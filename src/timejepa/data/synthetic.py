"""
Génération de séries synthétiques pour le pré-entraînement (G8 / P2.5).

Pourquoi ce module existe — trois trous mesurés qu'aucune donnée réelle ne bouche
--------------------------------------------------------------------------------
1. **Fréquences absentes** (E17) : les configs 10T/15T dégradent sur quatre
   datasets indépendants parce que le corpus n'a presque rien entre le 5 min
   (PEMS) et l'horaire. On ne peut pas télécharger ce qui n'existe pas ; on
   peut le générer.
2. **Séries courtes** (G7.1) : m1_*, monash_m3_*, tourism_* sont tous rejetés
   « médiane < 1280 » — une série annuelle fait ~30 points. Quinze des 97
   configs GIFT-Eval (A/Q/M/W) ont cette forme et le modèle n'en a JAMAIS vu.
3. **Géométrie de décimation** (G9) : `_sample_resolution_factor` exige
   `1280·f ≤ longueur_du_morceau` ; les morceaux réels font 2048, donc f=2 est
   déjà impossible. Le synthétique se génère à la longueur qu'on veut — 8192
   par défaut ici, ce qui autorise f ∈ {1..6} et fournit du même coup les
   paires (r, r′) du JEPA inter-résolution (G9.2).

Méthode — KernelSynth par caractéristiques de Fourier aléatoires
----------------------------------------------------------------
Chronos (KernelSynth) échantillonne des GP depuis des compositions de noyaux
{linéaire, RBF, périodique} ; FlowState fait pareil (CauKer). Échantillonner un
GP exact coûte une Cholesky O(N³) — à N=8192 c'est rédhibitoire. On utilise des
caractéristiques de Fourier aléatoires (Rahimi & Recht 2007) : pour un noyau
stationnaire, sommer K sinusoïdes à fréquences tirées de sa densité spectrale
converge vers le GP quand K croît. À K=64 par composante, l'échantillon est
indistinguable à l'œil et le coût est O(N·K).

Banque de composantes, tirées puis composées additivement (avec parfois une
enveloppe multiplicative, comme la composition ×) :
  * saisonnalités : 1–3 périodes log-uniformes dans [4, 2048] pas, chacune avec
    harmoniques décroissantes et phases aléatoires — couvre le cycle journalier
    vu à TOUTES les fréquences (24 pas à l'heure, 96 à 15 min, 144 à 10 min...) ;
  * dérive lisse : RBF par RFF, échelle de longueur log-uniforme [16, 1024] ;
  * tendance : linéaire ou par morceaux (point de rupture) ;
  * bruit : gaussien, parfois Student-t (queues lourdes — le plancher RevIN de
    G6 a montré que le corpus réel en contient) ;
  * rarement : sauts de niveau et impulsions.

La sortie est le MÊME format que prepare_lotsa.py : un `.npy` dense
[n_morceaux, longueur] float32 par « famille », memmappable, directement
globbable par `datasets: null`. Aucun code d'entraînement à modifier — pour
mélanger au corpus réel il suffit de déposer (ou symlinker) les fichiers dans
le répertoire du corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composantes
# ---------------------------------------------------------------------------

def _seasonal(n: int, rng: np.random.Generator,
              period_range=(4.0, 2048.0)) -> np.ndarray:
    """Une saisonnalité : période log-uniforme, 1-4 harmoniques décroissantes."""
    period = np.exp(rng.uniform(np.log(period_range[0]), np.log(period_range[1])))
    t = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    n_harm = rng.integers(1, 5)
    for h in range(1, n_harm + 1):
        amp = rng.uniform(0.3, 1.0) / h          # spectre décroissant
        phase = rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(2 * np.pi * h * t / period + phase)
    return out / max(np.std(out), 1e-8)


def _smooth_gp(n: int, rng: np.random.Generator, k: int = 64,
               lengthscale_range=(16.0, 1024.0)) -> np.ndarray:
    """
    GP RBF approché par caractéristiques de Fourier aléatoires : la densité
    spectrale d'un RBF d'échelle ℓ est une gaussienne d'écart-type 1/(2πℓ).
    """
    ls = np.exp(rng.uniform(np.log(lengthscale_range[0]),
                            np.log(lengthscale_range[1])))
    freqs = rng.normal(0.0, 1.0 / (2 * np.pi * ls), size=k)
    phases = rng.uniform(0, 2 * np.pi, size=k)
    t = np.arange(n, dtype=np.float64)
    out = np.cos(2 * np.pi * np.outer(t, freqs) + phases).sum(axis=1)
    out *= np.sqrt(2.0 / k)
    return out / max(np.std(out), 1e-8)


def _trend(n: int, rng: np.random.Generator) -> np.ndarray:
    """Tendance linéaire, ou par morceaux avec un point de rupture."""
    t = np.linspace(-1.0, 1.0, n)
    slope = rng.uniform(-1.0, 1.0)
    out = slope * t
    if rng.random() < 0.3:                        # rupture de pente
        cp = rng.integers(n // 4, 3 * n // 4)
        out[cp:] += rng.uniform(-1.0, 1.0) * (t[cp:] - t[cp])
    return out


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.15:                       # queues lourdes
        return rng.standard_t(df=rng.integers(3, 8), size=n)
    return rng.normal(0.0, 1.0, size=n)


def _events(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sauts de niveau et impulsions, rares — le réel en est plein."""
    n = len(x)
    if rng.random() < 0.15:                       # saut de niveau
        cp = rng.integers(n // 8, 7 * n // 8)
        x[cp:] += rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
    if rng.random() < 0.10:                       # impulsions
        idx = rng.integers(0, n, size=rng.integers(1, 4))
        x[idx] += rng.uniform(3.0, 8.0, size=len(idx)) * rng.choice([-1, 1], size=len(idx))
    return x


# ---------------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSpec:
    """
    Une « famille » synthétique = une distribution sur les compositions.

    `period_range` est le levier de couverture fréquentielle : une famille aux
    périodes courtes (24-150 pas) imite un cycle journalier vu en sub-horaire,
    une famille aux périodes longues (300-2048) imite du bas-fréquence.
    """
    name: str
    chunk_length: int = 8192
    period_range: tuple = (4.0, 2048.0)
    p_seasonal: float = 0.9
    p_smooth: float = 0.7
    p_trend: float = 0.6
    noise_scale: tuple = (0.05, 0.5)


def sample_series(spec: SyntheticSpec, rng: np.random.Generator) -> np.ndarray:
    n = spec.chunk_length
    parts = []
    if rng.random() < spec.p_seasonal:
        for _ in range(rng.integers(1, 4)):
            parts.append(rng.uniform(0.5, 2.0) * _seasonal(n, rng, spec.period_range))
    if rng.random() < spec.p_smooth:
        parts.append(rng.uniform(0.5, 2.0) * _smooth_gp(n, rng))
    if rng.random() < spec.p_trend:
        parts.append(rng.uniform(0.5, 2.0) * _trend(n, rng))
    if not parts:                                 # jamais une série vide
        parts.append(_seasonal(n, rng, spec.period_range))

    x = np.sum(parts, axis=0)
    if rng.random() < 0.25:                       # composition multiplicative
        x *= (1.0 + 0.5 * _smooth_gp(n, rng))
    x += rng.uniform(*spec.noise_scale) * _noise(n, rng)
    x = _events(x, rng)

    # Échelle et niveau aléatoires : RevIN normalise par instance, mais le
    # corpus réel n'est pas centré-réduit et le synthétique ne doit pas être
    # reconnaissable à sa normalisation.
    x = x * np.exp(rng.uniform(-1.0, 4.0)) + rng.uniform(-10.0, 1000.0)
    return x.astype(np.float32)


def write_synthetic_family(out_path: Path, spec: SyntheticSpec, n_chunks: int,
                           seed: int, log_every: int = 5000) -> Path:
    """Écrit une famille en `.npy` dense [n_chunks, chunk_length] float32."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_chunks, spec.chunk_length), dtype=np.float32)
    for i in range(n_chunks):
        out[i] = sample_series(spec, rng)
        if log_every and (i + 1) % log_every == 0:
            logger.info(f"  {spec.name}: {i + 1:,}/{n_chunks:,}")
    assert np.isfinite(out).all(), f"{spec.name}: valeurs non finies générées"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    logger.info(f"✓ {out_path.name}: {n_chunks:,} morceaux x {spec.chunk_length}")
    return out_path


# Les trois familles par défaut, une par trou mesuré. Les noms commencent par
# `synthetic_` pour être reconnaissables dans les logs du sampler et exclus
# d'un glob de corpus réel si besoin (pattern négatif).
DEFAULT_FAMILIES = (
    # Cycles courts (24-150 pas) : le journalier vu en sub-horaire — E17.
    SyntheticSpec("synthetic_subhourly", chunk_length=8192,
                  period_range=(24.0, 150.0)),
    # Tout-venant KernelSynth : périodes larges, le fond de diversité.
    SyntheticSpec("synthetic_broadband", chunk_length=8192,
                  period_range=(4.0, 2048.0)),
    # Basse fréquence : périodes 4-52 — la FORME des séries annuelles/
    # trimestrielles/mensuelles (G7.1). Audit 2026-08-20 (T4) : la version à
    # morceaux 1280 était décorative — 1 fenêtre par morceau, donc 25 k
    # fenêtres, plafond d'oversampling 3x épuisé à ~2 % de l'époque (au LR
    # maximal, rien de retenu), et inéligible à toute paire k1>1. À 8192 :
    # 865 fenêtres/morceau, poids sampler comparable aux autres familles, et
    # la famille participe à l'apprentissage inter-résolution. Les cycles
    # restent courts (4-52 pas) : c'est la période qui fait la « basse
    # fréquence », pas la longueur du morceau.
    SyntheticSpec("synthetic_lowfreq", chunk_length=8192,
                  period_range=(4.0, 52.0), p_trend=0.8),
)
