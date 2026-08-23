"""
Mise à l'échelle robuste arcsinh (G8.4) — COMPOSÉE autour de RevIN, jamais à sa place.

Le contrat, décidé le 2026-08-20 (PLAN.md G8.4) :

    entrée :  x' = arcsinh((x − médiane(ctx)) / MAD_scaled(ctx))
    sortie :  x  = sinh(x') · MAD_scaled + médiane

RevIN et tout son contrat de dénormalisation (freeze / to_input_frame /
denormalize_target_space — les cicatrices B2/B3/B10) restent INTACTS : ils
opèrent simplement dans l'espace compressé. Le rollout entier, la loss de
finetune (pinball en espace normalisé) et le prédicteur vivent dans cet espace ;
seules les sorties `*_denorm` repassent en brut via l'inverse.

Pourquoi ça marche là où le z-score casse (mesuré, pas supposé) :
* Plancher epsilon (G6) : un contexte quasi constant met l'échelle RevIN à
  ~0.003 et projette la cible à des MILLIERS de sigma. arcsinh est
  logarithmique en queue : arcsinh(6300) ≈ 9.4 — l'aberration devient un
  nombre ordinaire au lieu de dévorer le gradient.
* Queues lourdes (E17, moitié « domaine ») : un spike ×100 dans une trace de
  VM gonfle le std et écrase le reste du signal vers zéro. La MAD ignore le
  spike ; arcsinh compresse le spike lui-même. bitbrains/car_parts/bizitobs —
  nos pires configs — sont exactement ce régime. Précédent : Toto (Datadog,
  né du cloudops) utilise un « robust arcsinh scaler ».

Propriétés qui rendent la composition sûre :
* arcsinh est STRICTEMENT MONOTONE → les quantiles sont équivariants : la tête
  probabiliste prédit dans l'espace compressé et l'inverse point à point rend
  des quantiles valides et ordonnés en espace brut. La médiane commute avec
  l'inverse (median(sinh(q)) = sinh(median(q))) ; on n'utilise jamais la
  moyenne, qui elle ne commuterait pas.
* MAD × 1.4826 est un estimateur consistant de sigma pour une gaussienne, et
  arcsinh(z) ≈ z pour |z| ≲ 1 : sur une fenêtre « sage », la transformation est
  quasi l'identité du z-score — le comportement existant est préservé là où il
  marchait, seules les queues changent.
* AUCUN paramètre appris ici. Les statistiques sont des attributs d'exécution
  (comme revin.mean/std) — mais un buffer marqueur est enregistré pour que les
  checkpoints se DÉCLARENT : charger un checkpoint arcsinh dans un modèle nu
  (ou l'inverse) donnerait des chiffres silencieusement faux, le marqueur rend
  le mismatch détectable par le contrat de refus P3.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# MAD -> écart-type équivalent pour une gaussienne. Garde arcsinh≈z-score sur
# les fenêtres bien élevées.
MAD_TO_SIGMA = 1.4826


class RobustScale(nn.Module):
    """
    Transformation robuste par instance et par canal, statistiques du CONTEXTE.

    Même sémantique de cycle de vie que RevIN : `fit(context)` calcule et stocke
    (médiane, échelle) ; `transform`/`inverse` les réutilisent — la cible et
    chaque sortie sont donc traitées dans le repère du contexte, jamais le leur
    (le repère de la cible fuiterait le futur).
    """

    # Repli et plancher d'échelle — correctif du 2026-08-22, mesuré sur le
    # finetune mix (CRPS à 10^10..inf sur bitbrains/kdd à 5-10 % d'époque).
    #
    # La pathologie : sur une fenêtre « plate + spikes » (VM idle — 29 % des
    # contextes de bitbrains_rnd ont MAD EXACTEMENT 0), l'ancien plancher
    # eps=1e-8 donnait une échelle 1e-8, donc un repère compressé décalé de
    # ln(1e8) ≈ 18 : la cible atterrissait à |arcsinh| ≈ 20-38, et l'INVERSE
    # sinh ré-amplifiait exponentiellement — sinh(34) ≈ 3e14, overflow float32
    # à ~89. Même un fan parfaitement entraîné dans ce repère s'inversait en
    # intervalles bruts astronomiques : structurel, pas transitoire. C'est la
    # pathologie du plancher epsilon de G6, ressuscitée un étage plus haut.
    #
    # Le correctif exploite une asymétrie : une échelle TROP GRANDE est bénigne
    # (arcsinh devient quasi linéaire et RevIN — qui suit dans la composition —
    # renormalise : dégradation gracieuse vers le comportement z-score) ; une
    # échelle trop petite est catastrophique (l'offset logarithmique explose à
    # l'inverse). MAIS le repli doit être CONDITIONNEL, pas un max : un
    # max(MAD, 0.1·std) inconditionnel laisserait un spike isolé regonfler
    # l'échelle sur une fenêtre saine — précisément la pathologie E17 que la
    # MAD existe pour ignorer (attrapé par le test spike existant). Donc :
    #   échelle = MAD·1.4826                     si MAD·1.4826 > 0.01·std
    #           = max(0.1·std, eps)              sinon (MAD effondrée)
    # * fenêtres saines (MAD ≈ std, spike ou pas) : MAD garde la main,
    #   comportement STRICTEMENT identique à avant le correctif ;
    # * plates + spikes (MAD ~ 0 face au std) : 0.1·std prend le relais — le
    #   seul estimateur d'échelle encore disponible sur ce régime ;
    # * constantes strictes (std = 0 aussi) : eps = 1e-3 borne le repère à
    #   arcsinh(X/1e-3) ≈ ln(2000·X) — fini et log-borné, plus jamais +18.
    STD_FALLBACK = 0.1
    MAD_COLLAPSE_GATE = 0.01

    # Enveloppe de prévision relative au contexte — correctif G8.4b du
    # 2026-08-23, mesuré sur le run mix_zs_1ep3e4 à 15 % d'époque :
    # bitbrains_fast_storage/H/short a affiché un CRPS de 18 305 724 (la config
    # vaut 0.62-0.67 sur toutes les évals voisines), soit x1.19 sur l'agrégat
    # geomean des 97 configs À LUI SEUL. Diagnostic : ce n'est PAS l'échelle
    # plancher (le repère était borné) — c'est la tête quantile mi-entraînée
    # qui émet un quantile de queue |z| ≈ 15 en espace compressé, que sinh
    # ré-amplifie en sinh(15)·échelle ≈ 10^6·échelle. Le plancher ne peut rien
    # contre un z voyou : la garde correcte est en AVAL, sur l'inverse.
    #
    # Le prior encodé : une prévision ne sort pas de
    #     [min(ctx) − K·w, max(ctx) + K·w],  w = max(étendue(ctx), échelle)
    # avec K = 10 — dix fois l'étendue du contexte au-delà de ses bornes, très
    # au-dessus de tout futur plausible des benchmarks, très en dessous des
    # accidents sinh. Précédent structurel : le vocabulaire de Chronos borne
    # ses sorties à ±15σ par construction. Le clamp est monotone (l'ordre des
    # quantiles survit) et INACTIF sur toute prévision raisonnable — seuls les
    # accidents de queue sont touchés. `w` est protégé par l'échelle pour les
    # contextes dégénérés (étendue 0). Bonus mesurable attendu : borne aussi
    # le biais haussier x10-30 des fenêtres quasi-nulles (l'observation
    # london_smart_meters qui a ouvert G8.4b).
    FORECAST_ENVELOPE = 10.0

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        # Marqueur d'auto-description du checkpoint (cf. docstring). Aucun
        # gradient, aucune influence sur le calcul.
        self.register_buffer("is_robust", torch.ones(1))
        self.median: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.ctx_min: torch.Tensor | None = None
        self.ctx_max: torch.Tensor | None = None

    def fit(self, context: torch.Tensor) -> None:
        """context: [B, L, C] — stats sur L, par instance et par canal."""
        med = context.median(dim=1, keepdim=True).values                # [B,1,C]
        mad = (context - med).abs().median(dim=1, keepdim=True).values  # [B,1,C]
        std = context.std(dim=1, keepdim=True)                          # [B,1,C]
        mad_sigma = mad * MAD_TO_SIGMA
        fallback = (self.STD_FALLBACK * std).clamp_min(self.eps)
        self.median = med.detach()
        self.scale = torch.where(
            mad_sigma > self.MAD_COLLAPSE_GATE * std, mad_sigma, fallback
        ).clamp_min(self.eps).detach()
        # Bornes du contexte pour l'enveloppe de prévision (cf. FORECAST_ENVELOPE).
        self.ctx_min = context.amin(dim=1, keepdim=True).detach()
        self.ctx_max = context.amax(dim=1, keepdim=True).detach()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.median is not None, "fit(context) d'abord"
        return torch.asinh((x - self.median) / self.scale)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse point à point, borné par l'enveloppe de contexte (cf.
        FORECAST_ENVELOPE). Pour un tenseur de quantiles [B, n, Q], la
        monotonie de sinh ET du clamp préserve l'ordre des niveaux — aucun
        re-tri requis. Toute valeur dont l'inverse exact tombe DANS l'enveloppe
        (c.-à-d. toute prévision raisonnable, et tout aller-retour
        transform→inverse de données réelles) est restituée exactement.
        """
        assert self.median is not None, "fit(context) d'abord"
        med, scale = self.median, self.scale
        if x.dim() == med.dim() and x.shape[-1] != med.shape[-1]:
            # sorties quantiles [B, n, Q] contre stats [B, 1, C=1] : les stats
            # se diffusent sur la dimension Q (univarié, C=1).
            pass  # le broadcast [B,1,1] -> [B,n,Q] est déjà correct
        raw = torch.sinh(x) * scale + med
        half = self.FORECAST_ENVELOPE * torch.maximum(self.ctx_max - self.ctx_min, scale)
        return raw.clamp(self.ctx_min - half, self.ctx_max + half)
