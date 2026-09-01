"""
RateIN — canonicalisation du taux d'échantillonnage à l'inférence.

Décision 2026-08-31 (verdict de prémisse G9.3) : les deux modèles sub-3M qui
nous battent traitent l'échelle temporelle À L'ENTRÉE — FlowState ajuste son
pas interne par la saisonnalité fournie, TinyCast détecte la période par FFT
(zéro paramètre) et réaligne. Notre loi interne E1 dit la même chose : la
compétence suit le NOMBRE DE CYCLES vus dans le contexte (bande de
fonctionnement ~16-48 pas/cycle, soit 2-6 positions de patch), et
l'interpolation d'entrée est catastrophique (ECL ×4 : skill −136 %). D'où :

  * détection de période CAUSALE (rfft + test de Fisher, zéro paramètre —
    même statut de fairness que la médiane/MAD de RobustScale : une
    statistique du contexte, une règle uniforme pour les 97 configs) ;
  * DÉCIMATION SEULE vers la bande [16, 48] pas/cycle (jamais k<1 : on ne
    fabrique pas de points) ;
  * forecast à h' = ceil(h/k) sur la grille décimée (bonus : moins de
    rollouts sur les configs long-terme haute fréquence), puis
    ré-interpolation du fan complet vers la grille native.

k=1 (défaut, et choix du détecteur sur toute série sans pic significatif ou
déjà dans la bande) = chemin d'éval STRICTEMENT identique — épinglé par test.
C'est aussi le test de falsification le moins cher de l'hypothèse xres : si
même l'oracle-k ne gagne rien, la géométrie d'échelle n'est pas le mécanisme
de la queue.
"""

from typing import Optional

import numpy as np

# Bande de fonctionnement cible en pas/période (E1 : 2-6 positions de patch
# à stride 8). Le k choisi est le PLUS PETIT facteur qui y ramène la période.
# La grille monte à 48 : un cycle journalier vu en minutes (période 1440)
# demande k≈32 ; à k élevé l'historique disponible borne de lui-même le
# contexte effectif (decimate prend ce qu'il y a — le modèle est
# longueur-agnostique). Le repli ne passe JAMAIS sous la bande (on ne
# sur-décime pas un cycle).
BAND_LO, BAND_HI = 16, 48
K_CANDIDATES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


def detect_period(history: np.ndarray, max_window: int = 8192,
                  alpha: float = 0.05,
                  min_period: int = 16) -> Optional[int]:
    """Période dominante des `max_window` derniers points, ou None.

    Test de significativité de Fisher sur le périodogramme (g-statistic),
    seuil Bonferroni sur le nombre de fréquences testées — le protocole
    TinyCast, conservateur par construction : sur du bruit blanc, la
    probabilité de détection est ~alpha, et sans détection k restera 1.
    Exigences : >= 2 périodes complètes dans la fenêtre, période >= min_period
    (en deçà, la bande [16,48] est déjà atteinte, inutile de décimer).
    """
    x = np.asarray(history, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[-max_window:]
    n = len(x)
    if n < 4 * min_period:
        return None
    x = x - x.mean()
    if not np.any(x):
        return None

    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec[1:]                                    # sans la composante DC
    freqs_idx = np.arange(1, len(spec) + 1)
    periods = n / freqs_idx
    # candidates : au moins 2 périodes complètes, et période >= min_period
    valid = (periods <= n / 2) & (periods >= min_period)
    if not valid.any():
        return None
    m = int(valid.sum())
    g = spec[valid] / spec[valid].sum()
    # Seuil de Fisher (approximation premier terme) : g* tel que
    # m·(1−g*)^(m−1) = alpha. Appliqué à TOUS les pics (conservateur), et on
    # retient la PLUS PETITE période significative, pas la plus forte —
    # mesuré au smoke (2026-08-31) : sur electricity/H, le pic dominant est
    # l'hebdo (168 → k=6) et décimer détruit la structure intra-journalière
    # que le modèle exploitait ; si un cycle significatif vit déjà dans la
    # bande, la bonne décision est k=1.
    g_star = 1.0 - (alpha / m) ** (1.0 / (m - 1))
    # Maxima locaux seulement : la fuite spectrale d'un pic vrai éclabousse
    # les bins voisins au-dessus du seuil, et « la plus petite période
    # significative » deviendrait un lobe (mesuré : sinusoïde P=96 détectée
    # 93). Un lobe n'est pas un maximum local du périodogramme.
    gp = np.concatenate(([0.0], g, [0.0]))
    local_max = (g >= gp[:-2]) & (g >= gp[2:])
    sig = (g >= g_star) & local_max
    if not sig.any():
        return None
    return int(round(periods[valid][sig].min()))


def choose_k(period: Optional[int]) -> int:
    """Plus petit k qui ramène period/k dans [BAND_LO, BAND_HI] ; 1 sinon."""
    if period is None or period <= BAND_HI:
        return 1
    for k in K_CANDIDATES:
        if BAND_LO <= period / k <= BAND_HI:
            return k
    # période énorme sans k exact dans la grille : prendre le plus grand k
    # qui ne passe pas SOUS la bande (jamais sur-décimer un cycle).
    fallback = [k for k in K_CANDIDATES if period / k >= BAND_LO]
    return fallback[-1] if fallback else 1


def decimate(x: np.ndarray, k: int) -> np.ndarray:
    """Mean-pool par blocs de k, ALIGNÉ À DROITE (le dernier point du dernier
    bloc est le dernier point de la série — l'origine du forecast ne bouge
    pas) ; l'excédent de gauche est tronqué."""
    if k == 1:
        return x
    n = (len(x) // k) * k
    return x[len(x) - n:].reshape(-1, k).mean(axis=1)


def reinterp_fan(fan_dec: np.ndarray, h: int, k: int) -> np.ndarray:
    """[h', Q] sur grille décimée -> [h, Q] sur grille native.

    Le bloc décimé i couvre les pas natifs [i·k, (i+1)·k) ; sa valeur est
    posée au CENTRE du bloc (i·k + (k−1)/2) et chaque niveau de quantile est
    interpolé linéairement entre centres (extrapolation constante aux bords).
    La monotonie des niveaux survit : combinaison convexe de vecteurs triés.
    """
    if k == 1:
        return fan_dec[:h]
    h_dec = fan_dec.shape[0]
    centers = np.arange(h_dec) * k + (k - 1) / 2.0
    t = np.arange(h, dtype=np.float64)
    out = np.empty((h, fan_dec.shape[1]), dtype=fan_dec.dtype)
    for q in range(fan_dec.shape[1]):
        out[:, q] = np.interp(t, centers, fan_dec[:, q])
    return out
