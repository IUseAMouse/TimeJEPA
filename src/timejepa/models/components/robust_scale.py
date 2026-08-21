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

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # Marqueur d'auto-description du checkpoint (cf. docstring). Aucun
        # gradient, aucune influence sur le calcul.
        self.register_buffer("is_robust", torch.ones(1))
        self.median: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, context: torch.Tensor) -> None:
        """context: [B, L, C] — stats sur L, par instance et par canal."""
        med = context.median(dim=1, keepdim=True).values                # [B,1,C]
        mad = (context - med).abs().median(dim=1, keepdim=True).values  # [B,1,C]
        self.median = med.detach()
        self.scale = (mad * MAD_TO_SIGMA).clamp_min(self.eps).detach()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.median is not None, "fit(context) d'abord"
        return torch.asinh((x - self.median) / self.scale)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse point à point. Pour un tenseur de quantiles [B, n, Q], la
        monotonie de sinh préserve l'ordre des niveaux — aucun re-tri requis.
        """
        assert self.median is not None, "fit(context) d'abord"
        med, scale = self.median, self.scale
        if x.dim() == med.dim() and x.shape[-1] != med.shape[-1]:
            # sorties quantiles [B, n, Q] contre stats [B, 1, C=1] : les stats
            # se diffusent sur la dimension Q (univarié, C=1).
            pass  # le broadcast [B,1,1] -> [B,n,Q] est déjà correct
        return torch.sinh(x) * scale + med
