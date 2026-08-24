"""
Tests du TTA (evaluate_gift.tta_forecast) : flip, shifts de translation, et le
théorème d'invariance d'échelle.

Invariants, par gravité :
1. Sans options = strictement model.forecast (aucun chemin parasite).
2. LE THÉORÈME : f(kx) = k·f(x) EXACTEMENT sous RobustScale+RevIN (médiane et
   MAD 1-homogènes ⇒ entrée normalisée identique) — la TTA d'échelle est un
   no-op prouvé, seul le SIGNE (flip) porte de l'information.
3. L'alignement des shifts : sur une rampe avec un oracle de continuation,
   les variantes décalées réalignées prédisent EXACTEMENT la base — la
   moyenne est l'identité. Tout décalage d'indice ferait échouer ce test.
4. Le masque de couverture : les positions de queue (non couvertes par les
   variantes décalées) ne moyennent que les variantes qui les voient.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_gift import tta_forecast                                   # noqa: E402
from timejepa.models import JEPATST                                      # noqa: E402


class _RampOracle:
    """Continuation parfaite d'une rampe de pente 1 : f(ctx)[j] = ctx[-1]+j+1.
    Avec un vrai décalage d'origine, la variante s prédit t−s+j+1 — le
    réalignement doit redonner exactement la base."""
    patching = SimpleNamespace(stride=8, patch_size=16)

    def forecast(self, ctx, n):
        base = ctx[:, -1:]
        steps = torch.arange(1, n + 1, dtype=ctx.dtype).unsqueeze(0)
        med = (base + steps).unsqueeze(-1)                    # [B, n, 1]
        fan = torch.cat([med - 1.0, med, med + 1.0], dim=-1)  # [B, n, 3]
        return {"forecast_denorm": med, "quantiles_denorm": fan,
                "quantile_levels": (0.1, 0.5, 0.9)}


class _MarkerOracle:
    """Renvoie une constante qui identifie la variante par la longueur de son
    contexte — pour vérifier le masque de couverture position par position."""
    patching = SimpleNamespace(stride=8, patch_size=16)

    def __init__(self, full_len):
        self.full_len = full_len

    def forecast(self, ctx, n):
        val = 0.0 if ctx.shape[1] == self.full_len else 1.0
        med = torch.full((ctx.shape[0], n, 1), val)
        return {"forecast_denorm": med}


def test_no_options_is_passthrough():
    m = _RampOracle()
    batch = torch.arange(64, dtype=torch.float32).unsqueeze(0)
    out = tta_forecast(m, batch, h=16)
    ref = m.forecast(batch, n=16)
    assert torch.equal(out["forecast_denorm"], ref["forecast_denorm"])


def test_scale_tta_is_a_provable_noop():
    """f(kx) = k·f(x) au flottant près : l'entrée normalisée est identique
    (médiane et MAD sont 1-homogènes), la dénormalisation multiplie par k."""
    m = JEPATST(input_length=384, prediction_length=96, patch_size=16,
                stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                predictor_num_layers=1, predictor_num_heads=4,
                predictor_d_ff=64, decoder_type="quantile",
                robust_scale=True).eval()
    m.set_pretrain_mode(False)
    x = torch.randn(2, 384, 1) * 5 + 40
    with torch.no_grad():
        f_x = m.forecast(x, n=96)["quantiles_denorm"]
        f_3x = m.forecast(3.0 * x, n=96)["quantiles_denorm"]
    torch.testing.assert_close(f_3x, 3.0 * f_x, rtol=1e-4, atol=1e-3)


def test_shift_alignment_is_exact_on_a_ramp():
    m = _RampOracle()
    batch = torch.arange(160, dtype=torch.float32).unsqueeze(0)  # rampe
    base = m.forecast(batch, n=32)["forecast_denorm"]
    out = tta_forecast(m, batch, h=32, shifts=[2, 4, 6])
    # variantes s : tronquées de s à droite PUIS ré-alignées au stride à
    # gauche — sur la rampe, l'oracle réaligné redonne exactement la base
    torch.testing.assert_close(out["forecast_denorm"], base)
    # le fan aussi (moyenne pondérée de vecteurs identiques)
    fan = out["quantiles_denorm"]
    assert (fan[..., 1:] >= fan[..., :-1]).all()


def test_coverage_mask_counts_only_covering_variants():
    h, s = 16, 4
    m = _MarkerOracle(full_len=160)
    batch = torch.zeros(1, 160)
    out = tta_forecast(m, batch, h=h, shifts=[s])["forecast_denorm"].squeeze()
    # positions 0..h−s−1 : moyenne(base=0, shift=1) = 0.5 ; queue : base seule
    assert torch.allclose(out[:h - s], torch.full((h - s,), 0.5))
    assert torch.allclose(out[h - s:], torch.zeros(s))


def test_flip_combines_with_shifts():
    m = _RampOracle()
    batch = torch.arange(160, dtype=torch.float32).unsqueeze(0)
    out = tta_forecast(m, batch, h=32, flip=True, shifts=[2])
    assert torch.isfinite(out["forecast_denorm"]).all()
    fan = out["quantiles_denorm"]
    assert (fan[..., 1:] >= fan[..., :-1]).all()


def test_shift_larger_than_horizon_is_dropped():
    """m4_yearly a h=6 : un shift >= h ne couvre aucune position — il doit
    être écarté, pas casser l'alignement (bug mesuré 2026-08-25)."""
    m = _RampOracle()
    batch = torch.arange(160, dtype=torch.float32).unsqueeze(0)
    out = tta_forecast(m, batch, h=6, shifts=[7])
    base = m.forecast(batch, n=6)["forecast_denorm"]
    torch.testing.assert_close(out["forecast_denorm"], base)
