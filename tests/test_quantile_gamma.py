"""
G4.2 tests - uniform conformal calibration of the quantiles.

Invariants, by severity:
1. gamma = 1 everywhere: BIT-IDENTICAL output (flag-on inert by default).
2. The median (point forecast) is NEVER modified, whatever gamma - the
   MASE-invariant contract, the equivalent of the ESJEPA gate.
3. gamma > 0 preserves fan monotonicity.
4. gamma_for_level recovers the exact factor on an artificially narrowed
   Gaussian fan: q_k = 0.5*(true q_k) => gamma ~ 2 (the calibration works).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_gift import apply_quantile_gamma                       # noqa: E402
from calibrate_quantiles import gamma_for_level                      # noqa: E402


def _fake_out(B=2, h=16, Q=9, four_dim=False):
    torch.manual_seed(0)
    med = torch.randn(B, h, 1)
    spread = torch.linspace(-1, 1, Q).view(1, 1, Q) * torch.rand(B, h, 1)
    q = med + spread                                    # monotone per level
    if four_dim:
        q = q.unsqueeze(-1)
    return {"forecast_denorm": med, "quantiles_denorm": q,
            "quantile_levels": tuple(np.linspace(0.1, 0.9, Q))}


def test_gamma_one_is_bit_identical():
    out = _fake_out()
    res = apply_quantile_gamma(out, torch.ones(9))
    assert torch.equal(res["quantiles_denorm"], out["quantiles_denorm"])
    assert torch.equal(res["forecast_denorm"], out["forecast_denorm"])


def test_median_never_moves():
    out = _fake_out()
    res = apply_quantile_gamma(out, torch.full((9,), 2.5))
    assert torch.equal(res["forecast_denorm"], out["forecast_denorm"])
    # the fan's median level (index 4, level 0.5) coincides with med => it
    # does not move either when gamma_0.5 = 1 (calibrator contract)
    g = torch.full((9,), 2.5); g[4] = 1.0
    res = apply_quantile_gamma(out, g)
    torch.testing.assert_close(res["quantiles_denorm"][..., 4],
                               out["quantiles_denorm"][..., 4])


def test_monotonicity_preserved():
    out = _fake_out()
    g = torch.tensor([1.8, 1.5, 1.3, 1.1, 1.0, 1.1, 1.3, 1.5, 1.8])
    fan = apply_quantile_gamma(out, g)["quantiles_denorm"]
    assert (fan[..., 1:] >= fan[..., :-1] - 1e-6).all()


def test_four_dim_path():
    out = _fake_out(four_dim=True)
    res = apply_quantile_gamma(out, torch.full((9,), 2.0))["quantiles_denorm"]
    ref = apply_quantile_gamma(
        {"forecast_denorm": out["forecast_denorm"],
         "quantiles_denorm": out["quantiles_denorm"][..., 0]},
        torch.full((9,), 2.0))["quantiles_denorm"]
    torch.testing.assert_close(res[..., 0], ref)


def test_none_gamma_and_missing_quantiles_are_noops():
    out = _fake_out()
    assert apply_quantile_gamma(out, None) is out
    out2 = {"forecast_denorm": out["forecast_denorm"]}
    assert apply_quantile_gamma(out2, torch.ones(9)) is out2


def test_gamma_for_level_recovers_true_factor():
    """Gaussian fan narrowed by a factor 2: the calibration must return ~2,
    above (0.9) as well as below (0.1) the median."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    from scipy.stats import norm
    for k in (0.1, 0.9):
        q_k = 0.5 * norm.ppf(k)          # fan 2x too narrow, med = 0
        r = y / q_k
        g = gamma_for_level(r, k)
        assert abs(g - 2.0) < 0.05, (k, g)


def test_gamma_for_level_small_sample_is_neutral():
    assert gamma_for_level(np.ones(10), 0.9) == 1.0


# ---------------------------------------------------------------------------
# Integration: the collection loop itself (2026-08-26 bug - accumulator state
# leaked between datasets, only the first was initialized).
# ---------------------------------------------------------------------------

from types import SimpleNamespace                                    # noqa: E402

from calibrate_quantiles import calibrate_dataset                    # noqa: E402


class _NarrowFanOracle:
    """Predicts med=0 and a Gaussian fan 2x too narrow for N(0,1) targets:
    the calibration must return gamma ~ 2 at levels 0.1/0.9."""
    patching = SimpleNamespace(stride=8, patch_size=16)
    LEVELS = (0.1, 0.5, 0.9)
    Z = (-1.2816, 0.0, 1.2816)

    def forecast(self, ctx, n):
        B = ctx.shape[0]
        med = torch.zeros(B, n, 1)
        fan = torch.stack([torch.full((B, n), 0.5 * z) for z in self.Z], dim=-1)
        return {"forecast_denorm": med, "quantiles_denorm": fan,
                "quantile_levels": self.LEVELS}


def _items(seed, n=64, L=64, h=16):
    rng = np.random.default_rng(seed)
    return [{"context": rng.standard_normal(L).astype(np.float32),
             "target": rng.standard_normal(h).astype(np.float32)}
            for _ in range(n)]


def test_calibrate_dataset_two_datasets_no_state_leak():
    m = _NarrowFanOracle()
    dev = torch.device("cpu")
    for seed in (0, 1):                       # TWO successive datasets
        items = _items(seed)
        levels, stats = calibrate_dataset(
            m, lambda j: items[int(j)], np.arange(len(items)), h=16,
            batch_size=16, flip=False, device=dev)
        assert list(levels) == [0.1, 0.5, 0.9]
        # fan 2x too narrow => gamma ~ 2 at the extremes, 1.0 pinned at center
        assert abs(stats["gamma"][0] - 2.0) < 0.35, stats["gamma"]
        assert abs(stats["gamma"][2] - 2.0) < 0.35, stats["gamma"]
        assert stats["gamma"][1] == 1.0
        # coverage before: q90 too low => P(y<=q90) ~ Phi(0.64) ~ 0.74
        assert 0.6 < stats["coverage_before"][2] < 0.85
