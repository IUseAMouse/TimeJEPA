"""RateIN pooled ratios and the energy-based rate detector (2026-09-06)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_gift import _energy_series_k, _pool_ratios  # noqa: E402
from timejepa.evaluation.ratein import K_CANDIDATES  # noqa: E402
from timejepa.models import JEPATST  # noqa: E402


# ------------------------------------------------------------ pooling rule

def test_pooled_ratio_follows_amplitude_geomean_does_not():
    # two series: a large-amplitude one where k=2 hurts, a small one where
    # k=2 helps a lot. Equal weights (geomean) favour k=2; the CRPS-like
    # pooled sum follows the large series and rejects it.
    scores = {0: {1: [10.0], 2: [12.0]},        # big: k=2 20% worse
              1: {1: [0.1], 2: [0.05]}}         # small: k=2 50% better
    geo, n = _pool_ratios(scores, pooled=False)
    pooled, _ = _pool_ratios(scores, pooled=True)
    assert n == 2
    assert geo[2] < 1.0 < pooled[2]
    assert abs(pooled[2] - 12.05 / 10.1) < 1e-9


def test_pool_ratios_coverage_rule_and_k1_absent():
    scores = {i: {1: [1.0], 2: [0.5]} for i in range(6)}
    scores[0][4] = [0.1]                        # k=4 scored on 1/6 series
    r, n = _pool_ratios(scores, pooled=True)
    assert n == 6 and 1 not in r and 2 in r and 4 not in r


# ------------------------------------------------------- energy detector

def _judge():
    torch.manual_seed(0)
    return JEPATST(input_length=256, prediction_length=64, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="mlp").eval()


def test_energy_detector_returns_uniform_k_and_ratio_table():
    judge = _judge()
    rng = np.random.default_rng(1)
    t = np.arange(4000, dtype=np.float32)
    series = [np.sin(2 * np.pi * t / 96) + 0.1 * rng.standard_normal(len(t))
              for _ in range(3)]
    series = [s.astype(np.float32) for s in series]
    ks, diag = _energy_series_k(judge, series, h=32, windows=1, max_len=256,
                                stride=8, patch=16, device=torch.device("cpu"),
                                batch_size=8)
    assert set(ks) == {0, 1, 2} and len(set(ks.values())) == 1   # one k per config
    assert ks[0] in K_CANDIDATES and diag["K"] == ks[0]
    assert diag["margin"] == 0.0 and diag["judge_span"] == 64
    assert diag["n_base"] == 3
    assert all(r > 0 for r in diag["ratios"].values())
    # ratios are relative to k=1: a k is chosen only if its ratio is < 1
    if diag["K"] != 1:
        assert diag["ratios"][str(diag["K"])] < 1.0
    # large k on a short past is disqualified, never crashes
    short = [s[:400] for s in series]
    ks2, diag2 = _energy_series_k(judge, short, h=32, windows=1, max_len=256,
                                  stride=8, patch=16, device=torch.device("cpu"),
                                  batch_size=8)
    assert "48" not in diag2["ratios"] and set(ks2) == {0, 1, 2}
