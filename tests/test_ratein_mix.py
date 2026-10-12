"""RateIN-mix weights: the margin becomes a temperature, k=1 always a candidate."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_gift import MIX_TAU, _mix_weights  # noqa: E402


def test_no_gain_anywhere_is_pure_k1():
    w = _mix_weights({"2": 1.02, "4": 1.10, "8": 1.30})
    # k=1 leads; a near-tie (2% worse) keeps a hedge weight, a 30% loss none
    assert max(w, key=w.get) == 1 and w[1] > 0.5
    assert w[2] > 0.2 and 8 not in w
    assert math.isclose(sum(w.values()), 1.0)


def test_margin_equals_one_nat():
    # a k beating k=1 by the old margin gets weight ~e relative to k=1
    # (exactly exp(-ln(1 - tau) / tau), e to first order)
    w = _mix_weights({"4": 1.0 - MIX_TAU}, min_weight=0.0, max_components=10)
    assert math.isclose(w[4] / w[1], math.exp(-math.log(1 - MIX_TAU) / MIX_TAU))
    assert abs(w[4] / w[1] - math.e) / math.e < 0.05


def test_large_gain_becomes_selection():
    w = _mix_weights({"4": 0.70, "8": 0.95})
    assert set(w) == {4}                     # k=1 and k=8 fall under 2%
    assert math.isclose(w[4], 1.0)


def test_near_tie_is_hedged_and_sorted():
    w = _mix_weights({"3": 0.97, "16": 0.96})
    assert list(w) == [1, 3, 16]
    assert all(v > 0.15 for v in w.values())
    assert w[16] > w[3] > w[1]


def test_component_cap_and_disqualified_absent():
    ratios = {str(k): 0.9 - 0.001 * k for k in (2, 3, 4, 6, 8, 12)}
    w = _mix_weights(ratios)
    assert len(w) == 4 and 1 not in w        # cap 4, k=1 far behind
    # disqualified k (absent from the table) never gets weight
    assert 48 not in _mix_weights({"2": 0.9})
