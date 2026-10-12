"""ratein_selection_gap: case split and counterfactuals on synthetic caches."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ratein_selection_gap import analyse, load_dir  # noqa: E402


def _write(d: Path, config: str, crps: float, k_hist: dict, oracle=None,
           bt_ratios=None):
    (d / "per_config").mkdir(parents=True, exist_ok=True)
    e = {"config": config, "model": {"CRPS": crps},
         "ratein": {"k_hist": k_hist, "frac_k_gt1": 0.0}}
    if bt_ratios is not None:
        e["ratein"]["backtest"] = {"K": int(max(k_hist, key=k_hist.get)),
                                   "margin": 0.05, "n_base": 10,
                                   "ratios": bt_ratios}
    if oracle is not None:
        per_k = oracle
        best = min(per_k, key=per_k.get)
        e["oracle"] = {"per_k_crps": per_k, "best_k": int(best),
                       "gain_vs_k1": 1 - per_k[best] / per_k["1"]}
    (d / "per_config" / (config.replace("/", "__") + ".json")).write_text(
        json.dumps(e))


def test_case_split_and_counterfactual(tmp_path):
    bt, orc = tmp_path / "bt", tmp_path / "oracle"
    # missed: selector k=1 (saw 0.97, below the 5% margin), oracle k=4 at 0.8
    _write(bt, "a/5T/long", 1.0, {"1": 10}, bt_ratios={"4": 0.97, "8": 1.1})
    _write(orc, "a/5T/long", 0.8, {"4": 10}, oracle={"1": 1.0, "4": 0.8, "8": 0.9})
    # wrong_k: selector k=2 (0.9), oracle k=8 (0.6)
    _write(bt, "b/H/short", 0.9, {"2": 10})
    _write(orc, "b/H/short", 0.6, {"8": 10}, oracle={"1": 1.0, "2": 0.9, "8": 0.6})
    # false_pos: selector k=3 (1.2), oracle k=1 (1.0)
    _write(bt, "c/D/short", 1.2, {"3": 10})
    _write(orc, "c/D/short", 1.0, {"1": 10}, oracle={"1": 1.0, "3": 1.2})
    # match
    _write(bt, "d/W/short", 0.5, {"1": 10})
    _write(orc, "d/W/short", 0.5, {"1": 10}, oracle={"1": 0.5, "2": 0.7})

    rep = analyse(load_dir(bt), load_dir(orc))
    assert rep["n_configs"] == 4
    cases = {r["config"]: r["case"] for r in rep["rows"]}
    assert cases == {"a/5T/long": "missed", "b/H/short": "wrong_k",
                     "c/D/short": "false_pos", "d/W/short": "match"}
    assert rep["missed_below_margin"] == ["a/5T/long"]
    # geomeans and residual
    geo_bt = (1.0 * 0.9 * 1.2 * 0.5) ** 0.25
    geo_or = (0.8 * 0.6 * 1.0 * 0.5) ** 0.25
    assert math.isclose(rep["geomean_bt"], geo_bt)
    assert math.isclose(rep["geomean_oracle"], geo_or)
    # fixing every case at oracle quality recovers the oracle geomean
    total_gap = sum(v["gap_log"] for v in rep["by_case"].values())
    assert math.isclose(math.exp(math.log(geo_bt) - total_gap / 4), geo_or)
    # the biggest contributor is the wrong_k config (0.9 -> 0.6)
    assert rep["rows"][0]["config"] == "b/H/short"
    assert rep["by_case"]["match"]["gap_log"] == 0.0
