"""from_pretrained / export round trip and the judge module."""

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path

from timejepa.evaluation import create_model_from_config
from timejepa.hub import export, from_pretrained
from timejepa.judge import energy

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"


def _small_model():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name="lotsa_tiny_v3_eval")
    torch.manual_seed(0)
    return create_model_from_config(cfg).eval(), cfg


def test_export_round_trip_is_exact(tmp_path):
    model, cfg = _small_model()
    out = export(model, cfg.model, tmp_path / "hf", judge_standalone_targets=True)
    assert (out / "config.json").exists() and (out / "model.safetensors").exists()
    back = from_pretrained(str(out))
    assert back.judge_standalone_targets is True
    x = torch.randn(2, 512, 1)
    with torch.no_grad():
        a = model.forecast(x, n=64)["quantiles_denorm"]
        b = back.forecast(x, n=64)["quantiles_denorm"]
    assert torch.equal(a, b)
    # no target-encoder copy and no runtime buffer in the exported weights
    from safetensors.torch import load_file
    keys = load_file(str(out / "model.safetensors")).keys()
    assert not any("target_encoder" in k for k in keys)
    assert not any(k.endswith((".mean", ".std")) and "revin" in k for k in keys)


def test_energy_matches_probe_instance_and_both_conventions():
    model, _ = _small_model()
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "probe_energy", Path(__file__).resolve().parents[1] / "scripts" / "probe_energy.py")
    probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    rng = np.random.default_rng(1)
    ctx = rng.standard_normal(512).astype(np.float32)
    cands = rng.standard_normal((4, 96)).astype(np.float32)
    with torch.no_grad():
        for standalone in (False, True):
            a = probe.probe_instance(model, ctx, cands, torch.device("cpu"), standalone=standalone)
            b = energy(model, ctx, cands, standalone=standalone)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        e_ctx = energy(model, ctx, cands, standalone=False)[1]
        e_alone = energy(model, ctx, cands, standalone=True)[1]
    assert not np.array_equal(e_ctx, e_alone)
    model.judge_standalone_targets = True
    with torch.no_grad():
        assert np.array_equal(energy(model, ctx, cands)[1], e_alone)
