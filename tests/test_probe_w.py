"""xres coherence probe (2026-09-06): probe_instance takes w; identity at init
(zero FiLM), a real effect once the FiLM is perturbed, refusal without FiLM."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_energy import probe_instance  # noqa: E402
from timejepa.models import JEPATST  # noqa: E402


def _model(xres: bool):
    torch.manual_seed(0)
    return JEPATST(input_length=256, prediction_length=64, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="mlp",
                   cross_resolution=xres).eval()


def test_w_is_identity_at_init_then_real_after_perturbation():
    m = _model(True)
    rng = np.random.default_rng(0)
    ctx = rng.standard_normal(256).astype(np.float32)
    cands = rng.standard_normal((5, 32)).astype(np.float32)
    dev = torch.device("cpu")
    with torch.no_grad():
        _, e1 = probe_instance(m, ctx, cands, dev, standalone=True, w=1.0)
        _, e_half = probe_instance(m, ctx, cands, dev, standalone=True, w=0.5)
        assert np.allclose(e1, e_half)                 # zero-init FiLM
        m.predictor.w_film.weight.add_(0.5)
        _, e_half2 = probe_instance(m, ctx, cands, dev, standalone=True, w=0.5)
        _, e1b = probe_instance(m, ctx, cands, dev, standalone=True, w=1.0)
        assert np.allclose(e1, e1b)                    # log2(1) = 0: w=1 untouched
        assert not np.allclose(e1, e_half2)            # w=1/2 now conditions


def test_w_refused_without_film():
    m = _model(False)
    ctx = np.zeros(256, dtype=np.float32)
    cands = np.zeros((3, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        probe_instance(m, ctx, cands, torch.device("cpu"), standalone=True, w=0.5)
