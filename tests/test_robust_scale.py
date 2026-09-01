"""
Tests for the robust arcsinh scaler (G8.4, components/robust_scale.py).

Four invariants, by severity:
1. Flag off = strictly NOTHING (state_dict, compute paths) - protects all
   reproduced checkpoints.
2. Checkpoints self-describe: a flag/checkpoint mismatch REFUSES at load time
   instead of producing silently wrong numbers (the flag weighs no
   parameters, only the marker betrays it).
3. The two measured pathologies are fixed: epsilon floor (G6, targets at
   10^3 sigma) and a spike crushing the signal (E17, domain half).
4. Monotonicity: denormalized quantiles stay ordered and in raw scale.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST                                      # noqa: E402
from timejepa.models.components.robust_scale import RobustScale          # noqa: E402
from timejepa.evaluation.loading import load_checkpoint                  # noqa: E402
from timejepa.training.finetune_module import FinetuneModule             # noqa: E402


def _model(robust=False, decoder="quantile"):
    return JEPATST(input_length=384, prediction_length=96, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type=decoder,
                   robust_scale=robust)


# ---------------------------------------------------------------------------
# 1. Off = nothing
# ---------------------------------------------------------------------------

def test_flag_off_changes_nothing():
    m = _model(robust=False)
    assert m.robust_scaler is None
    assert not any("robust" in k for k in m.state_dict())


# ---------------------------------------------------------------------------
# 2. Checkpoint self-description
# ---------------------------------------------------------------------------

def _save(model, path):
    sd = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd}, path)


def test_robust_ckpt_refused_by_plain_model(tmp_path):
    ckpt = tmp_path / "robust.ckpt"
    _save(_model(robust=True), ckpt)
    with pytest.raises(RuntimeError, match="core components"):
        load_checkpoint(_model(robust=False), str(ckpt), torch.device("cpu"))


def test_plain_ckpt_refused_by_robust_model(tmp_path):
    ckpt = tmp_path / "plain.ckpt"
    _save(_model(robust=False), ckpt)
    with pytest.raises(RuntimeError, match="core components"):
        load_checkpoint(_model(robust=True), str(ckpt), torch.device("cpu"))


def test_matching_robust_ckpt_loads(tmp_path):
    ckpt = tmp_path / "robust.ckpt"
    _save(_model(robust=True), ckpt)
    load_checkpoint(_model(robust=True), str(ckpt), torch.device("cpu"))


# ---------------------------------------------------------------------------
# 3. The measured pathologies
# ---------------------------------------------------------------------------

def test_round_trip_identity():
    rs = RobustScale()
    x = torch.randn(4, 384, 1) * 7 + 100
    rs.fit(x)
    torch.testing.assert_close(rs.inverse(rs.transform(x)), x,
                               rtol=1e-4, atol=1e-4)


def test_epsilon_floor_case_is_tamed():
    """
    G6: near-constant context + moving target. Under RevIN alone, the
    normalized target went to THOUSANDS of sigma. arcsinh is logarithmic in
    the tail: the same target becomes an ordinary number.
    """
    rs = RobustScale()
    ctx = torch.full((2, 384, 1), 5.0) + torch.randn(2, 384, 1) * 1e-4
    tgt = torch.full((2, 96, 1), 25.0)                 # jump of 20 units
    rs.fit(ctx)
    t = rs.transform(tgt)
    assert t.abs().max() < 20, f"transformed target at {t.abs().max():.1f} - the tail is not compressed"


def test_spike_does_not_crush_the_signal():
    """
    E17 domain half: an x1000 spike in the context. std lets it crush the
    whole signal toward zero; MAD ignores it, arcsinh compresses the spike
    ITSELF. The body of the signal must keep a working variance.
    """
    body = torch.sin(torch.linspace(0, 20, 384)).reshape(1, 384, 1)
    spiked = body.clone()
    spiked[0, 100, 0] = 1000.0
    rs = RobustScale()
    rs.fit(spiked)
    t = rs.transform(spiked)
    body_std = t[0, 200:, 0].std()                     # far from the spike
    assert body_std > 0.3, f"signal crushed (std {body_std:.3f}) - MAD did not kick in"
    assert t[0, 100, 0] < 15, "the spike itself must be compressed, not propagated"


# ---------------------------------------------------------------------------
# 4. Monotonicity and end-to-end
# ---------------------------------------------------------------------------

def test_forecast_denorm_lives_in_raw_scale_and_quantiles_stay_sorted():
    m = _model(robust=True).eval()
    ctx = torch.randn(3, 384, 1) * 4 + 1000            # shifted raw scale
    with torch.no_grad():
        out = m.forecast(ctx, n=96)
    fd, qd = out["forecast_denorm"], out["quantiles_denorm"]
    assert torch.isfinite(fd).all() and torch.isfinite(qd).all()
    # raw scale recovered (untrained model: close to the context level)
    assert 500 < fd.mean() < 1500, f"denorm out of raw scale ({fd.mean():.0f})"
    # sinh is monotone: the fan stays ordered level by level
    assert (qd[..., 1:] >= qd[..., :-1] - 1e-4).all(), "quantiles unordered after inverse"


def test_finetune_loss_path_runs_in_compressed_space():
    m = _model(robust=True)
    module = FinetuneModule(model=m)
    ctx, tgt = torch.randn(2, 384, 1) * 3 + 50, torch.randn(2, 96, 1) * 3 + 50
    loss, results, target = module._forward_and_loss(ctx, tgt)
    assert torch.isfinite(loss)
    # the target compared by the pinball lives in the compressed+RevIN space:
    # O(1) magnitudes, not the raw scale ~50
    assert target.abs().mean() < 10


def test_flat_plus_spikes_context_stays_invertible():
    """
    Regression (2026-08-22, mix finetune): a "flat + spikes" window
    (idle VM - 29% of bitbrains_rnd contexts have MAD exactly 0) gave a
    floor scale of 1e-8, an anchor shifted by ln(1e8) ~ 18, and the sinh
    inverse exploded (CRPS 1e10..inf measured). With the 0.1*std fallback,
    the anchor stays bounded and the round trip of a reasonable fan stays
    FINITE and sane.
    """
    ctx = torch.zeros(1, 384, 1)
    ctx[0, ::40, 0] = 100.0                    # spikes, MAD = 0, std > 0
    tgt = torch.full((1, 96, 1), 50.0)
    rs = RobustScale()
    rs.fit(ctx)
    t = rs.transform(tgt)
    assert t.abs().max() < 15, f"anchor still degenerate ({t.abs().max():.1f})"
    # a fan of width 2 around the target must invert to finite values of the
    # same order of magnitude as the data, not 1e10
    fan = torch.stack([t - 2, t, t + 2], dim=-1)
    raw = rs.inverse(fan)
    assert torch.isfinite(raw).all()
    assert raw.abs().max() < 1e5, f"inverse still explosive ({raw.abs().max():.3g})"


def test_strictly_constant_context_is_log_bounded():
    """Strictly constant context (std=0 too): the eps=1e-3 floor bounds the
    anchor at ~ln(2000*X) instead of ln(1e8*X) - no more +18 offset."""
    ctx = torch.full((1, 384, 1), 5.0)
    tgt = torch.full((1, 96, 1), 25.0)         # jump of 20
    rs = RobustScale()
    rs.fit(ctx)
    t = rs.transform(tgt)
    assert t.abs().max() < 12
    assert torch.isfinite(rs.inverse(t + 3)).all()


def test_rogue_tail_quantile_is_capped_by_context_envelope():
    """
    Regression G8.4b (2026-08-23, run mix_zs_1ep3e4 at 15%):
    bitbrains_fast_storage/H/short - flat + spikes context, the half-trained
    quantile head emits a tail quantile z ~ 15 in compressed space, and
    sinh(15)*scale ~ 10^6*scale disfigures the GIFT aggregate (CRPS 1.8e7
    measured, x1.19 on the geomean of the 97 configs BY ITSELF). The context
    envelope bounds the inverse to [min-K*w, max+K*w]; the clamp is monotone
    and inactive on reasonable values.
    """
    ctx = torch.zeros(1, 384, 1)
    ctx[0, ::40, 0] = 100.0                    # flat + spikes: range 100
    rs = RobustScale()
    rs.fit(ctx)
    rogue = torch.full((1, 96, 1), 15.0)       # rogue z in compressed space
    raw = rs.inverse(rogue)
    hi = 100.0 + RobustScale.FORECAST_ENVELOPE * 100.0
    assert torch.isfinite(raw).all()
    assert raw.max() <= hi + 1e-3, f"rogue quantile not bounded ({raw.max():.3g})"
    # monotonicity: a fan bracketing the rogue stays ordered after the clamp
    fan = torch.stack([rogue - 1, rogue, rogue + 1], dim=-1)
    inv = rs.inverse(fan)
    assert (inv[..., 1:] >= inv[..., :-1]).all()
    # and values INSIDE the envelope stay exact (round trip intact)
    torch.testing.assert_close(rs.inverse(rs.transform(ctx)), ctx,
                               rtol=1e-4, atol=1e-4)


def test_healthy_windows_unchanged_by_std_fallback():
    """On a Gaussian window, MAD*1.4826 ~ std > 0.1*std: the fallback is
    inactive and the transform stays what it was before the fix."""
    x = torch.randn(4, 384, 1) * 7 + 100
    rs = RobustScale()
    rs.fit(x)
    med = x.median(dim=1, keepdim=True).values
    mad = (x - med).abs().median(dim=1, keepdim=True).values * 1.4826
    torch.testing.assert_close(rs.scale, mad, rtol=1e-5, atol=1e-6)
