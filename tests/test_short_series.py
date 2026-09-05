"""
Corpus v4 - short-series windows (2026-09-05).

Invariants pinned:
1. Defaults are bit-identical: no sidecar, or flag off, leaves the window set
   and the item dict unchanged.
2. With a _reallen sidecar, rows shorter than ctx+pred stop yielding sliding
   windows whose target lies in the left pad (the v3 defect) and, with the
   flag on, yield boundary windows: real target tail scored, padded tail
   masked, no decimation.
3. The masked pinball equals the plain pinball on a full mask and ignores the
   values at masked positions.
4. The finetune module accepts target_mask (point and quantile heads, with
   and without the anchor).
5. iter_dense_chunks records the real length of every emitted chunk.
"""

import numpy as np
import pytest
import torch

from timejepa.data.dataset import TimeSeriesDataset
from timejepa.data.lotsa import iter_dense_chunks
from timejepa.models import JEPATST
from timejepa.models.decoders.quantile_head import pinball_loss
from timejepa.training.finetune_module import FinetuneModule

CTX, PRED, L = 64, 16, 128          # small geometry: ctx + pred = 80 < L
N_REAL_SHORT = 30


def _corpus(tmp_path, with_sidecar):
    rng = np.random.default_rng(0)
    full = rng.standard_normal(L).astype(np.float32)
    short_real = (np.linspace(0.0, 1.0, N_REAL_SHORT) + 5.0).astype(np.float32)
    short = np.concatenate(
        [np.full(L - N_REAL_SHORT, short_real[0], dtype=np.float32), short_real])
    path = tmp_path / "fam.npy"
    np.save(path, np.stack([full, short]))
    if with_sidecar:
        (tmp_path / "_reallen").mkdir()
        np.save(tmp_path / "_reallen" / "fam.npy",
                np.array([L, N_REAL_SHORT], dtype=np.int32))
    return path, short_real


def _ds(path, **kw):
    return TimeSeriesDataset(path, context_length=CTX, prediction_length=PRED,
                             stride=8, return_tensor=True, **kw)


N_STD = len(range(0, L - (CTX + PRED) + 1, 8))      # sliding windows per full row


# ------------------------------------------------------------ 1. defaults inert

def test_no_sidecar_is_bit_identical(tmp_path):
    path, _ = _corpus(tmp_path, with_sidecar=False)
    off = _ds(path)
    assert len(off) == 2 * N_STD
    assert "target_mask" not in off[0]
    on = _ds(path, short_series_windows=True)
    assert len(on) == 2 * N_STD                    # no sidecar: same windows
    assert bool(on[0]["target_mask"].all())        # key present, all real


def test_sidecar_flag_off_drops_pad_target_windows_only(tmp_path):
    path, _ = _corpus(tmp_path, with_sidecar=True)
    ds = _ds(path)
    assert len(ds) == N_STD                        # the short row yields none
    assert set(int(i) for i in ds.window_indices[:, 0]) == {0}
    assert "target_mask" not in ds[0]


# ------------------------------------------------------- 2. boundary windows

def test_boundary_windows_score_real_tail_and_mask_the_pad(tmp_path):
    path, short_real = _corpus(tmp_path, with_sidecar=True)
    ds = _ds(path, short_series_windows=True, short_min_context=16,
             short_min_target=4)
    short_idx = [i for i in range(len(ds)) if ds.window_indices[i, 0] == 1]
    assert len(short_idx) >= 1
    assert len(ds) == N_STD + len(short_idx)
    pad_len = L - N_REAL_SHORT
    for i in short_idx:
        item = ds[i]
        start = int(ds.window_indices[i, 1])
        boundary = start + CTX
        assert pad_len + 16 <= boundary <= L - 4
        ctx, tgt, mask = item["context"], item["target"], item["target_mask"]
        assert ctx.shape[-1] == CTX and tgt.shape[-1] == PRED
        assert item["resolution_factor"] == 1
        n_real_ctx = boundary - pad_len
        n_real_tgt = min(PRED, L - boundary)
        # context: flat prefix, then the real steps before the boundary
        assert torch.allclose(ctx[-n_real_ctx:],
                              torch.from_numpy(short_real[:n_real_ctx]))
        assert torch.all(ctx[:-n_real_ctx] == float(short_real[0]))
        # target: real tail, then edge padding, masked accordingly
        assert torch.allclose(
            tgt[:n_real_tgt],
            torch.from_numpy(short_real[n_real_ctx:n_real_ctx + n_real_tgt]))
        assert int(mask.sum()) == n_real_tgt
        assert bool(mask[:n_real_tgt].all()) and not bool(mask[n_real_tgt:].any())
        if n_real_tgt < PRED:
            assert torch.all(tgt[n_real_tgt:] == tgt[n_real_tgt - 1])
    # full rows keep an all-True mask
    full_idx = next(i for i in range(len(ds)) if ds.window_indices[i, 0] == 0)
    assert bool(ds[full_idx]["target_mask"].all())


def test_short_rows_never_decimate(tmp_path):
    path, _ = _corpus(tmp_path, with_sidecar=True)
    ds = _ds(path, short_series_windows=True,
             multi_resolution_factors=[1, 2], p_multi_resolution=1.0)
    short_idx = [i for i in range(len(ds)) if ds.window_indices[i, 0] == 1]
    for i in short_idx:
        item = ds.get_item(i, allow_multi_resolution=True)
        assert item["resolution_factor"] == 1


# ------------------------------------------------------------ 3. masked pinball

def test_masked_pinball_equals_plain_on_full_mask_and_ignores_padding():
    torch.manual_seed(0)
    q = torch.sort(torch.randn(3, 10, 9), dim=-1).values
    y = torch.randn(3, 10, 1)
    levels = [0.1 * k for k in range(1, 10)]
    full = torch.ones(3, 10, dtype=torch.bool)
    assert torch.allclose(pinball_loss(q, y, levels, mask=full),
                          pinball_loss(q, y, levels))
    mask = full.clone()
    mask[:, 6:] = False
    base = pinball_loss(q, y, levels, mask=mask)
    y2 = y.clone()
    y2[:, 6:] += 100.0                                 # only masked positions
    assert torch.allclose(pinball_loss(q, y2, levels, mask=mask), base)
    manual = pinball_loss(q[:, :6], y[:, :6], levels)
    assert torch.allclose(base, manual)


# ------------------------------------------------------- 4. finetune module

def _module(decoder_type, lambda_anchor=0.0):
    model = JEPATST(input_length=512, prediction_length=128, patch_size=16,
                    stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                    predictor_num_layers=1, predictor_num_heads=4,
                    predictor_d_ff=64, decoder_type=decoder_type)
    m = FinetuneModule(model=model, finetune_mode="full_finetune",
                       lambda_anchor=lambda_anchor)
    m.model.eval()
    return m


@pytest.mark.parametrize("decoder_type", ["mlp", "quantile"])
def test_finetune_accepts_target_mask(decoder_type):
    torch.manual_seed(0)
    m = _module(decoder_type)
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    full = torch.ones(2, 128, dtype=torch.bool)
    with torch.no_grad():
        plain, _, _ = m._forward_and_loss(x, y)
        masked_full, _, _ = m._forward_and_loss(x, y, target_mask=full)
        part = full.clone()
        part[1, 100:] = False
        masked_part, _, _ = m._forward_and_loss(x, y, target_mask=part)
    assert torch.allclose(plain, masked_full)
    assert torch.isfinite(masked_part)


def test_anchor_skips_items_with_padded_targets():
    torch.manual_seed(0)
    m = _module("quantile", lambda_anchor=0.5)
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    part = torch.ones(2, 128, dtype=torch.bool)
    part[1, 64:] = False
    with torch.no_grad():
        loss, _, _ = m._forward_and_loss(x, y, target_mask=part)
    assert torch.isfinite(loss) and m._last_anchor is not None
    nothing_full = torch.zeros(2, 128, dtype=torch.bool)
    nothing_full[:, :8] = True
    with torch.no_grad():
        m._forward_and_loss(x, y, target_mask=nothing_full)
    assert float(m._last_anchor) == 0.0


# ------------------------------------------------------------- 5. sidecar data

def test_iter_dense_chunks_records_real_lengths():
    series = [np.arange(500, dtype=np.float32),      # padded to 1280
              np.arange(1280, dtype=np.float32),     # exact length
              np.arange(50, dtype=np.float32)]       # rejected
    real_lens = []
    chunks = list(iter_dense_chunks(iter(series), chunk_length=1280,
                                    min_length=384, pad_to=1280,
                                    real_lens=real_lens))
    assert len(chunks) == 2
    assert real_lens == [500, 1280]


def test_padded_short_series_kept_whole_and_not_counted_lost():
    """v4 prep (2026-09-05): with pad_to, a series between min_length and
    chunk_length is emitted whole (its tail included) and the 'LOST to
    chunking' counter stays at zero - that counter is for dense blocks."""
    from timejepa.data.lotsa import ChunkStats
    s = np.arange(72, dtype=np.float32)
    stats, real_lens = ChunkStats(), []
    chunks = list(iter_dense_chunks(iter([s]), chunk_length=1280, min_length=20,
                                    pad_to=1280, stats=stats, real_lens=real_lens))
    assert len(chunks) == 1 and real_lens == [72]
    assert np.array_equal(chunks[0][-72:], s)          # most recent steps kept
    assert stats.lost_to_chunking == 0 and stats.emitted == 1
