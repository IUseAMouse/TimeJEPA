"""
Tests de la GRILLE DE CONFIGS Hydra (extraits de test_p0_regressions.py, audit
du 2026-08-19 — déplacés à l'identique, aucun test réécrit).

Couvre : composition de chaque config d'expérience (B21), dimensions des
échelles lotsa_* contre leurs références, arms geo ne différant que par leur
variable déclarée, noms de checkpoints sans '=' (B21). Ce sont les tests les
plus lents de la suite (composition Hydra) : marqués `slow`.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST                                      # noqa: E402
from timejepa.models.components.revin import RevIN                       # noqa: E402
from timejepa.training.utils.baselines import (                          # noqa: E402
    seasonal_naive_forecast,
    last_value_forecast,
    mean_forecast,
    linear_trend_forecast,
    compute_all_baselines,
    get_seasonality,
)
from timejepa.training.utils.metrics import (                            # noqa: E402
    mase,
    nd,
    weighted_quantile_loss,
    compute_forecasting_metrics_extended,
)


@pytest.mark.parametrize("size,reference", [("mini", "mini"), ("base", "base")])
def test_lotsa_scale_configs_match_their_reference_dimensions(size, reference):
    """
    The dimensions are written out rather than inherited from mini.yaml/base.yaml,
    because those carry their own data block which would clobber the LOTSA
    corpus. Written-out values drift; this pins them.
    """
    ref = _compose(reference)
    pre = _compose(f"lotsa_{size}")
    for block in ("encoder", "predictor"):
        for key in ("d_model", "n_layers", "d_ff"):
            assert pre.model[block][key] == ref.model[block][key], f"{block}.{key}"
    # ...while the corpus and geometry stay those of the LOTSA round
    base = _compose("lotsa_tiny")
    assert pre.data.data_dir == base.data.data_dir
    assert pre.data.use_mmap is True
    assert pre.model.seq_length == base.model.seq_length
    assert pre.model.decoder.type == "quantile"
    assert pre.training.loss.type == "sigreg"


@pytest.mark.parametrize("size", ["tiny", "mini", "base"])
def test_lotsa_eval_config_matches_the_trained_model(size):
    """
    A shape mismatch at eval time only WARNS before producing silently wrong
    numbers — the trap already hit on the p32 arm. Capacity must match too, not
    just geometry.
    """
    zs = _compose(f"lotsa_{size}_zeroshot")
    ev = _compose(f"lotsa_{size}_eval")

    for key in ("seq_length", "prediction_length", "patch_length", "stride"):
        assert ev.model[key] == zs.model[key], f"eval/{key} drifted"
    for block in ("encoder", "predictor"):
        for key in ("d_model", "n_layers", "d_ff"):
            assert ev.model[block][key] == zs.model[block][key], f"eval/{block}.{key}"
    assert ev.model.decoder.type == zs.model.decoder.type

    # The eval config reads the HELD-OUT corpus, never LOTSA
    assert "lotsa" not in str(ev.data.data_dir).lower()
    assert ev.data.get("use_mmap", False) is False

    # The zero-shot arm trains its decoder on LOTSA only
    assert zs.data.datasets_finetune is None
    assert "lotsa" in str(zs.data.data_dir)


def test_lotsa_configs_share_one_effective_batch_regime():
    """
    Effective batch is batch x accumulation x GPUs. tiny_geo used accumulation 6
    on one or two cards; inheriting it here gave 6144 across four, a learning-rate
    regime unrelated to any previous run, and forced an override on every command.
    All three scales are now calibrated for four GPUs.
    """
    effective = {
        size: _compose(f"lotsa_{size}").data.batch_size
        * _compose(f"lotsa_{size}").trainer.accumulate_grad_batches
        * 4
        for size in ("tiny", "mini", "base")
    }
    assert all(1000 <= v <= 2048 for v in effective.values()), effective


def test_lotsa_configs_do_not_disturb_existing_ones():
    """LOTSA configs are additions; tiny_geo must be untouched by their presence."""
    base = _compose("tiny_geo")
    assert base.data.get("use_mmap", False) is False
    assert "lotsa" not in str(base.data.data_dir).lower()

    pre = _compose("lotsa_tiny")
    assert pre.data.use_mmap is True
    assert "lotsa" in str(pre.data.data_dir)
    assert pre.data.datasets is None          # glob the directory
    assert pre.model.seq_length == base.model.seq_length      # same geometry
    assert pre.model.patch_length == base.model.patch_length

    ft = _compose("lotsa_tiny_finetune")
    assert ft.training.mode == "finetune"
    # The domain-adapted arm stays on the Monash corpus (contaminated, documented)
    assert ft.data.data_dir == base.data.data_dir
    assert ft.data.get("use_mmap", False) is False
    assert len(ft.data.datasets_finetune) == len(base.data.datasets_finetune)

    # The zero-shot arm — the primary protocol — must train its decoder on LOTSA
    # only, so that Monash and Nixtla stay unseen at every stage.
    zs = _compose("lotsa_tiny_zeroshot")
    assert zs.training.mode == "finetune"
    assert zs.training.finetune_mode == "full_finetune"
    assert zs.data.datasets_finetune is None      # glob the LOTSA directory
    assert "lotsa" in str(zs.data.data_dir)
    assert zs.data.use_mmap is True
    assert zs.model.name != ft.model.name         # separate checkpoint trees

    # The eval config must carry the TRAINING geometry but the EVALUATION data.
    # If the two ever diverge, this breaks here rather than silently producing
    # wrong numbers: at eval time a geometry mismatch only WARNS.
    ev = _compose("lotsa_tiny_eval")
    for key in ("seq_length", "prediction_length", "patch_length", "stride"):
        assert ev.model[key] == zs.model[key], f"eval/{key} drifted from the trained model"
    assert ev.model.decoder.type == zs.model.decoder.type
    # ...and it must evaluate on the held-out corpus, never on LOTSA
    assert ev.data.data_dir == base.data.data_dir
    assert "lotsa" not in str(ev.data.data_dir).lower()
    assert ev.data.get("use_mmap", False) is False


def _compose(name):
    from hydra import initialize, compose
    root = Path(__file__).resolve().parents[1]
    with initialize(version_base=None, config_path="../configs/model"):
        return compose(config_name=name)


@pytest.mark.parametrize("config_name", ["tiny", "tiny_geo", "tiny_geo_p32",
                                         "tiny_geo_vicreg", "tiny_geo_scratch"])
def test_checkpoint_filename_has_no_equals_sign(config_name):
    """
    B21. auto_insert_metric_name lived in the config but was never forwarded to
    ModelCheckpoint, so Lightning kept its default (True) and prefixed each
    metric name on top of the template's own text:
    'epochepoch=00_val_lossval_loss=0.3445.ckpt'. Hydra's override grammar
    treats '=' as a separator, so every downstream finetune and eval command
    needed quoting gymnastics — and one of them failed outright with backslashes
    surviving literally into the path.
    """
    cfg = _compose(config_name)
    assert "=" not in cfg.checkpoint.filename
    assert cfg.checkpoint.auto_insert_metric_name is False


def test_geo_arms_differ_only_in_their_declared_variable():
    """
    Each arm of the geometry grid is a named config, not a pile of overrides:
    a forgotten `model.patch_length=32` at EVAL time only warns
    ('re-initialising patching.*') and yields silently wrong numbers.
    Everything except the arm's own variable must match the base.
    """
    base = _compose("tiny_geo")
    # Round-wide defaults: no override should ever be needed for these
    assert base.training.loss.type == "sigreg"
    assert base.model.decoder.type == "quantile"
    assert len(base.data.datasets_finetune) == len(base.data.datasets)

    p32 = _compose("tiny_geo_p32")
    assert (p32.model.patch_length, p32.model.stride) == (32, 16)
    assert p32.model.name != base.model.name
    for key in ("seq_length", "prediction_length"):
        assert p32.model[key] == base.model[key]
    assert p32.training.loss.type == base.training.loss.type
    assert p32.model.decoder.type == base.model.decoder.type

    vic = _compose("tiny_geo_vicreg")
    assert vic.training.loss.type == "vicreg"
    assert vic.model.name != base.model.name
    assert (vic.model.patch_length, vic.model.stride) == (base.model.patch_length,
                                                          base.model.stride)

    scratch = _compose("tiny_geo_scratch")
    assert scratch.training.mode == "finetune"
    # Mandatory: gradual_unfreeze would freeze randomly initialised weights
    assert scratch.training.finetune_mode == "full_finetune"
    assert "pretrained_encoder_path" not in scratch.training
