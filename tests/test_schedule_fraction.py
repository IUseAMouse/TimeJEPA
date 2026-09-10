"""schedule_fraction (2026-09-08): the cosine anneals at the real budget and
the run is bounded to it; inert at 1.0."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.models import JEPATST  # noqa: E402
from timejepa.training.finetune_module import FinetuneModule  # noqa: E402


def _model():
    torch.manual_seed(0)
    return JEPATST(input_length=512, prediction_length=128, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="quantile")


def _t_max(module, steps_per_epoch=1000, accumulate=1):
    module.trainer = SimpleNamespace(datamodule=SimpleNamespace(
        train_dataloader=lambda: range(steps_per_epoch)),
        accumulate_grad_batches=accumulate)
    sched = module.configure_optimizers()["lr_scheduler"]["scheduler"]
    return sched._schedulers[1].T_max


def test_schedule_counts_optimizer_steps_under_accumulation():
    """2026-09-10: with accumulate_grad_batches 3 the scheduler (stepped per
    optimizer step) was sized in batches - warmup and cosine 3x too long, the
    whole 30% TimeSSM run was warmup. Batches / accumulation now."""
    m = FinetuneModule(model=_model(), finetune_mode="full_finetune", max_epochs=1,
                       warmup_epochs=0.1, lr_scheduler="cosine", schedule_fraction=0.3)
    assert _t_max(m, steps_per_epoch=3000, accumulate=3) == 300 - 100
    m.trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataloader=lambda: range(3000)),
                                accumulate_grad_batches=3)
    sched = m.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert sched._milestones == [100]


def test_cosine_shortened_to_the_fraction():
    full = FinetuneModule(model=_model(), finetune_mode="full_finetune", max_epochs=1,
                          warmup_epochs=0.1, lr_scheduler="cosine")
    short = FinetuneModule(model=_model(), finetune_mode="full_finetune", max_epochs=1,
                           warmup_epochs=0.1, lr_scheduler="cosine", schedule_fraction=0.3)
    assert _t_max(full) == 1000 - 100
    assert _t_max(short) == 300 - 100
    with pytest.raises(ValueError):
        FinetuneModule(model=_model(), finetune_mode="full_finetune", schedule_fraction=0.0)


def test_trainer_bounded_to_the_fraction():
    from omegaconf import OmegaConf
    from train import apply_schedule_fraction
    cfg = OmegaConf.create({"training": {"schedule_fraction": 0.3}, "trainer": {}})
    kw = apply_schedule_fraction({}, cfg)
    assert kw["limit_train_batches"] == 0.3
    kw = apply_schedule_fraction({"limit_train_batches": 0.5}, cfg)
    assert kw["limit_train_batches"] == 0.5            # an explicit limit wins
    cfg1 = OmegaConf.create({"training": {}, "trainer": {}})
    assert "limit_train_batches" not in apply_schedule_fraction({}, cfg1)


def test_anneal_config_declares_the_fraction_only():
    from hydra import compose, initialize_config_dir
    cdir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=cdir):
        base = compose(config_name="lotsa_mini_v3_head8_zeroshot")
        arm = compose(config_name="lotsa_mini_v3_head8_anneal30_zeroshot")
    assert arm.model.name == "timejepa_lotsa_mini_v3_head8_anneal30_zs"
    assert float(arm.training.schedule_fraction) == 0.3
    assert base.training.get("schedule_fraction") is None
    assert arm.training.max_epochs == base.training.max_epochs == 1
    assert arm.model.decoder.quantile_hidden_dim == base.model.decoder.quantile_hidden_dim
