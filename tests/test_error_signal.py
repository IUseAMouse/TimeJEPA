"""
Tests de l'arm ESJEPA (ErrorSignal-JEPA, model.error_signal).

Invariants, par gravité :
1. Flag off = strictement RIEN : state_dict, dicts de sortie, composantes de
   loss et INFÉRENCE bit-identiques — protège tous les checkpoints reproduits
   (rétrocompatibilité demandée explicitement par l'utilisateur).
2. Contrats de checkpoint : refus P3.2 dans les DEUX sens (ckpt ESJEPA vs
   modèle nu), la voie z SURVIT au finetune, save_pretrained_encoder l'emporte.
3. Identité à l'init : gate d'étalement zéro-init => le fan est celui de la
   baseline, la modulation n'apparaît que si le gradient la demande.
4. La physique de z_target : grille des patchs respectée, déterminisme, AUCUN
   lookahead (la stat du patch t ne dépend que des valeurs <= sa fin).
5. Refus bruyants : gardes de construction et de forward bilatérales.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST                                      # noqa: E402
from timejepa.models.decoders.quantile_head import QuantileHead          # noqa: E402
from timejepa.models.decoders.linear_decoder import ForecastingHead      # noqa: E402
from timejepa.evaluation.loading import load_checkpoint                  # noqa: E402
from timejepa.training.finetune_module import FinetuneModule             # noqa: E402
from timejepa.training.jepa_pretrain_module import JEPAPretrainModule    # noqa: E402


def _model(error_signal=False, decoder="quantile", predictor="transformer"):
    return JEPATST(input_length=384, prediction_length=96, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_type=predictor, predictor_num_layers=1,
                   predictor_num_heads=4, predictor_d_ff=64,
                   decoder_type=decoder, error_signal=error_signal)


def _save(model, path):
    sd = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd}, path)


def _pretrain_batch():
    torch.manual_seed(0)
    return torch.randn(2, 384, 1), torch.randn(2, 96, 1)


# ---------------------------------------------------------------------------
# 1. Flag off = rien (rétrocompatibilité)
# ---------------------------------------------------------------------------

def test_default_state_dict_has_no_z_keys():
    m = _model(error_signal=False)
    assert not any("z_head" in k or "z_gate" in k for k in m.state_dict())


def test_default_forward_dicts_unchanged():
    m = _model(error_signal=False).eval()
    ctx, tgt = _pretrain_batch()
    with torch.no_grad():
        out = m.forward_pretrain(ctx, tgt)
        assert set(out) == {"predictions", "targets", "context_embeddings"}
        m.set_pretrain_mode(False)
        res = m.forward_finetune(ctx)
    assert "z" not in res


def test_z_loss_component_absent_by_default():
    m = _model(error_signal=False)
    module = JEPAPretrainModule(model=m, loss_type="mse")
    ctx, tgt = _pretrain_batch()
    out = m.forward_pretrain(ctx, tgt)
    _, components = module._compute_loss(out["predictions"], out["targets"], out)
    assert "z" not in components


def test_legacy_checkpoint_still_loads_and_forecasts():
    """Rétrocompatibilité d'inférence (demande utilisateur) : un checkpoint
    d'AVANT l'arm (aucune clé z) chargé dans un modèle flag-off doit passer le
    contrat de chargement et dérouler un forecast complet, rollout compris."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "legacy.ckpt"
        _save(_model(error_signal=False), ckpt)
        m = _model(error_signal=False)
        load_checkpoint(m, str(ckpt), torch.device("cpu"))
        m.set_pretrain_mode(False)
        ctx = torch.randn(2, 384, 1) * 3 + 50
        with torch.no_grad():
            short = m.forecast(ctx, n=96)
            rolled = m.forecast(ctx, n=192)
        for out in (short, rolled):
            assert torch.isfinite(out["forecast_denorm"]).all()
            assert "quantiles_denorm" in out


# ---------------------------------------------------------------------------
# 2. Contrats de checkpoint
# ---------------------------------------------------------------------------

def test_plain_ckpt_refused_by_esjepa_model(tmp_path):
    ckpt = tmp_path / "plain.ckpt"
    _save(_model(error_signal=False), ckpt)
    with pytest.raises(RuntimeError, match="core components"):
        load_checkpoint(_model(error_signal=True), str(ckpt), torch.device("cpu"))


def test_esjepa_ckpt_refused_by_plain_model(tmp_path):
    """Le sens 'unexpected' : évaluer un checkpoint d'arm en amputant sa voie z
    serait silencieusement faux — c'est ce test qui motive la généralisation du
    bras unexpected du refus P3.2 aux préfixes core."""
    ckpt = tmp_path / "esjepa.ckpt"
    _save(_model(error_signal=True), ckpt)
    with pytest.raises(RuntimeError, match="core components"):
        load_checkpoint(_model(error_signal=False), str(ckpt), torch.device("cpu"))


def test_finetune_load_keeps_predictor_z_head(tmp_path):
    """LE contrat de l'arm : la voie z survit au finetune (contrairement au
    recon_head pretrain-only de G6)."""
    src = _model(error_signal=True)
    ckpt = tmp_path / "esjepa.ckpt"
    _save(src, ckpt)
    dst = _model(error_signal=True)
    FinetuneModule(model=dst, pretrained_encoder_path=str(ckpt))
    for (ks, vs), (kd, vd) in zip(src.predictor.z_head.state_dict().items(),
                                  dst.predictor.z_head.state_dict().items()):
        assert ks == kd and torch.equal(vs, vd)


def test_save_pretrained_encoder_carries_z_head(tmp_path):
    m = _model(error_signal=True)
    path = tmp_path / "enc.pt"
    m.save_pretrained_encoder(str(path))
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert any("z_head" in k for k in saved["predictor"])


# ---------------------------------------------------------------------------
# 3. Identité à l'init (gate zéro-init)
# ---------------------------------------------------------------------------

def test_z_gate_identity_at_init():
    torch.manual_seed(0)
    head = QuantileHead(d_model=32, patch_size=16, stride=8,
                        prediction_length=96, use_error_signal=True).eval()
    latents, ctx = torch.randn(2, 11, 32), torch.randn(2, 47, 32)
    with torch.no_grad():
        a = head(latents, ctx, z=torch.randn(2, 11, 4))
        b = head(latents, ctx, z=torch.randn(2, 11, 4) * 100)
    # gate zéro-init : la sortie ignore z EXACTEMENT (pas allclose)
    assert torch.equal(a, b)
    # et le chemin gates=None est identique au chemin gates=exp(0)=1
    raw = torch.randn(2, 96, head.num_quantiles)
    with torch.no_grad():
        assert torch.equal(head._make_monotone(raw),
                           head._make_monotone(raw, torch.zeros(2, 96, 2)))


def test_z_gate_actually_widens_when_nonzero():
    torch.manual_seed(0)
    head = QuantileHead(d_model=32, patch_size=16, stride=8,
                        prediction_length=96, use_error_signal=True).eval()
    raw = torch.randn(2, 96, head.num_quantiles)
    with torch.no_grad():
        base = head._make_monotone(raw)
        wide = head._make_monotone(raw, torch.full((2, 96, 2), 1.0))
    width_base = base[..., -1] - base[..., 0]
    width_wide = wide[..., -1] - wide[..., 0]
    assert (width_wide > width_base).all()
    # la médiane est intouchable par construction
    mid = head.median_idx
    assert torch.equal(base[..., mid], wide[..., mid])


# ---------------------------------------------------------------------------
# 4. La physique de z_target
# ---------------------------------------------------------------------------

def test_residual_stats_geometry_and_determinism():
    m = _model(error_signal=True)
    ctx, tgt = _pretrain_batch()
    z1 = m._residual_stats(ctx, tgt, 11)
    z2 = m._residual_stats(ctx, tgt, 11)
    assert z1.shape == (2, 11, 4)
    assert torch.equal(z1, z2)
    assert torch.isfinite(z1).all()
    assert not z1.requires_grad


def test_residual_stats_have_no_lookahead():
    """La stat du patch 0 (pas 0-15 de la cible) ne doit dépendre d'AUCUNE
    valeur au-delà de la fin du patch — le lissage est causal."""
    m = _model(error_signal=True)
    ctx, tgt = _pretrain_batch()
    z_ref = m._residual_stats(ctx, tgt, 11)
    tgt_pert = tgt.clone()
    tgt_pert[:, 16:] += 1000.0
    z_pert = m._residual_stats(ctx, tgt_pert, 11)
    assert torch.equal(z_ref[:, 0, :], z_pert[:, 0, :])
    assert not torch.equal(z_ref[:, -1, :], z_pert[:, -1, :])


def test_flat_target_stats_are_floored():
    """Cible constante (solar de nuit) : le plancher 1e-3 borne les
    log-échelles, pas de -inf."""
    m = _model(error_signal=True)
    ctx = torch.randn(2, 384, 1)
    tgt = torch.zeros(2, 96, 1)
    z = m._residual_stats(ctx, tgt, 11)
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# 5. Refus bruyants et gardes
# ---------------------------------------------------------------------------

def test_return_z_without_head_raises():
    m = _model(error_signal=False)
    emb = torch.randn(2, 47, 32)
    with pytest.raises(ValueError, match="error_signal"):
        m.predictor.forward_simple(emb, num_targets=11, return_z=True)


def test_quantile_head_requires_z_when_built_for_it():
    head = QuantileHead(d_model=32, patch_size=16, stride=8,
                        prediction_length=96, use_error_signal=True)
    with pytest.raises(ValueError, match="use_error_signal"):
        head(torch.randn(2, 11, 32), torch.randn(2, 47, 32))


def test_z_to_head_without_module_raises():
    head = QuantileHead(d_model=32, patch_size=16, stride=8,
                        prediction_length=96, use_error_signal=False)
    with pytest.raises(ValueError, match="use_error_signal"):
        head(torch.randn(2, 11, 32), torch.randn(2, 47, 32),
             z=torch.randn(2, 11, 4))


def test_point_forecasting_head_refuses_error_signal():
    with pytest.raises(ValueError, match="quantile"):
        ForecastingHead(d_model=32, patch_size=16, stride=8,
                        prediction_length=96, decoder_type="mlp",
                        error_signal=True)


def test_esjepa_rejects_mlp_predictor():
    with pytest.raises(ValueError, match="transformer"):
        _model(error_signal=True, predictor="mlp")


def test_module_rejects_esjepa_with_reconstruction_target():
    with pytest.raises(ValueError, match="reconstruction_target"):
        JEPAPretrainModule(model=_model(error_signal=True), loss_type="mse",
                           error_signal=True, reconstruction_target=True)


def test_module_rejects_flag_mismatch_with_model():
    with pytest.raises(ValueError, match="model.error_signal"):
        JEPAPretrainModule(model=_model(error_signal=False), loss_type="mse",
                           error_signal=True)


# ---------------------------------------------------------------------------
# 6. Bout-en-bout de l'arm
# ---------------------------------------------------------------------------

def test_pretrain_forward_and_loss_carry_z():
    m = _model(error_signal=True)
    module = JEPAPretrainModule(model=m, loss_type="mse", error_signal=True)
    ctx, tgt = _pretrain_batch()
    out = m.forward_pretrain(ctx, tgt)
    assert out["z_predictions"].shape == (2, 11, 4)
    assert out["z_targets"].shape == (2, 11, 4)
    loss, components = module._compute_loss(out["predictions"], out["targets"], out)
    assert "z" in components and torch.isfinite(loss)
    loss.backward()
    # la loss z remonte bien dans le tronc du prédicteur (mécanisme voulu)
    assert m.predictor.z_head[0].weight.grad is not None


def test_rollout_carries_z_through_fan():
    m = _model(error_signal=True).eval()
    m.set_pretrain_mode(False)
    ctx = torch.randn(2, 384, 1) * 3 + 50
    with torch.no_grad():
        out = m.forecast(ctx, n=192)
    assert out["quantiles_denorm"].shape[1] == 192
    assert torch.isfinite(out["quantiles_denorm"]).all()
    # single-shot expose z pour le vérificateur G12
    with torch.no_grad():
        single = m.forward_finetune(ctx)
    assert single["z"].shape == (2, 11, 4)


def test_esjepa_configs_compose_and_build():
    """Le trio Hydra compose et construit un modèle portant les clés z aux
    TROIS sites (modèle, décodeur reconstruit train/loading) — y compris la
    config d'ÉVAL, celle qui a historiquement perdu des flags."""
    from hydra import compose, initialize
    from timejepa.evaluation.loading import create_model_from_config
    with initialize(version_base=None, config_path="../configs/model"):
        for name in ("lotsa_tiny_esjepa", "lotsa_tiny_esjepa_zeroshot",
                     "lotsa_tiny_esjepa_eval"):
            cfg = compose(config_name=name)
            assert cfg.model.error_signal is True
            model = create_model_from_config(cfg)
            assert hasattr(model.predictor, "z_head"), name
            assert hasattr(model.decoder.decoder, "z_gate"), name
