"""
Tests du scaler robuste arcsinh (G8.4, components/robust_scale.py).

Quatre invariants, par gravité :
1. Flag off = strictement RIEN (state_dict, chemins de calcul) — protège tous
   les checkpoints reproduits.
2. Les checkpoints s'auto-décrivent : un mismatch flag/checkpoint REFUSE au
   chargement au lieu de produire des chiffres silencieusement faux (le flag
   ne pèse aucun poids, seul le marqueur le trahit).
3. Les deux pathologies mesurées sont réparées : plancher epsilon (G6, cibles à
   10³σ) et spike qui écrase le signal (E17, moitié domaine).
4. La monotonie : les quantiles dénormalisés restent ordonnés et en échelle brute.
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
# 1. Off = rien
# ---------------------------------------------------------------------------

def test_flag_off_changes_nothing():
    m = _model(robust=False)
    assert m.robust_scaler is None
    assert not any("robust" in k for k in m.state_dict())


# ---------------------------------------------------------------------------
# 2. Auto-description des checkpoints
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
# 3. Les pathologies mesurées
# ---------------------------------------------------------------------------

def test_round_trip_identity():
    rs = RobustScale()
    x = torch.randn(4, 384, 1) * 7 + 100
    rs.fit(x)
    torch.testing.assert_close(rs.inverse(rs.transform(x)), x,
                               rtol=1e-4, atol=1e-4)


def test_epsilon_floor_case_is_tamed():
    """
    G6 : contexte quasi constant + cible qui bouge. Sous RevIN seul, la cible
    normalisée partait à des MILLIERS de sigma. arcsinh est logarithmique en
    queue : la même cible devient un nombre ordinaire.
    """
    rs = RobustScale()
    ctx = torch.full((2, 384, 1), 5.0) + torch.randn(2, 384, 1) * 1e-4
    tgt = torch.full((2, 96, 1), 25.0)                 # saut de 20 unités
    rs.fit(ctx)
    t = rs.transform(tgt)
    assert t.abs().max() < 20, f"cible transformée à {t.abs().max():.1f} — la queue n'est pas compressée"


def test_spike_does_not_crush_the_signal():
    """
    E17 moitié domaine : un spike ×1000 dans le contexte. Le std le laisse
    écraser tout le signal vers zéro ; la MAD l'ignore, arcsinh compresse le
    spike LUI-MÊME. Le corps du signal doit garder une variance de travail.
    """
    body = torch.sin(torch.linspace(0, 20, 384)).reshape(1, 384, 1)
    spiked = body.clone()
    spiked[0, 100, 0] = 1000.0
    rs = RobustScale()
    rs.fit(spiked)
    t = rs.transform(spiked)
    body_std = t[0, 200:, 0].std()                     # loin du spike
    assert body_std > 0.3, f"signal écrasé (std {body_std:.3f}) — la MAD n'a pas joué"
    assert t[0, 100, 0] < 15, "le spike lui-même doit être compressé, pas propagé"


# ---------------------------------------------------------------------------
# 4. Monotonie et bout-en-bout
# ---------------------------------------------------------------------------

def test_forecast_denorm_lives_in_raw_scale_and_quantiles_stay_sorted():
    m = _model(robust=True).eval()
    ctx = torch.randn(3, 384, 1) * 4 + 1000            # échelle brute décalée
    with torch.no_grad():
        out = m.forecast(ctx, n=96)
    fd, qd = out["forecast_denorm"], out["quantiles_denorm"]
    assert torch.isfinite(fd).all() and torch.isfinite(qd).all()
    # échelle brute retrouvée (modèle non entraîné : proche du niveau du contexte)
    assert 500 < fd.mean() < 1500, f"denorm hors échelle brute ({fd.mean():.0f})"
    # sinh est monotone : le fan reste ordonné niveau à niveau
    assert (qd[..., 1:] >= qd[..., :-1] - 1e-4).all(), "quantiles désordonnés après inverse"


def test_finetune_loss_path_runs_in_compressed_space():
    m = _model(robust=True)
    module = FinetuneModule(model=m)
    ctx, tgt = torch.randn(2, 384, 1) * 3 + 50, torch.randn(2, 96, 1) * 3 + 50
    loss, results, target = module._forward_and_loss(ctx, tgt)
    assert torch.isfinite(loss)
    # la cible comparée à la pinball vit bien dans l'espace compressé+RevIN :
    # ordres de grandeur O(1), pas l'échelle brute ~50
    assert target.abs().mean() < 10


def test_flat_plus_spikes_context_stays_invertible():
    """
    Régression (2026-08-22, finetune mix) : une fenêtre « plate + spikes »
    (VM idle — 29 % des contextes bitbrains_rnd ont MAD exactement 0) donnait
    une échelle plancher 1e-8, un repère décalé de ln(1e8) ≈ 18, et l'inverse
    sinh explosait (CRPS 1e10..inf mesurés). Avec le repli 0.1·std, le repère
    reste borné et l'aller-retour d'un fan raisonnable reste FINI et sensé.
    """
    ctx = torch.zeros(1, 384, 1)
    ctx[0, ::40, 0] = 100.0                    # spikes, MAD = 0, std > 0
    tgt = torch.full((1, 96, 1), 50.0)
    rs = RobustScale()
    rs.fit(ctx)
    t = rs.transform(tgt)
    assert t.abs().max() < 15, f"repère encore dégénéré ({t.abs().max():.1f})"
    # un fan de largeur 2 autour de la cible doit s'inverser en valeurs finies
    # et du même ordre de grandeur que la donnée, pas en 1e10
    fan = torch.stack([t - 2, t, t + 2], dim=-1)
    raw = rs.inverse(fan)
    assert torch.isfinite(raw).all()
    assert raw.abs().max() < 1e5, f"inverse encore explosif ({raw.abs().max():.3g})"


def test_strictly_constant_context_is_log_bounded():
    """Contexte strictement constant (std=0 aussi) : le plancher eps=1e-3 borne
    le repère à ~ln(2000·X) au lieu de ln(1e8·X) — plus d'offset +18."""
    ctx = torch.full((1, 384, 1), 5.0)
    tgt = torch.full((1, 96, 1), 25.0)         # saut de 20
    rs = RobustScale()
    rs.fit(ctx)
    t = rs.transform(tgt)
    assert t.abs().max() < 12
    assert torch.isfinite(rs.inverse(t + 3)).all()


def test_healthy_windows_unchanged_by_std_fallback():
    """Sur une fenêtre gaussienne, MAD·1.4826 ≈ std > 0.1·std : le repli est
    inactif et la transformation reste celle d'avant le correctif."""
    x = torch.randn(4, 384, 1) * 7 + 100
    rs = RobustScale()
    rs.fit(x)
    med = x.median(dim=1, keepdim=True).values
    mad = (x - med).abs().median(dim=1, keepdim=True).values * 1.4826
    torch.testing.assert_close(rs.scale, mad, rtol=1e-5, atol=1e-6)
