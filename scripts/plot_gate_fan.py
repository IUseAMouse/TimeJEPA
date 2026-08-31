#!/usr/bin/env python
"""
Figure papier : le fan AVEC et SANS le gate z, sur une fenêtre GIFT réelle.

    python scripts/plot_gate_fan.py \\
        --checkpoint checkpoints/champions/esjepa45_mase0.8739_crps0.5981.ckpt \\
        --model-config lotsa_tiny_esjepa_eval \\
        --config solar/10T/short --window 5

L'artefact expérimental à montrer (E21, review 2026-08-31 : « une figure
vaudrait plus que le paragraphe ») : à checkpoint IDENTIQUE, couper le gate
(chirurgie z_gate -> 0 à l'éval, aucun réentraînement) fait exploser le fan
vers son enveloppe worst-case (+18.7 pts de CRPS agrégé, couverture
0.790 -> 0.968) pendant que la MÉDIANE ne bouge pas d'un bit — l'invariance
structurelle de la médiane, visible à l'œil sur une seule fenêtre.

Sortie : paper/figures/gate_fan_<config>.pdf (et .png), deux panneaux
partageant l'axe : vérité + médiane + bande q10-q90, gate ON / gate OFF.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir                    # noqa: E402

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.evaluation import gift                                # noqa: E402
from evaluate_gift import prepare_context                           # noqa: E402


def fan_for(model, ctx: np.ndarray, h: int, device) -> np.ndarray:
    with torch.no_grad():
        out = model.forecast(torch.from_numpy(ctx).reshape(1, -1, 1)
                             .float().to(device), n=h)
    q = out["quantiles_denorm"][0].cpu().numpy()
    return q[..., 0] if q.ndim == 3 else q                          # [h, 9]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint FINETUNÉ d'un arm z (le gate doit exister)")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--config", default="solar/10T/short",
                    help="config GIFT (solar/10T : là où le gate resserre de -30%%)")
    ap.add_argument("--window", type=int, default=0, help="index d'instance")
    ap.add_argument("--gift-root", default="data/gift_eval")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    gate = model.decoder.decoder.z_gate
    if float(gate.weight.abs().sum()) == 0.0:
        raise SystemExit("z_gate à zéro : ce checkpoint n'a pas de gate appris "
                         "(il faut un checkpoint d'arm z finetuné).")

    h = gift.prediction_length(args.config)
    series = gift.load_series(Path(args.gift_root), args.config)
    windows = gift.num_windows(args.config, min(len(s) for s in series))
    insts = [i for i in gift.iter_test_instances(series, h, windows)]
    inst = insts[args.window % len(insts)]
    ctx = prepare_context(inst.context, model.input_length,
                          model.patching.stride, model.patching.patch_size)

    fan_on = fan_for(model, ctx, h, device)
    saved = (gate.weight.detach().clone(), gate.bias.detach().clone())
    with torch.no_grad():                       # chirurgie, puis restauration
        gate.weight.zero_(); gate.bias.zero_()
    fan_off = fan_for(model, ctx, h, device)
    with torch.no_grad():
        gate.weight.copy_(saved[0]); gate.bias.copy_(saved[1])

    med_delta = float(np.abs(fan_on[:, 4] - fan_off[:, 4]).max())
    print(f"médiane |on-off| max : {med_delta:.2e}  (doit être ~0 — invariance)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_ctx = np.arange(-min(len(ctx), 3 * h), 0)
    t_h = np.arange(h)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, fan, title in ((axes[0], fan_on, "gate on"),
                           (axes[1], fan_off, "gate off")):
        ax.plot(t_ctx, ctx[-len(t_ctx):], color="0.55", lw=0.9)
        ax.plot(t_h, inst.target[:h], color="black", lw=1.1, label="truth")
        ax.plot(t_h, fan[:, 4], color="tab:blue", lw=1.3, label="median")
        ax.fill_between(t_h, fan[:, 0], fan[:, 8], color="tab:blue",
                        alpha=0.25, lw=0, label="q10-q90")
        ax.axvline(0, color="red", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{args.config} — same checkpoint, gate surgery only "
                 f"(median identical, |Δ| ≤ {med_delta:.1e})", fontsize=9)
    fig.tight_layout()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = out / f"gate_fan_{args.config.replace('/', '_')}"
    fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.png", dpi=200)
    print(f"Figure : {stem}.pdf")


if __name__ == "__main__":
    main()
