"""Write a Hugging Face model directory from a TimeJEPA checkpoint.

    python scripts/export_hf.py --checkpoint checkpoints/timejepa_lotsa_tiny_v3/pretrain_True/epoch01_valloss0.6420.ckpt \
        --model-config lotsa_tiny_v3_eval --standalone-targets \
        --card docs/model_card_timejepa_ebm.md --out export/TimeJEPA-EBM

The directory holds config.json and model.safetensors (read by
`timejepa.hub.from_pretrained`), the original checkpoint, and the model card
as README.md with its Hugging Face header. The exported weights are checked
against the checkpoint: same forecast, and same energies on random candidates,
bit for bit. Nothing is uploaded: that is `hf auth login` then
`hf upload <repo> <dir> . --repo-type model`.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.hub import export, from_pretrained  # noqa: E402
from timejepa.judge import energy  # noqa: E402

HERE = Path(__file__).resolve().parents[1]

HF_HEADER = """---
license: apache-2.0
library_name: timejepa
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - jepa
  - energy-based-model
  - zero-shot
---

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", required=True, help="eval config of the lineage")
    ap.add_argument("--standalone-targets", action="store_true",
                    help="the checkpoint's pretrain encoded targets alone (contextualized_targets: false)")
    ap.add_argument("--card", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights-license", default=None,
                    help="license file copied next to the weights as LICENSE (the header says apache-2.0)")
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(HERE / "configs" / "model")):
        cfg = compose(config_name=args.model_config)
    model = create_model_from_config(cfg)
    load_checkpoint(model, args.checkpoint, torch.device("cpu"))
    out = export(model, cfg.model, args.out, judge_standalone_targets=args.standalone_targets)
    shutil.copy(args.checkpoint, out / Path(args.checkpoint).name)
    card = (HERE / args.card).read_text()
    (out / "README.md").write_text(HF_HEADER + card)
    if args.weights_license:
        shutil.copy(args.weights_license, out / "LICENSE")
    assets = HERE / "docs" / "assets"
    for name in sorted(p.name for p in assets.iterdir() if f"assets/{p.name}" in card):
        (out / "assets").mkdir(exist_ok=True)
        shutil.copy(assets / name, out / "assets" / name)

    back = from_pretrained(str(out))
    assert back.judge_standalone_targets == args.standalone_targets
    rng = np.random.default_rng(0)
    ctx = rng.standard_normal(512).astype(np.float32)
    cands = rng.standard_normal((5, 96)).astype(np.float32)
    with torch.no_grad():
        ea = energy(model, ctx, cands, standalone=args.standalone_targets)[1]
        eb = energy(back, ctx, cands)[1]
        fa = model.forecast(torch.from_numpy(ctx)[None, :, None], n=96)["quantiles_denorm"]
        fb = back.forecast(torch.from_numpy(ctx)[None, :, None], n=96)["quantiles_denorm"]
    assert np.array_equal(ea, eb), "exported weights do not reproduce the energies"
    assert torch.equal(fa, fb), "exported weights do not reproduce the forecast"
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {sorted(f.name for f in out.iterdir())}, {size / 1e6:.1f} MB, round trip exact")


if __name__ == "__main__":
    main()
