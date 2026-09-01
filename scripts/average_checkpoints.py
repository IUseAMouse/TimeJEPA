# DEPRECATED (2026-09-01 audit) - one-shot script from a closed round
# (G7.3c soup experiment); kept per the no-delete policy.
"""
Weight averaging ("soup" / SWA) over Lightning checkpoints of one run.

    python scripts/average_checkpoints.py \
        checkpoints/timejepa_lotsa_tiny_mix_zs_1ep3e4/pretrain_False/epoch00_valloss0.58*.ckpt \
        --out checkpoints/champions/soup_1ep3e4.ckpt

    # optional weighting (same order as the files):
    #   --weights 1,2,1

Why (2026-08-24, G7.3c verdict): finetune random-walks in the plateau -
checkpoints of one window oscillate around a basin, tracking the family mix
of the batch stream. The weight AVERAGE is a more central point of the basin
than any individual checkpoint (SWA, Izmailov et al. 2018; the big labs'
weight EMA is the same object). Cost: zero GPU. Verdict: a GIFT eval of the
soup against the selected champion.

Contract:
* only accepts checkpoints with the SAME key set and shapes - loud refusal
  otherwise (averaging two architectures would be silently wrong);
* only FLOATING tensors are averaged; ints/bools (counters, markers) are
  taken from the first checkpoint, with a warning if they differ;
* the output is a minimal Lightning-format checkpoint {'state_dict': ...} -
  loadable by evaluate_gift/evaluate and FinetuneModule like any checkpoint
  (optimizer and scheduler states are NOT kept: a soup is not "resumed", it
  is evaluated or re-annealed).
"""

import argparse
import sys
from pathlib import Path

import torch


def load_state_dict(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    sys.exit(f"{path}: unrecognized format (neither Lightning nor bare state_dict)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("checkpoints", nargs="+", help="Lightning checkpoints to average (>=2)")
    ap.add_argument("--out", required=True, help="path of the averaged checkpoint")
    ap.add_argument("--weights", default=None,
                    help="relative weights, e.g. '1,2,1' (default: uniform)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.checkpoints]
    if len(paths) < 2:
        sys.exit("at least 2 checkpoints are needed to average")
    for p in paths:
        if not p.exists():
            sys.exit(f"not found: {p}")

    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(paths):
            sys.exit(f"{len(weights)} weights for {len(paths)} checkpoints")
    else:
        weights = [1.0] * len(paths)
    total = sum(weights)
    weights = [w / total for w in weights]

    print(f"Soup of {len(paths)} checkpoints:")
    for p, w in zip(paths, weights):
        print(f"  {w:.3f} x {p.name}")

    ref = load_state_dict(paths[0])
    ref_keys = set(ref)
    avg = {}
    non_float_diffs = []

    for k, v in ref.items():
        if torch.is_tensor(v) and v.is_floating_point():
            avg[k] = v.double() * weights[0]
        else:
            avg[k] = v  # taken from the first

    for path, w in zip(paths[1:], weights[1:]):
        sd = load_state_dict(path)
        if set(sd) != ref_keys:
            only_a = sorted(ref_keys - set(sd))[:5]
            only_b = sorted(set(sd) - ref_keys)[:5]
            sys.exit(f"{path.name}: keys differ from the first checkpoint "
                     f"(missing: {only_a} ... / extra: {only_b} ...) - "
                     f"averaging two architectures would be silently wrong")
        for k, v in sd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                if v.shape != ref[k].shape:
                    sys.exit(f"{path.name}: shape of {k} {tuple(v.shape)} != "
                             f"{tuple(ref[k].shape)}")
                avg[k] = avg[k] + v.double() * w
            else:
                same = (torch.equal(v, ref[k]) if torch.is_tensor(v) else v == ref[k])
                if not same:
                    non_float_diffs.append(k)

    for k, v in avg.items():
        if torch.is_tensor(v) and v.is_floating_point():
            avg[k] = v.to(ref[k].dtype)

    if non_float_diffs:
        print(f"  warning: {len(non_float_diffs)} non-float keys differ between "
              f"checkpoints (taken from the first): {non_float_diffs[:5]} ...")

    n_avg = sum(1 for k, v in ref.items()
                if torch.is_tensor(v) and v.is_floating_point())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": avg}, out)
    print(f"{n_avg} float tensors averaged -> {out}")
    print("  Verdict via GIFT eval, never via val_loss (E18i/G7.3c).")


if __name__ == "__main__":
    main()
