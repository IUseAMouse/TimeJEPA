"""
Audit of BATCH COMPOSITION over the epoch (TemperatureSampler).

    python scripts/audit_batch_schedule.py --config-name lotsa_tiny_mix_zeroshot
    python scripts/audit_batch_schedule.py --config-name lotsa_tiny_mix --mode pretrain

Why (2026-08-24): the assumption "batch composition is stable over the epoch"
is FALSE by construction - `TemperatureSampler.__iter__` removes a family
from ALL remaining batches once it hits its `max_oversample_ratio` cap (the
`continue` in datamodule.py), and the batch SHRINKS (lost slots are not
reallocated). Small families, oversampled by T=0.5, die out early; the end of
the epoch is dominated by the big families. Candidate mechanism for the
end-of-finetune GIFT drift (G7.3c: both mix runs bend at the SAME step -
extinction is deterministic).

The script iterates the REAL sampler (no data read: it only produces
indices) and reports:
  * the extinction of each capped family (batch, % of the epoch);
  * composition per epoch decile (batch size, capped vs free family share,
    share of the 5 largest);
  * the start/end imbalance summary.

NB: run single-process, world_size=1 - the NUMBER of batches differs from a
3-GPU run (/3) but the composition PROFILE in % of the epoch is identical.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_datamodule(cfg, is_pretrain: bool, force_ration: bool = False):
    """Replicates the scripts/train.py construction (same config keys)."""
    from omegaconf import OmegaConf
    from timejepa.data.datamodule import MultiDatasetMonashDataModule

    aug_root = cfg.get('augmentations') or {}
    aug_cfg = aug_root.get('pretrain' if is_pretrain else 'finetune') if aug_root else None
    if aug_cfg is not None:
        aug_cfg = OmegaConf.to_container(aug_cfg, resolve=True)
        if not aug_cfg.get('enabled', True):
            aug_cfg = None

    return MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir,
        context_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        datasets=cfg.data.get('datasets') if is_pretrain else cfg.data.get('datasets_finetune'),
        dataset_pattern=cfg.data.get('dataset_pattern', '*.npy'),
        combine_mode=cfg.data.get('combine_mode', 'concatenate'),
        balanced_sampling=cfg.data.balanced_sampling,
        sampling_temperature=cfg.data.sampling_temperature,
        max_oversample_ratio=cfg.data.max_oversample_ratio,
        ration_oversample=force_ration or bool(cfg.data.get('ration_oversample', False)),
        batch_size=cfg.data.batch_size,
        stride=cfg.data.stride,
        normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type,
        clip_outliers=cfg.data.clip_outliers,
        clip_sigma=cfg.data.clip_sigma,
        train_val_test_split=cfg.data.train_val_test_split,
        augmentation_config=aug_cfg,
        multi_resolution_factors=(
            list(cfg.data.get('multi_resolution_factors') or [1]) if is_pretrain else [1]),
        p_multi_resolution=(
            float(cfg.data.get('p_multi_resolution', 0.0)) if is_pretrain else 0.0),
        cross_resolution=bool(cfg.model.get('cross_resolution', False)) if is_pretrain else False,
        seed=cfg.data.seed,
        num_workers=0,
        persistent_workers=False,
        use_mmap=bool(cfg.data.get('use_mmap', False)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config-name", default="lotsa_tiny_mix_zeroshot")
    ap.add_argument("--mode", choices=["auto", "pretrain", "finetune"], default="auto",
                    help="auto = read from the config's training.mode")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--ration", action="store_true",
                    help="force G10.2 rationing (to compare without touching the config)")
    args = ap.parse_args()

    from hydra import compose, initialize
    with initialize(version_base=None, config_path="../configs/model"):
        cfg = compose(config_name=args.config_name)
    is_pretrain = (cfg.training.mode == "pretrain") if args.mode == "auto" \
        else (args.mode == "pretrain")

    dm = build_datamodule(cfg, is_pretrain, force_ration=args.ration)
    dm.prepare_data()
    dm.setup('fit')

    sampler = getattr(dm, "_train_sampler", None)
    if sampler is None:
        sys.exit("no TemperatureSampler (balanced_sampling disabled?) - "
                 "nothing to audit, the DataLoader shuffles uniformly.")

    names = list(getattr(dm, "dataset_names_order", []) or
                 [f"dataset_{i}" for i in range(sampler.num_datasets)])
    sizes = np.asarray(sampler.dataset_sizes)
    offsets = np.asarray(sampler.dataset_offsets)
    bounds = np.concatenate([offsets, [offsets[-1] + sizes[-1]]])
    n_batches = len(sampler)
    n_dec = args.deciles
    capped = np.array([r >= sampler.max_oversample_ratio - 1e-9
                       for r in sampler.oversample_ratios])

    print(f"\n{'='*74}\nBATCH SCHEDULE AUDIT - {args.config_name} "
          f"({'pretrain' if is_pretrain else 'finetune'})")
    print(f"{sampler.num_datasets} families | {n_batches} batches/epoch "
          f"(world_size=1) | nominal batch {sampler.actual_batch_size} | "
          f"T={sampler.temperature} | cap {sampler.max_oversample_ratio}x | "
          f"rationing {'ACTIVE' if getattr(sampler, 'ration_oversample', False) else 'inactive'}")
    print(f"{'='*74}")

    dec_counts = np.zeros((n_dec, sampler.num_datasets), dtype=np.int64)
    dec_nbatch = np.zeros(n_dec, dtype=np.int64)
    extinction = {}

    for b_idx, batch in enumerate(sampler):
        d = min(b_idx * n_dec // n_batches, n_dec - 1)
        fam = np.searchsorted(bounds, np.asarray(batch), side='right') - 1
        cnt = np.bincount(fam, minlength=sampler.num_datasets)
        dec_counts[d] += cnt
        dec_nbatch[d] += 1
        for i in np.nonzero((cnt == 0) & (np.array(sampler.samples_per_dataset) > 0))[0]:
            extinction.setdefault(int(i), b_idx)
        if b_idx + 1 >= n_batches:
            break

    # real extinction = first batch where the family is missing and never returns
    print("\nEXTINCTIONS (capped family removed from the remaining batches):")
    ext_rows = [(names[i], b, 100.0 * b / n_batches)
                for i, b in sorted(extinction.items(), key=lambda kv: kv[1])
                if capped[i]]
    if not ext_rows:
        print("  none - every family covers the whole epoch")
    for name, b, pct in ext_rows[:25]:
        print(f"  {name:<42s} extinct at batch {b:>6d}  ({pct:5.1f} % of the epoch)")
    if len(ext_rows) > 25:
        print(f"  ... and {len(ext_rows) - 25} more")

    top5 = np.argsort(sizes)[-5:]
    print(f"\nCOMPOSITION PER DECILE (batch share, decile mean):")
    print(f"  {'decile':<8s}{'batch size':>13s}{'capped':>12s}"
          f"{'free':>9s}{'top-5 sizes':>15s}")
    for d in range(n_dec):
        tot = dec_counts[d].sum()
        if tot == 0:
            continue
        bs = tot / max(dec_nbatch[d], 1)
        print(f"  {d*100//n_dec:>3d}-{(d+1)*100//n_dec:<3d}%"
              f"{bs:>12.1f}{dec_counts[d][capped].sum()/tot:>11.1%}"
              f"{dec_counts[d][~capped].sum()/tot:>9.1%}"
              f"{dec_counts[d][top5].sum()/tot:>14.1%}")

    first, last = dec_counts[0] / max(dec_counts[0].sum(), 1), \
        dec_counts[-1] / max(dec_counts[-1].sum(), 1)
    movers = np.argsort(np.abs(last - first))[::-1][:8]
    print("\nBIGGEST MOVES start -> end of epoch:")
    for i in movers:
        print(f"  {names[i]:<42s} {first[i]:6.2%} -> {last[i]:6.2%}"
              f"  ({'capped' if capped[i] else 'free'})")

    # Full table - THE instrument of the v3 sizing ("target batch first"):
    # mean batch share over the epoch, per family, sorted. Added 2026-08-27:
    # the audit showed neither the synthetic shares nor the long tail,
    # exactly what step 2 of the runbook must read.
    total_counts = dec_counts.sum(axis=0)
    tot = max(total_counts.sum(), 1)
    order = np.argsort(total_counts)[::-1]
    print(f"\nBATCH SHARE PER FAMILY (epoch mean, {len(names)} families):")
    print(f"  {'family':<42s}{'windows':>12s}{'share':>8s}  status")
    synth_total = 0.0
    for i in order:
        share = total_counts[i] / tot
        if names[i].startswith("synthetic"):
            synth_total += share
        print(f"  {names[i]:<42s}{sampler.samples_per_dataset[i]:>12,d}"
              f"{share:>8.2%}  {'capped' if capped[i] else 'free'}"
              f"{'   <- SYNTH' if names[i].startswith('synthetic') else ''}")
    print(f"\n  SYNTHETIC TOTAL: {synth_total:.1%} of the batch"
          f"  (v3 target: 40-50%)")


if __name__ == "__main__":
    main()
