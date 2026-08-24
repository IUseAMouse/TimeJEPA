#!/usr/bin/env python
"""
Moyenne de poids (« soup » / SWA) sur des checkpoints Lightning d'un même run.

    python scripts/average_checkpoints.py \
        checkpoints/timejepa_lotsa_tiny_mix_zs_1ep3e4/pretrain_False/epoch00_valloss0.58*.ckpt \
        --out checkpoints/champions/soup_1ep3e4.ckpt

    # pondération optionnelle (même ordre que les fichiers) :
    #   --weights 1,2,1

Pourquoi (2026-08-24, verdict G7.3c) : le finetune marche aléatoirement dans le
plateau — les checkpoints d'une même fenêtre oscillent autour d'un bassin en
pistant le mélange de familles du flux de batchs. La MOYENNE des poids est un
point plus central du bassin que n'importe quel checkpoint individuel (SWA,
Izmailov et al. 2018 ; l'EMA de poids des gros labos est le même objet). Coût :
zéro GPU. Verdict : une éval GIFT du soup contre le champion sélectionné.

Contrat :
* n'accepte que des checkpoints au MÊME ensemble de clés et mêmes formes —
  refus bruyant sinon (moyenner deux architectures serait silencieusement faux) ;
* seuls les tenseurs FLOTTANTS sont moyennés ; les entiers/booléens (compteurs,
  marqueurs) sont pris du premier checkpoint, avec avertissement s'ils diffèrent ;
* la sortie est un checkpoint au format Lightning minimal {'state_dict': ...} —
  chargeable par evaluate_gift/evaluate et FinetuneModule comme n'importe quel
  checkpoint (l'optimiseur et les états de scheduler ne sont PAS conservés :
  un soup ne se « reprend » pas, il s'évalue ou se recuit).
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
    sys.exit(f"✗ {path}: format non reconnu (ni Lightning, ni state_dict nu)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("checkpoints", nargs="+", help="checkpoints Lightning à moyenner (>=2)")
    ap.add_argument("--out", required=True, help="chemin du checkpoint moyenné")
    ap.add_argument("--weights", default=None,
                    help="poids relatifs, ex. '1,2,1' (défaut : uniforme)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.checkpoints]
    if len(paths) < 2:
        sys.exit("✗ il faut au moins 2 checkpoints à moyenner")
    for p in paths:
        if not p.exists():
            sys.exit(f"✗ introuvable : {p}")

    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(paths):
            sys.exit(f"✗ {len(weights)} poids pour {len(paths)} checkpoints")
    else:
        weights = [1.0] * len(paths)
    total = sum(weights)
    weights = [w / total for w in weights]

    print(f"Soup de {len(paths)} checkpoints :")
    for p, w in zip(paths, weights):
        print(f"  {w:.3f} · {p.name}")

    ref = load_state_dict(paths[0])
    ref_keys = set(ref)
    avg = {}
    non_float_diffs = []

    for k, v in ref.items():
        if torch.is_tensor(v) and v.is_floating_point():
            avg[k] = v.double() * weights[0]
        else:
            avg[k] = v  # pris du premier

    for path, w in zip(paths[1:], weights[1:]):
        sd = load_state_dict(path)
        if set(sd) != ref_keys:
            only_a = sorted(ref_keys - set(sd))[:5]
            only_b = sorted(set(sd) - ref_keys)[:5]
            sys.exit(f"✗ {path.name}: clés différentes du premier checkpoint "
                     f"(manquantes: {only_a} … / en trop: {only_b} …) — "
                     f"moyenner deux architectures serait silencieusement faux")
        for k, v in sd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                if v.shape != ref[k].shape:
                    sys.exit(f"✗ {path.name}: forme de {k} {tuple(v.shape)} != "
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
        print(f"  ⚠ {len(non_float_diffs)} clés non-flottantes diffèrent entre "
              f"checkpoints (prises du premier) : {non_float_diffs[:5]} …")

    n_avg = sum(1 for k, v in ref.items()
                if torch.is_tensor(v) and v.is_floating_point())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": avg}, out)
    print(f"✓ {n_avg} tenseurs flottants moyennés -> {out}")
    print("  Verdict par éval GIFT, jamais par val_loss (E18i/G7.3c).")


if __name__ == "__main__":
    main()
