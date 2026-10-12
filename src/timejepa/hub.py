"""Load TimeJEPA weights from a Hugging Face repository or a local directory.

    from timejepa.hub import from_pretrained
    judge = from_pretrained("IUseAMouse/TimeJEPA-EBM")          # a pretrain checkpoint
    model = from_pretrained("IUseAMouse/timejepa-mini")         # a finetuned forecaster

A pretrained directory holds `config.json` (the `model` block of the Hydra
config, as `create_model_from_config` reads it, plus `judge_standalone_targets`)
and `model.safetensors` (the state_dict without the target-encoder copy and
the RevIN runtime buffers). `scripts/export_hf.py` writes such a directory
from a training checkpoint.
"""

import json
from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import OmegaConf

from .evaluation.loading import create_model_from_config

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
RUNTIME_BUFFERS = (".mean", ".std")
EXTRA_KEYS = ("judge_standalone_targets",)


def _runtime(key: str) -> bool:
    return "revin" in key and key.endswith(RUNTIME_BUFFERS)


def _resolve(repo_or_path: str, revision: Optional[str], token: Optional[str]) -> Path:
    p = Path(repo_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_or_path, revision=revision, token=token,
                                  allow_patterns=[CONFIG_FILE, WEIGHTS_FILE, "*.json", "*.yaml"]))


def from_pretrained(repo_or_path: str, device: Union[str, torch.device] = "cpu",
                    revision: Optional[str] = None, token: Optional[str] = None):
    """Build the model described by `config.json` and load `model.safetensors`.
    `repo_or_path` is a Hugging Face repository id or a local directory."""
    from safetensors.torch import load_file
    root = _resolve(repo_or_path, revision, token)
    config = json.loads((root / CONFIG_FILE).read_text())
    extras = {k: config.pop(k) for k in EXTRA_KEYS if k in config}
    model = create_model_from_config(OmegaConf.create({"model": config}))
    state = load_file(str(root / WEIGHTS_FILE))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The decoder holds the model's RevIN: its keys alias `revin.*`, which
    # the export writes once; a missing alias is loaded through its twin.
    own = model.state_dict()
    loaded = {own[k].data_ptr() for k in state if k in own}
    missing = [k for k in missing if not _runtime(k) and "target_encoder" not in k
               and own[k].data_ptr() not in loaded]
    if missing or unexpected:
        raise RuntimeError(f"weights/model mismatch: missing {missing[:6]}, unexpected {unexpected[:6]}")
    for k, v in extras.items():
        setattr(model, k, v)
    return model.to(torch.device(device)).eval()


def export(model, config, out_dir: Union[str, Path],
           judge_standalone_targets: Optional[bool] = None) -> Path:
    """Write `config.json` and `model.safetensors` for `from_pretrained`.
    `config` is the `model` block (a DictConfig or a dict)."""
    from safetensors.torch import save_file
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    block = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    if judge_standalone_targets is not None:
        block["judge_standalone_targets"] = bool(judge_standalone_targets)
    (out / CONFIG_FILE).write_text(json.dumps(block, indent=2) + "\n")
    state, seen = {}, set()
    for k, v in model.state_dict().items():
        if _runtime(k) or "target_encoder" in k or v.data_ptr() in seen:
            continue                      # aliases (decoder.revin.* = revin.*) written once
        seen.add(v.data_ptr())
        state[k] = v.detach().cpu().contiguous()
    save_file(state, str(out / WEIGHTS_FILE))
    return out
