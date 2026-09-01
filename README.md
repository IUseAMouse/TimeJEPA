# TimeJEPA: Small Joint-Embedding Foundation Forecasters

![TimeJEPA forecasting a von Karman vortex street, one pixel per series](docs/assets/vortex.gif)

TimeJEPA is a family of univariate forecasters built on a Joint Embedding
Predictive Architecture and trained end to end on three RTX 3090s. A single
checkpoint is evaluated zero-shot on all 97 GIFT-Eval configurations, with no
per-dataset adaptation of any kind. The animation above is the 1.14M-parameter
checkpoint forecasting a fluid simulation it has never seen: the red frame
marks the end of the context, and the model calibrates its own uncertainty on
the way (empirical 80% coverage on this scene: 0.79).

## Results

### TimeJEPA-tiny, 1.14M parameters

| Inference | MASE ratio | CRPS ratio |
|---|---|---|
| plain | 0.895 | 0.624 |
| + sign-flip averaging | 0.863 | 0.598 |
| + RateIN | **0.815** | **0.559** |
| seasonal naive | 1.000 | 1.000 |

Geometric mean over the 97 configurations, against the official seasonal
naive numbers. Both inference layers are causal and identical across all
datasets. Sign-flip averaging blends each forecast with the forecast of the
negated context. RateIN mean-pools each context so that its dominant cycle
falls inside the band the model was trained on, at a rate chosen by
backtesting on the past of each series.

Among the sub-10M models of the GIFT-Eval leaderboard, aggregated with the
same formula (parameter counts from the leaderboard metadata):

| Model | Parameters | CRPS ratio |
|---|---|---|
| FlowState-r1.1 | 9.1M | 0.487 |
| TTM-R3 | 1.4M | 0.520 |
| Toto-2.0-4m | 4.1M | 0.524 |
| TempoPFN | n/a | 0.533 |
| TinyCast | 0.1M | 0.545 |
| goia-forecast-nano | 4.7M | 0.553 |
| Kairos-10m | 9.9M | 0.554 |
| Metamorph-4.5M | n/a | 0.555 |
| **TimeJEPA-tiny** | **1.14M** | **0.559** |
| YingLong-6m | 7.3M | 0.609 |

Every number traces to a dated entry in the experiment registry
([docs/EXPERIMENTAL_LOG.md](docs/EXPERIMENTAL_LOG.md)), where predictions are
written down before the runs. The technical report lives in
[paper/](paper/).

### TimeJEPA-mini, 3.4M parameters

Pretraining in progress.

## Usage

```python
import numpy as np
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir

from timejepa.evaluation import create_model_from_config, load_checkpoint

with initialize_config_dir(config_dir=str(Path("configs/model").resolve()),
                           version_base=None):
    cfg = compose(config_name="lotsa_tiny_v3_eval")

model = create_model_from_config(cfg)
model = load_checkpoint(model, "checkpoints/<checkpoint>.ckpt",
                        torch.device("cpu")).eval()

history = np.asarray(my_series, dtype=np.float32)    # raw values, any scale
x = torch.from_numpy(history[-1024:])[None, :, None]

with torch.no_grad():
    out = model.forecast(x, n=192)                   # any horizon

median = out["forecast_denorm"][0, :, 0]             # [n]
fan = out["quantiles_denorm"][0]                     # [n, 9], levels 0.1 to 0.9
```

Normalization lives inside the model: feed raw values, read raw values back.
The quantile fan cannot cross by construction. Interpolate NaNs in the
context before calling. `scripts/evaluate_gift.py` is the reference
inference pipeline; sign-flip averaging and RateIN live there.

Checkpoints are not distributed yet. They will be published on Hugging Face
together with the preprint. Until then, the training commands below produce
one in about a day on three GPUs.

## Setup

[uv](https://github.com/astral-sh/uv) manages the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/IUseAMouse/TimeJEPA.git
cd TimeJEPA
make install
```

Evaluation data, the 97 GIFT-Eval configurations (a few GB):

```bash
make gift-download
```

Pretraining corpus: LOTSA plus a seeded synthetic generator, streamed and
converted once. Subsets overlapping any evaluation benchmark are excluded by
verified name patterns; the decontamination audit is in the experiment
registry.

```bash
python scripts/prepare_lotsa.py --out data/processed/lotsa
```

The TTM hybrid experiments need one extra dependency:

```bash
uv pip install -e ".[ttm]"
```

## Architecture

![TimeJEPA architecture](docs/assets/architecture.png)

The model forecasts in embedding space. An encoder cuts the context into
patches of 16 steps and embeds them with a RoPE transformer. A predictor
reads those embeddings through learned queries, one query per future patch,
and produces the embeddings of the future. A quantile head decodes each of
them into nine quantiles: the median is regressed directly and the other
levels are stacked on top of it as positive widths, so the fan cannot cross.

Two normalizations let one checkpoint work at any scale: a robust arcsinh
compression that tames spikes, then a per-window standardization. Both are
inverted at the output.

During pretraining, the target embeddings come from a slow moving-average
copy of the encoder, and a variance regularizer keeps the embedding space
from collapsing. The forecaster is then finetuned for one epoch with a
pinball loss on the quantile fan.

## Training and evaluation

```bash
# Pretrain (lotsa_tiny_v3 is the current recipe)
python scripts/train.py --config-name lotsa_tiny_v3 wandb.run_name=my-run

# Finetune (one cosine epoch, every checkpoint kept)
python scripts/train.py --config-name lotsa_tiny_v3_zeroshot \
    '+training.pretrained_encoder_path="checkpoints/.../<ckpt>"'

# GIFT-Eval, 97 configs, under 15 minutes on one 3090
python scripts/evaluate_gift.py --config-name lotsa_tiny_v3_eval \
    '+checkpoint_path="checkpoints/.../<ckpt>"' +tta_flip=true +ratein=backtest

# Video demo: every pixel forecast as an independent series
python scripts/forecast_video.py

# Build the paper
cd paper && make
```

## Project structure

```text
src/timejepa/             # data, models, training, evaluation
scripts/                  # entry points: train, evaluate_gift, probes, demos
configs/model/            # one-variable config lineages
docs/EXPERIMENTAL_LOG.md  # dated experiment registry, predictions first
paper/                    # LaTeX technical report
evaluation/               # per-config JSON results, cached
```

## Citation

Paper in preparation. Until the arXiv version lands:

```bibtex
@misc{vincent2026timejepa,
  author = {Vincent, Yvann},
  title  = {TimeJEPA: Small Joint-Embedding Foundation Forecasters},
  year   = {2026},
  url    = {https://github.com/IUseAMouse/TimeJEPA}
}
```

## License

[MIT](LICENSE)
