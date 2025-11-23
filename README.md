# TimeJEPA: Joint Embedding Predictive Architecture for Time Series

Research implementation of JEPA applied to Multivariate Time Series Forecasting and Representation Learning.

## 🔬 Overview

This project explores the application of non-generative self-supervised learning (SSL) to time series. Unlike Masked Autoencoders which focus on pixel-level reconstruction, TimeJEPA aims to learn semantic representations by predicting the latent embeddings of future time patches.

**Key Features:**
*   **Architecture:** Joint Embedding Predictive Architecture (JEPA) adapted for 1D signals.
*   **Framework:** Built with **PyTorch Lightning** for modularity and **DeepSpeed** integration.
*   **Logging:** Native **MLFlow** integration for experiment tracking.
*   **Data:** Automated pipelines for the **Monash Time Series Forecasting Archive**.

## 🏗️ Architecture

TimeJEPA models (JEPA-TST for now) use a non-generative approach to Time Series forecasting. Instead of predicting the raw signals, TimeJEPA models forecast the latent representation of future values.

![TimeJEPA Architecture](docs/assets/architecture.png)

*Figure 1: The Context Encoder (bottom) learns to predict the representations of the Target Encoder (top). The weights of the Target Encoder are an exponential moving average of the Context Encoder*

## 🚀 Quick Start

### Prerequisites
We use [uv](https://github.com/astral-sh/uv) for extremely fast dependency management and environment isolation.

### Installation

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/IUseAMouse/TimeJEPA.git
    cd ts-jepa
    ```

3.  **Setup Environment:**
    ```bash
    make install
    ```
    This command creates the virtual environment and installs all dependencies defined in `pyproject.toml`.

## 📁 Project Structure

The project follows a `src`-layout for better packaging and reproducibility:

```text
timejepa/
├── data/                   # Data directory (ignored by git)
│   ├── raw/                # Raw .ts files from Monash
│   ├── processed/          # Pre-processed tensors (.npy/.pt)
│   └── checkpoints/        # Model checkpoints
├── src/timejepa/            # Main package
│   ├── data/               # DataModules & Parsers
│   ├── models/             # Encoders, Predictors, Heads
│   ├── training/           # LightningModules
│   └── utils/              # Masking strategies, metrics
├── scripts/                # Executable scripts (entry points)
├── notebooks/              # Jupyter notebooks for analysis
├── Makefile                # Command shortcuts
└── pyproject.toml          # Dependency definitions
```

## 🎯 Usage

### 1. Download & Prepare Data
We utilize the Monash Time Series Archive. The following command downloads specific datasets (e.g., Electricity, Traffic) and converts them to efficient memory-mapped formats.

```bash
make download-data
```

### 2. Train Model (Pre-training)
Launch a JEPA pre-training run. This uses PyTorch Lightning and logs to MLFlow.

```bash
make train
```

### 3. Evaluation (Linear Probing / Fine-tuning)
Evaluate the learned representations on forecasting tasks.

```bash
make evaluate
```

## 🛠️ Development

To ensure code quality and reproducibility, we use `ruff` and `mypy`.

*   **Format code:**
    ```bash
    make format
    ```
*   **Lint code:**
    ```bash
    make lint
    ```
*   **Clean artifacts:**
    ```bash
    make clean
    ```

## 📊 Datasets

The code is designed to work seamlessly with datasets from the [Monash Time Series Forecasting Repository](https://zenodo.org/communities/forecasting), including:
*   Electricity
*   Traffic
*   Weather
*   ETT (via adapter)

<!-- ## 📐 Model Configuration Methodology

### Scaling Law-Based Design

This project follows a **data-driven approach** to determine model architecture, based on scaling laws adapted from language model research (Chinchilla, GPT-3).

#### 1. Dataset Size Analysis

First, compute total available datapoints across all training datasets:

```bash
python scripts/compute_model_config.py --data-dir data
```

This outputs:
- **D** = Total datapoints across all series
- Per-dataset statistics
- Recommended model configurations

#### 2. Optimal Parameter Count

We use the approximation:

```
Optimal Parameters ≈ (D × Epochs) / 8
```

Where:
- **D** = Total datapoints
- **Epochs** = Training epochs (typically 50 for pretraining)
- **8** = Tokens-per-parameter ratio (Chinchilla-derived)

**Rationale**: This ensures the model has sufficient capacity to learn from the data without severe over/under-parameterization.

#### 3. Architecture Dimensions

Given target parameters P, we compute:

```
P ≈ 12 × num_layers × d_model²
```

We test multiple `(num_layers, d_model)` combinations and select based on:
- **Computational budget** (GPU memory, training time)
- **Closest to optimal P**
- **Standard dimensions** (multiples of 64 for efficiency)

**Example configurations**:

| Dataset Size (D) | Optimal Params | Config |
|------------------|----------------|--------|
| 1-5M             | ~10M           | L=4, d=256, H=4  |
| 5-20M            | ~25M           | L=6, d=384, H=6  |
| 20-50M           | ~60M           | L=8, d=512, H=8  |
| 50-100M          | ~150M          | L=12, d=768, H=12 |

#### 4. Context/Prediction Lengths

**Pretraining** (JEPA):
- Variable lengths sampled from ranges
- `context_length`: [96, 512]
- `prediction_length`: [24, 96]
- Teaches the model multi-scale representations

**Finetuning** (Benchmarks):
- Fixed lengths matching benchmark standards
- Common: `context=512`, `prediction=96`
- Also evaluate at multiple horizons (96, 192, 336, 720)

#### 5. Patch Size

Fixed at **16** based on:
- PatchTST findings (optimal for most time series)
- Computational efficiency (512/16 = 32 patches)
- Representational granularity

#### 6. Iterative Refinement

1. **Baseline**: Train small model (10-20M params) on subset
2. **Validate**: Check convergence, overfitting signs
3. **Scale**: Adjust based on validation curves:
   - Underfitting → Increase capacity
   - Overfitting → More data, regularization, or reduce capacity
4. **Repeat**: Until satisfactory performance

### Current Configuration

**Model** (baseline, will update after dataset analysis):
- `d_model`: 256
- `num_layers`: 8
- `num_heads`: 8
- `Total params`: ~25M

**Training**:
- Pretrain: 50 epochs, variable lengths
- Finetune: 20 epochs, fixed benchmark lengths

**To update configuration**:
1. Download all datasets: `make download-all`
2. Analyze and get recommendations: `make analyze-data`
3. Update `configs/model/jepa.yaml` with recommended values
4. Begin training: `make train`

### References

- Chinchilla Scaling Laws (Hoffmann et al., 2022)
- PatchTST (Nie et al., 2023)
- Granite TTM (IBM Research, 2024) -->

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@software{timejepa2025,
  author = {Yvann Vincent},
  title = {TimeJEPA: Joint Embedding Predictive Architecture for Time Series},
  year = {2025},
  url = {https://github.com/IUseAMouse/TimeJEPA}
}
```

## 📄 License

This project is licensed under the MIT License.