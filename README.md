# TimeJEPA: Joint Embedding Predictive Architecture for Time Series

Research implementation of JEPA applied to Multivariate Time Series Forecasting and Representation Learning.

## 🔬 Overview

This project explores the application of non-generative self-supervised learning (SSL) to time series. Unlike Masked Autoencoders which focus on pixel-level reconstruction, TimeJEPA aims to learn semantic representations by predicting the latent embeddings of future time patches.

**Key Features:**
*   **Architecture:** Joint Embedding Predictive Architecture (JEPA) adapted for 1D signals.
*   **Framework:** Built with **PyTorch Lightning** for modularity and **DeepSpeed** integration.
*   **Logging:** Native **MLFlow** integration for experiment tracking.
*   **Data:** Automated pipelines for the **Monash Time Series Forecasting Archive**.

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
# Or manually:
python scripts/train.py --config configs/pretrain_electricity.yaml
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