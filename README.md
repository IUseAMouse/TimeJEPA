# TimeJEPA: Joint Embedding Predictive Architecture for Time Series

Research implementation of JEPA applied to Univariate Time Series Forecasting and Representation Learning.

## 🔬 Overview

This project explores the application of non-generative self-supervised learning to time series. Unlike generative models that focus on pixel-level reconstruction, TimeJEPA first aims to learn forecasting-specific semantic representations by predicting the latent embeddings of future time patches.

**Key Features:**
*   **Architecture:** Joint Embedding Predictive Architecture adapted for 1D signals forecasting.
*   **Framework:** Built with **PyTorch** and **PyTorch Lightning**.
*   **4 models:** TimeJEPA-tiny, TimeJEPA-mini, TimeJEPA-base and TimeJEPA-large.

## 🏗️ Architecture

TimeJEPA models  use a non-generative approach to Time Series forecasting. Instead of predicting the raw signals, TimeJEPA models forecast the latent representation of future values.

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
    cd TimeJEPA
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
│   ├── raw/                # Raw .ts/.tsf files from Monash ziped
│   └── processed/          # Pre-processed tensors (.npy/.pt)
├── src/timejepa/            # Main package
│   ├── data/               # DataModules & Parsers
│   ├── models/             # Encoders, Predictors, Heads
│   └── training/           # LightningModules
├── scripts/                # Executable scripts (entry points)
├── Makefile                # Command shortcuts
└── pyproject.toml          # Dependency definitions
```

## 🎯 Usage

### 1. Download & Prepare Data
We utilize the Monash Time Series Archive. The following command downloads specific datasets (e.g., Electricity, Traffic) and converts them to efficient memory-mapped formats.

```bash
make download-data
```

### 2. Pre-Train Model
Launch a JEPA pre-training run. This uses PyTorch Lightning and logs to WandB. Choose a config between *tiny, mini, base and large*.

```bash
make train CONFIG=tiny
```

### 3. Fine tuning
Evaluate the learned representations on forecasting tasks, choose your checkpoint:

```bash
make finetune CHECKPOINT=checkpoint
```

You can also override config parameters:

```bash
make finetune CHECKPOINT=checkpoints/timejepa_mini/pretrain_True/best.ckpt LR=1e-4 MODE=gradual_unfrezze STRIDE=128 EPOCHS=20
```


## 📊 Datasets

The code is designed to work seamlessly with datasets from the [Monash Time Series Forecasting Repository](https://zenodo.org/communities/forecasting), including:
*   Electricity
*   Traffic
*   Weather
*   Oikolab Weather
*   Rain Temperature
*   Australian Weather
*   M4 Hourly
*   Australian Electricity Demand
*   Bitcoin 
*   KDD Cup 2018
*   Saugeenday River flow
*   Solar Power 4 Seconds
*   Solar Power 10 Minutes
*   Sunspot Daily
*   US Births
*   Wind Power 4 Seconds
*   NN5 Daily
*   Melbourne Pedestrian Count
*   Wikipedia Web Traffic Extended
*   London Smart Meters
*   FRED-MD
*   Wind Farms Minutely
*   Rideshare

## 📝 Citation

Coming soon

## 📄 License

This project is licensed under the MIT License.