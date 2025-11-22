"""
Pytest configuration and fixtures.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def device():
    """PyTorch device (CPU for testing)."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def seq_length():
    """Default sequence length for tests."""
    return 64


@pytest.fixture
def num_channels():
    """Default number of channels for tests."""
    return 3


@pytest.fixture
def patch_length():
    """Default patch length for tests."""
    return 8


@pytest.fixture
def d_model():
    """Default model dimension for tests."""
    return 64


@pytest.fixture
def n_heads():
    """Default number of attention heads."""
    return 4


@pytest.fixture
def n_layers():
    """Default number of layers."""
    return 2


@pytest.fixture
def sample_timeseries(batch_size, seq_length, num_channels, device):
    """Generate random time series data."""
    return torch.randn(batch_size, seq_length, num_channels, device=device)


@pytest.fixture
def sample_patches(batch_size, seq_length, patch_length, num_channels, d_model, device):
    """Generate random patch embeddings."""
    num_patches = (seq_length - patch_length) // (patch_length // 2) + 1
    return torch.randn(batch_size, num_patches, d_model, device=device)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        "seq_length": 64,
        "patch_length": 8,
        "stride": 4,
        "num_channels": 3,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "prediction_length": 16,
    }


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)