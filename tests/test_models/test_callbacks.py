"""
Unit tests for training callbacks.
"""

import pytest
import torch
import pytorch_lightning as pl

from src.timejepa.training.callbacks import EMACallback, GradientClipCallback


class DummyModule(pl.LightningModule):
    """Dummy Lightning module for testing."""
    
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)
        self.target_layer = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        return self.layer(x)


class TestEMACallback:
    """Tests for EMA callback."""
    
    def test_initialization(self):
        """Test callback initialization."""
        callback = EMACallback(
            momentum_base=0.996,
            momentum_final=1.0,
            max_epochs=100
        )
        
        assert callback.momentum_base == 0.996
        assert callback.momentum_final == 1.0
        assert callback.max_epochs == 100
    
    def test_momentum_schedule(self):
        """Test momentum schedule computation."""
        callback = EMACallback(
            momentum_base=0.996,
            momentum_final=1.0,
            max_epochs=100,
            schedule='cosine'
        )
        
        # At epoch 0, momentum should be close to base
        m0 = callback._compute_momentum(epoch=0, total_epochs=100)
        assert abs(m0 - 0.996) < 0.01
        
        # At final epoch, momentum should be close to final
        m_final = callback._compute_momentum(epoch=99, total_epochs=100)
        assert abs(m_final - 1.0) < 0.01
        
        # Intermediate epochs should be in between
        m_mid = callback._compute_momentum(epoch=50, total_epochs=100)
        assert 0.996 < m_mid < 1.0


class TestGradientClipCallback:
    """Tests for gradient clipping callback."""
    
    def test_initialization(self):
        """Test callback initialization."""
        callback = GradientClipCallback(max_norm=1.0)
        
        assert callback.max_norm == 1.0