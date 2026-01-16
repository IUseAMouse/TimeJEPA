"""
Unit tests for JEPA-TST model.

/!\ Deprecated since I removed the create_jepa_tst_mini, tiny, base and large methods to favor loading configs with hydra
"""

import pytest
import torch
from pathlib import Path

from src.timejepa.models.jepa_tst import JEPATST, create_jepa_tst_tiny, create_jepa_tst_base


class TestJEPATST:
    """Tests for JEPA-TST model."""
    
    def test_pretrain_mode(self, sample_timeseries, device):
        """Test pretrain mode."""
        B, L, C = sample_timeseries.shape
        
        model = create_jepa_tst_tiny(
            seq_length=L,
            num_channels=C,
            patch_length=8,
            stride=4
        ).to(device)
        
        # Set to pretrain mode
        model.set_pretrain_mode(True)
        
        # Create masks
        num_patches = model.encoder.patching.num_patches
        context_mask = torch.rand(B, num_patches, device=device) > 0.3
        target_mask = ~context_mask
        
        # Forward pass
        outputs = model(
            sample_timeseries,
            context_mask=context_mask,
            target_mask=target_mask
        )
        
        # Check outputs
        assert 'predicted_target' in outputs
        assert 'actual_target' in outputs
        assert outputs['predicted_target'].shape == outputs['actual_target'].shape
    
    def test_finetune_mode(self, sample_timeseries, device):
        """Test finetune (forecasting) mode."""
        B, L, C = sample_timeseries.shape
        prediction_length = 16
        
        model = create_jepa_tst_tiny(
            seq_length=L,
            num_channels=C,
            patch_length=8,
            stride=4,
            prediction_length=prediction_length
        ).to(device)
        
        # Set to finetune mode
        model.set_pretrain_mode(False)
        
        # Forward pass
        forecast = model.forecast(sample_timeseries, prediction_length=prediction_length)
        
        # Check output shape
        assert forecast.shape == (B, prediction_length, C)
    
    def test_freeze_unfreeze(self, sample_timeseries, device):
        """Test freezing and unfreezing encoder."""
        B, L, C = sample_timeseries.shape
        
        model = create_jepa_tst_tiny(
            seq_length=L,
            num_channels=C,
            patch_length=8,
            stride=4
        ).to(device)
        
        # Initially, encoder should be trainable
        encoder_params_before = sum(p.requires_grad for p in model.encoder.parameters())
        assert encoder_params_before > 0
        
        # Freeze encoder
        model.freeze_encoder()
        encoder_params_frozen = sum(p.requires_grad for p in model.encoder.parameters())
        assert encoder_params_frozen == 0
        
        # Unfreeze encoder
        model.unfreeze_encoder()
        encoder_params_unfrozen = sum(p.requires_grad for p in model.encoder.parameters())
        assert encoder_params_unfrozen == encoder_params_before
    
    def test_save_load_pretrained(self, sample_timeseries, device, temp_dir):
        """Test saving and loading pretrained encoder."""
        B, L, C = sample_timeseries.shape
        
        model = create_jepa_tst_tiny(
            seq_length=L,
            num_channels=C,
            patch_length=8,
            stride=4
        ).to(device)
        
        # Save encoder
        save_path = temp_dir / "encoder.pt"
        model.save_pretrained_encoder(str(save_path))
        
        assert save_path.exists()
        
        # Create new model and load weights
        model2 = create_jepa_tst_tiny(
            seq_length=L,
            num_channels=C,
            patch_length=8,
            stride=4
        ).to(device)
        
        model2.load_pretrained_encoder(str(save_path))
        
        # Check that weights match
        for p1, p2 in zip(model.encoder.parameters(), model2.encoder.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_different_model_sizes(self, device):
        """Test creating different model sizes."""
        configs = [
            create_jepa_tst_tiny,
            create_jepa_tst_base,
        ]
        
        for create_fn in configs:
            model = create_fn(
                seq_length=64,
                num_channels=3,
                patch_length=8,
                stride=4
            ).to(device)
            
            # Test forward pass
            x = torch.randn(2, 64, 3, device=device)
            forecast = model.forecast(x, prediction_length=16)
            
            assert forecast.shape == (2, 16, 3)