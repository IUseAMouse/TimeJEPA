"""
Unit tests for model components.
"""

import pytest
import torch

from src.timejepa.models.components.revin import RevIN
from src.timejepa.models.components.rope import RotaryPositionEmbedding
from src.timejepa.models.components.patching import Patching
from src.timejepa.models.components.attention import MultiHeadAttention, TransformerBlock


class TestRevIN:
    """Tests for RevIN normalization."""
    
    def test_forward_backward(self, sample_timeseries):
        """Test forward and backward pass."""
        B, L, C = sample_timeseries.shape
        revin = RevIN(num_features=C)
        
        # Forward (normalize)
        normalized, stats = revin(sample_timeseries, mode='norm')
        
        # Check shape
        assert normalized.shape == sample_timeseries.shape
        
        # Backward (denormalize)
        restored = revin(normalized, mode='denorm', stats=stats)
        
        # Check restoration (should be close to original)
        assert torch.allclose(restored, sample_timeseries, atol=1e-5)
    
    def test_affine_transform(self, sample_timeseries):
        """Test learnable affine parameters."""
        C = sample_timeseries.shape[-1]
        revin = RevIN(num_features=C, affine=True)
        
        # Check parameters exist
        assert revin.affine_weight is not None
        assert revin.affine_bias is not None
        assert revin.affine_weight.shape == (C,)
        assert revin.affine_bias.shape == (C,)
    
    def test_no_affine(self, sample_timeseries):
        """Test without affine parameters."""
        C = sample_timeseries.shape[-1]
        revin = RevIN(num_features=C, affine=False)
        
        assert revin.affine_weight is None
        assert revin.affine_bias is None


class TestRoPE:
    """Tests for Rotary Position Embedding."""
    
    def test_forward(self, sample_patches, d_model):
        """Test RoPE application."""
        B, L, D = sample_patches.shape
        rope = RotaryPositionEmbedding(dim=D)
        
        # Apply RoPE
        embedded = rope(sample_patches)
        
        # Check shape preserved
        assert embedded.shape == sample_patches.shape
    
    def test_different_sequence_lengths(self, batch_size, d_model, device):
        """Test with different sequence lengths."""
        rope = RotaryPositionEmbedding(dim=d_model)
        
        for seq_len in [16, 32, 64]:
            x = torch.randn(batch_size, seq_len, d_model, device=device)
            out = rope(x)
            assert out.shape == x.shape


class TestPatching:
    """Tests for time series patching."""
    
    def test_patch_and_embed(self, sample_timeseries, patch_length, d_model):
        """Test patching and embedding."""
        B, L, C = sample_timeseries.shape
        stride = patch_length // 2
        
        patching = Patching(
            patch_length=patch_length,
            stride=stride,
            in_channels=C,
            d_model=d_model
        )
        
        # Apply patching
        patches = patching(sample_timeseries)
        
        # Check output shape
        expected_num_patches = (L - patch_length) // stride + 1
        assert patches.shape == (B, expected_num_patches, d_model)
    
    def test_stride_variations(self, batch_size, seq_length, num_channels, d_model, device):
        """Test different stride values."""
        patch_length = 8
        
        for stride in [4, 8, 16]:
            x = torch.randn(batch_size, seq_length, num_channels, device=device)
            patching = Patching(patch_length, stride, num_channels, d_model)
            patches = patching(x)
            
            expected_num = (seq_length - patch_length) // stride + 1
            assert patches.shape[1] == expected_num


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention."""
    
    def test_self_attention(self, sample_patches, d_model, n_heads):
        """Test self-attention."""
        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        
        output, attn_weights = attn(sample_patches, sample_patches, sample_patches)
        
        # Check output shape
        assert output.shape == sample_patches.shape
        
        # Check attention weights shape
        B, L = sample_patches.shape[:2]
        assert attn_weights.shape == (B, n_heads, L, L)
    
    def test_cross_attention(self, batch_size, d_model, n_heads, device):
        """Test cross-attention with different Q and K/V lengths."""
        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        
        q = torch.randn(batch_size, 10, d_model, device=device)
        kv = torch.randn(batch_size, 20, d_model, device=device)
        
        output, attn_weights = attn(q, kv, kv)
        
        assert output.shape == q.shape
        assert attn_weights.shape == (batch_size, n_heads, 10, 20)
    
    def test_attention_mask(self, sample_patches, d_model, n_heads):
        """Test with attention mask."""
        B, L = sample_patches.shape[:2]
        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        
        # Create causal mask
        mask = torch.triu(torch.ones(L, L, device=sample_patches.device), diagonal=1).bool()
        
        output, _ = attn(sample_patches, sample_patches, sample_patches, mask=mask)
        
        assert output.shape == sample_patches.shape


class TestTransformerBlock:
    """Tests for Transformer Block."""
    
    def test_forward(self, sample_patches, d_model, n_heads):
        """Test forward pass."""
        block = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=256,
            dropout=0.1
        )
        
        output = block(sample_patches)
        
        assert output.shape == sample_patches.shape
    
    def test_residual_connection(self, sample_patches, d_model, n_heads):
        """Test that residual connections work."""
        block = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=256,
            dropout=0.0  # No dropout for this test
        )
        
        # With residual, output should be different from input
        output = block(sample_patches)
        
        # But not too different (residual connection helps)
        diff = (output - sample_patches).abs().mean()
        assert diff > 0  # Should be different
        assert diff < sample_patches.abs().mean()  # But not completely different