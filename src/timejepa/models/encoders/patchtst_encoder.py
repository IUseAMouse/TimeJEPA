# Deprecated, now I simply use the bare encoder
"""
PatchTST-style Encoder with RoPE.

This is the main encoder that processes patched time series.
Architecture: Patching → Embedding → Stack of Transformer Blocks → Output

Key features:
- Uses RoPE for position encoding
- Pre-LN transformer blocks (more stable)
- RevIN for normalization
- 24 layers, d_model=512 (from scaling law)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from ..components.revin import RevIN
from ..components.patching import PatchEmbedding
from ..components.attention import TransformerBlock


class PatchTSTEncoder(nn.Module):
    """
    PatchTST-style encoder with RoPE.
    
    Architecture:
        Input [B, L] 
        → RevIN 
        → Patching [B, num_patches, d_model]
        → Transformer Blocks (24 layers)
        → Output [B, num_patches, d_model]
    
    Args:
        d_model: Model dimension (512)
        num_layers: Number of transformer layers (24)
        num_heads: Number of attention heads (8)
        d_ff: Feed-forward dimension (2048)
        patch_size: Size of each patch (16)
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        activation: Activation function ('gelu' or 'relu')
        use_revin: Whether to use RevIN normalization
        num_features: Number of input features (1 for univariate)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 24,
        num_heads: int = 8,
        d_ff: int = 2048,
        patch_size: int = 16,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_revin: bool = True,
        num_features: int = 1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.patch_size = patch_size
        self.use_revin = use_revin
        self.num_features = num_features
        
        # RevIN normalization
        if use_revin:
            self.revin = RevIN(
                num_features=num_features,
                affine=True
            )
        
        # Patch embedding
        # Note: We don't use learned positional encoding here since we use RoPE
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            num_features=num_features,
            dropout=dropout,
            use_pos_encoding=False  # RoPE handles position
        )
        
        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                causal=False  # Non-causal for JEPA
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input time series [B, L] or [B, L, C]
            attention_mask: Optional attention mask [B, L] or [B, num_patches]
            return_hidden_states: If True, returns all layer outputs
            
        Returns:
            Encoded representations [B, num_patches, d_model]
            If return_hidden_states=True, returns tuple of (output, hidden_states)
        """
        # RevIN normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # Patch embedding
        x = self.patch_embedding(x)  # [B, num_patches, d_model]
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        if return_hidden_states:
            hidden_states.append(x)
            return x, hidden_states
        
        return x
    
    def get_num_patches(self, seq_len: int) -> int:
        """Calculate number of patches for a given sequence length."""
        return self.patch_embedding.patching.get_num_patches(seq_len)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize output using stored RevIN statistics.
        
        Args:
            x: Normalized output
            
        Returns:
            Denormalized output
        """
        if self.use_revin:
            return self.revin(x, mode='denorm')
        return x


class ChannelIndependentPatchTSTEncoder(nn.Module):
    """
    Channel-independent variant of PatchTST Encoder.
    
    Processes each channel (feature) independently, then optionally combines.
    Useful for multivariate time series where channels are not strongly correlated.
    
    Args:
        Same as PatchTSTEncoder, but num_features > 1
        channel_mixing: How to combine channels ('mean', 'concat', or None)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 24,
        num_heads: int = 8,
        d_ff: int = 2048,
        patch_size: int = 16,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_revin: bool = True,
        num_features: int = 1,
        channel_mixing: Optional[str] = 'mean'  # 'mean', 'concat', None
    ):
        super().__init__()
        
        assert num_features > 1, "Use PatchTSTEncoder for univariate (num_features=1)"
        
        self.num_features = num_features
        self.channel_mixing = channel_mixing
        
        # Create separate encoder for each channel
        self.encoders = nn.ModuleList([
            PatchTSTEncoder(
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                patch_size=patch_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
                activation=activation,
                use_revin=use_revin,
                num_features=1  # Each encoder handles 1 feature
            )
            for _ in range(num_features)
        ])
        
        # Optional channel mixing layer
        if channel_mixing == 'concat':
            self.channel_mixer = nn.Linear(d_model * num_features, d_model)
        elif channel_mixing == 'mean':
            self.channel_mixer = None  # Just average
        else:
            self.channel_mixer = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [B, L, C] where C = num_features
            attention_mask: Optional mask
            
        Returns:
            Encoded representations [B, num_patches, d_model]
        """
        assert x.dim() == 3, "Input must be [B, L, C]"
        assert x.shape[-1] == self.num_features
        
        # Process each channel independently
        channel_outputs = []
        for i, encoder in enumerate(self.encoders):
            channel_input = x[..., i:i+1]  # [B, L, 1]
            channel_output = encoder(channel_input, attention_mask)
            channel_outputs.append(channel_output)
        
        # Combine channels
        if self.channel_mixing == 'concat':
            # Concatenate along feature dim and project
            x = torch.cat(channel_outputs, dim=-1)  # [B, num_patches, d_model*C]
            x = self.channel_mixer(x)  # [B, num_patches, d_model]
        elif self.channel_mixing == 'mean':
            # Average across channels
            x = torch.stack(channel_outputs, dim=0).mean(dim=0)
        else:
            # Return as list or just first channel
            x = channel_outputs[0]
        
        return x