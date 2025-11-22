# src/timejepa/models/components/patching.py
"""
Patching module for time series.

Similar to Vision Transformer (ViT) patching but for 1D sequences.
Converts continuous time series into discrete patches.

Key idea: [B, L] -> [B, num_patches, patch_size * d_model]
"""

import torch
import torch.nn as nn
from typing import Optional


class Patching(nn.Module):
    """
    Convert time series into patches and embed them.
    
    Takes a time series of length L and converts it into num_patches patches,
    each of length patch_size. Then projects each patch to d_model dimensions.
    
    Args:
        patch_size: Number of timesteps in each patch
        d_model: Embedding dimension
        num_features: Number of input features/channels (default: 1 for univariate)
        stride: Stride for patching (if None, uses patch_size for non-overlapping)
        padding: Whether to pad the sequence to make it divisible by patch_size
    """
    
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_features: int = 1,
        stride: Optional[int] = None,
        padding: bool = True
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_features = num_features
        self.stride = stride if stride is not None else patch_size
        self.padding = padding
        
        # Linear projection from (patch_size * num_features) -> d_model
        self.projection = nn.Linear(patch_size * num_features, d_model)
        
        # Optional: learnable value embedding (as in PatchTST)
        # This can help the model learn better representations
        self.value_embedding = nn.Linear(num_features, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to patches.
        
        Args:
            x: Input tensor [B, L] or [B, L, C]
            
        Returns:
            Patched tensor [B, num_patches, d_model]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Handle 2D input [B, L] -> [B, L, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, L, 1]
        
        num_features = x.shape[-1]
        
        # Pad if necessary
        if self.padding:
            # Calculate padding needed
            remainder = (seq_len - self.patch_size) % self.stride
            if remainder != 0:
                pad_len = self.stride - remainder
                # Pad at the end with last value (or could use zeros)
                x = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
                seq_len = x.shape[1]
        
        # Calculate number of patches
        num_patches = (seq_len - self.patch_size) // self.stride + 1
        
        # Create patches using unfold
        # x: [B, L, C] -> patches: [B, num_patches, patch_size, C]
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        # patches: [B, num_patches, C, patch_size]
        patches = patches.transpose(2, 3)  # [B, num_patches, patch_size, C]
        
        # Flatten patches: [B, num_patches, patch_size * C]
        patches = patches.reshape(batch_size, num_patches, -1)
        
        # Project to d_model
        patches = self.projection(patches)  # [B, num_patches, d_model]
        
        return patches
    
    def get_num_patches(self, seq_len: int) -> int:
        """
        Calculate number of patches for a given sequence length.
        
        Args:
            seq_len: Length of input sequence
            
        Returns:
            Number of patches
        """
        if self.padding:
            remainder = (seq_len - self.patch_size) % self.stride
            if remainder != 0:
                seq_len += (self.stride - remainder)
        
        return (seq_len - self.patch_size) // self.stride + 1


class PatchEmbedding(nn.Module):
    """
    Full patch embedding with optional positional encoding.
    
    This is a higher-level module that combines patching with
    optional learned positional embeddings (separate from RoPE).
    
    Args:
        patch_size: Size of each patch
        d_model: Embedding dimension
        num_features: Number of input features
        max_patches: Maximum number of patches (for learned pos encoding)
        dropout: Dropout rate
        use_pos_encoding: Whether to add learned positional encoding
    """
    
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_features: int = 1,
        max_patches: int = 512,
        dropout: float = 0.1,
        use_pos_encoding: bool = False  # We use RoPE instead usually
    ):
        super().__init__()
        
        self.patching = Patching(patch_size, d_model, num_features)
        self.dropout = nn.Dropout(dropout)
        
        # Learned positional encoding (optional, usually don't use with RoPE)
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed patches.
        
        Args:
            x: Input [B, L] or [B, L, C]
            
        Returns:
            Embedded patches [B, num_patches, d_model]
        """
        # Patchify
        x = self.patching(x)  # [B, num_patches, d_model]
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            num_patches = x.shape[1]
            x = x + self.pos_encoding[:, :num_patches, :]
        
        # Dropout
        x = self.dropout(x)
        
        return x


class UnPatching(nn.Module):
    """
    Convert patches back to continuous time series.
    
    This is used in the decoder to reconstruct the original sequence
    from patch-level predictions.
    
    Args:
        patch_size: Size of each patch
        d_model: Embedding dimension
        num_features: Number of output features
        stride: Stride used in patching
    """
    
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_features: int = 1,
        stride: Optional[int] = None
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_features = num_features
        self.stride = stride if stride is not None else patch_size
        
        # Project from d_model -> (patch_size * num_features)
        self.projection = nn.Linear(d_model, patch_size * num_features)
    
    def forward(
        self, 
        x: torch.Tensor, 
        target_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Convert patches back to continuous sequence.
        
        Args:
            x: Patched tensor [B, num_patches, d_model]
            target_len: Target sequence length (if None, inferred from patches)
            
        Returns:
            Continuous sequence [B, L, C]
        """
        batch_size, num_patches, _ = x.shape
        
        # Project to patch values
        x = self.projection(x)  # [B, num_patches, patch_size * C]
        
        # Reshape to [B, num_patches, patch_size, C]
        x = x.reshape(batch_size, num_patches, self.patch_size, self.num_features)
        
        # If non-overlapping patches (stride == patch_size), simple reshape
        if self.stride == self.patch_size:
            # Merge patches: [B, num_patches * patch_size, C]
            x = x.reshape(batch_size, -1, self.num_features)
        else:
            # Overlapping patches: need to average overlapping regions
            seq_len = (num_patches - 1) * self.stride + self.patch_size
            output = torch.zeros(
                batch_size, seq_len, self.num_features,
                device=x.device, dtype=x.dtype
            )
            counts = torch.zeros(seq_len, device=x.device, dtype=x.dtype)
            
            for i in range(num_patches):
                start = i * self.stride
                end = start + self.patch_size
                output[:, start:end, :] += x[:, i, :, :]
                counts[start:end] += 1
            
            x = output / counts.unsqueeze(0).unsqueeze(-1)
        
        # Trim to target length if specified
        if target_len is not None:
            x = x[:, :target_len, :]
        
        return x