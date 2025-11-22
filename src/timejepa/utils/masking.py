# src/timejepa/utils/masking.py
"""
Masking strategies for JEPA training.
"""
import logging
from typing import Tuple, Literal

import torch

logger = logging.getLogger(__name__)


class MaskGenerator:
    """Base class for mask generation strategies."""
    
    def __call__(self, batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate binary mask.
        
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            device: Device to create mask on
        
        Returns:
            Boolean mask of shape (batch_size, seq_length)
            True = masked (hidden), False = visible
        """
        raise NotImplementedError


class RandomMask(MaskGenerator):
    """Randomly mask individual timesteps."""
    
    def __init__(self, mask_ratio: float = 0.15):
        """
        Args:
            mask_ratio: Fraction of timesteps to mask
        """
        self.mask_ratio = mask_ratio
    
    def __call__(self, batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
        """Generate random mask."""
        num_masked = int(seq_length * self.mask_ratio)
        
        mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            masked_indices = torch.randperm(seq_length, device=device)[:num_masked]
            mask[i, masked_indices] = True
        
        return mask


class BlockMask(MaskGenerator):
    """Mask contiguous blocks (more realistic for time series)."""
    
    def __init__(
        self,
        mask_ratio: float = 0.15,
        min_block_size: int = 5,
        max_block_size: int = 20
    ):
        """
        Args:
            mask_ratio: Target fraction of timesteps to mask
            min_block_size: Minimum block size
            max_block_size: Maximum block size
        """
        self.mask_ratio = mask_ratio
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
    
    def __call__(self, batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
        """Generate block mask."""
        mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        
        num_masked_target = int(seq_length * self.mask_ratio)
        
        for i in range(batch_size):
            num_masked = 0
            
            while num_masked < num_masked_target:
                # Random block size
                block_size = torch.randint(
                    self.min_block_size,
                    self.max_block_size + 1,
                    (1,),
                    device=device
                ).item()
                
                # Random start position
                max_start = max(0, seq_length - block_size)
                if max_start == 0:
                    break
                
                start = torch.randint(0, max_start, (1,), device=device).item()
                end = min(start + block_size, seq_length)
                
                # Mask the block
                mask[i, start:end] = True
                num_masked += (end - start)
        
        return mask


class ContextTargetMask(MaskGenerator):
    """
    Generate complementary context and target masks for JEPA.
    
    Context: visible past
    Target: hidden future to predict
    """
    
    def __init__(self, target_ratio: float = 0.3):
        """
        Args:
            target_ratio: Fraction of sequence to use as target
        """
        self.target_ratio = target_ratio
    
    def __call__(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate context and target masks.
        
        Returns:
            Tuple of (context_mask, target_mask)
            Both are boolean tensors of shape (batch_size, seq_length)
        """
        target_len = int(seq_length * self.target_ratio)
        context_len = seq_length - target_len
        
        # Simple split: first part = context, last part = target
        context_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        
        context_mask[:, :context_len] = True
        target_mask[:, context_len:] = True
        
        return context_mask, target_mask


def get_mask_generator(name: str, **kwargs) -> MaskGenerator:
    """Factory function for mask generators."""
    generators = {
        "random": RandomMask,
        "block": BlockMask,
        "context_target": ContextTargetMask,
    }
    
    if name not in generators:
        raise ValueError(f"Unknown mask generator: {name}. Available: {list(generators.keys())}")
    
    return generators[name](**kwargs)