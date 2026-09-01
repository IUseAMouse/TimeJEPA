# DEPRECATED (2026-08-19 audit) - masked-reconstruction path, unused since the
# JEPA objective replaced it. Kept per the no-delete policy; do not import.
"""
Masking strategies for JEPA pretraining.

Creates context and target masks for patch-based JEPA training.
"""

import torch
from typing import Tuple, Literal, Optional
import math


class MaskingStrategy:
    """
    Base class for masking strategies.
    
    Creates boolean masks indicating which patches are context vs target.
    """
    
    def __init__(
        self,
        num_patches: int,
        context_ratio: float = 0.7,
        target_ratio: float = 0.3,
        allow_overlap: bool = False,
    ):
        """
        Args:
            num_patches: Total number of patches in sequence
            context_ratio: Fraction of patches to use as context
            target_ratio: Fraction of patches to use as target
            allow_overlap: Whether context and target can overlap
        """
        self.num_patches = num_patches
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio
        self.allow_overlap = allow_overlap
        
        self.num_context = max(1, int(num_patches * context_ratio))
        self.num_target = max(1, int(num_patches * target_ratio))
    
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate context and target masks.
        
        Args:
            batch_size: Number of masks to generate
            device: Device to create tensors on
        
        Returns:
            context_mask: [B, num_patches] boolean tensor (True = context)
            target_mask: [B, num_patches] boolean tensor (True = target)
        """
        raise NotImplementedError


class RandomMasking(MaskingStrategy):
    """
    Random patch masking.
    
    Randomly sample context and target patches independently.
    """
    
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        context_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            # Random context indices
            context_indices = torch.randperm(self.num_patches)[:self.num_context]
            context_mask[b, context_indices] = True
            
            if self.allow_overlap:
                # Target can overlap with context
                target_indices = torch.randperm(self.num_patches)[:self.num_target]
            else:
                # Target must be disjoint from context
                available = torch.where(~context_mask[b])[0]
                if len(available) < self.num_target:
                    # Not enough non-context patches, allow some overlap
                    target_indices = torch.randperm(self.num_patches)[:self.num_target]
                else:
                    perm = torch.randperm(len(available))[:self.num_target]
                    target_indices = available[perm]
            
            target_mask[b, target_indices] = True
        
        return context_mask, target_mask


class BlockMasking(MaskingStrategy):
    """
    Contiguous block masking.
    
    Context and target are contiguous blocks (like in I-JEPA).
    This is better for temporal structure.
    """
    
    def __init__(
        self,
        num_patches: int,
        context_ratio: float = 0.7,
        target_ratio: float = 0.15,
        num_target_blocks: int = 4,
        allow_overlap: bool = False,
        min_block_size: int = 1,
    ):
        """
        Args:
            num_target_blocks: Number of separate target blocks to predict
            min_block_size: Minimum size of each target block
        """
        super().__init__(num_patches, context_ratio, target_ratio, allow_overlap)
        self.num_target_blocks = num_target_blocks
        self.min_block_size = min_block_size
        
        # Calculate block sizes
        self.target_block_size = max(
            min_block_size,
            self.num_target // num_target_blocks
        )
    
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        context_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            # Create context as a contiguous block
            context_start = torch.randint(0, self.num_patches - self.num_context + 1, (1,)).item()
            context_end = context_start + self.num_context
            context_mask[b, context_start:context_end] = True
            
            # Create multiple target blocks
            for _ in range(self.num_target_blocks):
                if self.allow_overlap:
                    # Target can be anywhere
                    max_start = self.num_patches - self.target_block_size
                    if max_start > 0:
                        target_start = torch.randint(0, max_start + 1, (1,)).item()
                        target_end = target_start + self.target_block_size
                        target_mask[b, target_start:target_end] = True
                else:
                    # Target must be outside context
                    # Try to place target block before or after context
                    available_before = context_start
                    available_after = self.num_patches - context_end
                    
                    if available_before >= self.target_block_size:
                        # Can place before context
                        target_start = torch.randint(0, available_before - self.target_block_size + 1, (1,)).item()
                        target_end = target_start + self.target_block_size
                        target_mask[b, target_start:target_end] = True
                    elif available_after >= self.target_block_size:
                        # Can place after context
                        target_start = context_end + torch.randint(0, available_after - self.target_block_size + 1, (1,)).item()
                        target_end = target_start + self.target_block_size
                        target_mask[b, target_start:target_end] = True
                    else:
                        # Not enough space, allow overlap
                        max_start = self.num_patches - self.target_block_size
                        if max_start > 0:
                            target_start = torch.randint(0, max_start + 1, (1,)).item()
                            target_end = target_start + self.target_block_size
                            target_mask[b, target_start:target_end] = True
        
        return context_mask, target_mask


class TemporalMasking(MaskingStrategy):
    """
    Temporal masking: context = past, target = future.
    
    This is the most natural for time series forecasting.
    Context is always before target in time.
    """
    
    def __init__(
        self,
        num_patches: int,
        context_ratio: float = 0.7,
        target_ratio: float = 0.3,
    ):
        super().__init__(
            num_patches=num_patches,
            context_ratio=context_ratio,
            target_ratio=target_ratio,
            allow_overlap=False  # No overlap in temporal masking
        )
    
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        context_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        
        # Context is first num_context patches
        context_mask[:, :self.num_context] = True
        
        # Target is next num_target patches after context
        target_start = self.num_context
        target_end = min(target_start + self.num_target, self.num_patches)
        target_mask[:, target_start:target_end] = True
        
        return context_mask, target_mask


class RandomTemporalMasking(MaskingStrategy):
    """
    Randomized temporal masking.
    
    Context = random past window
    Target = future window after context
    
    This adds variability while maintaining temporal ordering.
    """
    
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        context_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            # Random start for context (but leave room for target)
            max_context_start = self.num_patches - self.num_context - self.num_target
            max_context_start = max(0, max_context_start)
            
            context_start = torch.randint(0, max_context_start + 1, (1,)).item()
            context_end = context_start + self.num_context
            
            context_mask[b, context_start:context_end] = True
            
            # Target starts after context
            target_start = context_end
            target_end = min(target_start + self.num_target, self.num_patches)
            target_mask[b, target_start:target_end] = True
        
        return context_mask, target_mask


def get_masking_strategy(
    strategy_name: Literal['random', 'block', 'temporal', 'random_temporal'],
    num_patches: int,
    context_ratio: float = 0.7,
    target_ratio: float = 0.3,
    **kwargs
) -> MaskingStrategy:
    """
    Factory function to create masking strategies.
    
    Args:
        strategy_name: Name of strategy
        num_patches: Number of patches
        context_ratio: Fraction for context
        target_ratio: Fraction for target
        **kwargs: Additional arguments for specific strategies
    
    Returns:
        MaskingStrategy instance
    """
    strategies = {
        'random': RandomMasking,
        'block': BlockMasking,
        'temporal': TemporalMasking,
        'random_temporal': RandomTemporalMasking,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown masking strategy: {strategy_name}. "
                        f"Choose from {list(strategies.keys())}")
    
    strategy_cls = strategies[strategy_name]
    
    # Base arguments for all strategies
    base_args = {
        'num_patches': num_patches,
        'context_ratio': context_ratio,
        'target_ratio': target_ratio,
    }
    
    # Strategy-specific arguments
    if strategy_name == 'block':
        # BlockMasking accepts additional arguments
        block_args = {}
        if 'num_target_blocks' in kwargs:
            block_args['num_target_blocks'] = kwargs['num_target_blocks']
        if 'n_context_blocks' in kwargs:  # Support both names
            block_args['num_target_blocks'] = kwargs['n_context_blocks']
        if 'allow_overlap' in kwargs:
            block_args['allow_overlap'] = kwargs['allow_overlap']
        if 'min_block_size' in kwargs:
            block_args['min_block_size'] = kwargs['min_block_size']
        
        return strategy_cls(**base_args, **block_args)
    
    elif strategy_name == 'random':
        # RandomMasking accepts allow_overlap
        random_args = {}
        if 'allow_overlap' in kwargs:
            random_args['allow_overlap'] = kwargs['allow_overlap']
        
        return strategy_cls(**base_args, **random_args)
    
    else:
        # TemporalMasking and RandomTemporalMasking only use base args
        return strategy_cls(**base_args)