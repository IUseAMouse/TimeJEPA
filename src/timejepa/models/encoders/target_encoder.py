# src/timejepa/models/encoders/target_encoder.py
"""
Target Encoder for JEPA.

This is a wrapper around PatchTSTEncoder that maintains a separate copy
of the encoder weights, updated via Exponential Moving Average (EMA).

Key JEPA principle:
- Online encoder (context): Updated via backprop
- Target encoder: Updated via EMA (slow-moving copy)
- This creates a momentum-based consistency regularization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from copy import deepcopy

from .patchtst_encoder import PatchTSTEncoder

import math

# Ligne 15 : Changer type hint
class TargetEncoder(nn.Module):
    """
    Target encoder with Exponential Moving Average (EMA) updates.
    
    This wraps an encoder and maintains its weights as a slow-moving
    average of the online encoder's weights.
    
    Key features:
    - Initialized as copy of online encoder
    - Weights updated via EMA (not gradient descent)
    - No gradients computed for target encoder (faster, memory efficient)
    - Cosine EMA schedule (tau increases during training)
    
    Args:
        encoder: The online encoder to create target from (any nn.Module)
        ema_decay: Base EMA decay rate (default: 0.998)
        ema_decay_end: Final EMA decay rate (default: 1.0)
    """
    
    def __init__(
        self,
        encoder: nn.Module,  # 🔥 Générique maintenant
        ema_decay: float = 0.996,
        ema_decay_end: float = 1.0
    ):
        super().__init__()
        
        self.ema_decay_base = ema_decay
        self.ema_decay_end = ema_decay_end
        self.ema_decay = ema_decay  # Current decay
        
        # Create a deep copy of the encoder
        self.encoder = deepcopy(encoder)
        
        # Freeze all parameters (no gradient computation)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Set to eval mode (important for BatchNorm/Dropout if any)
        self.encoder.eval()
    
    def _compute_ema_decay(self, step: int, max_steps: int) -> float:
        """
        Compute EMA decay with cosine schedule.
        
        Formula from BYOL/MoCo v3:
        tau = tau_end - (tau_end - tau_base) * (cos(π * k / K) + 1) / 2
        
        where k = current step, K = max steps
        
        Args:
            step: Current training step
            max_steps: Total training steps
            
        Returns:
            Current EMA decay rate
        """
        if step >= max_steps:
            return self.ema_decay_end
        
        progress = step / max_steps
        decay = self.ema_decay_end - (self.ema_decay_end - self.ema_decay_base) * \
                (math.cos(math.pi * progress) + 1) / 2
        
        return decay
    
    @torch.no_grad()
    def update(self, online_encoder: nn.Module, step: int = 0, max_steps: int = 1000):
        """
        Update target encoder weights using EMA.
        
        Formula: θ_target = τ * θ_target + (1 - τ) * θ_online
        
        Args:
            online_encoder: The online encoder to copy from
            step: Current training step (for EMA schedule)
            max_steps: Total training steps (for EMA schedule)
        """
        # Update decay with schedule
        self.ema_decay = self._compute_ema_decay(step, max_steps)
        
        # Get state dicts
        online_state = online_encoder.state_dict()
        target_state = self.encoder.state_dict()
        
        # Update each parameter
        for key in online_state.keys():
            if 'num_batches_tracked' in key:
                # Skip batch norm tracking (not a learned parameter)
                continue
            
            target_state[key].data.copy_(
                self.ema_decay * target_state[key].data + 
                (1.0 - self.ema_decay) * online_state[key].data
            )
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass (no gradients).
        
        Args:
            x: Input (format depends on encoder type)
            attention_mask: Optional attention mask
            
        Returns:
            Encoded representations
        """
        # Force eval mode (in case someone called .train())
        self.encoder.eval()
        
        # Forward pass without gradients
        return self.encoder(x, attention_mask)
    
    def get_encoder(self) -> nn.Module:
        """Get the underlying encoder (for inspection/saving)."""
        return self.encoder
    
    def copy_from(self, online_encoder: nn.Module):
        """
        Directly copy weights from online encoder (initialization).
        
        Args:
            online_encoder: Encoder to copy from
        """
        self.encoder.load_state_dict(online_encoder.state_dict())
    
    def get_current_ema_decay(self) -> float:
        """Get current EMA decay rate."""
        return self.ema_decay


class EMAUpdater:
    """
    Helper class to manage EMA updates with cosine schedule.
    
    The EMA decay rate can be scheduled to increase during training:
    - Start: lower decay (faster updates)
    - End: higher decay (slower updates, more stable)
    
    This follows BYOL/MoCo v3 practice.
    
    Args:
        base_decay: Base EMA decay rate (e.g., 0.996)
        final_decay: Final EMA decay rate (e.g., 0.999)
        total_steps: Total number of training steps
        use_schedule: Whether to use cosine schedule
    """
    
    def __init__(
        self,
        base_decay: float = 0.996,
        final_decay: float = 0.9996,
        total_steps: int = 100000,
        use_schedule: bool = True
    ):
        self.base_decay = base_decay
        self.final_decay = final_decay
        self.total_steps = total_steps
        self.use_schedule = use_schedule
        self.current_step = 0
    
    def get_current_decay(self) -> float:
        """Get current decay rate based on schedule."""
        if not self.use_schedule:
            return self.base_decay
        
        # Cosine schedule
        progress = min(self.current_step / self.total_steps, 1.0)
        decay = self.final_decay - (self.final_decay - self.base_decay) * \
                (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
        
        return float(decay)
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1
    
    def reset(self):
        """Reset step counter."""
        self.current_step = 0


class DualEncoderWrapper(nn.Module):
    """
    Wrapper that manages both online and target encoders.
    
    This is a convenience class that bundles the online encoder,
    target encoder, and EMA updater together.
    
    Usage:
        dual_encoder = DualEncoderWrapper(encoder_config)
        
        # In training loop:
        z_online = dual_encoder.online(x_context)
        z_target = dual_encoder.target(x_target)
        
        # After backward:
        dual_encoder.update_target()
    
    Args:
        encoder_config: Config dict for PatchTSTEncoder
        ema_decay: Base EMA decay rate
        ema_final_decay: Final EMA decay rate
        total_steps: Total training steps (for EMA schedule)
    """
    
    def __init__(
        self,
        encoder_config: Dict,
        ema_decay: float = 0.996,
        ema_final_decay: float = 0.9996,
        total_steps: int = 100000
    ):
        super().__init__()
        
        # Create online encoder
        self.online = PatchTSTEncoder(**encoder_config)
        
        # Create target encoder (copy of online)
        self.target = TargetEncoder(self.online, ema_decay=ema_decay)
        
        # EMA updater with schedule
        self.ema_updater = EMAUpdater(
            base_decay=ema_decay,
            final_decay=ema_final_decay,
            total_steps=total_steps,
            use_schedule=True
        )
    
    def forward(self, x: torch.Tensor, mode: str = 'online') -> torch.Tensor:
        """
        Forward pass through specified encoder.
        
        Args:
            x: Input tensor
            mode: 'online' or 'target'
            
        Returns:
            Encoded representations
        """
        if mode == 'online':
            return self.online(x)
        elif mode == 'target':
            return self.target(x)
        else:
            raise ValueError(f"mode must be 'online' or 'target', got {mode}")
    
    def update_target(self):
        """Update target encoder using current EMA decay."""
        current_decay = self.ema_updater.get_current_decay()
        self.target.ema_decay = current_decay
        self.target.update(self.online)
        self.ema_updater.step()
    
    def get_ema_decay(self) -> float:
        """Get current EMA decay rate."""
        return self.ema_updater.get_current_decay()