# ARCHIVED - not wired to live code, do not import (see scripts/archive/README.md).
"""
Compute optimal TimeJEPA configuration based on dataset size and scaling laws.

Warning: deprecated, especially since I wasn't computing effective tokens per training
by taking patching into consideration at this point. 
"""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class JEPAConfig:
    """TimeJEPA model configuration."""
    name: str
    d_model: int
    n_heads: int
    encoder_layers: int
    predictor_layers: int
    decoder_layers: int
    d_ff_ratio: int = 4
    
    @property
    def d_ff(self) -> int:
        return self.d_model * self.d_ff_ratio
    
    @property
    def encoder_params(self) -> int:
        """Transformer encoder parameters."""
        per_layer = (
            4 * self.d_model ** 2 +          # Self-attention (Q, K, V, O)
            2 * self.d_model * self.d_ff +   # FFN
            4 * self.d_model                  # LayerNorms
        )
        return self.encoder_layers * per_layer
    
    @property
    def predictor_params(self) -> int:
        """Predictor transformer parameters."""
        per_layer = (
            4 * self.d_model ** 2 +
            2 * self.d_model * self.d_ff +
            4 * self.d_model
        )
        return self.predictor_layers * per_layer
    
    @property
    def decoder_params(self) -> int:
        """Attentive decoder parameters."""
        # Cross-attention + FFN style decoder
        per_layer = (
            4 * self.d_model ** 2 +          # Cross-attention
            2 * self.d_model * (self.d_model * 2) +  # MLP (hidden_dim = 2*d)
            2 * self.d_model
        )
        return self.decoder_layers * per_layer
    
    @property
    def embedding_params(self) -> int:
        """Patch embedding parameters."""
        patch_size = 16  # Default
        return patch_size * self.d_model + self.d_model  # Linear + bias
    
    @property
    def total_params(self) -> int:
        """Total trainable parameters (encoder + predictor + decoder)."""
        # Note: target_encoder is EMA copy, not separately trained
        return (
            self.encoder_params + 
            self.predictor_params + 
            self.decoder_params + 
            self.embedding_params
        )
    
    def __str__(self) -> str:
        return (
            f"{self.name}: d={self.d_model}, h={self.n_heads}, "
            f"L_enc={self.encoder_layers}, L_pred={self.predictor_layers}, "
            f"L_dec={self.decoder_layers} -> {self.total_params/1e6:.1f}M params"
        )


# Predefined configs (like your YAML)
PRESET_CONFIGS = [
    JEPAConfig("nano",   d_model=64,  n_heads=2, encoder_layers=2, predictor_layers=1, decoder_layers=1),
    JEPAConfig("tiny",   d_model=128, n_heads=4, encoder_layers=3, predictor_layers=2, decoder_layers=2),
    JEPAConfig("small",  d_model=256, n_heads=4, encoder_layers=4, predictor_layers=2, decoder_layers=2),
    JEPAConfig("base",   d_model=384, n_heads=6, encoder_layers=6, predictor_layers=3, decoder_layers=2),
    JEPAConfig("large",  d_model=512, n_heads=8, encoder_layers=8, predictor_layers=4, decoder_layers=3),
    JEPAConfig("xlarge", d_model=768, n_heads=12, encoder_layers=12, predictor_layers=4, decoder_layers=3),
]


def compute_effective_tokens(
    total_points: int,
    epochs: int,
    context_length: int = 384,
    stride: int = 1
) -> dict:
    """
    Compute effective training tokens accounting for windowing and epochs.
    """
    # Approximate number of windows (samples)
    # Each series of length L gives roughly (L - context_length) / stride windows
    # Simplified: assume each point contributes ~1 window on average
    num_windows = total_points // stride
    
    # Each window = context_length tokens
    tokens_per_epoch = num_windows * context_length
    
    # Effective tokens with diminishing returns for repeated epochs
    # Using sqrt scaling for epochs > 1 (empirical heuristic)
    if epochs <= 1:
        effective_epochs = epochs
    else:
        # First epoch counts fully, subsequent have diminishing returns
        effective_epochs = 1 + np.sqrt(epochs - 1)
    
    effective_tokens = tokens_per_epoch * effective_epochs
    
    return {
        'total_points': total_points,
        'epochs': epochs,
        'effective_epochs': effective_epochs,
        'tokens_per_epoch': tokens_per_epoch,
        'total_tokens': tokens_per_epoch * epochs,
        'effective_tokens': effective_tokens,
    }


def compute_optimal_params(
    effective_tokens: float,
    regime: str = "representation"
) -> Tuple[int, dict]:
    """
    Compute optimal parameter count based on scaling laws.
    
    Regimes:
    - "chinchilla": Original LLM scaling (tokens/params ~ 20)
    - "representation": For JEPA/contrastive (more capacity needed)
    - "conservative": Avoid overfitting (larger ratio)
    """
    ratios = {
        "chinchilla": 20,       # Original Chinchilla
        "representation": 12,   # JEPA needs more capacity for good representations
        "conservative": 30,     # When data is limited or noisy
    }
    
    ratio = ratios.get(regime, 15)
    optimal = int(effective_tokens / ratio)
    
    # Compute range (±50%)
    param_range = {
        'min': int(optimal * 0.5),
        'optimal': optimal,
        'max': int(optimal * 1.5),
        'regime': regime,
        'ratio': ratio,
    }
    
    return optimal, param_range


def find_best_config(
    target_params: int,
    param_range: dict,
    configs: List[JEPAConfig] = PRESET_CONFIGS
) -> Tuple[JEPAConfig, List[JEPAConfig]]:
    """
    Find best matching config and viable alternatives.
    """
    # Sort by distance to target
    scored = sorted(configs, key=lambda c: abs(c.total_params - target_params))
    
    best = scored[0]
    
    # Find configs within acceptable range
    viable = [c for c in configs 
              if param_range['min'] <= c.total_params <= param_range['max']]
    
    return best, viable


def generate_custom_config(
    target_params: int,
    encoder_layers: int = 6
) -> JEPAConfig:
    """
    Generate a custom config to match target params.
    """
    # Solve for d_model given layers and target params
    # Simplified: total ~ 14 * L_total * d^2 
    L_total = encoder_layers + 2 + 2  # encoder + predictor + decoder
    
    d_model = int(np.sqrt(target_params / (14 * L_total)))
    
    # Round to multiple of 64
    d_model = max(64, ((d_model + 32) // 64) * 64)
    
    # Determine heads
    if d_model >= 512:
        n_heads = 8
    elif d_model >= 384:
        n_heads = 6
    elif d_model >= 256:
        n_heads = 4
    else:
        n_heads = max(2, d_model // 32)
    
    # Ensure divisibility
    d_model = (d_model // n_heads) * n_heads
    
    # Scale predictor/decoder with encoder
    predictor_layers = max(2, encoder_layers // 2)
    decoder_layers = 2
    
    return JEPAConfig(
        name="custom",
        d_model=d_model,
        n_heads=n_heads,
        encoder_layers=encoder_layers,
        predictor_layers=predictor_layers,
        decoder_layers=decoder_layers
    )


def print_analysis(
    total_points: int,
    epochs: int,
    context_length: int = 384
):
    """Full scaling law analysis."""
    
    print("\n" + "=" * 80)
    print("TIMEJEPA SCALING LAW ANALYSIS")
    print("=" * 80)
    
    token_stats = compute_effective_tokens(total_points, epochs, context_length)
    
    print(f"\nDATA STATISTICS:")
    print(f"  Total datapoints:         {token_stats['total_points']:>15,}")
    print(f"  Context length:           {context_length:>15}")
    print(f"  Epochs:                   {token_stats['epochs']:>15}")
    print(f"  Tokens per epoch:         {token_stats['tokens_per_epoch']:>15,}")
    print(f"  Total tokens (raw):       {token_stats['total_tokens']:>15,}")
    print(f"  Effective epochs*:        {token_stats['effective_epochs']:>15.2f}")
    print(f"  Effective tokens*:        {token_stats['effective_tokens']:>15,.0f}")
    print(f"\n  * Accounts for diminishing returns of repeated data")
    
    print(f"\nOPTIMAL PARAMETER COUNTS:")
    print("-" * 80)
    
    for regime in ["chinchilla", "representation", "conservative"]:
        optimal, param_range = compute_optimal_params(
            token_stats['effective_tokens'], 
            regime
        )
        print(f"  {regime.capitalize():15} (ratio {param_range['ratio']:2}:1): "
              f"{param_range['min']/1e6:>6.1f}M - {param_range['optimal']/1e6:>6.1f}M - {param_range['max']/1e6:>6.1f}M")
    
    
    optimal_params, param_range = compute_optimal_params(
        token_stats['effective_tokens'], 
        "representation"
    )
    
    print(f"\n" + "=" * 80)
    print("PRESET CONFIGURATIONS")
    print("=" * 80)
    print(f"\n{'Name':<10} {'d_model':>8} {'heads':>6} {'L_enc':>6} {'L_pred':>7} {'L_dec':>6} {'Params':>12} {'Status':<15}")
    print("-" * 80)
    
    best_config, viable_configs = find_best_config(optimal_params, param_range)
    
    for config in PRESET_CONFIGS:
        params_str = f"{config.total_params/1e6:.1f}M"
        
        if config.name == best_config.name:
            status = "⭐ RECOMMENDED"
        elif config in viable_configs:
            status = "Viable"
        elif config.total_params < param_range['min']:
            status = "Too small"
        else:
            status = "Too large"
        
        print(f"{config.name:<10} {config.d_model:>8} {config.n_heads:>6} "
              f"{config.encoder_layers:>6} {config.predictor_layers:>7} {config.decoder_layers:>6} "
              f"{params_str:>12} {status:<15}")
    
    
    print(f"\n" + "=" * 80)
    print("CUSTOM CONFIGURATION (optimized for your data)")
    print("=" * 80)
    
    for n_layers in [4, 6, 8]:
        custom = generate_custom_config(optimal_params, n_layers)
        indicator = " ⭐" if n_layers == 6 else ""
        print(f"  {custom}{indicator}")
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDED YAML CONFIG")
    print("=" * 80)
    
    rec = best_config if best_config.total_params <= param_range['max'] else generate_custom_config(optimal_params, 6)
    
    print(f"""
model:
  name: "timejepa_{rec.name}"
  seq_length: {context_length}
  patch_length: 16
  stride: 8
  num_channels: 1
  prediction_length: 96
  
  encoder:
    d_model: {rec.d_model}
    n_heads: {rec.n_heads}
    n_layers: {rec.encoder_layers}
    d_ff: {rec.d_ff}
    dropout: 0.1
    
  predictor:
    d_model: {rec.d_model}
    n_heads: {rec.n_heads}
    n_layers: {rec.predictor_layers}
    d_ff: {rec.d_ff}
    
  decoder:
    type: "attentive"
    d_model: {rec.d_model}
    hidden_dim: {rec.d_model * 2}
    n_layers: {rec.decoder_layers}

# Estimated total parameters: ~{rec.total_params/1e6:.1f}M
# Optimal range for your data: {param_range['min']/1e6:.1f}M - {param_range['max']/1e6:.1f}M
""")
    
    
    print("=" * 80)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 80)
    
    # Batch size heuristic
    params_M = rec.total_params / 1e6
    if params_M < 5:
        batch_rec = "256-512"
    elif params_M < 20:
        batch_rec = "128-256"
    elif params_M < 50:
        batch_rec = "64-128"
    else:
        batch_rec = "32-64"
    
    warmup_steps = max(500, int(token_stats['tokens_per_epoch'] / 256 * 0.05))
    
    print(f"""
  Batch size (24GB GPU):    {batch_rec}
  Learning rate:            1e-4 to 3e-4
  Weight decay:             0.05
  Warmup steps:             ~{warmup_steps:,}
  Gradient accumulation:    Adjust to reach effective batch ~512-1024
  
  With {epochs} epochs on {total_points/1e6:.0f}M points:
      - Monitor val_loss for overfitting after epoch ~{max(5, epochs//3)}
      - Consider early stopping with patience=5-10
      - Use dropout=0.1-0.2 for regularization
""")


def main():
    parser = argparse.ArgumentParser(description="TimeJEPA Scaling Law Calculator")
    parser.add_argument("--total-points", type=int, required=True, help="Total datapoints")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--context-length", type=int, default=384, help="Context window length")
    
    args = parser.parse_args()
    print_analysis(args.total_points, args.epochs, args.context_length)


if __name__ == "__main__":
    main()