"""
Compute optimal model configuration based on dataset size and scaling laws.

Usage:
    python scripts/compute_model_config.py --data-dir data/processed
    python scripts/compute_model_config.py --total-points 25000000 --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.config import Config

logger = logging.getLogger(__name__)


def count_dataset_points(data_dir: Path, config: Config) -> Dict[str, int]:
    """
    Count total datapoints across all processed datasets.
    
    Returns:
        dict with 'total_series', 'total_points', and per-dataset stats
    """
    processed_dir = data_dir / "processed"
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    datasets = config.list_datasets()
    
    stats = {
        'total_series': 0,
        'total_points': 0,
        'datasets': {}
    }
    
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"{'Dataset':<25} | {'Series':>8} | {'Points':>15}")
    print("-" * 70)
    
    for dataset_name in datasets:
        data_path = processed_dir / f"{dataset_name}.npy"
        
        if not data_path.exists():
            print(f"{dataset_name:<25} | {'NOT FOUND':>8} | {'-':>15}")
            continue
        
        try:
            # Load data
            data = np.load(data_path, allow_pickle=True)
            
            # Handle different formats
            if isinstance(data, np.ndarray):
                if data.dtype == object:
                    # Variable length series
                    num_series = len(data)
                    num_points = sum(len(x) for x in data)
                else:
                    # Fixed length (N, T) or (N, T, C)
                    num_series = data.shape[0]
                    num_points = data.shape[0] * data.shape[1]
            elif isinstance(data, np.lib.npyio.NpzFile):
                # Compressed format
                arr = data['data']
                if arr.dtype == object:
                    num_series = len(arr)
                    num_points = sum(len(x) for x in arr)
                else:
                    num_series = arr.shape[0]
                    num_points = arr.shape[0] * arr.shape[1]
            else:
                # Dict format with stats
                num_series = data.item()['stats']['num_series']
                num_points = data.item()['stats']['total_points']
            
            stats['total_series'] += num_series
            stats['total_points'] += num_points
            stats['datasets'][dataset_name] = {
                'series': num_series,
                'points': num_points
            }
            
            print(f"{dataset_name:<25} | {num_series:>8,} | {num_points:>15,}")
            
        except Exception as e:
            logger.warning(f"Error processing {dataset_name}: {e}")
            print(f"{dataset_name:<25} | {'ERROR':>8} | {'-':>15}")
    
    print("=" * 70)
    print(f"{'TOTAL':<25} | {stats['total_series']:>8,} | {stats['total_points']:>15,}")
    print("=" * 70)
    
    return stats


def compute_optimal_config(
    total_points: int,
    epochs: int = 50,
    avg_context_length: int = 256,
    target_params: Optional[int] = None
) -> None:
    """
    Compute optimal model dimensions based on scaling laws.
    
    Args:
        total_points: Total number of datapoints (D)
        epochs: Number of training epochs
        avg_context_length: Average context window length
        target_params: Optional target parameter count (overrides scaling law)
    """
    print("\n" + "=" * 70)
    print("SCALING LAW ANALYSIS")
    print("=" * 70)
    
    # Total training tokens
    total_tokens = total_points
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total datapoints (D):     {total_points:>15,}")
    print(f"  Training epochs:          {epochs:>15}")
    print(f"  Total training tokens:    {total_tokens:>15,}")
    print(f"  Avg context length:       {avg_context_length:>15}")
    
    # Compute optimal params using Chinchilla-like scaling
    # Rule: 1 param ≈ 6-10 tokens for optimal training
    # We use 8 as middle ground
    if target_params is None:
        optimal_params = total_tokens / 8
    else:
        optimal_params = target_params
        print(f"\n⚠️  Using target param count: {target_params:,}")
    
    print(f"\n🎯 Target Parameters:")
    print(f"  Optimal parameter count:  {optimal_params:>15,.0f}")
    
    # Compute configurations
    # For a Transformer: params ≈ 12 * L * d^2
    # (assumes d_ff = 4 * d_model, which contributes most params)
    
    print(f"\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Layers':>7} | {'d_model':>8} | {'n_heads':>8} | {'d_ff':>8} | {'Params':>12}")
    print("-" * 70)
    
    configs = []
    
    for num_layers in [4, 6, 8, 10, 12]:
        # Solve for d_model: params = 12 * L * d^2
        d_model = int(np.sqrt(optimal_params / (12 * num_layers)))
        
        # Round to multiple of 64 for efficiency (GPU/TPU friendly)
        d_model = ((d_model + 31) // 64) * 64
        
        # Ensure minimum size
        d_model = max(d_model, 128)
        
        # Compute num_heads (d_model should be divisible by num_heads)
        if d_model >= 512:
            num_heads = 8
        elif d_model >= 384:
            num_heads = 6
        elif d_model >= 256:
            num_heads = 4
        else:
            num_heads = 4
        
        # Adjust d_model to be divisible by num_heads
        d_model = (d_model // num_heads) * num_heads
        
        d_ff = 4 * d_model
        
        # Recompute actual params
        # Simplified: embedding + layers + head
        # Each layer: self-attn (4 * d^2) + FFN (2 * d * d_ff) ≈ 12 * d^2
        actual_params = 12 * num_layers * (d_model ** 2)
        
        configs.append({
            'num_layers': num_layers,
            'd_model': d_model,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'params': actual_params
        })
        
        print(f"{num_layers:>7} | {d_model:>8} | {num_heads:>8} | {d_ff:>8} | {actual_params:>12,}")
    
    # Find closest to optimal
    closest_idx = min(range(len(configs)), 
                     key=lambda i: abs(configs[i]['params'] - optimal_params))
    
    best = configs[closest_idx]
    
    print("=" * 70)
    print(f"\n✅ RECOMMENDED CONFIGURATION:")
    print(f"""
  num_layers: {best['num_layers']}
  d_model: {best['d_model']}
  num_heads: {best['num_heads']}
  d_ff: {best['d_ff']}
  
  Total parameters: ~{best['params']:,}
    """)
    
    # Additional recommendations
    print("=" * 70)
    print("ADDITIONAL RECOMMENDATIONS")
    print("=" * 70)
    
    # Context/prediction lengths based on data
    recommended_context = min(512, max(96, int(avg_context_length * 1.5)))
    recommended_context = (recommended_context // 16) * 16  # Round to patch_size
    
    recommended_pred = min(96, recommended_context // 4)
    recommended_pred = (recommended_pred // 16) * 16
    
    print(f"""
Training Configuration:
  context_length: {recommended_context}
  prediction_length: {recommended_pred}
  patch_size: 16
  
  Training strategy:
    - Pretrain with variable lengths: context=[96, {recommended_context}], pred=[24, {recommended_pred}]
    - Finetune with fixed: context={recommended_context}, pred={recommended_pred}
  
  Optimizer:
    - lr: 1e-4 (or use lr_finder)
    - weight_decay: 0.01
    - warmup_steps: {max(1000, total_points // (32 * 100))}  # ~1% of total steps
    
  Batch size (estimate for your GPU):
    - For {best['params']:,} params:
      * 12GB GPU: batch_size ~ 32-64
      * 24GB GPU: batch_size ~ 64-128
      * 48GB GPU: batch_size ~ 128-256
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Compute optimal model configuration based on dataset size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (will look in data/processed)"
    )
    
    parser.add_argument(
        "--total-points",
        type=int,
        help="Manually specify total datapoints (skip counting)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--avg-context",
        type=int,
        default=256,
        help="Average context length in your data"
    )
    
    parser.add_argument(
        "--target-params",
        type=int,
        help="Target parameter count (overrides scaling law calculation)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (default: configs/datasets.yaml)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Load config
    try:
        config_path = Path(args.config) if args.config else None
        config = Config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Count or use provided points
    if args.total_points:
        total_points = args.total_points
        print(f"\n📊 Using manually specified datapoints: {total_points:,}")
    else:
        try:
            stats = count_dataset_points(Path(args.data_dir), config)
            total_points = stats['total_points']
            
            if total_points == 0:
                logger.error("No datasets found or all have 0 points")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to count datapoints: {e}")
            sys.exit(1)
    
    # Compute optimal configuration
    compute_optimal_config(
        total_points=total_points,
        epochs=args.epochs,
        avg_context_length=args.avg_context,
        target_params=args.target_params
    )
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()