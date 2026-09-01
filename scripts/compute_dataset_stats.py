import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timejepa.config import Config

def main():
    config = Config()
    data_dir = Path("data/processed")
    
    datasets = config.list_datasets()
    
    total_series = 0
    total_points = 0
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for dataset_name in datasets:
        data_path = data_dir / f"{dataset_name}.npy"
        
        if not data_path.exists():
            print(f"{dataset_name}: NOT FOUND")
            continue
        
        # Load
        data = np.load(data_path, allow_pickle=True)
        
        if isinstance(data, np.ndarray):
            if data.dtype == object:
                # Variable length
                series_count = len(data)
                points = sum(len(x) for x in data)
            else:
                # Fixed length
                series_count, length = data.shape[:2]
                points = series_count * length
        else:
            # Dict with stats
            series_count = data['stats']['num_series']
            points = data['stats']['total_points']
        
        total_series += series_count
        total_points += points
        
        print(f"{dataset_name:25} | {series_count:6} series | {points:12,} points")
    
    print("="*60)
    print(f"{'TOTAL':25} | {total_series:6} series | {total_points:12,} points")
    print("="*60)
    print(f"\nD = {total_points:,} total datapoints\n")

if __name__ == "__main__":
    main()