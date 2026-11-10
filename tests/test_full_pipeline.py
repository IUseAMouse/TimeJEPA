"""
Full pipeline test: loader -> transform -> forward -> backward.
Adapted for JEPATST.
"""
import tempfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.dataset import TimeSeriesDataset
from timejepa.data.datamodule import MonashDataModule
from timejepa.data.normalizer import get_normalizer


def test_normalizer_identity():
    """Checks that the identity normalizer does not modify the data."""
    print("\n" + "="*80)
    print("TEST 1: Identity Normalizer")
    print("="*80)
    
    data = np.random.randn(10, 100).astype(np.float32)
    
    # Test identity
    normalizer = get_normalizer("identity")
    normalizer.fit(data)
    transformed = normalizer.transform(data)
    
    assert np.allclose(data, transformed), "Identity should not change data!"
    print("Identity normalizer: data unchanged")
    
    normalizer_std = get_normalizer("standard")
    normalizer_std.fit(data)
    transformed_std = normalizer_std.transform(data)
    
    assert not np.allclose(data, transformed_std), "Standard should change data!"
    assert np.abs(transformed_std.mean()) < 1e-5, "Mean should be ~0"
    assert np.abs(transformed_std.std() - 1.0) < 0.1, "Std should be ~1"
    print("Standard normalizer: data normalized correctly")
    print(f"   Original: mean={data.mean():.3f}, std={data.std():.3f}")
    print(f"   Standard: mean={transformed_std.mean():.6f}, std={transformed_std.std():.3f}")


def test_dataset_with_normalizers():
    """Tests the dataset with different normalizers."""
    print("\n" + "="*80)
    print("TEST 2: Dataset with normalizers")
    print("="*80)
    
   
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        data = np.random.randn(5, 200).astype(np.float32)
        np.save(f.name, data)
        temp_path = Path(f.name)
    
    try:
        
        dataset_identity = TimeSeriesDataset(
            data_path=temp_path,
            context_length=50,
            prediction_length=10,
            stride=10,
            normalizer=None,  
            normalize_mode="global",
        )
        
        sample = dataset_identity[0]
        context = sample['context'].numpy()
        
       
        print(f"Identity dataset:")
        print(f"   Context shape: {context.shape}")
        print(f"   Context mean: {context.mean():.3f}, std: {context.std():.3f}")
        print(f"   Original mean: {data.mean():.3f}, std: {data.std():.3f}")
        print(f"   Normalizer: {dataset_identity.normalizer.__class__.__name__}")
        

        normalizer_std = get_normalizer("standard")
        dataset_standard = TimeSeriesDataset(
            data_path=temp_path,
            context_length=50,
            prediction_length=10,
            stride=10,
            normalizer=normalizer_std,
            normalize_mode="global",
        )
        
        sample_std = dataset_standard[0]
        context_std = sample_std['context'].numpy()
        
        print(f"\nStandard dataset:")
        print(f"   Context mean: {context_std.mean():.6f}, std: {context_std.std():.3f}")
        print(f"   Normalizer: {dataset_standard.normalizer.__class__.__name__}")
        
        assert not np.allclose(context, context_std), "Identity vs Standard should differ!"
        
    finally:
        temp_path.unlink()


def test_datamodule_normalizer_types():
    """Tests MonashDataModule with different normalizer_type values."""
    print("\n" + "="*80)
    print("TEST 3: MonashDataModule normalizer_type")
    print("="*80)
    
    # Create temporary data
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        data = np.random.randn(5, 200).astype(np.float32)
        np.save(f.name, data)
        temp_path = Path(f.name)
    
    try:
        # Test identity
        dm_identity = MonashDataModule(
            data_path=temp_path,
            context_length=50,
            prediction_length=10,
            batch_size=2,
            normalizer_type="identity",
            normalize_mode="global",
            num_workers=0,
        )
        dm_identity.prepare_data()
        dm_identity.setup()
        
        print(f"DataModule with identity:")
        print(f"   Normalizer: {dm_identity.normalizer.__class__.__name__}")
        assert dm_identity.normalizer.__class__.__name__ == "IdentityNormalizer"
        
        # Test standard
        dm_standard = MonashDataModule(
            data_path=temp_path,
            context_length=50,
            prediction_length=10,
            batch_size=2,
            normalizer_type="standard",
            normalize_mode="global",
            num_workers=0,
        )
        dm_standard.prepare_data()
        dm_standard.setup()
        
        print(f"\nDataModule with standard:")
        print(f"   Normalizer: {dm_standard.normalizer.__class__.__name__}")
        assert dm_standard.normalizer.__class__.__name__ == "StandardScaler"
        
        # Compare the data
        batch_identity = next(iter(dm_identity.train_dataloader()))
        batch_standard = next(iter(dm_standard.train_dataloader()))
        
        print(f"\nBatch comparison:")
        print(f"   Identity batch mean: {batch_identity['context'].mean():.3f}")
        print(f"   Standard batch mean: {batch_standard['context'].mean():.6f}")
        
    finally:
        temp_path.unlink()
