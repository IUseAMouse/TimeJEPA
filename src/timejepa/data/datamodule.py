# src/timejepa/data/datamodule.py
"""
PyTorch Lightning DataModule for time series datasets.
"""
import logging
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import TimeSeriesDataset, MultiHorizonDataset
from .normalizer import Normalizer, get_normalizer

logger = logging.getLogger(__name__)


class MonashDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Monash time series datasets.
    
    Handles train/val/test splits, normalization, and DataLoader configuration.
    """
    
    def __init__(
        self,
        data_path: Path,
        context_length: int,
        prediction_length: int,
        batch_size: int = 32,
        stride: int = 1,
        normalize_mode: Literal["per_series", "global"] = "per_series",
        normalizer_type: str = "identity",
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_series: Optional[int] = None,
        min_series_length: Optional[int] = None,
        seed: int = 42
    ):
        """
        Args:
            data_path: Path to .npy file
            context_length: Context window length
            prediction_length: Prediction window length
            batch_size: Batch size for DataLoaders
            stride: Stride for sliding windows
            normalize_mode: 'per_series' or 'global'
            normalizer_type: 'standard', 'minmax', or 'identity'
            train_val_test_split: Tuple of (train, val, test) fractions
            num_workers: Number of workers for DataLoader
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            max_series: Limit number of series
            min_series_length: Filter short series
            seed: Random seed for splits
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_path = Path(data_path)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.stride = stride
        self.normalize_mode = normalize_mode
        self.normalizer_type = normalizer_type
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.max_series = max_series
        self.min_series_length = min_series_length
        self.seed = seed
        
        self.normalizer: Optional[Normalizer] = None
        self.train_dataset: Optional[TimeSeriesDataset] = None
        self.val_dataset: Optional[TimeSeriesDataset] = None
        self.test_dataset: Optional[TimeSeriesDataset] = None
    
    def prepare_data(self):
        """
        Download or prepare data (called only on main process).
        Here we just check if data exists.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please run: make download-data"
            )
        
        logger.info(f"Data file found: {self.data_path}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets (called on every process).
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            # 🔥 FIX: Créer le normalizer AVANT le dataset
            normalizer = get_normalizer(self.normalizer_type)
            
            # Create full dataset
            full_dataset = TimeSeriesDataset(
                data_path=self.data_path,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                stride=self.stride,
                normalizer=normalizer,  # 🔥 FIX: Passer le normalizer créé
                normalize_mode=self.normalize_mode,
                return_tensor=True,
                max_series=self.max_series,
                min_series_length=self.min_series_length
            )
            
            # Store normalizer
            self.normalizer = full_dataset.get_normalizer()
            
            # Split into train/val/test
            total_len = len(full_dataset)
            train_frac, val_frac, test_frac = self.train_val_test_split
            
            train_len = int(total_len * train_frac)
            val_len = int(total_len * val_frac)
            test_len = total_len - train_len - val_len
            
            logger.info(f"Splitting {total_len} windows: "
                    f"train={train_len}, val={val_len}, test={test_len}")
            
            # Use random_split for simplicity
            generator = torch.Generator().manual_seed(self.seed)
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_len, val_len, test_len],
                generator=generator
            )
            
            logger.info(f"✓ Setup complete with {self.normalizer.__class__.__name__}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True  # Drop last incomplete batch
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False  # Don't persist for test
        )
    
    def get_normalizer(self) -> Normalizer:
        """Get fitted normalizer for evaluation."""
        if self.normalizer is None:
            raise RuntimeError("DataModule must be setup before getting normalizer")
        return self.normalizer
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up resources."""
        pass


class MultiHorizonDataModule(MonashDataModule):
    """
    DataModule for multi-horizon forecasting.
    
    Predicts at multiple time scales simultaneously.
    """
    
    def __init__(
        self,
        data_path: Path,
        context_length: int,
        prediction_lengths: List[int],  # Multiple horizons
        batch_size: int = 32,
        stride: int = 1,
        normalize_mode: Literal["per_series", "global"] = "per_series",
        normalizer_type: str = "standard",
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_series: Optional[int] = None,
        seed: int = 42
    ):
        self.prediction_lengths = prediction_lengths
        
        # Use max horizon for base class
        super().__init__(
            data_path=data_path,
            context_length=context_length,
            prediction_length=max(prediction_lengths),
            batch_size=batch_size,
            stride=stride,
            normalize_mode=normalize_mode,
            normalizer_type=normalizer_type,
            train_val_test_split=train_val_test_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            max_series=max_series,
            seed=seed
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup with MultiHorizonDataset."""
        if stage == "fit" or stage is None:
            full_dataset = MultiHorizonDataset(
                data_path=self.data_path,
                context_length=self.context_length,
                prediction_lengths=self.prediction_lengths,
                stride=self.stride,
                normalizer=None,
                normalize_mode=self.normalize_mode,
                return_tensor=True,
                max_series=self.max_series
            )
            
            self.normalizer = full_dataset.get_normalizer()
            
            # Split
            total_len = len(full_dataset)
            train_frac, val_frac, test_frac = self.train_val_test_split
            
            train_len = int(total_len * train_frac)
            val_len = int(total_len * val_frac)
            test_len = total_len - train_len - val_len
            
            import torch
            generator = torch.Generator().manual_seed(self.seed)
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_len, val_len, test_len],
                generator=generator
            )
            
            logger.info(f"✓ Setup complete for multi-horizon training "
                       f"with horizons: {self.prediction_lengths}")
            

class MultiDatasetMonashDataModule(pl.LightningDataModule):
    """
    Extension de MonashDataModule pour charger automatiquement 
    plusieurs datasets du dossier data/processed/.
    
    Compatible avec la structure existante de MonashDataModule.
    """
    
    def __init__(
        self,
        data_dir: Path,
        context_length: int,
        prediction_length: int,
        datasets: Optional[List[str]] = None,  # None = auto-load all
        dataset_pattern: str = "*.npy",
        combine_mode: Literal["concatenate", "separate"] = "concatenate",
        batch_size: int = 32,
        stride: int = 1,
        normalize_mode: Literal["per_series", "global"] = "per_series",
        normalizer_type: str = "identity",
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_series: Optional[int] = None,
        min_series_length: Optional[int] = None,
        dataset_overrides: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing .npy files (e.g., "data/processed")
            context_length: Context window length
            prediction_length: Prediction window length
            datasets: List of dataset names to load (None = all)
            dataset_pattern: Glob pattern for dataset files
            combine_mode: How to combine datasets ('concatenate' or 'separate')
            ... (autres params identiques à MonashDataModule)
            dataset_overrides: Per-dataset config overrides
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = Path(data_dir)
        self.datasets = datasets
        self.dataset_pattern = dataset_pattern
        self.combine_mode = combine_mode
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.stride = stride
        self.normalize_mode = normalize_mode
        self.normalizer_type = normalizer_type
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.max_series = max_series
        self.min_series_length = min_series_length
        self.dataset_overrides = dataset_overrides or {}
        self.seed = seed
        
        # Store individual datamodules
        self.datamodules: Dict[str, MonashDataModule] = {}
        self.dataset_files: Dict[str, Path] = {}
        
        # Combined datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalizer = None
    
    def prepare_data(self):
        """Discover all .npy files in data_dir."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all .npy files
        all_files = list(self.data_dir.glob(self.dataset_pattern))
        
        if not all_files:
            raise FileNotFoundError(
                f"No datasets found in {self.data_dir} matching {self.dataset_pattern}"
            )
        
        # Filter by requested datasets
        if self.datasets is None:
            # Use all found datasets
            self.dataset_files = {f.stem: f for f in all_files}
        elif isinstance(self.datasets, str):
            # Single dataset
            target_file = self.data_dir / f"{self.datasets}.npy"
            if not target_file.exists():
                raise FileNotFoundError(f"Dataset not found: {target_file}")
            self.dataset_files = {self.datasets: target_file}
        else:
            # List of datasets
            for name in self.datasets:
                target_file = self.data_dir / f"{name}.npy"
                if not target_file.exists():
                    logger.warning(f"Dataset not found: {target_file}, skipping...")
                    continue
                self.dataset_files[name] = target_file
        
        if not self.dataset_files:
            raise ValueError("No valid datasets found after filtering")
        
        logger.info(f"Found {len(self.dataset_files)} dataset(s): {list(self.dataset_files.keys())}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup individual datamodules and combine them."""
        from torch.utils.data import ConcatDataset
        
        if stage == "fit" or stage is None:
            train_datasets = []
            val_datasets = []
            test_datasets = []
            
            # Create a datamodule for each dataset
            for dataset_name, file_path in self.dataset_files.items():
                logger.info(f"Setting up datamodule for: {dataset_name}")
                
                # Get per-dataset overrides
                overrides = self.dataset_overrides.get(dataset_name, {})
                
                # Create individual datamodule (réutilise ton MonashDataModule)
                dm = MonashDataModule(
                    data_path=file_path,
                    context_length=self.context_length,
                    prediction_length=self.prediction_length,
                    batch_size=overrides.get('batch_size', self.batch_size),
                    stride=self.stride,
                    normalize_mode=self.normalize_mode,
                    normalizer_type=self.normalizer_type,
                    train_val_test_split=self.train_val_test_split,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    max_series=overrides.get('max_series', self.max_series),
                    min_series_length=self.min_series_length,
                    seed=self.seed
                )
                
                # Setup this datamodule
                dm.prepare_data()
                dm.setup(stage)
                
                # Store it
                self.datamodules[dataset_name] = dm
                
                # Collect datasets
                train_datasets.append(dm.train_dataset)
                val_datasets.append(dm.val_dataset)
                test_datasets.append(dm.test_dataset)
                
                logger.info(f"  ✓ {dataset_name}: "
                          f"train={len(dm.train_dataset)}, "
                          f"val={len(dm.val_dataset)}, "
                          f"test={len(dm.test_dataset)}")
            
            # Combine datasets
            if self.combine_mode == "concatenate":
                self.train_dataset = ConcatDataset(train_datasets)
                self.val_dataset = ConcatDataset(val_datasets)
                self.test_dataset = ConcatDataset(test_datasets)
                
                logger.info(f"Combined datasets: "
                          f"train={len(self.train_dataset)}, "
                          f"val={len(self.val_dataset)}, "
                          f"test={len(self.test_dataset)}")
            else:
                # Use first dataset (can be extended for other strategies)
                first_dm = list(self.datamodules.values())[0]
                self.train_dataset = first_dm.train_dataset
                self.val_dataset = first_dm.val_dataset
                self.test_dataset = first_dm.test_dataset
                logger.info(f"Using single dataset: {list(self.datamodules.keys())[0]}")
            
            # Use normalizer from first dataset
            self.normalizer = list(self.datamodules.values())[0].normalizer
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False
        )
    
    def get_normalizer(self) -> Normalizer:
        """Get fitted normalizer."""
        if self.normalizer is None:
            raise RuntimeError("DataModule must be setup before getting normalizer")
        return self.normalizer