# src/timejepa/data/datamodule.py
"""
PyTorch Lightning DataModule for time series datasets.
"""
import logging
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .dataset import TimeSeriesDataset, MultiHorizonDataset
from .normalization import Normalizer, get_normalizer

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
            # Create full dataset
            full_dataset = TimeSeriesDataset(
                data_path=self.data_path,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                stride=self.stride,
                normalizer=None,  # Will be fitted
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
            # For time series, you might want temporal splits instead
            import torch
            generator = torch.Generator().manual_seed(self.seed)
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_len, val_len, test_len],
                generator=generator
            )
            
            logger.info("✓ Setup complete for training")
        
        if stage == "test" and self.test_dataset is None:
            # If only testing, load test set
            logger.info("Setting up test dataset")
            # This is a simplified version - in practice, load saved normalizer
            raise NotImplementedError("Test-only mode requires saved normalizer")
    
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