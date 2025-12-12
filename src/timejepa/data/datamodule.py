# src/timejepa/data/datamodule.py
"""
PyTorch Lightning DataModule for time series datasets.
"""
import logging
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .dataset import TimeSeriesDataset, MultiHorizonDataset, AugmentedSubset
from .normalizer import Normalizer, get_normalizer

logger = logging.getLogger(__name__)


# ❌ TrainingSubset supprimé — remplacé par AugmentedSubset


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
        clip_outliers: bool = True,
        clip_sigma: float = 5.0,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_series: Optional[int] = None,
        min_series_length: Optional[int] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
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
        self.clip_outliers = clip_outliers
        self.clip_sigma = clip_sigma
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.max_series = max_series
        self.min_series_length = min_series_length
        self.augmentation_config = augmentation_config
        self.seed = seed
        
        self.normalizer: Optional[Normalizer] = None
        self.train_dataset: Optional[AugmentedSubset] = None
        self.val_dataset: Optional[AugmentedSubset] = None
        self.test_dataset: Optional[AugmentedSubset] = None
        
        # Garder une référence au dataset complet
        self._full_dataset: Optional[TimeSeriesDataset] = None
    
    def prepare_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please run: make download-data"
            )
        logger.info(f"Data file found: {self.data_path}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            normalizer = get_normalizer(
                self.normalizer_type, 
                clip_outliers=self.clip_outliers,
                clip_sigma=self.clip_sigma
            )
            
            # Créer le dataset complet
            self._full_dataset = TimeSeriesDataset(
                data_path=self.data_path,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                stride=self.stride,
                normalizer=normalizer, 
                normalize_mode=self.normalize_mode,
                return_tensor=True,
                max_series=self.max_series,
                min_series_length=self.min_series_length,
                augmentations=self.augmentation_config,
            )
            
            self.normalizer = self._full_dataset.get_normalizer()
            
            # Calculer les tailles
            total_len = len(self._full_dataset)
            train_frac, val_frac, test_frac = self.train_val_test_split
            
            train_len = int(total_len * train_frac)
            val_len = int(total_len * val_frac)
            test_len = total_len - train_len - val_len
            
            logger.info(f"Splitting {total_len} windows: "
                       f"train={train_len}, val={val_len}, test={test_len}")
            
            # Générer les indices mélangés
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(total_len, generator=generator).tolist()
            
            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len + val_len]
            test_indices = indices[train_len + val_len:]
            
            # ✅ Créer les AugmentedSubset (et ne PAS les écraser !)
            self.train_dataset = AugmentedSubset(
                self._full_dataset, 
                train_indices, 
                apply_augmentation=True
            )
            self.val_dataset = AugmentedSubset(
                self._full_dataset, 
                val_indices, 
                apply_augmentation=False
            )
            self.test_dataset = AugmentedSubset(
                self._full_dataset, 
                test_indices, 
                apply_augmentation=False
            )
            
            logger.info(f"✓ Setup complete with {self.normalizer.__class__.__name__}")
            if self.augmentation_config:
                logger.info(f"✓ Augmentations enabled for training only")
    
    def train_dataloader(self) -> DataLoader:
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False
        )
    
    def get_normalizer(self) -> Normalizer:
        if self.normalizer is None:
            raise RuntimeError("DataModule must be setup before getting normalizer")
        return self.normalizer
    
    def teardown(self, stage: Optional[str] = None):
        pass


class MultiHorizonDataModule(MonashDataModule):
    """DataModule for multi-horizon forecasting."""
    
    def __init__(
        self,
        data_path: Path,
        context_length: int,
        prediction_lengths: List[int],
        batch_size: int = 32,
        stride: int = 1,
        normalize_mode: Literal["per_series", "global"] = "per_series",
        normalizer_type: str = "standard",
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_series: Optional[int] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.prediction_lengths = prediction_lengths
        
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
            augmentation_config=augmentation_config,
            seed=seed
        )
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self._full_dataset = MultiHorizonDataset(
                data_path=self.data_path,
                context_length=self.context_length,
                prediction_lengths=self.prediction_lengths,
                stride=self.stride,
                normalizer=None,
                normalize_mode=self.normalize_mode,
                return_tensor=True,
                max_series=self.max_series
            )
            
            self.normalizer = self._full_dataset.get_normalizer()
            
            total_len = len(self._full_dataset)
            train_frac, val_frac, test_frac = self.train_val_test_split
            
            train_len = int(total_len * train_frac)
            val_len = int(total_len * val_frac)
            test_len = total_len - train_len - val_len
            
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(total_len, generator=generator).tolist()
            
            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len + val_len]
            test_indices = indices[train_len + val_len:]
            
            # Note: MultiHorizonDataset n'a pas d'augmentations pour l'instant
            # On peut utiliser AugmentedSubset mais les augmentations ne seront pas appliquées
            self.train_dataset = AugmentedSubset(
                self._full_dataset, train_indices, apply_augmentation=True
            )
            self.val_dataset = AugmentedSubset(
                self._full_dataset, val_indices, apply_augmentation=False
            )
            self.test_dataset = AugmentedSubset(
                self._full_dataset, test_indices, apply_augmentation=False
            )
            
            logger.info(f"✓ Setup complete for multi-horizon training "
                       f"with horizons: {self.prediction_lengths}")


class MultiDatasetMonashDataModule(pl.LightningDataModule):
    """
    Extension de MonashDataModule pour charger automatiquement 
    plusieurs datasets du dossier data/processed/.
    """
    
    def __init__(
        self,
        data_dir: Path,
        context_length: int,
        prediction_length: int,
        datasets: Optional[List[str]] = None,
        dataset_pattern: str = "*.npy",
        combine_mode: Literal["concatenate", "separate"] = "concatenate",
        batch_size: int = 64,
        stride: int = 1,
        normalize_mode: Literal["per_series", "global"] = "per_series",
        normalizer_type: str = "identity",
        clip_outliers: bool = True,
        clip_sigma: float = 5.0,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_series: Optional[int] = None,
        min_series_length: Optional[int] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,  # 👈 AJOUTÉ
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
        self.clip_outliers = clip_outliers
        self.clip_sigma = clip_sigma
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.max_series = max_series
        self.min_series_length = min_series_length
        self.augmentation_config = augmentation_config  # 👈 AJOUTÉ
        self.dataset_overrides = dataset_overrides or {}
        self.seed = seed
        
        self.datamodules: Dict[str, MonashDataModule] = {}
        self.dataset_files: Dict[str, Path] = {}
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalizer = None
    
    def prepare_data(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        all_files = list(self.data_dir.glob(self.dataset_pattern))
        
        if not all_files:
            raise FileNotFoundError(
                f"No datasets found in {self.data_dir} matching {self.dataset_pattern}"
            )
        
        if self.datasets is None or self.datasets == []:
            logger.warning("Use all found datasets")
            self.dataset_files = {f.stem: f for f in all_files}
        elif isinstance(self.datasets, str):
            logger.warning(f"Loading datasets : {self.datasets}")
            target_file = self.data_dir / f"{self.datasets}.npy"
            if not target_file.exists():
                raise FileNotFoundError(f"Dataset not found: {target_file}")
            self.dataset_files = {self.datasets: target_file}
        else:
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
        if stage == "fit" or stage is None:
            train_datasets = []
            val_datasets = []
            test_datasets = []
            
            for dataset_name, file_path in self.dataset_files.items():
                logger.info(f"Setting up datamodule for: {dataset_name}")
                
                overrides = self.dataset_overrides.get(dataset_name, {})
                
                # 👇 Passer augmentation_config aux sous-modules
                dm = MonashDataModule(
                    data_path=file_path,
                    context_length=self.context_length,
                    prediction_length=self.prediction_length,
                    batch_size=overrides.get('batch_size', self.batch_size),
                    stride=self.stride,
                    normalize_mode=self.normalize_mode,
                    normalizer_type=self.normalizer_type,
                    clip_outliers=self.clip_outliers,
                    clip_sigma=self.clip_sigma,
                    train_val_test_split=self.train_val_test_split,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    max_series=overrides.get('max_series', self.max_series),
                    min_series_length=self.min_series_length,
                    augmentation_config=overrides.get('augmentation_config', self.augmentation_config),  # 👈
                    seed=self.seed
                )
                
                dm.prepare_data()
                dm.setup(stage)
                
                self.datamodules[dataset_name] = dm
                
                train_datasets.append(dm.train_dataset)
                val_datasets.append(dm.val_dataset)
                test_datasets.append(dm.test_dataset)
                
                logger.info(f"  ✓ {dataset_name}: "
                          f"train={len(dm.train_dataset)}, "
                          f"val={len(dm.val_dataset)}, "
                          f"test={len(dm.test_dataset)}")
            
            if self.combine_mode == "concatenate":
                self.train_dataset = ConcatDataset(train_datasets)
                self.val_dataset = ConcatDataset(val_datasets)
                self.test_dataset = ConcatDataset(test_datasets)
                
                logger.info(f"Combined datasets: "
                          f"train={len(self.train_dataset)}, "
                          f"val={len(self.val_dataset)}, "
                          f"test={len(self.test_dataset)}")
            else:
                first_dm = list(self.datamodules.values())[0]
                self.train_dataset = first_dm.train_dataset
                self.val_dataset = first_dm.val_dataset
                self.test_dataset = first_dm.test_dataset
                logger.info(f"Using single dataset: {list(self.datamodules.keys())[0]}")
            
            self.normalizer = list(self.datamodules.values())[0].normalizer
            
            if self.augmentation_config:
                logger.info(f"✓ Augmentations enabled for training across all datasets")
    
    def train_dataloader(self) -> DataLoader:
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False
        )
    
    def get_normalizer(self) -> Normalizer:
        if self.normalizer is None:
            raise RuntimeError("DataModule must be setup before getting normalizer")
        return self.normalizer