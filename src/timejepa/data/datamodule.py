"""
PyTorch Lightning DataModule for time series datasets.
"""
import logging
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, Sampler

from .dataset import TimeSeriesDataset, MultiHorizonDataset, AugmentedSubset
from .normalizer import Normalizer, get_normalizer

logger = logging.getLogger(__name__)


class TemperatureSampler(Sampler):
    """
    Temperature-based sampling for imbalanced multi-dataset training.
    
    Memory-efficient implementation: generates indices on-the-fly instead of
    materializing 500M+ indices in memory.
    
    With temperature T:
        p_i ∝ n_i^T
    
    - T=1.0: proportional to size (large datasets dominate)
    - T=0.5: square-root sampling (balanced compromise)  
    - T→0: uniform across datasets
    """
    
    def __init__(
        self,
        dataset_sizes: List[int],
        batch_size: int,
        temperature: float = 0.5,
        max_oversample_ratio: float = 5.0,
        num_batches_per_epoch: Optional[int] = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """
        Args:
            dataset_sizes: Size of each dataset in the ConcatDataset
            batch_size: Total batch size
            temperature: Sampling temperature
                - T=1.0: proportional (large datasets dominate)
                - T=0.5: sqrt sampling (balanced)
                - T=0.25: more uniform
            max_oversample_ratio: Max times a small dataset can be repeated per epoch
            num_batches_per_epoch: Override number of batches (None = auto)
            drop_last: Drop incomplete batches
            shuffle: Shuffle within datasets
            seed: Random seed
        """
        super().__init__(None)
        
        self.dataset_sizes = dataset_sizes
        self.num_datasets = len(dataset_sizes)
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_oversample_ratio = max_oversample_ratio
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
    
        if rank is None or world_size is None:
            if torch.distributed.is_initialized():
                self.rank = torch.distributed.get_rank()
                self.world_size = torch.distributed.get_world_size()
            else:
                self.rank = 0
                self.world_size = 1
        else:
            self.rank = rank
            self.world_size = world_size
        
        # Compute temperature-smoothed sampling probabilities
        # FIXED: sizes^T (not sizes^(1/T))
        # T=0.5 → sqrt, T→0 → uniform, T=1 → proportional
        sizes = torch.tensor(dataset_sizes, dtype=torch.float32)
        
        if temperature == 0:
            # Uniform across datasets
            self.sampling_probs = torch.ones(self.num_datasets) / self.num_datasets
        else:
            smoothed = sizes ** temperature
            self.sampling_probs = smoothed / smoothed.sum()
        
        # Calculate samples per dataset per batch
        self.samples_per_dataset = (self.sampling_probs * batch_size).floor().int()
        self.samples_per_dataset = torch.clamp(self.samples_per_dataset, min=1)
        
        # Distribute remainder to highest-probability datasets
        remainder = batch_size - self.samples_per_dataset.sum().item()
        if remainder > 0:
            top_k_count = min(remainder, self.num_datasets)
            top_k = torch.topk(self.sampling_probs, top_k_count).indices
            for idx in top_k[:remainder]:
                self.samples_per_dataset[idx] += 1
        elif remainder < 0:
            # Remove from lowest-probability datasets
            sorted_indices = torch.argsort(self.sampling_probs)
            idx = 0
            while self.samples_per_dataset.sum() > batch_size and idx < self.num_datasets:
                min_idx = sorted_indices[idx].item()
                if self.samples_per_dataset[min_idx] > 1:
                    self.samples_per_dataset[min_idx] -= 1
                else:
                    idx += 1
        
        self.samples_per_dataset = self.samples_per_dataset.tolist()
        self.actual_batch_size = sum(self.samples_per_dataset)
        
        # Store offsets only - NOT all indices (memory efficient!)
        self.dataset_offsets = [0]
        for size in dataset_sizes[:-1]:
            self.dataset_offsets.append(self.dataset_offsets[-1] + size)
        
        self._compute_epoch_plan()
        
        if num_batches_per_epoch is not None:
            self._num_batches = num_batches_per_epoch
        
        if self.rank == 0:
            self._log_sampling_info()
    
    def _compute_epoch_plan(self):
        """Compute number of batches and effective sampling ratios."""
        max_size = max(self.dataset_sizes)
        largest_idx = self.dataset_sizes.index(max_size)
        samples_for_largest = self.samples_per_dataset[largest_idx]
        
        if samples_for_largest > 0:
            batches_for_largest = max_size // samples_for_largest
        else:
            batches_for_largest = max_size
        
        # DDP: divide batches among workers
        total_batches = max(1, batches_for_largest)
        self._num_batches = total_batches // self.world_size
        
        self.effective_samples = []
        self.oversample_ratios = []
        
        for i, size in enumerate(self.dataset_sizes):
            effective = self._num_batches * self.samples_per_dataset[i] * self.world_size
            ratio = effective / size if size > 0 else 1.0
            
            if ratio > self.max_oversample_ratio:
                capped_effective = int(size * self.max_oversample_ratio)
                self.effective_samples.append(capped_effective)
            else:
                self.effective_samples.append(effective)
            
            self.oversample_ratios.append(min(ratio, self.max_oversample_ratio))
    
    def _log_sampling_info(self):
        logger.info(f"TemperatureSampler initialized (T={self.temperature}, "
                   f"max_oversample={self.max_oversample_ratio}x):")
        logger.info(f"  Batch size: {self.actual_batch_size} across {self.num_datasets} datasets")
        logger.info(f"  Batches/epoch/GPU: {self._num_batches} (world_size={self.world_size})")
        
        total_original = sum(self.dataset_sizes)
        
        for i, size in enumerate(self.dataset_sizes):
            samples_batch = self.samples_per_dataset[i]
            prob = self.sampling_probs[i].item()
            ratio = self.oversample_ratios[i]
            original_frac = size / total_original
            
            if ratio > 1.5:
                status = f"⬆ {ratio:.1f}x oversample"
            elif ratio < 0.8:
                status = f"⬇ {ratio:.0%} coverage"
            else:
                status = "✓ balanced"
            
            logger.info(
                f"  [{i}] {size:>12,} samples | "
                f"{samples_batch:>3}/batch ({prob:>5.1%}) | "
                f"was {original_frac:>5.1%} | {status}"
            )
    
    def __iter__(self):
        """
        Memory-efficient iterator using on-the-fly random sampling.
        
        Instead of materializing 500M indices (~14GB+), we generate
        random indices per batch using numpy RNG (~few KB).
        """
        # Deterministic seed: reproducible across restarts, varies by epoch/rank
        rng = np.random.default_rng(self.seed + self.epoch * 1000 + self.rank)
        
        # Track samples drawn per dataset for max_oversample enforcement
        samples_drawn = [0] * self.num_datasets
        max_samples = [int(size * self.max_oversample_ratio) for size in self.dataset_sizes]
        
        for batch_idx in range(self._num_batches):
            batch = []
            
            for i in range(self.num_datasets):
                n_samples = self.samples_per_dataset[i]
                dataset_size = self.dataset_sizes[i]
                offset = self.dataset_offsets[i]
                
                # Enforce max_oversample_ratio
                remaining = max_samples[i] - samples_drawn[i]
                if remaining <= 0:
                    continue
                
                actual_samples = min(n_samples, remaining)
                
                # Generate random indices on-the-fly (NOT stored!)
                local_indices = rng.integers(0, dataset_size, size=actual_samples)
                batch.extend((offset + local_indices).tolist())
                samples_drawn[i] += actual_samples
            
            if len(batch) > 0:
                if self.shuffle:
                    rng.shuffle(batch)
                yield batch
        
        self.epoch += 1
    
    def __len__(self) -> int:
        return self._num_batches
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch


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
        multi_resolution_factors: Optional[List[int]] = None,
        p_multi_resolution: float = 0.0,
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
        self.multi_resolution_factors = multi_resolution_factors
        self.p_multi_resolution = p_multi_resolution
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
                multi_resolution_factors=self.multi_resolution_factors,
                p_multi_resolution=self.p_multi_resolution,
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
            
            self.train_dataset = AugmentedSubset(
                self._full_dataset, 
                range(0, train_len),
                apply_augmentation=True
            )
            self.val_dataset = AugmentedSubset(
                self._full_dataset, 
                range(train_len, train_len + val_len),
                apply_augmentation=False
            )
            self.test_dataset = AugmentedSubset(
                self._full_dataset, 
                range(train_len + val_len, total_len),
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
    
    Supports temperature-based sampling to balance dataset representation
    without severe oversampling of small datasets.
    """
    
    def __init__(
        self,
        data_dir: Path,
        context_length: int,
        prediction_length: int,
        datasets: Optional[List[str]] = None,
        exclude_datasets: Optional[List[str]] = None,
        dataset_pattern: str = "*.npy",
        combine_mode: Literal["concatenate", "separate"] = "concatenate",
        # Sampling strategy
        balanced_sampling: bool = True,
        sampling_temperature: float = 0.5,  # sqrt sampling by default
        max_oversample_ratio: float = 5.0,  # cap repetitions
        # Standard params
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
        augmentation_config: Optional[Dict[str, Any]] = None,
        multi_resolution_factors: Optional[List[int]] = None,
        p_multi_resolution: float = 0.0,
        dataset_overrides: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing .npy files (e.g., "data/processed")
            context_length: Context window length
            prediction_length: Prediction window length
            datasets: List of dataset names to load (None = all)
            exclude_datasets: List of dataset names to exclude (e.g., ['bitcoin'])
            dataset_pattern: Glob pattern for dataset files
            combine_mode: How to combine datasets ('concatenate' or 'separate')
            balanced_sampling: If True, use temperature-based sampling
            sampling_temperature: Temperature for sampling distribution
                - T=1.0: proportional to size (large datasets dominate)
                - T=0.5: square-root sampling (balanced compromise) [default]
                - T=0.25: more uniform (small datasets better represented)
            max_oversample_ratio: Max times a dataset can repeat per epoch
            ... (autres params identiques à MonashDataModule)
            dataset_overrides: Per-dataset config overrides
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = Path(data_dir)
        self.datasets = datasets
        self.exclude_datasets = exclude_datasets or []
        self.dataset_pattern = dataset_pattern
        self.combine_mode = combine_mode
        self.balanced_sampling = balanced_sampling
        self.sampling_temperature = sampling_temperature
        self.max_oversample_ratio = max_oversample_ratio
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
        self.multi_resolution_factors = multi_resolution_factors
        self.p_multi_resolution = p_multi_resolution
        self.dataset_overrides = dataset_overrides or {}
        self.seed = seed
        
        self.datamodules: Dict[str, MonashDataModule] = {}
        self.dataset_files: Dict[str, Path] = {}
        
        # Track dataset sizes for sampling
        self.train_dataset_sizes: List[int] = []
        self.val_dataset_sizes: List[int] = []
        self.test_dataset_sizes: List[int] = []
        self.dataset_names_order: List[str] = []
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalizer = None
        
        # Samplers
        self._train_sampler: Optional[TemperatureSampler] = None
        self._val_sampler: Optional[TemperatureSampler] = None
    
    def prepare_data(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        all_files = list(self.data_dir.glob(self.dataset_pattern))
        
        if not all_files:
            raise FileNotFoundError(
                f"No datasets found in {self.data_dir} matching {self.dataset_pattern}"
            )
        
        if self.datasets is None or self.datasets == []:
            logger.info("Using all found datasets")
            self.dataset_files = {f.stem: f for f in all_files}
        elif isinstance(self.datasets, str):
            logger.info(f"Loading single dataset: {self.datasets}")
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
        
        # Apply exclusions
        if self.exclude_datasets:
            for name in self.exclude_datasets:
                if name in self.dataset_files:
                    logger.info(f"Excluding dataset: {name}")
                    del self.dataset_files[name]
        
        if not self.dataset_files:
            raise ValueError("No valid datasets found after filtering")
        
        logger.info(f"Found {len(self.dataset_files)} dataset(s): {list(self.dataset_files.keys())}")
        if self.exclude_datasets:
            logger.info(f"Excluded: {self.exclude_datasets}")
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_datasets = []
            val_datasets = []
            test_datasets = []
            
            # Reset tracking
            self.train_dataset_sizes = []
            self.val_dataset_sizes = []
            self.test_dataset_sizes = []
            self.dataset_names_order = []
            
            for dataset_name, file_path in self.dataset_files.items():
                logger.info(f"Setting up datamodule for: {dataset_name}")
                
                overrides = self.dataset_overrides.get(dataset_name, {})
                
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
                    augmentation_config=overrides.get('augmentation_config', self.augmentation_config),
                    multi_resolution_factors=self.multi_resolution_factors,
                    p_multi_resolution=self.p_multi_resolution,
                    seed=self.seed
                )
                
                dm.prepare_data()
                dm.setup(stage)
                
                self.datamodules[dataset_name] = dm
                
                train_datasets.append(dm.train_dataset)
                val_datasets.append(dm.val_dataset)
                test_datasets.append(dm.test_dataset)
                
                # Track sizes for sampling
                self.train_dataset_sizes.append(len(dm.train_dataset))
                self.val_dataset_sizes.append(len(dm.val_dataset))
                self.test_dataset_sizes.append(len(dm.test_dataset))
                self.dataset_names_order.append(dataset_name)
                
                logger.info(f"  ✓ {dataset_name}: "
                          f"train={len(dm.train_dataset):,}, "
                          f"val={len(dm.val_dataset):,}, "
                          f"test={len(dm.test_dataset):,}")
            
            if self.combine_mode == "concatenate":
                self.train_dataset = ConcatDataset(train_datasets)
                self.val_dataset = ConcatDataset(val_datasets)
                self.test_dataset = ConcatDataset(test_datasets)
                
                logger.info(f"Combined datasets: "
                          f"train={len(self.train_dataset):,}, "
                          f"val={len(self.val_dataset):,}, "
                          f"test={len(self.test_dataset):,}")
                
                # Create temperature samplers if enabled and multiple datasets
                if self.balanced_sampling and len(self.dataset_files) > 1:
                    logger.info(f"Creating TemperatureSampler for training "
                              f"(T={self.sampling_temperature}, max_oversample={self.max_oversample_ratio}x)")
                    
                    self._train_sampler = TemperatureSampler(
                        dataset_sizes=self.train_dataset_sizes,
                        batch_size=self.batch_size,
                        temperature=self.sampling_temperature,
                        max_oversample_ratio=self.max_oversample_ratio,
                        drop_last=True,
                        shuffle=True,
                        seed=self.seed
                    )
                    
                    # Val sampler: T=1.0 (proportional) for fair evaluation
                    self._val_sampler = TemperatureSampler(
                        dataset_sizes=self.val_dataset_sizes,
                        batch_size=self.batch_size,
                        temperature=1.0,  # Proportional for validation
                        max_oversample_ratio=1.0,  # No oversampling for val
                        drop_last=False,
                        shuffle=False,
                        seed=self.seed
                    )
                    
                    logger.info("✓ Temperature-based sampling ENABLED")
                else:
                    self._train_sampler = None
                    self._val_sampler = None
                    if len(self.dataset_files) > 1:
                        logger.warning("⚠ Balanced sampling DISABLED - large datasets will dominate!")
            else:
                first_dm = list(self.datamodules.values())[0]
                self.train_dataset = first_dm.train_dataset
                self.val_dataset = first_dm.val_dataset
                self.test_dataset = first_dm.test_dataset
                self._train_sampler = None
                self._val_sampler = None
                logger.info(f"Using single dataset: {list(self.datamodules.keys())[0]}")
            
            self.normalizer = list(self.datamodules.values())[0].normalizer
            
            if self.augmentation_config:
                logger.info(f"✓ Augmentations enabled for training across all datasets")
    
    def train_dataloader(self) -> DataLoader:
        if self._train_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self._train_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            )
        else:
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
        # Pour la validation, on évalue tous les samples sans sampling biaisé
        # pour avoir des métriques représentatives de chaque dataset
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
    
    def on_train_epoch_start(self):
        """Update sampler epoch for proper shuffling."""
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self.trainer.current_epoch)
    
    def get_dataset_info(self) -> Dict[str, Dict[str, int]]:
        """Return info about each dataset for logging/debugging."""
        return {
            name: {
                "train": self.train_dataset_sizes[i],
                "val": self.val_dataset_sizes[i],
                "test": self.test_dataset_sizes[i],
            }
            for i, name in enumerate(self.dataset_names_order)
        }