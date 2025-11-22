import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class Config:
    """Configuration manager for loading and validating configs."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, uses default.
        """
        if config_path is None:
            # Default to configs/datasets.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "datasets.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise ConfigError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {self.config_path}: {e}") from e
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset."""
        datasets = self.config.get('datasets', {})
        
        if dataset_name not in datasets:
            available = ', '.join(datasets.keys())
            raise ConfigError(
                f"Dataset '{dataset_name}' not found in config. "
                f"Available datasets: {available}"
            )
        
        return datasets[dataset_name]
    
    def get_download_config(self) -> Dict[str, Any]:
        """Get download configuration."""
        return self.config.get('download', {})
    
    def list_datasets(self) -> list:
        """List all available datasets."""
        return list(self.config.get('datasets', {}).keys())