import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.config import Config
from timejepa.data.downloader import Downloader, DownloadError
from timejepa.data.parser import DataArchive, TSFileParser, save_processed_data, ParsingError


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def download_and_process_dataset(
    dataset_name: str,
    data_dir: Path,
    config: Config,
    force_download: bool = False,
    force_process: bool = False,
    compress: bool = False
) -> Optional[Path]:
    """
    Download and process a single dataset.
    
    Args:
        dataset_name: Name of dataset
        data_dir: Root data directory
        config: Configuration object
        force_download: Force re-download even if file exists
        force_process: Force re-process even if output exists
        compress: Compress output file
        
    Returns:
        Path to processed data file
    """
    logger = logging.getLogger(__name__)
    
    # Get configs
    dataset_config = config.get_dataset_config(dataset_name)
    download_config = config.get_download_config()
    
    # Setup paths
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / f"{dataset_name}.zip"
    output_path = processed_dir / f"{dataset_name}.npy"
    
    # Check if already processed
    if output_path.exists() and not force_process:
        logger.info(f"Processed data already exists: {output_path}")
        logger.info("Use --force-process to reprocess")
        return output_path
    
    # 1. Download
    url = dataset_config['url']
    
    if zip_path.exists() and not force_download:
        logger.info(f"Zip file already exists: {zip_path}")
    else:
        try:
            downloader = Downloader(**download_config)
            downloader.download(url, zip_path, resume=True)
        except DownloadError as e:
            logger.error(f"Failed to download dataset: {e}")
            return None
    
    # 2. Extract
    try:
        extract_dir = raw_dir / dataset_name
        DataArchive.extract_zip(zip_path, extract_dir)
    except Exception as e:
        logger.error(f"Failed to extract archive: {e}")
        return None
    
    # 3. Find .ts files
    expected_files = dataset_config.get('expected_files', ['*.ts'])
    ts_files = []
    for pattern in expected_files:
        ts_files.extend(extract_dir.rglob(pattern))
    
    if not ts_files:
        logger.error(f"No .ts files found in {extract_dir}")
        logger.error(f"Expected patterns: {expected_files}")
        return None
    
    logger.info(f"Found {len(ts_files)} .ts file(s)")
    
    # 4. Parse
    # Take the first .ts file or merge if multiple
    target_file = ts_files[0]
    logger.info(f"Processing {target_file}")
    
    try:
        parser = TSFileParser(handle_missing='nan')
        series_list = parser.parse(target_file)
    except ParsingError as e:
        logger.error(f"Failed to parse file: {e}")
        return None
    
    # 5. Save
    try:
        save_processed_data(series_list, output_path, compress=compress)
        logger.info(f"✓ Successfully processed {dataset_name}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download and process time series datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of dataset to download (or 'all' for all datasets)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory for data storage"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (default: configs/datasets.yaml)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Force re-process even if output exists"
    )
    
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output files"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config_path = Path(args.config) if args.config else None
        config = Config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # List datasets if requested
    if args.list:
        datasets = config.list_datasets()
        print("\nAvailable datasets:")
        for name in datasets:
            dataset_config = config.get_dataset_config(name)
            desc = dataset_config.get('description', 'No description')
            print(f"  - {name}: {desc}")
        sys.exit(0)
    
    # Require dataset name
    if not args.dataset:
        parser.error("--dataset is required (or use --list to see available datasets)")
    
    data_dir = Path(args.data_dir)
    
    # Process dataset(s)
    if args.dataset.lower() == 'all':
        datasets = config.list_datasets()
        logger.info(f"Processing all {len(datasets)} datasets...")
        
        success_count = 0
        for dataset_name in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'='*60}")
            
            result = download_and_process_dataset(
                dataset_name,
                data_dir,
                config,
                args.force_download,
                args.force_process,
                args.compress
            )
            
            if result:
                success_count += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Completed: {success_count}/{len(datasets)} datasets processed successfully")
        logger.info(f"{'='*60}")
        
    else:
        result = download_and_process_dataset(
            args.dataset,
            data_dir,
            config,
            args.force_download,
            args.force_process,
            args.compress
        )
        
        if not result:
            sys.exit(1)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()