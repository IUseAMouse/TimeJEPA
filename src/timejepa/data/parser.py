import logging
import zipfile
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ParsingError(Exception):
    """Custom exception for parsing errors."""
    pass


class TSFileParser:
    """Parser for .ts time series files (Monash/sktime format)."""
    
    def __init__(self, handle_missing: str = 'nan'):
        """
        Args:
            handle_missing: How to handle missing values ('nan', 'zero', 'drop')
        """
        if handle_missing not in ['nan', 'zero', 'drop']:
            raise ValueError("handle_missing must be 'nan', 'zero', or 'drop'")
        self.handle_missing = handle_missing
    
    def parse(self, file_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Parse a .ts or .tsf file and return list of time series.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix not in ['.ts', '.tsf']:
            logger.warning(f"File {file_path} has unexpected extension: {file_path.suffix}")
        
        logger.info(f"Parsing {file_path}...")
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                used_encoding = encoding
                logger.info(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ParsingError(f"Failed to decode {file_path} with any known encoding")
        
        lines = content.strip().split('\n')
        
        data = []
        started = False
        line_num = 0
        error_count = 0
        
        for line in tqdm(lines, desc="Parsing lines"):
            line_num += 1
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line.lower().startswith("@data"):
                started = True
                continue
            
            if started:
                try:
                    series = self._parse_line(line)
                    if series is not None and len(series) > 0:
                        data.append(series)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Error parsing line {line_num}: {e}")
                    if error_count > 100:
                        raise ParsingError(f"Too many parsing errors (>{error_count})")
        
        if not data:
            raise ParsingError("No valid time series found in file")
        
        logger.info(f"Successfully parsed {len(data)} time series")
        if error_count > 0:
            logger.warning(f"Encountered {error_count} parsing errors")
        
        return data
    
    def _parse_line(self, line: str) -> Optional[np.ndarray]:
        """Parse a single line into a time series array."""
        # Split by colon to separate metadata from time series data
        parts = line.split(':')
        
        # The last part typically contains the time series values
        ts_string = parts[-1].strip()
        
        # Split by comma to get individual values
        values = ts_string.split(',')
        
        ts_array = []
        for val in values:
            val = val.strip()
            
            if val == '?' or val == '':
                # Missing value
                if self.handle_missing == 'nan':
                    ts_array.append(np.nan)
                elif self.handle_missing == 'zero':
                    ts_array.append(0.0)
                # 'drop' means skip this value
            else:
                try:
                    ts_array.append(float(val))
                except ValueError:
                    # Not a number, might be a class label - skip
                    pass
        
        if ts_array:
            return np.array(ts_array, dtype=np.float32)
        return None


class DataArchive:
    """Utility class for handling compressed archives."""
    
    @staticmethod
    def extract_zip(
        zip_path: Union[str, Path],
        extract_to: Union[str, Path],
        overwrite: bool = False
    ) -> Path:
        """
        Extract a zip file.
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to extraction directory
        """
        zip_path = Path(zip_path)
        extract_to = Path(extract_to)
        
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting {zip_path} to {extract_to}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                
                if not overwrite:
                    # Filter out already extracted files
                    members = [m for m in members if not (extract_to / m).exists()]
                
                if members:
                    for member in tqdm(members, desc="Extracting"):
                        zip_ref.extract(member, extract_to)
                else:
                    logger.info("All files already extracted")
            
            logger.info(f"Extraction complete")
            return extract_to
            
        except zipfile.BadZipFile as e:
            raise IOError(f"Corrupted zip file: {zip_path}") from e


def save_processed_data(
    data: List[np.ndarray],
    output_path: Union[str, Path],
    compress: bool = False
) -> dict:
    """Save processed time series data and return statistics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        raise ValueError("Cannot save empty data")
    
    # Compute statistics
    lengths = [len(x) for x in data]
    stats = {
        'num_series': len(data),
        'total_points': sum(lengths),
        'min_len': min(lengths),
        'max_len': max(lengths),
        'mean_len': np.mean(lengths),
        'median_len': np.median(lengths)
    }
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  - Number of series: {stats['num_series']}")
    logger.info(f"  - Total datapoints: {stats['total_points']:,}")
    logger.info(f"  - Length range: [{stats['min_len']}, {stats['max_len']}]")
    logger.info(f"  - Mean/Median length: {stats['mean_len']:.1f} / {stats['median_len']:.1f}")
    
    # Stack or keep as object array
    if len(set(lengths)) == 1:
        final_data = np.stack(data)
    else:
        final_data = np.array(data, dtype=object)
    
    # Save JUST the data array (not a dict)
    if compress:
        np.savez_compressed(output_path.with_suffix('.npz'), data=final_data)
    else:
        np.save(output_path, final_data)  # ← FIX ICI : sauve juste l'array
    
    logger.info(f"Saved to {output_path}")
    
    return stats