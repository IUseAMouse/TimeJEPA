import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Custom exception for download errors."""
    pass


class Downloader:
    """Robust downloader with retry logic and resume capability."""
    
    def __init__(
        self,
        chunk_size: int = 8192,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configure session with retries
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def download(
        self,
        url: str,
        save_path: Union[str, Path],
        resume: bool = True,
        verify_ssl: bool = True
    ) -> Path:
        """
        Download a file with progress bar and resume capability.
        
        Args:
            url: URL to download from
            save_path: Where to save the file
            resume: Whether to resume partial downloads
            verify_ssl: Whether to verify SSL certificates
            
        Returns:
            Path to downloaded file
            
        Raises:
            DownloadError: If download fails after all retries
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists and is complete
        if save_path.exists() and not resume:
            logger.info(f"File already exists: {save_path}")
            return save_path
        
        # Determine resume position
        resume_byte_pos = save_path.stat().st_size if (resume and save_path.exists()) else 0
        
        headers = {}
        if resume_byte_pos:
            headers['Range'] = f'bytes={resume_byte_pos}-'
            logger.info(f"Resuming download from byte {resume_byte_pos}")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading {url} to {save_path} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.session.get(
                    url,
                    stream=True,
                    timeout=self.timeout,
                    headers=headers,
                    verify=verify_ssl
                )
                response.raise_for_status()
                
                # Get total file size
                total_size = int(response.headers.get('content-length', 0))
                if resume_byte_pos:
                    total_size += resume_byte_pos
                
                mode = 'ab' if resume_byte_pos else 'wb'
                
                with open(save_path, mode) as file, tqdm(
                    desc=save_path.name,
                    initial=resume_byte_pos,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            size = file.write(chunk)
                            progress_bar.update(size)
                
                logger.info(f"Successfully downloaded to {save_path}")
                return save_path
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise DownloadError(f"Failed to download {url} after {self.max_retries} attempts") from e
            
            except Exception as e:
                logger.error(f"Unexpected error during download: {e}")
                raise DownloadError(f"Download failed: {e}") from e
        
        raise DownloadError(f"Failed to download {url}")
    
    @staticmethod
    def verify_checksum(file_path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify file integrity using checksum."""
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        actual_hash = hash_func.hexdigest()
        return actual_hash == expected_hash