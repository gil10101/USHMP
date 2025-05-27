#!/usr/bin/env python3
"""
Download Zillow Home Value Index (ZHVI) data.

This script downloads the latest ZHVI data from Zillow Research and saves it
to the data/raw directory for further processing.
"""

import os
import sys
import time
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_data.log')
    ]
)
logger = logging.getLogger(__name__)


class ZillowDataDownloader:
    """Download and manage Zillow ZHVI data."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the downloader.
        
        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # ZHVI data URL
        self.zhvi_url = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def download_zhvi_data(self, force_download: bool = False) -> Optional[Path]:
        """
        Download ZHVI data from Zillow.
        
        Args:
            force_download: If True, download even if file exists
            
        Returns:
            Path to downloaded file or None if failed
        """
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"zhvi_zip_data_{timestamp}.csv"
        filepath = self.data_dir / filename
        
        # Check if recent file exists (unless force download)
        if not force_download:
            existing_file = self._find_recent_zhvi_file()
            if existing_file:
                logger.info(f"Recent ZHVI file found: {existing_file}")
                return existing_file
        
        logger.info(f"Downloading ZHVI data from: {self.zhvi_url}")
        logger.info(f"Saving to: {filepath}")
        
        try:
            # Add timestamp parameter to URL to avoid caching
            url_with_timestamp = f"{self.zhvi_url}?t={timestamp}"
            
            response = self.session.get(url_with_timestamp, timeout=300)
            response.raise_for_status()
            
            # Check if response contains CSV data
            content_type = response.headers.get('content-type', '').lower()
            if 'text/csv' not in content_type and 'application/csv' not in content_type:
                logger.warning(f"Unexpected content type: {content_type}")
            
            # Save the file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            file_size = filepath.stat().st_size
            logger.info(f"Successfully downloaded {file_size:,} bytes to {filepath}")
            
            # Validate the downloaded file
            if self._validate_csv_file(filepath):
                logger.info("File validation successful")
                return filepath
            else:
                logger.error("File validation failed")
                filepath.unlink()  # Delete invalid file
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download ZHVI data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return None
    
    def _find_recent_zhvi_file(self, max_age_hours: int = 24) -> Optional[Path]:
        """
        Find a recently downloaded ZHVI file.
        
        Args:
            max_age_hours: Maximum age of file to consider recent
            
        Returns:
            Path to recent file or None if not found
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in self.data_dir.glob("zhvi_zip_data_*.csv"):
            file_age = current_time - file_path.stat().st_mtime
            if file_age < max_age_seconds:
                return file_path
        
        return None
    
    def _validate_csv_file(self, filepath: Path) -> bool:
        """
        Validate the downloaded CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read first few lines to check format
                header = f.readline().strip()
                
                # Check if header contains expected columns
                expected_columns = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
                if not any(col in header for col in expected_columns):
                    logger.error("CSV header doesn't contain expected ZHVI columns")
                    return False
                
                # Check if file has data beyond header
                second_line = f.readline().strip()
                if not second_line:
                    logger.error("CSV file appears to be empty (no data rows)")
                    return False
                
                logger.info(f"CSV validation passed. Header: {header[:100]}...")
                return True
                
        except Exception as e:
            logger.error(f"Error validating CSV file: {e}")
            return False
    
    def list_downloaded_files(self) -> list[Path]:
        """
        List all downloaded ZHVI files.
        
        Returns:
            List of paths to downloaded files
        """
        return list(self.data_dir.glob("zhvi_zip_data_*.csv"))
    
    def cleanup_old_files(self, keep_count: int = 3) -> None:
        """
        Remove old downloaded files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent files to keep
        """
        files = sorted(self.list_downloaded_files(), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(files) > keep_count:
            for old_file in files[keep_count:]:
                logger.info(f"Removing old file: {old_file}")
                old_file.unlink()


def main():
    """Main function to run the download script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Zillow ZHVI data")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force download even if recent file exists"
    )
    parser.add_argument(
        "--data-dir", 
        default="data/raw", 
        help="Directory to save downloaded data (default: data/raw)"
    )
    parser.add_argument(
        "--cleanup", 
        action="store_true", 
        help="Clean up old downloaded files"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List existing downloaded files"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ZillowDataDownloader(data_dir=args.data_dir)
    
    if args.list:
        files = downloader.list_downloaded_files()
        if files:
            logger.info("Downloaded ZHVI files:")
            for file_path in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                size = file_path.stat().st_size
                mtime = time.ctime(file_path.stat().st_mtime)
                logger.info(f"  {file_path.name} ({size:,} bytes, {mtime})")
        else:
            logger.info("No downloaded ZHVI files found")
        return
    
    if args.cleanup:
        downloader.cleanup_old_files()
    
    # Download data
    result = downloader.download_zhvi_data(force_download=args.force)
    
    if result:
        logger.info(f"Download completed successfully: {result}")
        sys.exit(0)
    else:
        logger.error("Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 