"""
Zillow Data Processor

This module handles loading, cleaning, and processing Zillow Home Value Index (ZHVI)
data for time series forecasting. It provides methods to extract time series data
for specific ZIP codes and prepare it for model input.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from utils import validate_zip_code, ValidationError

logger = logging.getLogger(__name__)


class ZillowDataProcessor:
    """
    Processor for Zillow ZHVI data.
    
    This class handles:
    - Loading ZHVI CSV data
    - Extracting time series for specific ZIP codes
    - Data validation and cleaning
    - Metadata extraction
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the ZHVI CSV file
        """
        self.data_path = data_path
        self.data = None
        self.date_columns = None
        self.metadata_columns = None
        self.loaded = False
        
    def load_data(self) -> None:
        """Load the ZHVI data from CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ZHVI data file not found: {self.data_path}")
        
        try:
            logger.info(f"Loading ZHVI data from {self.data_path}")
            
            # Load the data
            self.data = pd.read_csv(self.data_path)
            
            # Identify date columns (columns that look like dates)
            self.date_columns = []
            self.metadata_columns = []
            
            for col in self.data.columns:
                try:
                    # Try to parse as date
                    pd.to_datetime(col)
                    self.date_columns.append(col)
                except:
                    # Not a date column, it's metadata
                    self.metadata_columns.append(col)
            
            # Sort date columns chronologically
            self.date_columns = sorted(self.date_columns, key=pd.to_datetime)
            
            logger.info(f"Loaded data: {len(self.data)} ZIP codes, {len(self.date_columns)} time periods")
            logger.info(f"Date range: {self.date_columns[0]} to {self.date_columns[-1]}")
            
            self.loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load ZHVI data: {e}")
            raise
    
    def get_zip_time_series(self, zip_code: str) -> Optional[pd.Series]:
        """
        Get time series data for a specific ZIP code.
        
        Args:
            zip_code: 5-digit ZIP code
            
        Returns:
            Time series data as pandas Series with datetime index, or None if not found
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        if not validate_zip_code(zip_code):
            raise ValueError(f"Invalid ZIP code format: {zip_code}")
        
        # Find the ZIP code in the data
        # The ZIP code might be in 'RegionName' column or similar
        zip_code_int = int(zip_code)
        
        # Try different possible column names for ZIP codes
        possible_zip_columns = ['RegionName', 'ZipCode', 'ZIP', 'zip_code']
        zip_row = None
        
        for col in possible_zip_columns:
            if col in self.data.columns:
                mask = self.data[col] == zip_code_int
                if mask.any():
                    zip_row = self.data[mask].iloc[0]
                    break
        
        if zip_row is None:
            logger.warning(f"ZIP code {zip_code} not found in data")
            return None
        
        # Extract time series values
        time_series_values = zip_row[self.date_columns]
        
        # Create datetime index
        datetime_index = pd.to_datetime(self.date_columns)
        
        # Create series with datetime index
        time_series = pd.Series(
            time_series_values.values,
            index=datetime_index,
            name=f"ZIP_{zip_code}"
        )
        
        # Remove NaN values from the end (future dates)
        time_series = time_series.dropna()
        
        if len(time_series) == 0:
            logger.warning(f"No valid data found for ZIP code {zip_code}")
            return None
        
        logger.info(f"Retrieved {len(time_series)} data points for ZIP {zip_code}")
        logger.info(f"Data range: {time_series.index[0]} to {time_series.index[-1]}")
        logger.info(f"Current value: ${time_series.iloc[-1]:,.2f}")
        
        return time_series
    
    def get_zip_metadata(self, zip_code: str) -> Dict[str, Any]:
        """
        Get metadata for a specific ZIP code.
        
        Args:
            zip_code: 5-digit ZIP code
            
        Returns:
            Dictionary containing ZIP code metadata
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        if not validate_zip_code(zip_code):
            raise ValueError(f"Invalid ZIP code format: {zip_code}")
        
        # Find the ZIP code in the data
        zip_code_int = int(zip_code)
        
        # Try different possible column names for ZIP codes
        possible_zip_columns = ['RegionName', 'ZipCode', 'ZIP', 'zip_code']
        zip_row = None
        
        for col in possible_zip_columns:
            if col in self.data.columns:
                mask = self.data[col] == zip_code_int
                if mask.any():
                    zip_row = self.data[mask].iloc[0]
                    break
        
        if zip_row is None:
            return {"error": f"ZIP code {zip_code} not found"}
        
        # Extract metadata
        metadata = {}
        for col in self.metadata_columns:
            if col in zip_row.index and pd.notna(zip_row[col]):
                metadata[col] = zip_row[col]
        
        return metadata
    
    def get_available_zip_codes(self, min_data_points: int = 24) -> List[str]:
        """
        Get list of ZIP codes with sufficient data.
        
        Args:
            min_data_points: Minimum number of data points required
            
        Returns:
            List of ZIP codes with sufficient data
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        available_zips = []
        
        # Find ZIP code column
        zip_column = None
        possible_zip_columns = ['RegionName', 'ZipCode', 'ZIP', 'zip_code']
        
        for col in possible_zip_columns:
            if col in self.data.columns:
                zip_column = col
                break
        
        if zip_column is None:
            logger.error("No ZIP code column found in data")
            return []
        
        for _, row in self.data.iterrows():
            zip_code = str(int(row[zip_column])).zfill(5)
            
            # Count non-null data points
            data_points = row[self.date_columns].notna().sum()
            
            if data_points >= min_data_points:
                available_zips.append(zip_code)
        
        logger.info(f"Found {len(available_zips)} ZIP codes with ≥{min_data_points} data points")
        return available_zips
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the loaded data.
        
        Returns:
            Dictionary containing data summary
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        # Calculate summary statistics
        total_zip_codes = len(self.data)
        total_time_periods = len(self.date_columns)
        
        # Calculate data completeness
        price_data = self.data[self.date_columns]
        total_cells = price_data.size
        non_null_cells = price_data.notna().sum().sum()
        completeness_pct = (non_null_cells / total_cells) * 100
        
        # Price statistics
        all_prices = price_data.values.flatten()
        all_prices = all_prices[~pd.isna(all_prices)]
        
        summary = {
            "total_zip_codes": total_zip_codes,
            "total_time_periods": total_time_periods,
            "date_range": {
                "start": self.date_columns[0],
                "end": self.date_columns[-1]
            },
            "data_completeness_pct": completeness_pct,
            "price_statistics": {
                "count": len(all_prices),
                "mean": float(np.mean(all_prices)),
                "median": float(np.median(all_prices)),
                "std": float(np.std(all_prices)),
                "min": float(np.min(all_prices)),
                "max": float(np.max(all_prices)),
                "percentiles": {
                    "25th": float(np.percentile(all_prices, 25)),
                    "75th": float(np.percentile(all_prices, 75)),
                    "90th": float(np.percentile(all_prices, 90)),
                    "95th": float(np.percentile(all_prices, 95))
                }
            },
            "metadata_columns": self.metadata_columns
        }
        
        return summary
    
    def validate_zip_data(self, zip_code: str, min_data_points: int = 12) -> Tuple[bool, str]:
        """
        Validate that a ZIP code has sufficient data for modeling.
        
        Args:
            zip_code: ZIP code to validate
            min_data_points: Minimum number of data points required
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            time_series = self.get_zip_time_series(zip_code)
            
            if time_series is None:
                return False, f"ZIP code {zip_code} not found in data"
            
            if len(time_series) < min_data_points:
                return False, f"Insufficient data: {len(time_series)} points (need {min_data_points})"
            
            # Check for recent data (within last 2 years)
            latest_date = time_series.index[-1]
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=2)
            
            if latest_date < cutoff_date:
                return False, f"Data too old: latest data from {latest_date.strftime('%Y-%m')}"
            
            # Check for reasonable price values
            if time_series.min() <= 0:
                return False, "Invalid price data: contains zero or negative values"
            
            if time_series.max() > 50_000_000:  # $50M seems unreasonable for most homes
                return False, "Invalid price data: contains unreasonably high values"
            
            return True, f"Valid: {len(time_series)} data points from {time_series.index[0].strftime('%Y-%m')} to {time_series.index[-1].strftime('%Y-%m')}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


def quick_load_zip_data(zip_code: str, data_path: str = "data/raw/zhvi_zip.csv") -> Optional[pd.Series]:
    """
    Quick function to load time series data for a ZIP code.
    
    Args:
        zip_code: 5-digit ZIP code
        data_path: Path to ZHVI data file
        
    Returns:
        Time series data or None if not found
    """
    processor = ZillowDataProcessor(data_path)
    processor.load_data()
    return processor.get_zip_time_series(zip_code)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    data_path = "data/raw/zhvi_zip.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please download ZHVI data first using: python scripts/download_data.py")
        sys.exit(1)
    
    # Initialize processor
    processor = ZillowDataProcessor(data_path)
    processor.load_data()
    
    # Get data summary
    summary = processor.get_data_summary()
    print("\nData Summary:")
    print(f"Total ZIP codes: {summary['total_zip_codes']:,}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Data completeness: {summary['data_completeness_pct']:.1f}%")
    print(f"Price range: ${summary['price_statistics']['min']:,.0f} - ${summary['price_statistics']['max']:,.0f}")
    
    # Test with a few ZIP codes
    test_zips = ["90210", "77449", "10001"]
    
    for zip_code in test_zips:
        print(f"\nTesting ZIP code {zip_code}:")
        
        # Validate
        is_valid, message = processor.validate_zip_data(zip_code)
        print(f"Validation: {message}")
        
        if is_valid:
            # Get metadata
            metadata = processor.get_zip_metadata(zip_code)
            if 'City' in metadata and 'State' in metadata:
                print(f"Location: {metadata['City']}, {metadata['State']}")
            
            # Get time series
            ts = processor.get_zip_time_series(zip_code)
            if ts is not None:
                print(f"Current value: ${ts.iloc[-1]:,.2f}")
                print(f"1-year change: {((ts.iloc[-1] / ts.iloc[-13]) - 1) * 100:+.1f}%" if len(ts) >= 13 else "N/A") 