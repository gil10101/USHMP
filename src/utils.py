"""
Utility functions and constants for the Home Price Prediction project.

This module provides helper functions for configuration management,
data validation, and common utilities used across the project.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration management class that loads settings from environment variables
    and provides defaults for the home price prediction project.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML config file (optional)
        """
        self.config_file = config_file or "config/config.yaml"
        self._config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file if it exists."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
                self._config = {}
        else:
            logger.info(f"Config file {self.config_file} not found, using environment variables only")
    
    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """
        Get configuration value with fallback to environment variables.
        
        Args:
            key: Configuration key
            default: Default value if not found
            section: YAML section to look in
            
        Returns:
            Configuration value
        """
        # Try environment variable first (convert key to uppercase)
        env_key = key.upper()
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._convert_env_value(env_value)
        
        # Try YAML config
        if section and section in self._config:
            return self._config[section].get(key, default)
        elif key in self._config:
            return self._config[key]
        
        return default
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value


# Global configuration instance
config = Config()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = config.get("model_name", "amazon/chronos-t5-small", "model")
MODEL_CACHE_DIR = config.get("model_cache_dir", "./data/model_cache/")
MODEL_DEVICE = config.get("model_device", "auto")
MAX_CONTEXT_LENGTH = config.get("max_context_length", 512)
MIN_HISTORY_POINTS = config.get("min_history_points", 12)
DEFAULT_NUM_SAMPLES = config.get("default_num_samples", 100)

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_HOST = config.get("api_host", "0.0.0.0", "api")
API_PORT = config.get("api_port", 8000, "api")
API_DEBUG = config.get("api_debug", True)
API_TIMEOUT = config.get("api_timeout", 300)
MAX_CONCURRENT_REQUESTS = config.get("max_concurrent_requests", 10)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

ZHVI_DATA_PATH = config.get("zhvi_file", "./data/raw/zhvi_zip.csv", "data")
PROCESSED_DATA_DIR = config.get("processed_data_dir", "./data/processed/")
RAW_DATA_DIR = config.get("raw_data_dir", "./data/raw/")
MIN_DATA_MONTHS = config.get("min_data_points", 24, "data")
DATA_CACHE_DAYS = config.get("data_cache_days", 7)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = config.get("log_level", "INFO")
LOG_FILE = config.get("log_file", "")
LOG_FORMAT = config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# =============================================================================
# FORECAST CONFIGURATION
# =============================================================================

DEFAULT_FORECAST_HORIZONS = config.get("forecast_horizons", [3, 6, 12], "model")
DEFAULT_CONFIDENCE_LEVEL = config.get("confidence_level", 0.8)
MAX_FORECAST_HORIZON = config.get("max_forecast_horizon", 24)

# =============================================================================
# FEATURE FLAGS
# =============================================================================

ENABLE_BATCH_PREDICTIONS = config.get("enable_batch_predictions", True)
ENABLE_MODEL_WARMUP = config.get("enable_model_warmup", True)
ENABLE_DATA_VALIDATION = config.get("enable_data_validation", True)
DEVELOPMENT_MODE = config.get("development_mode", True)


def setup_logging(level: str = LOG_LEVEL, log_file: str = LOG_FILE) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Log file path (empty for console only)
    """
    # Create logs directory if logging to file
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),  # Console handler
            *([logging.FileHandler(log_file)] if log_file else [])  # File handler if specified
        ]
    )
    
    logger.info(f"Logging configured with level: {level}")


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        MODEL_CACHE_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        Path(LOG_FILE).parent if LOG_FILE else None
    ]
    
    for directory in directories:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


def validate_zip_code(zip_code: str) -> bool:
    """
    Validate ZIP code format.
    
    Args:
        zip_code: ZIP code to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Basic ZIP code validation (5 digits or 5+4 format)
    import re
    pattern = r'^\d{5}(-\d{4})?$'
    return bool(re.match(pattern, str(zip_code)))


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def get_model_info() -> Dict[str, Any]:
    """
    Get model configuration information.
    
    Returns:
        Dictionary with model configuration
    """
    return {
        "model_name": MODEL_NAME,
        "cache_dir": MODEL_CACHE_DIR,
        "device": MODEL_DEVICE,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "min_history_points": MIN_HISTORY_POINTS,
        "default_num_samples": DEFAULT_NUM_SAMPLES,
        "default_forecast_horizons": DEFAULT_FORECAST_HORIZONS
    }


def get_api_info() -> Dict[str, Any]:
    """
    Get API configuration information.
    
    Returns:
        Dictionary with API configuration
    """
    return {
        "host": API_HOST,
        "port": API_PORT,
        "debug": API_DEBUG,
        "timeout": API_TIMEOUT,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
    }


def validate_forecast_horizon(horizon: int) -> bool:
    """
    Validate forecast horizon.
    
    Args:
        horizon: Forecast horizon in months
        
    Returns:
        True if valid, False otherwise
    """
    return 1 <= horizon <= MAX_FORECAST_HORIZON


def validate_confidence_level(confidence: float) -> bool:
    """
    Validate confidence level.
    
    Args:
        confidence: Confidence level (0-1)
        
    Returns:
        True if valid, False otherwise
    """
    return 0.0 < confidence < 1.0


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_time_series_data(data: List[float], min_points: int = MIN_HISTORY_POINTS) -> None:
    """
    Validate time series data.
    
    Args:
        data: Time series data
        min_points: Minimum number of data points required
        
    Raises:
        ValidationError: If data is invalid
    """
    if not data:
        raise ValidationError("Time series data cannot be empty")
    
    if len(data) < min_points:
        raise ValidationError(f"Time series must have at least {min_points} data points, got {len(data)}")
    
    # Check for all NaN values
    import math
    valid_points = [x for x in data if not math.isnan(x)]
    if len(valid_points) < min_points:
        raise ValidationError(f"Time series must have at least {min_points} valid (non-NaN) data points")
    
    # Check for negative values (home prices should be positive)
    if any(x <= 0 for x in valid_points):
        raise ValidationError("Home prices must be positive values")


# Constants for data processing
MONTHS_PER_YEAR = 12
DAYS_PER_MONTH = 30.44 
SECONDS_PER_DAY = 86400

# ZIP code patterns for different regions
ZIP_CODE_PATTERNS = {
    "northeast": r"^0[0-9]{4}$|^1[0-9]{4}$",
    "southeast": r"^2[0-9]{4}$|^3[0-9]{4}$",
    "midwest": r"^4[0-9]{4}$|^5[0-9]{4}$|^6[0-9]{4}$",
    "west": r"^7[0-9]{4}$|^8[0-9]{4}$|^9[0-9]{4}$"
}

STATE_ZIP_RANGES = {
    # Northeast
    "CT": [(6001, 6999)],
    "MA": [(1001, 2799)],
    "ME": [(3901, 4999)],
    "NH": [(3001, 3899)],
    "NJ": [(7001, 8999)],
    "NY": [(10001, 14999)],
    "PA": [(15001, 19699)],
    "RI": [(2801, 2999)],
    "VT": [(5001, 5999)],
    
    # Southeast
    "AL": [(35004, 36925)],
    "AR": [(71601, 72959)],
    "DE": [(19701, 19980)],
    "FL": [(32003, 34997)],
    "GA": [(30002, 39901)],
    "KY": [(40003, 42788)],
    "LA": [(70001, 71497)],
    "MD": [(20588, 21930)],
    "MS": [(38601, 39776)],
    "NC": [(27006, 28909)],
    "SC": [(29001, 29948)],
    "TN": [(37010, 38589)],
    "VA": [(20105, 26886)],
    "WV": [(24701, 26886)],
    
    # Midwest
    "IA": [(50001, 52809)],
    "IL": [(60001, 62999)],
    "IN": [(46001, 47997)],
    "KS": [(66002, 67954)],
    "MI": [(48001, 49971)],
    "MN": [(55001, 56763)],
    "MO": [(63001, 65899)],
    "ND": [(58001, 58856)],
    "NE": [(68001, 69367)],
    "OH": [(43001, 45999)],
    "SD": [(57001, 57799)],
    "WI": [(53001, 54990)],
    
    # West
    "AK": [(99501, 99950)],
    "AZ": [(85001, 86556)],
    "CA": [(90001, 96162)],
    "CO": [(80001, 81658)],
    "HI": [(96701, 96898)],
    "ID": [(83201, 83877)],
    "MT": [(59001, 59937)],
    "NM": [(87001, 88441)],
    "NV": [(89001, 89883)],
    "OR": [(97001, 97920)],
    "UT": [(84001, 84791)],
    "WA": [(98001, 99403)],
    "WY": [(82001, 83128)]
}

METRO_ZIP_RANGES = {
    "new_york": [(10001, 10299), (10451, 10475), (11004, 11109), (11201, 11256)],
    "los_angeles": [(90001, 90084), (90086, 90089), (90091, 90096), (90099, 90189)],
    "chicago": [(60601, 60661), (60701, 60827)],
    "dallas": [(75001, 75398)],
    "philadelphia": [(19019, 19197)],
    "houston": [(77001, 77598)],
    "washington_dc": [(20001, 20599)],
    "miami": [(33101, 33299)],
    "atlanta": [(30301, 30398)],
    "boston": [(2101, 2137), (2201, 2297)],
    "san_francisco": [(94101, 94188)],
    "phoenix": [(85001, 85099)],
    "riverside": [(92201, 92399)],
    "detroit": [(48201, 48288)],
    "seattle": [(98101, 98199)],
    "minneapolis": [(55401, 55488)],
    "san_diego": [(92101, 92199)],
    "tampa": [(33601, 33699)],
    "denver": [(80201, 80299)],
    "baltimore": [(21201, 21298)]
}


