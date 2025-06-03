"""
Configuration Constants and Enumerations

This module provides system-wide constants and enumerations used throughout
the Home Price Prediction system.
"""

from enum import Enum
from typing import Final


# =============================================================================
# Model Constants
# =============================================================================

class ModelName(str, Enum):
    """Supported forecasting models."""
    CHRONOS_T5_SMALL = "amazon/chronos-t5-small"
    CHRONOS_T5_BASE = "amazon/chronos-t5-base"
    CHRONOS_T5_LARGE = "amazon/chronos-t5-large"


class DeviceType(str, Enum):
    """Supported device types for model inference."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class TorchDType(str, Enum):
    """Supported PyTorch data types."""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


# =============================================================================
# Data Processing Constants
# =============================================================================

class OutlierDetectionMethod(str, Enum):
    """Outlier detection methods."""
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"


class NormalizationMethod(str, Enum):
    """Data normalization methods."""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"


class DataFormat(str, Enum):
    """Supported data storage formats."""
    PARQUET = "parquet"
    CSV = "csv"
    FEATHER = "feather"
    PICKLE = "pickle"


class CompressionMethod(str, Enum):
    """Data compression methods."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    BROTLI = "brotli"


# =============================================================================
# API Constants
# =============================================================================

class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIVersion(str, Enum):
    """API versions."""
    V1 = "v1"
    V2 = "v2"


# Standard HTTP status codes
HTTP_STATUS: Final = {
    "OK": 200,
    "CREATED": 201,
    "ACCEPTED": 202,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "METHOD_NOT_ALLOWED": 405,
    "CONFLICT": 409,
    "UNPROCESSABLE_ENTITY": 422,
    "TOO_MANY_REQUESTS": 429,
    "INTERNAL_SERVER_ERROR": 500,
    "BAD_GATEWAY": 502,
    "SERVICE_UNAVAILABLE": 503,
}

# =============================================================================
# Database Constants
# =============================================================================

class DatabaseDriver(str, Enum):
    """Supported database drivers."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"


# =============================================================================
# Time Series Constants
# =============================================================================

# Standard forecast horizons in months
FORECAST_HORIZONS: Final = {
    "SHORT_TERM": [1, 3, 6],
    "MEDIUM_TERM": [6, 12, 18],
    "LONG_TERM": [12, 24, 36, 48],
    "ALL": [1, 3, 6, 12, 18, 24, 36, 48]
}

# Minimum data requirements
MIN_DATA_REQUIREMENTS: Final = {
    "MINIMAL": 6,      # Absolute minimum for testing
    "BASIC": 12,       # Basic forecasting
    "RECOMMENDED": 24, # Recommended minimum
    "OPTIMAL": 36,     # Optimal for accuracy
}

# Confidence levels
CONFIDENCE_LEVELS: Final = {
    "BASIC": [0.5, 0.8, 0.9],
    "EXTENDED": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    "STANDARD": [0.5, 0.8, 0.9, 0.95],
}

# =============================================================================
# Real Estate Constants
# =============================================================================

class PropertyType(str, Enum):
    """Property types for real estate data."""
    SINGLE_FAMILY = "single_family"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    MULTI_FAMILY = "multi_family"
    ALL = "all"


class PriceIndex(str, Enum):
    """Zillow price index types."""
    ZHVI = "zhvi"  # Zillow Home Value Index
    ZORI = "zori"  # Zillow Observed Rent Index
    ZHVI_PER_SQFT = "zhvi_per_sqft"


# US Census regions for geographic grouping
US_REGIONS: Final = {
    "NORTHEAST": [
        "Connecticut", "Maine", "Massachusetts", "New Hampshire",
        "New Jersey", "New York", "Pennsylvania", "Rhode Island", "Vermont"
    ],
    "MIDWEST": [
        "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
        "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"
    ],
    "SOUTH": [
        "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Kentucky",
        "Louisiana", "Maryland", "Mississippi", "North Carolina", "Oklahoma",
        "South Carolina", "Tennessee", "Texas", "Virginia", "West Virginia"
    ],
    "WEST": [
        "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho",
        "Montana", "Nevada", "New Mexico", "Oregon", "Utah", "Washington", "Wyoming"
    ]
}

# =============================================================================
# Monitoring Constants
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(str, Enum):
    """Alert levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# File System Constants
# =============================================================================

# Standard file extensions
FILE_EXTENSIONS: Final = {
    "DATA": [".csv", ".parquet", ".feather", ".json"],
    "MODEL": [".pkl", ".joblib", ".pt", ".pth"],
    "CONFIG": [".yaml", ".yml", ".json", ".toml"],
    "LOG": [".log", ".txt"],
}

# Default file permissions (octal)
FILE_PERMISSIONS: Final = {
    "READ_ONLY": 0o444,
    "READ_WRITE": 0o644,
    "EXECUTABLE": 0o755,
}

# =============================================================================
# Error Messages
# =============================================================================

ERROR_MESSAGES: Final = {
    "INVALID_ZIP_CODE": "Invalid ZIP code format. Expected format: 12345 or 12345-6789",
    "INSUFFICIENT_DATA": "Insufficient historical data for reliable forecasting",
    "MODEL_NOT_FOUND": "Specified model not found or not supported",
    "DATA_QUALITY_ISSUE": "Data quality issues detected. Please check input data",
    "RATE_LIMIT_EXCEEDED": "Rate limit exceeded. Please try again later",
    "UNAUTHORIZED_ACCESS": "Unauthorized access. Valid API key required",
    "INTERNAL_ERROR": "Internal server error. Please contact support",
}

# =============================================================================
# Default Values
# =============================================================================

# Default configuration values
DEFAULTS: Final = {
    "BATCH_SIZE": 32,
    "NUM_SAMPLES": 20,
    "CACHE_TTL_HOURS": 24,
    "REQUEST_TIMEOUT_SECONDS": 300,
    "MAX_RETRIES": 3,
    "RETRY_DELAY_SECONDS": 1,
}

# =============================================================================
# Validation Patterns
# =============================================================================

import re

# Regular expression patterns
REGEX_PATTERNS: Final = {
    "ZIP_CODE": re.compile(r"^\d{5}(-\d{4})?$"),
    "US_STATE": re.compile(r"^[A-Z]{2}$"),
    "EMAIL": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    "API_KEY": re.compile(r"^[A-Za-z0-9_-]{32,}$"),
    "SEMANTIC_VERSION": re.compile(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$"),
}

# =============================================================================
# Performance Thresholds
# =============================================================================

PERFORMANCE_THRESHOLDS: Final = {
    "SLOW_QUERY_SECONDS": 1.0,
    "MEMORY_WARNING_MB": 1000,
    "MEMORY_CRITICAL_MB": 2000,
    "CPU_WARNING_PERCENT": 80.0,
    "CPU_CRITICAL_PERCENT": 95.0,
    "DISK_WARNING_PERCENT": 85.0,
    "DISK_CRITICAL_PERCENT": 95.0,
} 