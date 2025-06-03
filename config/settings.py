"""
Application Settings and Configuration Models

This module defines the complete configuration schema for the Home Price Prediction system
using Pydantic v2 for validation and type safety.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"  
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelConfig(BaseModel):
    """Model-specific configuration."""
    
    # Core model settings
    name: str = Field(
        default="amazon/chronos-t5-small",
        min_length=1,
        description="HuggingFace model name or path"
    )
    device: str = Field(
        default="auto",
        pattern="^(auto|cpu|cuda|mps)$",
        description="Device for model inference: 'auto', 'cpu', 'cuda', 'mps'"
    )
    torch_dtype: str = Field(
        default="float32",
        pattern="^(float16|bfloat16|float32)$",
        description="PyTorch data type: 'float16', 'bfloat16', 'float32'"
    )
    
    # Forecasting parameters
    forecast_horizons_months: List[int] = Field(
        default=[3, 6, 12, 24],
        description="Forecast horizons in months"
    )
    max_history_months: int = Field(
        default=120,
        description="Maximum historical data to use (months)"
    )
    min_history_months: int = Field(
        default=24,
        description="Minimum historical data required (months)"
    )
    
    # Model performance settings
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for inference"
    )
    num_samples: int = Field(
        default=20,
        ge=1,
        description="Number of sample paths for probabilistic forecasting"
    )
    confidence_levels: List[float] = Field(
        default=[0.5, 0.8, 0.9],
        description="Confidence levels for prediction intervals"
    )
    
    # Caching and storage
    cache_predictions: bool = Field(
        default=True,
        description="Whether to cache model predictions"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        description="Cache time-to-live in hours"
    )
    
    @field_validator('confidence_levels')
    @classmethod
    def validate_confidence_levels(cls, v: List[float]) -> List[float]:
        """Validate confidence levels are between 0 and 1."""
        for level in v:
            if not 0 < level < 1:
                raise ValueError(f"Confidence level {level} must be between 0 and 1")
        return sorted(v)


class DataConfig(BaseModel):
    """Data processing and validation configuration."""
    
    # Data sources
    zhvi_data_url: str = Field(
        default="https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        description="URL for Zillow ZHVI data"
    )
    raw_data_file: str = Field(
        default="data/raw/zhvi_zip.csv",
        description="Path to raw ZHVI data file"
    )
    
    # Data validation
    min_data_points_required: int = Field(
        default=24,
        ge=12,
        description="Minimum data points required for forecasting"
    )
    max_missing_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Maximum ratio of missing values allowed"
    )
    outlier_detection_method: str = Field(
        default="iqr",
        pattern="^(iqr|zscore|isolation_forest)$",
        description="Outlier detection method: 'iqr', 'zscore', 'isolation_forest'"
    )
    outlier_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="Threshold for outlier detection"
    )
    
    # Data processing
    seasonal_adjustment: bool = Field(
        default=True,
        description="Apply seasonal adjustment to time series"
    )
    trend_removal: bool = Field(
        default=False,
        description="Remove trend component before forecasting"
    )
    normalization_method: str = Field(
        default="min_max",
        pattern="^(min_max|z_score|robust)$",
        description="Normalization method: 'min_max', 'z_score', 'robust'"
    )
    
    # Data storage
    processed_data_format: str = Field(
        default="parquet",
        pattern="^(parquet|csv|feather)$",
        description="Format for processed data: 'parquet', 'csv', 'feather'"
    )
    compression: str = Field(
        default="snappy",
        pattern="^(none|gzip|snappy|lz4|brotli)$",
        description="Compression method for data storage"
    )


class APIConfig(BaseModel):
    """API server configuration."""
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        min_length=1,
        description="API server host"
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port"
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of worker processes"
    )
    
    # Request handling
    max_request_size: int = Field(
        default=16 * 1024 * 1024,  # 16MB
        description="Maximum request size in bytes"
    )
    request_timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Request timeout in seconds"
    )
    rate_limit_requests_per_minute: int = Field(
        default=100,
        ge=1,
        description="Rate limit: requests per minute per IP"
    )
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    
    # API documentation
    title: str = Field(
        default="Home Price Prediction API",
        min_length=1,
        max_length=100,
        description="API title"
    )
    description: str = Field(
        default="Advanced time series forecasting for residential real estate prices",
        min_length=1,
        max_length=500,
        description="API description"
    )
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$",
        description="API version"
    )
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host format."""
        import ipaddress
        import re
        
        # Check if it's a valid IP address
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            pass
        
        # Check if it's a valid hostname (simple regex)
        hostname_pattern = re.compile(
            r'^([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        )
        
        if v in ["localhost", "0.0.0.0"] or hostname_pattern.match(v):
            return v
        
        raise ValueError(f"Invalid host format: {v}")
    
    @field_validator('cors_origins')
    @classmethod
    def validate_cors_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins."""
        import re
        
        url_pattern = re.compile(
            r'^(https?://)?([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+(\:[0-9]+)?(/.*)?$|^\*$'
        )
        
        for origin in v:
            if origin != "*" and not url_pattern.match(origin):
                raise ValueError(f"Invalid CORS origin format: {origin}")
        
        return v


class DatabaseConfig(BaseModel):
    """Database configuration (optional)."""
    
    # Connection settings
    url: Optional[str] = Field(
        default=None,
        description="Database connection URL"
    )
    driver: str = Field(
        default="sqlite",
        pattern="^(sqlite|postgresql|mysql|mongodb)$",
        description="Database driver: 'sqlite', 'postgresql', 'mysql', 'mongodb'"
    )
    
    # Connection pool
    pool_size: int = Field(
        default=5,
        ge=1,
        description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=10,
        ge=0,
        description="Maximum overflow connections"
    )
    pool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="Connection pool timeout"
    )


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    # Metrics collection
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=8001,
        ge=1024,
        le=65535,
        description="Metrics server port"
    )
    
    # Health checks
    health_check_interval_seconds: int = Field(
        default=30,
        ge=1,
        description="Health check interval"
    )
    
    # Performance monitoring
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    slow_query_threshold_seconds: float = Field(
        default=1.0,
        ge=0.1,
        description="Threshold for slow query logging"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    # Log levels
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Global log level"
    )
    
    # Log formatting
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for log messages"
    )
    
    # Log outputs
    console_enabled: bool = Field(
        default=True,
        description="Enable console logging"
    )
    file_enabled: bool = Field(
        default=True,
        description="Enable file logging"
    )
    log_file_path: str = Field(
        default="logs/app.log",
        min_length=1,
        description="Log file path"
    )
    
    # Log rotation
    max_log_size_mb: int = Field(
        default=100,
        ge=1,
        description="Maximum log file size in MB"
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        description="Number of log backup files to keep"
    )


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    # API Security
    api_key_required: bool = Field(
        default=False,
        description="Require API key for requests"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key"
    )
    
    # Rate limiting
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    
    # Input validation
    max_zip_codes_per_request: int = Field(
        default=10,
        ge=1,
        description="Maximum ZIP codes per prediction request"
    )
    allowed_zip_code_pattern: str = Field(
        default=r"^\d{5}(-\d{4})?$",
        description="Regex pattern for valid ZIP codes"
    )
    
    @field_validator('allowed_zip_code_pattern')
    @classmethod
    def validate_zip_pattern(cls, v: str) -> str:
        """Validate that the ZIP code pattern is a valid regex."""
        import re
        try:
            re.compile(v)
            return v
        except re.error as e:
            raise ValueError(f"Invalid regex pattern for ZIP codes: {e}")


class PathConfig(BaseModel):
    """File and directory path configuration."""
    
    # Data directories
    data_dir: Path = Field(
        default=Path("data"),
        description="Base data directory"
    )
    raw_data_dir: Path = Field(
        default=Path("data/raw"),
        description="Raw data directory"
    )
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        description="Processed data directory"
    )
    model_cache_dir: Path = Field(
        default=Path("data/model_cache"),
        description="Model cache directory"
    )
    
    # Output directories
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Output directory"
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Log directory"
    )
    temp_dir: Path = Field(
        default=Path("temp"),
        description="Temporary files directory"
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Create directories if they don't exist after model initialization."""
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="HPP_",  # Home Price Predictor prefix
        case_sensitive=False,
        extra="forbid",  # Prevent extra fields for security
        validate_assignment=True,  # Validate on assignment
    )
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    paths: PathConfig = Field(default_factory=PathConfig)


def load_config(
    environment: Optional[Environment] = None,
    config_path: Optional[Path] = None
) -> AppConfig:
    """
    Load configuration from YAML file and environment variables.
    
    Args:
        environment: Target environment (defaults to development)
        config_path: Custom path to config file
        
    Returns:
        AppConfig: Loaded and validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration validation fails
    """
    try:
        # Determine environment
        if environment is None:
            env_str = os.getenv("HPP_ENVIRONMENT", "development").lower()
            try:
                environment = Environment(env_str)
            except ValueError:
                # Default to development if invalid environment specified
                environment = Environment.DEVELOPMENT
        
        # Determine config file path
        if config_path is None:
            config_dir = Path(__file__).parent
            config_path = config_dir / f"{environment.value}.yaml"
            
            # Fallback to default config if environment-specific doesn't exist
            if not config_path.exists():
                config_path = config_dir / "default.yaml"
        
        # Ensure config file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load base configuration
        config_data: Dict[str, Any] = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)
                config_data = loaded_data or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read configuration file {config_path}: {e}")
        
        # Set environment in config data
        config_data['environment'] = environment.value
        
        # Create and return config instance
        try:
            return AppConfig(**config_data)
        except Exception as e:
            raise ValueError(f"Failed to create configuration from {config_path}: {e}")
            
    except Exception as e:
        # Re-raise with more context if not already a specific exception
        if isinstance(e, (FileNotFoundError, yaml.YAMLError, ValueError)):
            raise
        else:
            raise ValueError(f"Unexpected error loading configuration: {e}")


def validate_config(config: AppConfig) -> None:
    """
    Perform additional configuration validation.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    
    # Validate forecast horizons
    if not config.model.forecast_horizons_months:
        errors.append("At least one forecast horizon must be specified")
    
    # Check for duplicate forecast horizons
    if len(config.model.forecast_horizons_months) != len(set(config.model.forecast_horizons_months)):
        errors.append("Forecast horizons must be unique")
    
    # Validate data requirements
    if config.model.min_history_months > config.model.max_history_months:
        errors.append(
            f"min_history_months ({config.model.min_history_months}) "
            f"cannot be greater than max_history_months "
            f"({config.model.max_history_months})"
        )
    
    # Validate API configuration
    if config.api.port == config.monitoring.metrics_port:
        errors.append("API port and metrics port cannot be the same")
    
    # Validate URL format for data source
    if not config.data.zhvi_data_url.startswith(("http://", "https://")):
        errors.append("ZHVI data URL must be a valid HTTP/HTTPS URL")
    
    # Environment-specific validations
    if config.environment == Environment.PRODUCTION:
        if config.debug:
            errors.append("Debug mode should not be enabled in production")
        if not config.security.enable_rate_limiting:
            errors.append("Rate limiting should be enabled in production")
        if "*" in config.api.cors_origins:
            errors.append("Wildcard CORS origins should not be used in production")
    
    # Raise consolidated error if any validation failed
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_message) 