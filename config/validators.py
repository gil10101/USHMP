"""
Configuration Validation Utilities

This module provides validation functions and utilities for the configuration system.
It includes data science-specific validation logic, system resource checks, and
comprehensive error reporting for configuration issues.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
from pydantic import ValidationError

from .constants import (
    MIN_DATA_REQUIREMENTS,
    REGEX_PATTERNS,
)
from .settings import AppConfig, Environment

# Optional torch import for GPU validation
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


def validate_system_requirements(config: AppConfig) -> List[str]:
    """
    Validate system requirements for the given configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        List of validation warnings (empty if all requirements met)
    """
    warnings = []
    
    # Check available memory
    memory_info = psutil.virtual_memory()
    available_memory_gb = memory_info.available / (1024**3)
    
    if config.model.batch_size > 32 and available_memory_gb < 8:
        warnings.append(
            f"Large batch size ({config.model.batch_size}) with limited memory "
            f"({available_memory_gb:.1f}GB available). Consider reducing batch size."
        )
    
    # Check GPU availability if CUDA requested (only if torch is available)
    if HAS_TORCH and config.model.device in ["cuda", "auto"]:
        if not torch.cuda.is_available():
            warnings.append(
                "CUDA requested but not available. Model will fall back to CPU."
            )
        elif torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory < 4 and config.model.batch_size > 16:
                warnings.append(
                    f"Limited GPU memory ({gpu_memory:.1f}GB) with large batch size. "
                    "Consider reducing batch size or using CPU."
                )
    
    # Check MPS availability for Apple Silicon (only if torch is available)
    if HAS_TORCH and config.model.device == "mps":
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            warnings.append(
                "MPS requested but not available. Model will fall back to CPU."
            )
    
    # Check disk space for data and models
    data_path = Path(config.paths.data_dir)
    if data_path.exists():
        disk_usage = psutil.disk_usage(str(data_path))
        free_space_gb = disk_usage.free / (1024**3)
        
        if free_space_gb < 10:
            warnings.append(
                f"Limited disk space ({free_space_gb:.1f}GB free). "
                "Consider cleaning up data directory."
            )
    
    return warnings


def validate_model_configuration(config: AppConfig) -> List[str]:
    """
    Validate model-specific configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Validate forecast horizons
    if not config.model.forecast_horizons_months:
        errors.append("At least one forecast horizon must be specified")
    
    for horizon in config.model.forecast_horizons_months:
        if horizon <= 0:
            errors.append(f"Invalid forecast horizon: {horizon}. Must be positive.")
        if horizon > 60:
            errors.append(f"Very long forecast horizon: {horizon} months. Consider shorter horizons for better accuracy.")
    
    # Validate history requirements
    if config.model.min_history_months > config.model.max_history_months:
        errors.append(
            f"min_history_months ({config.model.min_history_months}) "
            f"cannot exceed max_history_months ({config.model.max_history_months})"
        )
    
    # Validate confidence levels
    for level in config.model.confidence_levels:
        if not 0 < level < 1:
            errors.append(f"Invalid confidence level: {level}. Must be between 0 and 1.")
    
    # Validate performance settings
    if config.model.batch_size <= 0:
        errors.append(f"Invalid batch size: {config.model.batch_size}. Must be positive.")
    
    if config.model.num_samples <= 0:
        errors.append(f"Invalid num_samples: {config.model.num_samples}. Must be positive.")
    
    return errors


def validate_data_configuration(config: AppConfig) -> List[str]:
    """
    Validate data processing configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Validate data quality thresholds
    if not 0 <= config.data.max_missing_ratio <= 1:
        errors.append(
            f"Invalid max_missing_ratio: {config.data.max_missing_ratio}. "
            "Must be between 0 and 1."
        )
    
    if config.data.outlier_threshold <= 0:
        errors.append(
            f"Invalid outlier_threshold: {config.data.outlier_threshold}. "
            "Must be positive."
        )
    
    # Validate minimum data requirements
    if config.data.min_data_points_required < MIN_DATA_REQUIREMENTS["MINIMAL"]:
        errors.append(
            f"min_data_points_required ({config.data.min_data_points_required}) "
            f"is below minimum safe threshold ({MIN_DATA_REQUIREMENTS['MINIMAL']})."
        )
    
    # Validate data URL format
    if not config.data.zhvi_data_url.startswith(("http://", "https://")):
        errors.append("zhvi_data_url must be a valid HTTP/HTTPS URL.")
    
    return errors


def validate_api_configuration(config: AppConfig) -> List[str]:
    """
    Validate API configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Validate port numbers
    if not 1024 <= config.api.port <= 65535:
        errors.append(f"Invalid API port: {config.api.port}. Must be between 1024-65535.")
    
    if not 1024 <= config.monitoring.metrics_port <= 65535:
        errors.append(f"Invalid metrics port: {config.monitoring.metrics_port}. Must be between 1024-65535.")
    
    if config.api.port == config.monitoring.metrics_port:
        errors.append("API port and metrics port must be different.")
    
    # Validate worker count
    if config.api.workers <= 0:
        errors.append(f"Invalid worker count: {config.api.workers}. Must be positive.")
    
    cpu_count = os.cpu_count() or 4
    if config.api.workers > cpu_count * 2:
        errors.append(
            f"Worker count ({config.api.workers}) is very high for system "
            f"with {cpu_count} CPUs. Consider reducing for better performance."
        )
    
    # Validate timeout settings
    if config.api.request_timeout_seconds <= 0:
        errors.append("Request timeout must be positive.")
    
    if config.api.rate_limit_requests_per_minute <= 0:
        errors.append("Rate limit must be positive.")
    
    return errors


def validate_security_configuration(config: AppConfig) -> List[str]:
    """
    Validate security configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Production security checks
    if config.environment == Environment.PRODUCTION:
        if not config.security.api_key_required:
            errors.append("API key should be required in production environment.")
        
        if not config.security.enable_rate_limiting:
            errors.append("Rate limiting should be enabled in production environment.")
        
        if config.debug:
            errors.append("Debug mode should be disabled in production environment.")
        
        if "*" in config.api.cors_origins:
            errors.append("Wildcard CORS origins should not be used in production.")
    
    # Validate ZIP code pattern
    try:
        re.compile(config.security.allowed_zip_code_pattern)
    except re.error as e:
        errors.append(f"Invalid ZIP code regex pattern: {e}")
    
    # Validate request limits
    if config.security.max_zip_codes_per_request <= 0:
        errors.append("max_zip_codes_per_request must be positive.")
    
    return errors


def validate_paths_configuration(config: AppConfig) -> List[str]:
    """
    Validate path configuration and create directories if needed.
    
    Args:
        config: Application configuration
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check if paths are absolute in production
    if config.environment == Environment.PRODUCTION:
        paths_to_check = [
            config.paths.data_dir,
            config.paths.output_dir,
            config.paths.log_dir,
        ]
        
        for path in paths_to_check:
            if not path.is_absolute():
                errors.append(f"Path should be absolute in production: {path}")
    
    # Try to create directories
    critical_dirs = [
        config.paths.data_dir,
        config.paths.raw_data_dir,
        config.paths.processed_data_dir,
        config.paths.log_dir,
    ]
    
    for directory in critical_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            errors.append(f"Permission denied creating directory: {directory}")
        except OSError as e:
            errors.append(f"Error creating directory {directory}: {e}")
    
    return errors


def validate_environment_consistency(config: AppConfig) -> List[str]:
    """
    Validate environment-specific configuration consistency.
    
    Args:
        config: Application configuration
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Development environment checks
    if config.environment == Environment.DEVELOPMENT:
        if config.api.workers > 2:
            errors.append("Consider using fewer workers in development environment.")
    
    # Testing environment checks
    elif config.environment == Environment.TESTING:
        if config.model.cache_predictions:
            errors.append("Caching should be disabled in testing environment for isolation.")
        
        if config.monitoring.enable_metrics:
            errors.append("Metrics collection should be disabled in testing environment.")
    
    # Production environment checks
    elif config.environment == Environment.PRODUCTION:
        if HAS_TORCH and config.model.device == "cpu" and torch.cuda.is_available():
            errors.append("Consider using GPU acceleration in production if available.")
        
        if config.logging.level.value == "DEBUG":
            errors.append("Debug logging should not be used in production.")
    
    return errors


def validate_resource_limits(config: AppConfig) -> List[str]:
    """
    Validate configuration against system resource limits.
    
    Args:
        config: Application configuration
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Memory usage estimation
    estimated_memory_mb = (
        config.model.batch_size * 
        config.model.num_samples * 
        max(config.model.forecast_horizons_months) * 
        0.1  # Rough estimation factor in MB per parameter
    )
    
    available_memory_mb = psutil.virtual_memory().available / (1024**2)
    
    if estimated_memory_mb > available_memory_mb * 0.8:
        warnings.append(
            f"Configuration may require {estimated_memory_mb:.0f}MB memory, "
            f"but only {available_memory_mb:.0f}MB available."
        )
    
    # CPU usage estimation
    cpu_count = os.cpu_count() or 4
    if config.api.workers > cpu_count:
        warnings.append(
            f"Worker count ({config.api.workers}) exceeds CPU count ({cpu_count}). "
            "This may cause context switching overhead."
        )
    
    return warnings


def run_comprehensive_validation(config: AppConfig) -> Tuple[List[str], List[str]]:
    """
    Run comprehensive validation on the configuration.
    
    Args:
        config: Application configuration to validate
        
    Returns:
        Tuple of (errors, warnings) lists
    """
    errors = []
    warnings = []
    
    # Run all validation functions
    validation_functions = [
        validate_model_configuration,
        validate_data_configuration,
        validate_api_configuration,
        validate_security_configuration,
        validate_paths_configuration,
        validate_environment_consistency,
    ]
    
    for validate_func in validation_functions:
        try:
            errors.extend(validate_func(config))
        except Exception as e:
            errors.append(f"Validation error in {validate_func.__name__}: {str(e)}")
    
    # Run warning functions
    warning_functions = [
        validate_system_requirements,
        validate_resource_limits,
    ]
    
    for warning_func in warning_functions:
        try:
            warnings.extend(warning_func(config))
        except Exception as e:
            warnings.append(f"Warning check error in {warning_func.__name__}: {str(e)}")
    
    return errors, warnings


def validate_zip_code(zip_code: str) -> bool:
    """
    Validate ZIP code format.
    
    Args:
        zip_code: ZIP code string to validate
        
    Returns:
        True if valid, False otherwise
    """
    return bool(REGEX_PATTERNS["ZIP_CODE"].match(zip_code))


def validate_forecast_horizons(horizons: List[int]) -> List[str]:
    """
    Validate forecast horizons.
    
    Args:
        horizons: List of forecast horizons in months
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if not horizons:
        errors.append("At least one forecast horizon must be specified")
        return errors
    
    for horizon in horizons:
        if not isinstance(horizon, int):
            errors.append(f"Forecast horizon must be integer, got {type(horizon).__name__}")
        elif horizon <= 0:
            errors.append(f"Forecast horizon must be positive, got {horizon}")
        elif horizon > 60:
            errors.append(f"Forecast horizon {horizon} months is very long, consider shorter horizons")
    
    return errors


def suggest_optimal_configuration(config: AppConfig) -> Dict[str, Any]:
    """
    Suggest optimal configuration based on system resources and environment.
    
    Args:
        config: Current configuration
        
    Returns:
        Dictionary of suggested configuration changes
    """
    suggestions = {}
    
    # System info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count() or 4
    has_gpu = HAS_TORCH and torch.cuda.is_available()
    
    # Model suggestions
    if memory_gb >= 16 and has_gpu:
        suggestions["model.batch_size"] = 64
        suggestions["model.device"] = "cuda"
        suggestions["model.torch_dtype"] = "float16"
    elif memory_gb >= 8:
        suggestions["model.batch_size"] = 32
        suggestions["model.device"] = "cpu"
    else:
        suggestions["model.batch_size"] = 16
        suggestions["model.device"] = "cpu"
    
    # API suggestions
    if config.environment == Environment.PRODUCTION:
        suggestions["api.workers"] = min(cpu_count, 4)
    else:
        suggestions["api.workers"] = 1
    
    # Environment-specific suggestions
    if config.environment == Environment.DEVELOPMENT:
        suggestions["logging.level"] = "DEBUG"
        suggestions["model.cache_ttl_hours"] = 1
    elif config.environment == Environment.PRODUCTION:
        suggestions["logging.level"] = "INFO"
        suggestions["security.api_key_required"] = True
        suggestions["security.enable_rate_limiting"] = True
    
    return suggestions


def validate_pydantic_config(config: AppConfig) -> List[str]:
    """
    Validate using Pydantic's built-in validation.
    
    Args:
        config: Application configuration to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    try:
        # Re-validate the config to catch any validation errors
        config.model_validate(config.model_dump())
    except ValidationError as e:
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"Validation error in {field_path}: {error['msg']}")
    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
    
    return errors


def validate_configuration_complete(config: AppConfig) -> Tuple[bool, List[str], List[str]]:
    """
    Complete configuration validation with all checks.
    
    Args:
        config: Application configuration to validate
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors, warnings = run_comprehensive_validation(config)
    
    # Add Pydantic validation errors
    pydantic_errors = validate_pydantic_config(config)
    errors.extend(pydantic_errors)
    
    is_valid = len(errors) == 0
    
    return is_valid, errors, warnings 