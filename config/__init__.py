"""
Configuration Management Module

This module provides centralized configuration management for the Home Price Prediction system.
It supports multiple environments (development, staging, production) with proper validation
and type safety using Pydantic.

Example:
    >>> from config import get_config
    >>> config = get_config()
    >>> print(config.model.name)
"""

from pathlib import Path
from typing import Optional

from .settings import (
    AppConfig,
    Environment,
    load_config,
    validate_config,
)

__all__ = [
    "get_config",
    "AppConfig", 
    "Environment",
    "CONFIG_DIR",
    "ROOT_DIR",
]

# Path constants
ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent

# Global config instance (lazy loaded)
_config_instance: Optional[AppConfig] = None


def get_config(
    environment: Optional[Environment] = None,
    config_path: Optional[Path] = None,
    force_reload: bool = False
) -> AppConfig:
    """
    Get the application configuration instance.
    
    This function implements a singleton pattern for config loading with support
    for different environments and config file paths.
    
    Args:
        environment: Target environment (dev/staging/prod). If None, uses ENVIRONMENT env var
        config_path: Custom path to config file. If None, uses default environment config
        force_reload: Force reload config even if already loaded
        
    Returns:
        AppConfig: Validated configuration instance
        
    Raises:
        ValidationError: If configuration validation fails
        FileNotFoundError: If config file doesn't exist
    """
    global _config_instance
    
    if _config_instance is None or force_reload:
        _config_instance = load_config(
            environment=environment,
            config_path=config_path
        )
        validate_config(_config_instance)
    
    return _config_instance


def reload_config() -> AppConfig:
    """
    Force reload the configuration.
    
    Returns:
        AppConfig: Newly loaded configuration instance
    """
    return get_config(force_reload=True) 