"""
Configuration package for Aetherveil Sentinel

This package provides comprehensive configuration management with:
- Pydantic validation for all configuration values
- Environment variable support
- Multiple environment configurations (development, staging, production)
- GCP Secret Manager integration
- Security-focused defaults
- Comprehensive logging configuration
"""

from .config import (
    AetherVeilConfig,
    DatabaseConfig,
    SecurityConfig,
    NetworkConfig,
    GCPConfig,
    AgentConfig,
    RLConfig,
    StealthConfig,
    LoggingConfig,
    config,
    get_config,
    get_secret,
    validate_config,
    setup_logging
)

__all__ = [
    'AetherVeilConfig',
    'DatabaseConfig',
    'SecurityConfig',
    'NetworkConfig',
    'GCPConfig',
    'AgentConfig',
    'RLConfig',
    'StealthConfig',
    'LoggingConfig',
    'config',
    'get_config',
    'get_secret',
    'validate_config',
    'setup_logging'
]