"""
Configuration management for Chimera
"""

import os
import yaml
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration from YAML files and environment variables
    Supports nested configuration access and environment variable overrides
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.env_prefix = "CHIMERA_"
        
    async def load(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                self.config = self._get_default_config()
            else:
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                    
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._get_default_config()
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get("persona.default", "balanced")
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set the value
        config[keys[-1]] = value
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.get(section, {})
        
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        
        # Mapping of environment variables to config keys
        env_mappings = {
            f"{self.env_prefix}DEBUG": "core.debug",
            f"{self.env_prefix}LOG_LEVEL": "logging.level",
            f"{self.env_prefix}DATA_DIR": "core.data_dir",
            f"{self.env_prefix}CONFIG_FILE": "core.config_file",
            "HTTP_PROXY": "network.http_proxy",
            "HTTPS_PROXY": "network.https_proxy",
            "SOCKS_PROXY": "network.socks_proxy",
            "TOR_PROXY": "network.tor_proxy",
            "OPENAI_API_KEY": "llm.api_keys.openai",
            "ANTHROPIC_API_KEY": "llm.api_keys.anthropic",
            "GOOGLE_API_KEY": "llm.api_keys.google",
            "DATABASE_URL": "memory.database_url",
            "MAX_THREADS": "core.max_concurrent_tasks",
            "REQUEST_DELAY_MIN": "web.min_delay",
            "REQUEST_DELAY_MAX": "web.max_delay",
            "BROWSER_HEADLESS": "web.headless",
            "VALIDATION_ENABLED": "validator.enabled",
            "NEUROPLASTICITY_ENABLED": "reasoner.neuroplasticity_enabled",
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value)
                self.set(config_key, converted_value)
                
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        
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
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        
        return {
            "core": {
                "max_concurrent_tasks": 4,
                "task_timeout": 3600,
                "auto_save_interval": 300,
                "event_queue_size": 1000,
                "debug": False,
                "data_dir": "./data"
            },
            "persona": {
                "default": "balanced",
                "available_personas": ["cautious", "balanced", "aggressive", "creative", "stealth_focused"],
                "persona_config_dir": "./configs/personas/"
            },
            "reasoner": {
                "decision_confidence_threshold": 0.7,
                "max_reasoning_depth": 5,
                "knowledge_weight_decay": 0.95,
                "learning_rate": 0.1,
                "neuroplasticity_enabled": True
            },
            "memory": {
                "graph_database": "./data/knowledge/graph.db",
                "max_nodes": 100000,
                "max_edges": 500000,
                "pruning_threshold": 0.1,
                "backup_interval": 3600,
                "compression_enabled": True
            },
            "web": {
                "search_engines": ["google", "bing", "duckduckgo"],
                "max_search_results": 50,
                "scraping_delay_range": [2, 8],
                "user_agent_rotation": True,
                "proxy_rotation": True,
                "javascript_enabled": True,
                "headless": True,
                "min_delay": 1.0,
                "max_delay": 5.0
            },
            "llm": {
                "providers": ["claude_web", "chatgpt_web", "gemini_web"],
                "fallback_enabled": True,
                "response_validation": True,
                "confidence_threshold": 0.8,
                "rate_limiting": True,
                "api_keys": {}
            },
            "planner": {
                "max_plan_complexity": 10,
                "optimization_enabled": True,
                "risk_assessment": True,
                "timeline_estimation": True,
                "resource_allocation": True
            },
            "executor": {
                "stealth_mode": True,
                "traffic_mixing": True,
                "decoy_requests": True,
                "ip_rotation": True,
                "timing_randomization": True,
                "tools": {
                    "nmap": True,
                    "gobuster": True,
                    "sqlmap": False,
                    "nuclei": True,
                    "subfinder": True
                }
            },
            "opsec": {
                "fingerprint_randomization": True,
                "cover_traffic_generation": True,
                "polymorphic_behavior": True,
                "detection_evasion": True,
                "attribution_resistance": True
            },
            "validator": {
                "sandbox_enabled": True,
                "safety_checks": True,
                "code_analysis": True,
                "behavior_monitoring": True,
                "rollback_capability": True,
                "enabled": True
            },
            "reporter": {
                "output_formats": ["markdown", "html", "json"],
                "template_dir": "./chimera/reporter/templates/",
                "include_technical_details": True,
                "include_remediation": True,
                "severity_scoring": "cvss"
            },
            "logging": {
                "level": "INFO",
                "file": "./data/logs/chimera.log",
                "rotation": "daily",
                "retention_days": 30,
                "structured_logging": True
            },
            "security": {
                "encryption_at_rest": True,
                "secure_communication": True,
                "credential_protection": True,
                "audit_logging": True
            },
            "network": {
                "http_proxy": None,
                "https_proxy": None,
                "socks_proxy": None,
                "tor_proxy": None
            },
            "limits": {
                "max_memory_usage": "2GB",
                "max_disk_usage": "10GB",
                "max_network_bandwidth": "100MB/s",
                "max_cpu_usage": 80
            }
        }