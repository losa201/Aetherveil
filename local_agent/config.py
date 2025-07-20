#!/usr/bin/env python3
"""
Configuration Management Module
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class LLMConfig:
    """Local LLM configuration"""
    model_dir: str = "./models"
    model_name: str = "codellama-7b-instruct.Q4_K_M.gguf"
    context_length: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    threads: int = 4
    gpu_layers: int = 0  # For GPU acceleration if available

@dataclass
class PlanningConfig:
    """Planning configuration"""
    cycle_delay: int = 300  # 5 minutes between cycles
    max_tasks_per_cycle: int = 5
    planning_timeout: int = 120  # 2 minutes for planning
    adaptive_planning: bool = True

@dataclass
class ExecutionConfig:
    """Execution configuration"""
    task_timeout: int = 300  # 5 minutes per task
    task_delay: int = 30    # 30 seconds between tasks
    max_parallel_tasks: int = 1  # Conservative for resource usage
    default_target: str = "127.0.0.1"  # Localhost for testing
    cleanup_temp_files: bool = True
    
    # Safety limits
    max_scan_rate: int = 100  # Max packets/second for network scans
    max_request_rate: int = 10  # Max requests/second for web tools

@dataclass
class StorageConfig:
    """Storage configuration"""
    base_dir: str = "./aetherveil_data"
    results_dir: str = "./aetherveil_data/results"
    temp_dir: str = "./aetherveil_data/temp"
    tools_dir: str = "./aetherveil_data/tools"
    reports_dir: str = "./aetherveil_data/reports"
    max_storage_mb: int = 1024  # 1GB max storage

@dataclass
class LearningConfig:
    """Learning configuration"""
    retrain_frequency: int = 10  # Every 10 cycles
    learning_rate: float = 0.01
    min_samples_for_learning: int = 5
    confidence_threshold: float = 0.7
    knowledge_retention_days: int = 30

@dataclass
class ReportingConfig:
    """Reporting configuration"""
    detailed_frequency: int = 5  # Every 5 cycles
    summary_frequency: int = 20  # Every 20 cycles
    enable_gcp_sync: bool = False
    
@dataclass
class GCPConfig:
    """GCP integration configuration"""
    enabled: bool = False
    project_id: str = ""
    dataset_id: str = "aetherveil_data"
    bucket_name: str = ""
    service_account_path: str = ""
    
    # BigQuery table names
    results_table: str = "pentesting_results"
    learning_table: str = "learning_data"
    metrics_table: str = "performance_metrics"

@dataclass
class ErrorRecoveryConfig:
    """Error recovery configuration"""
    max_retries: int = 3
    retry_delay: int = 60  # 1 minute
    exponential_backoff: bool = True
    circuit_breaker_threshold: int = 5

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_safe_mode: bool = True
    allowed_targets: list = field(default_factory=lambda: ["127.0.0.1", "localhost"])
    blocked_ports: list = field(default_factory=lambda: [22, 3389])  # SSH, RDP
    max_concurrent_scans: int = 1
    require_target_validation: bool = True

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize with defaults
        self.llm = LLMConfig()
        self.planning = PlanningConfig()
        self.execution = ExecutionConfig()
        self.storage = StorageConfig()
        self.learning = LearningConfig()
        self.reporting = ReportingConfig()
        self.gcp = GCPConfig()
        self.error_recovery = ErrorRecoveryConfig()
        self.security = SecurityConfig()
        
        # Load configuration
        self._load_config()
        self._apply_environment_overrides()
        self._validate_config()
        self._setup_directories()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data:
                    self._update_from_dict(config_data)
                    self.logger.info(f"✅ Configuration loaded from {self.config_path}")
            else:
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
                self._create_default_config()
                
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}, using defaults")
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        try:
            # Update LLM config
            if "llm" in config_data:
                llm_data = config_data["llm"]
                for key, value in llm_data.items():
                    if hasattr(self.llm, key):
                        setattr(self.llm, key, value)
            
            # Update Planning config
            if "planning" in config_data:
                planning_data = config_data["planning"]
                for key, value in planning_data.items():
                    if hasattr(self.planning, key):
                        setattr(self.planning, key, value)
            
            # Update Execution config
            if "execution" in config_data:
                exec_data = config_data["execution"]
                for key, value in exec_data.items():
                    if hasattr(self.execution, key):
                        setattr(self.execution, key, value)
            
            # Update Storage config
            if "storage" in config_data:
                storage_data = config_data["storage"]
                for key, value in storage_data.items():
                    if hasattr(self.storage, key):
                        setattr(self.storage, key, value)
            
            # Update Learning config
            if "learning" in config_data:
                learning_data = config_data["learning"]
                for key, value in learning_data.items():
                    if hasattr(self.learning, key):
                        setattr(self.learning, key, value)
            
            # Update GCP config
            if "gcp" in config_data:
                gcp_data = config_data["gcp"]
                for key, value in gcp_data.items():
                    if hasattr(self.gcp, key):
                        setattr(self.gcp, key, value)
            
            # Update Security config
            if "security" in config_data:
                security_data = config_data["security"]
                for key, value in security_data.items():
                    if hasattr(self.security, key):
                        setattr(self.security, key, value)
            
        except Exception as e:
            self.logger.error(f"Failed to update config from dict: {e}")
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            # LLM settings
            "AETHERVEIL_MODEL_DIR": ("llm", "model_dir"),
            "AETHERVEIL_MODEL_NAME": ("llm", "model_name"),
            "AETHERVEIL_GPU_LAYERS": ("llm", "gpu_layers"),
            
            # GCP settings
            "AETHERVEIL_GCP_ENABLED": ("gcp", "enabled"),
            "AETHERVEIL_GCP_PROJECT": ("gcp", "project_id"),
            "AETHERVEIL_GCP_SERVICE_ACCOUNT": ("gcp", "service_account_path"),
            
            # Security settings
            "AETHERVEIL_SAFE_MODE": ("security", "enable_safe_mode"),
            "AETHERVEIL_DEFAULT_TARGET": ("execution", "default_target"),
            
            # Storage settings
            "AETHERVEIL_DATA_DIR": ("storage", "base_dir"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                
                # Type conversion
                if hasattr(section_obj, key):
                    current_value = getattr(section_obj, key)
                    if isinstance(current_value, bool):
                        value = value.lower() in ("true", "1", "yes", "on")
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    
                    setattr(section_obj, key, value)
                    self.logger.debug(f"Applied env override: {env_var} = {value}")
    
    def _validate_config(self):
        """Validate configuration values"""
        issues = []
        
        # Validate LLM config
        if self.llm.context_length < 512:
            issues.append("LLM context_length too small (minimum 512)")
        
        if self.llm.temperature < 0 or self.llm.temperature > 2:
            issues.append("LLM temperature should be between 0 and 2")
        
        # Validate execution config
        if self.execution.task_timeout < 30:
            issues.append("Task timeout too small (minimum 30 seconds)")
        
        if self.execution.max_parallel_tasks > 5:
            issues.append("Max parallel tasks too high (recommended: 1-5)")
        
        # Validate storage config
        if self.storage.max_storage_mb < 100:
            issues.append("Max storage too small (minimum 100MB)")
        
        # Validate GCP config
        if self.gcp.enabled:
            if not self.gcp.project_id:
                issues.append("GCP project_id required when GCP is enabled")
            if not self.gcp.dataset_id:
                issues.append("GCP dataset_id required when GCP is enabled")
        
        # Log issues
        if issues:
            for issue in issues:
                self.logger.warning(f"Config validation: {issue}")
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.storage.base_dir,
            self.storage.results_dir,
            self.storage.temp_dir,
            self.storage.tools_dir,
            self.storage.reports_dir,
            self.llm.model_dir
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
    
    def _create_default_config(self):
        """Create default configuration file"""
        try:
            default_config = {
                "llm": {
                    "model_dir": "./models",
                    "model_name": "codellama-7b-instruct.Q4_K_M.gguf",
                    "context_length": 4096,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "threads": 4,
                    "gpu_layers": 0
                },
                "planning": {
                    "cycle_delay": 300,
                    "max_tasks_per_cycle": 5,
                    "planning_timeout": 120,
                    "adaptive_planning": True
                },
                "execution": {
                    "task_timeout": 300,
                    "task_delay": 30,
                    "max_parallel_tasks": 1,
                    "default_target": "127.0.0.1",
                    "cleanup_temp_files": True,
                    "max_scan_rate": 100,
                    "max_request_rate": 10
                },
                "storage": {
                    "base_dir": "./aetherveil_data",
                    "max_storage_mb": 1024
                },
                "learning": {
                    "retrain_frequency": 10,
                    "learning_rate": 0.01,
                    "min_samples_for_learning": 5,
                    "confidence_threshold": 0.7,
                    "knowledge_retention_days": 30
                },
                "gcp": {
                    "enabled": False,
                    "project_id": "",
                    "dataset_id": "aetherveil_data",
                    "bucket_name": "",
                    "service_account_path": ""
                },
                "security": {
                    "enable_safe_mode": True,
                    "allowed_targets": ["127.0.0.1", "localhost"],
                    "blocked_ports": [22, 3389],
                    "max_concurrent_scans": 1,
                    "require_target_validation": True
                }
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"✅ Created default configuration file: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create default config: {e}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "llm": {
                "model_dir": self.llm.model_dir,
                "model_name": self.llm.model_name,
                "context_length": self.llm.context_length,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "threads": self.llm.threads,
                "gpu_layers": self.llm.gpu_layers
            },
            "planning": {
                "cycle_delay": self.planning.cycle_delay,
                "max_tasks_per_cycle": self.planning.max_tasks_per_cycle,
                "planning_timeout": self.planning.planning_timeout,
                "adaptive_planning": self.planning.adaptive_planning
            },
            "execution": {
                "task_timeout": self.execution.task_timeout,
                "task_delay": self.execution.task_delay,
                "max_parallel_tasks": self.execution.max_parallel_tasks,
                "default_target": self.execution.default_target,
                "cleanup_temp_files": self.execution.cleanup_temp_files
            },
            "gcp": {
                "enabled": self.gcp.enabled,
                "project_id": self.gcp.project_id,
                "dataset_id": self.gcp.dataset_id
            },
            "security": {
                "enable_safe_mode": self.security.enable_safe_mode,
                "allowed_targets": self.security.allowed_targets,
                "blocked_ports": self.security.blocked_ports
            }
        }
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file"""
        try:
            save_path = path or self.config_path
            config_dict = self.get_config_dict()
            
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"✅ Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def update_setting(self, section: str, key: str, value: Any):
        """Update a specific configuration setting"""
        try:
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    self.logger.info(f"Updated {section}.{key} = {value}")
                else:
                    self.logger.warning(f"Unknown setting: {section}.{key}")
            else:
                self.logger.warning(f"Unknown section: {section}")
                
        except Exception as e:
            self.logger.error(f"Failed to update setting: {e}")
    
    def is_target_allowed(self, target: str) -> bool:
        """Check if target is allowed by security policy"""
        if not self.security.enable_safe_mode:
            return True
        
        # Check allowed targets
        if target in self.security.allowed_targets:
            return True
        
        # Check if target is localhost variants
        localhost_variants = ["127.0.0.1", "localhost", "::1", "0.0.0.0"]
        if target in localhost_variants:
            return True
        
        return False
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits for safe operation"""
        return {
            "max_parallel_tasks": self.execution.max_parallel_tasks,
            "task_timeout": self.execution.task_timeout,
            "max_scan_rate": self.execution.max_scan_rate,
            "max_request_rate": self.execution.max_request_rate,
            "max_storage_mb": self.storage.max_storage_mb,
            "max_concurrent_scans": self.security.max_concurrent_scans
        }