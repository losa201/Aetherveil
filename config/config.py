"""
Comprehensive configuration management for Aetherveil Sentinel
Production-ready configuration with Pydantic validation and environment variable handling
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from functools import lru_cache
import logging
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, validator, SecretStr, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration with validation"""
    
    # SQLite Configuration
    sqlite_path: str = Field(default="aetherveil.db", description="Path to SQLite database file")
    sqlite_pool_size: int = Field(default=20, ge=1, le=100, description="SQLite connection pool size")
    sqlite_timeout: int = Field(default=30, ge=1, le=300, description="SQLite timeout in seconds")
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j database URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: SecretStr = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    neo4j_pool_size: int = Field(default=50, ge=1, le=200, description="Neo4j connection pool size")
    neo4j_timeout: int = Field(default=30, ge=1, le=300, description="Neo4j timeout in seconds")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    redis_password: Optional[SecretStr] = Field(default=None, description="Redis password")
    redis_pool_size: int = Field(default=10, ge=1, le=100, description="Redis connection pool size")
    redis_timeout: int = Field(default=5, ge=1, le=60, description="Redis timeout in seconds")
    
    # Performance Settings
    connection_retry_count: int = Field(default=3, ge=1, le=10, description="Connection retry attempts")
    connection_retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Retry delay in seconds")
    
    @validator('sqlite_path')
    def validate_sqlite_path(cls, v):
        """Ensure SQLite path is valid"""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @validator('neo4j_uri')
    def validate_neo4j_uri(cls, v):
        """Validate Neo4j URI format"""
        if not v.startswith(('bolt://', 'neo4j://', 'bolt+s://', 'neo4j+s://')):
            raise ValueError('Neo4j URI must start with bolt://, neo4j://, bolt+s://, or neo4j+s://')
        return v


class SecurityConfig(BaseModel):
    """Security configuration with encryption and authentication"""
    
    # Encryption Settings
    encryption_key: SecretStr = Field(default="development_key_32_chars_long_123", description="256-bit encryption key for data at rest")
    jwt_secret: SecretStr = Field(default="development_jwt_secret_key_32_chars", description="JWT signing secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, le=168, description="JWT expiration in hours")
    
    # API Security
    api_key_length: int = Field(default=32, ge=16, le=64, description="API key length")
    api_rate_limit: int = Field(default=100, ge=1, le=10000, description="API rate limit per minute")
    api_burst_limit: int = Field(default=200, ge=1, le=20000, description="API burst limit")
    
    # Password Security
    password_min_length: int = Field(default=12, ge=8, le=128, description="Minimum password length")
    password_require_special: bool = Field(default=True, description="Require special characters in passwords")
    password_require_numbers: bool = Field(default=True, description="Require numbers in passwords")
    password_require_uppercase: bool = Field(default=True, description="Require uppercase letters")
    password_hash_rounds: int = Field(default=12, ge=10, le=16, description="Password hash rounds")
    
    # Session Security
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440, description="Session timeout in minutes")
    max_failed_attempts: int = Field(default=5, ge=1, le=20, description="Max failed login attempts")
    lockout_duration_minutes: int = Field(default=15, ge=1, le=1440, description="Account lockout duration")
    
    # TLS/SSL Configuration
    tls_enabled: bool = Field(default=True, description="Enable TLS/SSL")
    tls_cert_path: Optional[str] = Field(default=None, description="Path to TLS certificate")
    tls_key_path: Optional[str] = Field(default=None, description="Path to TLS private key")
    tls_ca_path: Optional[str] = Field(default=None, description="Path to TLS CA certificate")
    
    # Security Headers
    security_headers: Dict[str, str] = Field(default_factory=lambda: {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    })
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        """Validate encryption key length"""
        if len(v.get_secret_value()) < 32:
            raise ValueError('Encryption key must be at least 32 characters long')
        return v
    
    @validator('jwt_secret')
    def validate_jwt_secret(cls, v):
        """Validate JWT secret strength"""
        if len(v.get_secret_value()) < 32:
            raise ValueError('JWT secret must be at least 32 characters long')
        return v


class NetworkConfig(BaseModel):
    """Network configuration for services and communication"""
    
    # Main Service Ports
    coordinator_host: str = Field(default="0.0.0.0", description="Coordinator bind host")
    coordinator_port: int = Field(default=8000, ge=1024, le=65535, description="Coordinator HTTP port")
    coordinator_workers: int = Field(default=4, ge=1, le=32, description="Coordinator worker processes")
    
    # Agent Communication
    agent_port_range: Tuple[int, int] = Field(default=(8001, 8100), description="Agent port range")
    agent_heartbeat_interval: int = Field(default=30, ge=5, le=300, description="Agent heartbeat interval in seconds")
    agent_timeout: int = Field(default=120, ge=30, le=600, description="Agent timeout in seconds")
    
    # ZMQ Configuration
    zmq_port: int = Field(default=5555, ge=1024, le=65535, description="ZeroMQ port")
    zmq_hwm: int = Field(default=1000, ge=1, le=100000, description="ZeroMQ high water mark")
    zmq_linger: int = Field(default=1000, ge=0, le=30000, description="ZeroMQ linger time in ms")
    
    # gRPC Configuration
    grpc_port: int = Field(default=50051, ge=1024, le=65535, description="gRPC port")
    grpc_max_workers: int = Field(default=10, ge=1, le=100, description="gRPC max worker threads")
    grpc_max_message_size: int = Field(default=4194304, ge=1024, le=134217728, description="gRPC max message size")
    
    # Load Balancer Configuration
    load_balancer_enabled: bool = Field(default=False, description="Enable load balancer")
    load_balancer_algorithm: str = Field(default="round_robin", description="Load balancer algorithm")
    health_check_interval: int = Field(default=30, ge=5, le=300, description="Health check interval in seconds")
    
    # Network Security
    allowed_hosts: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"], description="Allowed hosts")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS allowed origins")
    
    # Connection Limits
    max_connections: int = Field(default=1000, ge=1, le=10000, description="Maximum concurrent connections")
    connection_timeout: int = Field(default=30, ge=1, le=300, description="Connection timeout in seconds")
    keep_alive_timeout: int = Field(default=5, ge=1, le=60, description="Keep-alive timeout in seconds")
    
    @validator('agent_port_range')
    def validate_port_range(cls, v):
        """Validate agent port range"""
        start, end = v
        if start >= end:
            raise ValueError('Start port must be less than end port')
        if end - start < 10:
            raise ValueError('Port range must be at least 10 ports')
        return v


class GCPConfig(BaseModel):
    """Google Cloud Platform configuration"""
    
    # Project Configuration
    project_id: str = Field(default="tidy-computing-465909-i3", description="GCP project ID")
    region: str = Field(default="europe-west1", description="GCP region")
    zone: str = Field(default="europe-west1-b", description="GCP zone")
    
    # Storage Configuration
    storage_bucket: str = Field(default="aetherveil-storage", description="GCS bucket name")
    storage_class: str = Field(default="STANDARD", description="GCS storage class")
    backup_bucket: str = Field(default="aetherveil-backups", description="GCS backup bucket")
    
    # Secret Manager Configuration
    secret_manager_prefix: str = Field(default="aetherveil", description="Secret Manager prefix")
    secret_manager_enabled: bool = Field(default=True, description="Enable Secret Manager")
    
    # Compute Engine Configuration
    compute_machine_type: str = Field(default="e2-standard-4", description="Compute Engine machine type")
    compute_disk_size: int = Field(default=100, ge=10, le=1000, description="Compute Engine disk size in GB")
    compute_disk_type: str = Field(default="pd-standard", description="Compute Engine disk type")
    
    # Kubernetes Configuration
    gke_cluster_name: str = Field(default="aetherveil-cluster", description="GKE cluster name")
    gke_node_count: int = Field(default=3, ge=1, le=100, description="GKE node count")
    gke_node_machine_type: str = Field(default="e2-standard-2", description="GKE node machine type")
    
    # Cloud Functions Configuration
    cloud_functions_region: str = Field(default="europe-west1", description="Cloud Functions region")
    cloud_functions_memory: int = Field(default=512, ge=128, le=8192, description="Cloud Functions memory in MB")
    cloud_functions_timeout: int = Field(default=300, ge=1, le=3600, description="Cloud Functions timeout in seconds")
    
    # Monitoring Configuration
    monitoring_enabled: bool = Field(default=True, description="Enable Cloud Monitoring")
    logging_enabled: bool = Field(default=True, description="Enable Cloud Logging")
    alerting_enabled: bool = Field(default=True, description="Enable Cloud Alerting")
    
    # Credentials
    service_account_path: Optional[str] = Field(default=None, description="Service account key file path")
    
    @validator('project_id')
    def validate_project_id(cls, v):
        """Validate GCP project ID format"""
        if not v or len(v) < 6:
            raise ValueError('GCP project ID must be at least 6 characters long')
        return v


class AgentConfig(BaseModel):
    """Agent configuration for swarm management"""
    
    # Swarm Configuration
    max_agents: int = Field(default=100, ge=1, le=1000, description="Maximum number of agents")
    agent_spawn_rate: int = Field(default=5, ge=1, le=50, description="Agent spawn rate per minute")
    agent_lifecycle_timeout: int = Field(default=3600, ge=60, le=86400, description="Agent lifecycle timeout in seconds")
    
    # Resource Limits
    max_cpu_per_agent: float = Field(default=0.5, ge=0.1, le=8.0, description="Max CPU cores per agent")
    max_memory_per_agent: int = Field(default=512, ge=128, le=8192, description="Max memory per agent in MB")
    max_disk_per_agent: int = Field(default=1024, ge=256, le=10240, description="Max disk per agent in MB")
    
    # Communication
    message_queue_size: int = Field(default=1000, ge=10, le=10000, description="Message queue size per agent")
    heartbeat_interval: int = Field(default=30, ge=5, le=300, description="Heartbeat interval in seconds")
    discovery_interval: int = Field(default=60, ge=10, le=600, description="Discovery interval in seconds")
    
    # Persistence
    state_persistence_enabled: bool = Field(default=True, description="Enable agent state persistence")
    checkpoint_interval: int = Field(default=300, ge=60, le=3600, description="Checkpoint interval in seconds")
    
    # Scaling
    auto_scaling_enabled: bool = Field(default=True, description="Enable auto-scaling")
    scale_up_threshold: float = Field(default=0.8, ge=0.1, le=1.0, description="Scale up threshold")
    scale_down_threshold: float = Field(default=0.3, ge=0.1, le=1.0, description="Scale down threshold")
    
    # Security
    agent_isolation_enabled: bool = Field(default=True, description="Enable agent isolation")
    sandbox_enabled: bool = Field(default=True, description="Enable sandbox mode")
    
    @validator('scale_up_threshold')
    def validate_scale_thresholds(cls, v, values):
        """Validate scaling thresholds"""
        if 'scale_down_threshold' in values and v <= values['scale_down_threshold']:
            raise ValueError('Scale up threshold must be greater than scale down threshold')
        return v


class RLConfig(BaseModel):
    """Reinforcement Learning configuration"""
    
    # Algorithm Configuration
    algorithm: str = Field(default="PPO", description="RL algorithm (PPO, A2C, SAC, etc.)")
    policy_type: str = Field(default="MlpPolicy", description="Policy network type")
    
    # Training Parameters
    learning_rate: float = Field(default=0.0003, ge=1e-6, le=1e-1, description="Learning rate")
    batch_size: int = Field(default=64, ge=8, le=1024, description="Batch size")
    buffer_size: int = Field(default=100000, ge=1000, le=1000000, description="Replay buffer size")
    
    # PPO Specific
    gamma: float = Field(default=0.99, ge=0.9, le=0.999, description="Discount factor")
    epsilon: float = Field(default=0.2, ge=0.1, le=0.5, description="PPO epsilon")
    entropy_coef: float = Field(default=0.01, ge=0.0, le=0.1, description="Entropy coefficient")
    value_coef: float = Field(default=0.5, ge=0.1, le=1.0, description="Value function coefficient")
    
    # Network Architecture
    net_arch: List[int] = Field(default_factory=lambda: [64, 64], description="Network architecture")
    activation_fn: str = Field(default="tanh", description="Activation function")
    
    # Training Configuration
    total_timesteps: int = Field(default=1000000, ge=10000, le=10000000, description="Total training timesteps")
    eval_freq: int = Field(default=10000, ge=1000, le=100000, description="Evaluation frequency")
    save_freq: int = Field(default=50000, ge=1000, le=1000000, description="Model save frequency")
    
    # Environment Configuration
    env_name: str = Field(default="AetherVeilEnv", description="Environment name")
    num_envs: int = Field(default=4, ge=1, le=32, description="Number of parallel environments")
    
    # Model Checkpointing
    checkpoint_dir: str = Field(default="./checkpoints", description="Checkpoint directory")
    model_save_path: str = Field(default="./models", description="Model save path")
    
    # Hyperparameter Optimization
    hyperopt_enabled: bool = Field(default=False, description="Enable hyperparameter optimization")
    hyperopt_trials: int = Field(default=50, ge=10, le=500, description="Hyperparameter optimization trials")
    
    @validator('net_arch')
    def validate_net_arch(cls, v):
        """Validate network architecture"""
        if not v or len(v) < 1:
            raise ValueError('Network architecture must have at least one layer')
        return v


class StealthConfig(BaseModel):
    """Stealth operation configuration"""
    
    # User Agents
    user_agents: List[str] = Field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
    ], description="User agent strings for stealth operations")
    
    # Proxy Configuration
    proxy_rotation: bool = Field(default=True, description="Enable proxy rotation")
    proxy_list: List[str] = Field(default_factory=list, description="List of proxy servers")
    proxy_timeout: int = Field(default=30, ge=5, le=300, description="Proxy timeout in seconds")
    proxy_retries: int = Field(default=3, ge=1, le=10, description="Proxy retry attempts")
    
    # Traffic Shaping
    jitter_min: float = Field(default=0.5, ge=0.1, le=5.0, description="Minimum jitter in seconds")
    jitter_max: float = Field(default=2.0, ge=0.1, le=10.0, description="Maximum jitter in seconds")
    request_delay: float = Field(default=1.0, ge=0.0, le=10.0, description="Base request delay in seconds")
    
    # Tor Configuration
    tor_enabled: bool = Field(default=False, description="Enable Tor network")
    tor_proxy: str = Field(default="socks5://127.0.0.1:9050", description="Tor proxy address")
    tor_control_port: int = Field(default=9051, ge=1024, le=65535, description="Tor control port")
    
    # Fingerprinting Evasion
    fingerprint_evasion: bool = Field(default=True, description="Enable fingerprint evasion")
    randomize_headers: bool = Field(default=True, description="Randomize HTTP headers")
    spoof_timezone: bool = Field(default=True, description="Spoof timezone")
    
    # Rate Limiting
    max_requests_per_minute: int = Field(default=30, ge=1, le=1000, description="Max requests per minute")
    burst_protection: bool = Field(default=True, description="Enable burst protection")
    
    @validator('jitter_max')
    def validate_jitter_range(cls, v, values):
        """Validate jitter range"""
        if 'jitter_min' in values and v <= values['jitter_min']:
            raise ValueError('Jitter max must be greater than jitter min')
        return v


class LoggingConfig(BaseModel):
    """Logging configuration"""
    
    # General Settings
    log_level: str = Field(default="INFO", description="Default log level")
    log_format: str = Field(default="%(asctime)s [%(levelname)s] %(name)s: %(message)s", description="Log format")
    
    # File Logging
    log_file: str = Field(default="aetherveil.log", description="Log file path")
    log_file_max_size: int = Field(default=10485760, ge=1024, le=1073741824, description="Max log file size in bytes")
    log_file_backup_count: int = Field(default=5, ge=1, le=100, description="Log file backup count")
    
    # Console Logging
    console_logging: bool = Field(default=True, description="Enable console logging")
    console_log_level: str = Field(default="INFO", description="Console log level")
    
    # Structured Logging
    json_logging: bool = Field(default=True, description="Enable JSON logging")
    include_trace: bool = Field(default=False, description="Include trace information")
    
    # Security Logging
    security_log_file: str = Field(default="security.log", description="Security log file")
    audit_log_file: str = Field(default="audit.log", description="Audit log file")
    
    # Performance Logging
    performance_logging: bool = Field(default=True, description="Enable performance logging")
    slow_query_threshold: float = Field(default=1.0, ge=0.1, le=60.0, description="Slow query threshold in seconds")
    
    @validator('log_level', 'console_log_level')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class AetherVeilConfig(BaseSettings):
    """Main configuration class for Aetherveil Sentinel"""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
        env_nested_delimiter='__'
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Enable debug mode")
    testing: bool = Field(default=False, description="Enable testing mode")
    
    # Application Info
    app_name: str = Field(default="Aetherveil Sentinel", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_description: str = Field(default="Advanced Penetration Testing AI Agent", description="Application description")
    
    # Configuration Sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    stealth: StealthConfig = Field(default_factory=StealthConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # External API Keys (loaded from environment or GCP Secret Manager)
    shodan_api_key: Optional[SecretStr] = Field(default=None, description="Shodan API key")
    censys_api_id: Optional[SecretStr] = Field(default=None, description="Censys API ID")
    censys_api_secret: Optional[SecretStr] = Field(default=None, description="Censys API secret")
    virustotal_api_key: Optional[SecretStr] = Field(default=None, description="VirusTotal API key")
    
    # Feature Flags
    features: Dict[str, bool] = Field(default_factory=lambda: {
        "knowledge_graph": True,
        "rl_optimization": True,
        "stealth_mode": True,
        "distributed_scanning": True,
        "auto_exploitation": False,
        "ai_decision_making": True,
        "continuous_learning": True,
        "threat_intelligence": True,
        "osint": True
    })
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment"""
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return f"sqlite:///{self.database.sqlite_path}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        password = ""
        if self.database.redis_password:
            password = f":{self.database.redis_password.get_secret_value()}@"
        return f"redis://{password}{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"
    
    def get_neo4j_url(self) -> str:
        """Get Neo4j connection URL"""
        return self.database.neo4j_uri
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary"""
        # Build handlers dict first
        handlers = {}
        
        if self.logging.console_logging:
            handlers["console"] = {
                "level": self.logging.console_log_level,
                "formatter": "standard",
                "class": "logging.StreamHandler"
            }
        
        handlers["file"] = {
            "level": self.logging.log_level,
            "formatter": "json" if self.logging.json_logging else "standard",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": self.logging.log_file,
            "maxBytes": self.logging.log_file_max_size,
            "backupCount": self.logging.log_file_backup_count
        }
        
        handlers["security"] = {
            "level": "WARNING",
            "formatter": "json" if self.logging.json_logging else "standard",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": self.logging.security_log_file,
            "maxBytes": self.logging.log_file_max_size,
            "backupCount": self.logging.log_file_backup_count
        }
        
        handlers["audit"] = {
            "level": "INFO",
            "formatter": "json" if self.logging.json_logging else "standard",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": self.logging.audit_log_file,
            "maxBytes": self.logging.log_file_max_size,
            "backupCount": self.logging.log_file_backup_count
        }
        
        # Build available handlers list
        available_handlers = []
        if "console" in handlers:
            available_handlers.append("console")
        if "file" in handlers:
            available_handlers.append("file")
        
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.logging.log_format
                },
                "json": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
                } if self.logging.json_logging else {
                    "format": self.logging.log_format
                }
            },
            "handlers": handlers,
            "loggers": {
                "": {
                    "handlers": available_handlers,
                    "level": self.logging.log_level,
                    "propagate": False
                },
                "aetherveil": {
                    "handlers": available_handlers,
                    "level": "DEBUG" if self.debug else self.logging.log_level,
                    "propagate": False
                },
                "security": {
                    "handlers": ["security"],
                    "level": "WARNING",
                    "propagate": False
                },
                "audit": {
                    "handlers": ["audit"],
                    "level": "INFO",
                    "propagate": False
                }
            }
        }


@lru_cache()
def get_config() -> AetherVeilConfig:
    """Get configuration instance with caching"""
    return AetherVeilConfig()


# Global configuration instance
config = get_config()


# Convenience functions for common operations
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get secret value from environment or GCP Secret Manager"""
    # First try environment variable
    value = os.getenv(key, default)
    
    # In production, try GCP Secret Manager
    if config.is_production() and config.gcp.secret_manager_enabled:
        try:
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{config.gcp.project_id}/secrets/{config.gcp.secret_manager_prefix}-{key.lower()}/versions/latest"
            
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logging.warning(f"Failed to get secret {key} from Secret Manager: {e}")
    
    return value


def validate_config() -> List[str]:
    """Validate configuration and return any errors"""
    errors = []
    
    try:
        # Validate configuration
        config = get_config()
        
        # Check required secrets in production
        if config.is_production():
            required_secrets = [
                'encryption_key',
                'jwt_secret'
            ]
            
            for secret in required_secrets:
                if not get_secret(secret):
                    errors.append(f"Missing required secret: {secret}")
        
        # Validate file paths
        if config.security.tls_cert_path and not Path(config.security.tls_cert_path).exists():
            errors.append(f"TLS certificate file not found: {config.security.tls_cert_path}")
        
        if config.security.tls_key_path and not Path(config.security.tls_key_path).exists():
            errors.append(f"TLS key file not found: {config.security.tls_key_path}")
        
        # Validate port conflicts
        ports = [
            config.network.coordinator_port,
            config.network.zmq_port,
            config.network.grpc_port,
            config.database.redis_port
        ]
        
        if len(ports) != len(set(ports)):
            errors.append("Port conflicts detected in configuration")
        
    except Exception as e:
        errors.append(f"Configuration validation error: {e}")
    
    return errors


def setup_logging():
    """Setup logging configuration"""
    import logging.config
    
    config = get_config()
    log_config = config.get_logging_config()
    
    # Create log directories
    for handler_name, handler_config in log_config["handlers"].items():
        if handler_config and "filename" in handler_config:
            log_file = Path(handler_config["filename"])
            log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.config.dictConfig(log_config)


# Initialize logging when module is imported
setup_logging()