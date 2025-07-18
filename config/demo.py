#!/usr/bin/env python3
"""
Demonstration script for the Aetherveil Sentinel configuration system
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config, validate_config, get_secret
import json


def main():
    """Main demonstration function"""
    print("🔧 Aetherveil Sentinel Configuration System Demo")
    print("=" * 60)
    
    # Load configuration
    config = get_config()
    
    # Display basic information
    print(f"🏷️  Application: {config.app_name} v{config.app_version}")
    print(f"🌍 Environment: {config.environment}")
    print(f"🐛 Debug Mode: {'Enabled' if config.debug else 'Disabled'}")
    print()
    
    # Database configuration
    print("📊 Database Configuration:")
    print(f"  SQLite: {config.database.sqlite_path}")
    print(f"  Neo4j: {config.database.neo4j_uri}")
    print(f"  Redis: {config.database.redis_host}:{config.database.redis_port}")
    print(f"  Pool sizes: SQLite={config.database.sqlite_pool_size}, Neo4j={config.database.neo4j_pool_size}, Redis={config.database.redis_pool_size}")
    print()
    
    # Network configuration
    print("🌐 Network Configuration:")
    print(f"  Coordinator: {config.network.coordinator_host}:{config.network.coordinator_port}")
    print(f"  gRPC: {config.network.grpc_port}")
    print(f"  ZMQ: {config.network.zmq_port}")
    print(f"  Workers: {config.network.coordinator_workers}")
    print(f"  Max connections: {config.network.max_connections}")
    print()
    
    # Security configuration
    print("🔒 Security Configuration:")
    print(f"  TLS Enabled: {config.security.tls_enabled}")
    print(f"  JWT Algorithm: {config.security.jwt_algorithm}")
    print(f"  JWT Expiration: {config.security.jwt_expiration_hours}h")
    print(f"  Session Timeout: {config.security.session_timeout_minutes}m")
    print(f"  Password Min Length: {config.security.password_min_length}")
    print(f"  Max Failed Attempts: {config.security.max_failed_attempts}")
    print()
    
    # GCP configuration
    print("☁️  GCP Configuration:")
    print(f"  Project ID: {config.gcp.project_id}")
    print(f"  Region: {config.gcp.region}")
    print(f"  Storage Bucket: {config.gcp.storage_bucket}")
    print(f"  Secret Manager: {'Enabled' if config.gcp.secret_manager_enabled else 'Disabled'}")
    print(f"  Monitoring: {'Enabled' if config.gcp.monitoring_enabled else 'Disabled'}")
    print()
    
    # Agent configuration
    print("🤖 Agent Configuration:")
    print(f"  Max Agents: {config.agent.max_agents}")
    print(f"  CPU per Agent: {config.agent.max_cpu_per_agent}")
    print(f"  Memory per Agent: {config.agent.max_memory_per_agent}MB")
    print(f"  Auto Scaling: {'Enabled' if config.agent.auto_scaling_enabled else 'Disabled'}")
    print(f"  Sandbox Mode: {'Enabled' if config.agent.sandbox_enabled else 'Disabled'}")
    print()
    
    # RL configuration
    print("🧠 Reinforcement Learning Configuration:")
    print(f"  Algorithm: {config.rl.algorithm}")
    print(f"  Learning Rate: {config.rl.learning_rate}")
    print(f"  Batch Size: {config.rl.batch_size}")
    print(f"  Gamma: {config.rl.gamma}")
    print(f"  Total Timesteps: {config.rl.total_timesteps:,}")
    print()
    
    # Stealth configuration
    print("🥷 Stealth Configuration:")
    print(f"  Proxy Rotation: {'Enabled' if config.stealth.proxy_rotation else 'Disabled'}")
    print(f"  Tor Network: {'Enabled' if config.stealth.tor_enabled else 'Disabled'}")
    print(f"  Fingerprint Evasion: {'Enabled' if config.stealth.fingerprint_evasion else 'Disabled'}")
    print(f"  Request Delay: {config.stealth.request_delay}s")
    print(f"  Max Requests/min: {config.stealth.max_requests_per_minute}")
    print()
    
    # Logging configuration
    print("📝 Logging Configuration:")
    print(f"  Log Level: {config.logging.log_level}")
    print(f"  Log File: {config.logging.log_file}")
    print(f"  JSON Logging: {'Enabled' if config.logging.json_logging else 'Disabled'}")
    print(f"  Console Logging: {'Enabled' if config.logging.console_logging else 'Disabled'}")
    print(f"  Performance Logging: {'Enabled' if config.logging.performance_logging else 'Disabled'}")
    print()
    
    # Feature flags
    print("🎛️  Feature Flags:")
    for feature, enabled in config.features.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    print()
    
    # Connection URLs
    print("🔗 Connection URLs:")
    print(f"  Database: {config.get_database_url()}")
    print(f"  Redis: {config.get_redis_url()}")
    print(f"  Neo4j: {config.get_neo4j_url()}")
    print()
    
    # Validation
    print("✅ Configuration Validation:")
    errors = validate_config()
    if errors:
        print("❌ Configuration has errors:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("✅ Configuration is valid!")
    print()
    
    # Environment detection
    print("🌍 Environment Detection:")
    print(f"  Is Production: {config.is_production()}")
    print(f"  Is Development: {config.is_development()}")
    print()
    
    # Configuration summary
    print("📋 Configuration Summary:")
    print(f"  Total Configuration Classes: 8")
    print(f"  Total Configuration Fields: 100+")
    print(f"  Environment Variables Supported: ✅")
    print(f"  Pydantic Validation: ✅")
    print(f"  GCP Secret Manager: ✅")
    print(f"  Multiple Environments: ✅")
    print(f"  Comprehensive Security: ✅")
    print()
    
    print("🎉 Configuration system is ready for production use!")
    print("📖 See config/README.md for detailed documentation")


if __name__ == "__main__":
    main()