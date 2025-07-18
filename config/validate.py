#!/usr/bin/env python3
"""
Configuration validation script for Aetherveil Sentinel

This script validates the configuration and reports any issues.
Can be run standalone or imported as a module.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import validate_config, get_config, setup_logging
import logging


def main():
    """Main validation function"""
    print("Aetherveil Sentinel Configuration Validator")
    print("=" * 50)
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Load configuration
        print("Loading configuration...")
        config = get_config()
        
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.debug}")
        print(f"Application: {config.app_name} v{config.app_version}")
        print()
        
        # Validate configuration
        print("Validating configuration...")
        errors = validate_config()
        
        if errors:
            print("❌ Configuration validation failed!")
            print("\nErrors found:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            return 1
        else:
            print("✅ Configuration validation passed!")
            
        # Display configuration summary
        print("\nConfiguration Summary:")
        print(f"  Database: SQLite ({config.database.sqlite_path})")
        print(f"  Neo4j: {config.database.neo4j_uri}")
        print(f"  Redis: {config.database.redis_host}:{config.database.redis_port}")
        print(f"  Coordinator: {config.network.coordinator_host}:{config.network.coordinator_port}")
        print(f"  gRPC: {config.network.grpc_port}")
        print(f"  ZMQ: {config.network.zmq_port}")
        print(f"  Max Agents: {config.agent.max_agents}")
        print(f"  RL Algorithm: {config.rl.algorithm}")
        print(f"  GCP Project: {config.gcp.project_id}")
        print(f"  Log Level: {config.logging.log_level}")
        
        # Feature flags
        print("\nFeature Flags:")
        for feature, enabled in config.features.items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {feature}")
        
        # Security check
        print("\nSecurity Settings:")
        print(f"  TLS Enabled: {'✅' if config.security.tls_enabled else '❌'}")
        print(f"  JWT Expiration: {config.security.jwt_expiration_hours}h")
        print(f"  Session Timeout: {config.security.session_timeout_minutes}m")
        print(f"  Password Min Length: {config.security.password_min_length}")
        print(f"  Max Failed Attempts: {config.security.max_failed_attempts}")
        
        # Network settings
        print("\nNetwork Settings:")
        print(f"  CORS Enabled: {'✅' if config.network.cors_enabled else '❌'}")
        print(f"  Load Balancer: {'✅' if config.network.load_balancer_enabled else '❌'}")
        print(f"  Max Connections: {config.network.max_connections}")
        print(f"  Connection Timeout: {config.network.connection_timeout}s")
        
        # Database settings
        print("\nDatabase Settings:")
        print(f"  SQLite Pool Size: {config.database.sqlite_pool_size}")
        print(f"  Neo4j Pool Size: {config.database.neo4j_pool_size}")
        print(f"  Redis Pool Size: {config.database.redis_pool_size}")
        print(f"  Connection Retries: {config.database.connection_retry_count}")
        
        # Agent settings
        print("\nAgent Settings:")
        print(f"  Auto Scaling: {'✅' if config.agent.auto_scaling_enabled else '❌'}")
        print(f"  Sandbox Mode: {'✅' if config.agent.sandbox_enabled else '❌'}")
        print(f"  Agent Isolation: {'✅' if config.agent.agent_isolation_enabled else '❌'}")
        print(f"  CPU per Agent: {config.agent.max_cpu_per_agent}")
        print(f"  Memory per Agent: {config.agent.max_memory_per_agent}MB")
        
        # Stealth settings
        print("\nStealth Settings:")
        print(f"  Proxy Rotation: {'✅' if config.stealth.proxy_rotation else '❌'}")
        print(f"  Tor Enabled: {'✅' if config.stealth.tor_enabled else '❌'}")
        print(f"  Fingerprint Evasion: {'✅' if config.stealth.fingerprint_evasion else '❌'}")
        print(f"  Request Delay: {config.stealth.request_delay}s")
        print(f"  Max Requests/min: {config.stealth.max_requests_per_minute}")
        
        print("\n✅ Configuration validation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Configuration validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())