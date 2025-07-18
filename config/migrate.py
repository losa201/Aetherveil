#!/usr/bin/env python3
"""
Configuration migration script for Aetherveil Sentinel

This script helps migrate configuration from old formats to new Pydantic-based configuration.
It can also generate configuration files from templates.
"""

import sys
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any

# Add the parent directory to the path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config, AetherVeilConfig
import logging


def migrate_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Migrate configuration from YAML format"""
    try:
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        env_vars = {}
        
        # Helper function to convert nested dict to env vars
        def dict_to_env_vars(data: Dict[str, Any], prefix: str = ""):
            for key, value in data.items():
                env_key = f"{prefix}{key.upper()}" if prefix else key.upper()
                
                if isinstance(value, dict):
                    dict_to_env_vars(value, f"{env_key}__")
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    env_vars[env_key] = ",".join(str(item) for item in value)
                else:
                    env_vars[env_key] = str(value)
        
        dict_to_env_vars(yaml_config)
        return env_vars
        
    except Exception as e:
        print(f"Error migrating from YAML: {e}")
        return {}


def migrate_from_json(json_path: str) -> Dict[str, Any]:
    """Migrate configuration from JSON format"""
    try:
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        
        env_vars = {}
        
        # Helper function to convert nested dict to env vars
        def dict_to_env_vars(data: Dict[str, Any], prefix: str = ""):
            for key, value in data.items():
                env_key = f"{prefix}{key.upper()}" if prefix else key.upper()
                
                if isinstance(value, dict):
                    dict_to_env_vars(value, f"{env_key}__")
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    env_vars[env_key] = ",".join(str(item) for item in value)
                else:
                    env_vars[env_key] = str(value)
        
        dict_to_env_vars(json_config)
        return env_vars
        
    except Exception as e:
        print(f"Error migrating from JSON: {e}")
        return {}


def generate_env_file(env_vars: Dict[str, Any], output_path: str):
    """Generate .env file from environment variables"""
    try:
        with open(output_path, 'w') as f:
            f.write("# Aetherveil Sentinel Configuration\n")
            f.write("# Migrated from legacy configuration\n\n")
            
            for key, value in sorted(env_vars.items()):
                f.write(f"{key}={value}\n")
        
        print(f"✅ Generated .env file: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating .env file: {e}")


def generate_config_template(environment: str = "development"):
    """Generate configuration template for specific environment"""
    try:
        config_path = Path(__file__).parent
        template_path = config_path / f".env.{environment}"
        
        if template_path.exists():
            print(f"✅ Configuration template already exists: {template_path}")
            return
        
        # Load base template
        base_template = config_path / ".env.example"
        if not base_template.exists():
            print(f"❌ Base template not found: {base_template}")
            return
        
        # Copy template and modify for environment
        with open(base_template, 'r') as f:
            content = f.read()
        
        # Environment-specific modifications
        if environment == "production":
            content = content.replace("ENVIRONMENT=development", "ENVIRONMENT=production")
            content = content.replace("DEBUG=true", "DEBUG=false")
            content = content.replace("localhost", "production-host")
        elif environment == "staging":
            content = content.replace("ENVIRONMENT=development", "ENVIRONMENT=staging")
            content = content.replace("DEBUG=true", "DEBUG=false")
            content = content.replace("localhost", "staging-host")
        
        with open(template_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Generated configuration template: {template_path}")
        
    except Exception as e:
        print(f"❌ Error generating configuration template: {e}")


def validate_migration():
    """Validate migrated configuration"""
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        # Check critical settings
        if config.is_production():
            print("⚠️  Production environment detected")
            if not config.security.tls_enabled:
                print("❌ TLS should be enabled in production")
            if config.debug:
                print("❌ Debug mode should be disabled in production")
        
        print(f"Environment: {config.environment}")
        print(f"Database: {config.database.sqlite_path}")
        print(f"Network: {config.network.coordinator_host}:{config.network.coordinator_port}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aetherveil Sentinel Configuration Migration Tool")
    parser.add_argument("--from-yaml", help="Migrate from YAML file")
    parser.add_argument("--from-json", help="Migrate from JSON file")
    parser.add_argument("--output", help="Output .env file path", default=".env")
    parser.add_argument("--template", help="Generate configuration template", choices=["development", "staging", "production"])
    parser.add_argument("--validate", help="Validate migrated configuration", action="store_true")
    
    args = parser.parse_args()
    
    print("Aetherveil Sentinel Configuration Migration Tool")
    print("=" * 50)
    
    env_vars = {}
    
    if args.from_yaml:
        print(f"Migrating from YAML: {args.from_yaml}")
        env_vars = migrate_from_yaml(args.from_yaml)
    
    if args.from_json:
        print(f"Migrating from JSON: {args.from_json}")
        env_vars = migrate_from_json(args.from_json)
    
    if env_vars:
        generate_env_file(env_vars, args.output)
    
    if args.template:
        print(f"Generating configuration template for: {args.template}")
        generate_config_template(args.template)
    
    if args.validate:
        print("Validating migrated configuration...")
        if not validate_migration():
            return 1
    
    if not any([args.from_yaml, args.from_json, args.template, args.validate]):
        parser.print_help()
        return 1
    
    print("\n✅ Migration completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())