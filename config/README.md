# Aetherveil Sentinel Configuration System

A comprehensive, production-ready configuration management system built with Pydantic for validation and type safety.

## Features

- **Pydantic Validation**: All configuration values are validated with proper type checking
- **Environment Variables**: Full support for environment variable overrides
- **Multiple Environments**: Development, staging, and production configurations
- **GCP Integration**: Built-in support for Google Cloud Platform services
- **Security Focus**: Comprehensive security settings with encryption and authentication
- **Logging Configuration**: Advanced logging with multiple handlers and formatters
- **Secret Management**: Integration with GCP Secret Manager for production secrets
- **Configuration Validation**: Built-in validation and error reporting
- **Migration Tools**: Scripts to help migrate from legacy configuration formats

## Quick Start

### 1. Installation

The configuration system requires additional dependencies:

```bash
pip install pydantic-settings python-json-logger
```

### 2. Basic Usage

```python
from aetherveil_sentinel.config import config

# Access configuration
print(f"Database path: {config.database.sqlite_path}")
print(f"API port: {config.network.coordinator_port}")
print(f"Environment: {config.environment}")

# Check environment
if config.is_production():
    print("Running in production mode")
```

### 3. Environment Variables

Copy the example environment file and customize:

```bash
cp .env.example .env
# Edit .env with your settings
```

Environment variables use double underscores for nested configuration:

```bash
DATABASE__SQLITE_PATH=/path/to/db.sqlite
NETWORK__COORDINATOR_PORT=8080
SECURITY__JWT_SECRET=your-secret-key
```

## Configuration Structure

### Core Configuration Classes

#### `AetherVeilConfig`
Main configuration class that includes all other configuration sections.

#### `DatabaseConfig`
Database configuration for SQLite, Neo4j, and Redis:
- Connection settings
- Pool configurations
- Timeout settings
- Retry behavior

#### `SecurityConfig`
Comprehensive security settings:
- Encryption keys and JWT secrets
- Password policies
- Session management
- TLS/SSL configuration
- Security headers

#### `NetworkConfig`
Network and service configuration:
- Service ports and hosts
- Load balancer settings
- Connection limits
- CORS configuration

#### `GCPConfig`
Google Cloud Platform integration:
- Project and region settings
- Storage bucket configuration
- Kubernetes cluster settings
- Monitoring and logging

#### `AgentConfig`
Agent swarm management:
- Resource limits
- Scaling configuration
- Communication settings
- Security and isolation

#### `RLConfig`
Reinforcement Learning settings:
- Algorithm parameters
- Training configuration
- Model checkpointing
- Hyperparameter optimization

#### `StealthConfig`
Stealth operation settings:
- Proxy rotation
- Traffic shaping
- Tor configuration
- Fingerprint evasion

#### `LoggingConfig`
Advanced logging configuration:
- Multiple log levels
- File and console handlers
- Structured (JSON) logging
- Security and audit logs

## Environment-Specific Configurations

### Development Environment

```bash
# Use development configuration
cp .env.development .env
```

Features:
- Debug mode enabled
- Relaxed security settings
- Local database connections
- Verbose logging
- Disabled external services

### Staging Environment

```bash
# Use staging configuration
cp .env.staging .env
```

Features:
- Production-like settings
- Moderate security
- External service integration
- Performance testing ready

### Production Environment

```bash
# Use production configuration
cp .env.production .env
```

Features:
- Maximum security settings
- TLS/SSL enabled
- GCP Secret Manager integration
- Comprehensive monitoring
- Optimized performance settings

## Configuration Validation

### Validate Configuration

```bash
# Validate current configuration
python -m aetherveil_sentinel.config.validate
```

### Programmatic Validation

```python
from aetherveil_sentinel.config import validate_config

errors = validate_config()
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
```

## Secret Management

### Development
Secrets are loaded from environment variables or `.env` file.

### Production
Secrets are automatically loaded from GCP Secret Manager when available:

```python
from aetherveil_sentinel.config import get_secret

api_key = get_secret("SHODAN_API_KEY")
```

Secret naming convention in GCP Secret Manager:
- Prefix: `aetherveil-prod-` (configurable)
- Format: `aetherveil-prod-shodan-api-key`

## Migration Tools

### Migrate from Legacy YAML

```bash
python -m aetherveil_sentinel.config.migrate --from-yaml old_config.yaml --output .env
```

### Migrate from Legacy JSON

```bash
python -m aetherveil_sentinel.config.migrate --from-json old_config.json --output .env
```

### Generate Configuration Template

```bash
python -m aetherveil_sentinel.config.migrate --template production
```

## Advanced Usage

### Custom Configuration

```python
from aetherveil_sentinel.config import AetherVeilConfig

# Create custom configuration
config = AetherVeilConfig(
    environment="custom",
    database__sqlite_path="/custom/path/db.sqlite",
    network__coordinator_port=9000
)
```

### Configuration Inheritance

```python
from aetherveil_sentinel.config import get_config

# Get cached configuration instance
config = get_config()

# Access nested configuration
db_config = config.database
security_config = config.security
```

### Environment-Specific Features

```python
from aetherveil_sentinel.config import config

# Check environment
if config.is_production():
    # Enable production features
    enable_monitoring()
    enable_security_headers()
elif config.is_development():
    # Enable development features
    enable_debug_mode()
    enable_test_endpoints()
```

## Configuration Reference

### Required Environment Variables

For production deployment, these environment variables must be set:

```bash
# Required for all environments
ENVIRONMENT=production
DATABASE__SQLITE_PATH=/var/lib/aetherveil/aetherveil.db

# Required for production
SECURITY__ENCRYPTION_KEY=your-32-char-encryption-key
SECURITY__JWT_SECRET=your-32-char-jwt-secret
GCP__PROJECT_ID=your-gcp-project-id
```

### Optional Environment Variables

All configuration values can be overridden via environment variables using the double underscore notation:

```bash
# Database configuration
DATABASE__NEO4J_URI=bolt://neo4j.example.com:7687
DATABASE__REDIS_HOST=redis.example.com

# Network configuration
NETWORK__COORDINATOR_PORT=8080
NETWORK__GRPC_PORT=50051

# Security configuration
SECURITY__TLS_ENABLED=true
SECURITY__SESSION_TIMEOUT_MINUTES=30

# Agent configuration
AGENT__MAX_AGENTS=500
AGENT__AUTO_SCALING_ENABLED=true

# And many more...
```

## Logging Configuration

The system provides advanced logging with multiple handlers:

```python
from aetherveil_sentinel.config import setup_logging

# Initialize logging
setup_logging()

# Use different loggers
import logging
logger = logging.getLogger("aetherveil")
security_logger = logging.getLogger("security")
audit_logger = logging.getLogger("audit")

# Log messages
logger.info("Application started")
security_logger.warning("Failed login attempt")
audit_logger.info("Configuration changed")
```

## Feature Flags

Control application features via configuration:

```python
from aetherveil_sentinel.config import config

if config.features["knowledge_graph"]:
    enable_knowledge_graph()

if config.features["rl_optimization"]:
    enable_rl_optimization()

if config.features["stealth_mode"]:
    enable_stealth_mode()
```

## Performance Considerations

- Configuration is cached using `@lru_cache()`
- Validation occurs only once during initialization
- Environment variables are read once at startup
- Secret Manager calls are minimized and cached

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use GCP Secret Manager** in production
3. **Enable TLS/SSL** in production
4. **Set strong password policies**
5. **Configure proper session timeouts**
6. **Enable security headers**
7. **Use environment-specific configurations**

## Troubleshooting

### Common Issues

1. **Configuration validation errors**
   ```bash
   python -m aetherveil_sentinel.config.validate
   ```

2. **Missing environment variables**
   ```bash
   # Check which variables are missing
   python -c "from aetherveil_sentinel.config import validate_config; print(validate_config())"
   ```

3. **Database connection issues**
   ```bash
   # Test database connections
   python -c "from aetherveil_sentinel.config import config; print(config.get_database_url())"
   ```

4. **Secret Manager access issues**
   ```bash
   # Check GCP credentials
   gcloud auth application-default login
   ```

### Debug Mode

Enable debug mode for detailed configuration logging:

```bash
DEBUG=true python your_application.py
```

## Contributing

When adding new configuration options:

1. Add the field to the appropriate config class
2. Include proper validation with Pydantic
3. Add default values
4. Update environment variable examples
5. Update documentation
6. Add validation tests

## License

This configuration system is part of the Aetherveil Sentinel project and follows the same licensing terms.