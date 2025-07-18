"""
Pytest configuration and fixtures for Aetherveil Sentinel tests.
"""
import pytest
import asyncio
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Optional
import redis
import neo4j
import docker
import yaml
import json
from datetime import datetime, timedelta

# Test configuration
TEST_CONFIG = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "aetherveil_test",
        "user": "test_user",
        "password": "test_password"
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "database": 1  # Use different database for tests
    },
    "neo4j": {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "testpassword"
    },
    "security": {
        "jwt_secret": "test-jwt-secret-key-for-testing",
        "encryption_key": "test-encryption-key-for-testing",
        "hash_algorithm": "sha256"
    },
    "agents": {
        "max_retries": 3,
        "timeout": 30,
        "concurrent_tasks": 5
    },
    "logging": {
        "level": "DEBUG",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG.copy()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_targets():
    """Provide sample targets for testing."""
    return [
        "example.com",
        "test.example.com",
        "192.168.1.1",
        "10.0.0.1",
        "https://example.com",
        "http://test.example.com:8080"
    ]


@pytest.fixture
def sample_vulnerabilities():
    """Provide sample vulnerability data."""
    return [
        {
            "cve": "CVE-2021-44228",
            "severity": "critical",
            "description": "Apache Log4j2 Remote Code Execution",
            "cvss_score": 10.0,
            "affected_versions": ["2.0-beta9", "2.15.0"],
            "solution": "Update to version 2.15.0 or higher"
        },
        {
            "cve": "CVE-2021-45046",
            "severity": "high",
            "description": "Apache Log4j2 Denial of Service",
            "cvss_score": 9.0,
            "affected_versions": ["2.0-beta9", "2.16.0"],
            "solution": "Update to version 2.16.0 or higher"
        }
    ]


@pytest.fixture
def sample_scan_results():
    """Provide sample scan results."""
    return {
        "target": "example.com",
        "timestamp": datetime.now().isoformat(),
        "ports": {
            "open": [80, 443, 22],
            "closed": [21, 23, 25],
            "filtered": [135, 139, 445]
        },
        "services": {
            "80": {"service": "http", "version": "Apache/2.4.41"},
            "443": {"service": "https", "version": "Apache/2.4.41"},
            "22": {"service": "ssh", "version": "OpenSSH 8.2"}
        },
        "vulnerabilities": [
            {
                "port": 80,
                "service": "http",
                "vulnerability": "HTTP methods",
                "severity": "info"
            }
        ]
    }


@pytest.fixture
def sample_osint_data():
    """Provide sample OSINT data."""
    return {
        "domain": "example.com",
        "whois": {
            "registrar": "Example Registrar Inc.",
            "creation_date": "1995-08-14",
            "expiration_date": "2025-08-13",
            "name_servers": ["ns1.example.com", "ns2.example.com"]
        },
        "dns_records": [
            {"type": "A", "value": "93.184.216.34"},
            {"type": "AAAA", "value": "2606:2800:220:1:248:1893:25c8:1946"},
            {"type": "MX", "value": "mail.example.com"}
        ],
        "subdomains": [
            "www.example.com",
            "api.example.com",
            "mail.example.com",
            "ftp.example.com"
        ],
        "certificates": [
            {
                "subject": "CN=example.com",
                "issuer": "DigiCert Inc",
                "valid_from": "2023-01-01",
                "valid_to": "2024-01-01",
                "fingerprint": "sha256:1234567890abcdef"
            }
        ]
    }


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = Mock(spec=redis.Redis)
    client.ping.return_value = True
    client.get.return_value = None
    client.set.return_value = True
    client.delete.return_value = 1
    client.exists.return_value = 0
    client.expire.return_value = True
    client.ttl.return_value = -1
    return client


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = Mock(spec=neo4j.Driver)
    session = Mock(spec=neo4j.Session)
    result = Mock(spec=neo4j.Result)
    
    # Configure the mocks
    driver.session.return_value = session
    session.run.return_value = result
    result.data.return_value = []
    session.close.return_value = None
    driver.close.return_value = None
    
    return driver


@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    client = Mock(spec=docker.DockerClient)
    container = Mock(spec=docker.models.containers.Container)
    
    # Configure container mock
    container.id = "test-container-id"
    container.name = "test-container"
    container.status = "running"
    container.logs.return_value = b"Container logs"
    container.stop.return_value = None
    container.remove.return_value = None
    
    # Configure client mock
    client.containers.run.return_value = container
    client.containers.get.return_value = container
    client.containers.list.return_value = [container]
    
    return client


@pytest.fixture
def mock_agent_registry():
    """Create a mock agent registry."""
    registry = Mock()
    registry.agents = {
        "recon-001": {
            "id": "recon-001",
            "type": "reconnaissance",
            "status": "running",
            "capabilities": ["dns_enumeration", "port_scanning"],
            "last_heartbeat": datetime.now().isoformat()
        },
        "scanner-001": {
            "id": "scanner-001",
            "type": "scanner",
            "status": "running",
            "capabilities": ["vulnerability_scanning"],
            "last_heartbeat": datetime.now().isoformat()
        }
    }
    
    registry.register_agent = Mock(return_value=True)
    registry.unregister_agent = Mock(return_value=True)
    registry.get_agent = Mock(side_effect=lambda agent_id: registry.agents.get(agent_id))
    registry.list_agents = Mock(return_value=list(registry.agents.values()))
    registry.get_agents_by_type = Mock(side_effect=lambda agent_type: [
        agent for agent in registry.agents.values() 
        if agent["type"] == agent_type
    ])
    
    return registry


@pytest.fixture
def mock_security_manager():
    """Create a mock security manager."""
    manager = Mock()
    manager.hash_password = Mock(return_value="hashed_password")
    manager.verify_password = Mock(return_value=True)
    manager.encrypt_data = Mock(return_value=b"encrypted_data")
    manager.decrypt_data = Mock(return_value=b"decrypted_data")
    manager.generate_token = Mock(return_value="jwt_token")
    manager.validate_token = Mock(return_value={"user_id": "test_user"})
    return manager


@pytest.fixture
def mock_task_queue():
    """Create a mock task queue."""
    queue = Mock()
    queue.tasks = []
    
    def enqueue_task(task):
        queue.tasks.append(task)
        return task["id"]
    
    def dequeue_task():
        if queue.tasks:
            return queue.tasks.pop(0)
        return None
    
    queue.enqueue = Mock(side_effect=enqueue_task)
    queue.dequeue = Mock(side_effect=dequeue_task)
    queue.size = Mock(return_value=len(queue.tasks))
    queue.clear = Mock(side_effect=lambda: queue.tasks.clear())
    
    return queue


@pytest.fixture
def sample_tasks():
    """Provide sample tasks for testing."""
    return [
        {
            "id": "task-001",
            "type": "reconnaissance",
            "target": "example.com",
            "priority": "high",
            "created_at": datetime.now().isoformat(),
            "payload": {
                "scan_type": "dns_enumeration",
                "options": {"timeout": 30}
            }
        },
        {
            "id": "task-002",
            "type": "scanning",
            "target": "192.168.1.1",
            "priority": "medium",
            "created_at": datetime.now().isoformat(),
            "payload": {
                "scan_type": "port_scan",
                "options": {"ports": "1-1000"}
            }
        }
    ]


@pytest.fixture
def mock_knowledge_graph():
    """Create a mock knowledge graph."""
    graph = Mock()
    graph.nodes = {}
    graph.relationships = {}
    
    def add_node(node_id, labels, properties):
        graph.nodes[node_id] = {
            "id": node_id,
            "labels": labels,
            "properties": properties
        }
        return node_id
    
    def add_relationship(source_id, target_id, relationship_type, properties):
        rel_id = f"{source_id}-{relationship_type}-{target_id}"
        graph.relationships[rel_id] = {
            "id": rel_id,
            "source": source_id,
            "target": target_id,
            "type": relationship_type,
            "properties": properties
        }
        return rel_id
    
    graph.add_node = Mock(side_effect=add_node)
    graph.add_relationship = Mock(side_effect=add_relationship)
    graph.get_node = Mock(side_effect=lambda node_id: graph.nodes.get(node_id))
    graph.query = Mock(return_value=[])
    
    return graph


@pytest.fixture
def mock_external_apis():
    """Create mocks for external APIs."""
    return {
        "shodan": Mock(
            search=Mock(return_value={"matches": []}),
            host=Mock(return_value={"ip": "93.184.216.34", "ports": [80, 443]})
        ),
        "virustotal": Mock(
            scan_url=Mock(return_value={"scan_id": "test-scan-id"}),
            get_report=Mock(return_value={"positives": 0, "total": 70})
        ),
        "censys": Mock(
            search=Mock(return_value={"results": []}),
            view=Mock(return_value={"ip": "93.184.216.34"})
        )
    }


@pytest.fixture
def test_database_url():
    """Provide test database URL."""
    return "postgresql://test_user:test_password@localhost:5432/aetherveil_test"


@pytest.fixture
def cleanup_files():
    """Fixture to cleanup test files after tests."""
    files_to_cleanup = []
    
    def add_file(filepath):
        files_to_cleanup.append(filepath)
    
    yield add_file
    
    # Cleanup
    for filepath in files_to_cleanup:
        if os.path.exists(filepath):
            os.remove(filepath)


@pytest.fixture
def mock_network_requests():
    """Mock network requests for testing."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post, \
         patch('requests.put') as mock_put, \
         patch('requests.delete') as mock_delete:
        
        # Configure default responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {}
        mock_get.return_value.text = ""
        
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {}
        
        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = {}
        
        mock_delete.return_value.status_code = 204
        
        yield {
            "get": mock_get,
            "post": mock_post,
            "put": mock_put,
            "delete": mock_delete
        }


@pytest.fixture
def performance_threshold():
    """Provide performance thresholds for testing."""
    return {
        "response_time": 1.0,  # 1 second
        "memory_usage": 100 * 1024 * 1024,  # 100MB
        "cpu_usage": 80.0,  # 80%
        "concurrent_requests": 100
    }


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "property: mark test as property-based test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "property" in str(item.fspath):
            item.add_marker(pytest.mark.property)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        
        # Add slow marker for tests that take more than 1 second
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ.update({
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "TESTING": "true",
        "REDIS_URL": "redis://localhost:6379/1",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "testpassword"
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Custom assertions for testing
def assert_valid_json(data):
    """Assert that data is valid JSON."""
    try:
        json.loads(data) if isinstance(data, str) else json.dumps(data)
    except (json.JSONDecodeError, TypeError):
        pytest.fail("Data is not valid JSON")


def assert_valid_uuid(uuid_string):
    """Assert that string is a valid UUID."""
    import uuid
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        pytest.fail(f"'{uuid_string}' is not a valid UUID")


def assert_valid_ipv4(ip_address):
    """Assert that string is a valid IPv4 address."""
    import ipaddress
    try:
        ipaddress.IPv4Address(ip_address)
    except ipaddress.AddressValueError:
        pytest.fail(f"'{ip_address}' is not a valid IPv4 address")


def assert_valid_domain(domain):
    """Assert that string is a valid domain name."""
    import re
    domain_pattern = re.compile(
        r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
    )
    if not domain_pattern.match(domain):
        pytest.fail(f"'{domain}' is not a valid domain name")


# Add custom assertions to pytest namespace
pytest.assert_valid_json = assert_valid_json
pytest.assert_valid_uuid = assert_valid_uuid
pytest.assert_valid_ipv4 = assert_valid_ipv4
pytest.assert_valid_domain = assert_valid_domain