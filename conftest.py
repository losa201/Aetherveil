"""
Pytest configuration and fixtures for Aetherveil Sentinel test suite.
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Generator
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import redis
from neo4j import GraphDatabase
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "database_url": "sqlite:///./test.db",
    "redis_url": "redis://localhost:6379/15",
    "neo4j_url": "bolt://localhost:7687",
    "test_mode": True,
    "skip_auth": True,
    "rate_limit_disabled": True,
    "log_level": "INFO"
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return TEST_CONFIG

@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(scope="function")
def mock_database():
    """Mock database connection for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    from coordinator.models import Base
    Base.metadata.create_all(bind=engine)
    
    def get_test_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    yield get_test_db

@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis connection for testing."""
    mock_redis = Mock()
    mock_redis.get = Mock(return_value=None)
    mock_redis.set = Mock(return_value=True)
    mock_redis.delete = Mock(return_value=True)
    mock_redis.exists = Mock(return_value=False)
    mock_redis.expire = Mock(return_value=True)
    mock_redis.incr = Mock(return_value=1)
    mock_redis.decr = Mock(return_value=0)
    mock_redis.hget = Mock(return_value=None)
    mock_redis.hset = Mock(return_value=True)
    mock_redis.hgetall = Mock(return_value={})
    mock_redis.flushall = Mock(return_value=True)
    return mock_redis

@pytest.fixture(scope="function")
def mock_neo4j():
    """Mock Neo4j connection for testing."""
    mock_driver = Mock()
    mock_session = Mock()
    mock_tx = Mock()
    
    mock_tx.run = Mock(return_value=[])
    mock_session.begin_transaction = Mock(return_value=mock_tx)
    mock_session.run = Mock(return_value=[])
    mock_session.close = Mock()
    mock_driver.session = Mock(return_value=mock_session)
    mock_driver.close = Mock()
    
    return mock_driver

@pytest.fixture(scope="function")
def api_client():
    """FastAPI test client."""
    from api.main import app
    
    with patch('api.main.get_database') as mock_db:
        mock_db.return_value = Mock()
        client = TestClient(app)
        yield client

@pytest.fixture(scope="function")
def mock_agent():
    """Mock agent for testing."""
    agent = Mock()
    agent.id = "test-agent-123"
    agent.name = "Test Agent"
    agent.status = "active"
    agent.capabilities = ["scanning", "reconnaissance"]
    agent.last_seen = datetime.utcnow()
    agent.health_score = 100
    agent.execute = AsyncMock(return_value={"status": "success", "data": {}})
    agent.get_status = Mock(return_value="active")
    agent.get_capabilities = Mock(return_value=["scanning", "reconnaissance"])
    return agent

@pytest.fixture(scope="function")
def mock_workflow():
    """Mock workflow for testing."""
    workflow = Mock()
    workflow.id = "test-workflow-123"
    workflow.name = "Test Workflow"
    workflow.status = "pending"
    workflow.tasks = []
    workflow.created_at = datetime.utcnow()
    workflow.updated_at = datetime.utcnow()
    workflow.execute = AsyncMock(return_value={"status": "success"})
    return workflow

@pytest.fixture(scope="function")
def sample_targets():
    """Sample targets for testing."""
    return [
        "example.com",
        "192.168.1.1",
        "testsite.org",
        "10.0.0.1",
        "demo.local"
    ]

@pytest.fixture(scope="function")
def sample_vulnerability_data():
    """Sample vulnerability data for testing."""
    return {
        "cve_id": "CVE-2023-1234",
        "severity": "high",
        "cvss_score": 8.5,
        "description": "Test vulnerability description",
        "affected_systems": ["example.com"],
        "remediation": "Test remediation steps",
        "references": ["https://example.com/advisory"],
        "discovered_at": datetime.utcnow().isoformat()
    }

@pytest.fixture(scope="function")
def sample_osint_data():
    """Sample OSINT data for testing."""
    return {
        "domain": "example.com",
        "emails": ["admin@example.com", "info@example.com"],
        "subdomains": ["www.example.com", "api.example.com"],
        "technologies": ["nginx", "php", "mysql"],
        "social_media": {
            "linkedin": "https://linkedin.com/company/example",
            "twitter": "https://twitter.com/example"
        },
        "dns_records": {
            "A": ["192.168.1.1"],
            "MX": ["mail.example.com"],
            "NS": ["ns1.example.com", "ns2.example.com"]
        }
    }

@pytest.fixture(scope="function")
def mock_authorization():
    """Mock authorization for testing."""
    auth = Mock()
    auth.target = "example.com"
    auth.scope = ["reconnaissance", "scanning"]
    auth.expires_at = datetime.utcnow() + timedelta(hours=1)
    auth.authorized_by = "test-user"
    auth.is_valid = Mock(return_value=True)
    auth.has_permission = Mock(return_value=True)
    return auth

@pytest.fixture(scope="function")
def mock_security_manager():
    """Mock security manager for testing."""
    security_manager = Mock()
    security_manager.validate_target = Mock(return_value=True)
    security_manager.check_authorization = Mock(return_value=True)
    security_manager.log_activity = Mock()
    security_manager.emergency_stop = Mock()
    security_manager.get_risk_score = Mock(return_value=5.0)
    return security_manager

@pytest.fixture(scope="function")
def mock_knowledge_graph():
    """Mock knowledge graph for testing."""
    kg = Mock()
    kg.add_node = Mock()
    kg.add_relationship = Mock()
    kg.query = Mock(return_value=[])
    kg.get_attack_paths = Mock(return_value=[])
    kg.get_vulnerabilities = Mock(return_value=[])
    kg.analyze_risk = Mock(return_value={"score": 7.5, "level": "high"})
    return kg

@pytest.fixture(scope="function")
def mock_rl_agent():
    """Mock reinforcement learning agent for testing."""
    rl_agent = Mock()
    rl_agent.predict = Mock(return_value={"action": "scan", "confidence": 0.8})
    rl_agent.learn = Mock()
    rl_agent.update_policy = Mock()
    rl_agent.get_performance = Mock(return_value={"accuracy": 0.85})
    return rl_agent

@pytest.fixture(scope="function")
def mock_report_generator():
    """Mock report generator for testing."""
    report_gen = Mock()
    report_gen.generate_report = Mock(return_value={"path": "/tmp/test_report.pdf"})
    report_gen.get_templates = Mock(return_value=["executive", "technical", "compliance"])
    report_gen.validate_data = Mock(return_value=True)
    return report_gen

@pytest.fixture(scope="function")
def test_jwt_token():
    """Generate test JWT token."""
    from coordinator.jwt_manager import JWTManager
    jwt_manager = JWTManager(secret_key="test-secret-key")
    return jwt_manager.create_token({"user": "test-user", "role": "admin"})

@pytest.fixture(scope="function")
def test_api_key():
    """Generate test API key."""
    import secrets
    return f"test-api-key-{secrets.token_hex(16)}"

@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services to prevent actual network calls during tests."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post, \
         patch('paramiko.SSHClient') as mock_ssh, \
         patch('nmap.PortScanner') as mock_nmap, \
         patch('scapy.all.send') as mock_send:
        
        # Mock HTTP requests
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "success"}
        mock_get.return_value.text = "<html><body>Test</body></html>"
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "success"}
        
        # Mock SSH
        mock_ssh_instance = Mock()
        mock_ssh_instance.connect = Mock()
        mock_ssh_instance.exec_command = Mock(return_value=(Mock(), Mock(), Mock()))
        mock_ssh.return_value = mock_ssh_instance
        
        # Mock Nmap
        mock_nmap_instance = Mock()
        mock_nmap_instance.scan = Mock(return_value={})
        mock_nmap_instance.all_hosts = Mock(return_value=[])
        mock_nmap.return_value = mock_nmap_instance
        
        # Mock Scapy
        mock_send.return_value = None
        
        yield

@pytest.fixture(scope="function")
def performance_monitor():
    """Performance monitoring fixture for benchmarking."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.process = psutil.Process()
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            self.end_time = time.time()
            self.end_memory = self.process.memory_info().rss
        
        def get_duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        def get_memory_usage(self):
            if self.start_memory and self.end_memory:
                return self.end_memory - self.start_memory
            return None
    
    return PerformanceMonitor()

# Custom assertions for testing
def assert_valid_uuid(uuid_string):
    """Assert that a string is a valid UUID."""
    import uuid
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def assert_valid_iso_datetime(datetime_string):
    """Assert that a string is a valid ISO datetime."""
    try:
        datetime.fromisoformat(datetime_string.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False

def assert_valid_ip(ip_string):
    """Assert that a string is a valid IP address."""
    import ipaddress
    try:
        ipaddress.ip_address(ip_string)
        return True
    except ValueError:
        return False

def assert_valid_url(url_string):
    """Assert that a string is a valid URL."""
    from urllib.parse import urlparse
    try:
        result = urlparse(url_string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# Test data generators
def generate_test_scan_results(count=10):
    """Generate test scan results."""
    import random
    results = []
    for i in range(count):
        results.append({
            "target": f"192.168.1.{i+1}",
            "port": random.randint(1, 65535),
            "service": random.choice(["http", "https", "ssh", "ftp", "smtp"]),
            "state": random.choice(["open", "closed", "filtered"]),
            "version": f"v{random.randint(1, 10)}.{random.randint(0, 9)}"
        })
    return results

def generate_test_vulnerabilities(count=5):
    """Generate test vulnerabilities."""
    import random
    vulnerabilities = []
    for i in range(count):
        vulnerabilities.append({
            "cve_id": f"CVE-2023-{1000+i}",
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "cvss_score": round(random.uniform(1.0, 10.0), 1),
            "description": f"Test vulnerability {i+1}",
            "affected_systems": [f"192.168.1.{random.randint(1, 10)}"],
            "remediation": f"Test remediation {i+1}"
        })
    return vulnerabilities

# pytest markers
pytestmark = pytest.mark.asyncio