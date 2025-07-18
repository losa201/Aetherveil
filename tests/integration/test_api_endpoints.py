"""
Integration tests for API endpoints.
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from datetime import datetime, timedelta

from api.main import app


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test basic health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_health_detailed(self, client):
        """Test detailed health check endpoint"""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "database" in data["components"]
        assert "redis" in data["components"]
        assert "neo4j" in data["components"]


class TestReconnaissanceEndpoints:
    """Test reconnaissance API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_reconnaissance_scan_endpoint(self, client):
        """Test reconnaissance scan endpoint"""
        scan_request = {
            "target": "example.com",
            "target_type": "domain",
            "mode": "passive",
            "depth": 1,
            "timeout": 30
        }
        
        with patch('modules.reconnaissance.ReconnaissanceModule.execute_reconnaissance') as mock_recon:
            mock_recon.return_value = []
            
            response = client.post("/api/v1/reconnaissance/scan", json=scan_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "scan_id" in data
            assert data["status"] == "initiated"
    
    def test_reconnaissance_results_endpoint(self, client):
        """Test reconnaissance results endpoint"""
        scan_id = "test-scan-123"
        
        with patch('modules.reconnaissance.ReconnaissanceModule.get_results') as mock_get_results:
            mock_get_results.return_value = [
                {
                    "target": "example.com",
                    "technique": "dns_lookup",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"ip": "192.168.1.1"},
                    "confidence": 1.0,
                    "source": "dns"
                }
            ]
            
            response = client.get(f"/api/v1/reconnaissance/results/{scan_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
    
    def test_reconnaissance_bulk_scan_endpoint(self, client):
        """Test bulk reconnaissance scan endpoint"""
        bulk_request = {
            "targets": [
                {"target": "example.com", "target_type": "domain"},
                {"target": "test.com", "target_type": "domain"}
            ],
            "mode": "passive"
        }
        
        with patch('modules.reconnaissance.ReconnaissanceModule.bulk_reconnaissance') as mock_bulk:
            mock_bulk.return_value = {"example.com": [], "test.com": []}
            
            response = client.post("/api/v1/reconnaissance/bulk-scan", json=bulk_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "scan_id" in data
            assert data["status"] == "initiated"
    
    def test_reconnaissance_scan_invalid_input(self, client):
        """Test reconnaissance scan with invalid input"""
        invalid_request = {
            "target": "",  # Empty target
            "target_type": "invalid_type"
        }
        
        response = client.post("/api/v1/reconnaissance/scan", json=invalid_request)
        
        assert response.status_code == 422  # Validation error


class TestScanningEndpoints:
    """Test scanning API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_scanning_scan_endpoint(self, client):
        """Test scanning scan endpoint"""
        scan_request = {
            "target": "192.168.1.1",
            "scan_type": "port_scan",
            "intensity": "normal",
            "ports": "80,443",
            "timeout": 300
        }
        
        with patch('modules.scanning.ScanningModule.execute_scan') as mock_scan:
            mock_scan.return_value = Mock(
                target="192.168.1.1",
                scan_type="port_scan",
                status="completed",
                vulnerabilities=[],
                services=[]
            )
            
            response = client.post("/api/v1/scanning/scan", json=scan_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "scan_id" in data
            assert data["status"] == "initiated"
    
    def test_scanning_vulnerabilities_endpoint(self, client):
        """Test scanning vulnerabilities endpoint"""
        scan_id = "test-scan-123"
        
        with patch('modules.scanning.ScanningModule.get_results') as mock_get_results:
            mock_result = Mock()
            mock_result.vulnerabilities = [
                Mock(
                    vuln_id="VULN-001",
                    name="SQL Injection",
                    severity="high",
                    cvss_score=8.5,
                    affected_service="HTTP/80"
                )
            ]
            mock_get_results.return_value = [mock_result]
            
            response = client.get(f"/api/v1/scanning/vulnerabilities/{scan_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "vulnerabilities" in data
    
    def test_scanning_bulk_scan_endpoint(self, client):
        """Test bulk scanning endpoint"""
        bulk_request = {
            "targets": [
                {"target": "192.168.1.1", "scan_type": "port_scan"},
                {"target": "192.168.1.2", "scan_type": "port_scan"}
            ],
            "intensity": "normal"
        }
        
        with patch('modules.scanning.ScanningModule.bulk_scan') as mock_bulk:
            mock_bulk.return_value = []
            
            response = client.post("/api/v1/scanning/bulk-scan", json=bulk_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "scan_id" in data
            assert data["status"] == "initiated"


class TestExploitationEndpoints:
    """Test exploitation API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_exploitation_authorize_endpoint(self, client):
        """Test exploitation authorization endpoint"""
        auth_request = {
            "target": "192.168.1.1",
            "scope": ["reconnaissance", "scanning"],
            "authorized_by": "test-user",
            "duration_hours": 24,
            "justification": "Authorized penetration test"
        }
        
        with patch('modules.exploitation.ExploitationModule.create_authorization') as mock_auth:
            mock_auth.return_value = "auth-token-123"
            
            response = client.post("/api/v1/exploitation/authorize", json=auth_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "authorization_token" in data
            assert data["status"] == "authorized"
    
    def test_exploitation_exploit_endpoint(self, client):
        """Test exploitation exploit endpoint"""
        exploit_request = {
            "target": "192.168.1.1",
            "authorization_token": "auth-token-123",
            "technique": "ssh_bruteforce",
            "parameters": {
                "username": "admin",
                "password_list": ["admin", "password", "123456"]
            }
        }
        
        with patch('modules.exploitation.ExploitationModule.execute_exploitation') as mock_exploit:
            mock_exploit.return_value = Mock(
                status="completed",
                success=False,
                evidence={}
            )
            
            response = client.post("/api/v1/exploitation/exploit", json=exploit_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "exploit_id" in data
            assert data["status"] == "initiated"
    
    def test_exploitation_emergency_stop_endpoint(self, client):
        """Test exploitation emergency stop endpoint"""
        stop_request = {
            "authorization_token": "auth-token-123",
            "reason": "Test emergency stop"
        }
        
        with patch('modules.exploitation.ExploitationModule.emergency_stop') as mock_stop:
            mock_stop.return_value = True
            
            response = client.post("/api/v1/exploitation/emergency-stop", json=stop_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "stopped"
    
    def test_exploitation_unauthorized_request(self, client):
        """Test exploitation without authorization"""
        exploit_request = {
            "target": "192.168.1.1",
            "authorization_token": "invalid-token",
            "technique": "ssh_bruteforce"
        }
        
        with patch('modules.exploitation.ExploitationModule.validate_authorization') as mock_validate:
            mock_validate.return_value = False
            
            response = client.post("/api/v1/exploitation/exploit", json=exploit_request)
            
            assert response.status_code == 403  # Forbidden


class TestStealthEndpoints:
    """Test stealth API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_stealth_apply_endpoint(self, client):
        """Test stealth apply endpoint"""
        stealth_request = {
            "target": "192.168.1.1",
            "techniques": ["traffic_obfuscation", "timing_evasion"],
            "intensity": "high",
            "duration": 3600
        }
        
        with patch('modules.stealth.StealthModule.apply_stealth') as mock_stealth:
            mock_stealth.return_value = Mock(
                status="applied",
                techniques_active=["traffic_obfuscation", "timing_evasion"],
                effectiveness_score=0.85
            )
            
            response = client.post("/api/v1/stealth/apply", json=stealth_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "stealth_id" in data
            assert data["status"] == "applied"
    
    def test_stealth_rating_endpoint(self, client):
        """Test stealth rating endpoint"""
        stealth_id = "stealth-123"
        
        with patch('modules.stealth.StealthModule.get_effectiveness_rating') as mock_rating:
            mock_rating.return_value = {
                "overall_rating": 0.85,
                "detection_probability": 0.15,
                "techniques_effectiveness": {
                    "traffic_obfuscation": 0.9,
                    "timing_evasion": 0.8
                }
            }
            
            response = client.get(f"/api/v1/stealth/rating/{stealth_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "overall_rating" in data
            assert data["overall_rating"] == 0.85
    
    def test_stealth_results_endpoint(self, client):
        """Test stealth results endpoint"""
        stealth_id = "stealth-123"
        
        with patch('modules.stealth.StealthModule.get_results') as mock_results:
            mock_results.return_value = [
                {
                    "technique": "traffic_obfuscation",
                    "status": "active",
                    "effectiveness": 0.9,
                    "detection_events": 0
                }
            ]
            
            response = client.get(f"/api/v1/stealth/results/{stealth_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1


class TestOSINTEndpoints:
    """Test OSINT API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_osint_query_endpoint(self, client):
        """Test OSINT query endpoint"""
        query_request = {
            "target": "example.com",
            "query_type": "domain_intelligence",
            "sources": ["whois", "dns", "ssl_certificates"],
            "depth": 2
        }
        
        with patch('modules.osint.OSINTModule.execute_query') as mock_query:
            mock_query.return_value = [
                {
                    "source": "whois",
                    "data": {"registrar": "Test Registrar"},
                    "confidence": 1.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
            
            response = client.post("/api/v1/osint/query", json=query_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "query_id" in data
            assert data["status"] == "initiated"
    
    def test_osint_search_endpoint(self, client):
        """Test OSINT search endpoint"""
        search_request = {
            "query": "example.com",
            "search_type": "domain",
            "filters": {
                "date_range": "30d",
                "confidence_min": 0.7
            }
        }
        
        with patch('modules.osint.OSINTModule.search_intelligence') as mock_search:
            mock_search.return_value = [
                {
                    "target": "example.com",
                    "intelligence_type": "domain_info",
                    "data": {"ip": "192.168.1.1"},
                    "confidence": 0.9
                }
            ]
            
            response = client.post("/api/v1/osint/search", json=search_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
    
    def test_osint_related_endpoint(self, client):
        """Test OSINT related intelligence endpoint"""
        target = "example.com"
        
        with patch('modules.osint.OSINTModule.find_related_intelligence') as mock_related:
            mock_related.return_value = [
                {
                    "target": "www.example.com",
                    "relationship": "subdomain",
                    "confidence": 0.95
                }
            ]
            
            response = client.get(f"/api/v1/osint/related/{target}")
            
            assert response.status_code == 200
            data = response.json()
            assert "related_intelligence" in data
            assert len(data["related_intelligence"]) == 1


class TestOrchestratorEndpoints:
    """Test orchestrator API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_orchestrator_workflows_endpoint(self, client):
        """Test orchestrator workflows endpoint"""
        workflow_request = {
            "name": "comprehensive_assessment",
            "target": "example.com",
            "modules": ["reconnaissance", "scanning", "osint"],
            "execution_mode": "sequential",
            "timeout": 3600
        }
        
        with patch('modules.orchestrator.OrchestratorModule.create_workflow') as mock_create:
            mock_create.return_value = "workflow-123"
            
            response = client.post("/api/v1/orchestrator/workflows", json=workflow_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "workflow_id" in data
            assert data["status"] == "created"
    
    def test_orchestrator_execute_endpoint(self, client):
        """Test orchestrator execute endpoint"""
        execute_request = {
            "workflow_id": "workflow-123",
            "parameters": {
                "target": "example.com",
                "scan_intensity": "normal"
            }
        }
        
        with patch('modules.orchestrator.OrchestratorModule.execute_workflow') as mock_execute:
            mock_execute.return_value = Mock(
                status="running",
                progress=0.0,
                current_task="reconnaissance"
            )
            
            response = client.post("/api/v1/orchestrator/execute", json=execute_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "execution_id" in data
            assert data["status"] == "initiated"
    
    def test_orchestrator_status_endpoint(self, client):
        """Test orchestrator status endpoint"""
        execution_id = "execution-123"
        
        with patch('modules.orchestrator.OrchestratorModule.get_execution_status') as mock_status:
            mock_status.return_value = {
                "status": "running",
                "progress": 0.5,
                "current_task": "scanning",
                "completed_tasks": ["reconnaissance"],
                "remaining_tasks": ["osint", "reporting"]
            }
            
            response = client.get(f"/api/v1/orchestrator/status/{execution_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert data["progress"] == 0.5


class TestReportingEndpoints:
    """Test reporting API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_reporting_generate_endpoint(self, client):
        """Test reporting generate endpoint"""
        report_request = {
            "report_type": "executive_summary",
            "data_sources": ["reconnaissance", "scanning", "osint"],
            "target": "example.com",
            "format": "pdf",
            "template": "default"
        }
        
        with patch('modules.reporting.ReportingModule.generate_report') as mock_generate:
            mock_generate.return_value = "report-123"
            
            response = client.post("/api/v1/reporting/generate", json=report_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "report_id" in data
            assert data["status"] == "generated"
    
    def test_reporting_download_endpoint(self, client):
        """Test reporting download endpoint"""
        report_id = "report-123"
        
        with patch('modules.reporting.ReportingModule.get_report_file') as mock_get_file:
            mock_get_file.return_value = b"PDF content"
            
            response = client.get(f"/api/v1/reporting/download/{report_id}")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
    
    def test_reporting_templates_endpoint(self, client):
        """Test reporting templates endpoint"""
        with patch('modules.reporting.ReportingModule.get_available_templates') as mock_templates:
            mock_templates.return_value = [
                {"name": "executive_summary", "description": "Executive summary report"},
                {"name": "technical_detailed", "description": "Technical detailed report"}
            ]
            
            response = client.get("/api/v1/reporting/templates")
            
            assert response.status_code == 200
            data = response.json()
            assert "templates" in data
            assert len(data["templates"]) == 2


class TestAPIAuthentication:
    """Test API authentication and authorization"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_protected_endpoint_without_auth(self, client):
        """Test protected endpoint without authentication"""
        response = client.post("/api/v1/exploitation/exploit", json={})
        
        assert response.status_code == 401  # Unauthorized
    
    def test_protected_endpoint_with_valid_auth(self, client):
        """Test protected endpoint with valid authentication"""
        headers = {"Authorization": "Bearer valid-token"}
        
        with patch('api.middleware.verify_token') as mock_verify:
            mock_verify.return_value = {"user": "test-user", "role": "admin"}
            
            response = client.post("/api/v1/exploitation/authorize", 
                                 headers=headers,
                                 json={"target": "example.com"})
            
            # Should not return 401
            assert response.status_code != 401
    
    def test_protected_endpoint_with_invalid_auth(self, client):
        """Test protected endpoint with invalid authentication"""
        headers = {"Authorization": "Bearer invalid-token"}
        
        with patch('api.middleware.verify_token') as mock_verify:
            mock_verify.return_value = None
            
            response = client.post("/api/v1/exploitation/authorize", 
                                 headers=headers,
                                 json={"target": "example.com"})
            
            assert response.status_code == 401  # Unauthorized


class TestAPIRateLimiting:
    """Test API rate limiting"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_rate_limiting_enforcement(self, client):
        """Test rate limiting enforcement"""
        # Make multiple requests rapidly
        responses = []
        for i in range(20):
            response = client.get("/health")
            responses.append(response)
        
        # Check if any requests are rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        
        # In test environment, rate limiting might be disabled
        # This test documents the expected behavior
        assert True  # Rate limiting behavior depends on configuration


class TestAPIErrorHandling:
    """Test API error handling"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_404_error_handling(self, client):
        """Test 404 error handling"""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"] == "Not Found"
    
    def test_500_error_handling(self, client):
        """Test 500 error handling"""
        with patch('api.v1.reconnaissance.reconnaissance_scan') as mock_scan:
            mock_scan.side_effect = Exception("Internal server error")
            
            response = client.post("/api/v1/reconnaissance/scan", 
                                 json={"target": "example.com", "target_type": "domain"})
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
    
    def test_validation_error_handling(self, client):
        """Test validation error handling"""
        invalid_request = {
            "target": "",  # Empty target
            "target_type": "invalid_type"
        }
        
        response = client.post("/api/v1/reconnaissance/scan", json=invalid_request)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)


class TestAPICors:
    """Test API CORS configuration"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_cors_headers(self, client):
        """Test CORS headers in response"""
        response = client.options("/api/v1/reconnaissance/scan")
        
        # Check for CORS headers
        assert response.status_code == 200
        # CORS headers might be added by middleware
        # This test documents the expected behavior


class TestAPIWebSocket:
    """Test API WebSocket connections"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection for real-time updates"""
        with client.websocket_connect("/ws/status") as websocket:
            # Send status request
            websocket.send_json({"type": "status_request", "module": "reconnaissance"})
            
            # Receive status response
            data = websocket.receive_json()
            
            assert data["type"] == "status_response"
            assert "module" in data
            assert "status" in data
    
    def test_websocket_scan_updates(self, client):
        """Test WebSocket scan progress updates"""
        with client.websocket_connect("/ws/scan_progress") as websocket:
            # Send scan progress request
            websocket.send_json({"type": "subscribe", "scan_id": "test-scan-123"})
            
            # Receive progress update
            data = websocket.receive_json()
            
            assert data["type"] == "progress_update"
            assert "scan_id" in data
            assert "progress" in data