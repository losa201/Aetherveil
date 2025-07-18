"""
Unit tests for the Scanning module.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import json
import nmap
import socket
import ssl
import subprocess
from typing import List, Dict, Any

from modules.scanning import (
    ScanningModule,
    NetworkScanner,
    VulnerabilityScanner,
    WebApplicationScanner,
    SSLScanner,
    ComplianceScanner,
    ScanTarget,
    ScanResult,
    Vulnerability,
    ScanType,
    ScanIntensity,
    VulnerabilitySeverity
)
from config.config import AetherVeilConfig


class TestScanTarget:
    """Test ScanTarget data class"""
    
    def test_scan_target_creation(self):
        """Test ScanTarget creation with default values"""
        target = ScanTarget(
            target="192.168.1.1",
            scan_type=ScanType.PORT_SCAN
        )
        
        assert target.target == "192.168.1.1"
        assert target.scan_type == ScanType.PORT_SCAN
        assert target.intensity == ScanIntensity.NORMAL
        assert target.ports is None
        assert target.timeout == 300
        assert target.options == {}
    
    def test_scan_target_with_custom_values(self):
        """Test ScanTarget creation with custom values"""
        options = {"exclude_hosts": "192.168.1.1", "max_rate": 1000}
        target = ScanTarget(
            target="192.168.1.0/24",
            scan_type=ScanType.VULNERABILITY_SCAN,
            intensity=ScanIntensity.AGGRESSIVE,
            ports="1-65535",
            timeout=600,
            options=options
        )
        
        assert target.target == "192.168.1.0/24"
        assert target.scan_type == ScanType.VULNERABILITY_SCAN
        assert target.intensity == ScanIntensity.AGGRESSIVE
        assert target.ports == "1-65535"
        assert target.timeout == 600
        assert target.options == options


class TestVulnerability:
    """Test Vulnerability data class"""
    
    def test_vulnerability_creation(self):
        """Test Vulnerability creation"""
        vulnerability = Vulnerability(
            vuln_id="VULN-001",
            name="SQL Injection",
            description="SQL injection vulnerability in login form",
            severity=VulnerabilitySeverity.HIGH,
            cvss_score=8.5,
            cve_id="CVE-2023-1234",
            affected_service="HTTP/80",
            evidence={"payload": "' OR 1=1--", "response": "Login successful"},
            remediation="Use parameterized queries"
        )
        
        assert vulnerability.vuln_id == "VULN-001"
        assert vulnerability.name == "SQL Injection"
        assert vulnerability.severity == VulnerabilitySeverity.HIGH
        assert vulnerability.cvss_score == 8.5
        assert vulnerability.cve_id == "CVE-2023-1234"
        assert vulnerability.affected_service == "HTTP/80"
        assert vulnerability.references == []


class TestScanResult:
    """Test ScanResult data class"""
    
    def test_scan_result_creation(self):
        """Test ScanResult creation"""
        timestamp = datetime.utcnow()
        duration = timedelta(seconds=30)
        vulnerabilities = [
            Vulnerability(
                vuln_id="VULN-001",
                name="Test Vulnerability",
                description="Test description",
                severity=VulnerabilitySeverity.MEDIUM,
                cvss_score=5.0,
                cve_id=None,
                affected_service="HTTP/80",
                evidence={},
                remediation="Test remediation"
            )
        ]
        services = [{"host": "192.168.1.1", "port": 80, "service": "http"}]
        
        result = ScanResult(
            target="192.168.1.1",
            scan_type=ScanType.PORT_SCAN,
            timestamp=timestamp,
            duration=duration,
            status="completed",
            vulnerabilities=vulnerabilities,
            services=services
        )
        
        assert result.target == "192.168.1.1"
        assert result.scan_type == ScanType.PORT_SCAN
        assert result.timestamp == timestamp
        assert result.duration == duration
        assert result.status == "completed"
        assert len(result.vulnerabilities) == 1
        assert len(result.services) == 1
        assert result.metadata == {}


class TestNetworkScanner:
    """Test NetworkScanner functionality"""
    
    @pytest.fixture
    def network_scanner(self):
        """Network scanner fixture"""
        return NetworkScanner()
    
    @pytest.mark.asyncio
    async def test_port_scan_success(self, network_scanner):
        """Test successful port scan"""
        with patch.object(network_scanner.nm, 'scan') as mock_scan:
            with patch.object(network_scanner.nm, 'all_hosts', return_value=['192.168.1.1']):
                # Mock nmap host data
                mock_host = Mock()
                mock_host.state.return_value = 'up'
                mock_host.all_protocols.return_value = ['tcp']
                mock_host.hostnames.return_value = [{'name': 'test.local', 'type': 'PTR'}]
                mock_host.get.return_value = {}
                
                # Mock port data
                mock_port_info = {
                    'state': 'open',
                    'name': 'http',
                    'product': 'Apache',
                    'version': '2.4.41',
                    'extrainfo': 'Ubuntu',
                    'conf': '10',
                    'cpe': 'cpe:/a:apache:http_server:2.4.41',
                    'script': {}
                }
                
                mock_protocol = Mock()
                mock_protocol.keys.return_value = [80]
                mock_protocol.__getitem__.return_value = mock_port_info
                
                mock_host.__getitem__.return_value = mock_protocol
                network_scanner.nm.__getitem__ = Mock(return_value=mock_host)
                
                result = await network_scanner.port_scan("192.168.1.1", "80")
                
                assert result["target"] == "192.168.1.1"
                assert result["ports_scanned"] == "80"
                assert "192.168.1.1" in result["hosts"]
                assert result["hosts"]["192.168.1.1"]["state"] == "up"
    
    @pytest.mark.asyncio
    async def test_port_scan_with_different_intensities(self, network_scanner):
        """Test port scan with different intensity levels"""
        with patch.object(network_scanner.nm, 'scan') as mock_scan:
            with patch.object(network_scanner.nm, 'all_hosts', return_value=[]):
                
                # Test stealth scan
                await network_scanner.port_scan("192.168.1.1", "80", ScanIntensity.STEALTH)
                mock_scan.assert_called_with(hosts="192.168.1.1", ports="80", arguments="-sS -T2 -f")
                
                # Test normal scan
                await network_scanner.port_scan("192.168.1.1", "80", ScanIntensity.NORMAL)
                mock_scan.assert_called_with(hosts="192.168.1.1", ports="80", arguments="-sS -T3")
                
                # Test aggressive scan
                await network_scanner.port_scan("192.168.1.1", "80", ScanIntensity.AGGRESSIVE)
                mock_scan.assert_called_with(hosts="192.168.1.1", ports="80", arguments="-sS -T4 -A")
                
                # Test comprehensive scan
                await network_scanner.port_scan("192.168.1.1", "80", ScanIntensity.COMPREHENSIVE)
                mock_scan.assert_called_with(hosts="192.168.1.1", ports="80", arguments="-sS -sV -sC -T4 -A --script vuln")
    
    @pytest.mark.asyncio
    async def test_port_scan_failure(self, network_scanner):
        """Test port scan failure handling"""
        with patch.object(network_scanner.nm, 'scan') as mock_scan:
            mock_scan.side_effect = Exception("Scan failed")
            
            with pytest.raises(Exception, match="Scan failed"):
                await network_scanner.port_scan("192.168.1.1", "80")
    
    @pytest.mark.asyncio
    async def test_service_detection(self, network_scanner):
        """Test service detection functionality"""
        with patch.object(network_scanner.nm, 'scan') as mock_scan:
            with patch.object(network_scanner.nm, 'all_hosts', return_value=['192.168.1.1']):
                with patch.object(network_scanner, '_grab_service_banner', return_value="HTTP/1.1 200 OK"):
                    
                    # Mock nmap service data
                    mock_host = Mock()
                    mock_host.all_protocols.return_value = ['tcp']
                    
                    mock_port_info = {
                        'state': 'open',
                        'name': 'http',
                        'product': 'Apache',
                        'version': '2.4.41',
                        'extrainfo': 'Ubuntu',
                        'tunnel': '',
                        'method': 'probed',
                        'conf': '10',
                        'cpe': 'cpe:/a:apache:http_server:2.4.41',
                        'script': {}
                    }
                    
                    mock_protocol = Mock()
                    mock_protocol.keys.return_value = [80]
                    mock_protocol.__getitem__.return_value = mock_port_info
                    
                    mock_host.__getitem__.return_value = mock_protocol
                    network_scanner.nm.__getitem__ = Mock(return_value=mock_host)
                    
                    services = await network_scanner.service_detection("192.168.1.1", "80")
                    
                    assert len(services) == 1
                    assert services[0]["host"] == "192.168.1.1"
                    assert services[0]["port"] == 80
                    assert services[0]["service"] == "http"
                    assert services[0]["product"] == "Apache"
                    assert services[0]["version"] == "2.4.41"
                    assert services[0]["banner"] == "HTTP/1.1 200 OK"
    
    @pytest.mark.asyncio
    async def test_grab_service_banner(self, network_scanner):
        """Test service banner grabbing"""
        with patch('socket.socket') as mock_socket:
            mock_sock = Mock()
            mock_sock.connect.return_value = None
            mock_sock.send.return_value = None
            mock_sock.recv.return_value = b"HTTP/1.1 200 OK\r\nServer: Apache/2.4.41\r\n"
            mock_sock.settimeout.return_value = None
            mock_sock.close.return_value = None
            mock_socket.return_value = mock_sock
            
            banner = await network_scanner._grab_service_banner("192.168.1.1", 80)
            
            assert "HTTP/1.1 200 OK" in banner
            assert "Apache/2.4.41" in banner
    
    @pytest.mark.asyncio
    async def test_grab_service_banner_failure(self, network_scanner):
        """Test service banner grabbing failure"""
        with patch('socket.socket') as mock_socket:
            mock_socket.side_effect = Exception("Connection failed")
            
            banner = await network_scanner._grab_service_banner("192.168.1.1", 80)
            
            assert banner is None


class TestVulnerabilityScanner:
    """Test VulnerabilityScanner functionality"""
    
    @pytest.fixture
    def vuln_scanner(self):
        """Vulnerability scanner fixture"""
        return VulnerabilityScanner()
    
    @pytest.mark.asyncio
    async def test_vulnerability_scan_success(self, vuln_scanner):
        """Test successful vulnerability scan"""
        # Mock service detection
        services = [
            {"host": "192.168.1.1", "port": 80, "service": "http", "product": "Apache", "version": "2.4.41"}
        ]
        
        with patch.object(vuln_scanner, '_check_web_vulnerabilities', return_value=[]):
            with patch.object(vuln_scanner, '_check_ssl_vulnerabilities', return_value=[]):
                with patch.object(vuln_scanner, '_check_service_vulnerabilities', return_value=[]):
                    with patch.object(vuln_scanner, '_check_default_credentials', return_value=[]):
                        
                        vulnerabilities = await vuln_scanner.vulnerability_scan("192.168.1.1", services)
                        
                        assert isinstance(vulnerabilities, list)
    
    @pytest.mark.asyncio
    async def test_check_web_vulnerabilities(self, vuln_scanner):
        """Test web vulnerability checking"""
        service = {"host": "192.168.1.1", "port": 80, "service": "http"}
        
        with patch.object(vuln_scanner, '_test_sql_injection', return_value=None):
            with patch.object(vuln_scanner, '_test_xss', return_value=None):
                with patch.object(vuln_scanner, '_test_csrf', return_value=None):
                    with patch.object(vuln_scanner, '_test_directory_traversal', return_value=None):
                        
                        vulnerabilities = await vuln_scanner._check_web_vulnerabilities(service)
                        
                        assert isinstance(vulnerabilities, list)
    
    @pytest.mark.asyncio
    async def test_sql_injection_detection(self, vuln_scanner):
        """Test SQL injection detection"""
        service = {"host": "192.168.1.1", "port": 80, "service": "http"}
        
        with patch('requests.get') as mock_get:
            # Mock vulnerable response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "SQL error: syntax error"
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            vulnerability = await vuln_scanner._test_sql_injection(service)
            
            assert vulnerability is not None
            assert vulnerability.name == "SQL Injection"
            assert vulnerability.severity == VulnerabilitySeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_xss_detection(self, vuln_scanner):
        """Test XSS vulnerability detection"""
        service = {"host": "192.168.1.1", "port": 80, "service": "http"}
        
        with patch('requests.get') as mock_get:
            # Mock vulnerable response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<script>alert('XSS')</script>"
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            vulnerability = await vuln_scanner._test_xss(service)
            
            assert vulnerability is not None
            assert vulnerability.name == "Cross-Site Scripting (XSS)"
            assert vulnerability.severity == VulnerabilitySeverity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_default_credentials_check(self, vuln_scanner):
        """Test default credentials checking"""
        service = {"host": "192.168.1.1", "port": 22, "service": "ssh"}
        
        with patch('paramiko.SSHClient') as mock_ssh:
            mock_ssh_client = Mock()
            mock_ssh_client.connect.return_value = None
            mock_ssh_client.close.return_value = None
            mock_ssh.return_value = mock_ssh_client
            
            vulnerability = await vuln_scanner._check_default_credentials(service)
            
            if vulnerability:
                assert vulnerability.name == "Default Credentials"
                assert vulnerability.severity == VulnerabilitySeverity.CRITICAL


class TestWebApplicationScanner:
    """Test WebApplicationScanner functionality"""
    
    @pytest.fixture
    def web_scanner(self):
        """Web application scanner fixture"""
        return WebApplicationScanner()
    
    @pytest.mark.asyncio
    async def test_web_application_scan(self, web_scanner):
        """Test web application scanning"""
        target = "https://example.com"
        
        with patch.object(web_scanner, '_crawl_website', return_value=["/"]):
            with patch.object(web_scanner, '_test_authentication_bypass', return_value=None):
                with patch.object(web_scanner, '_test_session_management', return_value=None):
                    with patch.object(web_scanner, '_test_input_validation', return_value=[]):
                        with patch.object(web_scanner, '_test_security_headers', return_value=[]):
                            
                            vulnerabilities = await web_scanner.web_application_scan(target)
                            
                            assert isinstance(vulnerabilities, list)
    
    @pytest.mark.asyncio
    async def test_crawl_website(self, web_scanner):
        """Test website crawling"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body><a href="/page1">Link</a></body></html>'
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            urls = await web_scanner._crawl_website("https://example.com")
            
            assert isinstance(urls, list)
            assert len(urls) > 0
    
    @pytest.mark.asyncio
    async def test_security_headers_check(self, web_scanner):
        """Test security headers checking"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}  # Missing security headers
            mock_get.return_value = mock_response
            
            vulnerabilities = await web_scanner._test_security_headers("https://example.com")
            
            assert isinstance(vulnerabilities, list)
            # Should find missing security headers
            assert len(vulnerabilities) > 0


class TestSSLScanner:
    """Test SSLScanner functionality"""
    
    @pytest.fixture
    def ssl_scanner(self):
        """SSL scanner fixture"""
        return SSLScanner()
    
    @pytest.mark.asyncio
    async def test_ssl_scan_success(self, ssl_scanner):
        """Test successful SSL scan"""
        with patch('socket.create_connection') as mock_create_conn:
            with patch('ssl.create_default_context') as mock_context:
                mock_cert = {
                    'subject': (('CN', 'example.com'),),
                    'issuer': (('CN', 'Test CA'),),
                    'version': 3,
                    'serialNumber': '12345',
                    'notBefore': 'Jan 1 00:00:00 2023 GMT',
                    'notAfter': 'Jan 1 00:00:00 2024 GMT',
                    'subjectAltName': (('DNS', 'www.example.com'),)
                }
                
                mock_cipher = ('TLS_AES_256_GCM_SHA384', 'TLSv1.3', 256)
                
                mock_ssl_sock = Mock()
                mock_ssl_sock.getpeercert.return_value = mock_cert
                mock_ssl_sock.cipher.return_value = mock_cipher
                mock_ssl_sock.version.return_value = 'TLSv1.3'
                
                mock_ctx = Mock()
                mock_ctx.wrap_socket.return_value.__enter__.return_value = mock_ssl_sock
                mock_context.return_value = mock_ctx
                
                mock_sock = Mock()
                mock_create_conn.return_value.__enter__.return_value = mock_sock
                
                vulnerabilities = await ssl_scanner.ssl_scan("example.com", 443)
                
                assert isinstance(vulnerabilities, list)
    
    @pytest.mark.asyncio
    async def test_ssl_scan_weak_cipher(self, ssl_scanner):
        """Test SSL scan detecting weak cipher"""
        with patch('socket.create_connection') as mock_create_conn:
            with patch('ssl.create_default_context') as mock_context:
                mock_cert = {
                    'subject': (('CN', 'example.com'),),
                    'issuer': (('CN', 'Test CA'),),
                    'version': 3,
                    'serialNumber': '12345',
                    'notBefore': 'Jan 1 00:00:00 2023 GMT',
                    'notAfter': 'Jan 1 00:00:00 2024 GMT'
                }
                
                # Weak cipher
                mock_cipher = ('RC4-MD5', 'TLSv1.0', 128)
                
                mock_ssl_sock = Mock()
                mock_ssl_sock.getpeercert.return_value = mock_cert
                mock_ssl_sock.cipher.return_value = mock_cipher
                mock_ssl_sock.version.return_value = 'TLSv1.0'
                
                mock_ctx = Mock()
                mock_ctx.wrap_socket.return_value.__enter__.return_value = mock_ssl_sock
                mock_context.return_value = mock_ctx
                
                mock_sock = Mock()
                mock_create_conn.return_value.__enter__.return_value = mock_sock
                
                vulnerabilities = await ssl_scanner.ssl_scan("example.com", 443)
                
                # Should detect weak cipher and protocol
                assert len(vulnerabilities) > 0
                weak_cipher_vuln = next((v for v in vulnerabilities if "weak cipher" in v.name.lower()), None)
                assert weak_cipher_vuln is not None
    
    @pytest.mark.asyncio
    async def test_ssl_scan_expired_certificate(self, ssl_scanner):
        """Test SSL scan detecting expired certificate"""
        with patch('socket.create_connection') as mock_create_conn:
            with patch('ssl.create_default_context') as mock_context:
                mock_cert = {
                    'subject': (('CN', 'example.com'),),
                    'issuer': (('CN', 'Test CA'),),
                    'version': 3,
                    'serialNumber': '12345',
                    'notBefore': 'Jan 1 00:00:00 2020 GMT',
                    'notAfter': 'Jan 1 00:00:00 2021 GMT'  # Expired
                }
                
                mock_cipher = ('TLS_AES_256_GCM_SHA384', 'TLSv1.3', 256)
                
                mock_ssl_sock = Mock()
                mock_ssl_sock.getpeercert.return_value = mock_cert
                mock_ssl_sock.cipher.return_value = mock_cipher
                mock_ssl_sock.version.return_value = 'TLSv1.3'
                
                mock_ctx = Mock()
                mock_ctx.wrap_socket.return_value.__enter__.return_value = mock_ssl_sock
                mock_context.return_value = mock_ctx
                
                mock_sock = Mock()
                mock_create_conn.return_value.__enter__.return_value = mock_sock
                
                vulnerabilities = await ssl_scanner.ssl_scan("example.com", 443)
                
                # Should detect expired certificate
                assert len(vulnerabilities) > 0
                expired_cert_vuln = next((v for v in vulnerabilities if "expired" in v.name.lower()), None)
                assert expired_cert_vuln is not None


class TestComplianceScanner:
    """Test ComplianceScanner functionality"""
    
    @pytest.fixture
    def compliance_scanner(self):
        """Compliance scanner fixture"""
        return ComplianceScanner()
    
    @pytest.mark.asyncio
    async def test_compliance_scan_nist(self, compliance_scanner):
        """Test NIST compliance scan"""
        scan_results = ScanResult(
            target="192.168.1.1",
            scan_type=ScanType.VULNERABILITY_SCAN,
            timestamp=datetime.utcnow(),
            duration=timedelta(minutes=5),
            status="completed",
            vulnerabilities=[],
            services=[]
        )
        
        findings = await compliance_scanner.compliance_scan(scan_results, "NIST")
        
        assert isinstance(findings, list)
    
    @pytest.mark.asyncio
    async def test_compliance_scan_owasp(self, compliance_scanner):
        """Test OWASP compliance scan"""
        vulnerability = Vulnerability(
            vuln_id="VULN-001",
            name="SQL Injection",
            description="SQL injection vulnerability",
            severity=VulnerabilitySeverity.HIGH,
            cvss_score=8.5,
            cve_id="CVE-2023-1234",
            affected_service="HTTP/80",
            evidence={},
            remediation="Use parameterized queries"
        )
        
        scan_results = ScanResult(
            target="192.168.1.1",
            scan_type=ScanType.WEB_APPLICATION_SCAN,
            timestamp=datetime.utcnow(),
            duration=timedelta(minutes=10),
            status="completed",
            vulnerabilities=[vulnerability],
            services=[]
        )
        
        findings = await compliance_scanner.compliance_scan(scan_results, "OWASP")
        
        assert isinstance(findings, list)
        assert len(findings) > 0


class TestScanningModule:
    """Test main scanning module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def scanning_module(self, config):
        """Scanning module fixture"""
        return ScanningModule(config)
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, scanning_module):
        """Test module initialization"""
        assert scanning_module.module_type.value == "scanning"
        assert scanning_module.status.value == "initialized"
        assert scanning_module.version == "1.0.0"
        assert len(scanning_module.results) == 0
    
    @pytest.mark.asyncio
    async def test_module_start_stop(self, scanning_module):
        """Test module start and stop"""
        # Test start
        success = await scanning_module.start()
        assert success is True
        assert scanning_module.status.value == "running"
        
        # Test stop
        success = await scanning_module.stop()
        assert success is True
        assert scanning_module.status.value == "stopped"
    
    @pytest.mark.asyncio
    async def test_execute_scan_port_scan(self, scanning_module):
        """Test port scan execution"""
        target = ScanTarget(
            target="192.168.1.1",
            scan_type=ScanType.PORT_SCAN,
            ports="80,443"
        )
        
        # Mock network scanner
        with patch.object(scanning_module.network_scanner, 'port_scan', return_value={"hosts": {}}):
            result = await scanning_module.execute_scan(target)
            
            assert isinstance(result, ScanResult)
            assert result.target == "192.168.1.1"
            assert result.scan_type == ScanType.PORT_SCAN
            assert result.status == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_scan_vulnerability_scan(self, scanning_module):
        """Test vulnerability scan execution"""
        target = ScanTarget(
            target="192.168.1.1",
            scan_type=ScanType.VULNERABILITY_SCAN
        )
        
        # Mock scanners
        with patch.object(scanning_module.network_scanner, 'service_detection', return_value=[]):
            with patch.object(scanning_module.vulnerability_scanner, 'vulnerability_scan', return_value=[]):
                result = await scanning_module.execute_scan(target)
                
                assert isinstance(result, ScanResult)
                assert result.target == "192.168.1.1"
                assert result.scan_type == ScanType.VULNERABILITY_SCAN
                assert result.status == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_scan_web_application_scan(self, scanning_module):
        """Test web application scan execution"""
        target = ScanTarget(
            target="https://example.com",
            scan_type=ScanType.WEB_APPLICATION_SCAN
        )
        
        # Mock web scanner
        with patch.object(scanning_module.web_scanner, 'web_application_scan', return_value=[]):
            result = await scanning_module.execute_scan(target)
            
            assert isinstance(result, ScanResult)
            assert result.target == "https://example.com"
            assert result.scan_type == ScanType.WEB_APPLICATION_SCAN
            assert result.status == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_scan_ssl_scan(self, scanning_module):
        """Test SSL scan execution"""
        target = ScanTarget(
            target="example.com",
            scan_type=ScanType.SSL_SCAN
        )
        
        # Mock SSL scanner
        with patch.object(scanning_module.ssl_scanner, 'ssl_scan', return_value=[]):
            result = await scanning_module.execute_scan(target)
            
            assert isinstance(result, ScanResult)
            assert result.target == "example.com"
            assert result.scan_type == ScanType.SSL_SCAN
            assert result.status == "completed"
    
    @pytest.mark.asyncio
    async def test_bulk_scan(self, scanning_module):
        """Test bulk scanning"""
        targets = [
            ScanTarget("192.168.1.1", ScanType.PORT_SCAN),
            ScanTarget("192.168.1.2", ScanType.PORT_SCAN)
        ]
        
        # Mock execute_scan
        with patch.object(scanning_module, 'execute_scan') as mock_execute:
            mock_result = ScanResult(
                target="test",
                scan_type=ScanType.PORT_SCAN,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=1),
                status="completed",
                vulnerabilities=[],
                services=[]
            )
            mock_execute.return_value = mock_result
            
            results = await scanning_module.bulk_scan(targets)
            
            assert isinstance(results, list)
            assert len(results) == 2
    
    def test_get_results_filtering(self, scanning_module):
        """Test result filtering"""
        # Add test results
        result1 = ScanResult(
            target="192.168.1.1",
            scan_type=ScanType.PORT_SCAN,
            timestamp=datetime.utcnow(),
            duration=timedelta(seconds=30),
            status="completed",
            vulnerabilities=[],
            services=[]
        )
        
        result2 = ScanResult(
            target="192.168.1.2",
            scan_type=ScanType.VULNERABILITY_SCAN,
            timestamp=datetime.utcnow(),
            duration=timedelta(minutes=5),
            status="completed",
            vulnerabilities=[],
            services=[]
        )
        
        scanning_module.results = [result1, result2]
        
        # Test filtering by target
        filtered = scanning_module.get_results(target="192.168.1.1")
        assert len(filtered) == 1
        assert filtered[0].target == "192.168.1.1"
        
        # Test filtering by scan type
        filtered = scanning_module.get_results(scan_type=ScanType.PORT_SCAN)
        assert len(filtered) == 1
        assert filtered[0].scan_type == ScanType.PORT_SCAN
        
        # Test no filtering
        filtered = scanning_module.get_results()
        assert len(filtered) == 2
    
    def test_export_results_json(self, scanning_module):
        """Test JSON export of results"""
        vulnerability = Vulnerability(
            vuln_id="VULN-001",
            name="Test Vulnerability",
            description="Test description",
            severity=VulnerabilitySeverity.HIGH,
            cvss_score=8.0,
            cve_id="CVE-2023-1234",
            affected_service="HTTP/80",
            evidence={},
            remediation="Test remediation"
        )
        
        result = ScanResult(
            target="192.168.1.1",
            scan_type=ScanType.VULNERABILITY_SCAN,
            timestamp=datetime.utcnow(),
            duration=timedelta(minutes=5),
            status="completed",
            vulnerabilities=[vulnerability],
            services=[]
        )
        
        scanning_module.results = [result]
        
        json_output = scanning_module.export_results("json")
        
        assert json_output != ""
        # Verify it's valid JSON
        parsed = json.loads(json_output)
        assert len(parsed) == 1
        assert parsed[0]["target"] == "192.168.1.1"
        assert parsed[0]["scan_type"] == "vulnerability_scan"
        assert len(parsed[0]["vulnerabilities"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_status(self, scanning_module):
        """Test module status reporting"""
        vulnerability = Vulnerability(
            vuln_id="VULN-001",
            name="Test Vulnerability",
            description="Test description",
            severity=VulnerabilitySeverity.HIGH,
            cvss_score=8.0,
            cve_id="CVE-2023-1234",
            affected_service="HTTP/80",
            evidence={},
            remediation="Test remediation"
        )
        
        result = ScanResult(
            target="192.168.1.1",
            scan_type=ScanType.VULNERABILITY_SCAN,
            timestamp=datetime.utcnow(),
            duration=timedelta(minutes=5),
            status="completed",
            vulnerabilities=[vulnerability],
            services=[]
        )
        
        scanning_module.results = [result]
        
        status = await scanning_module.get_status()
        
        assert status["module"] == "scanning"
        assert status["status"] == "initialized"
        assert status["version"] == "1.0.0"
        assert status["scans_completed"] == 1
        assert status["vulnerabilities_found"] == 1
        assert status["critical_vulnerabilities"] == 0
        assert status["high_vulnerabilities"] == 1


@pytest.mark.performance
class TestScanningPerformance:
    """Performance tests for scanning module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def scanning_module(self, config):
        """Scanning module fixture"""
        return ScanningModule(config)
    
    @pytest.mark.asyncio
    async def test_bulk_scan_performance(self, scanning_module, performance_monitor):
        """Test bulk scan performance"""
        # Create multiple targets
        targets = [
            ScanTarget(f"192.168.1.{i}", ScanType.PORT_SCAN)
            for i in range(1, 11)
        ]
        
        # Mock execute_scan
        with patch.object(scanning_module, 'execute_scan') as mock_execute:
            mock_result = ScanResult(
                target="test",
                scan_type=ScanType.PORT_SCAN,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=1),
                status="completed",
                vulnerabilities=[],
                services=[]
            )
            mock_execute.return_value = mock_result
            
            performance_monitor.start()
            
            results = await scanning_module.bulk_scan(targets)
            
            performance_monitor.stop()
            
            duration = performance_monitor.get_duration()
            assert duration is not None
            assert duration < 10.0  # Should complete within 10 seconds
            assert len(results) == 10


@pytest.mark.security
class TestScanningSecurity:
    """Security tests for scanning module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def scanning_module(self, config):
        """Scanning module fixture"""
        return ScanningModule(config)
    
    def test_input_validation_target(self, scanning_module):
        """Test input validation for scan targets"""
        # Test malicious targets
        malicious_targets = [
            "; rm -rf /",
            "../../..",
            "<script>alert('xss')</script>",
            "$(whoami)",
            "`id`"
        ]
        
        for target in malicious_targets:
            scan_target = ScanTarget(
                target=target,
                scan_type=ScanType.PORT_SCAN
            )
            
            # The module should handle these gracefully
            assert scan_target.target == target
    
    def test_port_range_validation(self, scanning_module):
        """Test port range validation"""
        # Test malicious port ranges
        malicious_ports = [
            "1-65535; rm -rf /",
            "../../../etc/passwd",
            "80,443,$(whoami)",
            "'; DROP TABLE ports; --"
        ]
        
        for ports in malicious_ports:
            scan_target = ScanTarget(
                target="192.168.1.1",
                scan_type=ScanType.PORT_SCAN,
                ports=ports
            )
            
            # Should handle malicious port ranges gracefully
            assert scan_target.ports == ports
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, scanning_module):
        """Test timeout handling in scans"""
        target = ScanTarget(
            target="192.168.1.1",
            scan_type=ScanType.PORT_SCAN,
            timeout=1  # Very short timeout
        )
        
        # Mock a slow scan
        with patch.object(scanning_module.network_scanner, 'port_scan') as mock_scan:
            mock_scan.side_effect = asyncio.TimeoutError("Scan timeout")
            
            result = await scanning_module.execute_scan(target)
            
            # Should handle timeout gracefully
            assert result.status == "failed"
    
    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(self, scanning_module):
        """Test resource limit enforcement"""
        # Test with many targets
        targets = [
            ScanTarget(f"192.168.1.{i}", ScanType.PORT_SCAN)
            for i in range(1, 101)  # 100 targets
        ]
        
        # Mock execute_scan to be fast
        with patch.object(scanning_module, 'execute_scan') as mock_execute:
            mock_result = ScanResult(
                target="test",
                scan_type=ScanType.PORT_SCAN,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=1),
                status="completed",
                vulnerabilities=[],
                services=[]
            )
            mock_execute.return_value = mock_result
            
            # Should handle many targets without resource exhaustion
            results = await scanning_module.bulk_scan(targets)
            
            assert isinstance(results, list)
            assert len(results) == 100