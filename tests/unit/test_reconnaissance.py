"""
Unit tests for the Reconnaissance module.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json
import dns.resolver
import dns.zone
import socket
import ssl
import requests
from typing import List, Dict, Any

from modules.reconnaissance import (
    ReconnaissanceModule,
    DNSReconnaissance,
    NetworkReconnaissance,
    WebReconnaissance,
    SSLReconnaissance,
    ReconTarget,
    ReconResult,
    ReconMode,
    TargetType
)
from config.config import AetherVeilConfig


class TestReconTarget:
    """Test ReconTarget data class"""
    
    def test_recon_target_creation(self):
        """Test ReconTarget creation with default values"""
        target = ReconTarget(
            target="example.com",
            target_type=TargetType.DOMAIN
        )
        
        assert target.target == "example.com"
        assert target.target_type == TargetType.DOMAIN
        assert target.mode == ReconMode.PASSIVE
        assert target.depth == 1
        assert target.timeout == 30
        assert target.metadata == {}
    
    def test_recon_target_with_custom_values(self):
        """Test ReconTarget creation with custom values"""
        metadata = {"scope": "external", "priority": "high"}
        target = ReconTarget(
            target="192.168.1.0/24",
            target_type=TargetType.IP_RANGE,
            mode=ReconMode.AGGRESSIVE,
            depth=3,
            timeout=60,
            metadata=metadata
        )
        
        assert target.target == "192.168.1.0/24"
        assert target.target_type == TargetType.IP_RANGE
        assert target.mode == ReconMode.AGGRESSIVE
        assert target.depth == 3
        assert target.timeout == 60
        assert target.metadata == metadata


class TestReconResult:
    """Test ReconResult data class"""
    
    def test_recon_result_creation(self):
        """Test ReconResult creation"""
        timestamp = datetime.utcnow()
        data = {"ip": "192.168.1.1", "port": 80}
        
        result = ReconResult(
            target="example.com",
            target_type=TargetType.DOMAIN,
            technique="port_scan",
            timestamp=timestamp,
            data=data,
            confidence=0.9,
            source="tcp_scan"
        )
        
        assert result.target == "example.com"
        assert result.target_type == TargetType.DOMAIN
        assert result.technique == "port_scan"
        assert result.timestamp == timestamp
        assert result.data == data
        assert result.confidence == 0.9
        assert result.source == "tcp_scan"
        assert result.metadata == {}


class TestDNSReconnaissance:
    """Test DNS reconnaissance functionality"""
    
    @pytest.fixture
    def dns_recon(self):
        """DNS reconnaissance fixture"""
        return DNSReconnaissance()
    
    @pytest.mark.asyncio
    async def test_dns_reconnaissance_success(self, dns_recon):
        """Test successful DNS reconnaissance"""
        with patch.object(dns_recon.resolver, 'resolve') as mock_resolve:
            # Mock DNS responses
            mock_a_record = Mock()
            mock_a_record.__str__ = Mock(return_value="192.168.1.1")
            mock_resolve.return_value = [mock_a_record]
            
            results = await dns_recon.dns_reconnaissance("example.com")
            
            assert len(results) > 0
            assert any(r.technique.startswith("dns_") for r in results)
            assert all(r.target == "example.com" for r in results)
            assert all(r.target_type == TargetType.DOMAIN for r in results)
            assert all(r.source == "dns_lookup" for r in results)
    
    @pytest.mark.asyncio
    async def test_dns_reconnaissance_no_records(self, dns_recon):
        """Test DNS reconnaissance with no records"""
        with patch.object(dns_recon.resolver, 'resolve') as mock_resolve:
            mock_resolve.side_effect = Exception("No records found")
            
            results = await dns_recon.dns_reconnaissance("nonexistent.com")
            
            assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_subdomain_enumeration(self, dns_recon):
        """Test subdomain enumeration"""
        with patch.object(dns_recon.resolver, 'resolve') as mock_resolve:
            # Mock successful resolution for www subdomain
            mock_a_record = Mock()
            mock_a_record.__str__ = Mock(return_value="192.168.1.1")
            
            def mock_resolve_side_effect(domain, record_type):
                if domain == "www.example.com":
                    return [mock_a_record]
                else:
                    raise Exception("No records found")
            
            mock_resolve.side_effect = mock_resolve_side_effect
            
            results = await dns_recon.enumerate_subdomains("example.com")
            
            # Should find www.example.com
            assert len(results) >= 1
            found_www = any(r.target == "www.example.com" for r in results)
            assert found_www
    
    @pytest.mark.asyncio
    async def test_subdomain_enumeration_custom_wordlist(self, dns_recon):
        """Test subdomain enumeration with custom wordlist"""
        custom_wordlist = ["api", "dev", "staging"]
        
        with patch.object(dns_recon.resolver, 'resolve') as mock_resolve:
            mock_a_record = Mock()
            mock_a_record.__str__ = Mock(return_value="192.168.1.1")
            
            def mock_resolve_side_effect(domain, record_type):
                if domain == "api.example.com":
                    return [mock_a_record]
                else:
                    raise Exception("No records found")
            
            mock_resolve.side_effect = mock_resolve_side_effect
            
            results = await dns_recon.enumerate_subdomains("example.com", custom_wordlist)
            
            # Should find api.example.com
            assert len(results) >= 1
            found_api = any(r.target == "api.example.com" for r in results)
            assert found_api


class TestNetworkReconnaissance:
    """Test network reconnaissance functionality"""
    
    @pytest.fixture
    def network_recon(self):
        """Network reconnaissance fixture"""
        return NetworkReconnaissance()
    
    @pytest.mark.asyncio
    async def test_port_discovery_open_port(self, network_recon):
        """Test port discovery with open port"""
        with patch('socket.socket') as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 0  # Success
            mock_sock.close.return_value = None
            mock_socket.return_value = mock_sock
            
            # Mock banner grabbing
            with patch.object(network_recon, '_grab_banner', return_value="HTTP/1.1 200 OK"):
                results = await network_recon.port_discovery("192.168.1.1", (80, 80))
                
                assert len(results) == 1
                assert results[0].target == "192.168.1.1:80"
                assert results[0].data["port"] == 80
                assert results[0].data["state"] == "open"
                assert results[0].data["banner"] == "HTTP/1.1 200 OK"
    
    @pytest.mark.asyncio
    async def test_port_discovery_closed_port(self, network_recon):
        """Test port discovery with closed port"""
        with patch('socket.socket') as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 1  # Connection refused
            mock_sock.close.return_value = None
            mock_socket.return_value = mock_sock
            
            results = await network_recon.port_discovery("192.168.1.1", (80, 80))
            
            assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_banner_grabbing_http(self, network_recon):
        """Test banner grabbing for HTTP service"""
        with patch('socket.socket') as mock_socket:
            mock_sock = Mock()
            mock_sock.connect.return_value = None
            mock_sock.send.return_value = None
            mock_sock.recv.return_value = b"HTTP/1.1 200 OK\r\nServer: nginx/1.18.0\r\n"
            mock_sock.close.return_value = None
            mock_socket.return_value = mock_sock
            
            banner = await network_recon._grab_banner("192.168.1.1", 80)
            
            assert "HTTP/1.1 200 OK" in banner
            assert "nginx/1.18.0" in banner
    
    @pytest.mark.asyncio
    async def test_banner_grabbing_failure(self, network_recon):
        """Test banner grabbing failure"""
        with patch('socket.socket') as mock_socket:
            mock_socket.side_effect = Exception("Connection failed")
            
            banner = await network_recon._grab_banner("192.168.1.1", 80)
            
            assert banner is None
    
    @pytest.mark.asyncio
    async def test_network_discovery_success(self, network_recon):
        """Test network discovery with active hosts"""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=1.23 ms"
            mock_run.return_value = mock_result
            
            # Mock _extract_ping_time
            with patch.object(network_recon, '_extract_ping_time', return_value=1.23):
                results = await network_recon.network_discovery("192.168.1.1/30")
                
                assert len(results) > 0
                assert all(r.technique == "icmp_ping" for r in results)
                assert all(r.data["state"] == "alive" for r in results)
    
    @pytest.mark.asyncio
    async def test_network_discovery_no_hosts(self, network_recon):
        """Test network discovery with no active hosts"""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1  # Ping failed
            mock_run.return_value = mock_result
            
            results = await network_recon.network_discovery("192.168.1.1/30")
            
            assert len(results) == 0
    
    def test_extract_ping_time_success(self, network_recon):
        """Test ping time extraction"""
        ping_output = "64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=1.23 ms"
        
        time_ms = network_recon._extract_ping_time(ping_output)
        
        assert time_ms == 1.23
    
    def test_extract_ping_time_failure(self, network_recon):
        """Test ping time extraction failure"""
        ping_output = "ping: cannot resolve example.com"
        
        time_ms = network_recon._extract_ping_time(ping_output)
        
        assert time_ms is None


class TestWebReconnaissance:
    """Test web reconnaissance functionality"""
    
    @pytest.fixture
    def web_recon(self):
        """Web reconnaissance fixture"""
        return WebReconnaissance()
    
    @pytest.mark.asyncio
    async def test_web_reconnaissance_success(self, web_recon):
        """Test successful web reconnaissance"""
        with patch.object(web_recon.session, 'head') as mock_head:
            mock_response = Mock()
            mock_response.headers = {
                'Server': 'nginx/1.18.0',
                'Content-Type': 'text/html'
            }
            mock_response.status_code = 200
            mock_response.url = "https://example.com/"
            mock_head.return_value = mock_response
            
            # Mock technology detection
            with patch.object(web_recon, '_detect_technologies', return_value={'server': 'nginx'}):
                # Mock directory enumeration
                with patch.object(web_recon, '_enumerate_directories', return_value=[]):
                    results = await web_recon.web_reconnaissance("https://example.com")
                    
                    assert len(results) >= 1
                    header_result = next(r for r in results if r.technique == "http_headers")
                    assert header_result.data["status_code"] == 200
                    assert "Server" in header_result.data["headers"]
    
    @pytest.mark.asyncio
    async def test_technology_detection(self, web_recon):
        """Test technology detection"""
        mock_response = Mock()
        mock_response.headers = {
            'Server': 'nginx/1.18.0',
            'X-Powered-By': 'PHP/7.4.0'
        }
        mock_response.url = "https://example.com"
        
        with patch.object(web_recon.session, 'get') as mock_get:
            mock_get_response = Mock()
            mock_get_response.text = "<html>wp-content/themes/test</html>"
            mock_get.return_value = mock_get_response
            
            technologies = await web_recon._detect_technologies(mock_response)
            
            assert technologies['server'] == 'nginx/1.18.0'
            assert technologies['framework'] == 'PHP/7.4.0'
            assert technologies['cms'] == 'WordPress'
    
    @pytest.mark.asyncio
    async def test_directory_enumeration(self, web_recon):
        """Test directory enumeration"""
        with patch.object(web_recon.session, 'head') as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-length': '1024'}
            mock_head.return_value = mock_response
            
            directories = await web_recon._enumerate_directories("https://example.com")
            
            assert len(directories) > 0
            assert any(d['path'] == 'admin' for d in directories)
            assert any(d['status_code'] == 200 for d in directories)


class TestSSLReconnaissance:
    """Test SSL reconnaissance functionality"""
    
    @pytest.fixture
    def ssl_recon(self):
        """SSL reconnaissance fixture"""
        return SSLReconnaissance()
    
    @pytest.mark.asyncio
    async def test_ssl_reconnaissance_success(self, ssl_recon):
        """Test successful SSL reconnaissance"""
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
        
        with patch('socket.create_connection') as mock_create_conn:
            mock_sock = Mock()
            mock_ssl_sock = Mock()
            mock_ssl_sock.getpeercert.return_value = mock_cert
            mock_ssl_sock.cipher.return_value = mock_cipher
            
            with patch('ssl.create_default_context') as mock_context:
                mock_ctx = Mock()
                mock_ctx.wrap_socket.return_value.__enter__.return_value = mock_ssl_sock
                mock_context.return_value = mock_ctx
                
                mock_create_conn.return_value.__enter__.return_value = mock_sock
                
                results = await ssl_recon.ssl_reconnaissance("example.com")
                
                assert len(results) == 2
                cert_result = next(r for r in results if r.technique == "ssl_certificate_analysis")
                cipher_result = next(r for r in results if r.technique == "ssl_cipher_analysis")
                
                assert cert_result.data["subject"]["CN"] == "example.com"
                assert cipher_result.data["cipher_suite"] == "TLS_AES_256_GCM_SHA384"
    
    @pytest.mark.asyncio
    async def test_ssl_reconnaissance_failure(self, ssl_recon):
        """Test SSL reconnaissance failure"""
        with patch('socket.create_connection') as mock_create_conn:
            mock_create_conn.side_effect = Exception("Connection failed")
            
            results = await ssl_recon.ssl_reconnaissance("example.com")
            
            assert len(results) == 0


class TestReconnaissanceModule:
    """Test main reconnaissance module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def recon_module(self, config):
        """Reconnaissance module fixture"""
        return ReconnaissanceModule(config)
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, recon_module):
        """Test module initialization"""
        assert recon_module.module_type.value == "reconnaissance"
        assert recon_module.status.value == "initialized"
        assert recon_module.version == "1.0.0"
        assert len(recon_module.results) == 0
    
    @pytest.mark.asyncio
    async def test_module_start_stop(self, recon_module):
        """Test module start and stop"""
        # Test start
        success = await recon_module.start()
        assert success is True
        assert recon_module.status.value == "running"
        
        # Test stop
        success = await recon_module.stop()
        assert success is True
        assert recon_module.status.value == "stopped"
    
    @pytest.mark.asyncio
    async def test_execute_reconnaissance_domain(self, recon_module):
        """Test reconnaissance execution for domain target"""
        target = ReconTarget(
            target="example.com",
            target_type=TargetType.DOMAIN,
            mode=ReconMode.PASSIVE
        )
        
        # Mock all reconnaissance methods
        with patch.object(recon_module.dns_recon, 'dns_reconnaissance', return_value=[]):
            with patch.object(recon_module.ssl_recon, 'ssl_reconnaissance', return_value=[]):
                with patch.object(recon_module.web_recon, 'web_reconnaissance', return_value=[]):
                    results = await recon_module.execute_reconnaissance(target)
                    
                    assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_execute_reconnaissance_ip_range(self, recon_module):
        """Test reconnaissance execution for IP range target"""
        target = ReconTarget(
            target="192.168.1.0/24",
            target_type=TargetType.IP_RANGE,
            mode=ReconMode.ACTIVE
        )
        
        # Mock network discovery
        with patch.object(recon_module.network_recon, 'network_discovery', return_value=[]):
            results = await recon_module.execute_reconnaissance(target)
            
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_execute_reconnaissance_single_ip(self, recon_module):
        """Test reconnaissance execution for single IP target"""
        target = ReconTarget(
            target="192.168.1.1",
            target_type=TargetType.SINGLE_IP,
            mode=ReconMode.ACTIVE
        )
        
        # Mock port discovery
        with patch.object(recon_module.network_recon, 'port_discovery', return_value=[]):
            results = await recon_module.execute_reconnaissance(target)
            
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_execute_reconnaissance_url(self, recon_module):
        """Test reconnaissance execution for URL target"""
        target = ReconTarget(
            target="https://example.com",
            target_type=TargetType.URL,
            mode=ReconMode.PASSIVE
        )
        
        # Mock web reconnaissance
        with patch.object(recon_module.web_recon, 'web_reconnaissance', return_value=[]):
            results = await recon_module.execute_reconnaissance(target)
            
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_bulk_reconnaissance(self, recon_module):
        """Test bulk reconnaissance execution"""
        targets = [
            ReconTarget("example.com", TargetType.DOMAIN),
            ReconTarget("test.com", TargetType.DOMAIN)
        ]
        
        # Mock execute_reconnaissance
        with patch.object(recon_module, 'execute_reconnaissance', return_value=[]):
            results = await recon_module.bulk_reconnaissance(targets)
            
            assert isinstance(results, dict)
            assert len(results) == 2
            assert "example.com" in results
            assert "test.com" in results
    
    def test_get_results_filtering(self, recon_module):
        """Test result filtering"""
        # Add test results
        test_results = [
            ReconResult(
                target="example.com",
                target_type=TargetType.DOMAIN,
                technique="dns_lookup",
                timestamp=datetime.utcnow(),
                data={},
                confidence=1.0,
                source="dns"
            ),
            ReconResult(
                target="test.com",
                target_type=TargetType.DOMAIN,
                technique="port_scan",
                timestamp=datetime.utcnow(),
                data={},
                confidence=0.9,
                source="tcp"
            )
        ]
        recon_module.results = test_results
        
        # Test filtering by target
        filtered = recon_module.get_results(target="example.com")
        assert len(filtered) == 1
        assert filtered[0].target == "example.com"
        
        # Test filtering by technique
        filtered = recon_module.get_results(technique="dns_lookup")
        assert len(filtered) == 1
        assert filtered[0].technique == "dns_lookup"
        
        # Test no filtering
        filtered = recon_module.get_results()
        assert len(filtered) == 2
    
    def test_export_results_json(self, recon_module):
        """Test JSON export of results"""
        # Add test result
        test_result = ReconResult(
            target="example.com",
            target_type=TargetType.DOMAIN,
            technique="dns_lookup",
            timestamp=datetime.utcnow(),
            data={"ip": "192.168.1.1"},
            confidence=1.0,
            source="dns"
        )
        recon_module.results = [test_result]
        
        json_output = recon_module.export_results("json")
        
        assert json_output != ""
        # Verify it's valid JSON
        parsed = json.loads(json_output)
        assert len(parsed) == 1
        assert parsed[0]["target"] == "example.com"
        assert parsed[0]["technique"] == "dns_lookup"
    
    @pytest.mark.asyncio
    async def test_get_status(self, recon_module):
        """Test module status reporting"""
        # Add test result
        test_result = ReconResult(
            target="example.com",
            target_type=TargetType.DOMAIN,
            technique="dns_lookup",
            timestamp=datetime.utcnow(),
            data={},
            confidence=1.0,
            source="dns"
        )
        recon_module.results = [test_result]
        
        status = await recon_module.get_status()
        
        assert status["module"] == "reconnaissance"
        assert status["status"] == "initialized"
        assert status["version"] == "1.0.0"
        assert status["results_count"] == 1
        assert "dns_lookup" in status["techniques_used"]
        assert "example.com" in status["targets_scanned"]
    
    @pytest.mark.asyncio
    async def test_get_status_no_results(self, recon_module):
        """Test module status reporting with no results"""
        status = await recon_module.get_status()
        
        assert status["module"] == "reconnaissance"
        assert status["results_count"] == 0
        assert status["last_activity"] is None
        assert status["techniques_used"] == []
        assert status["targets_scanned"] == []


class TestReconnaissanceModuleIntegration:
    """Integration tests for reconnaissance module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def recon_module(self, config):
        """Reconnaissance module fixture"""
        return ReconnaissanceModule(config)
    
    @pytest.mark.asyncio
    async def test_domain_reconnaissance_workflow(self, recon_module):
        """Test complete domain reconnaissance workflow"""
        target = ReconTarget(
            target="example.com",
            target_type=TargetType.DOMAIN,
            mode=ReconMode.PASSIVE
        )
        
        # Mock all external calls
        with patch.object(recon_module.dns_recon.resolver, 'resolve'):
            with patch('socket.create_connection'):
                with patch.object(recon_module.web_recon.session, 'head'):
                    with patch.object(recon_module.web_recon, '_detect_technologies', return_value={}):
                        with patch.object(recon_module.web_recon, '_enumerate_directories', return_value=[]):
                            await recon_module.start()
                            results = await recon_module.execute_reconnaissance(target)
                            await recon_module.stop()
                            
                            assert isinstance(results, list)
                            assert recon_module.status.value == "stopped"
    
    @pytest.mark.asyncio
    async def test_network_reconnaissance_workflow(self, recon_module):
        """Test complete network reconnaissance workflow"""
        target = ReconTarget(
            target="192.168.1.0/28",
            target_type=TargetType.IP_RANGE,
            mode=ReconMode.ACTIVE
        )
        
        # Mock subprocess for ping
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "64 bytes from 192.168.1.1: time=1.23 ms"
            mock_run.return_value = mock_result
            
            await recon_module.start()
            results = await recon_module.execute_reconnaissance(target)
            await recon_module.stop()
            
            assert isinstance(results, list)
            assert recon_module.status.value == "stopped"


@pytest.mark.performance
class TestReconnaissancePerformance:
    """Performance tests for reconnaissance module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def recon_module(self, config):
        """Reconnaissance module fixture"""
        return ReconnaissanceModule(config)
    
    @pytest.mark.asyncio
    async def test_bulk_reconnaissance_performance(self, recon_module, performance_monitor):
        """Test bulk reconnaissance performance"""
        # Create multiple targets
        targets = [
            ReconTarget(f"test{i}.com", TargetType.DOMAIN)
            for i in range(10)
        ]
        
        # Mock all external calls
        with patch.object(recon_module, 'execute_reconnaissance', return_value=[]):
            performance_monitor.start()
            
            results = await recon_module.bulk_reconnaissance(targets)
            
            performance_monitor.stop()
            
            duration = performance_monitor.get_duration()
            assert duration is not None
            assert duration < 10.0  # Should complete within 10 seconds
            assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_port_scanning_performance(self, recon_module, performance_monitor):
        """Test concurrent port scanning performance"""
        target = ReconTarget(
            target="192.168.1.1",
            target_type=TargetType.SINGLE_IP,
            mode=ReconMode.ACTIVE
        )
        
        # Mock socket operations
        with patch('socket.socket') as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 1  # Closed
            mock_sock.close.return_value = None
            mock_socket.return_value = mock_sock
            
            performance_monitor.start()
            
            results = await recon_module.network_recon.port_discovery("192.168.1.1", (1, 100))
            
            performance_monitor.stop()
            
            duration = performance_monitor.get_duration()
            assert duration is not None
            assert duration < 5.0  # Should complete within 5 seconds


@pytest.mark.security
class TestReconnaissanceSecurity:
    """Security tests for reconnaissance module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def recon_module(self, config):
        """Reconnaissance module fixture"""
        return ReconnaissanceModule(config)
    
    def test_input_validation_domain(self, recon_module):
        """Test input validation for domain targets"""
        # Test malicious domain names
        malicious_domains = [
            "../../../etc/passwd",
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../..",
            "file:///etc/passwd"
        ]
        
        for domain in malicious_domains:
            target = ReconTarget(
                target=domain,
                target_type=TargetType.DOMAIN
            )
            
            # The module should handle these gracefully without crashing
            assert target.target == domain  # Input preserved for analysis
    
    def test_network_range_validation(self, recon_module):
        """Test network range validation"""
        # Test with oversized networks
        large_network = "0.0.0.0/0"  # Entire IPv4 space
        
        target = ReconTarget(
            target=large_network,
            target_type=TargetType.IP_RANGE
        )
        
        # Should handle large networks gracefully
        assert target.target == large_network
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, recon_module):
        """Test timeout handling in reconnaissance"""
        target = ReconTarget(
            target="slow-server.com",
            target_type=TargetType.DOMAIN,
            timeout=1  # Very short timeout
        )
        
        # Mock slow DNS resolution
        with patch.object(recon_module.dns_recon.resolver, 'resolve') as mock_resolve:
            mock_resolve.side_effect = Exception("Timeout")
            
            results = await recon_module.execute_reconnaissance(target)
            
            # Should handle timeout gracefully
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_protection(self, recon_module):
        """Test protection against resource exhaustion"""
        # Test with many targets
        targets = [
            ReconTarget(f"test{i}.com", TargetType.DOMAIN)
            for i in range(100)
        ]
        
        # Mock reconnaissance to be fast
        with patch.object(recon_module, 'execute_reconnaissance', return_value=[]):
            # Should handle many targets without issues
            results = await recon_module.bulk_reconnaissance(targets)
            
            assert isinstance(results, dict)
            assert len(results) == 100