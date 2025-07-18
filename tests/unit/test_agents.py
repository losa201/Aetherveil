"""
Unit tests for Aetherveil Sentinel Agents
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

from aetherveil_sentinel.agents.base_agent import BaseAgent
from aetherveil_sentinel.agents.reconnaissance_agent import ReconnaissanceAgent
from aetherveil_sentinel.agents.scanner_agent import ScannerAgent
from aetherveil_sentinel.agents.osint_agent import OSINTAgent
from aetherveil_sentinel.agents.stealth_agent import StealthAgent
from aetherveil_sentinel.agents.exploiter_agent import ExploiterAgent


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    @pytest.fixture
    def base_agent(self):
        """Create a base agent for testing."""
        return BaseAgent(
            agent_id="test-agent-001",
            capabilities=["test", "mock"],
            config={"max_retries": 3, "timeout": 30}
        )
    
    def test_agent_initialization(self, base_agent):
        """Test agent initialization."""
        assert base_agent.agent_id == "test-agent-001"
        assert base_agent.capabilities == ["test", "mock"]
        assert base_agent.config["max_retries"] == 3
        assert base_agent.status == "initialized"
    
    @pytest.mark.asyncio
    async def test_agent_startup(self, base_agent):
        """Test agent startup process."""
        with patch.object(base_agent, '_initialize_resources', new_callable=AsyncMock) as mock_init:
            await base_agent.start()
            mock_init.assert_called_once()
            assert base_agent.status == "running"
    
    @pytest.mark.asyncio
    async def test_agent_shutdown(self, base_agent):
        """Test agent shutdown process."""
        base_agent.status = "running"
        with patch.object(base_agent, '_cleanup_resources', new_callable=AsyncMock) as mock_cleanup:
            await base_agent.stop()
            mock_cleanup.assert_called_once()
            assert base_agent.status == "stopped"
    
    @pytest.mark.asyncio
    async def test_task_execution(self, base_agent):
        """Test task execution with retries."""
        task = Mock()
        task.id = "test-task-001"
        task.type = "test"
        task.payload = {"target": "example.com"}
        
        with patch.object(base_agent, '_execute_task', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"status": "success", "result": "completed"}
            
            result = await base_agent.execute_task(task)
            
            assert result["status"] == "success"
            mock_exec.assert_called_once_with(task)
    
    @pytest.mark.asyncio
    async def test_task_execution_retry(self, base_agent):
        """Test task execution with retry logic."""
        task = Mock()
        task.id = "test-task-002"
        task.type = "test"
        task.payload = {"target": "example.com"}
        
        with patch.object(base_agent, '_execute_task', new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                Exception("Network error"),
                Exception("Timeout"),
                {"status": "success", "result": "completed on retry"}
            ]
            
            result = await base_agent.execute_task(task)
            
            assert result["status"] == "success"
            assert mock_exec.call_count == 3


class TestReconnaissanceAgent:
    """Test cases for ReconnaissanceAgent class."""
    
    @pytest.fixture
    def recon_agent(self):
        """Create a reconnaissance agent for testing."""
        return ReconnaissanceAgent(
            agent_id="recon-agent-001",
            capabilities=["dns_enumeration", "port_scanning", "subdomain_discovery"],
            config={"dns_servers": ["8.8.8.8", "1.1.1.1"], "port_range": "1-1000"}
        )
    
    @pytest.mark.asyncio
    async def test_dns_enumeration(self, recon_agent):
        """Test DNS enumeration capability."""
        target = "example.com"
        
        with patch('aetherveil_sentinel.modules.reconnaissance.dns_resolver') as mock_resolver:
            mock_resolver.query.return_value = [
                Mock(address="93.184.216.34"),
                Mock(address="2606:2800:220:1:248:1893:25c8:1946")
            ]
            
            result = await recon_agent.enumerate_dns(target)
            
            assert "dns_records" in result
            assert len(result["dns_records"]) > 0
    
    @pytest.mark.asyncio
    async def test_port_scanning(self, recon_agent):
        """Test port scanning capability."""
        target = "example.com"
        ports = [22, 80, 443]
        
        with patch('aetherveil_sentinel.modules.reconnaissance.port_scanner') as mock_scanner:
            mock_scanner.scan.return_value = {
                22: {"state": "closed", "service": "ssh"},
                80: {"state": "open", "service": "http"},
                443: {"state": "open", "service": "https"}
            }
            
            result = await recon_agent.scan_ports(target, ports)
            
            assert "open_ports" in result
            assert 80 in result["open_ports"]
            assert 443 in result["open_ports"]
    
    @pytest.mark.asyncio
    async def test_subdomain_discovery(self, recon_agent):
        """Test subdomain discovery capability."""
        target = "example.com"
        
        with patch('aetherveil_sentinel.modules.reconnaissance.subdomain_finder') as mock_finder:
            mock_finder.discover.return_value = [
                "www.example.com",
                "api.example.com",
                "mail.example.com"
            ]
            
            result = await recon_agent.discover_subdomains(target)
            
            assert "subdomains" in result
            assert len(result["subdomains"]) == 3
            assert "www.example.com" in result["subdomains"]


class TestScannerAgent:
    """Test cases for ScannerAgent class."""
    
    @pytest.fixture
    def scanner_agent(self):
        """Create a scanner agent for testing."""
        return ScannerAgent(
            agent_id="scanner-agent-001",
            capabilities=["vulnerability_scanning", "web_scanning", "ssl_scanning"],
            config={"scan_intensity": "normal", "timeout": 300}
        )
    
    @pytest.mark.asyncio
    async def test_vulnerability_scanning(self, scanner_agent):
        """Test vulnerability scanning capability."""
        target = "example.com"
        
        with patch('aetherveil_sentinel.modules.scanning.vulnerability_scanner') as mock_scanner:
            mock_scanner.scan.return_value = {
                "vulnerabilities": [
                    {"cve": "CVE-2021-44228", "severity": "critical", "description": "Log4j RCE"},
                    {"cve": "CVE-2021-45046", "severity": "high", "description": "Log4j DoS"}
                ],
                "scan_duration": 45.2
            }
            
            result = await scanner_agent.scan_vulnerabilities(target)
            
            assert "vulnerabilities" in result
            assert len(result["vulnerabilities"]) == 2
            assert result["vulnerabilities"][0]["severity"] == "critical"
    
    @pytest.mark.asyncio
    async def test_web_scanning(self, scanner_agent):
        """Test web application scanning."""
        target = "https://example.com"
        
        with patch('aetherveil_sentinel.modules.scanning.web_scanner') as mock_scanner:
            mock_scanner.scan.return_value = {
                "findings": [
                    {"type": "xss", "risk": "medium", "location": "/search"},
                    {"type": "sql_injection", "risk": "high", "location": "/login"}
                ],
                "technologies": ["Apache", "PHP", "MySQL"]
            }
            
            result = await scanner_agent.scan_web_application(target)
            
            assert "findings" in result
            assert "technologies" in result
            assert len(result["findings"]) == 2


class TestOSINTAgent:
    """Test cases for OSINTAgent class."""
    
    @pytest.fixture
    def osint_agent(self):
        """Create an OSINT agent for testing."""
        return OSINTAgent(
            agent_id="osint-agent-001",
            capabilities=["threat_intelligence", "domain_intelligence", "social_media_intelligence"],
            config={"api_keys": {"shodan": "test-key", "virustotal": "test-key"}}
        )
    
    @pytest.mark.asyncio
    async def test_threat_intelligence_gathering(self, osint_agent):
        """Test threat intelligence gathering."""
        target = "example.com"
        
        with patch('aetherveil_sentinel.modules.osint.threat_intel_collector') as mock_collector:
            mock_collector.gather.return_value = {
                "reputation": {"score": 85, "category": "clean"},
                "threat_feeds": [
                    {"source": "threatfox", "iocs": [], "last_seen": "2024-01-01"}
                ]
            }
            
            result = await osint_agent.gather_threat_intelligence(target)
            
            assert "reputation" in result
            assert result["reputation"]["score"] == 85
    
    @pytest.mark.asyncio
    async def test_domain_intelligence(self, osint_agent):
        """Test domain intelligence gathering."""
        target = "example.com"
        
        with patch('aetherveil_sentinel.modules.osint.domain_intel_collector') as mock_collector:
            mock_collector.analyze.return_value = {
                "whois": {"registrar": "Example Registrar", "created": "1995-08-14"},
                "dns_history": [{"date": "2024-01-01", "records": []}],
                "certificates": [{"issuer": "Let's Encrypt", "valid_from": "2024-01-01"}]
            }
            
            result = await osint_agent.analyze_domain(target)
            
            assert "whois" in result
            assert "dns_history" in result
            assert "certificates" in result


class TestStealthAgent:
    """Test cases for StealthAgent class."""
    
    @pytest.fixture
    def stealth_agent(self):
        """Create a stealth agent for testing."""
        return StealthAgent(
            agent_id="stealth-agent-001",
            capabilities=["traffic_obfuscation", "proxy_chaining", "anti_detection"],
            config={"proxy_chains": ["tor", "vpn"], "randomize_headers": True}
        )
    
    @pytest.mark.asyncio
    async def test_traffic_obfuscation(self, stealth_agent):
        """Test traffic obfuscation capability."""
        request_data = {"url": "https://example.com", "method": "GET"}
        
        with patch('aetherveil_sentinel.modules.stealth.traffic_obfuscator') as mock_obfuscator:
            mock_obfuscator.obfuscate.return_value = {
                "obfuscated_request": request_data,
                "techniques": ["header_randomization", "timing_randomization"]
            }
            
            result = await stealth_agent.obfuscate_traffic(request_data)
            
            assert "obfuscated_request" in result
            assert "techniques" in result
    
    @pytest.mark.asyncio
    async def test_proxy_chaining(self, stealth_agent):
        """Test proxy chaining capability."""
        target = "example.com"
        
        with patch('aetherveil_sentinel.modules.stealth.proxy_chain_manager') as mock_manager:
            mock_manager.create_chain.return_value = {
                "chain": ["proxy1:8080", "proxy2:3128", "proxy3:1080"],
                "anonymity_level": "high"
            }
            
            result = await stealth_agent.create_proxy_chain(target)
            
            assert "chain" in result
            assert len(result["chain"]) == 3
            assert result["anonymity_level"] == "high"


class TestExploiterAgent:
    """Test cases for ExploiterAgent class."""
    
    @pytest.fixture
    def exploiter_agent(self):
        """Create an exploiter agent for testing."""
        return ExploiterAgent(
            agent_id="exploiter-agent-001",
            capabilities=["exploit_execution", "payload_generation", "privilege_escalation"],
            config={"safety_checks": True, "max_attempts": 1}
        )
    
    @pytest.mark.asyncio
    async def test_exploit_validation(self, exploiter_agent):
        """Test exploit validation before execution."""
        exploit_data = {
            "target": "example.com",
            "vulnerability": "CVE-2021-44228",
            "payload": "test_payload"
        }
        
        with patch('aetherveil_sentinel.modules.exploitation.exploit_validator') as mock_validator:
            mock_validator.validate.return_value = {
                "is_valid": True,
                "safety_score": 95,
                "risk_level": "low"
            }
            
            result = await exploiter_agent.validate_exploit(exploit_data)
            
            assert result["is_valid"] is True
            assert result["safety_score"] == 95
    
    @pytest.mark.asyncio
    async def test_payload_generation(self, exploiter_agent):
        """Test payload generation capability."""
        target_info = {
            "os": "linux",
            "architecture": "x64",
            "vulnerability": "buffer_overflow"
        }
        
        with patch('aetherveil_sentinel.modules.exploitation.payload_generator') as mock_generator:
            mock_generator.generate.return_value = {
                "payload": "encoded_payload_data",
                "encoder": "base64",
                "size": 1024
            }
            
            result = await exploiter_agent.generate_payload(target_info)
            
            assert "payload" in result
            assert "encoder" in result
            assert result["size"] == 1024


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for agent interactions."""
    
    @pytest.mark.asyncio
    async def test_agent_coordination(self):
        """Test coordination between multiple agents."""
        recon_agent = ReconnaissanceAgent("recon-001", ["dns_enumeration"], {})
        scanner_agent = ScannerAgent("scanner-001", ["vulnerability_scanning"], {})
        
        # Mock the coordination process
        with patch.object(recon_agent, 'enumerate_dns', new_callable=AsyncMock) as mock_recon:
            with patch.object(scanner_agent, 'scan_vulnerabilities', new_callable=AsyncMock) as mock_scan:
                mock_recon.return_value = {"dns_records": [{"ip": "93.184.216.34"}]}
                mock_scan.return_value = {"vulnerabilities": []}
                
                # Simulate coordinated reconnaissance followed by scanning
                recon_result = await recon_agent.enumerate_dns("example.com")
                scan_result = await scanner_agent.scan_vulnerabilities("93.184.216.34")
                
                assert "dns_records" in recon_result
                assert "vulnerabilities" in scan_result
    
    @pytest.mark.asyncio
    async def test_agent_failure_handling(self):
        """Test agent behavior during failures."""
        agent = BaseAgent("test-agent", ["test"], {"max_retries": 2})
        
        task = Mock()
        task.id = "failing-task"
        task.type = "test"
        task.payload = {}
        
        with patch.object(agent, '_execute_task', new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Persistent failure")
            
            with pytest.raises(Exception):
                await agent.execute_task(task)
            
            assert mock_exec.call_count == 3  # Initial + 2 retries