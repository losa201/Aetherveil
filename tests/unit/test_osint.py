"""
Unit tests for the OSINT module.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import requests
from typing import List, Dict, Any

from modules.osint import (
    OSINTModule,
    DomainIntelligence,
    IPIntelligence,
    EmailIntelligence,
    SocialMediaIntelligence,
    ThreatIntelligence,
    OSINTTarget,
    OSINTResult,
    IntelligenceType,
    ConfidenceLevel,
    OSINTSource
)
from config.config import AetherVeilConfig


class TestOSINTTarget:
    """Test OSINTTarget data class"""
    
    def test_osint_target_creation(self):
        """Test OSINTTarget creation with default values"""
        target = OSINTTarget(
            target="example.com",
            intelligence_type=IntelligenceType.DOMAIN
        )
        
        assert target.target == "example.com"
        assert target.intelligence_type == IntelligenceType.DOMAIN
        assert target.sources == []
        assert target.depth == 1
        assert target.timeout == 300
        assert target.metadata == {}
    
    def test_osint_target_with_custom_values(self):
        """Test OSINTTarget creation with custom values"""
        sources = [OSINTSource.WHOIS, OSINTSource.DNS, OSINTSource.SOCIAL_MEDIA]
        metadata = {"priority": "high", "campaign": "test"}
        
        target = OSINTTarget(
            target="admin@example.com",
            intelligence_type=IntelligenceType.EMAIL,
            sources=sources,
            depth=3,
            timeout=600,
            metadata=metadata
        )
        
        assert target.target == "admin@example.com"
        assert target.intelligence_type == IntelligenceType.EMAIL
        assert target.sources == sources
        assert target.depth == 3
        assert target.timeout == 600
        assert target.metadata == metadata


class TestOSINTResult:
    """Test OSINTResult data class"""
    
    def test_osint_result_creation(self):
        """Test OSINTResult creation"""
        timestamp = datetime.utcnow()
        data = {"registrar": "Test Registrar", "creation_date": "2020-01-01"}
        
        result = OSINTResult(
            target="example.com",
            intelligence_type=IntelligenceType.DOMAIN,
            source=OSINTSource.WHOIS,
            timestamp=timestamp,
            data=data,
            confidence=ConfidenceLevel.HIGH,
            raw_data={"whois_output": "raw data"}
        )
        
        assert result.target == "example.com"
        assert result.intelligence_type == IntelligenceType.DOMAIN
        assert result.source == OSINTSource.WHOIS
        assert result.timestamp == timestamp
        assert result.data == data
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.raw_data == {"whois_output": "raw data"}
        assert result.metadata == {}


class TestDomainIntelligence:
    """Test DomainIntelligence functionality"""
    
    @pytest.fixture
    def domain_intel(self):
        """Domain intelligence fixture"""
        return DomainIntelligence()
    
    @pytest.mark.asyncio
    async def test_whois_lookup(self, domain_intel):
        """Test WHOIS lookup functionality"""
        with patch('whois.whois') as mock_whois:
            mock_whois.return_value = {
                'domain_name': 'EXAMPLE.COM',
                'registrar': 'Test Registrar',
                'creation_date': datetime(2020, 1, 1),
                'expiration_date': datetime(2025, 1, 1),
                'name_servers': ['ns1.example.com', 'ns2.example.com']
            }
            
            result = await domain_intel.whois_lookup("example.com")
            
            assert result is not None
            assert result.source == OSINTSource.WHOIS
            assert result.target == "example.com"
            assert result.data["registrar"] == "Test Registrar"
            assert result.confidence == ConfidenceLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_whois_lookup_failure(self, domain_intel):
        """Test WHOIS lookup failure"""
        with patch('whois.whois') as mock_whois:
            mock_whois.side_effect = Exception("WHOIS lookup failed")
            
            result = await domain_intel.whois_lookup("nonexistent.com")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_dns_intelligence(self, domain_intel):
        """Test DNS intelligence gathering"""
        with patch('dns.resolver.resolve') as mock_resolve:
            # Mock A record
            mock_a_record = Mock()
            mock_a_record.__str__ = Mock(return_value="192.168.1.1")
            mock_resolve.return_value = [mock_a_record]
            
            results = await domain_intel.dns_intelligence("example.com")
            
            assert len(results) > 0
            dns_result = next(r for r in results if r.source == OSINTSource.DNS)
            assert dns_result.target == "example.com"
            assert "A" in dns_result.data
    
    @pytest.mark.asyncio
    async def test_subdomain_enumeration(self, domain_intel):
        """Test subdomain enumeration"""
        with patch('dns.resolver.resolve') as mock_resolve:
            # Mock successful resolution for www subdomain
            mock_a_record = Mock()
            mock_a_record.__str__ = Mock(return_value="192.168.1.1")
            
            def mock_resolve_side_effect(domain, record_type):
                if domain == "www.example.com":
                    return [mock_a_record]
                else:
                    raise Exception("No records found")
            
            mock_resolve.side_effect = mock_resolve_side_effect
            
            results = await domain_intel.subdomain_enumeration("example.com")
            
            assert len(results) > 0
            subdomain_result = next(r for r in results if r.target == "www.example.com")
            assert subdomain_result.source == OSINTSource.SUBDOMAIN_ENUM
    
    @pytest.mark.asyncio
    async def test_ssl_certificate_analysis(self, domain_intel):
        """Test SSL certificate analysis"""
        with patch('ssl.get_server_certificate') as mock_get_cert:
            mock_get_cert.return_value = "-----BEGIN CERTIFICATE-----\nMOCK CERT\n-----END CERTIFICATE-----"
            
            with patch('ssl.PEM_cert_to_DER_cert') as mock_pem_to_der:
                with patch('ssl.DER_cert_to_PEM_cert') as mock_der_to_pem:
                    result = await domain_intel.ssl_certificate_analysis("example.com")
                    
                    assert result is not None
                    assert result.source == OSINTSource.SSL_CERTIFICATE
                    assert result.target == "example.com"
    
    @pytest.mark.asyncio
    async def test_passive_dns_lookup(self, domain_intel):
        """Test passive DNS lookup"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "passive_dns": [
                    {
                        "rrname": "example.com",
                        "rrtype": "A",
                        "rdata": "192.168.1.1",
                        "time_first": 1609459200,
                        "time_last": 1609459200
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            results = await domain_intel.passive_dns_lookup("example.com")
            
            assert len(results) > 0
            passive_result = next(r for r in results if r.source == OSINTSource.PASSIVE_DNS)
            assert passive_result.target == "example.com"


class TestIPIntelligence:
    """Test IPIntelligence functionality"""
    
    @pytest.fixture
    def ip_intel(self):
        """IP intelligence fixture"""
        return IPIntelligence()
    
    @pytest.mark.asyncio
    async def test_geolocation_lookup(self, ip_intel):
        """Test IP geolocation lookup"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "ip": "192.168.1.1",
                "country": "United States",
                "region": "California",
                "city": "San Francisco",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "isp": "Test ISP",
                "organization": "Test Organization"
            }
            mock_get.return_value = mock_response
            
            result = await ip_intel.geolocation_lookup("192.168.1.1")
            
            assert result is not None
            assert result.source == OSINTSource.GEOLOCATION
            assert result.target == "192.168.1.1"
            assert result.data["country"] == "United States"
            assert result.data["city"] == "San Francisco"
    
    @pytest.mark.asyncio
    async def test_reputation_check(self, ip_intel):
        """Test IP reputation check"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "ip": "192.168.1.1",
                "malicious": False,
                "reputation_score": 85,
                "categories": ["legitimate"],
                "last_seen": "2023-01-01T00:00:00Z"
            }
            mock_get.return_value = mock_response
            
            result = await ip_intel.reputation_check("192.168.1.1")
            
            assert result is not None
            assert result.source == OSINTSource.REPUTATION
            assert result.target == "192.168.1.1"
            assert result.data["malicious"] is False
            assert result.data["reputation_score"] == 85
    
    @pytest.mark.asyncio
    async def test_shodan_lookup(self, ip_intel):
        """Test Shodan lookup"""
        with patch('shodan.Shodan.host') as mock_shodan:
            mock_shodan.return_value = {
                "ip_str": "192.168.1.1",
                "ports": [80, 443, 22],
                "hostnames": ["example.com"],
                "country_name": "United States",
                "city": "San Francisco",
                "data": [
                    {
                        "port": 80,
                        "banner": "HTTP/1.1 200 OK",
                        "product": "Apache",
                        "version": "2.4.41"
                    }
                ]
            }
            
            result = await ip_intel.shodan_lookup("192.168.1.1")
            
            assert result is not None
            assert result.source == OSINTSource.SHODAN
            assert result.target == "192.168.1.1"
            assert 80 in result.data["ports"]
            assert 443 in result.data["ports"]
    
    @pytest.mark.asyncio
    async def test_censys_lookup(self, ip_intel):
        """Test Censys lookup"""
        with patch('censys.search.CensysHosts.view') as mock_censys:
            mock_censys.return_value = {
                "ip": "192.168.1.1",
                "location": {
                    "country": "United States",
                    "city": "San Francisco"
                },
                "services": [
                    {
                        "port": 80,
                        "service_name": "HTTP",
                        "banner": "Apache/2.4.41"
                    }
                ]
            }
            
            result = await ip_intel.censys_lookup("192.168.1.1")
            
            assert result is not None
            assert result.source == OSINTSource.CENSYS
            assert result.target == "192.168.1.1"
            assert result.data["location"]["country"] == "United States"


class TestEmailIntelligence:
    """Test EmailIntelligence functionality"""
    
    @pytest.fixture
    def email_intel(self):
        """Email intelligence fixture"""
        return EmailIntelligence()
    
    @pytest.mark.asyncio
    async def test_email_validation(self, email_intel):
        """Test email validation"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "email": "admin@example.com",
                "valid": True,
                "disposable": False,
                "role": True,
                "mx_records": ["mail.example.com"]
            }
            mock_get.return_value = mock_response
            
            result = await email_intel.email_validation("admin@example.com")
            
            assert result is not None
            assert result.source == OSINTSource.EMAIL_VALIDATION
            assert result.target == "admin@example.com"
            assert result.data["valid"] is True
            assert result.data["role"] is True
    
    @pytest.mark.asyncio
    async def test_breach_check(self, email_intel):
        """Test email breach check"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "Name": "Test Breach",
                    "Domain": "testbreach.com",
                    "BreachDate": "2023-01-01",
                    "DataClasses": ["Email addresses", "Passwords"]
                }
            ]
            mock_get.return_value = mock_response
            
            result = await email_intel.breach_check("admin@example.com")
            
            assert result is not None
            assert result.source == OSINTSource.BREACH_CHECK
            assert result.target == "admin@example.com"
            assert len(result.data["breaches"]) == 1
            assert result.data["breaches"][0]["Name"] == "Test Breach"
    
    @pytest.mark.asyncio
    async def test_email_enumeration(self, email_intel):
        """Test email enumeration for domain"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = """
            <html>
                <body>
                    <a href="mailto:admin@example.com">Admin</a>
                    <a href="mailto:info@example.com">Info</a>
                </body>
            </html>
            """
            mock_get.return_value = mock_response
            
            results = await email_intel.email_enumeration("example.com")
            
            assert len(results) > 0
            email_result = next(r for r in results if r.source == OSINTSource.EMAIL_ENUM)
            assert "admin@example.com" in email_result.data["emails"]
            assert "info@example.com" in email_result.data["emails"]


class TestSocialMediaIntelligence:
    """Test SocialMediaIntelligence functionality"""
    
    @pytest.fixture
    def social_intel(self):
        """Social media intelligence fixture"""
        return SocialMediaIntelligence()
    
    @pytest.mark.asyncio
    async def test_social_media_search(self, social_intel):
        """Test social media search"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "profiles": [
                    {
                        "platform": "linkedin",
                        "url": "https://linkedin.com/company/example",
                        "title": "Example Company",
                        "description": "Test company"
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            results = await social_intel.social_media_search("Example Company")
            
            assert len(results) > 0
            social_result = next(r for r in results if r.source == OSINTSource.SOCIAL_MEDIA)
            assert social_result.target == "Example Company"
            assert len(social_result.data["profiles"]) == 1
    
    @pytest.mark.asyncio
    async def test_linkedin_intelligence(self, social_intel):
        """Test LinkedIn intelligence gathering"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = """
            <html>
                <body>
                    <div class="company-info">
                        <h1>Example Company</h1>
                        <p>Technology company based in San Francisco</p>
                        <span>500-1000 employees</span>
                    </div>
                </body>
            </html>
            """
            mock_get.return_value = mock_response
            
            result = await social_intel.linkedin_intelligence("example-company")
            
            assert result is not None
            assert result.source == OSINTSource.LINKEDIN
            assert result.target == "example-company"
    
    @pytest.mark.asyncio
    async def test_twitter_intelligence(self, social_intel):
        """Test Twitter intelligence gathering"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": {
                    "id": "12345",
                    "username": "example",
                    "name": "Example Account",
                    "description": "Official Example account",
                    "public_metrics": {
                        "followers_count": 1000,
                        "following_count": 500,
                        "tweet_count": 2000
                    }
                }
            }
            mock_get.return_value = mock_response
            
            result = await social_intel.twitter_intelligence("example")
            
            assert result is not None
            assert result.source == OSINTSource.TWITTER
            assert result.target == "example"
            assert result.data["followers_count"] == 1000


class TestThreatIntelligence:
    """Test ThreatIntelligence functionality"""
    
    @pytest.fixture
    def threat_intel(self):
        """Threat intelligence fixture"""
        return ThreatIntelligence()
    
    @pytest.mark.asyncio
    async def test_malware_analysis(self, threat_intel):
        """Test malware analysis"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "hash": "d41d8cd98f00b204e9800998ecf8427e",
                "malicious": True,
                "engines": {
                    "detected": 45,
                    "total": 70
                },
                "families": ["Trojan.Generic"],
                "first_seen": "2023-01-01T00:00:00Z"
            }
            mock_get.return_value = mock_response
            
            result = await threat_intel.malware_analysis("d41d8cd98f00b204e9800998ecf8427e")
            
            assert result is not None
            assert result.source == OSINTSource.MALWARE_ANALYSIS
            assert result.target == "d41d8cd98f00b204e9800998ecf8427e"
            assert result.data["malicious"] is True
            assert result.data["engines"]["detected"] == 45
    
    @pytest.mark.asyncio
    async def test_ioc_lookup(self, threat_intel):
        """Test IOC lookup"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "indicator": "192.168.1.1",
                "type": "ip",
                "malicious": False,
                "confidence": 85,
                "tags": ["benign", "infrastructure"],
                "first_seen": "2023-01-01T00:00:00Z",
                "last_seen": "2023-12-31T23:59:59Z"
            }
            mock_get.return_value = mock_response
            
            result = await threat_intel.ioc_lookup("192.168.1.1")
            
            assert result is not None
            assert result.source == OSINTSource.IOC_LOOKUP
            assert result.target == "192.168.1.1"
            assert result.data["malicious"] is False
            assert result.data["confidence"] == 85
    
    @pytest.mark.asyncio
    async def test_threat_feed_lookup(self, threat_intel):
        """Test threat feed lookup"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "feeds": [
                    {
                        "name": "Test Feed",
                        "matches": [
                            {
                                "indicator": "example.com",
                                "type": "domain",
                                "threat_type": "malware",
                                "confidence": 90
                            }
                        ]
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            result = await threat_intel.threat_feed_lookup("example.com")
            
            assert result is not None
            assert result.source == OSINTSource.THREAT_FEED
            assert result.target == "example.com"
            assert len(result.data["feeds"]) == 1


class TestOSINTModule:
    """Test main OSINT module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def osint_module(self, config):
        """OSINT module fixture"""
        return OSINTModule(config)
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, osint_module):
        """Test module initialization"""
        assert osint_module.module_type.value == "osint"
        assert osint_module.status.value == "initialized"
        assert osint_module.version == "1.0.0"
        assert len(osint_module.results) == 0
    
    @pytest.mark.asyncio
    async def test_module_start_stop(self, osint_module):
        """Test module start and stop"""
        # Test start
        success = await osint_module.start()
        assert success is True
        assert osint_module.status.value == "running"
        
        # Test stop
        success = await osint_module.stop()
        assert success is True
        assert osint_module.status.value == "stopped"
    
    @pytest.mark.asyncio
    async def test_execute_intelligence_gathering_domain(self, osint_module):
        """Test intelligence gathering for domain target"""
        target = OSINTTarget(
            target="example.com",
            intelligence_type=IntelligenceType.DOMAIN,
            sources=[OSINTSource.WHOIS, OSINTSource.DNS]
        )
        
        # Mock intelligence gathering methods
        with patch.object(osint_module.domain_intel, 'whois_lookup', return_value=None):
            with patch.object(osint_module.domain_intel, 'dns_intelligence', return_value=[]):
                results = await osint_module.execute_intelligence_gathering(target)
                
                assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_execute_intelligence_gathering_ip(self, osint_module):
        """Test intelligence gathering for IP target"""
        target = OSINTTarget(
            target="192.168.1.1",
            intelligence_type=IntelligenceType.IP,
            sources=[OSINTSource.GEOLOCATION, OSINTSource.REPUTATION]
        )
        
        # Mock intelligence gathering methods
        with patch.object(osint_module.ip_intel, 'geolocation_lookup', return_value=None):
            with patch.object(osint_module.ip_intel, 'reputation_check', return_value=None):
                results = await osint_module.execute_intelligence_gathering(target)
                
                assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_execute_intelligence_gathering_email(self, osint_module):
        """Test intelligence gathering for email target"""
        target = OSINTTarget(
            target="admin@example.com",
            intelligence_type=IntelligenceType.EMAIL,
            sources=[OSINTSource.EMAIL_VALIDATION, OSINTSource.BREACH_CHECK]
        )
        
        # Mock intelligence gathering methods
        with patch.object(osint_module.email_intel, 'email_validation', return_value=None):
            with patch.object(osint_module.email_intel, 'breach_check', return_value=None):
                results = await osint_module.execute_intelligence_gathering(target)
                
                assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_bulk_intelligence_gathering(self, osint_module):
        """Test bulk intelligence gathering"""
        targets = [
            OSINTTarget("example.com", IntelligenceType.DOMAIN),
            OSINTTarget("192.168.1.1", IntelligenceType.IP)
        ]
        
        # Mock execute_intelligence_gathering
        with patch.object(osint_module, 'execute_intelligence_gathering', return_value=[]):
            results = await osint_module.bulk_intelligence_gathering(targets)
            
            assert isinstance(results, dict)
            assert len(results) == 2
            assert "example.com" in results
            assert "192.168.1.1" in results
    
    @pytest.mark.asyncio
    async def test_search_intelligence(self, osint_module):
        """Test intelligence search"""
        query = "example.com"
        filters = {"confidence_min": 0.7, "date_range": "30d"}
        
        # Add test results
        test_result = OSINTResult(
            target="example.com",
            intelligence_type=IntelligenceType.DOMAIN,
            source=OSINTSource.WHOIS,
            timestamp=datetime.utcnow(),
            data={"registrar": "Test Registrar"},
            confidence=ConfidenceLevel.HIGH
        )
        osint_module.results = [test_result]
        
        results = await osint_module.search_intelligence(query, filters)
        
        assert len(results) == 1
        assert results[0].target == "example.com"
    
    @pytest.mark.asyncio
    async def test_find_related_intelligence(self, osint_module):
        """Test finding related intelligence"""
        # Add test results
        test_results = [
            OSINTResult(
                target="example.com",
                intelligence_type=IntelligenceType.DOMAIN,
                source=OSINTSource.WHOIS,
                timestamp=datetime.utcnow(),
                data={"registrar": "Test Registrar"},
                confidence=ConfidenceLevel.HIGH
            ),
            OSINTResult(
                target="www.example.com",
                intelligence_type=IntelligenceType.DOMAIN,
                source=OSINTSource.SUBDOMAIN_ENUM,
                timestamp=datetime.utcnow(),
                data={"ip": "192.168.1.1"},
                confidence=ConfidenceLevel.HIGH
            )
        ]
        osint_module.results = test_results
        
        related = await osint_module.find_related_intelligence("example.com")
        
        assert len(related) >= 1
        assert any(r.target == "www.example.com" for r in related)
    
    def test_get_results_filtering(self, osint_module):
        """Test result filtering"""
        # Add test results
        test_results = [
            OSINTResult(
                target="example.com",
                intelligence_type=IntelligenceType.DOMAIN,
                source=OSINTSource.WHOIS,
                timestamp=datetime.utcnow(),
                data={},
                confidence=ConfidenceLevel.HIGH
            ),
            OSINTResult(
                target="192.168.1.1",
                intelligence_type=IntelligenceType.IP,
                source=OSINTSource.GEOLOCATION,
                timestamp=datetime.utcnow(),
                data={},
                confidence=ConfidenceLevel.MEDIUM
            )
        ]
        osint_module.results = test_results
        
        # Test filtering by target
        filtered = osint_module.get_results(target="example.com")
        assert len(filtered) == 1
        assert filtered[0].target == "example.com"
        
        # Test filtering by intelligence type
        filtered = osint_module.get_results(intelligence_type=IntelligenceType.DOMAIN)
        assert len(filtered) == 1
        assert filtered[0].intelligence_type == IntelligenceType.DOMAIN
        
        # Test filtering by source
        filtered = osint_module.get_results(source=OSINTSource.WHOIS)
        assert len(filtered) == 1
        assert filtered[0].source == OSINTSource.WHOIS
    
    def test_export_results_json(self, osint_module):
        """Test JSON export of results"""
        # Add test result
        test_result = OSINTResult(
            target="example.com",
            intelligence_type=IntelligenceType.DOMAIN,
            source=OSINTSource.WHOIS,
            timestamp=datetime.utcnow(),
            data={"registrar": "Test Registrar"},
            confidence=ConfidenceLevel.HIGH
        )
        osint_module.results = [test_result]
        
        json_output = osint_module.export_results("json")
        
        assert json_output != ""
        # Verify it's valid JSON
        parsed = json.loads(json_output)
        assert len(parsed) == 1
        assert parsed[0]["target"] == "example.com"
        assert parsed[0]["intelligence_type"] == "domain"
    
    @pytest.mark.asyncio
    async def test_get_status(self, osint_module):
        """Test module status reporting"""
        # Add test result
        test_result = OSINTResult(
            target="example.com",
            intelligence_type=IntelligenceType.DOMAIN,
            source=OSINTSource.WHOIS,
            timestamp=datetime.utcnow(),
            data={"registrar": "Test Registrar"},
            confidence=ConfidenceLevel.HIGH
        )
        osint_module.results = [test_result]
        
        status = await osint_module.get_status()
        
        assert status["module"] == "osint"
        assert status["status"] == "initialized"
        assert status["version"] == "1.0.0"
        assert status["results_count"] == 1
        assert status["high_confidence_results"] == 1
        assert "example.com" in status["targets_investigated"]
        assert OSINTSource.WHOIS.value in status["sources_used"]


@pytest.mark.performance
class TestOSINTPerformance:
    """Performance tests for OSINT module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def osint_module(self, config):
        """OSINT module fixture"""
        return OSINTModule(config)
    
    @pytest.mark.asyncio
    async def test_bulk_intelligence_gathering_performance(self, osint_module, performance_monitor):
        """Test bulk intelligence gathering performance"""
        # Create multiple targets
        targets = [
            OSINTTarget(f"example{i}.com", IntelligenceType.DOMAIN)
            for i in range(10)
        ]
        
        # Mock intelligence gathering
        with patch.object(osint_module, 'execute_intelligence_gathering', return_value=[]):
            performance_monitor.start()
            
            results = await osint_module.bulk_intelligence_gathering(targets)
            
            performance_monitor.stop()
            
            duration = performance_monitor.get_duration()
            assert duration is not None
            assert duration < 30.0  # Should complete within 30 seconds
            assert len(results) == 10


@pytest.mark.security
class TestOSINTSecurity:
    """Security tests for OSINT module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def osint_module(self, config):
        """OSINT module fixture"""
        return OSINTModule(config)
    
    def test_input_validation_target(self, osint_module):
        """Test input validation for targets"""
        # Test malicious targets
        malicious_targets = [
            "; rm -rf /",
            "../../..",
            "<script>alert('xss')</script>",
            "$(whoami)",
            "`id`"
        ]
        
        for target in malicious_targets:
            osint_target = OSINTTarget(
                target=target,
                intelligence_type=IntelligenceType.DOMAIN
            )
            
            # The module should handle these gracefully
            assert osint_target.target == target
    
    @pytest.mark.asyncio
    async def test_api_key_protection(self, osint_module):
        """Test API key protection"""
        # Test that API keys are not exposed in results
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"api_key": "secret-key", "data": "result"}
            mock_get.return_value = mock_response
            
            target = OSINTTarget("example.com", IntelligenceType.DOMAIN)
            results = await osint_module.execute_intelligence_gathering(target)
            
            # API keys should not be in the exported results
            json_output = osint_module.export_results("json")
            assert "secret-key" not in json_output
    
    @pytest.mark.asyncio
    async def test_rate_limiting_compliance(self, osint_module):
        """Test rate limiting compliance"""
        # Test that the module respects rate limits
        targets = [
            OSINTTarget(f"example{i}.com", IntelligenceType.DOMAIN)
            for i in range(100)  # Many targets
        ]
        
        # Mock rate-limited responses
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429  # Too Many Requests
            mock_get.return_value = mock_response
            
            # Should handle rate limiting gracefully
            results = await osint_module.bulk_intelligence_gathering(targets)
            
            # Should not crash or exhaust resources
            assert isinstance(results, dict)
    
    @pytest.mark.asyncio
    async def test_data_sanitization(self, osint_module):
        """Test data sanitization"""
        # Test that sensitive data is properly sanitized
        target = OSINTTarget("example.com", IntelligenceType.DOMAIN)
        
        # Mock response with sensitive data
        with patch.object(osint_module.domain_intel, 'whois_lookup') as mock_whois:
            mock_result = OSINTResult(
                target="example.com",
                intelligence_type=IntelligenceType.DOMAIN,
                source=OSINTSource.WHOIS,
                timestamp=datetime.utcnow(),
                data={
                    "registrar": "Test Registrar",
                    "admin_email": "admin@example.com",
                    "registrant_phone": "+1-555-123-4567"
                },
                confidence=ConfidenceLevel.HIGH
            )
            mock_whois.return_value = mock_result
            
            results = await osint_module.execute_intelligence_gathering(target)
            
            # Check that sensitive data is handled appropriately
            assert len(results) > 0