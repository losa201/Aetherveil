"""
Enhanced Threat Intelligence Integration Module
Provides real-world API integrations for comprehensive OSINT and threat intelligence
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import base64
from urllib.parse import urlencode, quote
import xml.etree.ElementTree as ET

from ...config.config import config
from ...coordinator.security_manager import security_manager

logger = logging.getLogger(__name__)


class ThreatIntelligenceProvider(Enum):
    """Threat intelligence providers"""
    SHODAN = "shodan"
    CENSYS = "censys"
    VIRUSTOTAL = "virustotal"
    ABUSE_IPDB = "abuseipdb"
    MISP = "misp"
    CIRCL = "circl"
    HYBRID_ANALYSIS = "hybrid_analysis"
    URLVOID = "urlvoid"
    GREYNOISE = "greynoise"
    ALIENVAULT = "alienvault"
    THREATCROWD = "threatcrowd"
    SPAMHAUS = "spamhaus"
    MALWARE_BAZAAR = "malware_bazaar"
    CVE_MITRE = "cve_mitre"
    NVD = "nvd"


@dataclass
class ThreatIntelligenceResult:
    """Threat intelligence result structure"""
    provider: ThreatIntelligenceProvider
    query: str
    result_type: str
    data: Dict[str, Any]
    confidence: float
    severity: str
    timestamp: datetime
    ttl: int = 3600  # Cache TTL in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "provider": self.provider.value,
            "query": self.query,
            "result_type": self.result_type,
            "data": self.data,
            "confidence": self.confidence,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl
        }


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit slot"""
        async with self.lock:
            now = time.time()
            # Remove old calls outside time window
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                oldest_call = min(self.calls)
                wait_time = self.time_window - (now - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            self.calls.append(now)


class EnhancedThreatIntelligence:
    """Enhanced threat intelligence aggregator with multiple API integrations"""
    
    def __init__(self):
        self.session = None
        self.rate_limiters = {}
        self.api_keys = {}
        self.base_urls = {}
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour default cache
        
        # Initialize API configurations
        self._initialize_api_configs()
        
        # Initialize rate limiters
        self._initialize_rate_limiters()
    
    def _initialize_api_configs(self):
        """Initialize API configurations"""
        self.api_keys = {
            ThreatIntelligenceProvider.SHODAN: security_manager.get_secret("SHODAN_API_KEY"),
            ThreatIntelligenceProvider.CENSYS: {
                "api_id": security_manager.get_secret("CENSYS_API_ID"),
                "api_secret": security_manager.get_secret("CENSYS_API_SECRET")
            },
            ThreatIntelligenceProvider.VIRUSTOTAL: security_manager.get_secret("VIRUSTOTAL_API_KEY"),
            ThreatIntelligenceProvider.ABUSE_IPDB: security_manager.get_secret("ABUSEIPDB_API_KEY"),
            ThreatIntelligenceProvider.HYBRID_ANALYSIS: security_manager.get_secret("HYBRID_ANALYSIS_API_KEY"),
            ThreatIntelligenceProvider.GREYNOISE: security_manager.get_secret("GREYNOISE_API_KEY"),
            ThreatIntelligenceProvider.ALIENVAULT: security_manager.get_secret("ALIENVAULT_API_KEY"),
        }
        
        self.base_urls = {
            ThreatIntelligenceProvider.SHODAN: "https://api.shodan.io",
            ThreatIntelligenceProvider.CENSYS: "https://search.censys.io/api/v2",
            ThreatIntelligenceProvider.VIRUSTOTAL: "https://www.virustotal.com/vtapi/v2",
            ThreatIntelligenceProvider.ABUSE_IPDB: "https://api.abuseipdb.com/api/v2",
            ThreatIntelligenceProvider.HYBRID_ANALYSIS: "https://www.hybrid-analysis.com/api/v2",
            ThreatIntelligenceProvider.GREYNOISE: "https://api.greynoise.io/v3",
            ThreatIntelligenceProvider.ALIENVAULT: "https://otx.alienvault.com/api/v1",
            ThreatIntelligenceProvider.THREATCROWD: "https://www.threatcrowd.org/searchApi/v2",
            ThreatIntelligenceProvider.MALWARE_BAZAAR: "https://mb-api.abuse.ch/api/v1",
            ThreatIntelligenceProvider.CVE_MITRE: "https://cve.mitre.org/cgi-bin/cvename.cgi",
            ThreatIntelligenceProvider.NVD: "https://services.nvd.nist.gov/rest/json/cves/2.0",
        }
    
    def _initialize_rate_limiters(self):
        """Initialize rate limiters for each provider"""
        self.rate_limiters = {
            ThreatIntelligenceProvider.SHODAN: RateLimiter(max_calls=100, time_window=60),
            ThreatIntelligenceProvider.CENSYS: RateLimiter(max_calls=120, time_window=60),
            ThreatIntelligenceProvider.VIRUSTOTAL: RateLimiter(max_calls=4, time_window=60),
            ThreatIntelligenceProvider.ABUSE_IPDB: RateLimiter(max_calls=1000, time_window=86400),
            ThreatIntelligenceProvider.HYBRID_ANALYSIS: RateLimiter(max_calls=100, time_window=60),
            ThreatIntelligenceProvider.GREYNOISE: RateLimiter(max_calls=1000, time_window=86400),
            ThreatIntelligenceProvider.ALIENVAULT: RateLimiter(max_calls=10000, time_window=3600),
            ThreatIntelligenceProvider.THREATCROWD: RateLimiter(max_calls=1, time_window=10),
            ThreatIntelligenceProvider.MALWARE_BAZAAR: RateLimiter(max_calls=100, time_window=60),
        }
    
    async def initialize(self):
        """Initialize HTTP session and connections"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=False  # For testing, enable SSL in production
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "AetherVeil-Sentinel/1.0 (Security Research)",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        
        logger.info("Enhanced threat intelligence initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def _make_api_request(self, provider: ThreatIntelligenceProvider, 
                               endpoint: str, params: Optional[Dict] = None,
                               headers: Optional[Dict] = None, 
                               method: str = "GET",
                               data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make rate-limited API request"""
        if not self.session:
            await self.initialize()
        
        # Apply rate limiting
        if provider in self.rate_limiters:
            await self.rate_limiters[provider].acquire()
        
        # Build URL
        base_url = self.base_urls.get(provider, "")
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        # Prepare headers
        request_headers = headers or {}
        
        # Add authentication headers
        if provider == ThreatIntelligenceProvider.SHODAN:
            if params is None:
                params = {}
            params["key"] = self.api_keys[provider]
        
        elif provider == ThreatIntelligenceProvider.CENSYS:
            auth_string = f"{self.api_keys[provider]['api_id']}:{self.api_keys[provider]['api_secret']}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            request_headers["Authorization"] = f"Basic {encoded_auth}"
        
        elif provider == ThreatIntelligenceProvider.VIRUSTOTAL:
            request_headers["X-Apikey"] = self.api_keys[provider]
        
        elif provider == ThreatIntelligenceProvider.ABUSE_IPDB:
            request_headers["Key"] = self.api_keys[provider]
        
        elif provider == ThreatIntelligenceProvider.HYBRID_ANALYSIS:
            request_headers["api-key"] = self.api_keys[provider]
        
        elif provider == ThreatIntelligenceProvider.GREYNOISE:
            request_headers["key"] = self.api_keys[provider]
        
        elif provider == ThreatIntelligenceProvider.ALIENVAULT:
            request_headers["X-OTX-API-KEY"] = self.api_keys[provider]
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                headers=request_headers,
                json=data if method in ["POST", "PUT"] else None
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limited
                    logger.warning(f"Rate limited by {provider.value}")
                    await asyncio.sleep(60)  # Wait 1 minute
                    return await self._make_api_request(provider, endpoint, params, headers, method, data)
                else:
                    logger.error(f"API request failed: {response.status} - {await response.text()}")
                    return {"error": f"HTTP {response.status}"}
        
        except Exception as e:
            logger.error(f"API request exception: {e}")
            return {"error": str(e)}
    
    def _get_cache_key(self, provider: ThreatIntelligenceProvider, query: str, query_type: str) -> str:
        """Generate cache key"""
        key_data = f"{provider.value}:{query_type}:{query}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ThreatIntelligenceResult]:
        """Get cached result"""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < result.ttl:
                return result
            else:
                del self.cache[cache_key]
        return None
    
    async def _cache_result(self, cache_key: str, result: ThreatIntelligenceResult):
        """Cache result"""
        self.cache[cache_key] = (result, time.time())
        
        # Clean old cache entries
        if len(self.cache) > 1000:  # Max 1000 cached entries
            # Remove oldest 100 entries
            sorted_cache = sorted(self.cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:100]:
                del self.cache[key]
    
    async def search_shodan(self, query: str, facets: Optional[str] = None) -> ThreatIntelligenceResult:
        """Search Shodan for hosts and services"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.SHODAN, query, "search")
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        params = {"q": query}
        if facets:
            params["facets"] = facets
        
        response = await self._make_api_request(
            ThreatIntelligenceProvider.SHODAN,
            "/shodan/host/search",
            params=params
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            confidence = 0.9
            severity = "medium" if response.get("total", 0) > 0 else "low"
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.SHODAN,
            query=query,
            result_type="search",
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def get_shodan_host_info(self, ip: str) -> ThreatIntelligenceResult:
        """Get detailed host information from Shodan"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.SHODAN, ip, "host")
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        response = await self._make_api_request(
            ThreatIntelligenceProvider.SHODAN,
            f"/shodan/host/{ip}"
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            # Calculate severity based on open ports and vulnerabilities
            ports = response.get("ports", [])
            vulns = response.get("vulns", [])
            
            if len(vulns) > 0:
                severity = "high"
                confidence = 0.95
            elif len(ports) > 10:
                severity = "medium"
                confidence = 0.8
            else:
                severity = "low"
                confidence = 0.7
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.SHODAN,
            query=ip,
            result_type="host",
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def search_censys(self, query: str, index: str = "hosts") -> ThreatIntelligenceResult:
        """Search Censys for hosts and certificates"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.CENSYS, query, f"search_{index}")
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        data = {
            "q": query,
            "per_page": 100,
            "cursor": ""
        }
        
        response = await self._make_api_request(
            ThreatIntelligenceProvider.CENSYS,
            f"/search/{index}",
            method="GET",
            params=data
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            total_results = response.get("result", {}).get("total", 0)
            confidence = 0.85
            severity = "medium" if total_results > 0 else "low"
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.CENSYS,
            query=query,
            result_type=f"search_{index}",
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def check_virustotal_ip(self, ip: str) -> ThreatIntelligenceResult:
        """Check IP reputation on VirusTotal"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.VIRUSTOTAL, ip, "ip")
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        params = {"ip": ip}
        response = await self._make_api_request(
            ThreatIntelligenceProvider.VIRUSTOTAL,
            "/ip-address/report",
            params=params
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            # Calculate severity based on detection ratio
            positives = response.get("positives", 0)
            total = response.get("total", 1)
            detection_ratio = positives / total if total > 0 else 0
            
            if detection_ratio > 0.1:
                severity = "high"
                confidence = 0.95
            elif detection_ratio > 0.05:
                severity = "medium"
                confidence = 0.8
            else:
                severity = "low"
                confidence = 0.7
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.VIRUSTOTAL,
            query=ip,
            result_type="ip",
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def check_virustotal_domain(self, domain: str) -> ThreatIntelligenceResult:
        """Check domain reputation on VirusTotal"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.VIRUSTOTAL, domain, "domain")
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        params = {"domain": domain}
        response = await self._make_api_request(
            ThreatIntelligenceProvider.VIRUSTOTAL,
            "/domain/report",
            params=params
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            # Calculate severity based on detection ratio
            positives = response.get("positives", 0)
            total = response.get("total", 1)
            detection_ratio = positives / total if total > 0 else 0
            
            if detection_ratio > 0.1:
                severity = "high"
                confidence = 0.95
            elif detection_ratio > 0.05:
                severity = "medium"
                confidence = 0.8
            else:
                severity = "low"
                confidence = 0.7
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.VIRUSTOTAL,
            query=domain,
            result_type="domain",
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def check_abuseipdb(self, ip: str, days: int = 30) -> ThreatIntelligenceResult:
        """Check IP reputation on AbuseIPDB"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.ABUSE_IPDB, ip, f"check_{days}")
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        params = {
            "ipAddress": ip,
            "maxAgeInDays": days,
            "verbose": ""
        }
        
        response = await self._make_api_request(
            ThreatIntelligenceProvider.ABUSE_IPDB,
            "/check",
            params=params
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            # Calculate severity based on abuse confidence
            abuse_confidence = response.get("data", {}).get("abuseConfidencePercentage", 0)
            
            if abuse_confidence > 75:
                severity = "high"
                confidence = 0.95
            elif abuse_confidence > 25:
                severity = "medium"
                confidence = 0.8
            else:
                severity = "low"
                confidence = 0.7
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.ABUSE_IPDB,
            query=ip,
            result_type="check",
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def get_greynoise_ip(self, ip: str) -> ThreatIntelligenceResult:
        """Get IP context from GreyNoise"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.GREYNOISE, ip, "context")
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        response = await self._make_api_request(
            ThreatIntelligenceProvider.GREYNOISE,
            f"/context/{ip}"
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            # Calculate severity based on GreyNoise classification
            classification = response.get("classification", "unknown")
            
            if classification == "malicious":
                severity = "high"
                confidence = 0.9
            elif classification == "suspicious":
                severity = "medium"
                confidence = 0.8
            else:
                severity = "low"
                confidence = 0.7
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.GREYNOISE,
            query=ip,
            result_type="context",
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def get_otx_indicators(self, indicator: str, indicator_type: str = "IPv4") -> ThreatIntelligenceResult:
        """Get threat indicators from AlienVault OTX"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.ALIENVAULT, indicator, indicator_type)
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        response = await self._make_api_request(
            ThreatIntelligenceProvider.ALIENVAULT,
            f"/indicators/{indicator_type}/{indicator}/general"
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            # Calculate severity based on pulse count and reputation
            pulse_count = response.get("pulse_info", {}).get("count", 0)
            
            if pulse_count > 5:
                severity = "high"
                confidence = 0.9
            elif pulse_count > 0:
                severity = "medium"
                confidence = 0.8
            else:
                severity = "low"
                confidence = 0.6
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.ALIENVAULT,
            query=indicator,
            result_type=indicator_type,
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def search_threatcrowd(self, resource: str, resource_type: str = "ip") -> ThreatIntelligenceResult:
        """Search ThreatCrowd for threat intelligence"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.THREATCROWD, resource, resource_type)
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        params = {resource_type: resource}
        response = await self._make_api_request(
            ThreatIntelligenceProvider.THREATCROWD,
            f"/{resource_type}/report/",
            params=params
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            # Calculate severity based on response code and associated data
            response_code = response.get("response_code", "0")
            
            if response_code == "1":
                # Has associated malware or suspicious activity
                hashes = response.get("hashes", [])
                if len(hashes) > 0:
                    severity = "high"
                    confidence = 0.85
                else:
                    severity = "medium"
                    confidence = 0.7
            else:
                severity = "low"
                confidence = 0.5
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.THREATCROWD,
            query=resource,
            result_type=resource_type,
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now()
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def get_cve_details(self, cve_id: str) -> ThreatIntelligenceResult:
        """Get CVE details from NVD"""
        cache_key = self._get_cache_key(ThreatIntelligenceProvider.NVD, cve_id, "cve")
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        params = {"cveId": cve_id}
        response = await self._make_api_request(
            ThreatIntelligenceProvider.NVD,
            "",
            params=params
        )
        
        if "error" in response:
            confidence = 0.0
            severity = "unknown"
        else:
            # Calculate severity based on CVSS score
            vulnerabilities = response.get("vulnerabilities", [])
            if vulnerabilities:
                cve_item = vulnerabilities[0]
                metrics = cve_item.get("cve", {}).get("metrics", {})
                
                # Try to get CVSS v3 score first, then v2
                cvss_score = 0.0
                if "cvssMetricV31" in metrics:
                    cvss_score = metrics["cvssMetricV31"][0]["cvssData"]["baseScore"]
                elif "cvssMetricV30" in metrics:
                    cvss_score = metrics["cvssMetricV30"][0]["cvssData"]["baseScore"]
                elif "cvssMetricV2" in metrics:
                    cvss_score = metrics["cvssMetricV2"][0]["cvssData"]["baseScore"]
                
                if cvss_score >= 9.0:
                    severity = "critical"
                    confidence = 0.95
                elif cvss_score >= 7.0:
                    severity = "high"
                    confidence = 0.9
                elif cvss_score >= 4.0:
                    severity = "medium"
                    confidence = 0.8
                else:
                    severity = "low"
                    confidence = 0.7
            else:
                severity = "unknown"
                confidence = 0.5
        
        result = ThreatIntelligenceResult(
            provider=ThreatIntelligenceProvider.NVD,
            query=cve_id,
            result_type="cve",
            data=response,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now(),
            ttl=86400  # CVE data changes infrequently, cache for 24 hours
        )
        
        await self._cache_result(cache_key, result)
        return result
    
    async def aggregate_threat_intelligence(self, indicator: str, 
                                          indicator_type: str = "ip") -> Dict[str, Any]:
        """Aggregate threat intelligence from multiple sources"""
        results = []
        
        # Determine which providers to query based on indicator type
        if indicator_type == "ip":
            providers = [
                ("shodan", self.get_shodan_host_info),
                ("virustotal", self.check_virustotal_ip),
                ("abuseipdb", self.check_abuseipdb),
                ("greynoise", self.get_greynoise_ip),
                ("otx", lambda x: self.get_otx_indicators(x, "IPv4")),
                ("threatcrowd", lambda x: self.search_threatcrowd(x, "ip"))
            ]
        elif indicator_type == "domain":
            providers = [
                ("virustotal", self.check_virustotal_domain),
                ("otx", lambda x: self.get_otx_indicators(x, "domain")),
                ("threatcrowd", lambda x: self.search_threatcrowd(x, "domain"))
            ]
        elif indicator_type == "cve":
            providers = [
                ("nvd", self.get_cve_details)
            ]
        else:
            providers = []
        
        # Query all providers concurrently
        tasks = []
        for provider_name, provider_func in providers:
            tasks.append(provider_func(indicator))
        
        if tasks:
            provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            provider_results = []
        
        # Process results
        for i, result in enumerate(provider_results):
            if isinstance(result, Exception):
                logger.error(f"Provider {providers[i][0]} failed: {result}")
                continue
            
            if result:
                results.append(result.to_dict())
        
        # Calculate aggregated confidence and severity
        if results:
            confidences = [r["confidence"] for r in results]
            severities = [r["severity"] for r in results]
            
            # Weighted average confidence
            avg_confidence = sum(confidences) / len(confidences)
            
            # Highest severity wins
            severity_order = ["low", "medium", "high", "critical"]
            max_severity = "low"
            for sev in severities:
                if sev in severity_order:
                    if severity_order.index(sev) > severity_order.index(max_severity):
                        max_severity = sev
            
            # Check for any high-confidence malicious indicators
            malicious_indicators = [r for r in results 
                                  if r["confidence"] > 0.8 and r["severity"] in ["high", "critical"]]
            
            reputation_score = self._calculate_reputation_score(results)
            
            aggregated_result = {
                "indicator": indicator,
                "indicator_type": indicator_type,
                "overall_confidence": avg_confidence,
                "overall_severity": max_severity,
                "reputation_score": reputation_score,
                "malicious_indicators": len(malicious_indicators),
                "total_sources": len(results),
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "summary": self._generate_summary(results)
            }
        else:
            aggregated_result = {
                "indicator": indicator,
                "indicator_type": indicator_type,
                "overall_confidence": 0.0,
                "overall_severity": "unknown",
                "reputation_score": 0.0,
                "malicious_indicators": 0,
                "total_sources": 0,
                "results": [],
                "timestamp": datetime.now().isoformat(),
                "summary": "No threat intelligence data available"
            }
        
        return aggregated_result
    
    def _calculate_reputation_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate reputation score (0-100, where 0 is clean and 100 is malicious)"""
        if not results:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        for result in results:
            confidence = result["confidence"]
            severity = result["severity"]
            
            # Convert severity to numeric score
            severity_scores = {
                "low": 10,
                "medium": 40,
                "high": 80,
                "critical": 100
            }
            
            severity_score = severity_scores.get(severity, 0)
            
            # Weight by confidence
            weighted_score = severity_score * confidence
            score += weighted_score
            total_weight += confidence
        
        if total_weight > 0:
            return min(100.0, score / total_weight)
        
        return 0.0
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate human-readable summary of threat intelligence"""
        if not results:
            return "No threat intelligence data available"
        
        high_confidence_results = [r for r in results if r["confidence"] > 0.8]
        malicious_results = [r for r in results if r["severity"] in ["high", "critical"]]
        
        if malicious_results:
            providers = [r["provider"] for r in malicious_results]
            return f"Potentially malicious indicator detected by {', '.join(providers)}"
        elif high_confidence_results:
            return f"Analyzed by {len(results)} sources, {len(high_confidence_results)} high-confidence results"
        else:
            return f"Analyzed by {len(results)} sources with mixed confidence levels"
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all threat intelligence providers"""
        provider_status = {}
        
        for provider in ThreatIntelligenceProvider:
            has_api_key = bool(self.api_keys.get(provider))
            rate_limiter = self.rate_limiters.get(provider)
            
            if rate_limiter:
                recent_calls = len([call for call in rate_limiter.calls 
                                  if time.time() - call < rate_limiter.time_window])
                rate_limit_status = f"{recent_calls}/{rate_limiter.max_calls}"
            else:
                rate_limit_status = "N/A"
            
            provider_status[provider.value] = {
                "has_api_key": has_api_key,
                "rate_limit_status": rate_limit_status,
                "base_url": self.base_urls.get(provider, "N/A")
            }
        
        return provider_status
    
    async def bulk_ip_analysis(self, ip_list: List[str]) -> Dict[str, Any]:
        """Perform bulk IP analysis"""
        results = {}
        
        # Process IPs in batches to avoid overwhelming APIs
        batch_size = 10
        for i in range(0, len(ip_list), batch_size):
            batch = ip_list[i:i + batch_size]
            
            tasks = []
            for ip in batch:
                tasks.append(self.aggregate_threat_intelligence(ip, "ip"))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Bulk analysis failed for {batch[j]}: {result}")
                    results[batch[j]] = {"error": str(result)}
                else:
                    results[batch[j]] = result
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        return results