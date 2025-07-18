"""
Reconnaissance Module for Aetherveil Sentinel

Provides comprehensive reconnaissance capabilities including passive and active
information gathering techniques for defensive security assessments.

Security Level: DEFENSIVE_ONLY
"""

import asyncio
import socket
import subprocess
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Network, AddressValueError
import dns.resolver
import dns.reversename
import whois
import requests
from urllib.parse import urlparse
import ssl
import concurrent.futures
from datetime import datetime, timedelta

from ..config.config import AetherVeilConfig
from . import ModuleType, ModuleStatus, register_module

logger = logging.getLogger(__name__)

class ReconMode(Enum):
    """Reconnaissance operation modes"""
    PASSIVE = "passive"
    ACTIVE = "active"
    STEALTH = "stealth"
    AGGRESSIVE = "aggressive"

class TargetType(Enum):
    """Types of reconnaissance targets"""
    DOMAIN = "domain"
    IP_RANGE = "ip_range"
    SINGLE_IP = "single_ip"
    URL = "url"
    ORGANIZATION = "organization"

@dataclass
class ReconTarget:
    """Reconnaissance target specification"""
    target: str
    target_type: TargetType
    mode: ReconMode = ReconMode.PASSIVE
    depth: int = 1
    timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReconResult:
    """Reconnaissance operation result"""
    target: str
    target_type: TargetType
    technique: str
    timestamp: datetime
    data: Dict[str, Any]
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class DNSReconnaissance:
    """DNS-based reconnaissance techniques"""
    
    def __init__(self):
        self.resolver = dns.resolver.Resolver()
        self.resolver.timeout = 10
        self.resolver.lifetime = 30
        
    async def enumerate_subdomains(self, domain: str, wordlist: List[str] = None) -> List[ReconResult]:
        """Enumerate subdomains using various techniques"""
        results = []
        
        # Default subdomain wordlist
        if wordlist is None:
            wordlist = [
                'www', 'mail', 'ftp', 'smtp', 'pop', 'ns1', 'ns2', 'admin',
                'test', 'dev', 'staging', 'api', 'app', 'web', 'secure',
                'vpn', 'remote', 'blog', 'shop', 'store', 'forum', 'wiki'
            ]
        
        # Subdomain brute force
        for subdomain in wordlist:
            try:
                full_domain = f"{subdomain}.{domain}"
                answers = self.resolver.resolve(full_domain, 'A')
                ips = [str(rdata) for rdata in answers]
                
                result = ReconResult(
                    target=full_domain,
                    target_type=TargetType.DOMAIN,
                    technique="subdomain_bruteforce",
                    timestamp=datetime.utcnow(),
                    data={"ips": ips, "subdomain": subdomain},
                    confidence=0.9,
                    source="dns_enumeration"
                )
                results.append(result)
                
            except Exception as e:
                logger.debug(f"Subdomain {subdomain}.{domain} not found: {e}")
                
        # Zone transfer attempt (passive)
        try:
            ns_records = self.resolver.resolve(domain, 'NS')
            for ns in ns_records:
                try:
                    zone = dns.zone.from_xfr(dns.query.xfr(str(ns), domain))
                    for name, node in zone.nodes.items():
                        if name != dns.name.from_text('@'):
                            full_name = f"{name}.{domain}"
                            result = ReconResult(
                                target=full_name,
                                target_type=TargetType.DOMAIN,
                                technique="zone_transfer",
                                timestamp=datetime.utcnow(),
                                data={"nameserver": str(ns)},
                                confidence=1.0,
                                source="dns_zone_transfer"
                            )
                            results.append(result)
                except Exception as e:
                    logger.debug(f"Zone transfer failed for {ns}: {e}")
                    
        except Exception as e:
            logger.debug(f"No NS records found for {domain}: {e}")
            
        return results
    
    async def dns_reconnaissance(self, domain: str) -> List[ReconResult]:
        """Comprehensive DNS reconnaissance"""
        results = []
        record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA', 'PTR']
        
        for record_type in record_types:
            try:
                answers = self.resolver.resolve(domain, record_type)
                records = [str(rdata) for rdata in answers]
                
                result = ReconResult(
                    target=domain,
                    target_type=TargetType.DOMAIN,
                    technique=f"dns_{record_type.lower()}_lookup",
                    timestamp=datetime.utcnow(),
                    data={"records": records, "type": record_type},
                    confidence=1.0,
                    source="dns_lookup"
                )
                results.append(result)
                
            except Exception as e:
                logger.debug(f"No {record_type} records for {domain}: {e}")
                
        return results

class NetworkReconnaissance:
    """Network-based reconnaissance techniques"""
    
    def __init__(self):
        self.timeout = 5
        
    async def port_discovery(self, target: str, port_range: Tuple[int, int] = (1, 1000)) -> List[ReconResult]:
        """Discover open ports on target"""
        results = []
        start_port, end_port = port_range
        
        # Common ports for quick scan
        common_ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995]
        
        async def scan_port(ip: str, port: int) -> Optional[ReconResult]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                result = sock.connect_ex((ip, port))
                sock.close()
                
                if result == 0:
                    # Try to grab banner
                    banner = await self._grab_banner(ip, port)
                    
                    return ReconResult(
                        target=f"{ip}:{port}",
                        target_type=TargetType.SINGLE_IP,
                        technique="port_scan",
                        timestamp=datetime.utcnow(),
                        data={"port": port, "state": "open", "banner": banner},
                        confidence=1.0,
                        source="tcp_connect_scan"
                    )
            except Exception as e:
                logger.debug(f"Port scan error for {ip}:{port}: {e}")
                
            return None
        
        # Scan common ports first
        tasks = [scan_port(target, port) for port in common_ports if start_port <= port <= end_port]
        port_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in port_results:
            if isinstance(result, ReconResult):
                results.append(result)
                
        return results
    
    async def _grab_banner(self, ip: str, port: int) -> Optional[str]:
        """Attempt to grab service banner"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((ip, port))
            
            # Send basic HTTP request for web services
            if port in [80, 443, 8080, 8443]:
                sock.send(b"HEAD / HTTP/1.0\r\n\r\n")
            
            banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
            sock.close()
            return banner
            
        except Exception as e:
            logger.debug(f"Banner grab failed for {ip}:{port}: {e}")
            return None
    
    async def network_discovery(self, network: str) -> List[ReconResult]:
        """Discover active hosts in network range"""
        results = []
        
        try:
            net = IPv4Network(network, strict=False)
        except AddressValueError:
            logger.error(f"Invalid network range: {network}")
            return results
        
        async def ping_host(ip: str) -> Optional[ReconResult]:
            try:
                # Use ping command for host discovery
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', '1', str(ip)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    return ReconResult(
                        target=str(ip),
                        target_type=TargetType.SINGLE_IP,
                        technique="icmp_ping",
                        timestamp=datetime.utcnow(),
                        data={"state": "alive", "response_time": self._extract_ping_time(result.stdout)},
                        confidence=0.9,
                        source="icmp_discovery"
                    )
            except Exception as e:
                logger.debug(f"Ping failed for {ip}: {e}")
                
            return None
        
        # Limit to reasonable network sizes
        if net.num_addresses > 1024:
            logger.warning(f"Network {network} too large, limiting scan")
            hosts = list(net.hosts())[:1024]
        else:
            hosts = list(net.hosts())
        
        # Concurrent ping scanning
        semaphore = asyncio.Semaphore(50)  # Limit concurrent operations
        
        async def bounded_ping(ip):
            async with semaphore:
                return await ping_host(str(ip))
        
        tasks = [bounded_ping(ip) for ip in hosts]
        ping_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in ping_results:
            if isinstance(result, ReconResult):
                results.append(result)
                
        return results
    
    def _extract_ping_time(self, ping_output: str) -> Optional[float]:
        """Extract ping response time from output"""
        try:
            match = re.search(r'time=(\d+\.?\d*)', ping_output)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return None

class WebReconnaissance:
    """Web application reconnaissance"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        self.timeout = 10
        
    async def web_reconnaissance(self, url: str) -> List[ReconResult]:
        """Comprehensive web reconnaissance"""
        results = []
        
        try:
            # HTTP headers analysis
            response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
            
            result = ReconResult(
                target=url,
                target_type=TargetType.URL,
                technique="http_headers",
                timestamp=datetime.utcnow(),
                data={
                    "headers": dict(response.headers),
                    "status_code": response.status_code,
                    "final_url": response.url
                },
                confidence=1.0,
                source="http_analysis"
            )
            results.append(result)
            
            # Technology detection
            tech_info = await self._detect_technologies(response)
            if tech_info:
                result = ReconResult(
                    target=url,
                    target_type=TargetType.URL,
                    technique="technology_detection",
                    timestamp=datetime.utcnow(),
                    data=tech_info,
                    confidence=0.8,
                    source="technology_fingerprinting"
                )
                results.append(result)
            
            # Directory enumeration (basic)
            dirs = await self._enumerate_directories(url)
            for dir_info in dirs:
                result = ReconResult(
                    target=f"{url}/{dir_info['path']}",
                    target_type=TargetType.URL,
                    technique="directory_enumeration",
                    timestamp=datetime.utcnow(),
                    data=dir_info,
                    confidence=0.7,
                    source="directory_discovery"
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Web reconnaissance failed for {url}: {e}")
            
        return results
    
    async def _detect_technologies(self, response) -> Dict[str, Any]:
        """Detect web technologies from HTTP response"""
        technologies = {}
        
        # Server detection
        server = response.headers.get('Server', '')
        if server:
            technologies['server'] = server
            
        # Framework detection
        x_powered_by = response.headers.get('X-Powered-By', '')
        if x_powered_by:
            technologies['framework'] = x_powered_by
            
        # CMS detection patterns
        cms_patterns = {
            'WordPress': ['wp-content', 'wp-includes'],
            'Drupal': ['sites/default', 'misc/drupal.js'],
            'Joomla': ['components/com_', 'templates/system']
        }
        
        try:
            content = self.session.get(response.url, timeout=self.timeout).text.lower()
            for cms, patterns in cms_patterns.items():
                if any(pattern in content for pattern in patterns):
                    technologies['cms'] = cms
                    break
        except Exception as e:
            logger.debug(f"Content analysis failed: {e}")
            
        return technologies
    
    async def _enumerate_directories(self, base_url: str) -> List[Dict[str, Any]]:
        """Basic directory enumeration"""
        directories = []
        common_dirs = [
            'admin', 'login', 'dashboard', 'api', 'docs', 'test',
            'backup', 'config', 'uploads', 'images', 'css', 'js'
        ]
        
        for directory in common_dirs:
            try:
                url = f"{base_url.rstrip('/')}/{directory}"
                response = self.session.head(url, timeout=5)
                
                if response.status_code in [200, 301, 302, 403]:
                    directories.append({
                        'path': directory,
                        'status_code': response.status_code,
                        'size': response.headers.get('content-length', 'unknown')
                    })
                    
            except Exception as e:
                logger.debug(f"Directory check failed for {directory}: {e}")
                
        return directories

class SSLReconnaissance:
    """SSL/TLS certificate reconnaissance"""
    
    async def ssl_reconnaissance(self, hostname: str, port: int = 443) -> List[ReconResult]:
        """Analyze SSL/TLS certificates and configuration"""
        results = []
        
        try:
            # Get certificate information
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    
                    # Certificate analysis
                    cert_result = ReconResult(
                        target=f"{hostname}:{port}",
                        target_type=TargetType.DOMAIN,
                        technique="ssl_certificate_analysis",
                        timestamp=datetime.utcnow(),
                        data={
                            "subject": dict(x[0] for x in cert.get('subject', [])),
                            "issuer": dict(x[0] for x in cert.get('issuer', [])),
                            "version": cert.get('version'),
                            "serial_number": cert.get('serialNumber'),
                            "not_before": cert.get('notBefore'),
                            "not_after": cert.get('notAfter'),
                            "san": cert.get('subjectAltName', [])
                        },
                        confidence=1.0,
                        source="ssl_analysis"
                    )
                    results.append(cert_result)
                    
                    # Cipher analysis
                    cipher_result = ReconResult(
                        target=f"{hostname}:{port}",
                        target_type=TargetType.DOMAIN,
                        technique="ssl_cipher_analysis",
                        timestamp=datetime.utcnow(),
                        data={
                            "cipher_suite": cipher[0] if cipher else None,
                            "ssl_version": cipher[1] if cipher else None,
                            "key_bits": cipher[2] if cipher else None
                        },
                        confidence=1.0,
                        source="ssl_analysis"
                    )
                    results.append(cipher_result)
                    
        except Exception as e:
            logger.error(f"SSL reconnaissance failed for {hostname}:{port}: {e}")
            
        return results

class ReconnaissanceModule:
    """Main reconnaissance module orchestrator"""
    
    def __init__(self, config: AetherVeilConfig):
        self.config = config
        self.module_type = ModuleType.RECONNAISSANCE
        self.status = ModuleStatus.INITIALIZED
        self.version = "1.0.0"
        
        # Initialize reconnaissance components
        self.dns_recon = DNSReconnaissance()
        self.network_recon = NetworkReconnaissance()
        self.web_recon = WebReconnaissance()
        self.ssl_recon = SSLReconnaissance()
        
        # Result storage
        self.results: List[ReconResult] = []
        
        logger.info("Reconnaissance module initialized")
        
    async def start(self) -> bool:
        """Start the reconnaissance module"""
        try:
            self.status = ModuleStatus.RUNNING
            logger.info("Reconnaissance module started")
            return True
        except Exception as e:
            self.status = ModuleStatus.ERROR
            logger.error(f"Failed to start reconnaissance module: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the reconnaissance module"""
        try:
            self.status = ModuleStatus.STOPPED
            logger.info("Reconnaissance module stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop reconnaissance module: {e}")
            return False
    
    async def execute_reconnaissance(self, target: ReconTarget) -> List[ReconResult]:
        """Execute reconnaissance against specified target"""
        results = []
        
        try:
            logger.info(f"Starting reconnaissance for {target.target} (mode: {target.mode.value})")
            
            if target.target_type == TargetType.DOMAIN:
                # DNS reconnaissance
                dns_results = await self.dns_recon.dns_reconnaissance(target.target)
                results.extend(dns_results)
                
                # Subdomain enumeration
                if target.mode in [ReconMode.ACTIVE, ReconMode.AGGRESSIVE]:
                    subdomain_results = await self.dns_recon.enumerate_subdomains(target.target)
                    results.extend(subdomain_results)
                
                # SSL reconnaissance
                ssl_results = await self.ssl_recon.ssl_reconnaissance(target.target)
                results.extend(ssl_results)
                
                # Web reconnaissance for HTTP/HTTPS
                for protocol in ['http', 'https']:
                    url = f"{protocol}://{target.target}"
                    web_results = await self.web_recon.web_reconnaissance(url)
                    results.extend(web_results)
                    
            elif target.target_type == TargetType.IP_RANGE:
                # Network discovery
                network_results = await self.network_recon.network_discovery(target.target)
                results.extend(network_results)
                
            elif target.target_type == TargetType.SINGLE_IP:
                # Port scanning
                if target.mode in [ReconMode.ACTIVE, ReconMode.AGGRESSIVE]:
                    port_results = await self.network_recon.port_discovery(target.target)
                    results.extend(port_results)
                    
            elif target.target_type == TargetType.URL:
                # Web reconnaissance
                web_results = await self.web_recon.web_reconnaissance(target.target)
                results.extend(web_results)
                
            # Store results
            self.results.extend(results)
            
            logger.info(f"Reconnaissance completed for {target.target}: {len(results)} results")
            
        except Exception as e:
            logger.error(f"Reconnaissance failed for {target.target}: {e}")
            
        return results
    
    async def bulk_reconnaissance(self, targets: List[ReconTarget]) -> Dict[str, List[ReconResult]]:
        """Execute reconnaissance against multiple targets"""
        all_results = {}
        
        # Process targets concurrently with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent operations
        
        async def process_target(target: ReconTarget):
            async with semaphore:
                results = await self.execute_reconnaissance(target)
                return target.target, results
        
        tasks = [process_target(target) for target in targets]
        target_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in target_results:
            if isinstance(result, tuple):
                target_name, results = result
                all_results[target_name] = results
            elif isinstance(result, Exception):
                logger.error(f"Target processing failed: {result}")
                
        return all_results
    
    def get_results(self, target: str = None, technique: str = None) -> List[ReconResult]:
        """Retrieve reconnaissance results with optional filtering"""
        filtered_results = self.results
        
        if target:
            filtered_results = [r for r in filtered_results if target in r.target]
            
        if technique:
            filtered_results = [r for r in filtered_results if r.technique == technique]
            
        return filtered_results
    
    def export_results(self, format: str = "json") -> str:
        """Export reconnaissance results in specified format"""
        if format == "json":
            results_dict = []
            for result in self.results:
                results_dict.append({
                    "target": result.target,
                    "target_type": result.target_type.value,
                    "technique": result.technique,
                    "timestamp": result.timestamp.isoformat(),
                    "data": result.data,
                    "confidence": result.confidence,
                    "source": result.source,
                    "metadata": result.metadata
                })
            return json.dumps(results_dict, indent=2)
        
        return ""
    
    async def get_status(self) -> Dict[str, Any]:
        """Get module status and statistics"""
        return {
            "module": "reconnaissance",
            "status": self.status.value,
            "version": self.version,
            "results_count": len(self.results),
            "last_activity": max([r.timestamp for r in self.results]).isoformat() if self.results else None,
            "techniques_used": list(set(r.technique for r in self.results)),
            "targets_scanned": list(set(r.target for r in self.results))
        }

# Register module on import
def create_reconnaissance_module(config: AetherVeilConfig) -> ReconnaissanceModule:
    """Factory function to create and register reconnaissance module"""
    module = ReconnaissanceModule(config)
    register_module("reconnaissance", module)
    return module

__all__ = [
    "ReconnaissanceModule",
    "ReconTarget", 
    "ReconResult",
    "ReconMode",
    "TargetType",
    "create_reconnaissance_module"
]