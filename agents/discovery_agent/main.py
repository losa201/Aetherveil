#!/usr/bin/env python3
"""
Aetherveil Discovery Agent
Performs defensive reconnaissance and asset discovery for authorized security testing.
"""

import os
import json
import asyncio
import aiohttp
import dns.resolver
import socket
import ssl
import logging
import time
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
import ipaddress
from google.cloud import pubsub_v1, firestore
import urllib.robotparser
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AssetDiscovery:
    """Discovered asset information"""
    asset_id: str
    discovery_type: str  # subdomain, ip, service, certificate
    domain: str
    subdomain: str
    ip_address: str
    ports: List[int]
    services: List[str]
    certificates: List[str]
    technologies: List[str]
    scope_type: str  # in_scope, out_of_scope, unknown
    confidence: str  # high, medium, low
    timestamp: datetime
    discovery_method: str

class SubdomainEnumerator:
    """Subdomain enumeration functionality"""
    
    def __init__(self):
        # Common subdomain wordlist (defensive approach)
        self.common_subdomains = [
            'www', 'api', 'mail', 'ftp', 'admin', 'test', 'dev', 'staging',
            'beta', 'app', 'portal', 'dashboard', 'login', 'auth', 'secure',
            'vpn', 'remote', 'support', 'help', 'docs', 'blog', 'news',
            'shop', 'store', 'payment', 'billing', 'account', 'profile',
            'cdn', 'static', 'assets', 'images', 'files', 'download',
            'upload', 'backup', 'archive', 'old', 'new', 'mobile', 'demo'
        ]
        
    async def enumerate_subdomains(self, domain: str) -> List[str]:
        """Enumerate subdomains using DNS queries"""
        discovered = []
        
        try:
            # DNS brute force with common subdomains
            for subdomain in self.common_subdomains:
                full_domain = f"{subdomain}.{domain}"
                
                try:
                    # Perform DNS lookup
                    answers = dns.resolver.resolve(full_domain, 'A')
                    if answers:
                        discovered.append(full_domain)
                        logger.info(f"Found subdomain: {full_domain}")
                        
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, Exception):
                    pass
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            # Certificate transparency logs (passive)
            ct_subdomains = await self.check_certificate_transparency(domain)
            discovered.extend(ct_subdomains)
            
        except Exception as e:
            logger.error(f"Error enumerating subdomains for {domain}: {e}")
        
        return list(set(discovered))  # Remove duplicates
    
    async def check_certificate_transparency(self, domain: str) -> List[str]:
        """Check certificate transparency logs for subdomains"""
        discovered = []
        
        try:
            # Using crt.sh API (public CT logs)
            url = f"https://crt.sh/?q=%.{domain}&output=json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for cert in data:
                            name_value = cert.get('name_value', '')
                            # Extract subdomains from certificate
                            for name in name_value.split('\n'):
                                name = name.strip()
                                if name.endswith(f'.{domain}') and '*' not in name:
                                    discovered.append(name)
                                    
        except Exception as e:
            logger.debug(f"Error checking CT logs for {domain}: {e}")
        
        return list(set(discovered))

class ServiceMapper:
    """Service and port scanning functionality"""
    
    def __init__(self):
        # Common ports for service discovery
        self.common_ports = [
            21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995,
            1723, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 8888, 9200, 27017
        ]
        
    async def scan_services(self, host: str) -> Dict[str, Any]:
        """Scan services on a host"""
        results = {
            'host': host,
            'ip_address': '',
            'open_ports': [],
            'services': [],
            'certificates': []
        }
        
        try:
            # Resolve IP address
            ip = socket.gethostbyname(host)
            results['ip_address'] = ip
            
            # Scan common ports
            open_ports = await self.scan_ports(ip)
            results['open_ports'] = open_ports
            
            # Identify services
            for port in open_ports:
                service_info = await self.identify_service(ip, port)
                if service_info:
                    results['services'].append(service_info)
            
            # Get SSL certificates for HTTPS services
            if 443 in open_ports:
                cert_info = await self.get_ssl_certificate(host, 443)
                if cert_info:
                    results['certificates'].append(cert_info)
                    
        except Exception as e:
            logger.error(f"Error scanning services for {host}: {e}")
        
        return results
    
    async def scan_ports(self, host: str) -> List[int]:
        """Scan common ports on a host"""
        open_ports = []
        
        # Use asyncio for concurrent port scanning
        tasks = []
        semaphore = asyncio.Semaphore(50)  # Limit concurrent connections
        
        async def check_port(port):
            async with semaphore:
                try:
                    # Connect with timeout
                    future = asyncio.open_connection(host, port)
                    reader, writer = await asyncio.wait_for(future, timeout=3)
                    writer.close()
                    await writer.wait_closed()
                    return port
                except:
                    return None
        
        # Create tasks for all ports
        for port in self.common_ports:
            tasks.append(check_port(port))
        
        # Execute with rate limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect open ports
        for result in results:
            if isinstance(result, int):
                open_ports.append(result)
        
        return sorted(open_ports)
    
    async def identify_service(self, host: str, port: int) -> Optional[Dict[str, Any]]:
        """Identify service running on a port"""
        try:
            # Basic service identification
            service_map = {
                21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'dns',
                80: 'http', 110: 'pop3', 135: 'rpc', 139: 'netbios',
                143: 'imap', 443: 'https', 993: 'imaps', 995: 'pop3s',
                3306: 'mysql', 3389: 'rdp', 5432: 'postgresql',
                6379: 'redis', 8080: 'http-alt', 8443: 'https-alt'
            }
            
            service_name = service_map.get(port, f'unknown-{port}')
            
            # Try to get service banner
            banner = await self.get_service_banner(host, port)
            
            return {
                'port': port,
                'service': service_name,
                'banner': banner,
                'confidence': 'medium' if banner else 'low'
            }
            
        except Exception as e:
            logger.debug(f"Error identifying service on {host}:{port}: {e}")
            return None
    
    async def get_service_banner(self, host: str, port: int) -> str:
        """Get service banner from a port"""
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=5)
            
            # Read initial banner
            banner = await asyncio.wait_for(reader.read(1024), timeout=3)
            
            writer.close()
            await writer.wait_closed()
            
            return banner.decode('utf-8', errors='ignore').strip()
            
        except:
            return ''
    
    async def get_ssl_certificate(self, host: str, port: int) -> Optional[Dict[str, Any]]:
        """Get SSL certificate information"""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            future = asyncio.open_connection(host, port, ssl=context)
            reader, writer = await asyncio.wait_for(future, timeout=10)
            
            # Get certificate
            ssl_object = writer.get_extra_info('ssl_object')
            if ssl_object:
                cert = ssl_object.getpeercert()
                
                writer.close()
                await writer.wait_closed()
                
                return {
                    'subject': cert.get('subject', []),
                    'issuer': cert.get('issuer', []),
                    'serial_number': cert.get('serialNumber', ''),
                    'not_before': cert.get('notBefore', ''),
                    'not_after': cert.get('notAfter', ''),
                    'subject_alt_names': [name[1] for name in cert.get('subjectAltName', [])]
                }
            
        except Exception as e:
            logger.debug(f"Error getting SSL certificate for {host}:{port}: {e}")
        
        return None

class TechnologyDetector:
    """Web technology detection"""
    
    def __init__(self):
        # Technology fingerprints
        self.fingerprints = {
            'Apache': ['Server: Apache', 'apache'],
            'Nginx': ['Server: nginx', 'Server: nginx/'],
            'IIS': ['Server: Microsoft-IIS', 'X-Powered-By: ASP.NET'],
            'PHP': ['X-Powered-By: PHP', 'Set-Cookie: PHPSESSID'],
            'WordPress': ['wp-content/', 'wp-includes/'],
            'Django': ['csrfmiddlewaretoken', 'django'],
            'React': ['react', '_next/static/'],
            'Angular': ['ng-version', 'angular'],
            'jQuery': ['jquery', 'jQuery'],
            'Bootstrap': ['bootstrap', 'Bootstrap']
        }
    
    async def detect_technologies(self, url: str) -> List[str]:
        """Detect web technologies"""
        technologies = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    content = await response.text()
                    headers = response.headers
                    
                    # Check headers and content for technology fingerprints
                    for tech, patterns in self.fingerprints.items():
                        for pattern in patterns:
                            if (pattern.lower() in str(headers).lower() or 
                                pattern.lower() in content.lower()):
                                technologies.append(tech)
                                break
                                
        except Exception as e:
            logger.debug(f"Error detecting technologies for {url}: {e}")
        
        return list(set(technologies))

class DiscoveryAgent:
    """Main discovery agent"""
    
    def __init__(self):
        self.project_id = os.getenv('PROJECT_ID', 'tidy-computing-465909-i3')
        self.pubsub_client = pubsub_v1.PublisherClient()
        self.subscriber_client = pubsub_v1.SubscriberClient()
        self.firestore_client = firestore.Client(project=self.project_id)
        
        # Components
        self.subdomain_enumerator = SubdomainEnumerator()
        self.service_mapper = ServiceMapper()
        self.tech_detector = TechnologyDetector()
        
        # Topics
        self.discovery_topic = f"projects/{self.project_id}/topics/aetherveil-asset-discoveries"
        self.request_subscription = f"projects/{self.project_id}/subscriptions/discovery-requests"
        
    async def start_listening(self):
        """Start listening for discovery requests"""
        logger.info("Starting Discovery Agent")
        
        try:
            future = self.subscriber_client.subscribe(
                self.request_subscription,
                callback=self.handle_discovery_request,
                flow_control=pubsub_v1.types.FlowControl(max_messages=10)
            )
            
            logger.info("Listening for discovery requests")
            
            # Keep the agent running
            try:
                future.result()  # Block until subscriber stops
            except KeyboardInterrupt:
                future.cancel()
                
        except Exception as e:
            logger.error(f"Error starting discovery agent: {e}")
    
    def handle_discovery_request(self, message):
        """Handle discovery request messages"""
        try:
            data = json.loads(message.data.decode())
            asyncio.create_task(self.process_discovery_request(data))
            message.ack()
            
        except Exception as e:
            logger.error(f"Error handling discovery request: {e}")
            message.nack()
    
    async def process_discovery_request(self, request_data: Dict[str, Any]):
        """Process a discovery request"""
        try:
            sweep_id = request_data.get('sweep_id')
            domains = request_data.get('domains', [])
            discovery_types = request_data.get('discovery_types', [])
            
            logger.info(f"Processing discovery sweep: {sweep_id}")
            
            all_discoveries = []
            
            for domain in domains:
                # Subdomain enumeration
                if 'subdomain_enum' in discovery_types:
                    subdomains = await self.subdomain_enumerator.enumerate_subdomains(domain)
                    
                    for subdomain in subdomains:
                        discovery = AssetDiscovery(
                            asset_id=f"subdomain_{int(time.time())}_{hash(subdomain)}",
                            discovery_type='subdomain',
                            domain=domain,
                            subdomain=subdomain,
                            ip_address='',
                            ports=[],
                            services=[],
                            certificates=[],
                            technologies=[],
                            scope_type='in_scope',  # Assume in scope for discovery
                            confidence='high',
                            timestamp=datetime.now(timezone.utc),
                            discovery_method='dns_enumeration'
                        )
                        all_discoveries.append(discovery)
                
                # Service mapping
                if 'service_mapping' in discovery_types:
                    targets = [domain] + [subdomain for subdomain in 
                              await self.subdomain_enumerator.enumerate_subdomains(domain)]
                    
                    for target in targets[:10]:  # Limit to prevent abuse
                        service_info = await self.service_mapper.scan_services(target)
                        
                        if service_info['open_ports']:
                            discovery = AssetDiscovery(
                                asset_id=f"service_{int(time.time())}_{hash(target)}",
                                discovery_type='service',
                                domain=domain,
                                subdomain=target,
                                ip_address=service_info['ip_address'],
                                ports=service_info['open_ports'],
                                services=[s['service'] for s in service_info['services']],
                                certificates=service_info['certificates'],
                                technologies=[],
                                scope_type='in_scope',
                                confidence='high',
                                timestamp=datetime.now(timezone.utc),
                                discovery_method='port_scan'
                            )
                            all_discoveries.append(discovery)
                
                # Technology detection
                if 'asset_discovery' in discovery_types:
                    web_targets = [f"http://{domain}", f"https://{domain}"]
                    
                    for url in web_targets:
                        try:
                            technologies = await self.tech_detector.detect_technologies(url)
                            
                            if technologies:
                                discovery = AssetDiscovery(
                                    asset_id=f"tech_{int(time.time())}_{hash(url)}",
                                    discovery_type='technology',
                                    domain=domain,
                                    subdomain=urlparse(url).netloc,
                                    ip_address='',
                                    ports=[],
                                    services=[],
                                    certificates=[],
                                    technologies=technologies,
                                    scope_type='in_scope',
                                    confidence='medium',
                                    timestamp=datetime.now(timezone.utc),
                                    discovery_method='technology_detection'
                                )
                                all_discoveries.append(discovery)
                                
                        except Exception as e:
                            logger.debug(f"Error detecting technologies for {url}: {e}")
            
            # Store discoveries and publish results
            await self.store_and_publish_discoveries(all_discoveries, sweep_id)
            
        except Exception as e:
            logger.error(f"Error processing discovery request: {e}")
    
    async def store_and_publish_discoveries(self, discoveries: List[AssetDiscovery], sweep_id: str):
        """Store discoveries in Firestore and publish to Pub/Sub"""
        try:
            # Store in Firestore
            discoveries_ref = self.firestore_client.collection('asset_discoveries')
            
            for discovery in discoveries:
                doc_data = asdict(discovery)
                doc_data['sweep_id'] = sweep_id
                doc_data['timestamp'] = discovery.timestamp.isoformat()
                discoveries_ref.add(doc_data)
            
            # Publish each discovery
            for discovery in discoveries:
                await self.publish_discovery(discovery, sweep_id)
            
            logger.info(f"Stored and published {len(discoveries)} discoveries for sweep {sweep_id}")
            
        except Exception as e:
            logger.error(f"Error storing/publishing discoveries: {e}")
    
    async def publish_discovery(self, discovery: AssetDiscovery, sweep_id: str):
        """Publish discovery to Pub/Sub"""
        try:
            discovery_data = asdict(discovery)
            discovery_data['sweep_id'] = sweep_id
            discovery_data['timestamp'] = discovery.timestamp.isoformat()
            
            message = json.dumps(discovery_data).encode('utf-8')
            future = self.pubsub_client.publish(self.discovery_topic, message)
            message_id = future.result()
            
            logger.debug(f"Published discovery: {message_id}")
            
        except Exception as e:
            logger.error(f"Error publishing discovery: {e}")

async def main():
    """Main entry point"""
    agent = DiscoveryAgent()
    await agent.start_listening()

if __name__ == "__main__":
    asyncio.run(main())