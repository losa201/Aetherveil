"""
Reconnaissance Agent for Aetherveil Sentinel
Specialized agent for reconnaissance and information gathering
"""

import asyncio
import logging
import socket
import subprocess
import dns.resolver
import dns.zone
import dns.query
import dns.reversename
import whois
import nmap
import ipaddress
import json
import random
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from urllib.parse import urlparse
import aiohttp
import ssl
import re

from .base_agent import BaseAgent
from .communication import AgentCommunicator
from config.config import config

logger = logging.getLogger(__name__)

class ReconnaissanceAgent(BaseAgent):
    """Advanced reconnaissance agent for information gathering"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            agent_type="reconnaissance",
            capabilities=[
                "dns_enumeration",
                "port_scanning",
                "service_detection",
                "subdomain_discovery",
                "whois_lookup",
                "ssl_analysis",
                "banner_grabbing",
                "zone_transfer",
                "reverse_dns"
            ]
        )
        
        self.communicator = AgentCommunicator(agent_id)
        self.nm = nmap.PortScanner()
        self.dns_resolver = dns.resolver.Resolver()
        self.common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 1433, 3306, 3389, 5432, 6379, 8080, 8443]
        self.wordlists = {
            "subdomains": [
                "www", "mail", "ftp", "admin", "test", "dev", "staging", "api", "app", "portal",
                "support", "help", "blog", "news", "shop", "store", "db", "database", "backup",
                "cdn", "assets", "static", "img", "images", "media", "files", "downloads"
            ],
            "directories": [
                "admin", "administrator", "login", "wp-admin", "phpmyadmin", "cpanel", "webmail",
                "api", "test", "dev", "staging", "backup", "old", "temp", "tmp", "config",
                "includes", "uploads", "files", "assets", "static", "css", "js", "images"
            ]
        }
        
    def _register_handlers(self):
        """Register task handlers"""
        self.register_task_handler("dns_enumeration", self.dns_enumeration)
        self.register_task_handler("port_scanning", self.port_scanning)
        self.register_task_handler("service_detection", self.service_detection)
        self.register_task_handler("subdomain_discovery", self.subdomain_discovery)
        self.register_task_handler("whois_lookup", self.whois_lookup)
        self.register_task_handler("ssl_analysis", self.ssl_analysis)
        self.register_task_handler("banner_grabbing", self.banner_grabbing)
        self.register_task_handler("zone_transfer", self.zone_transfer)
        self.register_task_handler("reverse_dns", self.reverse_dns)
        self.register_task_handler("comprehensive_recon", self.comprehensive_recon)

    async def initialize(self):
        """Initialize reconnaissance agent"""
        await super().initialize()
        await self.communicator.initialize()
        
        # Configure DNS resolver
        self.dns_resolver.nameservers = ['8.8.8.8', '8.8.4.4', '1.1.1.1']
        self.dns_resolver.timeout = 5
        self.dns_resolver.lifetime = 10
        
        logger.info(f"Reconnaissance agent {self.agent_id} initialized")

    async def execute_primary_function(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive reconnaissance"""
        return await self.comprehensive_recon(target, parameters)

    async def dns_enumeration(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform DNS enumeration"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "records": {},
                "nameservers": [],
                "mail_servers": []
            }
            
            # Common DNS record types
            record_types = ['A', 'AAAA', 'CNAME', 'MX', 'NS', 'TXT', 'SOA', 'PTR']
            
            for record_type in record_types:
                try:
                    answers = dns.resolver.resolve(target, record_type)
                    results["records"][record_type] = [str(rdata) for rdata in answers]
                    
                    # Extract nameservers
                    if record_type == 'NS':
                        results["nameservers"] = [str(rdata) for rdata in answers]
                    
                    # Extract mail servers
                    if record_type == 'MX':
                        results["mail_servers"] = [
                            {"priority": rdata.preference, "exchange": str(rdata.exchange)}
                            for rdata in answers
                        ]
                        
                except dns.resolver.NXDOMAIN:
                    results["records"][record_type] = []
                except dns.resolver.NoAnswer:
                    results["records"][record_type] = []
                except Exception as e:
                    logger.warning(f"DNS query failed for {record_type}: {e}")
                    results["records"][record_type] = []
            
            # Check for wildcard DNS
            random_subdomain = f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))}.{target}"
            try:
                dns.resolver.resolve(random_subdomain, 'A')
                results["wildcard_dns"] = True
            except:
                results["wildcard_dns"] = False
            
            self.log_activity("dns_enumeration", results)
            return results
            
        except Exception as e:
            logger.error(f"DNS enumeration failed: {e}")
            return {"error": str(e)}

    async def port_scanning(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform port scanning"""
        try:
            scan_type = parameters.get("scan_type", "tcp")
            port_range = parameters.get("port_range", "1-1000")
            stealth = parameters.get("stealth", True)
            
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "scan_type": scan_type,
                "port_range": port_range,
                "open_ports": [],
                "filtered_ports": [],
                "closed_ports": []
            }
            
            # Determine nmap arguments
            nmap_args = f"-p {port_range}"
            
            if stealth:
                nmap_args += " -sS"  # SYN scan
            else:
                nmap_args += " -sT"  # TCP connect scan
            
            if scan_type == "udp":
                nmap_args += " -sU"
            
            # Add timing and stealth options
            nmap_args += " -T3 --max-retries 2"
            
            # Perform scan with jitter
            await self.sleep_with_jitter(random.uniform(1, 3))
            
            scan_result = self.nm.scan(target, arguments=nmap_args)
            
            if target in scan_result['scan']:
                host_info = scan_result['scan'][target]
                
                for proto in host_info.get('protocols', []):
                    ports = host_info[proto]
                    
                    for port, info in ports.items():
                        port_data = {
                            "port": port,
                            "protocol": proto,
                            "state": info['state'],
                            "service": info.get('name', ''),
                            "version": info.get('version', ''),
                            "product": info.get('product', '')
                        }
                        
                        if info['state'] == 'open':
                            results["open_ports"].append(port_data)
                        elif info['state'] == 'filtered':
                            results["filtered_ports"].append(port_data)
                        else:
                            results["closed_ports"].append(port_data)
            
            self.log_activity("port_scanning", results)
            return results
            
        except Exception as e:
            logger.error(f"Port scanning failed: {e}")
            return {"error": str(e)}

    async def service_detection(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform service detection"""
        try:
            ports = parameters.get("ports", self.common_ports)
            
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "services": []
            }
            
            # Convert ports to string format
            port_range = ",".join(map(str, ports))
            
            # Perform service detection scan
            nmap_args = f"-p {port_range} -sV --version-intensity 5"
            
            await self.sleep_with_jitter(random.uniform(1, 3))
            
            scan_result = self.nm.scan(target, arguments=nmap_args)
            
            if target in scan_result['scan']:
                host_info = scan_result['scan'][target]
                
                for proto in host_info.get('protocols', []):
                    ports_info = host_info[proto]
                    
                    for port, info in ports_info.items():
                        if info['state'] == 'open':
                            service_data = {
                                "port": port,
                                "protocol": proto,
                                "service": info.get('name', ''),
                                "product": info.get('product', ''),
                                "version": info.get('version', ''),
                                "extrainfo": info.get('extrainfo', ''),
                                "cpe": info.get('cpe', '')
                            }
                            
                            results["services"].append(service_data)
            
            self.log_activity("service_detection", results)
            return results
            
        except Exception as e:
            logger.error(f"Service detection failed: {e}")
            return {"error": str(e)}

    async def subdomain_discovery(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform subdomain discovery"""
        try:
            method = parameters.get("method", "bruteforce")
            wordlist = parameters.get("wordlist", self.wordlists["subdomains"])
            
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "method": method,
                "subdomains": [],
                "total_found": 0
            }
            
            found_subdomains = set()
            
            if method == "bruteforce":
                # Brute force subdomain discovery
                for subdomain in wordlist:
                    full_domain = f"{subdomain}.{target}"
                    
                    try:
                        answers = dns.resolver.resolve(full_domain, 'A')
                        ips = [str(rdata) for rdata in answers]
                        
                        found_subdomains.add(full_domain)
                        results["subdomains"].append({
                            "subdomain": full_domain,
                            "ips": ips,
                            "method": "bruteforce"
                        })
                        
                        # Add small delay to avoid rate limiting
                        await self.sleep_with_jitter(0.1)
                        
                    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                        continue
                    except Exception as e:
                        logger.warning(f"Subdomain check failed for {full_domain}: {e}")
                        continue
            
            elif method == "certificate_transparency":
                # Certificate transparency logs
                await self._discover_subdomains_ct(target, found_subdomains, results)
            
            elif method == "search_engines":
                # Search engine discovery
                await self._discover_subdomains_search(target, found_subdomains, results)
            
            results["total_found"] = len(found_subdomains)
            self.log_activity("subdomain_discovery", results)
            return results
            
        except Exception as e:
            logger.error(f"Subdomain discovery failed: {e}")
            return {"error": str(e)}

    async def _discover_subdomains_ct(self, target: str, found_subdomains: Set[str], results: Dict[str, Any]):
        """Discover subdomains using certificate transparency logs"""
        try:
            ct_urls = [
                f"https://crt.sh/?q=%.{target}&output=json",
                f"https://api.certspotter.com/v1/issuances?domain={target}&include_subdomains=true&expand=dns_names"
            ]
            
            async with aiohttp.ClientSession() as session:
                for url in ct_urls:
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                if "crt.sh" in url:
                                    for entry in data:
                                        name_value = entry.get("name_value", "")
                                        for domain in name_value.split("\n"):
                                            domain = domain.strip()
                                            if domain and domain.endswith(target) and domain not in found_subdomains:
                                                found_subdomains.add(domain)
                                                results["subdomains"].append({
                                                    "subdomain": domain,
                                                    "ips": await self._resolve_domain(domain),
                                                    "method": "certificate_transparency"
                                                })
                                
                                elif "certspotter" in url:
                                    for entry in data:
                                        dns_names = entry.get("dns_names", [])
                                        for domain in dns_names:
                                            if domain and domain.endswith(target) and domain not in found_subdomains:
                                                found_subdomains.add(domain)
                                                results["subdomains"].append({
                                                    "subdomain": domain,
                                                    "ips": await self._resolve_domain(domain),
                                                    "method": "certificate_transparency"
                                                })
                    
                    except Exception as e:
                        logger.warning(f"CT log query failed for {url}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Certificate transparency discovery failed: {e}")

    async def _discover_subdomains_search(self, target: str, found_subdomains: Set[str], results: Dict[str, Any]):
        """Discover subdomains using search engines"""
        try:
            search_queries = [
                f"site:{target}",
                f"site:*.{target}",
                f"inurl:{target}",
                f"intitle:{target}"
            ]
            
            # This is a simplified implementation
            # In practice, you would use proper search engine APIs
            for query in search_queries:
                await self.sleep_with_jitter(2)  # Respect rate limits
                
                # Placeholder for search engine integration
                # You would implement actual search engine queries here
                pass
                
        except Exception as e:
            logger.error(f"Search engine discovery failed: {e}")

    async def _resolve_domain(self, domain: str) -> List[str]:
        """Resolve domain to IP addresses"""
        try:
            answers = dns.resolver.resolve(domain, 'A')
            return [str(rdata) for rdata in answers]
        except:
            return []

    async def whois_lookup(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform WHOIS lookup"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "whois_data": {}
            }
            
            # Perform WHOIS lookup
            whois_data = whois.whois(target)
            
            # Extract relevant information
            if whois_data:
                results["whois_data"] = {
                    "domain_name": whois_data.get("domain_name"),
                    "registrar": whois_data.get("registrar"),
                    "creation_date": str(whois_data.get("creation_date", "")),
                    "expiration_date": str(whois_data.get("expiration_date", "")),
                    "updated_date": str(whois_data.get("updated_date", "")),
                    "name_servers": whois_data.get("name_servers", []),
                    "status": whois_data.get("status", []),
                    "emails": whois_data.get("emails", []),
                    "org": whois_data.get("org"),
                    "country": whois_data.get("country"),
                    "state": whois_data.get("state"),
                    "city": whois_data.get("city"),
                    "address": whois_data.get("address")
                }
            
            self.log_activity("whois_lookup", results)
            return results
            
        except Exception as e:
            logger.error(f"WHOIS lookup failed: {e}")
            return {"error": str(e)}

    async def ssl_analysis(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform SSL/TLS analysis"""
        try:
            port = parameters.get("port", 443)
            
            results = {
                "target": target,
                "port": port,
                "timestamp": datetime.utcnow().isoformat(),
                "ssl_info": {}
            }
            
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Connect and get certificate
            with socket.create_connection((target, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=target) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    
                    results["ssl_info"] = {
                        "subject": dict(x[0] for x in cert.get('subject', [])),
                        "issuer": dict(x[0] for x in cert.get('issuer', [])),
                        "version": cert.get('version'),
                        "serial_number": cert.get('serialNumber'),
                        "not_before": cert.get('notBefore'),
                        "not_after": cert.get('notAfter'),
                        "signature_algorithm": cert.get('signatureAlgorithm'),
                        "san": cert.get('subjectAltName', []),
                        "cipher": {
                            "name": cipher[0] if cipher else None,
                            "version": cipher[1] if cipher else None,
                            "bits": cipher[2] if cipher else None
                        }
                    }
            
            self.log_activity("ssl_analysis", results)
            return results
            
        except Exception as e:
            logger.error(f"SSL analysis failed: {e}")
            return {"error": str(e)}

    async def banner_grabbing(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform banner grabbing"""
        try:
            ports = parameters.get("ports", [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995])
            
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "banners": []
            }
            
            for port in ports:
                try:
                    banner = await self._grab_banner(target, port)
                    if banner:
                        results["banners"].append({
                            "port": port,
                            "banner": banner,
                            "service": self._identify_service(port, banner)
                        })
                        
                except Exception as e:
                    logger.warning(f"Banner grab failed for port {port}: {e}")
                    continue
            
            self.log_activity("banner_grabbing", results)
            return results
            
        except Exception as e:
            logger.error(f"Banner grabbing failed: {e}")
            return {"error": str(e)}

    async def _grab_banner(self, target: str, port: int, timeout: int = 5) -> Optional[str]:
        """Grab banner from specific port"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target, port),
                timeout=timeout
            )
            
            # Send appropriate probe based on port
            if port == 80:
                writer.write(b"GET / HTTP/1.1\r\nHost: " + target.encode() + b"\r\n\r\n")
            elif port == 443:
                # For HTTPS, we need SSL context
                return None
            else:
                writer.write(b"\r\n")
            
            await writer.drain()
            
            # Read response
            banner = await asyncio.wait_for(
                reader.read(1024),
                timeout=timeout
            )
            
            writer.close()
            await writer.wait_closed()
            
            return banner.decode('utf-8', errors='ignore').strip()
            
        except Exception:
            return None

    def _identify_service(self, port: int, banner: str) -> str:
        """Identify service based on port and banner"""
        service_signatures = {
            21: ["FTP", "FileZilla", "vsftpd", "ProFTPD"],
            22: ["SSH", "OpenSSH", "Dropbear"],
            23: ["Telnet"],
            25: ["SMTP", "Postfix", "Sendmail", "Exchange"],
            53: ["DNS", "BIND", "dnsmasq"],
            80: ["HTTP", "Apache", "nginx", "IIS"],
            110: ["POP3"],
            143: ["IMAP"],
            443: ["HTTPS", "SSL", "TLS"],
            993: ["IMAPS"],
            995: ["POP3S"]
        }
        
        if port in service_signatures:
            for service in service_signatures[port]:
                if service.lower() in banner.lower():
                    return service
        
        return "unknown"

    async def zone_transfer(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt DNS zone transfer"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "zone_transfer": False,
                "records": []
            }
            
            # Get nameservers
            try:
                ns_answers = dns.resolver.resolve(target, 'NS')
                nameservers = [str(rdata) for rdata in ns_answers]
            except:
                return {"error": "Could not resolve nameservers"}
            
            # Try zone transfer with each nameserver
            for ns in nameservers:
                try:
                    zone = dns.zone.from_xfr(dns.query.xfr(ns, target))
                    results["zone_transfer"] = True
                    
                    for name, node in zone.nodes.items():
                        for rdataset in node.rdatasets:
                            for rdata in rdataset:
                                results["records"].append({
                                    "name": str(name),
                                    "type": dns.rdatatype.to_text(rdataset.rdtype),
                                    "value": str(rdata)
                                })
                    
                    break  # Success, no need to try other nameservers
                    
                except Exception as e:
                    logger.warning(f"Zone transfer failed for {ns}: {e}")
                    continue
            
            self.log_activity("zone_transfer", results)
            return results
            
        except Exception as e:
            logger.error(f"Zone transfer failed: {e}")
            return {"error": str(e)}

    async def reverse_dns(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reverse DNS lookup"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "reverse_dns": []
            }
            
            # If target is IP address, do reverse lookup
            if self._is_ip_address(target):
                try:
                    reverse_name = dns.reversename.from_address(target)
                    answers = dns.resolver.resolve(reverse_name, 'PTR')
                    results["reverse_dns"] = [str(rdata) for rdata in answers]
                except:
                    results["reverse_dns"] = []
            
            # If target is domain, resolve to IPs then reverse lookup
            else:
                try:
                    answers = dns.resolver.resolve(target, 'A')
                    for rdata in answers:
                        ip = str(rdata)
                        try:
                            reverse_name = dns.reversename.from_address(ip)
                            reverse_answers = dns.resolver.resolve(reverse_name, 'PTR')
                            results["reverse_dns"].extend([str(r) for r in reverse_answers])
                        except:
                            continue
                except:
                    results["reverse_dns"] = []
            
            self.log_activity("reverse_dns", results)
            return results
            
        except Exception as e:
            logger.error(f"Reverse DNS failed: {e}")
            return {"error": str(e)}

    def _is_ip_address(self, target: str) -> bool:
        """Check if target is an IP address"""
        try:
            ipaddress.ip_address(target)
            return True
        except ValueError:
            return False

    async def comprehensive_recon(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive reconnaissance"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "reconnaissance": {}
            }
            
            # Perform all reconnaissance tasks
            tasks = [
                ("dns_enumeration", self.dns_enumeration),
                ("whois_lookup", self.whois_lookup),
                ("subdomain_discovery", self.subdomain_discovery),
                ("port_scanning", self.port_scanning),
                ("service_detection", self.service_detection),
                ("banner_grabbing", self.banner_grabbing),
                ("ssl_analysis", self.ssl_analysis),
                ("zone_transfer", self.zone_transfer),
                ("reverse_dns", self.reverse_dns)
            ]
            
            for task_name, task_func in tasks:
                try:
                    task_result = await task_func(target, parameters)
                    results["reconnaissance"][task_name] = task_result
                    
                    # Add delay between tasks for stealth
                    await self.sleep_with_jitter(random.uniform(2, 5))
                    
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    results["reconnaissance"][task_name] = {"error": str(e)}
            
            # Send intelligence data to coordinator
            await self.send_intelligence_data(results)
            
            self.log_activity("comprehensive_recon", results)
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive reconnaissance failed: {e}")
            return {"error": str(e)}

    async def shutdown(self):
        """Shutdown reconnaissance agent"""
        try:
            await self.communicator.shutdown()
            await super().shutdown()
        except Exception as e:
            logger.error(f"Error shutting down reconnaissance agent: {e}")

def main():
    """Main function for running reconnaissance agent"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python reconnaissance_agent.py <agent_id>")
        sys.exit(1)
    
    agent_id = sys.argv[1]
    agent = ReconnaissanceAgent(agent_id)
    
    async def run_agent():
        try:
            await agent.initialize()
            
            # Keep agent running
            while agent.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Reconnaissance agent shutting down...")
        except Exception as e:
            logger.error(f"Agent error: {e}")
        finally:
            await agent.shutdown()
    
    asyncio.run(run_agent())

if __name__ == "__main__":
    main()