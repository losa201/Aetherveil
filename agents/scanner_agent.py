"""
Scanner Agent for Aetherveil Sentinel
Specialized agent for vulnerability scanning and security assessment
"""

import asyncio
import logging
import json
import random
import re
import ssl
import socket
import subprocess
import hashlib
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from urllib.parse import urlparse, urljoin
import aiohttp
import nmap
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_agent import BaseAgent
from .communication import AgentCommunicator
from config.config import config

logger = logging.getLogger(__name__)

class ScannerAgent(BaseAgent):
    """Advanced scanner agent for vulnerability assessment"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            agent_type="scanner",
            capabilities=[
                "vulnerability_scanning",
                "web_scanning",
                "network_scanning",
                "ssl_analysis",
                "configuration_analysis",
                "cve_scanning",
                "web_crawler",
                "directory_bruteforce",
                "injection_testing",
                "authentication_testing"
            ]
        )
        
        self.communicator = AgentCommunicator(agent_id)
        self.nm = nmap.PortScanner()
        self.session = None
        self.vulnerability_db = self._load_vulnerability_database()
        self.web_payloads = self._load_web_payloads()
        self.common_paths = [
            "/admin", "/administrator", "/login", "/wp-admin", "/phpmyadmin",
            "/cpanel", "/webmail", "/api", "/test", "/dev", "/staging",
            "/backup", "/old", "/temp", "/tmp", "/config", "/includes",
            "/uploads", "/files", "/assets", "/static", "/css", "/js",
            "/images", "/docs", "/documentation", "/readme", "/changelog"
        ]
        
    def _register_handlers(self):
        """Register task handlers"""
        self.register_task_handler("vulnerability_scanning", self.vulnerability_scanning)
        self.register_task_handler("web_scanning", self.web_scanning)
        self.register_task_handler("network_scanning", self.network_scanning)
        self.register_task_handler("ssl_analysis", self.ssl_analysis)
        self.register_task_handler("configuration_analysis", self.configuration_analysis)
        self.register_task_handler("cve_scanning", self.cve_scanning)
        self.register_task_handler("web_crawler", self.web_crawler)
        self.register_task_handler("directory_bruteforce", self.directory_bruteforce)
        self.register_task_handler("injection_testing", self.injection_testing)
        self.register_task_handler("authentication_testing", self.authentication_testing)
        self.register_task_handler("comprehensive_scan", self.comprehensive_scan)

    async def initialize(self):
        """Initialize scanner agent"""
        await super().initialize()
        await self.communicator.initialize()
        
        # Setup HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Scanner agent {self.agent_id} initialized")

    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Load vulnerability database"""
        # This would typically load from a file or database
        return {
            "web_vulns": {
                "sql_injection": {
                    "payloads": ["'", "1' OR '1'='1", "'; DROP TABLE users; --"],
                    "indicators": ["SQL", "mysql", "syntax error", "ORA-", "PostgreSQL"]
                },
                "xss": {
                    "payloads": ["<script>alert('XSS')</script>", "<img src=x onerror=alert('XSS')>"],
                    "indicators": ["<script>", "alert(", "onerror="]
                },
                "lfi": {
                    "payloads": ["../../../../etc/passwd", "..\\..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"],
                    "indicators": ["root:", "daemon:", "bin:", "sys:"]
                },
                "command_injection": {
                    "payloads": ["; ls", "| whoami", "&& id"],
                    "indicators": ["uid=", "gid=", "groups="]
                }
            },
            "service_vulns": {
                "ftp": {
                    "anonymous_login": {"method": "anonymous", "indicators": ["230", "Login successful"]},
                    "bounce_attack": {"method": "port", "indicators": ["200 PORT command successful"]}
                },
                "ssh": {
                    "weak_ciphers": {"method": "cipher", "indicators": ["arcfour", "des", "rc4"]},
                    "default_creds": {"method": "auth", "credentials": [("root", "root"), ("admin", "admin")]}
                },
                "smb": {
                    "null_sessions": {"method": "null", "indicators": ["STATUS_SUCCESS"]},
                    "eternal_blue": {"method": "exploit", "indicators": ["MS17-010"]}
                }
            }
        }

    def _load_web_payloads(self) -> Dict[str, List[str]]:
        """Load web testing payloads"""
        return {
            "sql_injection": [
                "'", "\"", "1' OR '1'='1", "1\" OR \"1\"=\"1", "'; DROP TABLE users; --",
                "1' UNION SELECT 1,2,3--", "1' AND 1=1--", "1' AND 1=2--"
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "javascript:alert('XSS')",
                "<iframe src=javascript:alert('XSS')>"
            ],
            "lfi": [
                "../../../../etc/passwd",
                "..\\..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "/etc/passwd",
                "C:\\windows\\system32\\drivers\\etc\\hosts",
                "php://filter/convert.base64-encode/resource=index.php"
            ],
            "command_injection": [
                "; ls", "| whoami", "&& id", "; cat /etc/passwd",
                "| cat /etc/passwd", "&& cat /etc/passwd"
            ]
        }

    async def execute_primary_function(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive vulnerability scanning"""
        return await self.comprehensive_scan(target, parameters)

    async def vulnerability_scanning(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform vulnerability scanning using nmap scripts"""
        try:
            scan_type = parameters.get("scan_type", "default")
            
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "scan_type": scan_type,
                "vulnerabilities": []
            }
            
            # Vulnerability scanning scripts
            vuln_scripts = [
                "vuln",
                "auth",
                "brute",
                "default",
                "discovery",
                "exploit",
                "external",
                "fuzzer",
                "malware",
                "safe"
            ]
            
            script_args = f"--script={','.join(vuln_scripts)}"
            nmap_args = f"{script_args} --script-args=unsafe=1"
            
            # Add stealth options
            if parameters.get("stealth", True):
                nmap_args += " -T3 -f"
            
            # Perform scan
            await self.sleep_with_jitter(random.uniform(2, 5))
            
            scan_result = self.nm.scan(target, arguments=nmap_args)
            
            if target in scan_result['scan']:
                host_info = scan_result['scan'][target]
                
                # Extract vulnerabilities from script output
                for proto in host_info.get('protocols', []):
                    ports = host_info[proto]
                    
                    for port, port_info in ports.items():
                        if 'script' in port_info:
                            for script_name, script_output in port_info['script'].items():
                                vulnerability = self._parse_nmap_script_output(
                                    script_name, script_output, target, port
                                )
                                if vulnerability:
                                    results["vulnerabilities"].append(vulnerability)
            
            # Additional custom vulnerability checks
            custom_vulns = await self._perform_custom_vulnerability_checks(target, parameters)
            results["vulnerabilities"].extend(custom_vulns)
            
            # Send vulnerability alerts
            for vuln in results["vulnerabilities"]:
                if vuln.get("severity") in ["high", "critical"]:
                    await self.send_vulnerability_found(vuln)
            
            self.log_activity("vulnerability_scanning", results)
            return results
            
        except Exception as e:
            logger.error(f"Vulnerability scanning failed: {e}")
            return {"error": str(e)}

    def _parse_nmap_script_output(self, script_name: str, output: str, target: str, port: int) -> Optional[Dict[str, Any]]:
        """Parse nmap script output to extract vulnerabilities"""
        try:
            vulnerability = {
                "id": hashlib.md5(f"{script_name}{target}{port}".encode()).hexdigest(),
                "target": target,
                "port": port,
                "script": script_name,
                "output": output,
                "severity": "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Classify severity based on script name and output
            if any(keyword in script_name.lower() for keyword in ['vuln', 'exploit', 'backdoor']):
                vulnerability["severity"] = "high"
            elif any(keyword in output.lower() for keyword in ['vulnerable', 'exploit', 'backdoor']):
                vulnerability["severity"] = "high"
            elif any(keyword in script_name.lower() for keyword in ['auth', 'brute', 'default']):
                vulnerability["severity"] = "medium"
            else:
                vulnerability["severity"] = "low"
            
            # Extract CVE if present
            cve_match = re.search(r'CVE-\d{4}-\d{4,7}', output)
            if cve_match:
                vulnerability["cve"] = cve_match.group()
            
            return vulnerability
            
        except Exception as e:
            logger.error(f"Failed to parse script output: {e}")
            return None

    async def _perform_custom_vulnerability_checks(self, target: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform custom vulnerability checks"""
        vulnerabilities = []
        
        try:
            # Check for common service vulnerabilities
            service_vulns = await self._check_service_vulnerabilities(target)
            vulnerabilities.extend(service_vulns)
            
            # Check for SSL/TLS vulnerabilities
            ssl_vulns = await self._check_ssl_vulnerabilities(target)
            vulnerabilities.extend(ssl_vulns)
            
            # Check for web application vulnerabilities
            web_vulns = await self._check_web_vulnerabilities(target)
            vulnerabilities.extend(web_vulns)
            
        except Exception as e:
            logger.error(f"Custom vulnerability checks failed: {e}")
        
        return vulnerabilities

    async def _check_service_vulnerabilities(self, target: str) -> List[Dict[str, Any]]:
        """Check for service-specific vulnerabilities"""
        vulnerabilities = []
        
        # This is a simplified implementation
        # In practice, you would check for specific service vulnerabilities
        common_services = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
        
        for port in common_services:
            try:
                # Check if port is open
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((target, port))
                sock.close()
                
                if result == 0:
                    # Port is open, check for vulnerabilities
                    vuln = await self._check_port_vulnerabilities(target, port)
                    if vuln:
                        vulnerabilities.append(vuln)
                        
            except Exception as e:
                logger.warning(f"Service check failed for port {port}: {e}")
                continue
        
        return vulnerabilities

    async def _check_port_vulnerabilities(self, target: str, port: int) -> Optional[Dict[str, Any]]:
        """Check for vulnerabilities on specific port"""
        try:
            # Example: Check for anonymous FTP
            if port == 21:
                return await self._check_ftp_anonymous(target, port)
            
            # Example: Check for SSH weak ciphers
            elif port == 22:
                return await self._check_ssh_weak_ciphers(target, port)
            
            # Example: Check for HTTP security headers
            elif port in [80, 443]:
                return await self._check_http_security_headers(target, port)
            
        except Exception as e:
            logger.warning(f"Port vulnerability check failed for {port}: {e}")
        
        return None

    async def _check_ftp_anonymous(self, target: str, port: int) -> Optional[Dict[str, Any]]:
        """Check for FTP anonymous login"""
        try:
            import ftplib
            
            ftp = ftplib.FTP()
            ftp.connect(target, port, timeout=10)
            
            try:
                ftp.login('anonymous', 'anonymous@example.com')
                ftp.quit()
                
                return {
                    "id": hashlib.md5(f"ftp_anonymous_{target}_{port}".encode()).hexdigest(),
                    "title": "FTP Anonymous Login Enabled",
                    "description": "FTP server allows anonymous login",
                    "target": target,
                    "port": port,
                    "severity": "medium",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except ftplib.error_perm:
                ftp.quit()
                return None
                
        except Exception:
            return None

    async def _check_ssh_weak_ciphers(self, target: str, port: int) -> Optional[Dict[str, Any]]:
        """Check for SSH weak ciphers"""
        try:
            # This is a simplified check
            # In practice, you would use a proper SSH client library
            cmd = f"ssh -o ConnectTimeout=5 -o BatchMode=yes {target} -p {port} 2>&1"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if "weak cipher" in result.stderr.lower():
                return {
                    "id": hashlib.md5(f"ssh_weak_cipher_{target}_{port}".encode()).hexdigest(),
                    "title": "SSH Weak Cipher",
                    "description": "SSH server uses weak ciphers",
                    "target": target,
                    "port": port,
                    "severity": "medium",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception:
            pass
        
        return None

    async def _check_http_security_headers(self, target: str, port: int) -> Optional[Dict[str, Any]]:
        """Check for HTTP security headers"""
        try:
            protocol = "https" if port == 443 else "http"
            url = f"{protocol}://{target}:{port}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    headers = response.headers
                    
                    missing_headers = []
                    security_headers = [
                        "X-Frame-Options",
                        "X-Content-Type-Options",
                        "X-XSS-Protection",
                        "Strict-Transport-Security",
                        "Content-Security-Policy"
                    ]
                    
                    for header in security_headers:
                        if header not in headers:
                            missing_headers.append(header)
                    
                    if missing_headers:
                        return {
                            "id": hashlib.md5(f"missing_headers_{target}_{port}".encode()).hexdigest(),
                            "title": "Missing Security Headers",
                            "description": f"Missing security headers: {', '.join(missing_headers)}",
                            "target": target,
                            "port": port,
                            "severity": "low",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
        except Exception:
            pass
        
        return None

    async def _check_ssl_vulnerabilities(self, target: str) -> List[Dict[str, Any]]:
        """Check for SSL/TLS vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Check SSL/TLS configuration
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((target, 443), timeout=10) as sock:
                with ssl_context.wrap_socket(sock, server_hostname=target) as ssock:
                    # Check for weak protocols
                    if ssock.version() in ['TLSv1', 'TLSv1.1', 'SSLv2', 'SSLv3']:
                        vulnerabilities.append({
                            "id": hashlib.md5(f"weak_tls_{target}".encode()).hexdigest(),
                            "title": "Weak TLS Protocol",
                            "description": f"Server supports weak TLS protocol: {ssock.version()}",
                            "target": target,
                            "port": 443,
                            "severity": "medium",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    # Check cipher strength
                    cipher = ssock.cipher()
                    if cipher and cipher[2] < 128:
                        vulnerabilities.append({
                            "id": hashlib.md5(f"weak_cipher_{target}".encode()).hexdigest(),
                            "title": "Weak SSL Cipher",
                            "description": f"Server uses weak cipher: {cipher[0]} ({cipher[2]} bits)",
                            "target": target,
                            "port": 443,
                            "severity": "medium",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
        except Exception as e:
            logger.warning(f"SSL vulnerability check failed: {e}")
        
        return vulnerabilities

    async def _check_web_vulnerabilities(self, target: str) -> List[Dict[str, Any]]:
        """Check for web application vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Check common web vulnerabilities
            for port in [80, 443]:
                protocol = "https" if port == 443 else "http"
                base_url = f"{protocol}://{target}:{port}"
                
                # Check if web server is running
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(base_url, timeout=10) as response:
                            if response.status == 200:
                                # Perform web vulnerability checks
                                web_vulns = await self._perform_web_vulnerability_checks(base_url, session)
                                vulnerabilities.extend(web_vulns)
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Web vulnerability check failed: {e}")
        
        return vulnerabilities

    async def _perform_web_vulnerability_checks(self, base_url: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Perform web vulnerability checks"""
        vulnerabilities = []
        
        try:
            # Check for directory traversal
            lfi_vulns = await self._check_lfi_vulnerabilities(base_url, session)
            vulnerabilities.extend(lfi_vulns)
            
            # Check for XSS vulnerabilities
            xss_vulns = await self._check_xss_vulnerabilities(base_url, session)
            vulnerabilities.extend(xss_vulns)
            
            # Check for SQL injection
            sql_vulns = await self._check_sql_injection(base_url, session)
            vulnerabilities.extend(sql_vulns)
            
        except Exception as e:
            logger.warning(f"Web vulnerability checks failed: {e}")
        
        return vulnerabilities

    async def _check_lfi_vulnerabilities(self, base_url: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Check for Local File Inclusion vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Test common LFI parameters
            test_params = ['file', 'page', 'include', 'path', 'document', 'folder', 'root', 'dir']
            lfi_payloads = self.web_payloads["lfi"]
            
            for param in test_params:
                for payload in lfi_payloads:
                    test_url = f"{base_url}/?{param}={payload}"
                    
                    try:
                        async with session.get(test_url, timeout=5) as response:
                            content = await response.text()
                            
                            # Check for LFI indicators
                            if any(indicator in content.lower() for indicator in ["root:", "daemon:", "bin:", "sys:"]):
                                vulnerabilities.append({
                                    "id": hashlib.md5(f"lfi_{base_url}_{param}".encode()).hexdigest(),
                                    "title": "Local File Inclusion",
                                    "description": f"LFI vulnerability found in parameter: {param}",
                                    "target": base_url,
                                    "parameter": param,
                                    "payload": payload,
                                    "severity": "high",
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                break
                                
                    except Exception:
                        continue
                        
                    await self.sleep_with_jitter(0.5)
                    
        except Exception as e:
            logger.warning(f"LFI check failed: {e}")
        
        return vulnerabilities

    async def _check_xss_vulnerabilities(self, base_url: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Check for Cross-Site Scripting vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Test common XSS parameters
            test_params = ['q', 'query', 'search', 'name', 'comment', 'message', 'input', 'data']
            xss_payloads = self.web_payloads["xss"]
            
            for param in test_params:
                for payload in xss_payloads:
                    test_url = f"{base_url}/?{param}={payload}"
                    
                    try:
                        async with session.get(test_url, timeout=5) as response:
                            content = await response.text()
                            
                            # Check for XSS indicators
                            if payload in content and not any(escape in content for escape in ["&lt;", "&gt;", "&amp;"]):
                                vulnerabilities.append({
                                    "id": hashlib.md5(f"xss_{base_url}_{param}".encode()).hexdigest(),
                                    "title": "Cross-Site Scripting (XSS)",
                                    "description": f"XSS vulnerability found in parameter: {param}",
                                    "target": base_url,
                                    "parameter": param,
                                    "payload": payload,
                                    "severity": "medium",
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                break
                                
                    except Exception:
                        continue
                        
                    await self.sleep_with_jitter(0.5)
                    
        except Exception as e:
            logger.warning(f"XSS check failed: {e}")
        
        return vulnerabilities

    async def _check_sql_injection(self, base_url: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Check for SQL injection vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Test common SQL injection parameters
            test_params = ['id', 'user', 'username', 'password', 'email', 'search', 'query', 'page']
            sql_payloads = self.web_payloads["sql_injection"]
            
            for param in test_params:
                for payload in sql_payloads:
                    test_url = f"{base_url}/?{param}={payload}"
                    
                    try:
                        async with session.get(test_url, timeout=5) as response:
                            content = await response.text()
                            
                            # Check for SQL error indicators
                            sql_errors = [
                                "mysql", "sql", "syntax error", "ora-", "postgresql",
                                "microsoft", "driver", "odbc", "warning", "error"
                            ]
                            
                            if any(error in content.lower() for error in sql_errors):
                                vulnerabilities.append({
                                    "id": hashlib.md5(f"sql_{base_url}_{param}".encode()).hexdigest(),
                                    "title": "SQL Injection",
                                    "description": f"SQL injection vulnerability found in parameter: {param}",
                                    "target": base_url,
                                    "parameter": param,
                                    "payload": payload,
                                    "severity": "high",
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                break
                                
                    except Exception:
                        continue
                        
                    await self.sleep_with_jitter(0.5)
                    
        except Exception as e:
            logger.warning(f"SQL injection check failed: {e}")
        
        return vulnerabilities

    async def web_scanning(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web application scanning"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "web_scan": {
                    "directories": [],
                    "files": [],
                    "technologies": [],
                    "vulnerabilities": []
                }
            }
            
            # Perform directory brute force
            directories = await self.directory_bruteforce(target, parameters)
            results["web_scan"]["directories"] = directories.get("directories", [])
            
            # Perform web crawling
            crawl_results = await self.web_crawler(target, parameters)
            results["web_scan"]["files"] = crawl_results.get("files", [])
            
            # Perform injection testing
            injection_results = await self.injection_testing(target, parameters)
            results["web_scan"]["vulnerabilities"] = injection_results.get("vulnerabilities", [])
            
            self.log_activity("web_scanning", results)
            return results
            
        except Exception as e:
            logger.error(f"Web scanning failed: {e}")
            return {"error": str(e)}

    async def directory_bruteforce(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform directory brute force"""
        try:
            wordlist = parameters.get("wordlist", self.common_paths)
            
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "directories": []
            }
            
            # Determine protocol
            protocol = "https" if parameters.get("ssl", False) else "http"
            port = parameters.get("port", 443 if protocol == "https" else 80)
            base_url = f"{protocol}://{target}:{port}"
            
            async with aiohttp.ClientSession() as session:
                for path in wordlist:
                    url = f"{base_url}{path}"
                    
                    try:
                        async with session.get(url, timeout=5) as response:
                            if response.status in [200, 301, 302, 401, 403]:
                                results["directories"].append({
                                    "path": path,
                                    "url": url,
                                    "status": response.status,
                                    "size": response.headers.get('Content-Length', 0)
                                })
                                
                    except Exception:
                        continue
                        
                    await self.sleep_with_jitter(0.2)
            
            self.log_activity("directory_bruteforce", results)
            return results
            
        except Exception as e:
            logger.error(f"Directory brute force failed: {e}")
            return {"error": str(e)}

    async def web_crawler(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web crawling"""
        try:
            max_depth = parameters.get("max_depth", 3)
            
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "files": [],
                "forms": [],
                "links": []
            }
            
            # Determine protocol
            protocol = "https" if parameters.get("ssl", False) else "http"
            port = parameters.get("port", 443 if protocol == "https" else 80)
            base_url = f"{protocol}://{target}:{port}"
            
            visited = set()
            to_visit = [base_url]
            
            async with aiohttp.ClientSession() as session:
                for depth in range(max_depth):
                    current_level = to_visit.copy()
                    to_visit.clear()
                    
                    for url in current_level:
                        if url in visited:
                            continue
                            
                        visited.add(url)
                        
                        try:
                            async with session.get(url, timeout=10) as response:
                                if response.status == 200:
                                    content = await response.text()
                                    soup = BeautifulSoup(content, 'html.parser')
                                    
                                    # Extract links
                                    for link in soup.find_all('a', href=True):
                                        href = link['href']
                                        full_url = urljoin(url, href)
                                        
                                        if full_url.startswith(base_url) and full_url not in visited:
                                            to_visit.append(full_url)
                                            results["links"].append(full_url)
                                    
                                    # Extract forms
                                    for form in soup.find_all('form'):
                                        form_data = {
                                            "action": form.get('action', ''),
                                            "method": form.get('method', 'GET'),
                                            "inputs": []
                                        }
                                        
                                        for input_field in form.find_all('input'):
                                            form_data["inputs"].append({
                                                "name": input_field.get('name', ''),
                                                "type": input_field.get('type', 'text'),
                                                "value": input_field.get('value', '')
                                            })
                                        
                                        results["forms"].append(form_data)
                                    
                                    # Record file
                                    results["files"].append({
                                        "url": url,
                                        "status": response.status,
                                        "size": len(content),
                                        "title": soup.title.string if soup.title else ""
                                    })
                                    
                        except Exception as e:
                            logger.warning(f"Crawl failed for {url}: {e}")
                            continue
                            
                        await self.sleep_with_jitter(0.5)
            
            self.log_activity("web_crawler", results)
            return results
            
        except Exception as e:
            logger.error(f"Web crawler failed: {e}")
            return {"error": str(e)}

    async def injection_testing(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform injection testing"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "vulnerabilities": []
            }
            
            # Determine protocol
            protocol = "https" if parameters.get("ssl", False) else "http"
            port = parameters.get("port", 443 if protocol == "https" else 80)
            base_url = f"{protocol}://{target}:{port}"
            
            # Test for various injection types
            injection_types = ["sql_injection", "xss", "lfi", "command_injection"]
            
            for injection_type in injection_types:
                vulns = await self._test_injection_type(base_url, injection_type)
                results["vulnerabilities"].extend(vulns)
            
            self.log_activity("injection_testing", results)
            return results
            
        except Exception as e:
            logger.error(f"Injection testing failed: {e}")
            return {"error": str(e)}

    async def _test_injection_type(self, base_url: str, injection_type: str) -> List[Dict[str, Any]]:
        """Test for specific injection type"""
        vulnerabilities = []
        
        try:
            payloads = self.web_payloads.get(injection_type, [])
            test_params = ['id', 'user', 'search', 'query', 'page', 'file', 'name', 'data']
            
            async with aiohttp.ClientSession() as session:
                for param in test_params:
                    for payload in payloads:
                        test_url = f"{base_url}/?{param}={payload}"
                        
                        try:
                            async with session.get(test_url, timeout=5) as response:
                                content = await response.text()
                                
                                # Check for injection indicators
                                if self._check_injection_indicators(content, injection_type, payload):
                                    vulnerabilities.append({
                                        "id": hashlib.md5(f"{injection_type}_{base_url}_{param}".encode()).hexdigest(),
                                        "title": injection_type.replace("_", " ").title(),
                                        "description": f"{injection_type} vulnerability found in parameter: {param}",
                                        "target": base_url,
                                        "parameter": param,
                                        "payload": payload,
                                        "severity": self._get_injection_severity(injection_type),
                                        "timestamp": datetime.utcnow().isoformat()
                                    })
                                    break
                                    
                        except Exception:
                            continue
                            
                        await self.sleep_with_jitter(0.3)
                        
        except Exception as e:
            logger.warning(f"Injection testing failed for {injection_type}: {e}")
        
        return vulnerabilities

    def _check_injection_indicators(self, content: str, injection_type: str, payload: str) -> bool:
        """Check for injection indicators in response"""
        content_lower = content.lower()
        
        if injection_type == "sql_injection":
            indicators = ["mysql", "sql", "syntax error", "ora-", "postgresql", "microsoft", "driver", "odbc"]
            return any(indicator in content_lower for indicator in indicators)
        
        elif injection_type == "xss":
            return payload in content and not any(escape in content for escape in ["&lt;", "&gt;", "&amp;"])
        
        elif injection_type == "lfi":
            indicators = ["root:", "daemon:", "bin:", "sys:"]
            return any(indicator in content_lower for indicator in indicators)
        
        elif injection_type == "command_injection":
            indicators = ["uid=", "gid=", "groups="]
            return any(indicator in content_lower for indicator in indicators)
        
        return False

    def _get_injection_severity(self, injection_type: str) -> str:
        """Get severity level for injection type"""
        severity_map = {
            "sql_injection": "high",
            "command_injection": "high",
            "lfi": "high",
            "xss": "medium"
        }
        return severity_map.get(injection_type, "medium")

    async def network_scanning(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform network scanning"""
        try:
            scan_type = parameters.get("scan_type", "tcp")
            
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "network_scan": {}
            }
            
            # Perform comprehensive network scan
            nmap_args = "-sS -sV -O --script=default,vuln"
            
            if scan_type == "stealth":
                nmap_args += " -f -T2"
            elif scan_type == "aggressive":
                nmap_args += " -A -T4"
            
            scan_result = self.nm.scan(target, arguments=nmap_args)
            results["network_scan"] = scan_result
            
            self.log_activity("network_scanning", results)
            return results
            
        except Exception as e:
            logger.error(f"Network scanning failed: {e}")
            return {"error": str(e)}

    async def ssl_analysis(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform SSL/TLS analysis"""
        try:
            port = parameters.get("port", 443)
            
            results = {
                "target": target,
                "port": port,
                "timestamp": datetime.utcnow().isoformat(),
                "ssl_analysis": {}
            }
            
            # Perform SSL/TLS analysis using nmap
            nmap_args = f"-p {port} --script=ssl-enum-ciphers,ssl-cert,ssl-date,ssl-heartbleed,ssl-poodle,ssl-dh-params"
            
            scan_result = self.nm.scan(target, arguments=nmap_args)
            results["ssl_analysis"] = scan_result
            
            self.log_activity("ssl_analysis", results)
            return results
            
        except Exception as e:
            logger.error(f"SSL analysis failed: {e}")
            return {"error": str(e)}

    async def configuration_analysis(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform configuration analysis"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "configuration_issues": []
            }
            
            # Check for common configuration issues
            config_checks = [
                self._check_default_credentials,
                self._check_directory_listing,
                self._check_information_disclosure,
                self._check_security_misconfigurations
            ]
            
            for check in config_checks:
                try:
                    issues = await check(target, parameters)
                    results["configuration_issues"].extend(issues)
                except Exception as e:
                    logger.warning(f"Configuration check failed: {e}")
                    continue
            
            self.log_activity("configuration_analysis", results)
            return results
            
        except Exception as e:
            logger.error(f"Configuration analysis failed: {e}")
            return {"error": str(e)}

    async def _check_default_credentials(self, target: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for default credentials"""
        # This is a simplified implementation
        return []

    async def _check_directory_listing(self, target: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for directory listing"""
        # This is a simplified implementation
        return []

    async def _check_information_disclosure(self, target: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for information disclosure"""
        # This is a simplified implementation
        return []

    async def _check_security_misconfigurations(self, target: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for security misconfigurations"""
        # This is a simplified implementation
        return []

    async def cve_scanning(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform CVE scanning"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "cve_scan": {}
            }
            
            # Perform CVE scanning using nmap
            nmap_args = "--script=vuln,exploit"
            
            scan_result = self.nm.scan(target, arguments=nmap_args)
            results["cve_scan"] = scan_result
            
            self.log_activity("cve_scanning", results)
            return results
            
        except Exception as e:
            logger.error(f"CVE scanning failed: {e}")
            return {"error": str(e)}

    async def authentication_testing(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform authentication testing"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "authentication_issues": []
            }
            
            # Test for weak authentication
            auth_issues = await self._test_weak_authentication(target, parameters)
            results["authentication_issues"].extend(auth_issues)
            
            self.log_activity("authentication_testing", results)
            return results
            
        except Exception as e:
            logger.error(f"Authentication testing failed: {e}")
            return {"error": str(e)}

    async def _test_weak_authentication(self, target: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test for weak authentication"""
        # This is a simplified implementation
        return []

    async def comprehensive_scan(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive scanning"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "comprehensive_scan": {}
            }
            
            # Perform all scanning tasks
            scan_tasks = [
                ("vulnerability_scanning", self.vulnerability_scanning),
                ("web_scanning", self.web_scanning),
                ("network_scanning", self.network_scanning),
                ("ssl_analysis", self.ssl_analysis),
                ("configuration_analysis", self.configuration_analysis),
                ("cve_scanning", self.cve_scanning),
                ("authentication_testing", self.authentication_testing)
            ]
            
            for task_name, task_func in scan_tasks:
                try:
                    task_result = await task_func(target, parameters)
                    results["comprehensive_scan"][task_name] = task_result
                    
                    # Add delay between tasks
                    await self.sleep_with_jitter(random.uniform(3, 8))
                    
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    results["comprehensive_scan"][task_name] = {"error": str(e)}
            
            # Send intelligence data to coordinator
            await self.send_intelligence_data(results)
            
            self.log_activity("comprehensive_scan", results)
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive scan failed: {e}")
            return {"error": str(e)}

    async def shutdown(self):
        """Shutdown scanner agent"""
        try:
            if self.session:
                self.session.close()
            await self.communicator.shutdown()
            await super().shutdown()
        except Exception as e:
            logger.error(f"Error shutting down scanner agent: {e}")

def main():
    """Main function for running scanner agent"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python scanner_agent.py <agent_id>")
        sys.exit(1)
    
    agent_id = sys.argv[1]
    agent = ScannerAgent(agent_id)
    
    async def run_agent():
        try:
            await agent.initialize()
            
            # Keep agent running
            while agent.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Scanner agent shutting down...")
        except Exception as e:
            logger.error(f"Agent error: {e}")
        finally:
            await agent.shutdown()
    
    asyncio.run(run_agent())

if __name__ == "__main__":
    main()