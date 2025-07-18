"""
OSINT Agent for Aetherveil Sentinel
Specialized agent for Open Source Intelligence gathering
"""

import asyncio
import logging
import json
import random
import re
import hashlib
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import aiohttp
import dns.resolver
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_agent import BaseAgent
from .communication import AgentCommunicator
from config.config import config

logger = logging.getLogger(__name__)

class OSINTAgent(BaseAgent):
    """Advanced OSINT agent for intelligence gathering"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            agent_type="osint",
            capabilities=[
                "social_media_intelligence",
                "threat_intelligence",
                "dark_web_monitoring",
                "breach_data_analysis",
                "reputation_analysis",
                "email_intelligence",
                "domain_intelligence",
                "person_intelligence",
                "company_intelligence",
                "ip_intelligence",
                "certificate_intelligence",
                "paste_monitoring"
            ]
        )
        
        self.communicator = AgentCommunicator(agent_id)
        self.session = None
        self.osint_sources = self._load_osint_sources()
        self.threat_feeds = self._load_threat_feeds()
        self.social_platforms = self._load_social_platforms()
        self.intelligence_cache = {}
        
    def _register_handlers(self):
        """Register task handlers"""
        self.register_task_handler("social_media_intelligence", self.social_media_intelligence)
        self.register_task_handler("threat_intelligence", self.threat_intelligence)
        self.register_task_handler("dark_web_monitoring", self.dark_web_monitoring)
        self.register_task_handler("breach_data_analysis", self.breach_data_analysis)
        self.register_task_handler("reputation_analysis", self.reputation_analysis)
        self.register_task_handler("email_intelligence", self.email_intelligence)
        self.register_task_handler("domain_intelligence", self.domain_intelligence)
        self.register_task_handler("person_intelligence", self.person_intelligence)
        self.register_task_handler("company_intelligence", self.company_intelligence)
        self.register_task_handler("ip_intelligence", self.ip_intelligence)
        self.register_task_handler("certificate_intelligence", self.certificate_intelligence)
        self.register_task_handler("paste_monitoring", self.paste_monitoring)
        self.register_task_handler("comprehensive_osint", self.comprehensive_osint)

    async def initialize(self):
        """Initialize OSINT agent"""
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
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': self.generate_random_user_agent()
        })
        
        logger.info(f"OSINT agent {self.agent_id} initialized")

    def _load_osint_sources(self) -> Dict[str, Any]:
        """Load OSINT sources configuration"""
        return {
            "search_engines": {
                "google": {
                    "url": "https://www.google.com/search",
                    "params": {"q": "{query}"},
                    "rate_limit": 10  # requests per minute
                },
                "bing": {
                    "url": "https://www.bing.com/search",
                    "params": {"q": "{query}"},
                    "rate_limit": 15
                },
                "duckduckgo": {
                    "url": "https://duckduckgo.com/",
                    "params": {"q": "{query}"},
                    "rate_limit": 20
                }
            },
            "whois_services": {
                "whois_json": {
                    "url": "https://whois.whoisjson.com/",
                    "rate_limit": 100
                },
                "ip_api": {
                    "url": "http://ip-api.com/json/",
                    "rate_limit": 45
                }
            },
            "certificate_transparency": {
                "crt_sh": {
                    "url": "https://crt.sh/",
                    "rate_limit": 100
                },
                "certspotter": {
                    "url": "https://api.certspotter.com/v1/",
                    "rate_limit": 100
                }
            },
            "threat_intelligence": {
                "virustotal": {
                    "url": "https://www.virustotal.com/vtapi/v2/",
                    "rate_limit": 4
                },
                "alienvault": {
                    "url": "https://otx.alienvault.com/api/v1/",
                    "rate_limit": 10
                },
                "shodan": {
                    "url": "https://api.shodan.io/",
                    "rate_limit": 100
                }
            },
            "paste_sites": {
                "pastebin": {
                    "url": "https://pastebin.com/",
                    "rate_limit": 60
                },
                "github": {
                    "url": "https://api.github.com/search/code",
                    "rate_limit": 30
                }
            }
        }

    def _load_threat_feeds(self) -> Dict[str, Any]:
        """Load threat intelligence feeds"""
        return {
            "malware_domains": {
                "malware_domain_list": "http://www.malwaredomainlist.com/hostslist/hosts.txt",
                "zeus_tracker": "https://zeustracker.abuse.ch/blocklist.php?download=domainblocklist",
                "malwaredomains": "http://mirror1.malwaredomains.com/files/justdomains"
            },
            "ip_reputation": {
                "spamhaus": "https://www.spamhaus.org/drop/drop.txt",
                "feodo_tracker": "https://feodotracker.abuse.ch/blocklist/?download=ipblocklist",
                "emergingthreats": "https://rules.emergingthreats.net/fwrules/emerging-Block-IPs.txt"
            },
            "vulnerability_feeds": {
                "cve_mitre": "https://cve.mitre.org/data/downloads/allitems.csv",
                "nvd": "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-recent.json.gz",
                "exploit_db": "https://www.exploit-db.com/rss.xml"
            }
        }

    def _load_social_platforms(self) -> Dict[str, Any]:
        """Load social media platforms configuration"""
        return {
            "twitter": {
                "search_url": "https://twitter.com/search",
                "user_url": "https://twitter.com/{username}",
                "rate_limit": 300
            },
            "facebook": {
                "search_url": "https://www.facebook.com/search/top/",
                "user_url": "https://www.facebook.com/{username}",
                "rate_limit": 200
            },
            "linkedin": {
                "search_url": "https://www.linkedin.com/search/results/people/",
                "user_url": "https://www.linkedin.com/in/{username}",
                "rate_limit": 100
            },
            "instagram": {
                "search_url": "https://www.instagram.com/explore/tags/",
                "user_url": "https://www.instagram.com/{username}",
                "rate_limit": 200
            },
            "github": {
                "search_url": "https://github.com/search",
                "user_url": "https://github.com/{username}",
                "rate_limit": 60
            },
            "reddit": {
                "search_url": "https://www.reddit.com/search/",
                "user_url": "https://www.reddit.com/user/{username}",
                "rate_limit": 60
            }
        }

    async def execute_primary_function(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive OSINT gathering"""
        return await self.comprehensive_osint(target, parameters)

    async def social_media_intelligence(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather social media intelligence"""
        try:
            search_type = parameters.get("search_type", "person")
            platforms = parameters.get("platforms", ["twitter", "facebook", "linkedin", "instagram"])
            
            results = {
                "target": target,
                "search_type": search_type,
                "timestamp": datetime.utcnow().isoformat(),
                "social_intelligence": {}
            }
            
            for platform in platforms:
                if platform in self.social_platforms:
                    try:
                        platform_data = await self._search_social_platform(platform, target, search_type)
                        results["social_intelligence"][platform] = platform_data
                        
                        # Add delay to respect rate limits
                        await self.sleep_with_jitter(2)
                        
                    except Exception as e:
                        logger.warning(f"Social media search failed for {platform}: {e}")
                        results["social_intelligence"][platform] = {"error": str(e)}
            
            self.log_activity("social_media_intelligence", results)
            return results
            
        except Exception as e:
            logger.error(f"Social media intelligence failed: {e}")
            return {"error": str(e)}

    async def _search_social_platform(self, platform: str, target: str, search_type: str) -> Dict[str, Any]:
        """Search specific social media platform"""
        try:
            platform_config = self.social_platforms[platform]
            
            results = {
                "platform": platform,
                "profiles": [],
                "posts": [],
                "metadata": {}
            }
            
            # Search for profiles
            if search_type == "person":
                profiles = await self._search_person_profiles(platform, target)
                results["profiles"] = profiles
            
            elif search_type == "company":
                profiles = await self._search_company_profiles(platform, target)
                results["profiles"] = profiles
            
            elif search_type == "email":
                profiles = await self._search_email_profiles(platform, target)
                results["profiles"] = profiles
            
            # Search for posts/content
            posts = await self._search_platform_content(platform, target)
            results["posts"] = posts
            
            return results
            
        except Exception as e:
            logger.error(f"Social platform search failed: {e}")
            return {"error": str(e)}

    async def _search_person_profiles(self, platform: str, target: str) -> List[Dict[str, Any]]:
        """Search for person profiles on platform"""
        try:
            profiles = []
            
            # This is a simplified implementation
            # In practice, you would use platform-specific APIs or web scraping
            
            if platform == "twitter":
                # Search Twitter for person
                profiles = await self._search_twitter_profiles(target)
            
            elif platform == "linkedin":
                # Search LinkedIn for person
                profiles = await self._search_linkedin_profiles(target)
            
            elif platform == "facebook":
                # Search Facebook for person
                profiles = await self._search_facebook_profiles(target)
            
            return profiles
            
        except Exception as e:
            logger.error(f"Person profile search failed: {e}")
            return []

    async def _search_company_profiles(self, platform: str, target: str) -> List[Dict[str, Any]]:
        """Search for company profiles on platform"""
        try:
            profiles = []
            
            # This is a simplified implementation
            # In practice, you would use platform-specific APIs or web scraping
            
            return profiles
            
        except Exception as e:
            logger.error(f"Company profile search failed: {e}")
            return []

    async def _search_email_profiles(self, platform: str, target: str) -> List[Dict[str, Any]]:
        """Search for email-associated profiles on platform"""
        try:
            profiles = []
            
            # This is a simplified implementation
            # In practice, you would use platform-specific APIs or web scraping
            
            return profiles
            
        except Exception as e:
            logger.error(f"Email profile search failed: {e}")
            return []

    async def _search_platform_content(self, platform: str, target: str) -> List[Dict[str, Any]]:
        """Search for content on platform"""
        try:
            posts = []
            
            # This is a simplified implementation
            # In practice, you would use platform-specific APIs or web scraping
            
            return posts
            
        except Exception as e:
            logger.error(f"Platform content search failed: {e}")
            return []

    async def _search_twitter_profiles(self, target: str) -> List[Dict[str, Any]]:
        """Search Twitter for profiles"""
        try:
            # This is a placeholder implementation
            # In practice, you would use Twitter API or web scraping
            return []
            
        except Exception as e:
            logger.error(f"Twitter profile search failed: {e}")
            return []

    async def _search_linkedin_profiles(self, target: str) -> List[Dict[str, Any]]:
        """Search LinkedIn for profiles"""
        try:
            # This is a placeholder implementation
            # In practice, you would use LinkedIn API or web scraping
            return []
            
        except Exception as e:
            logger.error(f"LinkedIn profile search failed: {e}")
            return []

    async def _search_facebook_profiles(self, target: str) -> List[Dict[str, Any]]:
        """Search Facebook for profiles"""
        try:
            # This is a placeholder implementation
            # In practice, you would use Facebook API or web scraping
            return []
            
        except Exception as e:
            logger.error(f"Facebook profile search failed: {e}")
            return []

    async def threat_intelligence(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather threat intelligence"""
        try:
            intel_type = parameters.get("intel_type", "domain")
            
            results = {
                "target": target,
                "intel_type": intel_type,
                "timestamp": datetime.utcnow().isoformat(),
                "threat_intelligence": {}
            }
            
            # Gather intelligence based on type
            if intel_type == "domain":
                results["threat_intelligence"] = await self._gather_domain_threat_intel(target)
            elif intel_type == "ip":
                results["threat_intelligence"] = await self._gather_ip_threat_intel(target)
            elif intel_type == "hash":
                results["threat_intelligence"] = await self._gather_hash_threat_intel(target)
            elif intel_type == "email":
                results["threat_intelligence"] = await self._gather_email_threat_intel(target)
            
            self.log_activity("threat_intelligence", results)
            return results
            
        except Exception as e:
            logger.error(f"Threat intelligence failed: {e}")
            return {"error": str(e)}

    async def _gather_domain_threat_intel(self, domain: str) -> Dict[str, Any]:
        """Gather threat intelligence for domain"""
        try:
            intel = {
                "domain": domain,
                "reputation": {},
                "malware_analysis": {},
                "certificate_info": {},
                "dns_analysis": {}
            }
            
            # Check domain reputation
            intel["reputation"] = await self._check_domain_reputation(domain)
            
            # Analyze certificates
            intel["certificate_info"] = await self._analyze_domain_certificates(domain)
            
            # DNS analysis
            intel["dns_analysis"] = await self._analyze_domain_dns(domain)
            
            return intel
            
        except Exception as e:
            logger.error(f"Domain threat intelligence failed: {e}")
            return {"error": str(e)}

    async def _gather_ip_threat_intel(self, ip: str) -> Dict[str, Any]:
        """Gather threat intelligence for IP address"""
        try:
            intel = {
                "ip": ip,
                "reputation": {},
                "geolocation": {},
                "asn_info": {},
                "port_analysis": {}
            }
            
            # Check IP reputation
            intel["reputation"] = await self._check_ip_reputation(ip)
            
            # Get geolocation
            intel["geolocation"] = await self._get_ip_geolocation(ip)
            
            # Get ASN information
            intel["asn_info"] = await self._get_ip_asn_info(ip)
            
            return intel
            
        except Exception as e:
            logger.error(f"IP threat intelligence failed: {e}")
            return {"error": str(e)}

    async def _gather_hash_threat_intel(self, hash_value: str) -> Dict[str, Any]:
        """Gather threat intelligence for hash"""
        try:
            intel = {
                "hash": hash_value,
                "malware_analysis": {},
                "antivirus_results": {},
                "behavior_analysis": {}
            }
            
            # Check hash reputation
            intel["malware_analysis"] = await self._check_hash_reputation(hash_value)
            
            return intel
            
        except Exception as e:
            logger.error(f"Hash threat intelligence failed: {e}")
            return {"error": str(e)}

    async def _gather_email_threat_intel(self, email: str) -> Dict[str, Any]:
        """Gather threat intelligence for email"""
        try:
            intel = {
                "email": email,
                "breach_data": {},
                "reputation": {},
                "domain_analysis": {}
            }
            
            # Check for breaches
            intel["breach_data"] = await self._check_email_breaches(email)
            
            # Analyze email domain
            domain = email.split('@')[1] if '@' in email else email
            intel["domain_analysis"] = await self._gather_domain_threat_intel(domain)
            
            return intel
            
        except Exception as e:
            logger.error(f"Email threat intelligence failed: {e}")
            return {"error": str(e)}

    async def _check_domain_reputation(self, domain: str) -> Dict[str, Any]:
        """Check domain reputation"""
        try:
            reputation = {
                "malware_detected": False,
                "phishing_detected": False,
                "spam_detected": False,
                "reputation_score": 0,
                "sources": []
            }
            
            # Check against threat feeds
            for feed_name, feed_urls in self.threat_feeds["malware_domains"].items():
                try:
                    response = await self._fetch_url(feed_urls)
                    if response and domain in response.lower():
                        reputation["malware_detected"] = True
                        reputation["sources"].append(feed_name)
                        
                except Exception as e:
                    logger.warning(f"Failed to check {feed_name}: {e}")
            
            return reputation
            
        except Exception as e:
            logger.error(f"Domain reputation check failed: {e}")
            return {"error": str(e)}

    async def _check_ip_reputation(self, ip: str) -> Dict[str, Any]:
        """Check IP reputation"""
        try:
            reputation = {
                "malicious": False,
                "spam": False,
                "botnet": False,
                "reputation_score": 0,
                "sources": []
            }
            
            # Check against IP reputation feeds
            for feed_name, feed_url in self.threat_feeds["ip_reputation"].items():
                try:
                    response = await self._fetch_url(feed_url)
                    if response and ip in response:
                        reputation["malicious"] = True
                        reputation["sources"].append(feed_name)
                        
                except Exception as e:
                    logger.warning(f"Failed to check {feed_name}: {e}")
            
            return reputation
            
        except Exception as e:
            logger.error(f"IP reputation check failed: {e}")
            return {"error": str(e)}

    async def _check_hash_reputation(self, hash_value: str) -> Dict[str, Any]:
        """Check hash reputation"""
        try:
            reputation = {
                "malicious": False,
                "detection_rate": 0,
                "first_seen": None,
                "last_seen": None,
                "sources": []
            }
            
            # This is a placeholder implementation
            # In practice, you would use VirusTotal API or similar services
            
            return reputation
            
        except Exception as e:
            logger.error(f"Hash reputation check failed: {e}")
            return {"error": str(e)}

    async def _check_email_breaches(self, email: str) -> Dict[str, Any]:
        """Check email for data breaches"""
        try:
            breaches = {
                "found_in_breaches": False,
                "breach_count": 0,
                "breaches": [],
                "pastes": []
            }
            
            # This is a placeholder implementation
            # In practice, you would use Have I Been Pwned API or similar services
            
            return breaches
            
        except Exception as e:
            logger.error(f"Email breach check failed: {e}")
            return {"error": str(e)}

    async def _get_ip_geolocation(self, ip: str) -> Dict[str, Any]:
        """Get IP geolocation"""
        try:
            url = f"http://ip-api.com/json/{ip}"
            response = await self._fetch_url(url)
            
            if response:
                data = json.loads(response)
                return {
                    "country": data.get("country", ""),
                    "country_code": data.get("countryCode", ""),
                    "region": data.get("regionName", ""),
                    "city": data.get("city", ""),
                    "latitude": data.get("lat", 0),
                    "longitude": data.get("lon", 0),
                    "isp": data.get("isp", ""),
                    "org": data.get("org", ""),
                    "as": data.get("as", "")
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"IP geolocation failed: {e}")
            return {"error": str(e)}

    async def _get_ip_asn_info(self, ip: str) -> Dict[str, Any]:
        """Get IP ASN information"""
        try:
            # This is a placeholder implementation
            # In practice, you would use ASN lookup services
            return {
                "asn": "",
                "org": "",
                "country": "",
                "allocated": ""
            }
            
        except Exception as e:
            logger.error(f"IP ASN lookup failed: {e}")
            return {"error": str(e)}

    async def _analyze_domain_certificates(self, domain: str) -> Dict[str, Any]:
        """Analyze domain certificates"""
        try:
            certificates = {
                "current_cert": {},
                "historical_certs": [],
                "subdomains": []
            }
            
            # Check certificate transparency logs
            ct_url = f"https://crt.sh/?q=%.{domain}&output=json"
            response = await self._fetch_url(ct_url)
            
            if response:
                data = json.loads(response)
                for cert in data[:10]:  # Limit to first 10 certificates
                    certificates["historical_certs"].append({
                        "id": cert.get("id", ""),
                        "logged_at": cert.get("entry_timestamp", ""),
                        "not_before": cert.get("not_before", ""),
                        "not_after": cert.get("not_after", ""),
                        "issuer": cert.get("issuer_name", ""),
                        "common_name": cert.get("common_name", ""),
                        "name_value": cert.get("name_value", "")
                    })
            
            return certificates
            
        except Exception as e:
            logger.error(f"Certificate analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_domain_dns(self, domain: str) -> Dict[str, Any]:
        """Analyze domain DNS"""
        try:
            dns_info = {
                "a_records": [],
                "aaaa_records": [],
                "mx_records": [],
                "ns_records": [],
                "txt_records": [],
                "cname_records": []
            }
            
            # Perform DNS lookups
            resolver = dns.resolver.Resolver()
            resolver.nameservers = ['8.8.8.8', '8.8.4.4']
            
            record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME']
            
            for record_type in record_types:
                try:
                    answers = resolver.resolve(domain, record_type)
                    records = [str(rdata) for rdata in answers]
                    dns_info[f"{record_type.lower()}_records"] = records
                    
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                    continue
                except Exception as e:
                    logger.warning(f"DNS lookup failed for {record_type}: {e}")
            
            return dns_info
            
        except Exception as e:
            logger.error(f"DNS analysis failed: {e}")
            return {"error": str(e)}

    async def _fetch_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch URL content with error handling"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.text()
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch URL {url}: {e}")
            return None

    async def dark_web_monitoring(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor dark web for intelligence"""
        try:
            search_type = parameters.get("search_type", "domain")
            
            results = {
                "target": target,
                "search_type": search_type,
                "timestamp": datetime.utcnow().isoformat(),
                "dark_web_intelligence": {}
            }
            
            # This is a placeholder implementation
            # In practice, you would use specialized dark web monitoring services
            results["dark_web_intelligence"] = {
                "mentions": [],
                "marketplaces": [],
                "forums": [],
                "paste_sites": []
            }
            
            self.log_activity("dark_web_monitoring", results)
            return results
            
        except Exception as e:
            logger.error(f"Dark web monitoring failed: {e}")
            return {"error": str(e)}

    async def breach_data_analysis(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze breach data"""
        try:
            data_type = parameters.get("data_type", "email")
            
            results = {
                "target": target,
                "data_type": data_type,
                "timestamp": datetime.utcnow().isoformat(),
                "breach_analysis": {}
            }
            
            if data_type == "email":
                results["breach_analysis"] = await self._analyze_email_breaches(target)
            elif data_type == "domain":
                results["breach_analysis"] = await self._analyze_domain_breaches(target)
            elif data_type == "username":
                results["breach_analysis"] = await self._analyze_username_breaches(target)
            
            self.log_activity("breach_data_analysis", results)
            return results
            
        except Exception as e:
            logger.error(f"Breach data analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_email_breaches(self, email: str) -> Dict[str, Any]:
        """Analyze email breaches"""
        try:
            # This is a placeholder implementation
            # In practice, you would use Have I Been Pwned API or similar services
            return {
                "breaches_found": 0,
                "breaches": [],
                "pastes": [],
                "last_breach": None
            }
            
        except Exception as e:
            logger.error(f"Email breach analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_domain_breaches(self, domain: str) -> Dict[str, Any]:
        """Analyze domain breaches"""
        try:
            # This is a placeholder implementation
            return {
                "breaches_found": 0,
                "breaches": [],
                "affected_users": 0
            }
            
        except Exception as e:
            logger.error(f"Domain breach analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_username_breaches(self, username: str) -> Dict[str, Any]:
        """Analyze username breaches"""
        try:
            # This is a placeholder implementation
            return {
                "breaches_found": 0,
                "breaches": [],
                "platforms": []
            }
            
        except Exception as e:
            logger.error(f"Username breach analysis failed: {e}")
            return {"error": str(e)}

    async def reputation_analysis(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reputation"""
        try:
            target_type = parameters.get("target_type", "domain")
            
            results = {
                "target": target,
                "target_type": target_type,
                "timestamp": datetime.utcnow().isoformat(),
                "reputation_analysis": {}
            }
            
            if target_type == "domain":
                results["reputation_analysis"] = await self._analyze_domain_reputation(target)
            elif target_type == "ip":
                results["reputation_analysis"] = await self._analyze_ip_reputation(target)
            elif target_type == "email":
                results["reputation_analysis"] = await self._analyze_email_reputation(target)
            
            self.log_activity("reputation_analysis", results)
            return results
            
        except Exception as e:
            logger.error(f"Reputation analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_domain_reputation(self, domain: str) -> Dict[str, Any]:
        """Analyze domain reputation"""
        try:
            reputation = await self._check_domain_reputation(domain)
            
            # Add additional reputation metrics
            reputation.update({
                "age": await self._get_domain_age(domain),
                "alexa_rank": await self._get_alexa_rank(domain),
                "social_mentions": await self._get_social_mentions(domain)
            })
            
            return reputation
            
        except Exception as e:
            logger.error(f"Domain reputation analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_ip_reputation(self, ip: str) -> Dict[str, Any]:
        """Analyze IP reputation"""
        try:
            reputation = await self._check_ip_reputation(ip)
            
            # Add additional reputation metrics
            reputation.update({
                "geolocation": await self._get_ip_geolocation(ip),
                "asn_info": await self._get_ip_asn_info(ip),
                "port_scan": await self._get_ip_port_scan(ip)
            })
            
            return reputation
            
        except Exception as e:
            logger.error(f"IP reputation analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_email_reputation(self, email: str) -> Dict[str, Any]:
        """Analyze email reputation"""
        try:
            reputation = {
                "deliverability": "unknown",
                "spam_score": 0,
                "breach_history": await self._check_email_breaches(email),
                "domain_reputation": {}
            }
            
            # Analyze email domain
            domain = email.split('@')[1] if '@' in email else email
            reputation["domain_reputation"] = await self._analyze_domain_reputation(domain)
            
            return reputation
            
        except Exception as e:
            logger.error(f"Email reputation analysis failed: {e}")
            return {"error": str(e)}

    async def _get_domain_age(self, domain: str) -> Optional[str]:
        """Get domain age"""
        try:
            # This is a placeholder implementation
            # In practice, you would use WHOIS data
            return None
            
        except Exception as e:
            logger.error(f"Domain age lookup failed: {e}")
            return None

    async def _get_alexa_rank(self, domain: str) -> Optional[int]:
        """Get Alexa rank"""
        try:
            # This is a placeholder implementation
            # In practice, you would use Alexa API
            return None
            
        except Exception as e:
            logger.error(f"Alexa rank lookup failed: {e}")
            return None

    async def _get_social_mentions(self, domain: str) -> int:
        """Get social media mentions count"""
        try:
            # This is a placeholder implementation
            # In practice, you would search social platforms
            return 0
            
        except Exception as e:
            logger.error(f"Social mentions lookup failed: {e}")
            return 0

    async def _get_ip_port_scan(self, ip: str) -> Dict[str, Any]:
        """Get IP port scan results"""
        try:
            # This is a placeholder implementation
            # In practice, you would use Shodan API
            return {
                "open_ports": [],
                "services": [],
                "vulnerabilities": []
            }
            
        except Exception as e:
            logger.error(f"IP port scan failed: {e}")
            return {"error": str(e)}

    async def email_intelligence(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather email intelligence"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "email_intelligence": {}
            }
            
            # Gather email intelligence
            results["email_intelligence"] = {
                "breach_data": await self._check_email_breaches(target),
                "reputation": await self._analyze_email_reputation(target),
                "related_accounts": await self._find_related_accounts(target),
                "domain_analysis": await self._analyze_email_domain(target)
            }
            
            self.log_activity("email_intelligence", results)
            return results
            
        except Exception as e:
            logger.error(f"Email intelligence failed: {e}")
            return {"error": str(e)}

    async def _find_related_accounts(self, email: str) -> List[Dict[str, Any]]:
        """Find related accounts for email"""
        try:
            accounts = []
            
            # Search social platforms for email
            for platform in self.social_platforms:
                try:
                    platform_accounts = await self._search_email_profiles(platform, email)
                    accounts.extend(platform_accounts)
                    
                except Exception as e:
                    logger.warning(f"Related accounts search failed for {platform}: {e}")
            
            return accounts
            
        except Exception as e:
            logger.error(f"Related accounts search failed: {e}")
            return []

    async def _analyze_email_domain(self, email: str) -> Dict[str, Any]:
        """Analyze email domain"""
        try:
            domain = email.split('@')[1] if '@' in email else email
            return await self._gather_domain_threat_intel(domain)
            
        except Exception as e:
            logger.error(f"Email domain analysis failed: {e}")
            return {"error": str(e)}

    async def domain_intelligence(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather domain intelligence"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "domain_intelligence": {}
            }
            
            # Gather domain intelligence
            results["domain_intelligence"] = {
                "threat_intel": await self._gather_domain_threat_intel(target),
                "certificate_intel": await self._analyze_domain_certificates(target),
                "subdomain_intel": await self._discover_subdomains(target),
                "dns_intel": await self._analyze_domain_dns(target)
            }
            
            self.log_activity("domain_intelligence", results)
            return results
            
        except Exception as e:
            logger.error(f"Domain intelligence failed: {e}")
            return {"error": str(e)}

    async def _discover_subdomains(self, domain: str) -> List[str]:
        """Discover subdomains"""
        try:
            subdomains = set()
            
            # Certificate transparency logs
            ct_url = f"https://crt.sh/?q=%.{domain}&output=json"
            response = await self._fetch_url(ct_url)
            
            if response:
                data = json.loads(response)
                for cert in data:
                    name_value = cert.get("name_value", "")
                    for name in name_value.split("\n"):
                        if name.endswith(domain):
                            subdomains.add(name.strip())
            
            return list(subdomains)
            
        except Exception as e:
            logger.error(f"Subdomain discovery failed: {e}")
            return []

    async def person_intelligence(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather person intelligence"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "person_intelligence": {}
            }
            
            # Gather person intelligence
            results["person_intelligence"] = {
                "social_media": await self._search_person_social_media(target),
                "professional_info": await self._search_person_professional(target),
                "public_records": await self._search_person_public_records(target),
                "data_breaches": await self._search_person_breaches(target)
            }
            
            self.log_activity("person_intelligence", results)
            return results
            
        except Exception as e:
            logger.error(f"Person intelligence failed: {e}")
            return {"error": str(e)}

    async def _search_person_social_media(self, person: str) -> Dict[str, Any]:
        """Search person on social media"""
        try:
            social_data = {}
            
            for platform in self.social_platforms:
                try:
                    profiles = await self._search_person_profiles(platform, person)
                    social_data[platform] = profiles
                    
                except Exception as e:
                    logger.warning(f"Social media search failed for {platform}: {e}")
            
            return social_data
            
        except Exception as e:
            logger.error(f"Person social media search failed: {e}")
            return {"error": str(e)}

    async def _search_person_professional(self, person: str) -> Dict[str, Any]:
        """Search person professional information"""
        try:
            # This is a placeholder implementation
            # In practice, you would search professional networks
            return {
                "linkedin": [],
                "company_websites": [],
                "press_releases": [],
                "patents": []
            }
            
        except Exception as e:
            logger.error(f"Person professional search failed: {e}")
            return {"error": str(e)}

    async def _search_person_public_records(self, person: str) -> Dict[str, Any]:
        """Search person public records"""
        try:
            # This is a placeholder implementation
            # In practice, you would search public record databases
            return {
                "court_records": [],
                "property_records": [],
                "business_records": [],
                "voter_records": []
            }
            
        except Exception as e:
            logger.error(f"Person public records search failed: {e}")
            return {"error": str(e)}

    async def _search_person_breaches(self, person: str) -> Dict[str, Any]:
        """Search person in data breaches"""
        try:
            # This is a placeholder implementation
            # In practice, you would search breach databases
            return {
                "email_breaches": [],
                "username_breaches": [],
                "password_breaches": []
            }
            
        except Exception as e:
            logger.error(f"Person breach search failed: {e}")
            return {"error": str(e)}

    async def company_intelligence(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather company intelligence"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "company_intelligence": {}
            }
            
            # Gather company intelligence
            results["company_intelligence"] = {
                "corporate_info": await self._search_company_corporate_info(target),
                "employees": await self._search_company_employees(target),
                "technology_stack": await self._search_company_technology(target),
                "social_media": await self._search_company_social_media(target),
                "news_mentions": await self._search_company_news(target)
            }
            
            self.log_activity("company_intelligence", results)
            return results
            
        except Exception as e:
            logger.error(f"Company intelligence failed: {e}")
            return {"error": str(e)}

    async def _search_company_corporate_info(self, company: str) -> Dict[str, Any]:
        """Search company corporate information"""
        try:
            # This is a placeholder implementation
            return {
                "registration_info": {},
                "financial_info": {},
                "legal_info": {},
                "subsidiaries": []
            }
            
        except Exception as e:
            logger.error(f"Company corporate info search failed: {e}")
            return {"error": str(e)}

    async def _search_company_employees(self, company: str) -> List[Dict[str, Any]]:
        """Search company employees"""
        try:
            # This is a placeholder implementation
            # In practice, you would search LinkedIn and other professional networks
            return []
            
        except Exception as e:
            logger.error(f"Company employees search failed: {e}")
            return []

    async def _search_company_technology(self, company: str) -> Dict[str, Any]:
        """Search company technology stack"""
        try:
            # This is a placeholder implementation
            # In practice, you would use services like Wappalyzer, BuiltWith, etc.
            return {
                "web_technologies": [],
                "cloud_services": [],
                "software_stack": [],
                "security_tools": []
            }
            
        except Exception as e:
            logger.error(f"Company technology search failed: {e}")
            return {"error": str(e)}

    async def _search_company_social_media(self, company: str) -> Dict[str, Any]:
        """Search company social media"""
        try:
            social_data = {}
            
            for platform in self.social_platforms:
                try:
                    profiles = await self._search_company_profiles(platform, company)
                    social_data[platform] = profiles
                    
                except Exception as e:
                    logger.warning(f"Company social media search failed for {platform}: {e}")
            
            return social_data
            
        except Exception as e:
            logger.error(f"Company social media search failed: {e}")
            return {"error": str(e)}

    async def _search_company_news(self, company: str) -> List[Dict[str, Any]]:
        """Search company news mentions"""
        try:
            # This is a placeholder implementation
            # In practice, you would search news aggregators
            return []
            
        except Exception as e:
            logger.error(f"Company news search failed: {e}")
            return []

    async def ip_intelligence(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather IP intelligence"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "ip_intelligence": {}
            }
            
            # Gather IP intelligence
            results["ip_intelligence"] = {
                "threat_intel": await self._gather_ip_threat_intel(target),
                "geolocation": await self._get_ip_geolocation(target),
                "asn_info": await self._get_ip_asn_info(target),
                "port_scan": await self._get_ip_port_scan(target),
                "reverse_dns": await self._get_ip_reverse_dns(target)
            }
            
            self.log_activity("ip_intelligence", results)
            return results
            
        except Exception as e:
            logger.error(f"IP intelligence failed: {e}")
            return {"error": str(e)}

    async def _get_ip_reverse_dns(self, ip: str) -> List[str]:
        """Get IP reverse DNS"""
        try:
            import socket
            
            try:
                hostname = socket.gethostbyaddr(ip)[0]
                return [hostname]
            except socket.herror:
                return []
            
        except Exception as e:
            logger.error(f"IP reverse DNS failed: {e}")
            return []

    async def certificate_intelligence(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather certificate intelligence"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "certificate_intelligence": {}
            }
            
            # Gather certificate intelligence
            results["certificate_intelligence"] = await self._analyze_domain_certificates(target)
            
            self.log_activity("certificate_intelligence", results)
            return results
            
        except Exception as e:
            logger.error(f"Certificate intelligence failed: {e}")
            return {"error": str(e)}

    async def paste_monitoring(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor paste sites"""
        try:
            search_type = parameters.get("search_type", "domain")
            
            results = {
                "target": target,
                "search_type": search_type,
                "timestamp": datetime.utcnow().isoformat(),
                "paste_monitoring": {}
            }
            
            # Search paste sites
            results["paste_monitoring"] = await self._search_paste_sites(target, search_type)
            
            self.log_activity("paste_monitoring", results)
            return results
            
        except Exception as e:
            logger.error(f"Paste monitoring failed: {e}")
            return {"error": str(e)}

    async def _search_paste_sites(self, target: str, search_type: str) -> Dict[str, Any]:
        """Search paste sites"""
        try:
            pastes = {
                "pastebin": [],
                "github": [],
                "other_sites": []
            }
            
            # Search GitHub for code
            if search_type in ["domain", "email", "password"]:
                github_results = await self._search_github_code(target)
                pastes["github"] = github_results
            
            # Search other paste sites
            # This is a placeholder implementation
            
            return pastes
            
        except Exception as e:
            logger.error(f"Paste sites search failed: {e}")
            return {"error": str(e)}

    async def _search_github_code(self, target: str) -> List[Dict[str, Any]]:
        """Search GitHub code"""
        try:
            # This is a placeholder implementation
            # In practice, you would use GitHub API
            return []
            
        except Exception as e:
            logger.error(f"GitHub code search failed: {e}")
            return []

    async def comprehensive_osint(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive OSINT gathering"""
        try:
            results = {
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "comprehensive_osint": {}
            }
            
            # Perform all OSINT tasks
            osint_tasks = [
                ("social_media_intelligence", self.social_media_intelligence),
                ("threat_intelligence", self.threat_intelligence),
                ("breach_data_analysis", self.breach_data_analysis),
                ("reputation_analysis", self.reputation_analysis),
                ("domain_intelligence", self.domain_intelligence),
                ("certificate_intelligence", self.certificate_intelligence),
                ("paste_monitoring", self.paste_monitoring)
            ]
            
            for task_name, task_func in osint_tasks:
                try:
                    task_result = await task_func(target, parameters)
                    results["comprehensive_osint"][task_name] = task_result
                    
                    # Add delay between tasks
                    await self.sleep_with_jitter(random.uniform(3, 7))
                    
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    results["comprehensive_osint"][task_name] = {"error": str(e)}
            
            # Send intelligence data to coordinator
            await self.send_intelligence_data(results)
            
            self.log_activity("comprehensive_osint", results)
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive OSINT failed: {e}")
            return {"error": str(e)}

    async def shutdown(self):
        """Shutdown OSINT agent"""
        try:
            if self.session:
                self.session.close()
            await self.communicator.shutdown()
            await super().shutdown()
        except Exception as e:
            logger.error(f"Error shutting down OSINT agent: {e}")

def main():
    """Main function for running OSINT agent"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python osint_agent.py <agent_id>")
        sys.exit(1)
    
    agent_id = sys.argv[1]
    agent = OSINTAgent(agent_id)
    
    async def run_agent():
        try:
            await agent.initialize()
            
            # Keep agent running
            while agent.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("OSINT agent shutting down...")
        except Exception as e:
            logger.error(f"Agent error: {e}")
        finally:
            await agent.shutdown()
    
    asyncio.run(run_agent())

if __name__ == "__main__":
    main()