#!/usr/bin/env python3
"""
Aetherveil Knowledge Learning Agent
Continuously learns from CVE feeds, HackerOne writeups, and security research
to enhance the Red Team AI's capabilities and detection accuracy.
"""

import os
import json
import asyncio
import aiohttp
import logging
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
import feedparser
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import yaml
from google.cloud import pubsub_v1, firestore, bigquery, aiplatform
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityKnowledge:
    """Structured security knowledge item"""
    knowledge_id: str
    timestamp: datetime
    source: str  # cve, hackerone, research, blog
    source_url: str
    title: str
    description: str
    vulnerability_type: str
    affected_technologies: List[str]
    attack_vectors: List[str]
    exploitation_techniques: List[str]
    detection_signatures: List[str]
    remediation_steps: List[str]
    severity_score: float
    confidence_score: float
    tags: List[str]
    references: List[str]
    learned_patterns: Dict[str, Any]
    ml_features: Dict[str, Any]

class CVEFeedProcessor:
    """Processes CVE feeds and extracts actionable intelligence"""
    
    def __init__(self):
        self.cve_sources = [
            'https://cve.mitre.org/data/downloads/allitems.xml',
            'https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-recent.json.gz',
            'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json'
        ]
        
        self.vulnerability_patterns = {
            'xss': [
                r'cross.?site.?scripting',
                r'reflected.*input',
                r'stored.*payload',
                r'dom.*manipulation'
            ],
            'sql_injection': [
                r'sql.*injection',
                r'database.*query',
                r'prepared.*statement',
                r'union.*select'
            ],
            'rce': [
                r'remote.*code.*execution',
                r'command.*injection',
                r'arbitrary.*code',
                r'shell.*execution'
            ],
            'ssrf': [
                r'server.?side.*request.*forgery',
                r'internal.*request',
                r'url.*fetch',
                r'metadata.*access'
            ]
        }
    
    async def fetch_recent_cves(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Fetch recent CVEs from multiple sources"""
        recent_cves = []
        
        try:
            # Fetch from NVD recent feed
            nvd_cves = await self.fetch_nvd_recent()
            recent_cves.extend(nvd_cves)
            
            # Fetch from CISA known exploited vulnerabilities
            cisa_cves = await self.fetch_cisa_kev()
            recent_cves.extend(cisa_cves)
            
            logger.info(f"Fetched {len(recent_cves)} recent CVEs")
            
        except Exception as e:
            logger.error(f"Error fetching CVEs: {e}")
        
        return recent_cves
    
    async def fetch_nvd_recent(self) -> List[Dict[str, Any]]:
        """Fetch recent CVEs from NVD"""
        cves = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Note: In production, would use proper NVD API with API key
                url = "https://services.nvd.nist.gov/rest/json/cves/1.0/"
                params = {
                    'modStartDate': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S:000 UTC-05:00'),
                    'modEndDate': datetime.now().strftime('%Y-%m-%dT%H:%M:%S:000 UTC-05:00')
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for cve_item in data.get('result', {}).get('CVE_Items', []):
                            cve_data = self.parse_nvd_cve(cve_item)
                            if cve_data:
                                cves.append(cve_data)
                                
        except Exception as e:
            logger.debug(f"Error fetching NVD CVEs: {e}")
        
        return cves
    
    def parse_nvd_cve(self, cve_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse NVD CVE item into structured format"""
        try:
            cve = cve_item.get('cve', {})
            cve_id = cve.get('CVE_data_meta', {}).get('ID', '')
            
            description = ''
            descriptions = cve.get('description', {}).get('description_data', [])
            if descriptions:
                description = descriptions[0].get('value', '')
            
            # Extract vulnerability type
            vuln_type = self.classify_vulnerability_type(description)
            
            # Extract affected technologies
            affected_tech = self.extract_affected_technologies(description, cve_item)
            
            # Get CVSS score
            cvss_score = 0.0
            impact = cve_item.get('impact', {})
            if 'baseMetricV3' in impact:
                cvss_score = impact['baseMetricV3'].get('cvssV3', {}).get('baseScore', 0.0)
            elif 'baseMetricV2' in impact:
                cvss_score = impact['baseMetricV2'].get('cvssV2', {}).get('baseScore', 0.0)
            
            return {
                'cve_id': cve_id,
                'description': description,
                'vulnerability_type': vuln_type,
                'affected_technologies': affected_tech,
                'cvss_score': cvss_score,
                'published_date': cve.get('publishedDate', ''),
                'modified_date': cve.get('lastModifiedDate', ''),
                'source_url': f"https://nvd.nist.gov/vuln/detail/{cve_id}"
            }
            
        except Exception as e:
            logger.debug(f"Error parsing CVE: {e}")
            return None
    
    def classify_vulnerability_type(self, description: str) -> str:
        """Classify vulnerability type based on description"""
        description_lower = description.lower()
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description_lower):
                    return vuln_type
        
        return 'unknown'
    
    def extract_affected_technologies(self, description: str, cve_item: Dict[str, Any]) -> List[str]:
        """Extract affected technologies from CVE data"""
        technologies = []
        
        # Extract from vendor/product data
        affects = cve_item.get('cve', {}).get('affects', {}).get('vendor', {}).get('vendor_data', [])
        for vendor in affects:
            vendor_name = vendor.get('vendor_name', '')
            if vendor_name and vendor_name != 'n/a':
                technologies.append(vendor_name)
            
            for product in vendor.get('product', {}).get('product_data', []):
                product_name = product.get('product_name', '')
                if product_name and product_name != 'n/a':
                    technologies.append(product_name)
        
        # Extract common technologies from description
        tech_keywords = [
            'apache', 'nginx', 'php', 'python', 'java', 'javascript', 'node.js',
            'wordpress', 'drupal', 'joomla', 'mysql', 'postgresql', 'mongodb',
            'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp'
        ]
        
        description_lower = description.lower()
        for keyword in tech_keywords:
            if keyword in description_lower:
                technologies.append(keyword)
        
        return list(set(technologies))  # Remove duplicates
    
    async def fetch_cisa_kev(self) -> List[Dict[str, Any]]:
        """Fetch CISA Known Exploited Vulnerabilities"""
        cves = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for vuln in data.get('vulnerabilities', []):
                            cve_data = {
                                'cve_id': vuln.get('cveID', ''),
                                'description': vuln.get('shortDescription', ''),
                                'vulnerability_type': self.classify_vulnerability_type(vuln.get('shortDescription', '')),
                                'affected_technologies': [vuln.get('vendorProject', ''), vuln.get('product', '')],
                                'cvss_score': 8.0,  # CISA KEV are high priority
                                'published_date': vuln.get('dateAdded', ''),
                                'source_url': f"https://nvd.nist.gov/vuln/detail/{vuln.get('cveID', '')}",
                                'actively_exploited': True
                            }
                            cves.append(cve_data)
                            
        except Exception as e:
            logger.debug(f"Error fetching CISA KEV: {e}")
        
        return cves

class HackerOneProcessor:
    """Processes HackerOne writeups and disclosed reports"""
    
    def __init__(self):
        self.hackerone_sources = [
            'https://hackerone.com/hacktivity.json',
            'https://hackerone.com/reports.json'
        ]
        
        self.technique_extractors = {
            'payload_patterns': [
                r'<script[^>]*>.*?</script>',
                r'\'.*?union.*?select.*?--',
                r';\s*(?:cat|ls|whoami|id)\s*',
                r'{{.*?}}',  # Template injection
                r'\${.*?}'   # Expression language injection
            ],
            'bypass_techniques': [
                r'waf.*?bypass',
                r'filter.*?bypass',
                r'encoding.*?bypass',
                r'double.*?encoding'
            ]
        }
    
    async def fetch_recent_reports(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Fetch recent HackerOne disclosed reports"""
        reports = []
        
        try:
            # Note: In production, would use HackerOne API with proper authentication
            async with aiohttp.ClientSession() as session:
                # Fetch recent disclosed reports
                url = "https://hackerone.com/hacktivity"
                params = {
                    'disclosed': 'true',
                    'page': 1
                }
                
                # This is a simplified example - would need proper API integration
                # For now, simulate with known public reports
                sample_reports = await self.get_sample_disclosed_reports()
                reports.extend(sample_reports)
                
        except Exception as e:
            logger.debug(f"Error fetching HackerOne reports: {e}")
        
        return reports
    
    async def get_sample_disclosed_reports(self) -> List[Dict[str, Any]]:
        """Get sample disclosed reports for learning"""
        return [
            {
                'report_id': 'sample_xss_1',
                'title': 'Reflected XSS in search parameter',
                'vulnerability_type': 'xss',
                'description': 'The search parameter reflects user input without proper sanitization',
                'payload': '<script>alert(document.domain)</script>',
                'bypass_technique': 'HTML entity encoding bypass',
                'severity': 'medium',
                'bounty_amount': 500,
                'program': 'example-program',
                'disclosure_date': datetime.now().isoformat(),
                'techniques_learned': ['html_entity_bypass', 'dom_manipulation']
            },
            {
                'report_id': 'sample_sqli_1', 
                'title': 'SQL Injection in login form',
                'vulnerability_type': 'sql_injection',
                'description': 'Login form vulnerable to SQL injection via username parameter',
                'payload': "admin' OR '1'='1' --",
                'bypass_technique': 'Comment-based bypass',
                'severity': 'high',
                'bounty_amount': 2000,
                'program': 'example-program',
                'disclosure_date': datetime.now().isoformat(),
                'techniques_learned': ['authentication_bypass', 'comment_injection']
            }
        ]
    
    def extract_techniques_from_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reusable techniques from HackerOne report"""
        techniques = {
            'payloads': [],
            'bypass_methods': [],
            'attack_vectors': [],
            'detection_evasion': []
        }
        
        description = report.get('description', '')
        payload = report.get('payload', '')
        
        # Extract payloads
        for pattern in self.technique_extractors['payload_patterns']:
            matches = re.findall(pattern, payload + ' ' + description, re.IGNORECASE)
            techniques['payloads'].extend(matches)
        
        # Extract bypass techniques
        for pattern in self.technique_extractors['bypass_techniques']:
            matches = re.findall(pattern, description, re.IGNORECASE)
            techniques['bypass_methods'].extend(matches)
        
        return techniques

class SecurityResearchProcessor:
    """Processes security research and blog posts"""
    
    def __init__(self):
        self.research_sources = [
            'https://feeds.feedburner.com/PortSwiggerResearch',
            'https://googleprojectzero.blogspot.com/feeds/posts/default',
            'https://blog.rapid7.com/rss.xml',
            'https://www.exploit-db.com/rss.xml'
        ]
    
    async def fetch_recent_research(self, days_back: int = 14) -> List[Dict[str, Any]]:
        """Fetch recent security research"""
        research_items = []
        
        for source_url in self.research_sources:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(source_url) as response:
                        if response.status == 200:
                            feed_content = await response.text()
                            feed = feedparser.parse(feed_content)
                            
                            for entry in feed.entries[:10]:  # Latest 10 items
                                research_item = self.parse_research_entry(entry, source_url)
                                if research_item:
                                    research_items.append(research_item)
                                    
            except Exception as e:
                logger.debug(f"Error fetching research from {source_url}: {e}")
        
        return research_items
    
    def parse_research_entry(self, entry: Any, source_url: str) -> Optional[Dict[str, Any]]:
        """Parse research feed entry"""
        try:
            return {
                'title': entry.get('title', ''),
                'description': entry.get('summary', ''),
                'url': entry.get('link', ''),
                'published_date': entry.get('published', ''),
                'source': source_url,
                'tags': [tag.get('term', '') for tag in entry.get('tags', [])],
                'content': entry.get('content', [{}])[0].get('value', '') if entry.get('content') else ''
            }
            
        except Exception as e:
            logger.debug(f"Error parsing research entry: {e}")
            return None

class MLModelTrainer:
    """Trains ML models on learned security knowledge"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        # Initialize Vertex AI
        aiplatform.init(project=project_id)
    
    async def train_vulnerability_classifier(self, knowledge_items: List[SecurityKnowledge]) -> str:
        """Train ML model to classify vulnerabilities"""
        try:
            # Prepare training data
            training_data = []
            for item in knowledge_items:
                training_data.append({
                    'text_features': f"{item.title} {item.description}",
                    'vulnerability_type': item.vulnerability_type,
                    'severity_score': item.severity_score,
                    'affected_technologies': ' '.join(item.affected_technologies),
                    'attack_vectors': ' '.join(item.attack_vectors)
                })
            
            # In production, would create and train actual Vertex AI model
            # For now, simulate model training
            model_id = f"vuln_classifier_{int(time.time())}"
            
            logger.info(f"Training vulnerability classifier with {len(training_data)} samples")
            
            # Simulate training time
            await asyncio.sleep(2)
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return ""
    
    async def train_payload_generator(self, payloads: List[str], vuln_types: List[str]) -> str:
        """Train model to generate payloads for different vulnerability types"""
        try:
            # Prepare payload training data
            training_pairs = list(zip(vuln_types, payloads))
            
            # In production, would train sequence-to-sequence model
            model_id = f"payload_generator_{int(time.time())}"
            
            logger.info(f"Training payload generator with {len(training_pairs)} examples")
            
            await asyncio.sleep(2)
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error training payload generator: {e}")
            return ""

class KnowledgeLearner:
    """Main knowledge learning agent"""
    
    def __init__(self):
        self.project_id = os.getenv('PROJECT_ID', 'tidy-computing-465909-i3')
        self.pubsub_client = pubsub_v1.PublisherClient()
        self.firestore_client = firestore.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        
        # Processors
        self.cve_processor = CVEFeedProcessor()
        self.hackerone_processor = HackerOneProcessor()
        self.research_processor = SecurityResearchProcessor()
        self.ml_trainer = MLModelTrainer(self.project_id)
        
        # Topics
        self.knowledge_topic = f"projects/{self.project_id}/topics/aetherveil-knowledge-updates"
        self.model_topic = f"projects/{self.project_id}/topics/aetherveil-model-updates"
    
    async def initialize(self):
        """Initialize the knowledge learner"""
        logger.info("Initializing Knowledge Learning Agent")
        
        # Setup knowledge storage
        await self.setup_knowledge_storage()
        
        # Start learning cycles
        await self.start_learning_cycles()
    
    async def setup_knowledge_storage(self):
        """Setup BigQuery tables for knowledge storage"""
        try:
            dataset_id = "security_knowledge"
            
            # Create dataset if not exists
            try:
                self.bigquery_client.get_dataset(dataset_id)
            except:
                dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
                dataset.location = "US"
                self.bigquery_client.create_dataset(dataset)
            
            # Knowledge items table
            knowledge_schema = [
                bigquery.SchemaField("knowledge_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("vulnerability_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("severity_score", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("affected_technologies", "STRING", mode="REPEATED"),
                bigquery.SchemaField("attack_vectors", "STRING", mode="REPEATED"),
                bigquery.SchemaField("detection_signatures", "STRING", mode="REPEATED"),
                bigquery.SchemaField("learned_patterns", "JSON", mode="NULLABLE"),
            ]
            
            self.create_table_if_not_exists(dataset_id, "security_knowledge", knowledge_schema)
            
            logger.info("Knowledge storage setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up knowledge storage: {e}")
    
    def create_table_if_not_exists(self, dataset_id: str, table_id: str, schema: List):
        """Create BigQuery table if it doesn't exist"""
        try:
            table_ref = self.bigquery_client.dataset(dataset_id).table(table_id)
            self.bigquery_client.get_table(table_ref)
        except:
            table = bigquery.Table(table_ref, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
            self.bigquery_client.create_table(table)
            logger.info(f"Created table: {dataset_id}.{table_id}")
    
    async def start_learning_cycles(self):
        """Start continuous learning cycles"""
        # Schedule different learning tasks
        asyncio.create_task(self.cve_learning_cycle())
        asyncio.create_task(self.hackerone_learning_cycle())
        asyncio.create_task(self.research_learning_cycle())
        asyncio.create_task(self.model_training_cycle())
    
    async def cve_learning_cycle(self):
        """Continuous CVE learning cycle"""
        while True:
            try:
                logger.info("Starting CVE learning cycle")
                
                # Fetch recent CVEs
                recent_cves = await self.cve_processor.fetch_recent_cves()
                
                # Process and store knowledge
                for cve_data in recent_cves:
                    knowledge_item = await self.process_cve_knowledge(cve_data)
                    if knowledge_item:
                        await self.store_knowledge(knowledge_item)
                
                logger.info(f"CVE learning cycle completed: {len(recent_cves)} items processed")
                
                # Wait 6 hours before next cycle
                await asyncio.sleep(6 * 3600)
                
            except Exception as e:
                logger.error(f"Error in CVE learning cycle: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def hackerone_learning_cycle(self):
        """Continuous HackerOne learning cycle"""
        while True:
            try:
                logger.info("Starting HackerOne learning cycle")
                
                # Fetch recent reports
                recent_reports = await self.hackerone_processor.fetch_recent_reports()
                
                # Process and store knowledge
                for report in recent_reports:
                    knowledge_item = await self.process_hackerone_knowledge(report)
                    if knowledge_item:
                        await self.store_knowledge(knowledge_item)
                
                logger.info(f"HackerOne learning cycle completed: {len(recent_reports)} items processed")
                
                # Wait 24 hours before next cycle
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                logger.error(f"Error in HackerOne learning cycle: {e}")
                await asyncio.sleep(3600)
    
    async def research_learning_cycle(self):
        """Continuous research learning cycle"""
        while True:
            try:
                logger.info("Starting research learning cycle")
                
                # Fetch recent research
                research_items = await self.research_processor.fetch_recent_research()
                
                # Process and store knowledge
                for item in research_items:
                    knowledge_item = await self.process_research_knowledge(item)
                    if knowledge_item:
                        await self.store_knowledge(knowledge_item)
                
                logger.info(f"Research learning cycle completed: {len(research_items)} items processed")
                
                # Wait 12 hours before next cycle
                await asyncio.sleep(12 * 3600)
                
            except Exception as e:
                logger.error(f"Error in research learning cycle: {e}")
                await asyncio.sleep(3600)
    
    async def model_training_cycle(self):
        """Periodic ML model training cycle"""
        while True:
            try:
                logger.info("Starting model training cycle")
                
                # Get recent knowledge for training
                knowledge_items = await self.get_recent_knowledge_for_training()
                
                if len(knowledge_items) > 100:  # Only train with sufficient data
                    # Train vulnerability classifier
                    classifier_model = await self.ml_trainer.train_vulnerability_classifier(knowledge_items)
                    
                    # Extract payloads and train payload generator
                    payloads = []
                    vuln_types = []
                    for item in knowledge_items:
                        if item.exploitation_techniques:
                            payloads.extend(item.exploitation_techniques)
                            vuln_types.extend([item.vulnerability_type] * len(item.exploitation_techniques))
                    
                    if payloads:
                        payload_model = await self.ml_trainer.train_payload_generator(payloads, vuln_types)
                    
                    logger.info("Model training cycle completed")
                else:
                    logger.info("Insufficient data for model training")
                
                # Wait 7 days before next training cycle
                await asyncio.sleep(7 * 24 * 3600)
                
            except Exception as e:
                logger.error(f"Error in model training cycle: {e}")
                await asyncio.sleep(24 * 3600)  # Wait 24 hours on error
    
    async def process_cve_knowledge(self, cve_data: Dict[str, Any]) -> Optional[SecurityKnowledge]:
        """Process CVE data into knowledge item"""
        try:
            return SecurityKnowledge(
                knowledge_id=f"cve_{cve_data['cve_id']}_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                source='cve',
                source_url=cve_data.get('source_url', ''),
                title=cve_data['cve_id'],
                description=cve_data['description'],
                vulnerability_type=cve_data['vulnerability_type'],
                affected_technologies=cve_data['affected_technologies'],
                attack_vectors=[cve_data['vulnerability_type']],
                exploitation_techniques=[],  # Would extract from description
                detection_signatures=[],     # Would generate based on vuln type
                remediation_steps=[],        # Would extract from advisories
                severity_score=cve_data['cvss_score'],
                confidence_score=0.9 if cve_data.get('actively_exploited') else 0.7,
                tags=['cve', cve_data['vulnerability_type']],
                references=[cve_data.get('source_url', '')],
                learned_patterns={},
                ml_features={}
            )
            
        except Exception as e:
            logger.debug(f"Error processing CVE knowledge: {e}")
            return None
    
    async def process_hackerone_knowledge(self, report: Dict[str, Any]) -> Optional[SecurityKnowledge]:
        """Process HackerOne report into knowledge item"""
        try:
            techniques = self.hackerone_processor.extract_techniques_from_report(report)
            
            return SecurityKnowledge(
                knowledge_id=f"h1_{report['report_id']}_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                source='hackerone',
                source_url=f"https://hackerone.com/reports/{report['report_id']}",
                title=report['title'],
                description=report['description'],
                vulnerability_type=report['vulnerability_type'],
                affected_technologies=[],
                attack_vectors=techniques['attack_vectors'],
                exploitation_techniques=techniques['payloads'],
                detection_signatures=[],
                remediation_steps=[],
                severity_score=self.severity_to_score(report.get('severity', 'medium')),
                confidence_score=0.95,  # HackerOne reports are high confidence
                tags=['hackerone', report['vulnerability_type']],
                references=[],
                learned_patterns=techniques,
                ml_features={}
            )
            
        except Exception as e:
            logger.debug(f"Error processing HackerOne knowledge: {e}")
            return None
    
    async def process_research_knowledge(self, item: Dict[str, Any]) -> Optional[SecurityKnowledge]:
        """Process research item into knowledge item"""
        try:
            # Extract vulnerability type from content
            content = item.get('content', '') + ' ' + item.get('description', '')
            vuln_type = self.cve_processor.classify_vulnerability_type(content)
            
            return SecurityKnowledge(
                knowledge_id=f"research_{hash(item['url']) % 100000}_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                source='research',
                source_url=item['url'],
                title=item['title'],
                description=item['description'],
                vulnerability_type=vuln_type,
                affected_technologies=[],
                attack_vectors=[],
                exploitation_techniques=[],
                detection_signatures=[],
                remediation_steps=[],
                severity_score=5.0,  # Default medium severity
                confidence_score=0.6,  # Research has variable confidence
                tags=item.get('tags', []) + ['research'],
                references=[item['url']],
                learned_patterns={},
                ml_features={}
            )
            
        except Exception as e:
            logger.debug(f"Error processing research knowledge: {e}")
            return None
    
    def severity_to_score(self, severity: str) -> float:
        """Convert severity string to numeric score"""
        severity_map = {
            'critical': 9.0,
            'high': 7.0,
            'medium': 5.0,
            'low': 3.0,
            'info': 1.0
        }
        return severity_map.get(severity.lower(), 5.0)
    
    async def store_knowledge(self, knowledge: SecurityKnowledge):
        """Store knowledge item in Firestore and BigQuery"""
        try:
            # Store in Firestore
            knowledge_ref = self.firestore_client.collection('security_knowledge')
            knowledge_data = asdict(knowledge)
            knowledge_data['timestamp'] = knowledge.timestamp.isoformat()
            knowledge_ref.add(knowledge_data)
            
            # Store in BigQuery
            await self.store_knowledge_bigquery(knowledge)
            
            # Publish knowledge update
            await self.publish_knowledge_update(knowledge)
            
        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
    
    async def store_knowledge_bigquery(self, knowledge: SecurityKnowledge):
        """Store knowledge in BigQuery"""
        try:
            table_ref = self.bigquery_client.dataset("security_knowledge").table("security_knowledge")
            
            row = asdict(knowledge)
            row['timestamp'] = knowledge.timestamp.isoformat()
            
            errors = self.bigquery_client.insert_rows_json(table_ref, [row])
            if errors:
                logger.error(f"BigQuery knowledge insert errors: {errors}")
                
        except Exception as e:
            logger.error(f"Error storing knowledge in BigQuery: {e}")
    
    async def publish_knowledge_update(self, knowledge: SecurityKnowledge):
        """Publish knowledge update to other agents"""
        try:
            knowledge_data = asdict(knowledge)
            knowledge_data['timestamp'] = knowledge.timestamp.isoformat()
            
            message = json.dumps(knowledge_data).encode('utf-8')
            future = self.pubsub_client.publish(self.knowledge_topic, message)
            message_id = future.result()
            
            logger.debug(f"Published knowledge update: {message_id}")
            
        except Exception as e:
            logger.error(f"Error publishing knowledge update: {e}")
    
    async def get_recent_knowledge_for_training(self) -> List[SecurityKnowledge]:
        """Get recent knowledge items for ML training"""
        try:
            knowledge_ref = self.firestore_client.collection('security_knowledge')
            
            # Get knowledge from last 30 days
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            query = knowledge_ref.where('timestamp', '>=', cutoff_date).limit(1000)
            
            docs = query.get()
            knowledge_items = []
            
            for doc in docs:
                data = doc.to_dict()
                # Convert back to SecurityKnowledge object
                data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                knowledge_items.append(SecurityKnowledge(**data))
            
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Error getting knowledge for training: {e}")
            return []

async def main():
    """Main entry point"""
    learner = KnowledgeLearner()
    await learner.initialize()
    
    # Keep the learner running
    try:
        while True:
            await asyncio.sleep(3600)  # Check every hour
            
    except KeyboardInterrupt:
        logger.info("Knowledge Learner shutting down")

if __name__ == "__main__":
    asyncio.run(main())