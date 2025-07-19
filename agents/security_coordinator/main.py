#!/usr/bin/env python3
"""
Aetherveil Security Coordinator Agent
Orchestrates defensive security testing operations across the multi-agent system.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import yaml
from google.cloud import pubsub_v1, firestore, bigquery
from google.cloud.pubsub_v1.futures import Future

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityTarget:
    """Security testing target definition"""
    target_id: str
    program_name: str
    domain: str
    subdomain: str
    ports: List[int]
    services: List[str]
    scope_type: str  # in_scope, out_of_scope, unknown
    priority: str    # critical, high, medium, low
    last_tested: Optional[datetime]
    test_types: List[str]

@dataclass
class ScanJob:
    """Security scan job definition"""
    job_id: str
    target: SecurityTarget
    scan_type: str
    agent_type: str
    priority: int
    created_at: datetime
    status: str  # queued, running, completed, failed
    assigned_agent: Optional[str]

class SecurityCoordinator:
    """Main coordinator for security testing operations"""
    
    def __init__(self):
        self.project_id = os.getenv('PROJECT_ID', 'tidy-computing-465909-i3')
        self.pubsub_client = pubsub_v1.PublisherClient()
        self.subscriber_client = pubsub_v1.SubscriberClient()
        self.firestore_client = firestore.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        
        # Pub/Sub topics
        self.topics = {
            'discovery_requests': f"projects/{self.project_id}/topics/aetherveil-discovery-requests",
            'assessment_requests': f"projects/{self.project_id}/topics/aetherveil-assessment-requests",
            'learning_requests': f"projects/{self.project_id}/topics/aetherveil-learning-requests",
            'report_requests': f"projects/{self.project_id}/topics/aetherveil-report-requests",
            'findings': f"projects/{self.project_id}/topics/aetherveil-security-findings",
            'discoveries': f"projects/{self.project_id}/topics/aetherveil-asset-discoveries",
            'alerts': f"projects/{self.project_id}/topics/aetherveil-security-alerts"
        }
        
        # Subscriptions
        self.subscriptions = {
            'findings': f"projects/{self.project_id}/subscriptions/coordinator-findings",
            'discoveries': f"projects/{self.project_id}/subscriptions/coordinator-discoveries"
        }
        
        # Agent capabilities
        self.agent_capabilities = {
            'discovery_agent': ['subdomain_enum', 'asset_discovery', 'service_mapping'],
            'assessment_agent': ['vulnerability_scan', 'config_audit', 'safe_validation'],
            'learning_agent': ['cve_analysis', 'technique_update', 'model_training'],
            'report_agent': ['report_generation', 'risk_scoring', 'remediation_advice']
        }
        
    async def initialize(self):
        """Initialize coordinator and start listening for events"""
        logger.info("Initializing Security Coordinator")
        
        # Initialize Firestore collections
        await self.setup_firestore_collections()
        
        # Start message handlers
        await self.start_message_handlers()
        
    async def setup_firestore_collections(self):
        """Setup Firestore collections for coordination"""
        try:
            # Targets collection
            targets_ref = self.firestore_client.collection('security_targets')
            
            # Jobs collection  
            jobs_ref = self.firestore_client.collection('scan_jobs')
            
            # Agent status collection
            agents_ref = self.firestore_client.collection('agent_status')
            
            logger.info("Firestore collections initialized")
            
        except Exception as e:
            logger.error(f"Error setting up Firestore: {e}")
    
    async def start_message_handlers(self):
        """Start Pub/Sub message handlers"""
        try:
            # Handle findings from agents
            findings_future = self.subscriber_client.subscribe(
                self.subscriptions['findings'],
                callback=self.handle_findings_message,
                flow_control=pubsub_v1.types.FlowControl(max_messages=100)
            )
            
            # Handle discoveries from discovery agent
            discoveries_future = self.subscriber_client.subscribe(
                self.subscriptions['discoveries'],
                callback=self.handle_discoveries_message,
                flow_control=pubsub_v1.types.FlowControl(max_messages=100)
            )
            
            logger.info("Message handlers started")
            
        except Exception as e:
            logger.error(f"Error starting message handlers: {e}")
    
    def handle_findings_message(self, message):
        """Handle findings messages from agents"""
        try:
            data = json.loads(message.data.decode())
            asyncio.create_task(self.process_finding(data))
            message.ack()
            
        except Exception as e:
            logger.error(f"Error handling findings message: {e}")
            message.nack()
    
    def handle_discoveries_message(self, message):
        """Handle discovery messages from agents"""
        try:
            data = json.loads(message.data.decode())
            asyncio.create_task(self.process_discovery(data))
            message.ack()
            
        except Exception as e:
            logger.error(f"Error handling discoveries message: {e}")
            message.nack()
    
    async def process_finding(self, finding_data: Dict[str, Any]):
        """Process a security finding"""
        try:
            logger.info(f"Processing finding: {finding_data.get('finding_id', 'unknown')}")
            
            # Store finding in Firestore
            findings_ref = self.firestore_client.collection('security_findings')
            findings_ref.add(finding_data)
            
            # Trigger report generation for high-severity findings
            severity = finding_data.get('severity', 'low')
            if severity in ['critical', 'high']:
                await self.trigger_report_generation(finding_data)
            
            # Update target status
            target_url = finding_data.get('target_url', '')
            if target_url:
                await self.update_target_status(target_url, 'tested')
            
        except Exception as e:
            logger.error(f"Error processing finding: {e}")
    
    async def process_discovery(self, discovery_data: Dict[str, Any]):
        """Process an asset discovery"""
        try:
            logger.info(f"Processing discovery: {discovery_data.get('asset_type', 'unknown')}")
            
            # Create new security target
            target = SecurityTarget(
                target_id=f"target_{int(datetime.now().timestamp())}",
                program_name=discovery_data.get('program_name', 'unknown'),
                domain=discovery_data.get('domain', ''),
                subdomain=discovery_data.get('subdomain', ''),
                ports=discovery_data.get('ports', []),
                services=discovery_data.get('services', []),
                scope_type=discovery_data.get('scope_type', 'unknown'),
                priority=self.calculate_target_priority(discovery_data),
                last_tested=None,
                test_types=['vulnerability_scan', 'config_audit']
            )
            
            # Store target in Firestore
            targets_ref = self.firestore_client.collection('security_targets')
            targets_ref.document(target.target_id).set(asdict(target))
            
            # Schedule assessment if in scope
            if target.scope_type == 'in_scope':
                await self.schedule_assessment(target)
            
        except Exception as e:
            logger.error(f"Error processing discovery: {e}")
    
    def calculate_target_priority(self, discovery_data: Dict[str, Any]) -> str:
        """Calculate priority for a discovered target"""
        # Simple priority calculation based on services and exposure
        services = discovery_data.get('services', [])
        ports = discovery_data.get('ports', [])
        
        # High priority for sensitive services
        high_risk_services = ['ssh', 'ftp', 'telnet', 'smtp', 'mysql', 'postgresql']
        if any(service.lower() in high_risk_services for service in services):
            return 'high'
        
        # Medium priority for web services
        web_ports = [80, 443, 8080, 8443]
        if any(port in web_ports for port in ports):
            return 'medium'
        
        return 'low'
    
    async def schedule_assessment(self, target: SecurityTarget):
        """Schedule security assessment for a target"""
        try:
            # Create scan job
            job = ScanJob(
                job_id=f"job_{int(datetime.now().timestamp())}",
                target=target,
                scan_type='comprehensive',
                agent_type='assessment_agent',
                priority=self.get_priority_score(target.priority),
                created_at=datetime.now(timezone.utc),
                status='queued',
                assigned_agent=None
            )
            
            # Store job in Firestore
            jobs_ref = self.firestore_client.collection('scan_jobs')
            jobs_ref.document(job.job_id).set(asdict(job))
            
            # Send assessment request
            await self.send_assessment_request(job)
            
            logger.info(f"Scheduled assessment job: {job.job_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling assessment: {e}")
    
    def get_priority_score(self, priority: str) -> int:
        """Convert priority string to numeric score"""
        priority_map = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return priority_map.get(priority, 1)
    
    async def send_assessment_request(self, job: ScanJob):
        """Send assessment request to agents"""
        try:
            request_data = {
                'job_id': job.job_id,
                'target': asdict(job.target),
                'scan_type': job.scan_type,
                'priority': job.priority,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            message = json.dumps(request_data).encode('utf-8')
            future = self.pubsub_client.publish(self.topics['assessment_requests'], message)
            message_id = future.result()
            
            logger.info(f"Sent assessment request: {message_id}")
            
        except Exception as e:
            logger.error(f"Error sending assessment request: {e}")
    
    async def trigger_report_generation(self, finding_data: Dict[str, Any]):
        """Trigger report generation for important findings"""
        try:
            report_request = {
                'finding_id': finding_data.get('finding_id'),
                'severity': finding_data.get('severity'),
                'target_url': finding_data.get('target_url'),
                'vulnerability_type': finding_data.get('vulnerability_type'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'report_type': 'immediate'
            }
            
            message = json.dumps(report_request).encode('utf-8')
            future = self.pubsub_client.publish(self.topics['report_requests'], message)
            message_id = future.result()
            
            logger.info(f"Triggered report generation: {message_id}")
            
        except Exception as e:
            logger.error(f"Error triggering report generation: {e}")
    
    async def update_target_status(self, target_url: str, status: str):
        """Update target testing status"""
        try:
            # Query for target by URL
            targets_ref = self.firestore_client.collection('security_targets')
            query = targets_ref.where('domain', '==', target_url).limit(1)
            
            docs = query.get()
            for doc in docs:
                doc.reference.update({
                    'last_tested': datetime.now(timezone.utc),
                    'status': status
                })
                break
            
        except Exception as e:
            logger.error(f"Error updating target status: {e}")
    
    async def start_discovery_sweep(self, program_config: Dict[str, Any]):
        """Start a discovery sweep for a bug bounty program"""
        try:
            logger.info(f"Starting discovery sweep for: {program_config.get('program_name')}")
            
            discovery_request = {
                'sweep_id': f"sweep_{int(datetime.now().timestamp())}",
                'program_name': program_config.get('program_name'),
                'domains': program_config.get('domains', []),
                'scope_type': 'in_scope',
                'discovery_types': ['subdomain_enum', 'asset_discovery', 'service_mapping'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            message = json.dumps(discovery_request).encode('utf-8')
            future = self.pubsub_client.publish(self.topics['discovery_requests'], message)
            message_id = future.result()
            
            logger.info(f"Started discovery sweep: {message_id}")
            
        except Exception as e:
            logger.error(f"Error starting discovery sweep: {e}")
    
    async def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all activities"""
        try:
            # Query findings from Firestore
            findings_ref = self.firestore_client.collection('security_findings')
            findings = findings_ref.limit(100).get()
            
            # Query targets
            targets_ref = self.firestore_client.collection('security_targets')
            targets = targets_ref.limit(100).get()
            
            # Generate summary
            summary = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_targets': len(targets),
                'total_findings': len(findings),
                'severity_breakdown': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                },
                'recent_activity': []
            }
            
            # Count findings by severity
            for finding in findings:
                finding_data = finding.to_dict()
                severity = finding_data.get('severity', 'low')
                if severity in summary['severity_breakdown']:
                    summary['severity_breakdown'][severity] += 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {}

async def main():
    """Main entry point"""
    coordinator = SecurityCoordinator()
    await coordinator.initialize()
    
    # Example: Start discovery for a program
    program_config = {
        'program_name': 'Example Bug Bounty Program',
        'domains': ['example.com', 'api.example.com']
    }
    
    await coordinator.start_discovery_sweep(program_config)
    
    # Keep coordinator running
    try:
        while True:
            summary = await coordinator.generate_summary_report()
            logger.info(f"System summary: {summary}")
            await asyncio.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        logger.info("Coordinator shutting down")

if __name__ == "__main__":
    asyncio.run(main())