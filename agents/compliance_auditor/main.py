#!/usr/bin/env python3
"""
Aetherveil Compliance Auditor Agent
Ensures all Red Team activities strictly adhere to scope and ethical boundaries.
Critical component for maintaining ethical bug bounty and authorized penetration testing.
"""

import os
import json
import asyncio
import logging
import ipaddress
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
import yaml
from google.cloud import pubsub_v1, firestore, bigquery, logging as cloud_logging
from urllib.parse import urlparse
import dns.resolver
import tldextract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScopeDefinition:
    """Defines testing scope for a program"""
    program_id: str
    program_name: str
    in_scope_domains: List[str]
    in_scope_ips: List[str]
    in_scope_subdomains: List[str]
    out_of_scope_domains: List[str]
    out_of_scope_ips: List[str]
    out_of_scope_paths: List[str]
    allowed_methods: List[str]
    rate_limits: Dict[str, int]
    blackout_windows: List[Dict[str, str]]
    special_instructions: str
    contact_email: str
    last_updated: datetime

@dataclass
class ComplianceViolation:
    """Records a compliance violation"""
    violation_id: str
    timestamp: datetime
    agent_id: str
    violation_type: str  # out_of_scope, rate_limit, blackout, unauthorized_method
    target: str
    description: str
    severity: str  # critical, high, medium, low
    action_taken: str
    prevented: bool

@dataclass
class ActivityLog:
    """Logs all Red Team activities for audit trail"""
    activity_id: str
    timestamp: datetime
    agent_id: str
    activity_type: str
    target: str
    method: str
    scope_validated: bool
    rate_limit_respected: bool
    details: Dict[str, Any]
    compliance_status: str

class ScopeValidator:
    """Validates if targets are within authorized scope"""
    
    def __init__(self):
        self.domain_cache = {}
        self.ip_cache = {}
    
    def validate_domain(self, target_domain: str, scope: ScopeDefinition) -> Tuple[bool, str]:
        """Validate if domain is in scope"""
        try:
            # Extract domain parts
            extracted = tldextract.extract(target_domain)
            root_domain = f"{extracted.domain}.{extracted.suffix}"
            
            # Check exact domain matches
            if target_domain in scope.in_scope_domains:
                return True, "exact_match"
            
            # Check if subdomain of in-scope domain
            for allowed_domain in scope.in_scope_domains:
                if target_domain.endswith(f".{allowed_domain}"):
                    return True, "subdomain_match"
                if allowed_domain.startswith("*.") and target_domain.endswith(allowed_domain[2:]):
                    return True, "wildcard_match"
            
            # Check root domain
            if root_domain in scope.in_scope_domains:
                return True, "root_domain_match"
            
            # Check against out-of-scope
            if target_domain in scope.out_of_scope_domains:
                return False, "explicitly_excluded"
            
            for excluded_domain in scope.out_of_scope_domains:
                if target_domain.endswith(f".{excluded_domain}"):
                    return False, "subdomain_excluded"
            
            return False, "not_in_scope"
            
        except Exception as e:
            logger.error(f"Error validating domain {target_domain}: {e}")
            return False, "validation_error"
    
    def validate_ip(self, target_ip: str, scope: ScopeDefinition) -> Tuple[bool, str]:
        """Validate if IP address is in scope"""
        try:
            target_addr = ipaddress.ip_address(target_ip)
            
            # Check exact IP matches
            if target_ip in scope.in_scope_ips:
                return True, "exact_ip_match"
            
            # Check IP ranges/networks
            for allowed_ip in scope.in_scope_ips:
                try:
                    if '/' in allowed_ip:  # CIDR notation
                        network = ipaddress.ip_network(allowed_ip, strict=False)
                        if target_addr in network:
                            return True, "ip_range_match"
                except:
                    pass
            
            # Check against out-of-scope IPs
            if target_ip in scope.out_of_scope_ips:
                return False, "ip_excluded"
            
            for excluded_ip in scope.out_of_scope_ips:
                try:
                    if '/' in excluded_ip:
                        network = ipaddress.ip_network(excluded_ip, strict=False)
                        if target_addr in network:
                            return False, "ip_range_excluded"
                except:
                    pass
            
            return False, "ip_not_in_scope"
            
        except Exception as e:
            logger.error(f"Error validating IP {target_ip}: {e}")
            return False, "ip_validation_error"
    
    def validate_url(self, target_url: str, scope: ScopeDefinition) -> Tuple[bool, str]:
        """Validate if URL is in scope"""
        try:
            parsed = urlparse(target_url)
            hostname = parsed.hostname
            path = parsed.path
            
            # Validate domain/IP
            if hostname:
                # Try as domain first
                domain_valid, domain_reason = self.validate_domain(hostname, scope)
                if domain_valid:
                    # Check path exclusions
                    for excluded_path in scope.out_of_scope_paths:
                        if path.startswith(excluded_path):
                            return False, f"path_excluded: {excluded_path}"
                    return True, domain_reason
                
                # Try as IP if domain validation failed
                try:
                    ip_valid, ip_reason = self.validate_ip(hostname, scope)
                    if ip_valid:
                        return True, ip_reason
                except:
                    pass
                
                return False, domain_reason
            
            return False, "no_hostname"
            
        except Exception as e:
            logger.error(f"Error validating URL {target_url}: {e}")
            return False, "url_validation_error"

class RateLimitEnforcer:
    """Enforces rate limits to prevent DoS and respect target infrastructure"""
    
    def __init__(self):
        self.request_history: Dict[str, List[datetime]] = {}
        self.global_limits = {
            'requests_per_second': 2,
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'concurrent_connections': 10
        }
    
    def check_rate_limit(self, target: str, scope: ScopeDefinition) -> Tuple[bool, str]:
        """Check if request is within rate limits"""
        now = datetime.now(timezone.utc)
        
        # Get target-specific limits or use defaults
        target_limits = scope.rate_limits.get(target, self.global_limits)
        
        # Initialize history if needed
        if target not in self.request_history:
            self.request_history[target] = []
        
        history = self.request_history[target]
        
        # Clean old entries
        cutoff_time = now - timedelta(hours=1)
        history = [req_time for req_time in history if req_time > cutoff_time]
        self.request_history[target] = history
        
        # Check various time windows
        checks = [
            (timedelta(seconds=1), target_limits.get('requests_per_second', 2), 'per_second'),
            (timedelta(minutes=1), target_limits.get('requests_per_minute', 60), 'per_minute'),
            (timedelta(hours=1), target_limits.get('requests_per_hour', 1000), 'per_hour')
        ]
        
        for window, limit, window_name in checks:
            window_start = now - window
            requests_in_window = len([t for t in history if t > window_start])
            
            if requests_in_window >= limit:
                return False, f"rate_limit_exceeded_{window_name}"
        
        # Record this request
        history.append(now)
        return True, "rate_limit_ok"
    
    def is_blackout_period(self, scope: ScopeDefinition) -> Tuple[bool, str]:
        """Check if current time is within blackout window"""
        now = datetime.now(timezone.utc)
        current_time = now.time()
        current_day = now.strftime('%A').lower()
        
        for blackout in scope.blackout_windows:
            # Check day of week
            if 'days' in blackout:
                allowed_days = [day.strip().lower() for day in blackout['days'].split(',')]
                if current_day not in allowed_days:
                    continue
            
            # Check time window
            if 'start_time' in blackout and 'end_time' in blackout:
                start_time = datetime.strptime(blackout['start_time'], '%H:%M').time()
                end_time = datetime.strptime(blackout['end_time'], '%H:%M').time()
                
                if start_time <= current_time <= end_time:
                    return True, f"blackout_period: {blackout.get('reason', 'scheduled_maintenance')}"
        
        return False, "not_blackout"

class ComplianceAuditor:
    """Main compliance auditing and enforcement agent"""
    
    def __init__(self):
        self.project_id = os.getenv('PROJECT_ID', 'tidy-computing-465909-i3')
        self.pubsub_client = pubsub_v1.PublisherClient()
        self.subscriber_client = pubsub_v1.SubscriberClient()
        self.firestore_client = firestore.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        self.cloud_logging_client = cloud_logging.Client(project=self.project_id)
        
        # Initialize components
        self.scope_validator = ScopeValidator()
        self.rate_enforcer = RateLimitEnforcer()
        
        # Topics and subscriptions
        self.topics = {
            'violations': f"projects/{self.project_id}/topics/aetherveil-compliance-violations",
            'activity_logs': f"projects/{self.project_id}/topics/aetherveil-activity-logs",
            'alerts': f"projects/{self.project_id}/topics/aetherveil-security-alerts"
        }
        
        self.subscriptions = {
            'all_activities': f"projects/{self.project_id}/subscriptions/compliance-all-activities"
        }
        
        # In-memory scope cache
        self.scope_cache: Dict[str, ScopeDefinition] = {}
        
    async def initialize(self):
        """Initialize the compliance auditor"""
        logger.info("Initializing Compliance Auditor Agent")
        
        # Load all scope definitions
        await self.load_scope_definitions()
        
        # Setup BigQuery tables for audit logging
        await self.setup_audit_tables()
        
        # Start monitoring all agent activities
        await self.start_activity_monitoring()
    
    async def load_scope_definitions(self):
        """Load scope definitions from Firestore"""
        try:
            scopes_ref = self.firestore_client.collection('program_scopes')
            docs = scopes_ref.stream()
            
            for doc in docs:
                scope_data = doc.to_dict()
                scope = ScopeDefinition(**scope_data)
                self.scope_cache[scope.program_id] = scope
                
            logger.info(f"Loaded {len(self.scope_cache)} scope definitions")
            
        except Exception as e:
            logger.error(f"Error loading scope definitions: {e}")
    
    async def setup_audit_tables(self):
        """Setup BigQuery tables for audit logging"""
        try:
            dataset_id = "compliance_audit"
            
            # Create dataset if not exists
            try:
                self.bigquery_client.get_dataset(dataset_id)
            except:
                dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
                dataset.location = "US"
                self.bigquery_client.create_dataset(dataset)
            
            # Activity logs table
            activity_schema = [
                bigquery.SchemaField("activity_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("agent_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("activity_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("target", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("method", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("scope_validated", "BOOLEAN", mode="REQUIRED"),
                bigquery.SchemaField("rate_limit_respected", "BOOLEAN", mode="REQUIRED"),
                bigquery.SchemaField("compliance_status", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("details", "JSON", mode="NULLABLE"),
            ]
            
            self.create_table_if_not_exists(dataset_id, "activity_logs", activity_schema)
            
            # Violations table
            violation_schema = [
                bigquery.SchemaField("violation_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("agent_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("violation_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("target", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("severity", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("action_taken", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("prevented", "BOOLEAN", mode="REQUIRED"),
            ]
            
            self.create_table_if_not_exists(dataset_id, "compliance_violations", violation_schema)
            
            logger.info("Audit tables setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up audit tables: {e}")
    
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
    
    async def start_activity_monitoring(self):
        """Start monitoring all agent activities"""
        try:
            future = self.subscriber_client.subscribe(
                self.subscriptions['all_activities'],
                callback=self.handle_activity_message,
                flow_control=pubsub_v1.types.FlowControl(max_messages=100)
            )
            
            logger.info("Started activity monitoring")
            
        except Exception as e:
            logger.error(f"Error starting activity monitoring: {e}")
    
    def handle_activity_message(self, message):
        """Handle activity messages from other agents"""
        try:
            data = json.loads(message.data.decode())
            asyncio.create_task(self.audit_activity(data))
            message.ack()
            
        except Exception as e:
            logger.error(f"Error handling activity message: {e}")
            message.nack()
    
    async def audit_activity(self, activity_data: Dict[str, Any]):
        """Audit an agent activity for compliance"""
        try:
            activity_id = activity_data.get('activity_id', f"audit_{int(datetime.now().timestamp())}")
            agent_id = activity_data.get('agent_id', 'unknown')
            target = activity_data.get('target', '')
            activity_type = activity_data.get('activity_type', 'unknown')
            
            logger.info(f"Auditing activity: {activity_id} from {agent_id}")
            
            # Determine program scope
            program_id = activity_data.get('program_id')
            if not program_id or program_id not in self.scope_cache:
                await self.record_violation(
                    agent_id, 'no_scope_definition', target,
                    f"No scope definition found for program: {program_id}",
                    'critical', 'activity_blocked', True
                )
                return
            
            scope = self.scope_cache[program_id]
            
            # Validate scope
            scope_valid, scope_reason = self.validate_target_scope(target, scope)
            
            # Check rate limits
            rate_ok, rate_reason = self.rate_enforcer.check_rate_limit(target, scope)
            
            # Check blackout periods
            blackout_active, blackout_reason = self.rate_enforcer.is_blackout_period(scope)
            
            # Determine compliance status
            compliance_status = "compliant"
            violations = []
            
            if not scope_valid:
                compliance_status = "violation"
                violations.append(('out_of_scope', scope_reason))
            
            if not rate_ok:
                compliance_status = "violation"
                violations.append(('rate_limit', rate_reason))
            
            if blackout_active:
                compliance_status = "violation"  
                violations.append(('blackout_period', blackout_reason))
            
            # Record violations
            for violation_type, reason in violations:
                await self.record_violation(
                    agent_id, violation_type, target, reason,
                    'high', 'activity_blocked', True
                )
            
            # Log activity
            await self.log_activity(ActivityLog(
                activity_id=activity_id,
                timestamp=datetime.now(timezone.utc),
                agent_id=agent_id,
                activity_type=activity_type,
                target=target,
                method=activity_data.get('method', 'unknown'),
                scope_validated=scope_valid,
                rate_limit_respected=rate_ok and not blackout_active,
                details=activity_data,
                compliance_status=compliance_status
            ))
            
        except Exception as e:
            logger.error(f"Error auditing activity: {e}")
    
    def validate_target_scope(self, target: str, scope: ScopeDefinition) -> Tuple[bool, str]:
        """Validate if target is within scope"""
        # Handle different target formats
        if target.startswith('http'):
            return self.scope_validator.validate_url(target, scope)
        elif '.' in target and not target.replace('.', '').isdigit():
            return self.scope_validator.validate_domain(target, scope)
        else:
            return self.scope_validator.validate_ip(target, scope)
    
    async def record_violation(self, agent_id: str, violation_type: str, target: str, 
                             description: str, severity: str, action_taken: str, prevented: bool):
        """Record a compliance violation"""
        try:
            violation = ComplianceViolation(
                violation_id=f"violation_{int(datetime.now().timestamp())}_{agent_id}",
                timestamp=datetime.now(timezone.utc),
                agent_id=agent_id,
                violation_type=violation_type,
                target=target,
                description=description,
                severity=severity,
                action_taken=action_taken,
                prevented=prevented
            )
            
            # Store in Firestore
            violations_ref = self.firestore_client.collection('compliance_violations')
            violations_ref.add(asdict(violation))
            
            # Log to BigQuery
            await self.store_violation_bigquery(violation)
            
            # Send alert for high/critical violations
            if severity in ['high', 'critical']:
                await self.send_compliance_alert(violation)
            
            logger.warning(f"Compliance violation recorded: {violation.violation_id}")
            
        except Exception as e:
            logger.error(f"Error recording violation: {e}")
    
    async def log_activity(self, activity: ActivityLog):
        """Log agent activity for audit trail"""
        try:
            # Store in Firestore
            activities_ref = self.firestore_client.collection('activity_logs')
            activity_dict = asdict(activity)
            activity_dict['timestamp'] = activity.timestamp.isoformat()
            activities_ref.add(activity_dict)
            
            # Log to BigQuery
            await self.store_activity_bigquery(activity)
            
            # Publish to Pub/Sub for real-time monitoring
            message = json.dumps(activity_dict).encode('utf-8')
            future = self.pubsub_client.publish(self.topics['activity_logs'], message)
            future.result()
            
        except Exception as e:
            logger.error(f"Error logging activity: {e}")
    
    async def store_violation_bigquery(self, violation: ComplianceViolation):
        """Store violation in BigQuery"""
        try:
            table_ref = self.bigquery_client.dataset("compliance_audit").table("compliance_violations")
            
            row = asdict(violation)
            row['timestamp'] = violation.timestamp.isoformat()
            
            errors = self.bigquery_client.insert_rows_json(table_ref, [row])
            if errors:
                logger.error(f"BigQuery violation insert errors: {errors}")
                
        except Exception as e:
            logger.error(f"Error storing violation in BigQuery: {e}")
    
    async def store_activity_bigquery(self, activity: ActivityLog):
        """Store activity in BigQuery"""
        try:
            table_ref = self.bigquery_client.dataset("compliance_audit").table("activity_logs")
            
            row = asdict(activity)
            row['timestamp'] = activity.timestamp.isoformat()
            
            errors = self.bigquery_client.insert_rows_json(table_ref, [row])
            if errors:
                logger.error(f"BigQuery activity insert errors: {errors}")
                
        except Exception as e:
            logger.error(f"Error storing activity in BigQuery: {e}")
    
    async def send_compliance_alert(self, violation: ComplianceViolation):
        """Send alert for compliance violations"""
        try:
            alert_data = {
                'alert_type': 'compliance_violation',
                'violation_id': violation.violation_id,
                'severity': violation.severity,
                'agent_id': violation.agent_id,
                'violation_type': violation.violation_type,
                'target': violation.target,
                'description': violation.description,
                'timestamp': violation.timestamp.isoformat(),
                'action_taken': violation.action_taken
            }
            
            message = json.dumps(alert_data).encode('utf-8')
            future = self.pubsub_client.publish(self.topics['alerts'], message)
            message_id = future.result()
            
            logger.warning(f"Compliance alert sent: {message_id}")
            
        except Exception as e:
            logger.error(f"Error sending compliance alert: {e}")
    
    async def validate_request(self, agent_id: str, target: str, method: str, 
                             program_id: str) -> Tuple[bool, str]:
        """Validate if a request is compliant before execution"""
        try:
            if program_id not in self.scope_cache:
                return False, "no_scope_definition"
            
            scope = self.scope_cache[program_id]
            
            # Scope validation
            scope_valid, scope_reason = self.validate_target_scope(target, scope)
            if not scope_valid:
                return False, f"out_of_scope: {scope_reason}"
            
            # Rate limit check
            rate_ok, rate_reason = self.rate_enforcer.check_rate_limit(target, scope)
            if not rate_ok:
                return False, f"rate_limit: {rate_reason}"
            
            # Blackout check
            blackout_active, blackout_reason = self.rate_enforcer.is_blackout_period(scope)
            if blackout_active:
                return False, f"blackout: {blackout_reason}"
            
            # Method validation
            if method not in scope.allowed_methods:
                return False, f"method_not_allowed: {method}"
            
            return True, "compliant"
            
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return False, f"validation_error: {str(e)}"
    
    async def generate_compliance_report(self, program_id: str) -> Dict[str, Any]:
        """Generate compliance report for a program"""
        try:
            # Query activities and violations from BigQuery
            query = f"""
            SELECT 
                compliance_status,
                COUNT(*) as count
            FROM `{self.project_id}.compliance_audit.activity_logs`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            GROUP BY compliance_status
            """
            
            activities_query = self.bigquery_client.query(query)
            activities_results = activities_query.result()
            
            violations_query = f"""
            SELECT 
                violation_type,
                severity,
                COUNT(*) as count
            FROM `{self.project_id}.compliance_audit.compliance_violations`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            GROUP BY violation_type, severity
            """
            
            violations_results = self.bigquery_client.query(violations_query).result()
            
            # Compile report
            report = {
                'program_id': program_id,
                'report_date': datetime.now(timezone.utc).isoformat(),
                'period': 'last_30_days',
                'activity_summary': {},
                'violation_summary': {},
                'compliance_score': 0.0
            }
            
            total_activities = 0
            compliant_activities = 0
            
            for row in activities_results:
                report['activity_summary'][row.compliance_status] = row.count
                total_activities += row.count
                if row.compliance_status == 'compliant':
                    compliant_activities += row.count
            
            for row in violations_results:
                key = f"{row.violation_type}_{row.severity}"
                report['violation_summary'][key] = row.count
            
            # Calculate compliance score
            if total_activities > 0:
                report['compliance_score'] = (compliant_activities / total_activities) * 100
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {}

async def main():
    """Main entry point"""
    auditor = ComplianceAuditor()
    await auditor.initialize()
    
    # Keep the auditor running
    try:
        while True:
            # Generate periodic compliance reports
            for program_id in auditor.scope_cache.keys():
                report = await auditor.generate_compliance_report(program_id)
                logger.info(f"Compliance report for {program_id}: {report.get('compliance_score', 0):.2f}%")
            
            await asyncio.sleep(3600)  # Run every hour
            
    except KeyboardInterrupt:
        logger.info("Compliance Auditor shutting down")

if __name__ == "__main__":
    asyncio.run(main())