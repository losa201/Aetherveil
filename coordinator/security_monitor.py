"""
Comprehensive Security Monitoring and Alerting System for Aetherveil Sentinel
Implements real-time security monitoring, threat detection, and automated alerting
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import statistics
from collections import defaultdict, deque
import threading
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import yaml
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from coordinator.security_manager import SecurityLevel, ThreatLevel
from coordinator.blockchain_logger import BlockchainLogger, LogLevel, EventType
from coordinator.rate_limiter import RateLimiter, SecurityEvent as RLSecurityEvent
from coordinator.rbac_manager import RBACManager

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Alert types"""
    SECURITY_BREACH = "security_breach"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    DDOS_ATTACK = "ddos_attack"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SYSTEM_FAILURE = "system_failure"
    DATA_LEAK = "data_leak"
    CERTIFICATE_EXPIRY = "certificate_expiry"
    CONFIGURATION_CHANGE = "configuration_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COMPLIANCE_VIOLATION = "compliance_violation"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"

class MonitoringMode(Enum):
    """Monitoring modes"""
    PASSIVE = "passive"
    ACTIVE = "active"
    HUNTING = "hunting"
    LEARNING = "learning"

@dataclass
class SecurityAlert:
    """Security alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    affected_entities: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)
    false_positive_score: float = 0.0
    confidence_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalated_to: Optional[str] = None
    escalated_at: Optional[datetime] = None

@dataclass
class ThreatIndicator:
    """Threat indicator"""
    indicator_id: str
    indicator_type: str
    value: str
    severity: ThreatLevel
    confidence: float
    source: str
    description: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringRule:
    """Monitoring rule for detection"""
    rule_id: str
    name: str
    description: str
    rule_type: str
    pattern: str
    threshold: float
    time_window: int
    severity: AlertSeverity
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertingChannel:
    """Alerting channel configuration"""
    channel_id: str
    name: str
    channel_type: str
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    alert_types: List[AlertType] = field(default_factory=list)

class AnomalyDetector:
    """Machine learning-based anomaly detector"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_history = deque(maxlen=10000)
        self.training_data = []
        
    def add_sample(self, features: List[float]):
        """Add sample for training"""
        self.feature_history.append(features)
        
        # Retrain periodically
        if len(self.feature_history) >= 100 and len(self.feature_history) % 100 == 0:
            self.train()
    
    def train(self):
        """Train the anomaly detection model"""
        try:
            if len(self.feature_history) < 50:
                return
            
            # Prepare training data
            X = np.array(list(self.feature_history))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled)
            self.is_trained = True
            
            logger.info(f"Anomaly detector trained with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {e}")
    
    def predict_anomaly(self, features: List[float]) -> Tuple[bool, float]:
        """Predict if features represent an anomaly"""
        try:
            if not self.is_trained:
                return False, 0.0
            
            # Scale features
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            anomaly_score = self.model.score_samples(X_scaled)[0]
            
            # Convert to probability
            probability = 1 / (1 + np.exp(anomaly_score))
            
            return prediction == -1, probability
            
        except Exception as e:
            logger.error(f"Failed to predict anomaly: {e}")
            return False, 0.0

class SecurityMonitor:
    """Comprehensive security monitoring system"""
    
    def __init__(self, config_path: str = "/app/config/security_monitor.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Components
        self.blockchain_logger: Optional[BlockchainLogger] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.rbac_manager: Optional[RBACManager] = None
        
        # Monitoring data
        self.alerts: Dict[str, SecurityAlert] = {}
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        self.alerting_channels: Dict[str, AlertingChannel] = {}
        
        # Detection engines
        self.anomaly_detector = AnomalyDetector()
        self.pattern_matchers: Dict[str, callable] = {}
        
        # Statistics and metrics
        self.metrics = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'false_positives': 0,
            'mean_response_time': 0.0,
            'threat_indicators_count': 0,
            'anomalies_detected': 0,
            'monitoring_rules_count': 0
        }
        
        # Real-time monitoring
        self.monitoring_mode = MonitoringMode.ACTIVE
        self.event_queue = asyncio.Queue()
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        self.suppression_rules: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # Thread safety
        self._lock = threading.Lock()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'monitoring': {
                'mode': 'active',
                'anomaly_detection': {
                    'enabled': True,
                    'contamination': 0.1,
                    'features': [
                        'request_rate',
                        'error_rate',
                        'response_time',
                        'authentication_failures',
                        'authorization_failures'
                    ]
                },
                'threat_intelligence': {
                    'enabled': True,
                    'sources': [],
                    'update_interval': 3600
                }
            },
            'alerting': {
                'enabled': True,
                'channels': {
                    'email': {
                        'enabled': True,
                        'smtp_server': 'localhost',
                        'smtp_port': 587,
                        'username': '',
                        'password': '',
                        'from_address': 'alerts@aetherveil-sentinel.com',
                        'to_addresses': []
                    },
                    'webhook': {
                        'enabled': True,
                        'url': '',
                        'headers': {},
                        'timeout': 30
                    },
                    'syslog': {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 514,
                        'facility': 'local0'
                    }
                },
                'escalation': {
                    'enabled': True,
                    'levels': [
                        {'severity': 'critical', 'timeout': 300},
                        {'severity': 'high', 'timeout': 600},
                        {'severity': 'medium', 'timeout': 1800}
                    ]
                }
            },
            'rules': {
                'authentication_failures': {
                    'threshold': 5,
                    'time_window': 300,
                    'severity': 'medium'
                },
                'rate_limit_exceeded': {
                    'threshold': 10,
                    'time_window': 60,
                    'severity': 'high'
                },
                'ddos_attack': {
                    'threshold': 1000,
                    'time_window': 60,
                    'severity': 'critical'
                }
            }
        }
    
    async def initialize(self, blockchain_logger: BlockchainLogger = None,
                        rate_limiter: RateLimiter = None,
                        rbac_manager: RBACManager = None):
        """Initialize security monitor"""
        try:
            # Set component references
            self.blockchain_logger = blockchain_logger
            self.rate_limiter = rate_limiter
            self.rbac_manager = rbac_manager
            
            # Initialize monitoring rules
            self._initialize_monitoring_rules()
            
            # Initialize alerting channels
            self._initialize_alerting_channels()
            
            # Initialize pattern matchers
            self._initialize_pattern_matchers()
            
            # Load threat indicators
            await self._load_threat_indicators()
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._event_processing_loop()),
                asyncio.create_task(self._anomaly_detection_loop()),
                asyncio.create_task(self._threat_intelligence_loop()),
                asyncio.create_task(self._alert_escalation_loop()),
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            logger.info("Security monitor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize security monitor: {e}")
            raise
    
    def _initialize_monitoring_rules(self):
        """Initialize monitoring rules"""
        try:
            rules_config = self.config.get('rules', {})
            
            for rule_name, rule_config in rules_config.items():
                rule = MonitoringRule(
                    rule_id=str(uuid.uuid4()),
                    name=rule_name,
                    description=rule_config.get('description', ''),
                    rule_type=rule_config.get('type', 'threshold'),
                    pattern=rule_config.get('pattern', ''),
                    threshold=rule_config.get('threshold', 0),
                    time_window=rule_config.get('time_window', 300),
                    severity=AlertSeverity(rule_config.get('severity', 'medium')),
                    conditions=rule_config.get('conditions', {}),
                    actions=rule_config.get('actions', [])
                )
                
                self.monitoring_rules[rule.rule_id] = rule
            
            logger.info(f"Initialized {len(self.monitoring_rules)} monitoring rules")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring rules: {e}")
    
    def _initialize_alerting_channels(self):
        """Initialize alerting channels"""
        try:
            channels_config = self.config.get('alerting', {}).get('channels', {})
            
            for channel_type, channel_config in channels_config.items():
                if channel_config.get('enabled', False):
                    channel = AlertingChannel(
                        channel_id=str(uuid.uuid4()),
                        name=channel_type,
                        channel_type=channel_type,
                        config=channel_config
                    )
                    
                    self.alerting_channels[channel.channel_id] = channel
            
            logger.info(f"Initialized {len(self.alerting_channels)} alerting channels")
            
        except Exception as e:
            logger.error(f"Failed to initialize alerting channels: {e}")
    
    def _initialize_pattern_matchers(self):
        """Initialize pattern matching functions"""
        try:
            # Authentication failure pattern
            self.pattern_matchers['auth_failure'] = self._match_auth_failure
            
            # Rate limit pattern
            self.pattern_matchers['rate_limit'] = self._match_rate_limit
            
            # DDoS pattern
            self.pattern_matchers['ddos'] = self._match_ddos
            
            # Anomalous behavior pattern
            self.pattern_matchers['anomaly'] = self._match_anomaly
            
            # Data leak pattern
            self.pattern_matchers['data_leak'] = self._match_data_leak
            
            logger.info(f"Initialized {len(self.pattern_matchers)} pattern matchers")
            
        except Exception as e:
            logger.error(f"Failed to initialize pattern matchers: {e}")
    
    async def _load_threat_indicators(self):
        """Load threat indicators from various sources"""
        try:
            # Load from local file
            indicators_file = Path("/app/data/threat_indicators.json")
            if indicators_file.exists():
                with open(indicators_file, 'r') as f:
                    indicators_data = json.load(f)
                
                for indicator_data in indicators_data:
                    indicator = ThreatIndicator(
                        indicator_id=indicator_data['indicator_id'],
                        indicator_type=indicator_data['indicator_type'],
                        value=indicator_data['value'],
                        severity=ThreatLevel(indicator_data['severity']),
                        confidence=indicator_data['confidence'],
                        source=indicator_data['source'],
                        description=indicator_data['description'],
                        created_at=datetime.fromisoformat(indicator_data['created_at']),
                        expires_at=datetime.fromisoformat(indicator_data['expires_at']) if indicator_data.get('expires_at') else None,
                        tags=indicator_data.get('tags', []),
                        metadata=indicator_data.get('metadata', {})
                    )
                    
                    self.threat_indicators[indicator.indicator_id] = indicator
            
            logger.info(f"Loaded {len(self.threat_indicators)} threat indicators")
            
        except Exception as e:
            logger.error(f"Failed to load threat indicators: {e}")
    
    async def process_event(self, event: Dict[str, Any]):
        """Process security event"""
        try:
            await self.event_queue.put(event)
        except Exception as e:
            logger.error(f"Failed to queue event: {e}")
    
    async def _event_processing_loop(self):
        """Main event processing loop"""
        while not self._shutdown:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event
                await self._process_security_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_security_event(self, event: Dict[str, Any]):
        """Process individual security event"""
        try:
            event_type = event.get('type', 'unknown')
            event_data = event.get('data', {})
            
            # Apply monitoring rules
            for rule in self.monitoring_rules.values():
                if rule.enabled and await self._evaluate_rule(rule, event):
                    await self._trigger_alert(rule, event)
            
            # Pattern matching
            for pattern_name, matcher in self.pattern_matchers.items():
                if await matcher(event):
                    await self._handle_pattern_match(pattern_name, event)
            
            # Threat indicator matching
            await self._check_threat_indicators(event)
            
            # Extract features for anomaly detection
            features = self._extract_features(event)
            if features:
                self.anomaly_detector.add_sample(features)
            
        except Exception as e:
            logger.error(f"Failed to process security event: {e}")
    
    async def _evaluate_rule(self, rule: MonitoringRule, event: Dict[str, Any]) -> bool:
        """Evaluate monitoring rule against event"""
        try:
            # Simple threshold-based evaluation
            if rule.rule_type == 'threshold':
                value = event.get('data', {}).get(rule.pattern, 0)
                return value >= rule.threshold
            
            # Pattern-based evaluation
            elif rule.rule_type == 'pattern':
                return rule.pattern in str(event)
            
            # Condition-based evaluation
            elif rule.rule_type == 'condition':
                return self._evaluate_conditions(rule.conditions, event)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.rule_id}: {e}")
            return False
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], event: Dict[str, Any]) -> bool:
        """Evaluate conditions against event"""
        try:
            for condition_key, condition_value in conditions.items():
                event_value = event.get('data', {}).get(condition_key)
                
                if isinstance(condition_value, dict):
                    operator = condition_value.get('operator', 'equals')
                    value = condition_value.get('value')
                    
                    if operator == 'equals' and event_value != value:
                        return False
                    elif operator == 'greater_than' and event_value <= value:
                        return False
                    elif operator == 'less_than' and event_value >= value:
                        return False
                    elif operator == 'contains' and value not in str(event_value):
                        return False
                else:
                    if event_value != condition_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate conditions: {e}")
            return False
    
    async def _trigger_alert(self, rule: MonitoringRule, event: Dict[str, Any]):
        """Trigger alert based on rule"""
        try:
            alert = SecurityAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.ANOMALOUS_BEHAVIOR,  # Default, would be determined by rule
                severity=rule.severity,
                title=f"Monitoring rule triggered: {rule.name}",
                description=f"Rule {rule.name} has been triggered",
                source=f"monitoring_rule_{rule.rule_id}",
                timestamp=datetime.utcnow(),
                indicators=event.get('data', {}),
                metadata={
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'event': event
                }
            )
            
            await self.create_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    async def _handle_pattern_match(self, pattern_name: str, event: Dict[str, Any]):
        """Handle pattern match"""
        try:
            # Create appropriate alert based on pattern
            alert_type_map = {
                'auth_failure': AlertType.AUTHENTICATION_FAILURE,
                'rate_limit': AlertType.RATE_LIMIT_EXCEEDED,
                'ddos': AlertType.DDOS_ATTACK,
                'anomaly': AlertType.ANOMALOUS_BEHAVIOR,
                'data_leak': AlertType.DATA_LEAK
            }
            
            alert_type = alert_type_map.get(pattern_name, AlertType.ANOMALOUS_BEHAVIOR)
            
            alert = SecurityAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=alert_type,
                severity=AlertSeverity.HIGH,
                title=f"Pattern detected: {pattern_name}",
                description=f"Security pattern {pattern_name} detected",
                source=f"pattern_matcher_{pattern_name}",
                timestamp=datetime.utcnow(),
                indicators=event.get('data', {}),
                metadata={
                    'pattern_name': pattern_name,
                    'event': event
                }
            )
            
            await self.create_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to handle pattern match: {e}")
    
    async def _check_threat_indicators(self, event: Dict[str, Any]):
        """Check event against threat indicators"""
        try:
            event_data = event.get('data', {})
            
            for indicator in self.threat_indicators.values():
                # Check if indicator is expired
                if indicator.expires_at and datetime.utcnow() > indicator.expires_at:
                    continue
                
                # Check for indicator match
                if self._matches_indicator(indicator, event_data):
                    await self._handle_threat_indicator_match(indicator, event)
            
        except Exception as e:
            logger.error(f"Failed to check threat indicators: {e}")
    
    def _matches_indicator(self, indicator: ThreatIndicator, event_data: Dict[str, Any]) -> bool:
        """Check if event matches threat indicator"""
        try:
            if indicator.indicator_type == 'ip':
                return event_data.get('ip_address') == indicator.value
            elif indicator.indicator_type == 'domain':
                return indicator.value in event_data.get('domain', '')
            elif indicator.indicator_type == 'hash':
                return event_data.get('file_hash') == indicator.value
            elif indicator.indicator_type == 'email':
                return event_data.get('email') == indicator.value
            elif indicator.indicator_type == 'user_agent':
                return indicator.value in event_data.get('user_agent', '')
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to match indicator: {e}")
            return False
    
    async def _handle_threat_indicator_match(self, indicator: ThreatIndicator, event: Dict[str, Any]):
        """Handle threat indicator match"""
        try:
            severity_map = {
                ThreatLevel.LOW: AlertSeverity.LOW,
                ThreatLevel.MEDIUM: AlertSeverity.MEDIUM,
                ThreatLevel.HIGH: AlertSeverity.HIGH,
                ThreatLevel.CRITICAL: AlertSeverity.CRITICAL,
                ThreatLevel.EMERGENCY: AlertSeverity.EMERGENCY
            }
            
            alert = SecurityAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.SECURITY_BREACH,
                severity=severity_map.get(indicator.severity, AlertSeverity.MEDIUM),
                title=f"Threat indicator match: {indicator.indicator_type}",
                description=f"Known threat indicator {indicator.value} detected",
                source=f"threat_intelligence_{indicator.source}",
                timestamp=datetime.utcnow(),
                confidence_score=indicator.confidence,
                indicators=event.get('data', {}),
                metadata={
                    'indicator_id': indicator.indicator_id,
                    'indicator_type': indicator.indicator_type,
                    'indicator_value': indicator.value,
                    'indicator_source': indicator.source,
                    'event': event
                }
            )
            
            await self.create_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to handle threat indicator match: {e}")
    
    def _extract_features(self, event: Dict[str, Any]) -> Optional[List[float]]:
        """Extract features for anomaly detection"""
        try:
            features = []
            event_data = event.get('data', {})
            
            # Extract numerical features
            features.append(event_data.get('request_rate', 0))
            features.append(event_data.get('error_rate', 0))
            features.append(event_data.get('response_time', 0))
            features.append(event_data.get('auth_failures', 0))
            features.append(event_data.get('authz_failures', 0))
            features.append(event_data.get('unique_ips', 0))
            features.append(event_data.get('payload_size', 0))
            
            # Convert timestamp to hour of day
            timestamp = event.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            features.append(timestamp.hour)
            
            return features if len(features) > 0 else None
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return None
    
    async def _anomaly_detection_loop(self):
        """Anomaly detection loop"""
        while not self._shutdown:
            try:
                # Collect recent events for anomaly detection
                await self._run_anomaly_detection()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_anomaly_detection(self):
        """Run anomaly detection on recent events"""
        try:
            if not self.config.get('monitoring', {}).get('anomaly_detection', {}).get('enabled', True):
                return
            
            # Get recent events (placeholder - would integrate with actual event source)
            recent_events = []  # This would be populated from actual event sources
            
            for event in recent_events:
                features = self._extract_features(event)
                if features:
                    is_anomaly, score = self.anomaly_detector.predict_anomaly(features)
                    
                    if is_anomaly and score > 0.7:  # High confidence threshold
                        await self._handle_anomaly_detection(event, score)
            
        except Exception as e:
            logger.error(f"Failed to run anomaly detection: {e}")
    
    async def _handle_anomaly_detection(self, event: Dict[str, Any], score: float):
        """Handle detected anomaly"""
        try:
            alert = SecurityAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.ANOMALOUS_BEHAVIOR,
                severity=AlertSeverity.HIGH if score > 0.9 else AlertSeverity.MEDIUM,
                title="Anomalous behavior detected",
                description=f"ML-based anomaly detection triggered with score {score:.2f}",
                source="anomaly_detector",
                timestamp=datetime.utcnow(),
                confidence_score=score,
                indicators=event.get('data', {}),
                metadata={
                    'anomaly_score': score,
                    'event': event
                }
            )
            
            await self.create_alert(alert)
            self.metrics['anomalies_detected'] += 1
            
        except Exception as e:
            logger.error(f"Failed to handle anomaly detection: {e}")
    
    async def _threat_intelligence_loop(self):
        """Threat intelligence update loop"""
        while not self._shutdown:
            try:
                if self.config.get('monitoring', {}).get('threat_intelligence', {}).get('enabled', True):
                    await self._update_threat_intelligence()
                
                # Update interval from config
                interval = self.config.get('monitoring', {}).get('threat_intelligence', {}).get('update_interval', 3600)
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in threat intelligence loop: {e}")
                await asyncio.sleep(3600)
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence from external sources"""
        try:
            # Placeholder for threat intelligence updates
            # In a real implementation, this would fetch from threat intelligence sources
            logger.debug("Updating threat intelligence...")
            
        except Exception as e:
            logger.error(f"Failed to update threat intelligence: {e}")
    
    async def _alert_escalation_loop(self):
        """Alert escalation loop"""
        while not self._shutdown:
            try:
                await self._check_alert_escalation()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in alert escalation loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_alert_escalation(self):
        """Check for alerts that need escalation"""
        try:
            escalation_config = self.config.get('alerting', {}).get('escalation', {})
            if not escalation_config.get('enabled', True):
                return
            
            now = datetime.utcnow()
            
            for alert in self.alerts.values():
                if alert.status != AlertStatus.ACTIVE:
                    continue
                
                # Check escalation rules
                for level in escalation_config.get('levels', []):
                    if alert.severity.value == level.get('severity'):
                        timeout = level.get('timeout', 300)
                        
                        if (now - alert.timestamp).total_seconds() > timeout:
                            await self._escalate_alert(alert)
                            break
            
        except Exception as e:
            logger.error(f"Failed to check alert escalation: {e}")
    
    async def _escalate_alert(self, alert: SecurityAlert):
        """Escalate alert"""
        try:
            alert.status = AlertStatus.ESCALATED
            alert.escalated_at = datetime.utcnow()
            
            # Send escalation notification
            await self._send_alert_notification(alert, escalation=True)
            
            logger.warning(f"Alert escalated: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to escalate alert: {e}")
    
    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while not self._shutdown:
            try:
                await self._collect_metrics()
                await asyncio.sleep(300)  # Collect every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(300)
    
    async def _collect_metrics(self):
        """Collect monitoring metrics"""
        try:
            # Update metrics
            self.metrics['total_alerts'] = len(self.alerts)
            self.metrics['active_alerts'] = len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE])
            self.metrics['resolved_alerts'] = len([a for a in self.alerts.values() if a.status == AlertStatus.RESOLVED])
            self.metrics['threat_indicators_count'] = len(self.threat_indicators)
            self.metrics['monitoring_rules_count'] = len(self.monitoring_rules)
            
            # Calculate mean response time
            resolved_alerts = [a for a in self.alerts.values() if a.status == AlertStatus.RESOLVED and a.resolved_at]
            if resolved_alerts:
                response_times = [(a.resolved_at - a.timestamp).total_seconds() for a in resolved_alerts]
                self.metrics['mean_response_time'] = statistics.mean(response_times)
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _health_check_loop(self):
        """Health check loop"""
        while not self._shutdown:
            try:
                await self._perform_health_check()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(300)
    
    async def _perform_health_check(self):
        """Perform health check"""
        try:
            # Check component health
            health_issues = []
            
            # Check queue size
            if self.event_queue.qsize() > 1000:
                health_issues.append("Event queue size is high")
            
            # Check active alerts
            active_alerts = len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE])
            if active_alerts > 100:
                health_issues.append("Too many active alerts")
            
            # Check anomaly detector
            if not self.anomaly_detector.is_trained:
                health_issues.append("Anomaly detector not trained")
            
            # Create health alert if issues found
            if health_issues:
                alert = SecurityAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=AlertType.SYSTEM_FAILURE,
                    severity=AlertSeverity.MEDIUM,
                    title="Security monitor health issues",
                    description=f"Health check found issues: {', '.join(health_issues)}",
                    source="health_check",
                    timestamp=datetime.utcnow(),
                    metadata={'health_issues': health_issues}
                )
                
                await self.create_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to perform health check: {e}")
    
    # Pattern matching functions
    async def _match_auth_failure(self, event: Dict[str, Any]) -> bool:
        """Match authentication failure pattern"""
        try:
            event_data = event.get('data', {})
            return (event_data.get('event_type') == 'authentication_failure' or
                    event_data.get('auth_result') == 'failed')
        except Exception:
            return False
    
    async def _match_rate_limit(self, event: Dict[str, Any]) -> bool:
        """Match rate limit pattern"""
        try:
            event_data = event.get('data', {})
            return event_data.get('event_type') == 'rate_limit_exceeded'
        except Exception:
            return False
    
    async def _match_ddos(self, event: Dict[str, Any]) -> bool:
        """Match DDoS pattern"""
        try:
            event_data = event.get('data', {})
            return (event_data.get('request_rate', 0) > 1000 or
                    event_data.get('event_type') == 'ddos_attack')
        except Exception:
            return False
    
    async def _match_anomaly(self, event: Dict[str, Any]) -> bool:
        """Match anomaly pattern"""
        try:
            features = self._extract_features(event)
            if features and self.anomaly_detector.is_trained:
                is_anomaly, score = self.anomaly_detector.predict_anomaly(features)
                return is_anomaly and score > 0.8
            return False
        except Exception:
            return False
    
    async def _match_data_leak(self, event: Dict[str, Any]) -> bool:
        """Match data leak pattern"""
        try:
            event_data = event.get('data', {})
            return (event_data.get('event_type') == 'data_access' and
                    event_data.get('data_size', 0) > 10000000)  # 10MB threshold
        except Exception:
            return False
    
    async def create_alert(self, alert: SecurityAlert):
        """Create new alert"""
        try:
            with self._lock:
                self.alerts[alert.alert_id] = alert
                self.metrics['total_alerts'] += 1
            
            # Send alert notification
            await self._send_alert_notification(alert)
            
            # Call alert handlers
            handlers = self.alert_handlers.get(alert.alert_type, [])
            for handler in handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            # Log to blockchain
            if self.blockchain_logger:
                self.blockchain_logger.log_security_event(
                    event_type=EventType.SECURITY_INCIDENT,
                    source="security_monitor",
                    message=f"Alert created: {alert.title}",
                    threat_level=ThreatLevel.HIGH if alert.severity == AlertSeverity.HIGH else ThreatLevel.MEDIUM,
                    data=alert.metadata
                )
            
            logger.info(f"Alert created: {alert.alert_id} - {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def _send_alert_notification(self, alert: SecurityAlert, escalation: bool = False):
        """Send alert notification through configured channels"""
        try:
            for channel in self.alerting_channels.values():
                if not channel.enabled:
                    continue
                
                # Check severity filter
                if channel.severity_filter and alert.severity not in channel.severity_filter:
                    continue
                
                # Check alert type filter
                if channel.alert_types and alert.alert_type not in channel.alert_types:
                    continue
                
                # Send notification
                await self._send_notification(channel, alert, escalation)
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    async def _send_notification(self, channel: AlertingChannel, alert: SecurityAlert, escalation: bool = False):
        """Send notification through specific channel"""
        try:
            if channel.channel_type == 'email':
                await self._send_email_notification(channel, alert, escalation)
            elif channel.channel_type == 'webhook':
                await self._send_webhook_notification(channel, alert, escalation)
            elif channel.channel_type == 'syslog':
                await self._send_syslog_notification(channel, alert, escalation)
            
        except Exception as e:
            logger.error(f"Failed to send notification through {channel.channel_type}: {e}")
    
    async def _send_email_notification(self, channel: AlertingChannel, alert: SecurityAlert, escalation: bool = False):
        """Send email notification"""
        try:
            config = channel.config
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = config.get('from_address', 'alerts@aetherveil-sentinel.com')
            msg['To'] = ', '.join(config.get('to_addresses', []))
            msg['Subject'] = f"{'[ESCALATED] ' if escalation else ''}Security Alert: {alert.title}"
            
            # Create email body
            body = f"""
Security Alert Details:

Alert ID: {alert.alert_id}
Type: {alert.alert_type.value}
Severity: {alert.severity.value}
Title: {alert.title}
Description: {alert.description}
Source: {alert.source}
Timestamp: {alert.timestamp}
Status: {alert.status.value}

Affected Entities: {', '.join(alert.affected_entities)}

Indicators:
{json.dumps(alert.indicators, indent=2)}

Remediation Steps:
{chr(10).join(f"- {step}" for step in alert.remediation_steps)}

Metadata:
{json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config.get('smtp_server', 'localhost'), config.get('smtp_port', 587))
            if config.get('username') and config.get('password'):
                server.starttls()
                server.login(config.get('username'), config.get('password'))
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_webhook_notification(self, channel: AlertingChannel, alert: SecurityAlert, escalation: bool = False):
        """Send webhook notification"""
        try:
            config = channel.config
            
            # Prepare webhook payload
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'status': alert.status.value,
                'escalated': escalation,
                'affected_entities': alert.affected_entities,
                'indicators': alert.indicators,
                'remediation_steps': alert.remediation_steps,
                'metadata': alert.metadata
            }
            
            # Send webhook
            url = config.get('url')
            headers = config.get('headers', {})
            timeout = config.get('timeout', 30)
            
            if url:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    async def _send_syslog_notification(self, channel: AlertingChannel, alert: SecurityAlert, escalation: bool = False):
        """Send syslog notification"""
        try:
            # Placeholder for syslog implementation
            logger.info(f"Syslog notification: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send syslog notification: {e}")
    
    def register_alert_handler(self, alert_type: AlertType, handler: Callable):
        """Register alert handler"""
        self.alert_handlers[alert_type].append(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.metrics,
            'monitoring_mode': self.monitoring_mode.value,
            'event_queue_size': self.event_queue.qsize(),
            'alerting_channels': len(self.alerting_channels),
            'anomaly_detector_trained': self.anomaly_detector.is_trained
        }
    
    def get_alerts(self, status: AlertStatus = None, severity: AlertSeverity = None,
                  limit: int = 100) -> List[SecurityAlert]:
        """Get alerts with optional filtering"""
        alerts = list(self.alerts.values())
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        return alerts[:limit]
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge alert"""
        try:
            alert = self.alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str):
        """Resolve alert"""
        try:
            alert = self.alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_by = resolved_by
                alert.resolved_at = datetime.utcnow()
                
                logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
    
    async def shutdown(self):
        """Shutdown security monitor"""
        try:
            self._shutdown = True
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            logger.info("Security monitor shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown security monitor: {e}")

# Global security monitor instance
security_monitor = SecurityMonitor()