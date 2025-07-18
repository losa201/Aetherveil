"""
Advanced Rate Limiting and DDoS Protection System for Aetherveil Sentinel
Implements intelligent rate limiting, adaptive throttling, and DDoS mitigation
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import ipaddress
import re
from pathlib import Path
import threading
import weakref

from coordinator.security_manager import SecurityLevel, ThreatLevel

logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    """Rate limit types"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"

class ActionType(Enum):
    """Actions to take when rate limit is exceeded"""
    BLOCK = "block"
    THROTTLE = "throttle"
    CAPTCHA = "captcha"
    DELAY = "delay"
    LOG_ONLY = "log_only"

class ThreatType(Enum):
    """Types of threats detected"""
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    BRUTE_FORCE = "brute_force"
    DDOS_ATTACK = "ddos_attack"
    SCRAPING = "scraping"
    MALFORMED_REQUESTS = "malformed_requests"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    rule_id: str
    name: str
    description: str
    pattern: str  # URL pattern or entity pattern
    limit_type: RateLimitType
    max_requests: int
    window_seconds: int
    burst_limit: int = 0
    block_duration: int = 300  # 5 minutes default
    action: ActionType = ActionType.BLOCK
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RateLimitState:
    """Rate limit state for tracking"""
    entity_id: str
    rule_id: str
    request_count: int
    window_start: datetime
    last_request: datetime
    tokens: float = 0.0
    blocked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityEvent:
    """Security event for monitoring"""
    event_id: str
    event_type: ThreatType
    entity_id: str
    ip_address: str
    user_agent: str
    endpoint: str
    timestamp: datetime
    severity: ThreatLevel
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

@dataclass
class DDoSDetectionConfig:
    """DDoS detection configuration"""
    request_threshold: int = 1000  # Requests per minute
    unique_ip_threshold: int = 100  # Unique IPs per minute
    error_rate_threshold: float = 0.5  # Error rate threshold
    geographic_spread_threshold: int = 10  # Countries threshold
    pattern_similarity_threshold: float = 0.8  # Pattern similarity
    detection_window: int = 60  # Detection window in seconds
    mitigation_duration: int = 300  # Mitigation duration in seconds

class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket"""
        with self._lock:
            now = time.time()
            
            # Refill tokens
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def available_tokens(self) -> int:
        """Get available tokens"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            return int(tokens)

class SlidingWindowCounter:
    """Sliding window counter for rate limiting"""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.requests = deque()
        self._lock = threading.Lock()
    
    def add_request(self, timestamp: float = None):
        """Add a request to the window"""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self.requests.append(timestamp)
            self._cleanup_old_requests(timestamp)
    
    def get_count(self, timestamp: float = None) -> int:
        """Get request count in current window"""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self._cleanup_old_requests(timestamp)
            return len(self.requests)
    
    def _cleanup_old_requests(self, current_time: float):
        """Remove old requests outside the window"""
        cutoff_time = current_time - self.window_size
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""
    
    def __init__(self, base_limit: int, max_limit: int, min_limit: int):
        self.base_limit = base_limit
        self.max_limit = max_limit
        self.min_limit = min_limit
        self.current_limit = base_limit
        self.load_history = deque(maxlen=100)
        self.adjustment_factor = 1.0
        self._lock = threading.Lock()
    
    def update_load(self, cpu_usage: float, memory_usage: float, error_rate: float):
        """Update system load metrics"""
        with self._lock:
            load_score = (cpu_usage + memory_usage + error_rate * 2) / 4
            self.load_history.append(load_score)
            
            # Calculate adjustment factor
            if len(self.load_history) >= 10:
                avg_load = sum(self.load_history) / len(self.load_history)
                
                if avg_load > 0.8:  # High load
                    self.adjustment_factor = max(0.5, self.adjustment_factor * 0.9)
                elif avg_load < 0.3:  # Low load
                    self.adjustment_factor = min(2.0, self.adjustment_factor * 1.1)
                
                self.current_limit = int(self.base_limit * self.adjustment_factor)
                self.current_limit = max(self.min_limit, min(self.max_limit, self.current_limit))
    
    def get_current_limit(self) -> int:
        """Get current adaptive limit"""
        with self._lock:
            return self.current_limit

class RateLimiter:
    """Advanced rate limiter with multiple algorithms"""
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.states: Dict[str, RateLimitState] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindowCounter] = {}
        self.adaptive_limiters: Dict[str, AdaptiveRateLimiter] = {}
        self.blocked_entities: Dict[str, datetime] = {}
        self.security_events: List[SecurityEvent] = []
        self.ddos_config = DDoSDetectionConfig()
        
        # Request tracking
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ip_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.endpoint_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Statistics
        self.statistics = {
            'total_requests': 0,
            'blocked_requests': 0,
            'throttled_requests': 0,
            'active_rules': 0,
            'blocked_entities': 0,
            'security_events': 0,
            'ddos_attacks_detected': 0,
            'false_positives': 0
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._ddos_detection_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize rate limiter"""
        try:
            # Create default rules
            self._create_default_rules()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._ddos_detection_task = asyncio.create_task(self._ddos_detection_loop())
            
            logger.info("Rate limiter initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {e}")
            raise
    
    def _create_default_rules(self):
        """Create default rate limiting rules"""
        default_rules = [
            {
                'rule_id': 'global_api_limit',
                'name': 'Global API Rate Limit',
                'description': 'Global rate limit for all API endpoints',
                'pattern': '/api/*',
                'limit_type': RateLimitType.SLIDING_WINDOW,
                'max_requests': 1000,
                'window_seconds': 60,
                'burst_limit': 50,
                'block_duration': 300,
                'action': ActionType.THROTTLE,
                'priority': 10
            },
            {
                'rule_id': 'auth_endpoint_limit',
                'name': 'Authentication Endpoint Limit',
                'description': 'Strict limit for authentication endpoints',
                'pattern': '/api/auth/*',
                'limit_type': RateLimitType.FIXED_WINDOW,
                'max_requests': 10,
                'window_seconds': 60,
                'burst_limit': 5,
                'block_duration': 600,
                'action': ActionType.BLOCK,
                'priority': 100
            },
            {
                'rule_id': 'admin_endpoint_limit',
                'name': 'Admin Endpoint Limit',
                'description': 'Strict limit for admin endpoints',
                'pattern': '/api/admin/*',
                'limit_type': RateLimitType.TOKEN_BUCKET,
                'max_requests': 50,
                'window_seconds': 60,
                'burst_limit': 10,
                'block_duration': 300,
                'action': ActionType.BLOCK,
                'priority': 90
            },
            {
                'rule_id': 'search_endpoint_limit',
                'name': 'Search Endpoint Limit',
                'description': 'Limit for search endpoints to prevent scraping',
                'pattern': '/api/search/*',
                'limit_type': RateLimitType.LEAKY_BUCKET,
                'max_requests': 100,
                'window_seconds': 60,
                'burst_limit': 20,
                'block_duration': 180,
                'action': ActionType.THROTTLE,
                'priority': 50
            }
        ]
        
        for rule_data in default_rules:
            rule = RateLimitRule(**rule_data)
            self.rules[rule.rule_id] = rule
            
            # Initialize corresponding data structures
            if rule.limit_type == RateLimitType.TOKEN_BUCKET:
                self.token_buckets[rule.rule_id] = TokenBucket(
                    capacity=rule.max_requests,
                    refill_rate=rule.max_requests / rule.window_seconds
                )
            elif rule.limit_type == RateLimitType.SLIDING_WINDOW:
                self.sliding_windows[rule.rule_id] = SlidingWindowCounter(rule.window_seconds)
            elif rule.limit_type == RateLimitType.ADAPTIVE:
                self.adaptive_limiters[rule.rule_id] = AdaptiveRateLimiter(
                    base_limit=rule.max_requests,
                    max_limit=rule.max_requests * 2,
                    min_limit=rule.max_requests // 2
                )
    
    def create_rule(self, name: str, description: str, pattern: str,
                   limit_type: RateLimitType, max_requests: int, window_seconds: int,
                   burst_limit: int = 0, block_duration: int = 300,
                   action: ActionType = ActionType.BLOCK, priority: int = 0,
                   conditions: Dict[str, Any] = None) -> str:
        """Create new rate limiting rule"""
        try:
            rule_id = str(uuid.uuid4())
            
            rule = RateLimitRule(
                rule_id=rule_id,
                name=name,
                description=description,
                pattern=pattern,
                limit_type=limit_type,
                max_requests=max_requests,
                window_seconds=window_seconds,
                burst_limit=burst_limit,
                block_duration=block_duration,
                action=action,
                priority=priority,
                conditions=conditions or {}
            )
            
            self.rules[rule_id] = rule
            
            # Initialize corresponding data structures
            if limit_type == RateLimitType.TOKEN_BUCKET:
                self.token_buckets[rule_id] = TokenBucket(
                    capacity=max_requests,
                    refill_rate=max_requests / window_seconds
                )
            elif limit_type == RateLimitType.SLIDING_WINDOW:
                self.sliding_windows[rule_id] = SlidingWindowCounter(window_seconds)
            elif limit_type == RateLimitType.ADAPTIVE:
                self.adaptive_limiters[rule_id] = AdaptiveRateLimiter(
                    base_limit=max_requests,
                    max_limit=max_requests * 2,
                    min_limit=max_requests // 2
                )
            
            logger.info(f"Created rate limit rule: {name} ({rule_id})")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to create rate limit rule: {e}")
            raise
    
    async def check_rate_limit(self, entity_id: str, endpoint: str, ip_address: str,
                              user_agent: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check rate limit for request"""
        try:
            now = datetime.utcnow()
            timestamp = time.time()
            
            # Update statistics
            self.statistics['total_requests'] += 1
            
            # Track request
            self.request_history[entity_id].append(timestamp)
            self.ip_requests[ip_address].append(timestamp)
            self.endpoint_requests[endpoint].append(timestamp)
            
            # Check if entity is blocked
            if entity_id in self.blocked_entities:
                if now < self.blocked_entities[entity_id]:
                    self.statistics['blocked_requests'] += 1
                    return {
                        'allowed': False,
                        'action': ActionType.BLOCK.value,
                        'reason': 'Entity is blocked',
                        'retry_after': int((self.blocked_entities[entity_id] - now).total_seconds()),
                        'blocked_until': self.blocked_entities[entity_id].isoformat()
                    }
                else:
                    # Unblock entity
                    del self.blocked_entities[entity_id]
            
            # Find matching rules
            matching_rules = self._find_matching_rules(endpoint, entity_id, ip_address, metadata)
            
            # Check each rule
            for rule in matching_rules:
                result = await self._check_rule(rule, entity_id, endpoint, ip_address, user_agent, metadata)
                
                if not result['allowed']:
                    return result
            
            # All checks passed
            return {
                'allowed': True,
                'action': None,
                'reason': 'Request within limits',
                'applied_rules': [rule.rule_id for rule in matching_rules]
            }
            
        except Exception as e:
            logger.error(f"Failed to check rate limit: {e}")
            return {
                'allowed': False,
                'action': ActionType.BLOCK.value,
                'reason': f'Rate limit check error: {e}'
            }
    
    def _find_matching_rules(self, endpoint: str, entity_id: str, ip_address: str,
                           metadata: Dict[str, Any]) -> List[RateLimitRule]:
        """Find rules that match the request"""
        matching_rules = []
        
        for rule in self.rules.values():
            if not rule.active:
                continue
            
            # Check pattern match
            if self._pattern_matches(rule.pattern, endpoint):
                # Check conditions
                if self._conditions_match(rule.conditions, entity_id, ip_address, metadata):
                    matching_rules.append(rule)
        
        # Sort by priority (higher priority first)
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        return matching_rules
    
    def _pattern_matches(self, pattern: str, endpoint: str) -> bool:
        """Check if pattern matches endpoint"""
        try:
            # Simple wildcard matching
            if '*' in pattern:
                pattern_regex = pattern.replace('*', '.*')
                return re.match(pattern_regex, endpoint) is not None
            else:
                return pattern == endpoint
        except Exception:
            return False
    
    def _conditions_match(self, conditions: Dict[str, Any], entity_id: str,
                         ip_address: str, metadata: Dict[str, Any]) -> bool:
        """Check if conditions match"""
        try:
            if not conditions:
                return True
            
            # Check IP range condition
            if 'ip_range' in conditions:
                try:
                    ip_range = conditions['ip_range']
                    if not ipaddress.ip_address(ip_address) in ipaddress.ip_network(ip_range):
                        return False
                except Exception:
                    return False
            
            # Check entity pattern condition
            if 'entity_pattern' in conditions:
                pattern = conditions['entity_pattern']
                if not re.match(pattern, entity_id):
                    return False
            
            # Check metadata conditions
            if 'metadata' in conditions:
                for key, value in conditions['metadata'].items():
                    if metadata.get(key) != value:
                        return False
            
            return True
            
        except Exception:
            return False
    
    async def _check_rule(self, rule: RateLimitRule, entity_id: str, endpoint: str,
                         ip_address: str, user_agent: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check specific rule"""
        try:
            state_key = f"{rule.rule_id}:{entity_id}"
            now = datetime.utcnow()
            timestamp = time.time()
            
            if rule.limit_type == RateLimitType.FIXED_WINDOW:
                return self._check_fixed_window(rule, state_key, entity_id, now)
            elif rule.limit_type == RateLimitType.SLIDING_WINDOW:
                return self._check_sliding_window(rule, state_key, entity_id, timestamp)
            elif rule.limit_type == RateLimitType.TOKEN_BUCKET:
                return self._check_token_bucket(rule, state_key, entity_id)
            elif rule.limit_type == RateLimitType.LEAKY_BUCKET:
                return self._check_leaky_bucket(rule, state_key, entity_id, timestamp)
            elif rule.limit_type == RateLimitType.ADAPTIVE:
                return self._check_adaptive(rule, state_key, entity_id, timestamp)
            
            return {'allowed': True, 'reason': 'Unknown limit type'}
            
        except Exception as e:
            logger.error(f"Failed to check rule {rule.rule_id}: {e}")
            return {'allowed': False, 'action': ActionType.BLOCK.value, 'reason': f'Rule check error: {e}'}
    
    def _check_fixed_window(self, rule: RateLimitRule, state_key: str, entity_id: str,
                           now: datetime) -> Dict[str, Any]:
        """Check fixed window rate limit"""
        try:
            state = self.states.get(state_key)
            
            if not state:
                # First request in window
                state = RateLimitState(
                    entity_id=entity_id,
                    rule_id=rule.rule_id,
                    request_count=1,
                    window_start=now,
                    last_request=now
                )
                self.states[state_key] = state
                return {'allowed': True, 'reason': 'First request in window'}
            
            # Check if we're still in the same window
            window_end = state.window_start + timedelta(seconds=rule.window_seconds)
            
            if now >= window_end:
                # New window
                state.request_count = 1
                state.window_start = now
                state.last_request = now
                return {'allowed': True, 'reason': 'New window started'}
            
            # Same window
            if state.request_count >= rule.max_requests:
                # Rate limit exceeded
                return self._handle_rate_limit_exceeded(rule, entity_id, state)
            
            state.request_count += 1
            state.last_request = now
            return {'allowed': True, 'reason': 'Within rate limit'}
            
        except Exception as e:
            logger.error(f"Failed to check fixed window: {e}")
            return {'allowed': False, 'action': ActionType.BLOCK.value, 'reason': f'Fixed window check error: {e}'}
    
    def _check_sliding_window(self, rule: RateLimitRule, state_key: str, entity_id: str,
                             timestamp: float) -> Dict[str, Any]:
        """Check sliding window rate limit"""
        try:
            window = self.sliding_windows.get(rule.rule_id)
            if not window:
                window = SlidingWindowCounter(rule.window_seconds)
                self.sliding_windows[rule.rule_id] = window
            
            # Get current count
            current_count = window.get_count(timestamp)
            
            if current_count >= rule.max_requests:
                # Rate limit exceeded
                state = self.states.get(state_key)
                if not state:
                    state = RateLimitState(
                        entity_id=entity_id,
                        rule_id=rule.rule_id,
                        request_count=current_count,
                        window_start=datetime.utcnow(),
                        last_request=datetime.utcnow()
                    )
                    self.states[state_key] = state
                
                return self._handle_rate_limit_exceeded(rule, entity_id, state)
            
            # Add request to window
            window.add_request(timestamp)
            
            return {'allowed': True, 'reason': 'Within sliding window limit'}
            
        except Exception as e:
            logger.error(f"Failed to check sliding window: {e}")
            return {'allowed': False, 'action': ActionType.BLOCK.value, 'reason': f'Sliding window check error: {e}'}
    
    def _check_token_bucket(self, rule: RateLimitRule, state_key: str, entity_id: str) -> Dict[str, Any]:
        """Check token bucket rate limit"""
        try:
            bucket = self.token_buckets.get(rule.rule_id)
            if not bucket:
                bucket = TokenBucket(
                    capacity=rule.max_requests,
                    refill_rate=rule.max_requests / rule.window_seconds
                )
                self.token_buckets[rule.rule_id] = bucket
            
            if bucket.consume(1):
                return {'allowed': True, 'reason': 'Token consumed from bucket'}
            else:
                # No tokens available
                state = self.states.get(state_key)
                if not state:
                    state = RateLimitState(
                        entity_id=entity_id,
                        rule_id=rule.rule_id,
                        request_count=rule.max_requests,
                        window_start=datetime.utcnow(),
                        last_request=datetime.utcnow()
                    )
                    self.states[state_key] = state
                
                return self._handle_rate_limit_exceeded(rule, entity_id, state)
            
        except Exception as e:
            logger.error(f"Failed to check token bucket: {e}")
            return {'allowed': False, 'action': ActionType.BLOCK.value, 'reason': f'Token bucket check error: {e}'}
    
    def _check_leaky_bucket(self, rule: RateLimitRule, state_key: str, entity_id: str,
                           timestamp: float) -> Dict[str, Any]:
        """Check leaky bucket rate limit"""
        try:
            state = self.states.get(state_key)
            
            if not state:
                state = RateLimitState(
                    entity_id=entity_id,
                    rule_id=rule.rule_id,
                    request_count=1,
                    window_start=datetime.utcnow(),
                    last_request=datetime.utcnow(),
                    tokens=rule.max_requests - 1
                )
                self.states[state_key] = state
                return {'allowed': True, 'reason': 'First request in leaky bucket'}
            
            # Calculate leaked tokens
            now = datetime.utcnow()
            elapsed = (now - state.last_request).total_seconds()
            leak_rate = rule.max_requests / rule.window_seconds
            leaked_tokens = elapsed * leak_rate
            
            # Update tokens
            state.tokens = min(rule.max_requests, state.tokens + leaked_tokens)
            state.last_request = now
            
            if state.tokens >= 1:
                state.tokens -= 1
                return {'allowed': True, 'reason': 'Request allowed by leaky bucket'}
            else:
                return self._handle_rate_limit_exceeded(rule, entity_id, state)
            
        except Exception as e:
            logger.error(f"Failed to check leaky bucket: {e}")
            return {'allowed': False, 'action': ActionType.BLOCK.value, 'reason': f'Leaky bucket check error: {e}'}
    
    def _check_adaptive(self, rule: RateLimitRule, state_key: str, entity_id: str,
                       timestamp: float) -> Dict[str, Any]:
        """Check adaptive rate limit"""
        try:
            limiter = self.adaptive_limiters.get(rule.rule_id)
            if not limiter:
                limiter = AdaptiveRateLimiter(
                    base_limit=rule.max_requests,
                    max_limit=rule.max_requests * 2,
                    min_limit=rule.max_requests // 2
                )
                self.adaptive_limiters[rule.rule_id] = limiter
            
            # Use sliding window with adaptive limit
            window = self.sliding_windows.get(f"{rule.rule_id}_adaptive")
            if not window:
                window = SlidingWindowCounter(rule.window_seconds)
                self.sliding_windows[f"{rule.rule_id}_adaptive"] = window
            
            current_count = window.get_count(timestamp)
            current_limit = limiter.get_current_limit()
            
            if current_count >= current_limit:
                state = self.states.get(state_key)
                if not state:
                    state = RateLimitState(
                        entity_id=entity_id,
                        rule_id=rule.rule_id,
                        request_count=current_count,
                        window_start=datetime.utcnow(),
                        last_request=datetime.utcnow()
                    )
                    self.states[state_key] = state
                
                return self._handle_rate_limit_exceeded(rule, entity_id, state)
            
            window.add_request(timestamp)
            return {'allowed': True, 'reason': f'Within adaptive limit ({current_limit})'}
            
        except Exception as e:
            logger.error(f"Failed to check adaptive limit: {e}")
            return {'allowed': False, 'action': ActionType.BLOCK.value, 'reason': f'Adaptive check error: {e}'}
    
    def _handle_rate_limit_exceeded(self, rule: RateLimitRule, entity_id: str,
                                   state: RateLimitState) -> Dict[str, Any]:
        """Handle rate limit exceeded"""
        try:
            now = datetime.utcnow()
            
            # Update statistics
            if rule.action == ActionType.BLOCK:
                self.statistics['blocked_requests'] += 1
            elif rule.action == ActionType.THROTTLE:
                self.statistics['throttled_requests'] += 1
            
            # Log security event
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=ThreatType.RATE_LIMIT_EXCEEDED,
                entity_id=entity_id,
                ip_address='',  # Would be filled by caller
                user_agent='',  # Would be filled by caller
                endpoint='',    # Would be filled by caller
                timestamp=now,
                severity=ThreatLevel.MEDIUM,
                description=f"Rate limit exceeded for rule {rule.name}",
                metadata={
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'limit_type': rule.limit_type.value,
                    'max_requests': rule.max_requests,
                    'window_seconds': rule.window_seconds,
                    'current_count': state.request_count
                }
            )
            self.security_events.append(event)
            
            # Take action based on rule
            if rule.action == ActionType.BLOCK:
                # Block entity
                block_until = now + timedelta(seconds=rule.block_duration)
                self.blocked_entities[entity_id] = block_until
                self.statistics['blocked_entities'] += 1
                
                return {
                    'allowed': False,
                    'action': ActionType.BLOCK.value,
                    'reason': f'Rate limit exceeded: {rule.name}',
                    'rule_id': rule.rule_id,
                    'retry_after': rule.block_duration,
                    'blocked_until': block_until.isoformat()
                }
            
            elif rule.action == ActionType.THROTTLE:
                # Calculate throttle delay
                throttle_delay = min(60, rule.window_seconds // rule.max_requests)
                
                return {
                    'allowed': False,
                    'action': ActionType.THROTTLE.value,
                    'reason': f'Rate limit exceeded: {rule.name}',
                    'rule_id': rule.rule_id,
                    'throttle_delay': throttle_delay,
                    'retry_after': throttle_delay
                }
            
            elif rule.action == ActionType.DELAY:
                # Calculate delay
                delay = min(30, rule.window_seconds // (rule.max_requests * 2))
                
                return {
                    'allowed': True,
                    'action': ActionType.DELAY.value,
                    'reason': f'Rate limit exceeded: {rule.name}',
                    'rule_id': rule.rule_id,
                    'delay': delay
                }
            
            elif rule.action == ActionType.LOG_ONLY:
                return {
                    'allowed': True,
                    'action': ActionType.LOG_ONLY.value,
                    'reason': f'Rate limit exceeded (log only): {rule.name}',
                    'rule_id': rule.rule_id
                }
            
            return {
                'allowed': False,
                'action': ActionType.BLOCK.value,
                'reason': f'Rate limit exceeded: {rule.name}',
                'rule_id': rule.rule_id
            }
            
        except Exception as e:
            logger.error(f"Failed to handle rate limit exceeded: {e}")
            return {
                'allowed': False,
                'action': ActionType.BLOCK.value,
                'reason': f'Rate limit handler error: {e}'
            }
    
    async def detect_ddos(self, time_window: int = 60) -> List[SecurityEvent]:
        """Detect DDoS attacks"""
        try:
            now = time.time()
            cutoff_time = now - time_window
            
            # Count requests in window
            total_requests = 0
            unique_ips = set()
            error_count = 0
            
            for requests in self.ip_requests.values():
                recent_requests = [r for r in requests if r > cutoff_time]
                total_requests += len(recent_requests)
                if recent_requests:
                    unique_ips.add(len(recent_requests))  # Simplified
            
            # Check thresholds
            events = []
            
            if total_requests > self.ddos_config.request_threshold:
                event = SecurityEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=ThreatType.DDOS_ATTACK,
                    entity_id='system',
                    ip_address='multiple',
                    user_agent='multiple',
                    endpoint='multiple',
                    timestamp=datetime.utcnow(),
                    severity=ThreatLevel.CRITICAL,
                    description=f"DDoS attack detected: {total_requests} requests in {time_window}s",
                    metadata={
                        'total_requests': total_requests,
                        'unique_ips': len(unique_ips),
                        'threshold': self.ddos_config.request_threshold,
                        'time_window': time_window
                    }
                )
                events.append(event)
                self.statistics['ddos_attacks_detected'] += 1
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to detect DDoS: {e}")
            return []
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await self._cleanup_expired_data()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_data(self):
        """Clean up expired data"""
        try:
            now = datetime.utcnow()
            cutoff_time = time.time() - 3600  # Keep data for 1 hour
            
            # Clean up blocked entities
            expired_blocks = [
                entity_id for entity_id, blocked_until in self.blocked_entities.items()
                if now > blocked_until
            ]
            
            for entity_id in expired_blocks:
                del self.blocked_entities[entity_id]
                self.statistics['blocked_entities'] -= 1
            
            # Clean up old states
            expired_states = [
                state_key for state_key, state in self.states.items()
                if (now - state.last_request).total_seconds() > 3600
            ]
            
            for state_key in expired_states:
                del self.states[state_key]
            
            # Clean up old request history
            for entity_id in list(self.request_history.keys()):
                requests = self.request_history[entity_id]
                # Remove old requests
                while requests and requests[0] < cutoff_time:
                    requests.popleft()
                
                # Remove empty deques
                if not requests:
                    del self.request_history[entity_id]
            
            # Clean up IP request history
            for ip in list(self.ip_requests.keys()):
                requests = self.ip_requests[ip]
                while requests and requests[0] < cutoff_time:
                    requests.popleft()
                
                if not requests:
                    del self.ip_requests[ip]
            
            # Clean up endpoint request history
            for endpoint in list(self.endpoint_requests.keys()):
                requests = self.endpoint_requests[endpoint]
                while requests and requests[0] < cutoff_time:
                    requests.popleft()
                
                if not requests:
                    del self.endpoint_requests[endpoint]
            
            # Clean up old security events
            cutoff_datetime = now - timedelta(days=7)
            self.security_events = [
                event for event in self.security_events
                if event.timestamp > cutoff_datetime
            ]
            
            logger.debug(f"Cleaned up {len(expired_blocks)} expired blocks and {len(expired_states)} expired states")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Update adaptive limiters with system metrics
                # This would typically integrate with system monitoring
                cpu_usage = 0.5  # Placeholder
                memory_usage = 0.5  # Placeholder
                error_rate = 0.1  # Placeholder
                
                for limiter in self.adaptive_limiters.values():
                    limiter.update_load(cpu_usage, memory_usage, error_rate)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _ddos_detection_loop(self):
        """Background DDoS detection loop"""
        while True:
            try:
                events = await self.detect_ddos()
                
                for event in events:
                    logger.warning(f"DDoS attack detected: {event.description}")
                    # Here you would trigger mitigation measures
                
                await asyncio.sleep(self.ddos_config.detection_window)
                
            except Exception as e:
                logger.error(f"Error in DDoS detection loop: {e}")
                await asyncio.sleep(60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            **self.statistics,
            'active_rules': len([r for r in self.rules.values() if r.active]),
            'total_rules': len(self.rules),
            'active_states': len(self.states),
            'request_history_size': sum(len(requests) for requests in self.request_history.values()),
            'ip_tracking_size': len(self.ip_requests),
            'endpoint_tracking_size': len(self.endpoint_requests),
            'security_events': len(self.security_events)
        }
    
    def get_security_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events"""
        return sorted(self.security_events, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def get_blocked_entities(self) -> List[Dict[str, Any]]:
        """Get currently blocked entities"""
        return [
            {
                'entity_id': entity_id,
                'blocked_until': blocked_until.isoformat(),
                'remaining_seconds': int((blocked_until - datetime.utcnow()).total_seconds())
            }
            for entity_id, blocked_until in self.blocked_entities.items()
        ]

# Global rate limiter instance
rate_limiter = RateLimiter()