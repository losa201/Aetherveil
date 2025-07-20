"""
Adaptive OPSEC Intelligence System
Learns from detection patterns and adapts operational security in real-time
"""

import asyncio
import logging
import random
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

from ..core.events import EventSystem, EventType, EventEmitter
from ..utils.network import NetworkUtils

logger = logging.getLogger(__name__)

@dataclass
class OPSECThreat:
    """Detected OPSEC threat or suspicious activity"""
    
    threat_id: str
    threat_type: str  # rate_limiting, behavioral_detection, fingerprinting, etc.
    severity: float  # 0.0 to 1.0
    confidence: float  # How confident we are this is a real threat
    indicators: List[str]  # Specific indicators that triggered detection
    source_ip: Optional[str]
    target_endpoint: Optional[str]
    timestamp: datetime
    countermeasures_applied: List[str]
    effectiveness: Optional[float] = None  # How effective our response was

@dataclass
class OPSECProfile:
    """Dynamic OPSEC profile for a target or environment"""
    
    target_id: str
    detection_sensitivity: float  # How sensitive their detection is
    common_patterns: List[str]  # Patterns they commonly detect
    safe_request_rate: float  # Requests per minute that seem safe
    safe_user_agents: List[str]  # User agents that don't trigger detection
    time_windows: List[Dict[str, Any]]  # Safe time windows for operations
    fingerprinting_methods: List[str]  # Known fingerprinting techniques used
    last_updated: datetime
    confidence_level: float  # How confident we are in this profile

class AdaptiveOPSEC(EventEmitter):
    """
    Adaptive OPSEC Intelligence System
    
    Features:
    - Real-time threat detection and response
    - Adaptive rate limiting based on target behavior
    - Dynamic fingerprint randomization
    - Behavioral pattern learning and avoidance
    - Automated countermeasure selection and deployment
    - OPSEC effectiveness measurement and optimization
    """
    
    def __init__(self, config, event_system: EventSystem):
        super().__init__(event_system, "AdaptiveOPSEC")
        
        self.config = config
        
        # Threat detection and response
        self.active_threats: Dict[str, OPSECThreat] = {}
        self.threat_history: deque = deque(maxlen=1000)
        self.target_profiles: Dict[str, OPSECProfile] = {}
        
        # Adaptive parameters
        self.base_request_rate = config.get("opsec.base_request_rate", 30)  # per minute
        self.max_request_rate = config.get("opsec.max_request_rate", 120)
        self.min_request_rate = config.get("opsec.min_request_rate", 5)
        self.detection_threshold = config.get("opsec.detection_threshold", 0.7)
        
        # Rate limiting state
        self.current_rates: Dict[str, float] = {}  # target -> current rate
        self.request_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.rate_adjustment_history: Dict[str, List[Tuple[datetime, float, str]]] = defaultdict(list)
        
        # Fingerprint pools
        self.user_agent_pool: List[str] = []
        self.header_profiles: List[Dict[str, str]] = []
        self.timing_profiles: List[Dict[str, float]] = []
        
        # Learning and adaptation
        self.detection_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.countermeasure_effectiveness: Dict[str, List[float]] = defaultdict(list)
        self.behavioral_models: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize adaptive OPSEC system"""
        
        await self._initialize_fingerprint_pools()
        await self._load_behavioral_models()
        
        # Start background tasks
        asyncio.create_task(self._threat_monitoring_loop())
        asyncio.create_task(self._profile_update_loop())
        asyncio.create_task(self._adaptation_loop())
        
        await self.emit_event(
            EventType.OPSEC_STEALTH_MODE_CHANGE,
            {"message": "Adaptive OPSEC initialized", "mode": "learning"}
        )
        
        logger.info("Adaptive OPSEC system initialized")
        
    async def assess_request_safety(self, target: str, endpoint: str, 
                                  request_type: str) -> Dict[str, Any]:
        """
        Assess the safety of making a request and provide recommendations
        
        Returns:
            {
                "safe": bool,
                "risk_level": float,
                "recommended_delay": float,
                "recommended_headers": dict,
                "warnings": list
            }
        """
        
        risk_factors = []
        risk_level = 0.0
        
        # Check current request rate
        current_rate = await self._calculate_current_rate(target)
        safe_rate = self._get_safe_rate(target)
        
        if current_rate > safe_rate:
            risk_factors.append(f"Request rate {current_rate:.1f}/min exceeds safe rate {safe_rate:.1f}/min")
            risk_level += 0.3
            
        # Check for recent threats
        recent_threats = [t for t in self.threat_history 
                         if t.timestamp > datetime.utcnow() - timedelta(minutes=30)
                         and getattr(t, 'target_endpoint', None) == endpoint]
        
        if recent_threats:
            risk_factors.append(f"{len(recent_threats)} recent threats detected on this endpoint")
            risk_level += len(recent_threats) * 0.2
            
        # Check target profile
        profile = self.target_profiles.get(target)
        if profile:
            # Check if this endpoint matches known detection patterns
            for pattern in profile.common_patterns:
                if pattern in endpoint:
                    risk_factors.append(f"Endpoint matches known detection pattern: {pattern}")
                    risk_level += 0.25
                    
        # Time-based risk assessment
        current_hour = datetime.utcnow().hour
        if profile and profile.time_windows:
            in_safe_window = any(
                tw["start_hour"] <= current_hour <= tw["end_hour"] 
                for tw in profile.time_windows
            )
            if not in_safe_window:
                risk_factors.append("Outside of safe operational time windows")
                risk_level += 0.2
                
        # Normalize risk level
        risk_level = min(risk_level, 1.0)
        
        # Generate recommendations
        recommended_delay = await self._calculate_optimal_delay(target, risk_level)
        recommended_headers = await self._select_safe_headers(target, endpoint)
        
        safety_assessment = {
            "safe": risk_level < self.detection_threshold,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommended_delay": recommended_delay,
            "recommended_headers": recommended_headers,
            "warnings": risk_factors if risk_level > 0.5 else []
        }
        
        return safety_assessment
        
    async def adapt_to_response(self, target: str, endpoint: str, response_code: int,
                               response_headers: Dict[str, str], response_time: float,
                               request_headers: Dict[str, str]):
        """Adapt OPSEC based on target response"""
        
        # Record request timestamp
        self.request_timestamps[target].append(time.time())
        
        # Analyze response for detection indicators
        threats_detected = await self._analyze_response_for_threats(
            target, endpoint, response_code, response_headers, response_time
        )
        
        for threat in threats_detected:
            await self._handle_threat(threat)
            
        # Update target profile
        await self._update_target_profile(target, endpoint, response_code, 
                                        response_headers, request_headers)
        
        # Adaptive rate adjustment
        await self._adapt_request_rate(target, response_code, threats_detected)
        
    async def get_optimal_fingerprint(self, target: str, context: str) -> Dict[str, Any]:
        """Get optimal fingerprint for target and context"""
        
        profile = self.target_profiles.get(target)
        
        # Select user agent
        if profile and profile.safe_user_agents:
            user_agent = random.choice(profile.safe_user_agents)
        else:
            user_agent = random.choice(self.user_agent_pool)
            
        # Select header profile
        base_headers = random.choice(self.header_profiles)
        
        # Customize headers based on profile
        if profile:
            # Avoid known fingerprinting headers
            for fp_method in profile.fingerprinting_methods:
                if fp_method == "accept_language":
                    base_headers["Accept-Language"] = "en-US,en;q=0.9"
                elif fp_method == "accept_encoding":
                    base_headers["Accept-Encoding"] = "gzip, deflate"
                    
        # Add randomization
        fingerprint = {
            "user_agent": user_agent,
            "headers": base_headers,
            "timing_profile": random.choice(self.timing_profiles),
            "viewport": self._generate_random_viewport(),
            "timezone": self._select_safe_timezone()
        }
        
        return fingerprint
        
    async def report_detection_event(self, target: str, detection_type: str, 
                                   indicators: List[str], severity: float):
        """Report a detection event for learning"""
        
        threat = OPSECThreat(
            threat_id=f"detection_{target}_{int(time.time())}",
            threat_type=detection_type,
            severity=severity,
            confidence=0.8,  # User-reported events have high confidence
            indicators=indicators,
            source_ip=None,
            target_endpoint=target,
            timestamp=datetime.utcnow(),
            countermeasures_applied=[]
        )
        
        self.active_threats[threat.threat_id] = threat
        self.threat_history.append(threat)
        
        # Immediate adaptive response
        await self._emergency_opsec_adjustment(target, detection_type)
        
        await self.emit_event(
            EventType.OPSEC_VIOLATION,
            {
                "target": target,
                "detection_type": detection_type,
                "severity": severity,
                "threat_id": threat.threat_id
            }
        )
        
    async def get_opsec_intelligence(self, target: str = None) -> Dict[str, Any]:
        """Get current OPSEC intelligence and recommendations"""
        
        intelligence = {
            "global_threat_level": await self._calculate_global_threat_level(),
            "active_threats": len(self.active_threats),
            "total_targets_profiled": len(self.target_profiles),
            "recent_detections": len([t for t in self.threat_history 
                                   if t.timestamp > datetime.utcnow() - timedelta(hours=24)]),
            "countermeasure_effectiveness": await self._analyze_countermeasure_effectiveness()
        }
        
        if target:
            profile = self.target_profiles.get(target)
            if profile:
                intelligence["target_profile"] = {
                    "detection_sensitivity": profile.detection_sensitivity,
                    "safe_request_rate": profile.safe_request_rate,
                    "confidence_level": profile.confidence_level,
                    "last_updated": profile.last_updated.isoformat(),
                    "known_patterns": len(profile.common_patterns),
                    "safe_time_windows": len(profile.time_windows)
                }
            else:
                intelligence["target_profile"] = None
                
            # Current operational status for target
            current_rate = await self._calculate_current_rate(target)
            intelligence["current_operations"] = {
                "request_rate": current_rate,
                "recent_requests": len(self.request_timestamps[target]),
                "last_rate_adjustment": self._get_last_rate_adjustment(target)
            }
            
        return intelligence
        
    # Private methods
    
    async def _initialize_fingerprint_pools(self):
        """Initialize pools of safe fingerprints"""
        
        # Diverse user agent pool
        self.user_agent_pool = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
        
        # Common header profiles
        self.header_profiles = [
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            },
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate"
            },
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.7,de;q=0.3",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive"
            }
        ]
        
        # Timing profiles (delays between actions)
        self.timing_profiles = [
            {"page_load_wait": 2.0, "action_delay": 0.5, "typing_speed": 0.1},
            {"page_load_wait": 3.5, "action_delay": 1.2, "typing_speed": 0.15},
            {"page_load_wait": 1.8, "action_delay": 0.8, "typing_speed": 0.08},
            {"page_load_wait": 4.0, "action_delay": 2.0, "typing_speed": 0.2}
        ]
        
    async def _load_behavioral_models(self):
        """Load behavioral models for different target types"""
        
        # Basic behavioral models
        self.behavioral_models = {
            "corporate_web": {
                "safe_hours": [9, 10, 11, 14, 15, 16],  # Business hours
                "detection_patterns": ["rapid_scanning", "uncommon_user_agents"],
                "response_characteristics": {"avg_response_time": 0.8, "error_rate": 0.05}
            },
            "e_commerce": {
                "safe_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "detection_patterns": ["cart_manipulation", "price_scraping"],
                "response_characteristics": {"avg_response_time": 1.2, "error_rate": 0.02}
            },
            "api_service": {
                "safe_hours": list(range(24)),  # 24/7 operation
                "detection_patterns": ["rate_limiting", "auth_probing"],
                "response_characteristics": {"avg_response_time": 0.3, "error_rate": 0.01}
            }
        }
        
    async def _calculate_current_rate(self, target: str) -> float:
        """Calculate current request rate for target"""
        
        timestamps = self.request_timestamps[target]
        if len(timestamps) < 2:
            return 0.0
            
        # Count requests in last minute
        current_time = time.time()
        recent_requests = [ts for ts in timestamps if current_time - ts <= 60]
        
        return len(recent_requests)
        
    def _get_safe_rate(self, target: str) -> float:
        """Get safe request rate for target"""
        
        profile = self.target_profiles.get(target)
        if profile:
            return profile.safe_request_rate
        else:
            return self.base_request_rate
            
    async def _calculate_optimal_delay(self, target: str, risk_level: float) -> float:
        """Calculate optimal delay before next request"""
        
        base_delay = 60.0 / self._get_safe_rate(target)  # Base delay for safe rate
        
        # Increase delay based on risk
        risk_multiplier = 1.0 + (risk_level * 3.0)  # Up to 4x delay at max risk
        
        # Add random jitter
        jitter = random.uniform(0.8, 1.2)
        
        optimal_delay = base_delay * risk_multiplier * jitter
        
        return optimal_delay
        
    async def _select_safe_headers(self, target: str, endpoint: str) -> Dict[str, str]:
        """Select safe headers for target"""
        
        profile = self.target_profiles.get(target)
        base_headers = random.choice(self.header_profiles).copy()
        
        if profile:
            # Customize based on known fingerprinting methods
            for method in profile.fingerprinting_methods:
                if method == "referer_tracking":
                    # Add believable referer
                    base_headers["Referer"] = f"https://www.google.com/"
                elif method == "cache_control":
                    base_headers["Cache-Control"] = "no-cache"
                    
        return base_headers
        
    async def _analyze_response_for_threats(self, target: str, endpoint: str, 
                                          response_code: int, response_headers: Dict[str, str],
                                          response_time: float) -> List[OPSECThreat]:
        """Analyze response for potential threats"""
        
        threats = []
        
        # Rate limiting detection
        if response_code == 429:
            threat = OPSECThreat(
                threat_id=f"rate_limit_{target}_{int(time.time())}",
                threat_type="rate_limiting",
                severity=0.8,
                confidence=0.95,
                indicators=["HTTP 429 response"],
                source_ip=None,
                target_endpoint=endpoint,
                timestamp=datetime.utcnow(),
                countermeasures_applied=[]
            )
            threats.append(threat)
            
        # Suspicious response time patterns
        profile = self.target_profiles.get(target)
        if profile:
            expected_time = profile.detection_sensitivity * 2.0  # Rough baseline
            if response_time > expected_time * 3:
                threat = OPSECThreat(
                    threat_id=f"slow_response_{target}_{int(time.time())}",
                    threat_type="behavioral_detection",
                    severity=0.4,
                    confidence=0.6,
                    indicators=[f"Response time {response_time:.2f}s exceeds expected"],
                    source_ip=None,
                    target_endpoint=endpoint,
                    timestamp=datetime.utcnow(),
                    countermeasures_applied=[]
                )
                threats.append(threat)
                
        # Security headers indicating monitoring
        suspicious_headers = [
            "x-rate-limit", "x-ratelimit", "retry-after", 
            "x-content-type-options", "x-frame-options",
            "content-security-policy"
        ]
        
        detected_headers = [h for h in response_headers.keys() 
                          if any(sh in h.lower() for sh in suspicious_headers)]
        
        if len(detected_headers) > 3:
            threat = OPSECThreat(
                threat_id=f"security_headers_{target}_{int(time.time())}",
                threat_type="fingerprinting",
                severity=0.3,
                confidence=0.7,
                indicators=[f"Multiple security headers: {detected_headers}"],
                source_ip=None,
                target_endpoint=endpoint,
                timestamp=datetime.utcnow(),
                countermeasures_applied=[]
            )
            threats.append(threat)
            
        # WAF detection patterns
        waf_indicators = ["cloudflare", "incapsula", "akamai", "blocked"]
        for header_name, header_value in response_headers.items():
            if any(indicator in header_value.lower() for indicator in waf_indicators):
                threat = OPSECThreat(
                    threat_id=f"waf_detection_{target}_{int(time.time())}",
                    threat_type="waf_detection",
                    severity=0.6,
                    confidence=0.8,
                    indicators=[f"WAF indicator in {header_name}: {header_value}"],
                    source_ip=None,
                    target_endpoint=endpoint,
                    timestamp=datetime.utcnow(),
                    countermeasures_applied=[]
                )
                threats.append(threat)
                break
                
        return threats
        
    async def _handle_threat(self, threat: OPSECThreat):
        """Handle detected threat with appropriate countermeasures"""
        
        self.active_threats[threat.threat_id] = threat
        self.threat_history.append(threat)
        
        countermeasures = []
        
        if threat.threat_type == "rate_limiting":
            # Reduce request rate
            target = threat.target_endpoint or "unknown"
            if target in self.current_rates:
                self.current_rates[target] *= 0.5  # Halve the rate
            countermeasures.append("reduced_request_rate")
            
        elif threat.threat_type == "fingerprinting":
            # Randomize fingerprint more aggressively
            countermeasures.append("enhanced_fingerprint_randomization")
            
        elif threat.threat_type == "behavioral_detection":
            # Increase delays and randomization
            countermeasures.append("increased_behavioral_randomization")
            
        elif threat.threat_type == "waf_detection":
            # Switch to more conservative approach
            countermeasures.append("conservative_mode_activated")
            
        threat.countermeasures_applied = countermeasures
        
        await self.emit_event(
            EventType.OPSEC_VIOLATION,
            {
                "threat_type": threat.threat_type,
                "severity": threat.severity,
                "countermeasures": countermeasures
            }
        )
        
        logger.warning(f"OPSEC threat detected: {threat.threat_type} (severity: {threat.severity})")
        
    async def _update_target_profile(self, target: str, endpoint: str, response_code: int,
                                   response_headers: Dict[str, str], request_headers: Dict[str, str]):
        """Update target profile based on response"""
        
        if target not in self.target_profiles:
            # Create new profile
            self.target_profiles[target] = OPSECProfile(
                target_id=target,
                detection_sensitivity=0.5,
                common_patterns=[],
                safe_request_rate=self.base_request_rate,
                safe_user_agents=[],
                time_windows=[],
                fingerprinting_methods=[],
                last_updated=datetime.utcnow(),
                confidence_level=0.1
            )
            
        profile = self.target_profiles[target]
        
        # Update based on response
        if response_code == 200:
            # Success - this approach seems safe
            user_agent = request_headers.get("User-Agent", "")
            if user_agent and user_agent not in profile.safe_user_agents:
                profile.safe_user_agents.append(user_agent)
                
            # Update safe request rate if we're not seeing issues
            current_rate = await self._calculate_current_rate(target)
            if current_rate > profile.safe_request_rate:
                profile.safe_request_rate = min(current_rate, self.max_request_rate)
                
        elif response_code in [429, 403, 406]:
            # Potential detection - reduce safe rate
            profile.safe_request_rate *= 0.8
            profile.detection_sensitivity = min(profile.detection_sensitivity + 0.1, 1.0)
            
        # Analyze response headers for fingerprinting methods
        for header_name in response_headers.keys():
            if "etag" in header_name.lower():
                if "etag_tracking" not in profile.fingerprinting_methods:
                    profile.fingerprinting_methods.append("etag_tracking")
            elif "set-cookie" in header_name.lower():
                if "cookie_tracking" not in profile.fingerprinting_methods:
                    profile.fingerprinting_methods.append("cookie_tracking")
                    
        # Update confidence and timestamp
        profile.confidence_level = min(profile.confidence_level + 0.02, 1.0)
        profile.last_updated = datetime.utcnow()
        
    async def _adapt_request_rate(self, target: str, response_code: int, threats: List[OPSECThreat]):
        """Adapt request rate based on response and threats"""
        
        current_rate = self.current_rates.get(target, self.base_request_rate)
        
        if threats:
            # Threats detected - reduce rate aggressively
            new_rate = current_rate * 0.6
            reason = f"threats_detected_{len(threats)}"
        elif response_code == 200:
            # Success - can slightly increase rate
            new_rate = min(current_rate * 1.05, self.max_request_rate)
            reason = "success_response"
        elif response_code in [429, 503]:
            # Server overload - reduce rate
            new_rate = current_rate * 0.4
            reason = f"server_response_{response_code}"
        else:
            # Other responses - slight reduction
            new_rate = current_rate * 0.9
            reason = f"response_code_{response_code}"
            
        # Apply minimum rate limit
        new_rate = max(new_rate, self.min_request_rate)
        
        if abs(new_rate - current_rate) > 1.0:  # Only record significant changes
            self.current_rates[target] = new_rate
            self.rate_adjustment_history[target].append(
                (datetime.utcnow(), new_rate, reason)
            )
            
            # Limit history
            if len(self.rate_adjustment_history[target]) > 50:
                self.rate_adjustment_history[target] = self.rate_adjustment_history[target][-25:]
                
    async def _emergency_opsec_adjustment(self, target: str, detection_type: str):
        """Emergency OPSEC adjustments when detection is reported"""
        
        # Immediate rate reduction
        if target in self.current_rates:
            self.current_rates[target] = max(self.current_rates[target] * 0.3, self.min_request_rate)
        else:
            self.current_rates[target] = self.min_request_rate
            
        # Enhanced randomization
        if detection_type == "fingerprinting":
            # Regenerate fingerprint pools
            await self._initialize_fingerprint_pools()
            
        # Update target profile with detection info
        if target in self.target_profiles:
            profile = self.target_profiles[target]
            profile.detection_sensitivity = min(profile.detection_sensitivity + 0.3, 1.0)
            profile.safe_request_rate = self.current_rates[target]
            
    def _generate_random_viewport(self) -> Tuple[int, int]:
        """Generate random but realistic viewport size"""
        
        common_viewports = [
            (1920, 1080), (1366, 768), (1440, 900), (1536, 864),
            (1280, 720), (1600, 900), (1024, 768), (1280, 1024)
        ]
        
        return random.choice(common_viewports)
        
    def _select_safe_timezone(self) -> str:
        """Select a safe timezone"""
        
        safe_timezones = [
            "America/New_York", "America/Los_Angeles", "America/Chicago",
            "Europe/London", "Europe/Berlin", "Asia/Tokyo", "Australia/Sydney"
        ]
        
        return random.choice(safe_timezones)
        
    async def _calculate_global_threat_level(self) -> float:
        """Calculate global threat level based on recent activity"""
        
        recent_threats = [t for t in self.threat_history 
                         if t.timestamp > datetime.utcnow() - timedelta(hours=24)]
        
        if not recent_threats:
            return 0.1  # Low baseline threat
            
        # Calculate based on threat count and severity
        threat_count_factor = min(len(recent_threats) / 10.0, 1.0)
        avg_severity = statistics.mean([t.severity for t in recent_threats])
        
        global_threat = (threat_count_factor * 0.4 + avg_severity * 0.6)
        
        return min(global_threat, 1.0)
        
    async def _analyze_countermeasure_effectiveness(self) -> Dict[str, float]:
        """Analyze effectiveness of different countermeasures"""
        
        effectiveness = {}
        
        for countermeasure_type in ["reduced_request_rate", "enhanced_fingerprint_randomization", 
                                  "increased_behavioral_randomization", "conservative_mode_activated"]:
            
            effectiveness_scores = self.countermeasure_effectiveness.get(countermeasure_type, [])
            
            if effectiveness_scores:
                effectiveness[countermeasure_type] = statistics.mean(effectiveness_scores)
            else:
                effectiveness[countermeasure_type] = 0.5  # Neutral
                
        return effectiveness
        
    def _get_last_rate_adjustment(self, target: str) -> Optional[Dict[str, Any]]:
        """Get last rate adjustment for target"""
        
        history = self.rate_adjustment_history.get(target, [])
        if history:
            timestamp, rate, reason = history[-1]
            return {
                "timestamp": timestamp.isoformat(),
                "rate": rate,
                "reason": reason
            }
        return None
        
    # Background tasks
    
    async def _threat_monitoring_loop(self):
        """Background threat monitoring and cleanup"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Clean up old active threats
                current_time = datetime.utcnow()
                expired_threats = []
                
                for threat_id, threat in self.active_threats.items():
                    if current_time - threat.timestamp > timedelta(hours=2):
                        expired_threats.append(threat_id)
                        
                for threat_id in expired_threats:
                    del self.active_threats[threat_id]
                    
                # Analyze threat patterns
                await self._analyze_threat_patterns()
                
            except Exception as e:
                logger.error(f"Error in threat monitoring loop: {e}")
                await asyncio.sleep(60)
                
    async def _profile_update_loop(self):
        """Background profile updates and optimization"""
        
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Update profile confidence based on age and usage
                for target, profile in self.target_profiles.items():
                    age_hours = (datetime.utcnow() - profile.last_updated).total_seconds() / 3600
                    
                    # Decay confidence over time
                    if age_hours > 24:
                        profile.confidence_level *= 0.95
                        
                    # Update time windows based on successful operations
                    current_hour = datetime.utcnow().hour
                    # This would be enhanced with actual success/failure tracking
                    
            except Exception as e:
                logger.error(f"Error in profile update loop: {e}")
                await asyncio.sleep(300)
                
    async def _adaptation_loop(self):
        """Background adaptation and learning"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Adapt mutation rates based on threat levels
                global_threat = await self._calculate_global_threat_level()
                
                if global_threat > 0.7:
                    # High threat - increase randomization
                    self.detection_threshold = max(0.5, self.detection_threshold - 0.1)
                elif global_threat < 0.3:
                    # Low threat - can be slightly more aggressive
                    self.detection_threshold = min(0.9, self.detection_threshold + 0.05)
                    
                # Learn from countermeasure effectiveness
                await self._update_countermeasure_strategies()
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                await asyncio.sleep(300)
                
    async def _analyze_threat_patterns(self):
        """Analyze patterns in threat detection"""
        
        # Group threats by type and look for patterns
        threat_types = defaultdict(list)
        
        for threat in list(self.threat_history)[-100:]:  # Last 100 threats
            threat_types[threat.threat_type].append(threat)
            
        # Update detection patterns
        for threat_type, threats in threat_types.items():
            if len(threats) > 5:  # Enough data to analyze
                # Analyze common indicators
                all_indicators = []
                for threat in threats:
                    all_indicators.extend(threat.indicators)
                    
                # Find common patterns
                from collections import Counter
                common_indicators = Counter(all_indicators).most_common(5)
                
                self.detection_patterns[threat_type] = [
                    {"indicator": indicator, "frequency": freq}
                    for indicator, freq in common_indicators
                ]
                
    async def _update_countermeasure_strategies(self):
        """Update countermeasure strategies based on effectiveness"""
        
        # This would implement learning from countermeasure effectiveness
        # For now, just log the analysis
        effectiveness = await self._analyze_countermeasure_effectiveness()
        
        logger.info(f"Countermeasure effectiveness analysis: {effectiveness}")
        
    async def shutdown(self):
        """Shutdown adaptive OPSEC system"""
        logger.info("Adaptive OPSEC system shutdown complete")