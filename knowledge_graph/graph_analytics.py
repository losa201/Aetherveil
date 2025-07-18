"""
Graph Analytics - Advanced analytics and intelligence correlation for the knowledge graph
Provides comprehensive analytics, pattern recognition, and intelligence correlation capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import math
import statistics
import numpy as np
from itertools import combinations
import pickle
from pathlib import Path

import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jaccard

from .graph_manager import GraphManager, GraphNode, GraphEdge
from .graph_schema import NodeType, RelationType, SeverityLevel, ThreatType
from .attack_path_analyzer import AttackPathAnalyzer
from .vulnerability_mapper import VulnerabilityMapper
from .graph_algorithms import GraphAlgorithms


@dataclass
class AnalyticsPattern:
    """Represents a discovered analytics pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    entities: List[str]
    confidence_score: float
    risk_level: str
    frequency: int
    first_observed: datetime
    last_observed: datetime
    indicators: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "entities": self.entities,
            "confidence_score": self.confidence_score,
            "risk_level": self.risk_level,
            "frequency": self.frequency,
            "first_observed": self.first_observed.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "indicators": self.indicators,
            "metadata": self.metadata
        }


@dataclass
class IntelligenceCorrelation:
    """Represents correlated intelligence across multiple sources"""
    correlation_id: str
    primary_entity: str
    related_entities: List[str]
    correlation_type: str
    strength: float
    sources: List[str]
    evidence: List[Dict[str, Any]]
    temporal_correlation: bool
    geographic_correlation: bool
    technical_correlation: bool
    discovered_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "correlation_id": self.correlation_id,
            "primary_entity": self.primary_entity,
            "related_entities": self.related_entities,
            "correlation_type": self.correlation_type,
            "strength": self.strength,
            "sources": self.sources,
            "evidence": self.evidence,
            "temporal_correlation": self.temporal_correlation,
            "geographic_correlation": self.geographic_correlation,
            "technical_correlation": self.technical_correlation,
            "discovered_at": self.discovered_at.isoformat()
        }


@dataclass
class TrendAnalysis:
    """Represents trend analysis results"""
    trend_id: str
    entity_type: str
    metric: str
    time_period: str
    trend_direction: str  # increasing, decreasing, stable, volatile
    change_rate: float
    statistical_significance: float
    data_points: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    analysis_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "trend_id": self.trend_id,
            "entity_type": self.entity_type,
            "metric": self.metric,
            "time_period": self.time_period,
            "trend_direction": self.trend_direction,
            "change_rate": self.change_rate,
            "statistical_significance": self.statistical_significance,
            "data_points": self.data_points,
            "anomalies": self.anomalies,
            "predictions": self.predictions,
            "analysis_date": self.analysis_date.isoformat()
        }


@dataclass
class RiskScore:
    """Comprehensive risk scoring"""
    entity_id: str
    overall_risk: float
    vulnerability_risk: float
    exposure_risk: float
    threat_risk: float
    temporal_risk: float
    network_risk: float
    contributing_factors: List[str]
    mitigation_recommendations: List[str]
    calculated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entity_id": self.entity_id,
            "overall_risk": self.overall_risk,
            "vulnerability_risk": self.vulnerability_risk,
            "exposure_risk": self.exposure_risk,
            "threat_risk": self.threat_risk,
            "temporal_risk": self.temporal_risk,
            "network_risk": self.network_risk,
            "contributing_factors": self.contributing_factors,
            "mitigation_recommendations": self.mitigation_recommendations,
            "calculated_at": self.calculated_at.isoformat()
        }


class GraphAnalytics:
    """Advanced graph analytics and intelligence correlation engine"""
    
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.logger = logging.getLogger(__name__)
        
        # Component dependencies
        self.attack_analyzer = AttackPathAnalyzer(graph_manager)
        self.vulnerability_mapper = VulnerabilityMapper(graph_manager)
        self.graph_algorithms = GraphAlgorithms(graph_manager)
        
        # Analytics cache
        self.pattern_cache = {}
        self.correlation_cache = {}
        self.trend_cache = {}
        self.risk_cache = {}
        
        # Analytics parameters
        self.pattern_confidence_threshold = 0.7
        self.correlation_strength_threshold = 0.6
        self.trend_window_days = 30
        self.max_patterns_per_analysis = 100
        
        # Pattern templates
        self.pattern_templates = {
            "lateral_movement": {
                "description": "Suspicious lateral movement pattern",
                "min_nodes": 3,
                "required_types": [NodeType.HOST, NodeType.CREDENTIAL],
                "required_relations": [RelationType.USES, RelationType.ACCESSES]
            },
            "privilege_escalation": {
                "description": "Privilege escalation pattern",
                "min_nodes": 2,
                "required_types": [NodeType.VULNERABILITY, NodeType.USER],
                "required_relations": [RelationType.EXPLOITS]
            },
            "data_exfiltration": {
                "description": "Data exfiltration pattern",
                "min_nodes": 3,
                "required_types": [NodeType.HOST, NodeType.NETWORK],
                "required_relations": [RelationType.CONNECTS_TO, RelationType.ACCESSES]
            },
            "command_control": {
                "description": "Command and control pattern",
                "min_nodes": 2,
                "required_types": [NodeType.HOST, NodeType.DOMAIN],
                "required_relations": [RelationType.COMMUNICATES]
            },
            "vulnerability_cluster": {
                "description": "Vulnerability clustering pattern",
                "min_nodes": 3,
                "required_types": [NodeType.VULNERABILITY, NodeType.HOST],
                "required_relations": [RelationType.AFFECTS]
            }
        }
        
        # Intelligence sources configuration
        self.intelligence_sources = {
            "internal_analysis": {"weight": 1.0, "reliability": 0.9},
            "threat_intel_feeds": {"weight": 0.8, "reliability": 0.7},
            "vulnerability_databases": {"weight": 0.9, "reliability": 0.95},
            "attack_frameworks": {"weight": 0.85, "reliability": 0.8},
            "behavioral_analysis": {"weight": 0.7, "reliability": 0.6}
        }
    
    async def analyze_patterns(self, pattern_types: Optional[List[str]] = None) -> List[AnalyticsPattern]:
        """Analyze the graph for security patterns"""
        try:
            patterns = []
            
            # Use specified pattern types or all available
            types_to_analyze = pattern_types or list(self.pattern_templates.keys())
            
            for pattern_type in types_to_analyze:
                detected_patterns = await self._detect_pattern_type(pattern_type)
                patterns.extend(detected_patterns)
            
            # Sort by confidence score
            patterns.sort(key=lambda p: p.confidence_score, reverse=True)
            
            # Cache results
            cache_key = f"patterns_{datetime.now().strftime('%Y%m%d')}"
            self.pattern_cache[cache_key] = patterns
            
            return patterns[:self.max_patterns_per_analysis]
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {e}")
            return []
    
    async def _detect_pattern_type(self, pattern_type: str) -> List[AnalyticsPattern]:
        """Detect specific pattern type"""
        template = self.pattern_templates.get(pattern_type)
        if not template:
            return []
        
        patterns = []
        
        try:
            if pattern_type == "lateral_movement":
                patterns = await self._detect_lateral_movement()
            elif pattern_type == "privilege_escalation":
                patterns = await self._detect_privilege_escalation()
            elif pattern_type == "data_exfiltration":
                patterns = await self._detect_data_exfiltration()
            elif pattern_type == "command_control":
                patterns = await self._detect_command_control()
            elif pattern_type == "vulnerability_cluster":
                patterns = await self._detect_vulnerability_clusters()
            
        except Exception as e:
            self.logger.error(f"Error detecting {pattern_type} patterns: {e}")
        
        return patterns
    
    async def _detect_lateral_movement(self) -> List[AnalyticsPattern]:
        """Detect lateral movement patterns"""
        patterns = []
        
        # Get credential nodes
        credential_nodes = await self.graph_manager.find_nodes(NodeType.CREDENTIAL, limit=50)
        
        for cred_node in credential_nodes:
            # Find hosts that use this credential
            hosts_using_cred = await self.graph_manager.get_neighbors(
                cred_node.id, 
                direction="incoming",
                edge_type=RelationType.USES
            )
            
            if len(hosts_using_cred) >= 3:  # Potential lateral movement
                # Calculate confidence based on various factors
                confidence = await self._calculate_lateral_movement_confidence(
                    cred_node, hosts_using_cred
                )
                
                if confidence >= self.pattern_confidence_threshold:
                    pattern = AnalyticsPattern(
                        pattern_id=f"lateral_movement_{cred_node.id}",
                        pattern_type="lateral_movement",
                        description=f"Potential lateral movement using credential {cred_node.id}",
                        entities=[cred_node.id] + [host.id for host in hosts_using_cred],
                        confidence_score=confidence,
                        risk_level=await self._calculate_risk_level(confidence),
                        frequency=len(hosts_using_cred),
                        first_observed=cred_node.created_at,
                        last_observed=datetime.now(),
                        indicators=[f"credential_reuse:{cred_node.id}"],
                        metadata={
                            "credential_type": cred_node.properties.get("type", "unknown"),
                            "affected_hosts": len(hosts_using_cred)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_privilege_escalation(self) -> List[AnalyticsPattern]:
        """Detect privilege escalation patterns"""
        patterns = []
        
        # Get vulnerability nodes
        vuln_nodes = await self.graph_manager.find_nodes(NodeType.VULNERABILITY, limit=100)
        
        for vuln_node in vuln_nodes:
            # Check if vulnerability enables privilege escalation
            vuln_description = vuln_node.properties.get("description", "").lower()
            if any(keyword in vuln_description for keyword in ["privilege", "escalation", "elevation"]):
                
                # Find affected hosts
                affected_hosts = await self.graph_manager.get_neighbors(
                    vuln_node.id,
                    direction="outgoing", 
                    edge_type=RelationType.AFFECTS
                )
                
                if affected_hosts:
                    severity = vuln_node.properties.get("severity", "medium")
                    confidence = 0.8 if severity in ["critical", "high"] else 0.6
                    
                    pattern = AnalyticsPattern(
                        pattern_id=f"privesc_{vuln_node.id}",
                        pattern_type="privilege_escalation",
                        description=f"Privilege escalation vulnerability {vuln_node.properties.get('cve_id', 'unknown')}",
                        entities=[vuln_node.id] + [host.id for host in affected_hosts],
                        confidence_score=confidence,
                        risk_level=await self._calculate_risk_level(confidence),
                        frequency=len(affected_hosts),
                        first_observed=vuln_node.created_at,
                        last_observed=datetime.now(),
                        indicators=[f"privilege_escalation_vuln:{vuln_node.properties.get('cve_id', 'unknown')}"],
                        metadata={
                            "cve_id": vuln_node.properties.get("cve_id"),
                            "severity": severity,
                            "affected_hosts": len(affected_hosts)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_data_exfiltration(self) -> List[AnalyticsPattern]:
        """Detect data exfiltration patterns"""
        patterns = []
        
        # Get network nodes (potential external destinations)
        network_nodes = await self.graph_manager.find_nodes(NodeType.NETWORK, limit=20)
        
        for network_node in network_nodes:
            # Find hosts connecting to this network
            connected_hosts = await self.graph_manager.get_neighbors(
                network_node.id,
                direction="incoming",
                edge_type=RelationType.CONNECTS_TO
            )
            
            # Check for external networks
            if network_node.properties.get("type") == "external" and connected_hosts:
                confidence = 0.6  # Base confidence for external connections
                
                # Increase confidence if multiple hosts connect
                if len(connected_hosts) > 1:
                    confidence += 0.2
                
                pattern = AnalyticsPattern(
                    pattern_id=f"exfiltration_{network_node.id}",
                    pattern_type="data_exfiltration",
                    description=f"Potential data exfiltration to external network {network_node.id}",
                    entities=[network_node.id] + [host.id for host in connected_hosts],
                    confidence_score=confidence,
                    risk_level=await self._calculate_risk_level(confidence),
                    frequency=len(connected_hosts),
                    first_observed=network_node.created_at,
                    last_observed=datetime.now(),
                    indicators=[f"external_connection:{network_node.id}"],
                    metadata={
                        "network_type": network_node.properties.get("type"),
                        "connecting_hosts": len(connected_hosts)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_command_control(self) -> List[AnalyticsPattern]:
        """Detect command and control patterns"""
        patterns = []
        
        # Get domain nodes (potential C&C servers)
        domain_nodes = await self.graph_manager.find_nodes(NodeType.DOMAIN, limit=30)
        
        for domain_node in domain_nodes:
            # Find hosts communicating with this domain
            communicating_hosts = await self.graph_manager.get_neighbors(
                domain_node.id,
                direction="incoming",
                edge_type=RelationType.COMMUNICATES
            )
            
            if communicating_hosts:
                # Check domain reputation indicators
                domain_name = domain_node.properties.get("name", "")
                suspicious_indicators = await self._check_domain_reputation(domain_name)
                
                confidence = 0.5  # Base confidence
                if suspicious_indicators:
                    confidence += 0.3
                if len(communicating_hosts) > 1:
                    confidence += 0.2
                
                if confidence >= self.pattern_confidence_threshold:
                    pattern = AnalyticsPattern(
                        pattern_id=f"c2_{domain_node.id}",
                        pattern_type="command_control",
                        description=f"Potential C&C communication to domain {domain_name}",
                        entities=[domain_node.id] + [host.id for host in communicating_hosts],
                        confidence_score=confidence,
                        risk_level=await self._calculate_risk_level(confidence),
                        frequency=len(communicating_hosts),
                        first_observed=domain_node.created_at,
                        last_observed=datetime.now(),
                        indicators=[f"c2_domain:{domain_name}"] + suspicious_indicators,
                        metadata={
                            "domain_name": domain_name,
                            "communicating_hosts": len(communicating_hosts),
                            "suspicious_indicators": len(suspicious_indicators)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_vulnerability_clusters(self) -> List[AnalyticsPattern]:
        """Detect vulnerability clustering patterns"""
        patterns = []
        
        # Get all vulnerability nodes
        vuln_nodes = await self.graph_manager.find_nodes(NodeType.VULNERABILITY, limit=100)
        
        # Group vulnerabilities by affected hosts
        host_vulns = defaultdict(list)
        for vuln in vuln_nodes:
            affected_hosts = await self.graph_manager.get_neighbors(
                vuln.id,
                direction="outgoing",
                edge_type=RelationType.AFFECTS
            )
            for host in affected_hosts:
                host_vulns[host.id].append(vuln)
        
        # Find hosts with multiple vulnerabilities
        for host_id, vulnerabilities in host_vulns.items():
            if len(vulnerabilities) >= 3:  # Cluster threshold
                # Calculate cluster risk
                severity_scores = []
                for vuln in vulnerabilities:
                    severity = vuln.properties.get("severity", "medium")
                    score = {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(severity, 1)
                    severity_scores.append(score)
                
                avg_severity = statistics.mean(severity_scores)
                confidence = min(0.9, 0.5 + (avg_severity / 4) * 0.4)
                
                pattern = AnalyticsPattern(
                    pattern_id=f"vuln_cluster_{host_id}",
                    pattern_type="vulnerability_cluster",
                    description=f"Vulnerability cluster on host {host_id}",
                    entities=[host_id] + [vuln.id for vuln in vulnerabilities],
                    confidence_score=confidence,
                    risk_level=await self._calculate_risk_level(confidence),
                    frequency=len(vulnerabilities),
                    first_observed=min(vuln.created_at for vuln in vulnerabilities),
                    last_observed=datetime.now(),
                    indicators=[f"vuln_cluster:{host_id}"],
                    metadata={
                        "host_id": host_id,
                        "vulnerability_count": len(vulnerabilities),
                        "average_severity": avg_severity
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def correlate_intelligence(self, time_window_hours: int = 24) -> List[IntelligenceCorrelation]:
        """Correlate intelligence across multiple sources and entities"""
        try:
            correlations = []
            
            # Get recent nodes for correlation
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Temporal correlation
            temporal_correlations = await self._find_temporal_correlations(cutoff_time)
            correlations.extend(temporal_correlations)
            
            # Technical correlation (similar properties/attributes)
            technical_correlations = await self._find_technical_correlations()
            correlations.extend(technical_correlations)
            
            # Behavioral correlation
            behavioral_correlations = await self._find_behavioral_correlations()
            correlations.extend(behavioral_correlations)
            
            # Sort by correlation strength
            correlations.sort(key=lambda c: c.strength, reverse=True)
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error correlating intelligence: {e}")
            return []
    
    async def _find_temporal_correlations(self, cutoff_time: datetime) -> List[IntelligenceCorrelation]:
        """Find temporal correlations in recent activities"""
        correlations = []
        
        try:
            # Get recent attacks and threats
            attack_nodes = await self.graph_manager.find_nodes(NodeType.ATTACK, limit=50)
            threat_nodes = await self.graph_manager.find_nodes(NodeType.THREAT, limit=30)
            
            # Group by time windows
            time_window_minutes = 60
            time_groups = defaultdict(list)
            
            for node in attack_nodes + threat_nodes:
                time_key = node.created_at.replace(
                    minute=node.created_at.minute // time_window_minutes * time_window_minutes,
                    second=0, microsecond=0
                )
                time_groups[time_key].append(node)
            
            # Find groups with multiple related entities
            for time_key, nodes in time_groups.items():
                if len(nodes) >= 2:
                    # Calculate correlation strength based on temporal proximity
                    time_span = max(node.created_at for node in nodes) - min(node.created_at for node in nodes)
                    strength = max(0.3, 1.0 - (time_span.total_seconds() / 3600))  # Stronger for closer times
                    
                    if strength >= self.correlation_strength_threshold:
                        correlation = IntelligenceCorrelation(
                            correlation_id=f"temporal_{time_key.timestamp()}",
                            primary_entity=nodes[0].id,
                            related_entities=[node.id for node in nodes[1:]],
                            correlation_type="temporal",
                            strength=strength,
                            sources=["internal_analysis"],
                            evidence=[{
                                "type": "temporal_proximity",
                                "time_window": time_window_minutes,
                                "entities": len(nodes)
                            }],
                            temporal_correlation=True,
                            geographic_correlation=False,
                            technical_correlation=False,
                            discovered_at=datetime.now()
                        )
                        correlations.append(correlation)
            
        except Exception as e:
            self.logger.error(f"Error finding temporal correlations: {e}")
        
        return correlations
    
    async def _find_technical_correlations(self) -> List[IntelligenceCorrelation]:
        """Find technical correlations based on similar properties"""
        correlations = []
        
        try:
            # Get vulnerability nodes for technical correlation
            vuln_nodes = await self.graph_manager.find_nodes(NodeType.VULNERABILITY, limit=100)
            
            # Group by CWE categories
            cwe_groups = defaultdict(list)
            for vuln in vuln_nodes:
                cwe_id = vuln.properties.get("cwe_id")
                if cwe_id:
                    cwe_category = cwe_id.split("-")[0] if "-" in cwe_id else cwe_id
                    cwe_groups[cwe_category].append(vuln)
            
            # Find groups with multiple vulnerabilities
            for cwe_category, vulns in cwe_groups.items():
                if len(vulns) >= 2:
                    # Calculate similarity based on properties
                    similarity_score = await self._calculate_vulnerability_similarity(vulns)
                    
                    if similarity_score >= self.correlation_strength_threshold:
                        correlation = IntelligenceCorrelation(
                            correlation_id=f"technical_{cwe_category}",
                            primary_entity=vulns[0].id,
                            related_entities=[vuln.id for vuln in vulns[1:]],
                            correlation_type="technical",
                            strength=similarity_score,
                            sources=["vulnerability_databases"],
                            evidence=[{
                                "type": "cwe_similarity",
                                "cwe_category": cwe_category,
                                "vulnerability_count": len(vulns)
                            }],
                            temporal_correlation=False,
                            geographic_correlation=False,
                            technical_correlation=True,
                            discovered_at=datetime.now()
                        )
                        correlations.append(correlation)
            
        except Exception as e:
            self.logger.error(f"Error finding technical correlations: {e}")
        
        return correlations
    
    async def _find_behavioral_correlations(self) -> List[IntelligenceCorrelation]:
        """Find behavioral correlations based on patterns"""
        correlations = []
        
        try:
            # Get host nodes and analyze their behavior patterns
            host_nodes = await self.graph_manager.find_nodes(NodeType.HOST, limit=50)
            
            # Create behavior vectors for each host
            behavior_vectors = {}
            for host in host_nodes:
                vector = await self._create_behavior_vector(host)
                behavior_vectors[host.id] = vector
            
            # Find hosts with similar behavior patterns
            host_ids = list(behavior_vectors.keys())
            for i in range(len(host_ids)):
                for j in range(i + 1, len(host_ids)):
                    host_a, host_b = host_ids[i], host_ids[j]
                    similarity = await self._calculate_behavior_similarity(
                        behavior_vectors[host_a], 
                        behavior_vectors[host_b]
                    )
                    
                    if similarity >= self.correlation_strength_threshold:
                        correlation = IntelligenceCorrelation(
                            correlation_id=f"behavioral_{host_a}_{host_b}",
                            primary_entity=host_a,
                            related_entities=[host_b],
                            correlation_type="behavioral",
                            strength=similarity,
                            sources=["behavioral_analysis"],
                            evidence=[{
                                "type": "behavior_similarity",
                                "similarity_score": similarity,
                                "behavior_dimensions": len(behavior_vectors[host_a])
                            }],
                            temporal_correlation=False,
                            geographic_correlation=False,
                            technical_correlation=False,
                            discovered_at=datetime.now()
                        )
                        correlations.append(correlation)
            
        except Exception as e:
            self.logger.error(f"Error finding behavioral correlations: {e}")
        
        return correlations
    
    async def analyze_trends(self, entity_types: Optional[List[str]] = None, 
                           time_period_days: int = 30) -> List[TrendAnalysis]:
        """Analyze trends in graph entities and activities"""
        try:
            trends = []
            
            # Default entity types to analyze
            types_to_analyze = entity_types or ["vulnerability", "threat", "attack"]
            
            for entity_type in types_to_analyze:
                entity_trends = await self._analyze_entity_trends(entity_type, time_period_days)
                trends.extend(entity_trends)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return []
    
    async def _analyze_entity_trends(self, entity_type: str, time_period_days: int) -> List[TrendAnalysis]:
        """Analyze trends for specific entity type"""
        trends = []
        
        try:
            # Get entities of specified type
            try:
                node_type = NodeType(entity_type)
                entities = await self.graph_manager.find_nodes(node_type, limit=200)
            except ValueError:
                return trends
            
            # Group entities by time periods
            time_buckets = await self._create_time_buckets(entities, time_period_days)
            
            # Analyze different metrics
            metrics = ["count", "severity_distribution", "geographic_distribution"]
            
            for metric in metrics:
                trend_data = await self._calculate_trend_metric(time_buckets, metric, entity_type)
                
                if trend_data:
                    trend = TrendAnalysis(
                        trend_id=f"{entity_type}_{metric}_{datetime.now().strftime('%Y%m%d')}",
                        entity_type=entity_type,
                        metric=metric,
                        time_period=f"{time_period_days}_days",
                        trend_direction=trend_data["direction"],
                        change_rate=trend_data["change_rate"],
                        statistical_significance=trend_data["significance"],
                        data_points=trend_data["data_points"],
                        anomalies=trend_data["anomalies"],
                        predictions=trend_data["predictions"],
                        analysis_date=datetime.now()
                    )
                    trends.append(trend)
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends for {entity_type}: {e}")
        
        return trends
    
    async def calculate_comprehensive_risk_scores(self, entity_ids: Optional[List[str]] = None) -> List[RiskScore]:
        """Calculate comprehensive risk scores for entities"""
        try:
            risk_scores = []
            
            # Get entities to analyze
            if entity_ids:
                entities = []
                for entity_id in entity_ids:
                    entity = await self.graph_manager.get_node(entity_id)
                    if entity:
                        entities.append(entity)
            else:
                # Get all critical entity types
                entities = []
                for node_type in [NodeType.HOST, NodeType.SERVICE, NodeType.NETWORK]:
                    nodes = await self.graph_manager.find_nodes(node_type, limit=50)
                    entities.extend(nodes)
            
            # Calculate risk for each entity
            for entity in entities:
                risk_score = await self._calculate_entity_risk_score(entity)
                risk_scores.append(risk_score)
            
            # Sort by overall risk
            risk_scores.sort(key=lambda r: r.overall_risk, reverse=True)
            
            return risk_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating risk scores: {e}")
            return []
    
    async def _calculate_entity_risk_score(self, entity: GraphNode) -> RiskScore:
        """Calculate comprehensive risk score for an entity"""
        # Initialize risk components
        vulnerability_risk = 0.0
        exposure_risk = 0.0
        threat_risk = 0.0
        temporal_risk = 0.0
        network_risk = 0.0
        
        contributing_factors = []
        mitigation_recommendations = []
        
        try:
            # Vulnerability risk
            vulnerabilities = await self.graph_manager.get_neighbors(
                entity.id, edge_type=RelationType.AFFECTS, direction="incoming"
            )
            
            if vulnerabilities:
                vuln_scores = []
                for vuln in vulnerabilities:
                    severity = vuln.properties.get("severity", "medium")
                    score = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}.get(severity, 0.3)
                    vuln_scores.append(score)
                    contributing_factors.append(f"Vulnerability: {vuln.properties.get('cve_id', 'unknown')}")
                
                vulnerability_risk = min(1.0, statistics.mean(vuln_scores) * (1 + math.log10(len(vuln_scores))))
                mitigation_recommendations.append("Patch critical and high severity vulnerabilities")
            
            # Exposure risk
            external_connections = await self.graph_manager.get_neighbors(
                entity.id, edge_type=RelationType.CONNECTS_TO
            )
            
            for conn in external_connections:
                if conn.properties.get("type") == "external":
                    exposure_risk += 0.3
                    contributing_factors.append("External network connection")
            
            exposure_risk = min(1.0, exposure_risk)
            if exposure_risk > 0.5:
                mitigation_recommendations.append("Review and restrict external network access")
            
            # Threat risk
            threats = await self.graph_manager.get_neighbors(
                entity.id, edge_type=RelationType.TARGETS, direction="incoming"
            )
            
            if threats:
                threat_scores = []
                for threat in threats:
                    severity_score = threat.properties.get("severity_score", 50.0)
                    threat_scores.append(severity_score / 100.0)
                    contributing_factors.append(f"Active threat: {threat.properties.get('name', 'unknown')}")
                
                threat_risk = min(1.0, statistics.mean(threat_scores))
                mitigation_recommendations.append("Implement threat-specific countermeasures")
            
            # Temporal risk (recent activity)
            entity_age = (datetime.now() - entity.created_at).days
            if entity_age < 7:
                temporal_risk = 0.3  # New entities have unknown risk
                contributing_factors.append("Recently discovered entity")
            elif entity_age < 30:
                temporal_risk = 0.1
            
            # Network risk (centrality in attack paths)
            try:
                centrality_measures = await self.graph_algorithms.calculate_centrality_measures([entity.id])
                if centrality_measures:
                    centrality = centrality_measures[0]
                    network_risk = (
                        centrality.betweenness_centrality * 0.4 +
                        centrality.degree_centrality * 0.3 +
                        centrality.pagerank * 0.3
                    )
                    if network_risk > 0.5:
                        contributing_factors.append("High network centrality")
                        mitigation_recommendations.append("Monitor critical network position")
            except Exception:
                pass
            
            # Calculate overall risk
            weights = {
                "vulnerability": 0.3,
                "exposure": 0.2,
                "threat": 0.25,
                "temporal": 0.1,
                "network": 0.15
            }
            
            overall_risk = (
                vulnerability_risk * weights["vulnerability"] +
                exposure_risk * weights["exposure"] +
                threat_risk * weights["threat"] +
                temporal_risk * weights["temporal"] +
                network_risk * weights["network"]
            )
            
            # Add general recommendations
            if overall_risk > 0.8:
                mitigation_recommendations.append("Immediate attention required - critical risk level")
            elif overall_risk > 0.6:
                mitigation_recommendations.append("High priority for security review")
            elif overall_risk > 0.4:
                mitigation_recommendations.append("Schedule security assessment")
            
        except Exception as e:
            self.logger.error(f"Error calculating risk for entity {entity.id}: {e}")
        
        return RiskScore(
            entity_id=entity.id,
            overall_risk=overall_risk,
            vulnerability_risk=vulnerability_risk,
            exposure_risk=exposure_risk,
            threat_risk=threat_risk,
            temporal_risk=temporal_risk,
            network_risk=network_risk,
            contributing_factors=contributing_factors,
            mitigation_recommendations=mitigation_recommendations,
            calculated_at=datetime.now()
        )
    
    # Helper methods
    
    async def _calculate_lateral_movement_confidence(self, cred_node: GraphNode, hosts: List[GraphNode]) -> float:
        """Calculate confidence for lateral movement pattern"""
        base_confidence = 0.6
        
        # Increase confidence based on number of hosts
        host_factor = min(0.3, len(hosts) * 0.05)
        
        # Increase confidence if hosts are in different networks
        network_diversity = len(set(host.properties.get("network", "unknown") for host in hosts))
        network_factor = min(0.2, network_diversity * 0.1)
        
        # Check credential strength
        strength = cred_node.properties.get("strength_score", 50)
        strength_factor = -0.1 if strength > 80 else 0.1  # Weak credentials more suspicious
        
        return min(1.0, base_confidence + host_factor + network_factor + strength_factor)
    
    async def _calculate_risk_level(self, confidence: float) -> str:
        """Calculate risk level from confidence score"""
        if confidence >= 0.9:
            return "critical"
        elif confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    async def _check_domain_reputation(self, domain_name: str) -> List[str]:
        """Check domain reputation for suspicious indicators"""
        indicators = []
        
        # Simple heuristic checks
        if len(domain_name) > 20:
            indicators.append("long_domain_name")
        
        if any(char.isdigit() for char in domain_name.replace(".", "")):
            indicators.append("contains_numbers")
        
        suspicious_tlds = [".tk", ".ml", ".ga", ".cf"]
        if any(domain_name.endswith(tld) for tld in suspicious_tlds):
            indicators.append("suspicious_tld")
        
        # Check for DGA-like patterns
        vowels = "aeiou"
        consonant_ratio = sum(1 for c in domain_name if c.isalpha() and c.lower() not in vowels) / len(domain_name)
        if consonant_ratio > 0.7:
            indicators.append("high_consonant_ratio")
        
        return indicators
    
    async def _calculate_vulnerability_similarity(self, vulnerabilities: List[GraphNode]) -> float:
        """Calculate similarity between vulnerabilities"""
        if len(vulnerabilities) < 2:
            return 0.0
        
        # Compare properties
        similarity_scores = []
        
        for i in range(len(vulnerabilities)):
            for j in range(i + 1, len(vulnerabilities)):
                vuln_a, vuln_b = vulnerabilities[i], vulnerabilities[j]
                
                # Compare severity
                severity_a = vuln_a.properties.get("severity", "unknown")
                severity_b = vuln_b.properties.get("severity", "unknown")
                severity_match = 1.0 if severity_a == severity_b else 0.0
                
                # Compare CWE
                cwe_a = vuln_a.properties.get("cwe_id", "").split("-")[0]
                cwe_b = vuln_b.properties.get("cwe_id", "").split("-")[0]
                cwe_match = 1.0 if cwe_a == cwe_b and cwe_a else 0.0
                
                # Compare CVSS score
                cvss_a = vuln_a.properties.get("cvss_score", 0)
                cvss_b = vuln_b.properties.get("cvss_score", 0)
                cvss_similarity = 1.0 - abs(cvss_a - cvss_b) / 10.0
                
                similarity = (severity_match * 0.3 + cwe_match * 0.5 + cvss_similarity * 0.2)
                similarity_scores.append(similarity)
        
        return statistics.mean(similarity_scores) if similarity_scores else 0.0
    
    async def _create_behavior_vector(self, host: GraphNode) -> List[float]:
        """Create behavior vector for a host"""
        vector = []
        
        # Connection patterns
        connections = await self.graph_manager.get_neighbors(host.id)
        vector.append(len(connections))  # Connection count
        
        # Service diversity
        services = [n for n in connections if n.type == NodeType.SERVICE]
        vector.append(len(services))
        
        # Vulnerability exposure
        vulnerabilities = [n for n in connections if n.type == NodeType.VULNERABILITY]
        vector.append(len(vulnerabilities))
        
        # Threat associations
        threats = await self.graph_manager.get_neighbors(
            host.id, edge_type=RelationType.TARGETS, direction="incoming"
        )
        vector.append(len(threats))
        
        # Host properties
        vector.append(1.0 if host.properties.get("internet_facing") else 0.0)
        vector.append(hash(host.properties.get("os", "unknown")) % 10)  # OS type encoding
        
        return vector
    
    async def _calculate_behavior_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        """Calculate behavioral similarity between two vectors"""
        if len(vector_a) != len(vector_b):
            return 0.0
        
        try:
            # Use cosine similarity
            similarity_matrix = cosine_similarity([vector_a], [vector_b])
            return float(similarity_matrix[0][0])
        except:
            # Fallback to simple correlation
            if len(vector_a) > 1:
                correlation, _ = pearsonr(vector_a, vector_b)
                return abs(correlation) if not math.isnan(correlation) else 0.0
            return 0.0
    
    async def _create_time_buckets(self, entities: List[GraphNode], time_period_days: int) -> Dict[str, List[GraphNode]]:
        """Create time buckets for trend analysis"""
        buckets = defaultdict(list)
        
        # Create daily buckets
        for entity in entities:
            bucket_key = entity.created_at.strftime("%Y-%m-%d")
            buckets[bucket_key].append(entity)
        
        return dict(buckets)
    
    async def _calculate_trend_metric(self, time_buckets: Dict[str, List[GraphNode]], 
                                    metric: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Calculate trend data for a specific metric"""
        try:
            data_points = []
            
            # Sort buckets by date
            sorted_dates = sorted(time_buckets.keys())
            
            for date_str in sorted_dates:
                entities = time_buckets[date_str]
                
                if metric == "count":
                    value = len(entities)
                elif metric == "severity_distribution":
                    severities = [e.properties.get("severity", "medium") for e in entities]
                    critical_count = severities.count("critical")
                    value = critical_count / len(severities) if severities else 0
                elif metric == "geographic_distribution":
                    # Placeholder for geographic analysis
                    value = len(set(e.properties.get("location", "unknown") for e in entities))
                else:
                    continue
                
                data_points.append({
                    "date": date_str,
                    "value": value,
                    "entity_count": len(entities)
                })
            
            if len(data_points) < 2:
                return None
            
            # Calculate trend direction and change rate
            values = [dp["value"] for dp in data_points]
            
            # Simple linear regression for trend
            if len(values) > 1:
                x = list(range(len(values)))
                correlation, _ = pearsonr(x, values)
                
                if correlation > 0.1:
                    direction = "increasing"
                elif correlation < -0.1:
                    direction = "decreasing"
                else:
                    direction = "stable"
                
                change_rate = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
                significance = abs(correlation)
            else:
                direction = "stable"
                change_rate = 0.0
                significance = 0.0
            
            # Detect anomalies (simple standard deviation approach)
            if len(values) > 3:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                threshold = mean_val + 2 * std_val
                
                anomalies = []
                for i, dp in enumerate(data_points):
                    if dp["value"] > threshold:
                        anomalies.append({
                            "date": dp["date"],
                            "value": dp["value"],
                            "deviation": dp["value"] - mean_val
                        })
            else:
                anomalies = []
            
            # Simple prediction (linear extrapolation)
            if len(values) >= 2:
                trend_slope = (values[-1] - values[-2])
                predicted_value = values[-1] + trend_slope
                predictions = [{
                    "horizon": "next_period",
                    "predicted_value": max(0, predicted_value),
                    "confidence": significance
                }]
            else:
                predictions = []
            
            return {
                "direction": direction,
                "change_rate": change_rate,
                "significance": significance,
                "data_points": data_points,
                "anomalies": anomalies,
                "predictions": predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trend metric {metric}: {e}")
            return None
    
    async def clear_cache(self) -> None:
        """Clear all analytics caches"""
        self.pattern_cache.clear()
        self.correlation_cache.clear()
        self.trend_cache.clear()
        self.risk_cache.clear()
        self.logger.info("Graph analytics cache cleared")
    
    async def export_analytics_report(self, format: str = "json") -> Dict[str, Any]:
        """Export comprehensive analytics report"""
        try:
            # Generate comprehensive analytics
            patterns = await self.analyze_patterns()
            correlations = await self.correlate_intelligence()
            trends = await self.analyze_trends()
            risk_scores = await self.calculate_comprehensive_risk_scores()
            
            # Compile report
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "format": format,
                    "analysis_period": "30_days",
                    "entities_analyzed": len(risk_scores)
                },
                "executive_summary": {
                    "total_patterns": len(patterns),
                    "high_risk_patterns": len([p for p in patterns if p.risk_level in ["critical", "high"]]),
                    "intelligence_correlations": len(correlations),
                    "trend_analyses": len(trends),
                    "overall_risk_level": await self._calculate_overall_risk_level(risk_scores)
                },
                "detailed_analysis": {
                    "security_patterns": [pattern.to_dict() for pattern in patterns],
                    "intelligence_correlations": [correlation.to_dict() for correlation in correlations],
                    "trend_analysis": [trend.to_dict() for trend in trends],
                    "risk_assessment": [risk.to_dict() for risk in risk_scores]
                },
                "recommendations": await self._generate_recommendations(patterns, correlations, risk_scores)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error exporting analytics report: {e}")
            return {"error": str(e)}
    
    async def _calculate_overall_risk_level(self, risk_scores: List[RiskScore]) -> str:
        """Calculate overall risk level from individual scores"""
        if not risk_scores:
            return "unknown"
        
        avg_risk = statistics.mean(score.overall_risk for score in risk_scores)
        
        if avg_risk >= 0.8:
            return "critical"
        elif avg_risk >= 0.6:
            return "high"
        elif avg_risk >= 0.4:
            return "medium"
        else:
            return "low"
    
    async def _generate_recommendations(self, patterns: List[AnalyticsPattern], 
                                      correlations: List[IntelligenceCorrelation],
                                      risk_scores: List[RiskScore]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Pattern-based recommendations
        high_risk_patterns = [p for p in patterns if p.risk_level in ["critical", "high"]]
        if high_risk_patterns:
            recommendations.append({
                "category": "pattern_mitigation",
                "priority": "high",
                "title": "Address High-Risk Security Patterns",
                "description": f"Found {len(high_risk_patterns)} high-risk patterns requiring immediate attention",
                "actions": [
                    "Investigate lateral movement patterns",
                    "Patch privilege escalation vulnerabilities", 
                    "Review external data flows",
                    "Monitor command and control communications"
                ]
            })
        
        # Correlation-based recommendations
        if correlations:
            recommendations.append({
                "category": "intelligence_correlation",
                "priority": "medium",
                "title": "Investigate Correlated Threats",
                "description": f"Found {len(correlations)} intelligence correlations",
                "actions": [
                    "Review temporal attack patterns",
                    "Analyze technical vulnerabilities clusters",
                    "Monitor behavioral anomalies"
                ]
            })
        
        # Risk-based recommendations
        critical_entities = [r for r in risk_scores if r.overall_risk >= 0.8]
        if critical_entities:
            recommendations.append({
                "category": "risk_mitigation",
                "priority": "critical",
                "title": "Secure Critical Risk Entities",
                "description": f"Found {len(critical_entities)} entities with critical risk levels",
                "actions": list(set(
                    action for entity in critical_entities 
                    for action in entity.mitigation_recommendations
                ))
            })
        
        return recommendations