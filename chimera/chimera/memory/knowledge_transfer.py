"""
Cross-Domain Knowledge Transfer Engine
Intelligently transfers successful patterns between different domains and contexts
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import re

from ..core.events import EventSystem, EventType, EventEmitter
from .knowledge_graph_lite import LiteKnowledgeGraph

logger = logging.getLogger(__name__)

@dataclass
class KnowledgePattern:
    """A transferable knowledge pattern"""
    
    pattern_id: str
    source_domain: str
    pattern_type: str  # technique, approach, configuration, etc.
    abstract_description: str  # Domain-agnostic description
    concrete_examples: List[Dict[str, Any]]  # Domain-specific examples
    success_metrics: Dict[str, float]  # Metrics showing success
    applicability_conditions: List[str]  # When this pattern applies
    abstraction_level: float  # How abstract/generalizable (0.0-1.0)
    transfer_success_rate: float  # Success rate when transferred
    creation_date: datetime
    last_transfer: Optional[datetime] = None

@dataclass
class TransferCandidate:
    """A candidate for knowledge transfer"""
    
    pattern: KnowledgePattern
    target_domain: str
    similarity_score: float  # How similar target domain is to source
    adaptation_complexity: float  # How complex adaptation will be
    expected_benefit: float  # Expected performance improvement
    confidence: float  # Confidence in successful transfer
    proposed_adaptations: List[str]  # Specific adaptations needed

@dataclass
class TransferResult:
    """Result of a knowledge transfer attempt"""
    
    transfer_id: str
    source_pattern: str
    target_domain: str
    adaptations_made: List[str]
    implementation_details: Dict[str, Any]
    success: bool
    performance_improvement: float
    lessons_learned: List[str]
    timestamp: datetime

class KnowledgeTransferEngine(EventEmitter):
    """
    Cross-Domain Knowledge Transfer Engine
    
    Features:
    - Pattern abstraction from successful techniques
    - Domain similarity analysis
    - Intelligent adaptation strategies
    - Transfer success prediction
    - Automated transfer execution
    - Continuous learning from transfer outcomes
    """
    
    def __init__(self, config, event_system: EventSystem, knowledge_graph: LiteKnowledgeGraph):
        super().__init__(event_system, "KnowledgeTransferEngine")
        
        self.config = config
        self.knowledge_graph = knowledge_graph
        
        # Pattern storage and analysis
        self.patterns: Dict[str, KnowledgePattern] = {}
        self.domain_characteristics: Dict[str, Dict[str, Any]] = {}
        self.transfer_history: List[TransferResult] = []
        
        # Transfer parameters
        self.min_pattern_success_rate = config.get("transfer.min_success_rate", 0.7)
        self.similarity_threshold = config.get("transfer.similarity_threshold", 0.6)
        self.max_adaptation_complexity = config.get("transfer.max_complexity", 0.8)
        
        # Learning state
        self.domain_relationships: Dict[Tuple[str, str], float] = {}
        self.adaptation_strategies: Dict[str, List[str]] = {}
        self.transfer_effectiveness: Dict[str, List[float]] = defaultdict(list)
        
    async def initialize(self):
        """Initialize knowledge transfer engine"""
        
        await self._initialize_domain_characteristics()
        await self._load_base_patterns()
        
        # Start background analysis
        asyncio.create_task(self._pattern_discovery_loop())
        asyncio.create_task(self._transfer_optimization_loop())
        
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {"message": "Knowledge transfer engine initialized", "patterns": len(self.patterns)}
        )
        
        logger.info("Knowledge transfer engine initialized")
        
    async def discover_transferable_patterns(self, source_domain: str, 
                                           min_success_rate: float = None) -> List[KnowledgePattern]:
        """
        Discover transferable patterns from a source domain
        
        Args:
            source_domain: Domain to extract patterns from
            min_success_rate: Minimum success rate to consider
            
        Returns:
            List of discovered patterns
        """
        
        min_success_rate = min_success_rate or self.min_pattern_success_rate
        
        # Query knowledge graph for successful techniques in domain
        domain_knowledge = await self.knowledge_graph.query_knowledge(
            f"{source_domain} successful technique", limit=50
        )
        
        discovered_patterns = []
        
        # Group related knowledge nodes
        technique_groups = self._group_related_techniques(domain_knowledge)
        
        for group_id, techniques in technique_groups.items():
            if len(techniques) < 3:  # Need multiple examples
                continue
                
            # Calculate success metrics for this group
            success_metrics = await self._calculate_group_success_metrics(techniques)
            
            if success_metrics.get("success_rate", 0.0) < min_success_rate:
                continue
                
            # Abstract the pattern
            pattern = await self._abstract_pattern(source_domain, group_id, techniques, success_metrics)
            
            if pattern:
                discovered_patterns.append(pattern)
                self.patterns[pattern.pattern_id] = pattern
                
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {"action": "patterns_discovered", "domain": source_domain, "count": len(discovered_patterns)}
        )
        
        return discovered_patterns
        
    async def find_transfer_candidates(self, target_domain: str, 
                                     performance_threshold: float = 0.6) -> List[TransferCandidate]:
        """
        Find promising candidates for transferring knowledge to target domain
        
        Args:
            target_domain: Domain to transfer knowledge to
            performance_threshold: Minimum expected performance improvement
            
        Returns:
            List of transfer candidates sorted by expected benefit
        """
        
        candidates = []
        
        # Analyze target domain characteristics
        target_characteristics = await self._analyze_domain_characteristics(target_domain)
        
        # Evaluate each pattern for transfer potential
        for pattern in self.patterns.values():
            if pattern.source_domain == target_domain:
                continue  # Skip same-domain transfers
                
            # Calculate similarity between source and target domains
            similarity = await self._calculate_domain_similarity(
                pattern.source_domain, target_domain, target_characteristics
            )
            
            if similarity < self.similarity_threshold:
                continue
                
            # Estimate adaptation complexity
            adaptation_complexity = await self._estimate_adaptation_complexity(
                pattern, target_domain, target_characteristics
            )
            
            if adaptation_complexity > self.max_adaptation_complexity:
                continue
                
            # Estimate expected benefit
            expected_benefit = await self._estimate_transfer_benefit(
                pattern, target_domain, similarity, adaptation_complexity
            )
            
            if expected_benefit < performance_threshold:
                continue
                
            # Generate specific adaptation proposals
            adaptations = await self._propose_adaptations(pattern, target_domain, target_characteristics)
            
            # Calculate transfer confidence
            confidence = await self._calculate_transfer_confidence(
                pattern, target_domain, similarity, adaptation_complexity, expected_benefit
            )
            
            candidate = TransferCandidate(
                pattern=pattern,
                target_domain=target_domain,
                similarity_score=similarity,
                adaptation_complexity=adaptation_complexity,
                expected_benefit=expected_benefit,
                confidence=confidence,
                proposed_adaptations=adaptations
            )
            
            candidates.append(candidate)
            
        # Sort by expected benefit and confidence
        candidates.sort(key=lambda c: c.expected_benefit * c.confidence, reverse=True)
        
        return candidates
        
    async def execute_knowledge_transfer(self, candidate: TransferCandidate) -> TransferResult:
        """
        Execute a knowledge transfer
        
        Args:
            candidate: Transfer candidate to execute
            
        Returns:
            Transfer result with success metrics
        """
        
        transfer_id = f"transfer_{candidate.pattern.pattern_id}_{candidate.target_domain}_{int(datetime.utcnow().timestamp())}"
        
        try:
            # Apply adaptations to create target-domain version
            adapted_knowledge = await self._adapt_pattern_to_domain(
                candidate.pattern, candidate.target_domain, candidate.proposed_adaptations
            )
            
            # Create knowledge nodes in target domain
            transfer_nodes = []
            for adaptation in adapted_knowledge:
                node_id = await self.knowledge_graph.add_knowledge(
                    content=adaptation["content"],
                    node_type=adaptation["type"],
                    source=f"knowledge_transfer_{transfer_id}",
                    metadata={
                        "source_pattern": candidate.pattern.pattern_id,
                        "source_domain": candidate.pattern.source_domain,
                        "target_domain": candidate.target_domain,
                        "transfer_id": transfer_id,
                        "adaptation_method": adaptation.get("method", "unknown")
                    }
                )
                transfer_nodes.append(node_id)
                
            # Create connections between transferred knowledge
            await self._create_transfer_connections(transfer_nodes, candidate.pattern)
            
            # Mark pattern as transferred
            candidate.pattern.last_transfer = datetime.utcnow()
            
            # Create transfer result
            result = TransferResult(
                transfer_id=transfer_id,
                source_pattern=candidate.pattern.pattern_id,
                target_domain=candidate.target_domain,
                adaptations_made=candidate.proposed_adaptations,
                implementation_details={
                    "nodes_created": len(transfer_nodes),
                    "adaptations_applied": len(adapted_knowledge),
                    "similarity_score": candidate.similarity_score,
                    "expected_benefit": candidate.expected_benefit
                },
                success=True,
                performance_improvement=0.0,  # Will be updated when measured
                lessons_learned=[],
                timestamp=datetime.utcnow()
            )
            
            self.transfer_history.append(result)
            
            await self.emit_event(
                EventType.KNOWLEDGE_LEARNED,
                {
                    "action": "knowledge_transferred",
                    "transfer_id": transfer_id,
                    "source_domain": candidate.pattern.source_domain,
                    "target_domain": candidate.target_domain
                }
            )
            
            logger.info(f"Knowledge transfer completed: {transfer_id}")
            
            return result
            
        except Exception as e:
            # Transfer failed
            result = TransferResult(
                transfer_id=transfer_id,
                source_pattern=candidate.pattern.pattern_id,
                target_domain=candidate.target_domain,
                adaptations_made=[],
                implementation_details={"error": str(e)},
                success=False,
                performance_improvement=0.0,
                lessons_learned=[f"Transfer failed: {str(e)}"],
                timestamp=datetime.utcnow()
            )
            
            self.transfer_history.append(result)
            
            logger.error(f"Knowledge transfer failed: {transfer_id} - {e}")
            
            return result
            
    async def evaluate_transfer_success(self, transfer_id: str, 
                                      performance_metrics: Dict[str, float]) -> bool:
        """
        Evaluate the success of a knowledge transfer
        
        Args:
            transfer_id: ID of the transfer to evaluate
            performance_metrics: Measured performance metrics
            
        Returns:
            True if transfer was successful
        """
        
        # Find transfer result
        transfer_result = None
        for result in self.transfer_history:
            if result.transfer_id == transfer_id:
                transfer_result = result
                break
                
        if not transfer_result:
            logger.warning(f"Transfer result not found: {transfer_id}")
            return False
            
        # Calculate performance improvement
        baseline_performance = performance_metrics.get("baseline", 0.5)
        new_performance = performance_metrics.get("current", 0.5)
        improvement = new_performance - baseline_performance
        
        transfer_result.performance_improvement = improvement
        
        # Determine success
        success = improvement > 0.1  # 10% improvement threshold
        
        # Update pattern transfer success rate
        pattern_id = transfer_result.source_pattern
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            
            # Update transfer success rate with exponential moving average
            alpha = 0.2
            success_value = 1.0 if success else 0.0
            pattern.transfer_success_rate = (
                (1 - alpha) * pattern.transfer_success_rate + alpha * success_value
            )
            
        # Learn from the outcome
        await self._learn_from_transfer_outcome(transfer_result, success, performance_metrics)
        
        # Update effectiveness tracking
        domain_pair = (transfer_result.source_pattern.split("_")[0], transfer_result.target_domain)
        self.transfer_effectiveness[str(domain_pair)].append(improvement)
        
        await self.emit_event(
            EventType.KNOWLEDGE_UPDATED,
            {
                "transfer_id": transfer_id,
                "success": success,
                "improvement": improvement
            }
        )
        
        return success
        
    async def get_transfer_analytics(self) -> Dict[str, Any]:
        """Get analytics about knowledge transfer performance"""
        
        if not self.transfer_history:
            return {"message": "No transfer history available"}
            
        # Overall statistics
        total_transfers = len(self.transfer_history)
        successful_transfers = len([r for r in self.transfer_history if r.success])
        success_rate = successful_transfers / total_transfers
        
        # Performance improvements
        improvements = [r.performance_improvement for r in self.transfer_history if r.success]
        avg_improvement = statistics.mean(improvements) if improvements else 0.0
        
        # Domain analysis
        source_domains = defaultdict(int)
        target_domains = defaultdict(int)
        domain_pairs = defaultdict(list)
        
        for result in self.transfer_history:
            source_domains[result.source_pattern.split("_")[0]] += 1
            target_domains[result.target_domain] += 1
            domain_pairs[(result.source_pattern.split("_")[0], result.target_domain)].append(
                result.performance_improvement
            )
            
        # Best performing domain pairs
        best_pairs = {}
        for (source, target), improvements in domain_pairs.items():
            if improvements:
                avg_improvement = statistics.mean(improvements)
                best_pairs[f"{source} -> {target}"] = {
                    "avg_improvement": avg_improvement,
                    "transfer_count": len(improvements),
                    "success_rate": len([i for i in improvements if i > 0.1]) / len(improvements)
                }
                
        # Pattern effectiveness
        pattern_stats = {}
        for pattern in self.patterns.values():
            pattern_transfers = [r for r in self.transfer_history 
                               if r.source_pattern == pattern.pattern_id]
            
            if pattern_transfers:
                pattern_stats[pattern.pattern_id] = {
                    "transfer_count": len(pattern_transfers),
                    "success_rate": len([r for r in pattern_transfers if r.success]) / len(pattern_transfers),
                    "avg_improvement": statistics.mean([r.performance_improvement for r in pattern_transfers]),
                    "abstraction_level": pattern.abstraction_level
                }
                
        return {
            "overall_statistics": {
                "total_transfers": total_transfers,
                "success_rate": success_rate,
                "average_improvement": avg_improvement,
                "total_patterns": len(self.patterns)
            },
            "domain_analysis": {
                "source_domains": dict(source_domains),
                "target_domains": dict(target_domains),
                "best_domain_pairs": dict(sorted(best_pairs.items(), 
                                               key=lambda x: x[1]["avg_improvement"], reverse=True)[:5])
            },
            "pattern_effectiveness": dict(sorted(pattern_stats.items(), 
                                               key=lambda x: x[1]["avg_improvement"], reverse=True)[:10]),
            "recent_transfers": len([r for r in self.transfer_history 
                                   if r.timestamp > datetime.utcnow() - timedelta(days=7)])
        }
        
    # Private methods
    
    async def _initialize_domain_characteristics(self):
        """Initialize characteristics of different domains"""
        
        self.domain_characteristics = {
            "web_application": {
                "techniques": ["injection_testing", "authentication_bypass", "session_manipulation"],
                "tools": ["burp_suite", "zap", "sqlmap"],
                "complexity": 0.7,
                "detection_likelihood": 0.6,
                "common_vulnerabilities": ["xss", "sql_injection", "csrf"]
            },
            "network_infrastructure": {
                "techniques": ["port_scanning", "service_enumeration", "protocol_analysis"],
                "tools": ["nmap", "nessus", "wireshark"],
                "complexity": 0.8,
                "detection_likelihood": 0.8,
                "common_vulnerabilities": ["misconfiguration", "weak_authentication", "unpatched_services"]
            },
            "api_security": {
                "techniques": ["endpoint_enumeration", "parameter_manipulation", "rate_limiting_bypass"],
                "tools": ["postman", "burp_suite", "custom_scripts"],
                "complexity": 0.6,
                "detection_likelihood": 0.5,
                "common_vulnerabilities": ["broken_authentication", "excessive_data_exposure", "injection"]
            },
            "mobile_application": {
                "techniques": ["static_analysis", "dynamic_analysis", "runtime_manipulation"],
                "tools": ["mobsf", "frida", "jadx"],
                "complexity": 0.9,
                "detection_likelihood": 0.4,
                "common_vulnerabilities": ["insecure_storage", "weak_crypto", "platform_misuse"]
            },
            "cloud_infrastructure": {
                "techniques": ["misconfiguration_analysis", "privilege_escalation", "lateral_movement"],
                "tools": ["scout_suite", "aws_cli", "azure_cli"],
                "complexity": 0.8,
                "detection_likelihood": 0.7,
                "common_vulnerabilities": ["iam_misconfig", "exposed_storage", "network_security_groups"]
            }
        }
        
    async def _load_base_patterns(self):
        """Load base transferable patterns"""
        
        # Example base patterns
        base_patterns = [
            {
                "pattern_id": "enumeration_pattern_001",
                "source_domain": "web_application",
                "pattern_type": "enumeration_technique",
                "abstract_description": "Systematic enumeration of endpoints using wordlists and pattern recognition",
                "concrete_examples": [
                    {"technique": "directory_bruteforcing", "tool": "gobuster", "wordlist": "common.txt"},
                    {"technique": "parameter_discovery", "tool": "ffuf", "wordlist": "parameters.txt"}
                ],
                "success_metrics": {"success_rate": 0.8, "coverage": 0.7},
                "applicability_conditions": ["web_interface_available", "http_service_detected"],
                "abstraction_level": 0.7
            },
            {
                "pattern_id": "credential_testing_pattern_001", 
                "source_domain": "network_infrastructure",
                "pattern_type": "authentication_testing",
                "abstract_description": "Systematic testing of default and weak credentials across services",
                "concrete_examples": [
                    {"technique": "default_credential_testing", "wordlist": "default_creds.txt"},
                    {"technique": "weak_password_testing", "wordlist": "common_passwords.txt"}
                ],
                "success_metrics": {"success_rate": 0.6, "time_efficiency": 0.8},
                "applicability_conditions": ["authentication_required", "multiple_services"],
                "abstraction_level": 0.8
            }
        ]
        
        for pattern_data in base_patterns:
            pattern = KnowledgePattern(
                pattern_id=pattern_data["pattern_id"],
                source_domain=pattern_data["source_domain"],
                pattern_type=pattern_data["pattern_type"],
                abstract_description=pattern_data["abstract_description"],
                concrete_examples=pattern_data["concrete_examples"],
                success_metrics=pattern_data["success_metrics"],
                applicability_conditions=pattern_data["applicability_conditions"],
                abstraction_level=pattern_data["abstraction_level"],
                transfer_success_rate=0.5,  # Neutral starting point
                creation_date=datetime.utcnow()
            )
            
            self.patterns[pattern.pattern_id] = pattern
            
    def _group_related_techniques(self, knowledge_nodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group related techniques from knowledge nodes"""
        
        groups = defaultdict(list)
        
        for node in knowledge_nodes:
            content = node.get("content", "").lower()
            
            # Simple grouping by technique type
            if any(word in content for word in ["scan", "enumerate", "discover"]):
                groups["enumeration_techniques"].append(node)
            elif any(word in content for word in ["inject", "payload", "exploit"]):
                groups["exploitation_techniques"].append(node)
            elif any(word in content for word in ["auth", "login", "credential"]):
                groups["authentication_techniques"].append(node)
            elif any(word in content for word in ["config", "setting", "parameter"]):
                groups["configuration_techniques"].append(node)
            else:
                groups["general_techniques"].append(node)
                
        return groups
        
    async def _calculate_group_success_metrics(self, techniques: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate success metrics for a group of techniques"""
        
        # This would analyze actual success data from knowledge graph
        # For now, simulate based on confidence scores
        
        confidences = [tech.get("confidence", 0.5) for tech in techniques]
        
        if confidences:
            success_rate = statistics.mean(confidences)
            coverage = min(len(techniques) / 10.0, 1.0)  # More techniques = better coverage
            efficiency = success_rate * coverage
        else:
            success_rate = coverage = efficiency = 0.5
            
        return {
            "success_rate": success_rate,
            "coverage": coverage,
            "efficiency": efficiency,
            "sample_size": len(techniques)
        }
        
    async def _abstract_pattern(self, source_domain: str, group_id: str, 
                              techniques: List[Dict[str, Any]], 
                              success_metrics: Dict[str, float]) -> Optional[KnowledgePattern]:
        """Abstract a pattern from a group of techniques"""
        
        if len(techniques) < 3:
            return None
            
        # Generate abstract description
        common_words = self._extract_common_concepts(techniques)
        abstract_description = f"Systematic approach using {', '.join(common_words[:3])} techniques"
        
        # Extract concrete examples
        concrete_examples = []
        for tech in techniques[:5]:  # Top 5 examples
            example = {
                "technique": tech.get("type", "unknown"),
                "description": tech.get("content", "")[:100],
                "confidence": tech.get("confidence", 0.5)
            }
            concrete_examples.append(example)
            
        # Generate applicability conditions
        conditions = self._derive_applicability_conditions(techniques, source_domain)
        
        # Calculate abstraction level
        abstraction_level = self._calculate_abstraction_level(techniques, source_domain)
        
        pattern = KnowledgePattern(
            pattern_id=f"{source_domain}_{group_id}_{int(datetime.utcnow().timestamp())}",
            source_domain=source_domain,
            pattern_type=group_id,
            abstract_description=abstract_description,
            concrete_examples=concrete_examples,
            success_metrics=success_metrics,
            applicability_conditions=conditions,
            abstraction_level=abstraction_level,
            transfer_success_rate=0.5,
            creation_date=datetime.utcnow()
        )
        
        return pattern
        
    def _extract_common_concepts(self, techniques: List[Dict[str, Any]]) -> List[str]:
        """Extract common concepts from techniques"""
        
        all_words = []
        for tech in techniques:
            content = tech.get("content", "")
            words = re.findall(r'\b\w+\b', content.lower())
            all_words.extend(words)
            
        # Count word frequency
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Filter out common words and get most frequent
        common_words = ["the", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by"]
        filtered_counts = {word: count for word, count in word_counts.items() 
                          if word not in common_words and len(word) > 3}
        
        return [word for word, count in sorted(filtered_counts.items(), 
                                             key=lambda x: x[1], reverse=True)[:10]]
        
    def _derive_applicability_conditions(self, techniques: List[Dict[str, Any]], 
                                       source_domain: str) -> List[str]:
        """Derive conditions when this pattern applies"""
        
        conditions = []
        
        # Domain-specific conditions
        domain_conditions = {
            "web_application": ["http_service_available", "web_interface_detected"],
            "network_infrastructure": ["network_access_available", "multiple_services_detected"],
            "api_security": ["api_endpoints_discovered", "authentication_mechanism_identified"],
            "mobile_application": ["application_binary_available", "runtime_environment_accessible"],
            "cloud_infrastructure": ["cloud_environment_access", "iam_permissions_available"]
        }
        
        conditions.extend(domain_conditions.get(source_domain, ["target_accessible"]))
        
        # Technique-specific conditions
        content_analysis = " ".join([tech.get("content", "") for tech in techniques]).lower()
        
        if "authentication" in content_analysis:
            conditions.append("authentication_mechanism_present")
        if "parameter" in content_analysis:
            conditions.append("parameterized_interface_available")
        if "file" in content_analysis or "directory" in content_analysis:
            conditions.append("file_system_accessible")
            
        return conditions
        
    def _calculate_abstraction_level(self, techniques: List[Dict[str, Any]], 
                                   source_domain: str) -> float:
        """Calculate how abstract/generalizable a pattern is"""
        
        # Factors that increase abstraction:
        # - More diverse techniques
        # - Less domain-specific terminology
        # - More general principles
        
        diversity_score = min(len(techniques) / 10.0, 1.0)
        
        # Analyze content for domain-specific terms
        all_content = " ".join([tech.get("content", "") for tech in techniques]).lower()
        
        domain_specific_terms = {
            "web_application": ["html", "javascript", "cookie", "session", "browser"],
            "network_infrastructure": ["port", "protocol", "packet", "socket", "tcp"],
            "api_security": ["endpoint", "json", "rest", "graphql", "oauth"],
            "mobile_application": ["android", "ios", "apk", "ipa", "mobile"],
            "cloud_infrastructure": ["aws", "azure", "gcp", "container", "serverless"]
        }
        
        domain_terms = domain_specific_terms.get(source_domain, [])
        domain_term_count = sum(1 for term in domain_terms if term in all_content)
        domain_specificity = min(domain_term_count / len(domain_terms), 1.0)
        
        # Higher abstraction = lower domain specificity + higher diversity
        abstraction_level = (diversity_score * 0.6) + ((1.0 - domain_specificity) * 0.4)
        
        return min(abstraction_level, 1.0)
        
    async def _analyze_domain_characteristics(self, domain: str) -> Dict[str, Any]:
        """Analyze characteristics of a domain"""
        
        if domain in self.domain_characteristics:
            return self.domain_characteristics[domain].copy()
            
        # If domain not known, derive characteristics from knowledge graph
        domain_knowledge = await self.knowledge_graph.query_knowledge(
            f"{domain} technique tool", limit=20
        )
        
        if not domain_knowledge:
            # Default characteristics
            return {
                "techniques": [],
                "tools": [],
                "complexity": 0.5,
                "detection_likelihood": 0.5,
                "common_vulnerabilities": []
            }
            
        # Extract characteristics from knowledge
        techniques = []
        tools = []
        
        for knowledge in domain_knowledge:
            content = knowledge.get("content", "").lower()
            
            # Extract techniques (words ending in common technique suffixes)
            technique_words = re.findall(r'\b\w*(?:ing|tion|ment|ysis)\b', content)
            techniques.extend(technique_words)
            
            # Extract tools (common security tool names)
            tool_indicators = ["nmap", "burp", "metasploit", "wireshark", "sqlmap", "gobuster"]
            for tool in tool_indicators:
                if tool in content:
                    tools.append(tool)
                    
        characteristics = {
            "techniques": list(set(techniques))[:10],
            "tools": list(set(tools))[:10],
            "complexity": 0.6,  # Default medium complexity
            "detection_likelihood": 0.5,  # Default medium detection
            "common_vulnerabilities": []
        }
        
        # Cache for future use
        self.domain_characteristics[domain] = characteristics
        
        return characteristics
        
    async def _calculate_domain_similarity(self, source_domain: str, target_domain: str,
                                         target_characteristics: Dict[str, Any]) -> float:
        """Calculate similarity between two domains"""
        
        if source_domain == target_domain:
            return 1.0
            
        source_chars = self.domain_characteristics.get(source_domain, {})
        
        # Check if we have a cached relationship
        domain_pair = (source_domain, target_domain)
        if domain_pair in self.domain_relationships:
            return self.domain_relationships[domain_pair]
            
        similarity_factors = []
        
        # Tool overlap
        source_tools = set(source_chars.get("tools", []))
        target_tools = set(target_characteristics.get("tools", []))
        if source_tools or target_tools:
            tool_similarity = len(source_tools & target_tools) / len(source_tools | target_tools)
            similarity_factors.append(tool_similarity * 0.3)
            
        # Technique overlap
        source_techniques = set(source_chars.get("techniques", []))
        target_techniques = set(target_characteristics.get("techniques", []))
        if source_techniques or target_techniques:
            technique_similarity = len(source_techniques & target_techniques) / len(source_techniques | target_techniques)
            similarity_factors.append(technique_similarity * 0.4)
            
        # Complexity similarity
        source_complexity = source_chars.get("complexity", 0.5)
        target_complexity = target_characteristics.get("complexity", 0.5)
        complexity_similarity = 1.0 - abs(source_complexity - target_complexity)
        similarity_factors.append(complexity_similarity * 0.2)
        
        # Detection likelihood similarity
        source_detection = source_chars.get("detection_likelihood", 0.5)
        target_detection = target_characteristics.get("detection_likelihood", 0.5)
        detection_similarity = 1.0 - abs(source_detection - target_detection)
        similarity_factors.append(detection_similarity * 0.1)
        
        # Calculate overall similarity
        overall_similarity = sum(similarity_factors) if similarity_factors else 0.5
        
        # Cache the result
        self.domain_relationships[domain_pair] = overall_similarity
        
        return overall_similarity
        
    async def _estimate_adaptation_complexity(self, pattern: KnowledgePattern, 
                                            target_domain: str, 
                                            target_characteristics: Dict[str, Any]) -> float:
        """Estimate complexity of adapting pattern to target domain"""
        
        complexity_factors = []
        
        # Abstraction level - higher abstraction = lower adaptation complexity
        abstraction_factor = 1.0 - pattern.abstraction_level
        complexity_factors.append(abstraction_factor * 0.4)
        
        # Tool compatibility
        pattern_tools = []
        for example in pattern.concrete_examples:
            if "tool" in example:
                pattern_tools.append(example["tool"])
                
        target_tools = target_characteristics.get("tools", [])
        tool_compatibility = 0.0
        if pattern_tools and target_tools:
            common_tools = set(pattern_tools) & set(target_tools)
            tool_compatibility = len(common_tools) / len(set(pattern_tools))
            
        tool_complexity = 1.0 - tool_compatibility
        complexity_factors.append(tool_complexity * 0.3)
        
        # Applicability conditions
        condition_complexity = len(pattern.applicability_conditions) / 10.0  # More conditions = more complex
        complexity_factors.append(min(condition_complexity, 1.0) * 0.2)
        
        # Domain complexity difference
        source_complexity = self.domain_characteristics.get(pattern.source_domain, {}).get("complexity", 0.5)
        target_complexity = target_characteristics.get("complexity", 0.5)
        complexity_diff = abs(target_complexity - source_complexity)
        complexity_factors.append(complexity_diff * 0.1)
        
        overall_complexity = sum(complexity_factors)
        
        return min(overall_complexity, 1.0)
        
    async def _estimate_transfer_benefit(self, pattern: KnowledgePattern, target_domain: str,
                                       similarity: float, adaptation_complexity: float) -> float:
        """Estimate expected benefit of transferring pattern"""
        
        # Base benefit from pattern's historical success
        base_benefit = pattern.success_metrics.get("success_rate", 0.5)
        
        # Adjust for similarity - more similar domains = higher benefit
        similarity_adjustment = similarity * 0.7 + 0.3  # Scale to 0.3-1.0
        
        # Adjust for adaptation complexity - more complex = lower benefit
        complexity_penalty = 1.0 - (adaptation_complexity * 0.5)
        
        # Adjust for pattern's transfer history
        transfer_history_bonus = pattern.transfer_success_rate * 0.2
        
        # Check if target domain lacks good patterns (opportunity factor)
        target_patterns = [p for p in self.patterns.values() if p.source_domain == target_domain]
        if not target_patterns:
            opportunity_bonus = 0.3  # High opportunity in underserved domain
        else:
            target_success_rates = [p.success_metrics.get("success_rate", 0.5) for p in target_patterns]
            avg_target_success = statistics.mean(target_success_rates)
            opportunity_bonus = max(0, (base_benefit - avg_target_success) * 0.2)
            
        expected_benefit = (
            base_benefit * similarity_adjustment * complexity_penalty + 
            transfer_history_bonus + opportunity_bonus
        )
        
        return min(expected_benefit, 1.0)
        
    async def _calculate_transfer_confidence(self, pattern: KnowledgePattern, target_domain: str,
                                           similarity: float, adaptation_complexity: float,
                                           expected_benefit: float) -> float:
        """Calculate confidence in transfer success"""
        
        confidence_factors = []
        
        # Pattern maturity (more examples = higher confidence)
        example_count_factor = min(len(pattern.concrete_examples) / 5.0, 1.0)
        confidence_factors.append(example_count_factor * 0.25)
        
        # Historical transfer success
        transfer_success_factor = pattern.transfer_success_rate
        confidence_factors.append(transfer_success_factor * 0.3)
        
        # Domain similarity
        confidence_factors.append(similarity * 0.25)
        
        # Adaptation simplicity
        adaptation_confidence = 1.0 - adaptation_complexity
        confidence_factors.append(adaptation_confidence * 0.2)
        
        overall_confidence = sum(confidence_factors)
        
        return min(overall_confidence, 1.0)
        
    async def _propose_adaptations(self, pattern: KnowledgePattern, target_domain: str,
                                 target_characteristics: Dict[str, Any]) -> List[str]:
        """Propose specific adaptations for target domain"""
        
        adaptations = []
        
        # Tool substitutions
        pattern_tools = set()
        for example in pattern.concrete_examples:
            if "tool" in example:
                pattern_tools.add(example["tool"])
                
        target_tools = set(target_characteristics.get("tools", []))
        
        # Find tool substitutions
        tool_mappings = {
            "gobuster": ["dirb", "ffuf", "wfuzz"],
            "burp_suite": ["zap", "custom_scripts"],
            "nmap": ["masscan", "zmap"],
            "sqlmap": ["custom_injection_scripts"],
            "metasploit": ["custom_exploits"]
        }
        
        for pattern_tool in pattern_tools:
            if pattern_tool not in target_tools:
                alternatives = tool_mappings.get(pattern_tool, [])
                available_alternatives = [alt for alt in alternatives if alt in target_tools]
                if available_alternatives:
                    adaptations.append(f"replace_{pattern_tool}_with_{available_alternatives[0]}")
                    
        # Technique adaptations
        source_complexity = self.domain_characteristics.get(pattern.source_domain, {}).get("complexity", 0.5)
        target_complexity = target_characteristics.get("complexity", 0.5)
        
        if target_complexity > source_complexity + 0.2:
            adaptations.append("increase_technique_sophistication")
        elif target_complexity < source_complexity - 0.2:
            adaptations.append("simplify_approach")
            
        # Detection adaptations
        source_detection = self.domain_characteristics.get(pattern.source_domain, {}).get("detection_likelihood", 0.5)
        target_detection = target_characteristics.get("detection_likelihood", 0.5)
        
        if target_detection > source_detection + 0.2:
            adaptations.append("enhance_stealth_measures")
            
        # Domain-specific adaptations
        domain_adaptations = {
            "api_security": ["add_api_specific_headers", "implement_rate_limiting_awareness"],
            "mobile_application": ["adapt_for_mobile_constraints", "add_platform_specific_checks"],
            "cloud_infrastructure": ["add_cloud_service_enumeration", "implement_iam_awareness"]
        }
        
        if target_domain in domain_adaptations:
            adaptations.extend(domain_adaptations[target_domain])
            
        return adaptations[:5]  # Limit to top 5 adaptations
        
    async def _adapt_pattern_to_domain(self, pattern: KnowledgePattern, target_domain: str,
                                     adaptations: List[str]) -> List[Dict[str, Any]]:
        """Adapt pattern to target domain"""
        
        adapted_knowledge = []
        
        # Adapt abstract description
        adapted_description = pattern.abstract_description
        for adaptation in adaptations:
            if "replace_" in adaptation:
                # Tool replacement
                parts = adaptation.split("_")
                if len(parts) >= 4:
                    old_tool = parts[1]
                    new_tool = parts[3]
                    adapted_description = adapted_description.replace(old_tool, new_tool)
                    
        adapted_knowledge.append({
            "content": f"Adapted for {target_domain}: {adapted_description}",
            "type": "adapted_technique",
            "method": "description_adaptation"
        })
        
        # Adapt concrete examples
        for i, example in enumerate(pattern.concrete_examples):
            adapted_example = example.copy()
            
            # Apply tool substitutions
            for adaptation in adaptations:
                if "replace_" in adaptation and "tool" in example:
                    parts = adaptation.split("_")
                    if len(parts) >= 4:
                        old_tool = parts[1]
                        new_tool = parts[3]
                        if example["tool"] == old_tool:
                            adapted_example["tool"] = new_tool
                            
            # Add domain-specific enhancements
            if "enhance_stealth_measures" in adaptations:
                adapted_example["stealth_enhancements"] = ["rate_limiting", "user_agent_rotation"]
                
            if "api_specific_headers" in adaptations:
                adapted_example["required_headers"] = ["Content-Type: application/json"]
                
            adapted_knowledge.append({
                "content": f"Adapted example {i+1}: {json.dumps(adapted_example)}",
                "type": "adapted_example",
                "method": "example_adaptation"
            })
            
        # Add domain-specific guidance
        guidance_content = f"Domain-specific guidance for {target_domain}: "
        
        if "api_security" in target_domain:
            guidance_content += "Focus on API endpoints, authentication mechanisms, and data exposure."
        elif "mobile" in target_domain:
            guidance_content += "Consider platform constraints, app sandboxing, and runtime protection."
        elif "cloud" in target_domain:
            guidance_content += "Account for cloud-specific services, IAM policies, and network configurations."
        else:
            guidance_content += "Apply general security testing principles with domain awareness."
            
        adapted_knowledge.append({
            "content": guidance_content,
            "type": "domain_guidance", 
            "method": "domain_specific_guidance"
        })
        
        return adapted_knowledge
        
    async def _create_transfer_connections(self, transfer_nodes: List[str], source_pattern: KnowledgePattern):
        """Create connections between transferred knowledge nodes"""
        
        # Connect all transferred nodes to each other
        for i, node1 in enumerate(transfer_nodes):
            for node2 in transfer_nodes[i+1:]:
                # This would use the knowledge graph's relationship creation
                # For now, just log the intention
                logger.debug(f"Would create relationship between {node1} and {node2}")
                
    async def _learn_from_transfer_outcome(self, transfer_result: TransferResult, 
                                         success: bool, metrics: Dict[str, float]):
        """Learn from transfer outcome to improve future transfers"""
        
        # Update adaptation strategy effectiveness
        for adaptation in transfer_result.adaptations_made:
            if adaptation not in self.adaptation_strategies:
                self.adaptation_strategies[adaptation] = []
                
            self.adaptation_strategies[adaptation].append(transfer_result.performance_improvement)
            
            # Limit history
            if len(self.adaptation_strategies[adaptation]) > 20:
                self.adaptation_strategies[adaptation] = self.adaptation_strategies[adaptation][-10:]
                
        # Update domain relationship understanding
        source_domain = transfer_result.source_pattern.split("_")[0]
        target_domain = transfer_result.target_domain
        domain_pair = (source_domain, target_domain)
        
        if success:
            # Strengthen domain relationship
            current_similarity = self.domain_relationships.get(domain_pair, 0.5)
            self.domain_relationships[domain_pair] = min(current_similarity + 0.1, 1.0)
        else:
            # Weaken domain relationship
            current_similarity = self.domain_relationships.get(domain_pair, 0.5)
            self.domain_relationships[domain_pair] = max(current_similarity - 0.1, 0.0)
            
        # Learn lessons for future transfers
        lessons = transfer_result.lessons_learned
        
        if not success:
            lessons.append(f"Failed transfer from {source_domain} to {target_domain}")
            
            if transfer_result.performance_improvement < -0.1:
                lessons.append("Negative performance impact - review adaptation strategy")
                
        transfer_result.lessons_learned = lessons
        
    # Background tasks
    
    async def _pattern_discovery_loop(self):
        """Background pattern discovery"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Discover new patterns from active domains
                for domain in self.domain_characteristics.keys():
                    new_patterns = await self.discover_transferable_patterns(domain, 0.6)
                    
                    if new_patterns:
                        logger.info(f"Discovered {len(new_patterns)} new patterns in {domain}")
                        
            except Exception as e:
                logger.error(f"Error in pattern discovery loop: {e}")
                await asyncio.sleep(300)
                
    async def _transfer_optimization_loop(self):
        """Background transfer optimization"""
        
        while True:
            try:
                await asyncio.sleep(7200)  # Every 2 hours
                
                # Analyze adaptation strategy effectiveness
                effective_strategies = {}
                
                for strategy, improvements in self.adaptation_strategies.items():
                    if len(improvements) > 3:
                        avg_improvement = statistics.mean(improvements)
                        if avg_improvement > 0.1:  # 10% improvement threshold
                            effective_strategies[strategy] = avg_improvement
                            
                logger.info(f"Effective adaptation strategies: {effective_strategies}")
                
                # Optimize domain relationships based on transfer history
                await self._optimize_domain_relationships()
                
            except Exception as e:
                logger.error(f"Error in transfer optimization loop: {e}")
                await asyncio.sleep(300)
                
    async def _optimize_domain_relationships(self):
        """Optimize domain relationships based on transfer history"""
        
        # Analyze successful transfers to update domain similarities
        successful_transfers = [r for r in self.transfer_history if r.success and r.performance_improvement > 0.1]
        
        domain_pairs = defaultdict(list)
        for transfer in successful_transfers:
            source_domain = transfer.source_pattern.split("_")[0]
            target_domain = transfer.target_domain
            domain_pairs[(source_domain, target_domain)].append(transfer.performance_improvement)
            
        # Update relationships based on actual transfer success
        for (source, target), improvements in domain_pairs.items():
            if len(improvements) > 2:  # Need multiple successful transfers
                avg_improvement = statistics.mean(improvements)
                
                # High success rate indicates high similarity
                if avg_improvement > 0.2:
                    current_similarity = self.domain_relationships.get((source, target), 0.5)
                    self.domain_relationships[(source, target)] = min(current_similarity + 0.15, 1.0)
                    
    async def shutdown(self):
        """Shutdown knowledge transfer engine"""
        logger.info("Knowledge transfer engine shutdown complete")