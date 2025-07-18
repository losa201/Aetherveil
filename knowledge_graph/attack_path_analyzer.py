"""
Attack Path Analyzer - Analyzes attack paths and shortest paths in the knowledge graph
Implements advanced algorithms for attack path discovery and analysis
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
from heapq import heappush, heappop
import json
import math

import networkx as nx
from neo4j.exceptions import ServiceUnavailable

from .graph_manager import GraphManager, GraphNode, GraphEdge
from .graph_schema import NodeType, RelationType, SeverityLevel, AttackStage


@dataclass
class AttackPath:
    """Represents an attack path through the graph"""
    path_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_cost: float
    risk_score: float
    complexity: int
    techniques: List[str]
    stages: List[AttackStage]
    prerequisites: List[str]
    impact: str
    confidence: float
    discovered_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "path_id": self.path_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "total_cost": self.total_cost,
            "risk_score": self.risk_score,
            "complexity": self.complexity,
            "techniques": self.techniques,
            "stages": [stage.value for stage in self.stages],
            "prerequisites": self.prerequisites,
            "impact": self.impact,
            "confidence": self.confidence,
            "discovered_at": self.discovered_at.isoformat()
        }


@dataclass
class AttackVector:
    """Represents an attack vector with associated metadata"""
    vector_id: str
    source_node: str
    target_node: str
    attack_type: str
    exploit_methods: List[str]
    required_access: str
    success_probability: float
    detection_difficulty: float
    impact_level: str
    mitigation_difficulty: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "vector_id": self.vector_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "attack_type": self.attack_type,
            "exploit_methods": self.exploit_methods,
            "required_access": self.required_access,
            "success_probability": self.success_probability,
            "detection_difficulty": self.detection_difficulty,
            "impact_level": self.impact_level,
            "mitigation_difficulty": self.mitigation_difficulty
        }


class AttackPathAnalyzer:
    """Advanced attack path analysis and shortest path algorithms"""
    
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.logger = logging.getLogger(__name__)
        
        # Caching for performance
        self.path_cache = {}
        self.vector_cache = {}
        self.cost_cache = {}
        
        # Algorithm parameters
        self.max_path_length = 10
        self.max_paths_per_query = 100
        self.min_confidence_threshold = 0.3
        self.cost_weights = {
            "exploit_difficulty": 0.3,
            "detection_risk": 0.2,
            "success_probability": 0.25,
            "impact_potential": 0.25
        }
        
        # Attack stage progression
        self.stage_progression = {
            AttackStage.RECONNAISSANCE: [AttackStage.WEAPONIZATION, AttackStage.DELIVERY],
            AttackStage.WEAPONIZATION: [AttackStage.DELIVERY],
            AttackStage.DELIVERY: [AttackStage.EXPLOITATION],
            AttackStage.EXPLOITATION: [AttackStage.INSTALLATION, AttackStage.COMMAND_CONTROL],
            AttackStage.INSTALLATION: [AttackStage.COMMAND_CONTROL],
            AttackStage.COMMAND_CONTROL: [AttackStage.ACTIONS_OBJECTIVES],
            AttackStage.ACTIONS_OBJECTIVES: []
        }
    
    async def find_shortest_attack_paths(self, source_node_id: str, target_node_id: str,
                                       max_paths: int = 5, 
                                       attack_stages: Optional[List[AttackStage]] = None) -> List[AttackPath]:
        """Find shortest attack paths between two nodes"""
        try:
            # Check cache first
            cache_key = f"{source_node_id}:{target_node_id}:{max_paths}"
            if cache_key in self.path_cache:
                return self.path_cache[cache_key]
            
            # Get nodes
            source_node = await self.graph_manager.get_node(source_node_id)
            target_node = await self.graph_manager.get_node(target_node_id)
            
            if not source_node or not target_node:
                return []
            
            # Use appropriate algorithm based on backend
            if self.graph_manager.use_fallback:
                paths = await self._find_paths_networkx(source_node_id, target_node_id, max_paths, attack_stages)
            else:
                paths = await self._find_paths_neo4j(source_node_id, target_node_id, max_paths, attack_stages)
            
            # Cache results
            self.path_cache[cache_key] = paths
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error finding attack paths: {e}")
            return []
    
    async def _find_paths_neo4j(self, source_id: str, target_id: str, 
                               max_paths: int, attack_stages: Optional[List[AttackStage]]) -> List[AttackPath]:
        """Find paths using Neo4j Cypher queries"""
        async with self.graph_manager.get_session() as session:
            if not session:
                return []
            
            # Build stage filter
            stage_filter = ""
            if attack_stages:
                stage_values = [stage.value for stage in attack_stages]
                stage_filter = f"AND ALL(rel in relationships(path) WHERE rel.stage IN {stage_values})"
            
            # Cypher query for finding paths
            query = f"""
            MATCH path = (source {{id: $source_id}})-[*1..{self.max_path_length}]->(target {{id: $target_id}})
            WHERE source.id <> target.id
            {stage_filter}
            WITH path, 
                 length(path) as path_length,
                 reduce(cost = 0, rel in relationships(path) | cost + rel.weight) as total_cost
            ORDER BY total_cost ASC, path_length ASC
            LIMIT $max_paths
            RETURN path, path_length, total_cost
            """
            
            result = await session.run(query, {
                "source_id": source_id,
                "target_id": target_id,
                "max_paths": max_paths
            })
            
            paths = []
            async for record in result:
                path_data = record["path"]
                path_length = record["path_length"]
                total_cost = record["total_cost"]
                
                # Extract nodes and edges from path
                nodes = []
                edges = []
                
                for i, node in enumerate(path_data.nodes):
                    node_props = dict(node)
                    node_type = NodeType(node_props.pop("type", "unknown"))
                    created_at = datetime.fromisoformat(node_props.pop("created_at", datetime.now().isoformat()))
                    updated_at = datetime.fromisoformat(node_props.pop("updated_at", datetime.now().isoformat()))
                    node_id = node_props.pop("id")
                    
                    nodes.append(GraphNode(
                        id=node_id,
                        type=node_type,
                        properties=node_props,
                        labels=list(node.labels),
                        created_at=created_at,
                        updated_at=updated_at
                    ))
                
                for i, rel in enumerate(path_data.relationships):
                    rel_props = dict(rel)
                    rel_type = RelationType(rel.type.lower())
                    created_at = datetime.fromisoformat(rel_props.pop("created_at", datetime.now().isoformat()))
                    weight = rel_props.pop("weight", 1.0)
                    rel_id = rel_props.pop("id", f"rel_{i}")
                    
                    edges.append(GraphEdge(
                        id=rel_id,
                        source=str(rel.start_node["id"]),
                        target=str(rel.end_node["id"]),
                        type=rel_type,
                        properties=rel_props,
                        created_at=created_at,
                        weight=weight
                    ))
                
                # Calculate additional metrics
                attack_path = await self._create_attack_path(nodes, edges, total_cost)
                paths.append(attack_path)
            
            return paths
    
    async def _find_paths_networkx(self, source_id: str, target_id: str, 
                                  max_paths: int, attack_stages: Optional[List[AttackStage]]) -> List[AttackPath]:
        """Find paths using NetworkX algorithms"""
        graph = self.graph_manager.networkx_graph
        
        if not graph.has_node(source_id) or not graph.has_node(target_id):
            return []
        
        paths = []
        
        try:
            # Use NetworkX to find shortest paths
            all_paths = list(nx.all_shortest_paths(graph, source_id, target_id, weight='weight'))
            
            # Limit number of paths
            limited_paths = all_paths[:max_paths]
            
            for path_nodes in limited_paths:
                # Get actual node objects
                nodes = []
                edges = []
                total_cost = 0
                
                for i, node_id in enumerate(path_nodes):
                    node_data = graph.nodes[node_id]
                    nodes.append(GraphNode(
                        id=node_id,
                        type=NodeType(node_data.get("type", "unknown")),
                        properties=node_data.get("properties", {}),
                        labels=node_data.get("labels", []),
                        created_at=node_data.get("created_at", datetime.now()),
                        updated_at=node_data.get("updated_at", datetime.now())
                    ))
                    
                    # Get edge to next node
                    if i < len(path_nodes) - 1:
                        next_node_id = path_nodes[i + 1]
                        edge_data = graph[node_id][next_node_id]
                        
                        # Handle multiple edges between nodes
                        if isinstance(edge_data, dict) and len(edge_data) > 0:
                            # Take the first edge (or one with minimum weight)
                            edge_key = min(edge_data.keys(), key=lambda k: edge_data[k].get('weight', 1.0))
                            edge_info = edge_data[edge_key]
                            
                            edges.append(GraphEdge(
                                id=edge_info.get("id", f"edge_{i}"),
                                source=node_id,
                                target=next_node_id,
                                type=RelationType(edge_info.get("type", "connects_to")),
                                properties=edge_info.get("properties", {}),
                                created_at=edge_info.get("created_at", datetime.now()),
                                weight=edge_info.get("weight", 1.0)
                            ))
                            
                            total_cost += edge_info.get("weight", 1.0)
                
                # Create attack path
                attack_path = await self._create_attack_path(nodes, edges, total_cost)
                paths.append(attack_path)
            
        except nx.NetworkXNoPath:
            # No path exists
            pass
        except Exception as e:
            self.logger.error(f"Error finding paths in NetworkX: {e}")
        
        return paths
    
    async def _create_attack_path(self, nodes: List[GraphNode], edges: List[GraphEdge], 
                                 total_cost: float) -> AttackPath:
        """Create AttackPath object with calculated metrics"""
        path_id = f"path_{hash(str([n.id for n in nodes]))}"
        
        # Calculate risk score
        risk_score = await self._calculate_risk_score(nodes, edges)
        
        # Extract techniques and stages
        techniques = []
        stages = []
        
        for edge in edges:
            if "technique" in edge.properties:
                techniques.append(edge.properties["technique"])
            if "stage" in edge.properties:
                try:
                    stages.append(AttackStage(edge.properties["stage"]))
                except ValueError:
                    pass
        
        # Calculate complexity
        complexity = len(nodes) + len(set(techniques))
        
        # Calculate confidence
        confidence = await self._calculate_confidence(nodes, edges)
        
        # Determine impact
        impact = await self._calculate_impact(nodes, edges)
        
        # Extract prerequisites
        prerequisites = []
        for node in nodes:
            if node.type == NodeType.CREDENTIAL:
                prerequisites.append(f"Credential: {node.properties.get('type', 'unknown')}")
            elif node.type == NodeType.VULNERABILITY:
                prerequisites.append(f"Vulnerability: {node.properties.get('cve_id', 'unknown')}")
        
        return AttackPath(
            path_id=path_id,
            nodes=nodes,
            edges=edges,
            total_cost=total_cost,
            risk_score=risk_score,
            complexity=complexity,
            techniques=techniques,
            stages=stages,
            prerequisites=prerequisites,
            impact=impact,
            confidence=confidence,
            discovered_at=datetime.now()
        )
    
    async def _calculate_risk_score(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> float:
        """Calculate risk score for attack path"""
        base_score = 0.0
        
        # Node-based scoring
        for node in nodes:
            if node.type == NodeType.VULNERABILITY:
                severity = node.properties.get("severity", "low")
                if severity == "critical":
                    base_score += 10.0
                elif severity == "high":
                    base_score += 7.0
                elif severity == "medium":
                    base_score += 5.0
                elif severity == "low":
                    base_score += 2.0
            
            elif node.type == NodeType.HOST:
                # High-value targets increase risk
                if "domain_controller" in node.properties.get("roles", []):
                    base_score += 8.0
                elif "server" in node.properties.get("type", ""):
                    base_score += 5.0
        
        # Edge-based scoring
        for edge in edges:
            if edge.type == RelationType.EXPLOITS:
                base_score += 6.0
            elif edge.type == RelationType.USES:
                base_score += 3.0
            elif edge.type == RelationType.ACCESSES:
                base_score += 4.0
        
        # Normalize to 0-100 scale
        return min(100.0, base_score)
    
    async def _calculate_confidence(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> float:
        """Calculate confidence score for attack path"""
        confidence = 1.0
        
        # Reduce confidence for each hop
        confidence *= (0.9 ** len(edges))
        
        # Reduce confidence for uncertain vulnerabilities
        for node in nodes:
            if node.type == NodeType.VULNERABILITY:
                if not node.properties.get("exploitable", True):
                    confidence *= 0.7
        
        return max(0.1, confidence)
    
    async def _calculate_impact(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        """Calculate impact level for attack path"""
        max_impact = "low"
        
        # Check for high-impact targets
        for node in nodes:
            if node.type == NodeType.HOST:
                if "domain_controller" in node.properties.get("roles", []):
                    max_impact = "critical"
                elif "database" in node.properties.get("services", []):
                    max_impact = "high"
            
            elif node.type == NodeType.VULNERABILITY:
                severity = node.properties.get("severity", "low")
                if severity == "critical" and max_impact not in ["critical"]:
                    max_impact = "critical"
                elif severity == "high" and max_impact not in ["critical", "high"]:
                    max_impact = "high"
                elif severity == "medium" and max_impact not in ["critical", "high", "medium"]:
                    max_impact = "medium"
        
        return max_impact
    
    async def find_attack_vectors(self, node_id: str, max_distance: int = 2) -> List[AttackVector]:
        """Find all attack vectors from a given node"""
        try:
            vectors = []
            
            # Get all nodes within max_distance
            if self.graph_manager.use_fallback:
                reachable_nodes = await self._get_reachable_nodes_networkx(node_id, max_distance)
            else:
                reachable_nodes = await self._get_reachable_nodes_neo4j(node_id, max_distance)
            
            # Analyze each reachable node for attack vectors
            for target_node_id in reachable_nodes:
                vector = await self._analyze_attack_vector(node_id, target_node_id)
                if vector:
                    vectors.append(vector)
            
            return vectors
            
        except Exception as e:
            self.logger.error(f"Error finding attack vectors: {e}")
            return []
    
    async def _get_reachable_nodes_neo4j(self, node_id: str, max_distance: int) -> List[str]:
        """Get reachable nodes within max distance using Neo4j"""
        async with self.graph_manager.get_session() as session:
            if not session:
                return []
            
            query = f"""
            MATCH (source {{id: $node_id}})-[*1..{max_distance}]->(target)
            WHERE source.id <> target.id
            RETURN DISTINCT target.id as target_id
            """
            
            result = await session.run(query, {"node_id": node_id})
            return [record["target_id"] async for record in result]
    
    async def _get_reachable_nodes_networkx(self, node_id: str, max_distance: int) -> List[str]:
        """Get reachable nodes within max distance using NetworkX"""
        graph = self.graph_manager.networkx_graph
        
        if not graph.has_node(node_id):
            return []
        
        reachable = set()
        current_level = {node_id}
        
        for distance in range(max_distance):
            next_level = set()
            for node in current_level:
                for neighbor in graph.successors(node):
                    if neighbor not in reachable and neighbor != node_id:
                        next_level.add(neighbor)
                        reachable.add(neighbor)
            current_level = next_level
            
            if not current_level:
                break
        
        return list(reachable)
    
    async def _analyze_attack_vector(self, source_id: str, target_id: str) -> Optional[AttackVector]:
        """Analyze attack vector between two nodes"""
        # Get nodes
        source_node = await self.graph_manager.get_node(source_id)
        target_node = await self.graph_manager.get_node(target_id)
        
        if not source_node or not target_node:
            return None
        
        # Determine attack type and methods
        attack_type = "unknown"
        exploit_methods = []
        required_access = "none"
        
        # Analyze based on node types
        if source_node.type == NodeType.HOST and target_node.type == NodeType.HOST:
            attack_type = "lateral_movement"
            exploit_methods = ["network_exploitation", "credential_reuse", "remote_services"]
            required_access = "network"
        
        elif source_node.type == NodeType.VULNERABILITY and target_node.type == NodeType.HOST:
            attack_type = "exploitation"
            exploit_methods = ["vulnerability_exploitation"]
            required_access = "network"
        
        elif source_node.type == NodeType.CREDENTIAL and target_node.type == NodeType.HOST:
            attack_type = "authentication"
            exploit_methods = ["credential_access", "password_attack"]
            required_access = "credential"
        
        # Calculate probabilities and difficulties
        success_probability = await self._calculate_success_probability(source_node, target_node)
        detection_difficulty = await self._calculate_detection_difficulty(source_node, target_node)
        mitigation_difficulty = await self._calculate_mitigation_difficulty(source_node, target_node)
        
        # Determine impact level
        impact_level = "low"
        if target_node.type == NodeType.HOST:
            if "domain_controller" in target_node.properties.get("roles", []):
                impact_level = "critical"
            elif "server" in target_node.properties.get("type", ""):
                impact_level = "high"
        
        return AttackVector(
            vector_id=f"vector_{source_id}_{target_id}",
            source_node=source_id,
            target_node=target_id,
            attack_type=attack_type,
            exploit_methods=exploit_methods,
            required_access=required_access,
            success_probability=success_probability,
            detection_difficulty=detection_difficulty,
            impact_level=impact_level,
            mitigation_difficulty=mitigation_difficulty
        )
    
    async def _calculate_success_probability(self, source_node: GraphNode, target_node: GraphNode) -> float:
        """Calculate success probability for attack vector"""
        base_probability = 0.3
        
        # Increase probability for known vulnerabilities
        if source_node.type == NodeType.VULNERABILITY:
            if source_node.properties.get("exploitable", False):
                base_probability += 0.4
            if source_node.properties.get("exploit_available", False):
                base_probability += 0.2
        
        # Increase probability for credential access
        if source_node.type == NodeType.CREDENTIAL:
            if source_node.properties.get("cracked", False):
                base_probability += 0.5
        
        # Adjust for target hardening
        if target_node.type == NodeType.HOST:
            security_level = target_node.properties.get("security_level", "medium")
            if security_level == "high":
                base_probability *= 0.7
            elif security_level == "low":
                base_probability *= 1.3
        
        return min(1.0, base_probability)
    
    async def _calculate_detection_difficulty(self, source_node: GraphNode, target_node: GraphNode) -> float:
        """Calculate detection difficulty for attack vector"""
        base_difficulty = 0.5
        
        # Stealth techniques increase difficulty
        if source_node.type == NodeType.ATTACK:
            if "stealth" in source_node.properties.get("techniques", []):
                base_difficulty += 0.3
        
        # Well-known vulnerabilities are easier to detect
        if source_node.type == NodeType.VULNERABILITY:
            if source_node.properties.get("public", True):
                base_difficulty -= 0.2
        
        return max(0.1, min(1.0, base_difficulty))
    
    async def _calculate_mitigation_difficulty(self, source_node: GraphNode, target_node: GraphNode) -> float:
        """Calculate mitigation difficulty for attack vector"""
        base_difficulty = 0.5
        
        # Fundamental vulnerabilities are harder to mitigate
        if source_node.type == NodeType.VULNERABILITY:
            if not source_node.properties.get("patch_available", True):
                base_difficulty += 0.4
        
        # Credential-based attacks can be mitigated with access controls
        if source_node.type == NodeType.CREDENTIAL:
            base_difficulty -= 0.2
        
        return max(0.1, min(1.0, base_difficulty))
    
    async def analyze_attack_surface(self, target_node_id: str, max_distance: int = 3) -> Dict[str, Any]:
        """Analyze the attack surface of a target node"""
        try:
            target_node = await self.graph_manager.get_node(target_node_id)
            if not target_node:
                return {}
            
            # Find all paths to target
            attack_paths = []
            attack_vectors = []
            
            # Get all nodes that could potentially attack the target
            if self.graph_manager.use_fallback:
                potential_sources = await self._get_potential_sources_networkx(target_node_id, max_distance)
            else:
                potential_sources = await self._get_potential_sources_neo4j(target_node_id, max_distance)
            
            # Analyze each potential source
            for source_id in potential_sources:
                paths = await self.find_shortest_attack_paths(source_id, target_node_id, max_paths=3)
                attack_paths.extend(paths)
                
                vector = await self._analyze_attack_vector(source_id, target_node_id)
                if vector:
                    attack_vectors.append(vector)
            
            # Calculate overall metrics
            total_paths = len(attack_paths)
            avg_path_length = sum(len(path.nodes) for path in attack_paths) / max(1, total_paths)
            max_risk_score = max((path.risk_score for path in attack_paths), default=0)
            
            # Identify critical vulnerabilities
            critical_vulnerabilities = []
            for path in attack_paths:
                for node in path.nodes:
                    if node.type == NodeType.VULNERABILITY and node.properties.get("severity") == "critical":
                        critical_vulnerabilities.append(node.id)
            
            critical_vulnerabilities = list(set(critical_vulnerabilities))
            
            return {
                "target_node": target_node.to_dict(),
                "attack_paths": [path.to_dict() for path in attack_paths],
                "attack_vectors": [vector.to_dict() for vector in attack_vectors],
                "metrics": {
                    "total_paths": total_paths,
                    "avg_path_length": avg_path_length,
                    "max_risk_score": max_risk_score,
                    "critical_vulnerabilities": len(critical_vulnerabilities)
                },
                "critical_vulnerabilities": critical_vulnerabilities,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing attack surface: {e}")
            return {}
    
    async def _get_potential_sources_neo4j(self, target_id: str, max_distance: int) -> List[str]:
        """Get potential attack sources using Neo4j"""
        async with self.graph_manager.get_session() as session:
            if not session:
                return []
            
            query = f"""
            MATCH (source)-[*1..{max_distance}]->(target {{id: $target_id}})
            WHERE source.id <> target.id
            AND (source.type IN ['vulnerability', 'credential', 'attack'] OR
                 (source.type = 'host' AND source.id <> $target_id))
            RETURN DISTINCT source.id as source_id
            """
            
            result = await session.run(query, {"target_id": target_id})
            return [record["source_id"] async for record in result]
    
    async def _get_potential_sources_networkx(self, target_id: str, max_distance: int) -> List[str]:
        """Get potential attack sources using NetworkX"""
        graph = self.graph_manager.networkx_graph
        
        if not graph.has_node(target_id):
            return []
        
        potential_sources = []
        
        # Get all nodes within max_distance that could lead to target
        for node_id in graph.nodes():
            if node_id == target_id:
                continue
            
            try:
                # Check if there's a path from node to target
                if nx.has_path(graph, node_id, target_id):
                    path_length = nx.shortest_path_length(graph, node_id, target_id)
                    if path_length <= max_distance:
                        node_data = graph.nodes[node_id]
                        node_type = node_data.get("type", "unknown")
                        
                        # Only consider relevant node types as sources
                        if node_type in ["vulnerability", "credential", "attack", "host"]:
                            potential_sources.append(node_id)
            except nx.NetworkXNoPath:
                continue
        
        return potential_sources
    
    async def get_critical_attack_paths(self, min_risk_score: float = 70.0) -> List[AttackPath]:
        """Get all critical attack paths in the graph"""
        try:
            # This would require a more complex query to find all high-risk paths
            # For now, we'll return cached high-risk paths
            critical_paths = []
            
            for cached_paths in self.path_cache.values():
                for path in cached_paths:
                    if path.risk_score >= min_risk_score:
                        critical_paths.append(path)
            
            # Sort by risk score
            critical_paths.sort(key=lambda p: p.risk_score, reverse=True)
            
            return critical_paths
            
        except Exception as e:
            self.logger.error(f"Error getting critical attack paths: {e}")
            return []
    
    async def clear_cache(self) -> None:
        """Clear all caches"""
        self.path_cache.clear()
        self.vector_cache.clear()
        self.cost_cache.clear()
        self.logger.info("Attack path analyzer cache cleared")