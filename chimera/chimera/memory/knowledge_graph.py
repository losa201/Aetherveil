"""
Neuroplastic knowledge graph for adaptive learning and memory
Implements a weighted, evolving knowledge representation
"""

import asyncio
import json
import logging
import sqlite3
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import networkx as nx

from ..core.events import EventSystem, EventType, EventEmitter
from .persistence import DataPersistence

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """A node in the knowledge graph representing a concept or piece of information"""
    
    id: str
    content: str
    node_type: str  # concept, technique, tool, target, finding, etc.
    confidence: float  # 0.0 to 1.0
    relevance: float  # 0.0 to 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int
    source: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)

@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph representing relationships between concepts"""
    
    source_id: str
    target_id: str
    relationship_type: str  # enables, contradicts, similar_to, part_of, etc.
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    created_at: datetime
    evidence: List[str]  # Sources that support this relationship
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

class KnowledgeGraph(EventEmitter):
    """
    Neuroplastic knowledge graph that evolves based on experiences
    
    Features:
    - Weighted nodes and edges that strengthen/weaken based on outcomes
    - Automatic pruning of low-value knowledge
    - Semantic search and reasoning capabilities
    - Temporal decay of unused knowledge
    - Cross-domain knowledge transfer
    """
    
    def __init__(self, config, event_system: EventSystem):
        super().__init__(event_system, "KnowledgeGraph")
        
        self.config = config
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[Tuple[str, str], KnowledgeEdge] = {}
        
        # Configuration
        self.max_nodes = config.get("memory.max_nodes", 100000)
        self.max_edges = config.get("memory.max_edges", 500000)
        self.pruning_threshold = config.get("memory.pruning_threshold", 0.1)
        self.decay_rate = config.get("memory.knowledge_weight_decay", 0.95)
        
        # Persistence
        self.db_path = config.get("memory.graph_database", "./data/knowledge/graph.db")
        self.persistence = DataPersistence(self.db_path)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_boost = 0.05
        self.relevance_decay = 0.02
        
        # Indexing for fast search
        self.content_index: Dict[str, Set[str]] = {}  # word -> set of node IDs
        self.type_index: Dict[str, Set[str]] = {}     # type -> set of node IDs
        
    async def initialize(self):
        """Initialize the knowledge graph"""
        try:
            # Ensure data directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize persistence
            await self.persistence.initialize()
            
            # Load existing knowledge
            await self._load_from_persistence()
            
            # Rebuild indices
            await self._rebuild_indices()
            
            # Start background maintenance
            asyncio.create_task(self._maintenance_loop())
            
            await self.emit_event(
                EventType.KNOWLEDGE_LEARNED,
                {"message": "Knowledge graph initialized", "nodes": len(self.nodes), "edges": len(self.edges)}
            )
            
            logger.info(f"Knowledge graph initialized with {len(self.nodes)} nodes and {len(self.edges)} edges")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}")
            raise
            
    async def add_knowledge(self, content: str, node_type: str, source: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add new knowledge to the graph"""
        
        node_id = self._generate_node_id(content, node_type)
        
        # Check if similar knowledge already exists
        similar_nodes = await self._find_similar_nodes(content, node_type)
        
        if similar_nodes:
            # Strengthen existing knowledge instead of creating duplicate
            best_match = similar_nodes[0]
            await self._strengthen_node(best_match["id"], source)
            return best_match["id"]
        
        # Create new knowledge node
        node = KnowledgeNode(
            id=node_id,
            content=content,
            node_type=node_type,
            confidence=0.5,  # Start with neutral confidence
            relevance=0.8,   # New knowledge starts with high relevance
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            source=source,
            metadata=metadata or {}
        )
        
        # Add to graph
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
        
        # Update indices
        self._update_content_index(node_id, content)
        self._update_type_index(node_id, node_type)
        
        # Auto-connect to related knowledge
        await self._auto_connect_knowledge(node_id)
        
        # Prune if necessary
        if len(self.nodes) > self.max_nodes:
            await self._prune_knowledge()
            
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {"node_id": node_id, "type": node_type, "source": source}
        )
        
        return node_id
        
    async def add_relationship(self, source_id: str, target_id: str, 
                              relationship_type: str, strength: float, 
                              evidence: List[str]) -> bool:
        """Add a relationship between two knowledge nodes"""
        
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot create relationship: nodes {source_id} or {target_id} not found")
            return False
            
        edge_key = (source_id, target_id)
        
        # Update existing edge or create new one
        if edge_key in self.edges:
            edge = self.edges[edge_key]
            edge.strength = min(edge.strength + strength * self.learning_rate, 1.0)
            edge.evidence.extend(evidence)
        else:
            edge = KnowledgeEdge(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                strength=strength,
                confidence=0.7,
                created_at=datetime.utcnow(),
                evidence=evidence
            )
            self.edges[edge_key] = edge
            
        # Update NetworkX graph
        self.graph.add_edge(source_id, target_id, **asdict(edge))
        
        return True
        
    async def query_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph for relevant information"""
        
        query_words = self._tokenize(query.lower())
        candidate_nodes = set()
        
        # Find nodes matching query terms
        for word in query_words:
            if word in self.content_index:
                candidate_nodes.update(self.content_index[word])
                
        # Score candidates
        scored_results = []
        for node_id in candidate_nodes:
            if node_id not in self.nodes:
                continue
                
            node = self.nodes[node_id]
            score = self._calculate_relevance_score(node, query_words)
            
            # Update access statistics
            node.last_accessed = datetime.utcnow()
            node.access_count += 1
            
            scored_results.append({
                "id": node_id,
                "content": node.content,
                "type": node.node_type,
                "confidence": node.confidence,
                "relevance_score": score,
                "source": node.source,
                "metadata": node.metadata
            })
            
        # Sort by relevance score
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return scored_results[:limit]
        
    async def learn_from_outcome(self, knowledge_ids: List[str], success: bool, 
                                confidence_change: float = 0.1):
        """Learn from the outcome of using specific knowledge"""
        
        for node_id in knowledge_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                if success:
                    # Strengthen successful knowledge
                    node.confidence = min(node.confidence + confidence_change, 1.0)
                    node.relevance = min(node.relevance + self.confidence_boost, 1.0)
                else:
                    # Weaken unsuccessful knowledge
                    node.confidence = max(node.confidence - confidence_change, 0.1)
                    
                # Update graph
                self.graph.nodes[node_id].update(asdict(node))
                
        await self.emit_event(
            EventType.KNOWLEDGE_UPDATED,
            {"knowledge_ids": knowledge_ids, "success": success}
        )
        
    async def learn_from_sources(self, sources: List[Dict[str, Any]]):
        """Learn from multiple information sources"""
        
        for source_data in sources:
            content = source_data.get("content", "")
            source_type = source_data.get("type", "web_search")
            source_name = source_data.get("source", "unknown")
            metadata = source_data.get("metadata", {})
            
            if content:
                # Extract key concepts and techniques
                concepts = await self._extract_concepts(content)
                
                for concept in concepts:
                    await self.add_knowledge(
                        content=concept["text"],
                        node_type=concept["type"],
                        source=source_name,
                        metadata={**metadata, "confidence": concept.get("confidence", 0.7)}
                    )
                    
    async def get_related_knowledge(self, node_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get knowledge related to a specific node"""
        
        if node_id not in self.nodes:
            return []
            
        related = []
        visited = set([node_id])
        
        # BFS to find related nodes
        queue = [(node_id, 0)]
        
        while queue and len(related) < 20:  # Limit results
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
                
            # Get connected nodes
            neighbors = list(self.graph.successors(current_id)) + list(self.graph.predecessors(current_id))
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited and neighbor_id in self.nodes:
                    visited.add(neighbor_id)
                    neighbor = self.nodes[neighbor_id]
                    
                    # Calculate relationship strength
                    edge_key = (current_id, neighbor_id) if (current_id, neighbor_id) in self.edges else (neighbor_id, current_id)
                    strength = self.edges[edge_key].strength if edge_key in self.edges else 0.5
                    
                    related.append({
                        "id": neighbor_id,
                        "content": neighbor.content,
                        "type": neighbor.node_type,
                        "confidence": neighbor.confidence,
                        "relationship_strength": strength,
                        "depth": depth + 1
                    })
                    
                    queue.append((neighbor_id, depth + 1))
                    
        # Sort by relationship strength and confidence
        related.sort(key=lambda x: x["relationship_strength"] * x["confidence"], reverse=True)
        
        return related
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        
        if not self.nodes:
            return {"node_count": 0, "edge_count": 0}
            
        # Node statistics
        node_types = {}
        confidence_sum = 0
        relevance_sum = 0
        
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
            confidence_sum += node.confidence
            relevance_sum += node.relevance
            
        avg_confidence = confidence_sum / len(self.nodes)
        avg_relevance = relevance_sum / len(self.nodes)
        
        # Edge statistics
        relationship_types = {}
        strength_sum = 0
        
        for edge in self.edges.values():
            relationship_types[edge.relationship_type] = relationship_types.get(edge.relationship_type, 0) + 1
            strength_sum += edge.strength
            
        avg_strength = strength_sum / len(self.edges) if self.edges else 0
        
        # Graph connectivity
        try:
            connectivity = nx.average_clustering(self.graph.to_undirected())
        except:
            connectivity = 0
            
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": node_types,
            "relationship_types": relationship_types,
            "average_confidence": avg_confidence,
            "average_relevance": avg_relevance,
            "average_relationship_strength": avg_strength,
            "graph_connectivity": connectivity,
            "database_size": Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        }
        
    async def save_state(self):
        """Save current state to persistence"""
        try:
            await self.persistence.save_nodes(list(self.nodes.values()))
            await self.persistence.save_edges(list(self.edges.values()))
            logger.debug("Knowledge graph state saved")
        except Exception as e:
            logger.error(f"Error saving knowledge graph state: {e}")
            
    # Private methods
    
    def _generate_node_id(self, content: str, node_type: str) -> str:
        """Generate a unique ID for a knowledge node"""
        import hashlib
        content_hash = hashlib.md5(f"{content}:{node_type}".encode()).hexdigest()[:12]
        return f"{node_type}_{content_hash}"
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for text processing"""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2]  # Filter short words
        
    def _update_content_index(self, node_id: str, content: str):
        """Update the content search index"""
        words = self._tokenize(content)
        for word in words:
            if word not in self.content_index:
                self.content_index[word] = set()
            self.content_index[word].add(node_id)
            
    def _update_type_index(self, node_id: str, node_type: str):
        """Update the type search index"""
        if node_type not in self.type_index:
            self.type_index[node_type] = set()
        self.type_index[node_type].add(node_id)
        
    async def _rebuild_indices(self):
        """Rebuild search indices"""
        self.content_index.clear()
        self.type_index.clear()
        
        for node_id, node in self.nodes.items():
            self._update_content_index(node_id, node.content)
            self._update_type_index(node_id, node.node_type)
            
    def _calculate_relevance_score(self, node: KnowledgeNode, query_words: List[str]) -> float:
        """Calculate relevance score for a node given query words"""
        
        content_words = self._tokenize(node.content)
        
        # Calculate word overlap
        word_overlap = len(set(query_words) & set(content_words))
        max_overlap = len(query_words)
        
        if max_overlap == 0:
            word_score = 0
        else:
            word_score = word_overlap / max_overlap
            
        # Combine with node properties
        relevance_score = (
            word_score * 0.4 +
            node.confidence * 0.3 +
            node.relevance * 0.2 +
            min(node.access_count / 10.0, 1.0) * 0.1
        )
        
        # Time decay factor
        days_since_created = (datetime.utcnow() - node.created_at).days
        time_factor = max(0.5, 1.0 - (days_since_created * 0.01))  # 1% decay per day, min 0.5
        
        return relevance_score * time_factor
        
    async def _find_similar_nodes(self, content: str, node_type: str, 
                                 threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find nodes similar to the given content"""
        
        query_words = self._tokenize(content)
        candidates = []
        
        # Get nodes of the same type
        if node_type in self.type_index:
            for node_id in self.type_index[node_type]:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    similarity = self._calculate_similarity(content, node.content)
                    
                    if similarity > threshold:
                        candidates.append({
                            "id": node_id,
                            "similarity": similarity,
                            "content": node.content
                        })
                        
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:5]  # Top 5 similar nodes
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)  # Jaccard similarity
        
    async def _strengthen_node(self, node_id: str, source: str):
        """Strengthen an existing knowledge node"""
        
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        # Increase confidence and relevance
        node.confidence = min(node.confidence + self.confidence_boost, 1.0)
        node.relevance = min(node.relevance + self.confidence_boost, 1.0)
        node.last_accessed = datetime.utcnow()
        node.access_count += 1
        
        # Add source to metadata
        if "sources" not in node.metadata:
            node.metadata["sources"] = []
        if source not in node.metadata["sources"]:
            node.metadata["sources"].append(source)
            
        # Update graph
        self.graph.nodes[node_id].update(asdict(node))
        
    async def _auto_connect_knowledge(self, node_id: str):
        """Automatically connect new knowledge to related existing knowledge"""
        
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        # Find potentially related nodes
        query_words = self._tokenize(node.content)
        related_candidates = set()
        
        for word in query_words:
            if word in self.content_index:
                related_candidates.update(self.content_index[word])
                
        # Remove self
        related_candidates.discard(node_id)
        
        # Create connections to highly similar nodes
        for candidate_id in list(related_candidates)[:10]:  # Limit to prevent explosion
            if candidate_id in self.nodes:
                candidate = self.nodes[candidate_id]
                similarity = self._calculate_similarity(node.content, candidate.content)
                
                if similarity > 0.6:  # High similarity threshold for auto-connection
                    relationship_type = "similar_to"
                    if node.node_type == candidate.node_type:
                        relationship_type = "related_to"
                        
                    await self.add_relationship(
                        source_id=node_id,
                        target_id=candidate_id,
                        relationship_type=relationship_type,
                        strength=similarity,
                        evidence=[f"Content similarity: {similarity:.2f}"]
                    )
                    
    async def _extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract key concepts from content (simplified implementation)"""
        
        concepts = []
        
        # Simple keyword extraction
        security_keywords = [
            "vulnerability", "exploit", "payload", "injection", "xss", "sql", "csrf",
            "authentication", "authorization", "encryption", "ssl", "tls", "certificate",
            "firewall", "ids", "ips", "siem", "scanning", "enumeration", "reconnaissance"
        ]
        
        tool_keywords = [
            "nmap", "burp", "metasploit", "sqlmap", "gobuster", "nuclei", "ffuf",
            "wireshark", "hashcat", "john", "hydra", "nikto", "dirb"
        ]
        
        content_lower = content.lower()
        
        # Extract security concepts
        for keyword in security_keywords:
            if keyword in content_lower:
                # Extract surrounding context
                start_idx = content_lower.find(keyword)
                context_start = max(0, start_idx - 50)
                context_end = min(len(content), start_idx + len(keyword) + 50)
                context = content[context_start:context_end].strip()
                
                concepts.append({
                    "text": context,
                    "type": "security_concept",
                    "confidence": 0.7
                })
                
        # Extract tool mentions
        for tool in tool_keywords:
            if tool in content_lower:
                start_idx = content_lower.find(tool)
                context_start = max(0, start_idx - 30)
                context_end = min(len(content), start_idx + len(tool) + 30)
                context = content[context_start:context_end].strip()
                
                concepts.append({
                    "text": context,
                    "type": "security_tool",
                    "confidence": 0.8
                })
                
        return concepts[:10]  # Limit concepts per source
        
    async def _prune_knowledge(self):
        """Remove low-value knowledge to maintain performance"""
        
        nodes_to_remove = []
        current_time = datetime.utcnow()
        
        for node_id, node in self.nodes.items():
            # Calculate node value score
            age_days = (current_time - node.created_at).days
            time_since_access = (current_time - node.last_accessed).days
            
            # Lower score is worse
            value_score = (
                node.confidence * 0.3 +
                node.relevance * 0.3 +
                min(node.access_count / 10.0, 1.0) * 0.2 +
                max(0, (30 - time_since_access) / 30) * 0.2  # Recency bonus
            )
            
            # Mark for removal if below threshold
            if value_score < self.pruning_threshold and age_days > 7:  # Must be at least a week old
                nodes_to_remove.append(node_id)
                
        # Remove low-value nodes
        removal_count = min(len(nodes_to_remove), len(self.nodes) // 10)  # Remove at most 10%
        
        for node_id in nodes_to_remove[:removal_count]:
            await self._remove_node(node_id)
            
        if removal_count > 0:
            await self.emit_event(
                EventType.KNOWLEDGE_PRUNED,
                {"removed_count": removal_count, "total_nodes": len(self.nodes)}
            )
            
            logger.info(f"Pruned {removal_count} low-value knowledge nodes")
            
    async def _remove_node(self, node_id: str):
        """Remove a node and its connections from the graph"""
        
        if node_id not in self.nodes:
            return
            
        # Remove from graph
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            
        # Remove from indices
        node = self.nodes[node_id]
        words = self._tokenize(node.content)
        for word in words:
            if word in self.content_index:
                self.content_index[word].discard(node_id)
                if not self.content_index[word]:
                    del self.content_index[word]
                    
        if node.node_type in self.type_index:
            self.type_index[node.node_type].discard(node_id)
            if not self.type_index[node.node_type]:
                del self.type_index[node.node_type]
                
        # Remove edges
        edges_to_remove = []
        for edge_key in self.edges:
            if edge_key[0] == node_id or edge_key[1] == node_id:
                edges_to_remove.append(edge_key)
                
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
            
        # Remove node
        del self.nodes[node_id]
        
    async def _maintenance_loop(self):
        """Background maintenance tasks"""
        
        while True:
            try:
                # Apply temporal decay
                await self._apply_temporal_decay()
                
                # Save state periodically
                await self.save_state()
                
                # Sleep for maintenance interval
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in knowledge graph maintenance: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
                
    async def _apply_temporal_decay(self):
        """Apply temporal decay to unused knowledge"""
        
        current_time = datetime.utcnow()
        
        for node in self.nodes.values():
            days_since_access = (current_time - node.last_accessed).days
            
            if days_since_access > 1:
                # Apply decay
                decay_factor = self.decay_rate ** days_since_access
                node.relevance *= decay_factor
                
                # Ensure minimum values
                node.relevance = max(node.relevance, 0.1)
                
    async def _load_from_persistence(self):
        """Load knowledge graph from persistent storage"""
        
        try:
            # Load nodes
            saved_nodes = await self.persistence.load_nodes()
            for node_data in saved_nodes:
                node = KnowledgeNode(**node_data)
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **asdict(node))
                
            # Load edges
            saved_edges = await self.persistence.load_edges()
            for edge_data in saved_edges:
                edge = KnowledgeEdge(**edge_data)
                edge_key = (edge.source_id, edge.target_id)
                self.edges[edge_key] = edge
                
                # Only add edge if both nodes exist
                if edge.source_id in self.nodes and edge.target_id in self.nodes:
                    self.graph.add_edge(edge.source_id, edge.target_id, **asdict(edge))
                    
            logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges from persistence")
            
        except Exception as e:
            logger.warning(f"Could not load from persistence: {e}")
            
    async def shutdown(self):
        """Shutdown the knowledge graph"""
        
        try:
            await self.save_state()
            await self.persistence.close()
            logger.info("Knowledge graph shutdown complete")
        except Exception as e:
            logger.error(f"Error during knowledge graph shutdown: {e}")