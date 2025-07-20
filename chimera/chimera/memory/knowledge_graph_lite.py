"""
Lightweight knowledge graph implementation without heavy dependencies
Optimized for minimal memory footprint and ARM64 performance
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
import re

from ..core.events import EventSystem, EventType, EventEmitter
from .persistence import DataPersistence

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """Lightweight knowledge node"""
    
    id: str
    content: str
    node_type: str
    confidence: float
    relevance: float
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
    """Lightweight knowledge relationship"""
    
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    confidence: float
    created_at: datetime
    evidence: List[str]
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

class LiteKnowledgeGraph(EventEmitter):
    """
    Lightweight neuroplastic knowledge graph
    No external ML dependencies - uses pure Python algorithms
    """
    
    def __init__(self, config, event_system: EventSystem):
        super().__init__(event_system, "LiteKnowledgeGraph")
        
        self.config = config
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[Tuple[str, str], KnowledgeEdge] = {}
        self.adjacency: Dict[str, Set[str]] = {}  # Simple adjacency list
        
        # Configuration
        self.max_nodes = config.get("memory.max_nodes", 10000)
        self.max_edges = config.get("memory.max_edges", 50000)
        self.pruning_threshold = config.get("memory.pruning_threshold", 0.1)
        self.decay_rate = config.get("memory.knowledge_weight_decay", 0.95)
        
        # Persistence
        self.db_path = config.get("memory.graph_database", "./data/knowledge/graph.db")
        self.persistence = DataPersistence(self.db_path)
        
        # Search indices
        self.content_index: Dict[str, Set[str]] = {}
        self.type_index: Dict[str, Set[str]] = {}
        
    async def initialize(self):
        """Initialize the lightweight knowledge graph"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            await self.persistence.initialize()
            await self._load_from_persistence()
            await self._rebuild_indices()
            
            # Start maintenance loop
            asyncio.create_task(self._maintenance_loop())
            
            await self.emit_event(
                EventType.KNOWLEDGE_LEARNED,
                {"message": "Lite knowledge graph initialized", "nodes": len(self.nodes)}
            )
            
            logger.info(f"Lite knowledge graph initialized with {len(self.nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to initialize lite knowledge graph: {e}")
            raise
            
    async def add_knowledge(self, content: str, node_type: str, source: str,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add knowledge with similarity checking"""
        
        node_id = self._generate_node_id(content, node_type)
        
        # Check for similar existing knowledge
        similar_nodes = await self._find_similar_nodes(content, node_type)
        
        if similar_nodes:
            # Strengthen existing knowledge
            best_match = similar_nodes[0]
            await self._strengthen_node(best_match["id"], source)
            return best_match["id"]
            
        # Create new node
        node = KnowledgeNode(
            id=node_id,
            content=content,
            node_type=node_type,
            confidence=0.5,
            relevance=0.8,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            source=source,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self._update_indices(node_id, content, node_type)
        
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
        
    async def query_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query knowledge using lightweight text search"""
        
        query_terms = self._tokenize(query.lower())
        candidate_nodes = set()
        
        # Find nodes matching query terms
        for term in query_terms:
            if term in self.content_index:
                candidate_nodes.update(self.content_index[term])
                
        # Score and rank candidates
        scored_results = []
        for node_id in candidate_nodes:
            if node_id not in self.nodes:
                continue
                
            node = self.nodes[node_id]
            score = self._calculate_relevance_score(node, query_terms)
            
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
            
        # Sort by relevance
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_results[:limit]
        
    async def learn_from_outcome(self, knowledge_ids: List[str], success: bool,
                                confidence_change: float = 0.1):
        """Learn from outcomes to strengthen/weaken knowledge"""
        
        for node_id in knowledge_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                if success:
                    node.confidence = min(node.confidence + confidence_change, 1.0)
                    node.relevance = min(node.relevance + 0.05, 1.0)
                else:
                    node.confidence = max(node.confidence - confidence_change, 0.1)
                    
        await self.emit_event(
            EventType.KNOWLEDGE_UPDATED,
            {"knowledge_ids": knowledge_ids, "success": success}
        )
        
    async def get_related_knowledge(self, node_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get related knowledge using adjacency traversal"""
        
        if node_id not in self.nodes:
            return []
            
        related = []
        visited = set([node_id])
        queue = [(node_id, 0)]
        
        while queue and len(related) < 20:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
                
            # Get neighbors from adjacency list
            neighbors = self.adjacency.get(current_id, set())
            
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
                    
        return sorted(related, key=lambda x: x["relationship_strength"] * x["confidence"], reverse=True)
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        
        if not self.nodes:
            return {"node_count": 0, "edge_count": 0}
            
        # Node type distribution
        node_types = {}
        total_confidence = 0
        total_relevance = 0
        
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
            total_confidence += node.confidence
            total_relevance += node.relevance
            
        avg_confidence = total_confidence / len(self.nodes)
        avg_relevance = total_relevance / len(self.nodes)
        
        # Edge statistics
        relationship_types = {}
        total_strength = 0
        
        for edge in self.edges.values():
            relationship_types[edge.relationship_type] = relationship_types.get(edge.relationship_type, 0) + 1
            total_strength += edge.strength
            
        avg_strength = total_strength / len(self.edges) if self.edges else 0
        
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": node_types,
            "relationship_types": relationship_types,
            "average_confidence": avg_confidence,
            "average_relevance": avg_relevance,
            "average_relationship_strength": avg_strength,
            "database_size": Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        }
        
    async def save_state(self):
        """Save current state"""
        try:
            await self.persistence.save_nodes(list(self.nodes.values()))
            await self.persistence.save_edges(list(self.edges.values()))
        except Exception as e:
            logger.error(f"Error saving knowledge graph state: {e}")
            
    # Private methods
    
    def _generate_node_id(self, content: str, node_type: str) -> str:
        """Generate unique node ID"""
        import hashlib
        content_hash = hashlib.md5(f"{content}:{node_type}".encode()).hexdigest()[:12]
        return f"{node_type}_{content_hash}"
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2]
        
    def _update_indices(self, node_id: str, content: str, node_type: str):
        """Update search indices"""
        # Content index
        words = self._tokenize(content)
        for word in words:
            if word not in self.content_index:
                self.content_index[word] = set()
            self.content_index[word].add(node_id)
            
        # Type index
        if node_type not in self.type_index:
            self.type_index[node_type] = set()
        self.type_index[node_type].add(node_id)
        
    async def _rebuild_indices(self):
        """Rebuild search indices"""
        self.content_index.clear()
        self.type_index.clear()
        self.adjacency.clear()
        
        for node_id, node in self.nodes.items():
            self._update_indices(node_id, node.content, node.node_type)
            
        # Build adjacency list
        for (source_id, target_id) in self.edges:
            if source_id not in self.adjacency:
                self.adjacency[source_id] = set()
            if target_id not in self.adjacency:
                self.adjacency[target_id] = set()
            self.adjacency[source_id].add(target_id)
            self.adjacency[target_id].add(source_id)
            
    def _calculate_relevance_score(self, node: KnowledgeNode, query_terms: List[str]) -> float:
        """Calculate relevance score using TF-IDF like approach"""
        
        content_terms = self._tokenize(node.content)
        
        # Term frequency
        tf_score = 0
        for term in query_terms:
            tf_score += content_terms.count(term)
            
        if not query_terms:
            tf_score = 0
        else:
            tf_score = tf_score / len(query_terms)
            
        # Combine with node properties
        relevance_score = (
            tf_score * 0.4 +
            node.confidence * 0.3 +
            node.relevance * 0.2 +
            min(node.access_count / 10.0, 1.0) * 0.1
        )
        
        # Time decay
        days_since_created = (datetime.utcnow() - node.created_at).days
        time_factor = max(0.5, 1.0 - (days_since_created * 0.01))
        
        return relevance_score * time_factor
        
    async def _find_similar_nodes(self, content: str, node_type: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar nodes using Jaccard similarity"""
        
        content_terms = set(self._tokenize(content))
        candidates = []
        
        # Get nodes of same type
        if node_type in self.type_index:
            for node_id in self.type_index[node_type]:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    node_terms = set(self._tokenize(node.content))
                    
                    # Jaccard similarity
                    if content_terms or node_terms:
                        intersection = content_terms & node_terms
                        union = content_terms | node_terms
                        similarity = len(intersection) / len(union) if union else 0
                        
                        if similarity > threshold:
                            candidates.append({
                                "id": node_id,
                                "similarity": similarity,
                                "content": node.content
                            })
                            
        return sorted(candidates, key=lambda x: x["similarity"], reverse=True)[:5]
        
    async def _strengthen_node(self, node_id: str, source: str):
        """Strengthen existing knowledge node"""
        
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        node.confidence = min(node.confidence + 0.05, 1.0)
        node.relevance = min(node.relevance + 0.05, 1.0)
        node.last_accessed = datetime.utcnow()
        node.access_count += 1
        
        # Add source to metadata
        if "sources" not in node.metadata:
            node.metadata["sources"] = []
        if source not in node.metadata["sources"]:
            node.metadata["sources"].append(source)
            
    async def _auto_connect_knowledge(self, node_id: str):
        """Auto-connect new knowledge to related nodes"""
        
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        node_terms = set(self._tokenize(node.content))
        
        # Find potential connections
        candidates = set()
        for term in list(node_terms)[:5]:  # Limit to avoid explosion
            if term in self.content_index:
                candidates.update(self.content_index[term])
                
        candidates.discard(node_id)
        
        # Create connections to highly similar nodes
        for candidate_id in list(candidates)[:10]:
            if candidate_id in self.nodes:
                candidate = self.nodes[candidate_id]
                candidate_terms = set(self._tokenize(candidate.content))
                
                # Calculate similarity
                intersection = node_terms & candidate_terms
                union = node_terms | candidate_terms
                similarity = len(intersection) / len(union) if union else 0
                
                if similarity > 0.6:
                    await self._add_edge(node_id, candidate_id, "similar_to", similarity)
                    
    async def _add_edge(self, source_id: str, target_id: str, rel_type: str, strength: float):
        """Add edge between nodes"""
        
        edge_key = (source_id, target_id)
        
        if edge_key not in self.edges:
            edge = KnowledgeEdge(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                strength=strength,
                confidence=0.7,
                created_at=datetime.utcnow(),
                evidence=[f"Auto-connected with similarity: {strength:.2f}"]
            )
            
            self.edges[edge_key] = edge
            
            # Update adjacency
            if source_id not in self.adjacency:
                self.adjacency[source_id] = set()
            if target_id not in self.adjacency:
                self.adjacency[target_id] = set()
            self.adjacency[source_id].add(target_id)
            self.adjacency[target_id].add(source_id)
            
    async def _prune_knowledge(self):
        """Remove low-value knowledge"""
        
        nodes_to_remove = []
        current_time = datetime.utcnow()
        
        for node_id, node in self.nodes.items():
            age_days = (current_time - node.created_at).days
            time_since_access = (current_time - node.last_accessed).days
            
            # Value score
            value_score = (
                node.confidence * 0.3 +
                node.relevance * 0.3 +
                min(node.access_count / 10.0, 1.0) * 0.2 +
                max(0, (30 - time_since_access) / 30) * 0.2
            )
            
            if value_score < self.pruning_threshold and age_days > 7:
                nodes_to_remove.append(node_id)
                
        # Remove lowest value nodes
        removal_count = min(len(nodes_to_remove), len(self.nodes) // 10)
        
        for node_id in nodes_to_remove[:removal_count]:
            await self._remove_node(node_id)
            
        if removal_count > 0:
            await self.emit_event(
                EventType.KNOWLEDGE_PRUNED,
                {"removed_count": removal_count, "total_nodes": len(self.nodes)}
            )
            
    async def _remove_node(self, node_id: str):
        """Remove node and its connections"""
        
        if node_id not in self.nodes:
            return
            
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
            
        # Remove from adjacency
        if node_id in self.adjacency:
            for neighbor in self.adjacency[node_id]:
                if neighbor in self.adjacency:
                    self.adjacency[neighbor].discard(node_id)
            del self.adjacency[node_id]
            
        # Remove node
        del self.nodes[node_id]
        
    async def _maintenance_loop(self):
        """Background maintenance"""
        
        while True:
            try:
                await self._apply_temporal_decay()
                await self.save_state()
                await asyncio.sleep(3600)  # 1 hour
            except Exception as e:
                logger.error(f"Error in knowledge graph maintenance: {e}")
                await asyncio.sleep(300)
                
    async def _apply_temporal_decay(self):
        """Apply decay to unused knowledge"""
        
        current_time = datetime.utcnow()
        
        for node in self.nodes.values():
            days_since_access = (current_time - node.last_accessed).days
            
            if days_since_access > 1:
                decay_factor = self.decay_rate ** days_since_access
                node.relevance *= decay_factor
                node.relevance = max(node.relevance, 0.1)
                
    async def _load_from_persistence(self):
        """Load from database"""
        
        try:
            saved_nodes = await self.persistence.load_nodes()
            for node_data in saved_nodes:
                node = KnowledgeNode(**node_data)
                self.nodes[node.id] = node
                
            saved_edges = await self.persistence.load_edges()
            for edge_data in saved_edges:
                edge = KnowledgeEdge(**edge_data)
                edge_key = (edge.source_id, edge.target_id)
                
                if edge.source_id in self.nodes and edge.target_id in self.nodes:
                    self.edges[edge_key] = edge
                    
            logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
            
        except Exception as e:
            logger.warning(f"Could not load from persistence: {e}")
            
    async def shutdown(self):
        """Shutdown the knowledge graph"""
        
        try:
            await self.save_state()
            await self.persistence.close()
            logger.info("Lite knowledge graph shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Alias for compatibility
KnowledgeGraph = LiteKnowledgeGraph