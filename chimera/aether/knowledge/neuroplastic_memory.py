"""
Aether Neuroplastic Memory: Advanced knowledge graph with contextual learning
"""

import asyncio
import logging
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import pickle
import sqlite3
from pathlib import Path
import re
from collections import defaultdict, deque
import math

# Optional Neo4j support (fallback to SQLite if not available)
try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available, using SQLite fallback")

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """Types of knowledge stored in the graph"""
    CONCEPT = "concept"
    INSIGHT = "insight"
    EXPERIENCE = "experience"
    MENTOR_INTERACTION = "mentor_interaction"
    CODE_PATTERN = "code_pattern"
    HYPOTHESIS = "hypothesis"
    SKILL = "skill"
    MEMORY = "memory"

class RelationshipType(Enum):
    """Types of relationships between knowledge nodes"""
    RELATED_TO = "RELATED_TO"
    CAUSED_BY = "CAUSED_BY"
    IMPROVES = "IMPROVES"
    CONTRADICTS = "CONTRADICTS"
    LEARNED_FROM = "LEARNED_FROM"
    APPLIED_IN = "APPLIED_IN"
    BUILDS_ON = "BUILDS_ON"
    SIMILAR_TO = "SIMILAR_TO"
    PART_OF = "PART_OF"
    TRIGGERED_BY = "TRIGGERED_BY"

class MemoryStrength(Enum):
    """Memory consolidation strength levels"""
    WEAK = 0.2
    MODERATE = 0.5
    STRONG = 0.8
    PERMANENT = 1.0

@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    knowledge_type: KnowledgeType
    title: str
    content: str
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    strength: float
    embedding: Optional[List[float]] = None
    
@dataclass 
class KnowledgeRelationship:
    """Represents a relationship between knowledge nodes"""
    relationship_id: str
    from_node_id: str
    to_node_id: str
    relationship_type: RelationshipType
    strength: float
    context: str
    created_at: datetime
    last_reinforced: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Insight:
    """Represents a synthesized insight"""
    insight_id: str
    content: str
    confidence: float
    supporting_nodes: List[str]
    context: Dict[str, Any]
    created_at: datetime
    validated: bool = False

@dataclass
class ConsolidationEvent:
    """Memory consolidation event"""
    event_id: str
    nodes_affected: List[str]
    consolidation_type: str
    strength_changes: Dict[str, float]
    timestamp: datetime

class SimpleEmbedding:
    """Simple embedding generator for knowledge nodes"""
    
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.embedding_dim = 128
        
    def get_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for text"""
        
        # Simple bag-of-words embedding
        words = re.findall(r'\w+', text.lower())
        word_counts = defaultdict(int)
        
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.vocab_size
                self.vocab_size += 1
            word_counts[word] += 1
            
        # Create simple embedding vector
        embedding = [0.0] * self.embedding_dim
        
        for word, count in word_counts.items():
            word_idx = self.vocab[word] % self.embedding_dim
            embedding[word_idx] += count / len(words)
            
        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
            
        return embedding
        
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        
        if not embedding1 or not embedding2:
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(x * x for x in embedding1))
        norm2 = math.sqrt(sum(x * x for x in embedding2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

class NeuroplasticMemory:
    """
    Advanced knowledge graph with neuroplastic learning and memory consolidation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.neo4j_driver = None
        self.sqlite_conn = None
        
        # Memory management
        self.consolidation_scheduler = ConsolidationScheduler(config)
        self.embedding_generator = SimpleEmbedding()
        
        # Local caches
        self.node_cache: Dict[str, KnowledgeNode] = {}
        self.relationship_cache: Dict[str, KnowledgeRelationship] = {}
        self.recent_insights: deque = deque(maxlen=100)
        
        # Learning parameters
        self.forgetting_curve_lambda = config.get('forgetting_curve_lambda', 0.1)
        self.consolidation_threshold = config.get('consolidation_threshold', 0.7)
        self.max_working_memory = config.get('max_working_memory', 50)
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'total_relationships': 0,
            'insights_generated': 0,
            'consolidation_events': 0,
            'last_consolidation': None
        }
        
    async def initialize(self):
        """Initialize the neuroplastic memory system"""
        
        logger.info("Initializing Neuroplastic Memory System...")
        
        try:
            # Try Neo4j first, fallback to SQLite
            if NEO4J_AVAILABLE and self.config.get('neo4j_uri'):
                await self._initialize_neo4j()
            else:
                await self._initialize_sqlite()
                
            # Start background consolidation
            asyncio.create_task(self._consolidation_loop())
            
            # Load existing statistics
            await self._load_statistics()
            
            logger.info(f"Neuroplastic Memory initialized - {self.stats['total_nodes']} nodes, "
                       f"{self.stats['total_relationships']} relationships")
            
        except Exception as e:
            logger.error(f"Failed to initialize neuroplastic memory: {e}")
            raise
            
    async def store_insight(self, insight: Insight, context: Dict[str, Any] = None) -> str:
        """Store new insight with rich contextual links"""
        
        try:
            # Generate unique node ID
            node_id = self._generate_node_id(insight.content)
            
            # Create knowledge node
            node = KnowledgeNode(
                node_id=node_id,
                knowledge_type=KnowledgeType.INSIGHT,
                title=f"Insight: {insight.content[:50]}...",
                content=insight.content,
                metadata={
                    'confidence': insight.confidence,
                    'context': context or {},
                    'supporting_nodes': insight.supporting_nodes,
                    'validated': insight.validated
                },
                tags=self._extract_tags(insight.content),
                created_at=insight.created_at,
                last_accessed=datetime.utcnow(),
                access_count=1,
                strength=insight.confidence,
                embedding=self.embedding_generator.get_embedding(insight.content)
            )
            
            # Store the node
            await self._store_node(node)
            
            # Create contextual relationships
            await self._create_contextual_relationships(node, context)
            
            # Add to recent insights for processing
            self.recent_insights.append(insight)
            
            self.stats['total_nodes'] += 1
            
            logger.info(f"Stored insight: {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Error storing insight: {e}")
            raise
            
    async def store_mentor_interaction(self, mentor: str, prompt: str, response: str, 
                                     metadata: Dict[str, Any] = None) -> str:
        """Store interaction with LLM mentor"""
        
        try:
            interaction_id = self._generate_node_id(f"{mentor}_{prompt}_{response}")
            
            node = KnowledgeNode(
                node_id=interaction_id,
                knowledge_type=KnowledgeType.MENTOR_INTERACTION,
                title=f"Interaction with {mentor}",
                content=f"Prompt: {prompt}\n\nResponse: {response}",
                metadata={
                    'mentor': mentor,
                    'prompt': prompt,
                    'response': response,
                    'response_quality': metadata.get('quality', 0.7) if metadata else 0.7,
                    'trust_score': metadata.get('trust_score', 0.5) if metadata else 0.5,
                    **(metadata or {})
                },
                tags=self._extract_tags(f"{prompt} {response}") + [f"mentor_{mentor}"],
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                strength=metadata.get('quality', 0.7) if metadata else 0.7,
                embedding=self.embedding_generator.get_embedding(f"{prompt} {response}")
            )
            
            await self._store_node(node)
            
            # Create relationships to related concepts
            await self._link_to_existing_knowledge(node)
            
            self.stats['total_nodes'] += 1
            
            logger.info(f"Stored mentor interaction: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error storing mentor interaction: {e}")
            raise
            
    async def retrieve_relevant_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeNode]:
        """Retrieve contextually relevant knowledge using semantic search"""
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.get_embedding(query)
            
            # Get all nodes (in production, this would be optimized)
            all_nodes = await self._get_all_nodes()
            
            # Calculate similarity scores
            scored_nodes = []
            for node in all_nodes:
                if node.embedding:
                    similarity = self.embedding_generator.similarity(query_embedding, node.embedding)
                    scored_nodes.append((similarity, node))
                    
            # Sort by similarity and access strength
            scored_nodes.sort(key=lambda x: x[0] * x[1].strength, reverse=True)
            
            # Update access counts for retrieved nodes
            relevant_nodes = [node for _, node in scored_nodes[:limit]]
            for node in relevant_nodes:
                await self._update_node_access(node.node_id)
                
            logger.info(f"Retrieved {len(relevant_nodes)} relevant nodes for query: {query[:50]}")
            return relevant_nodes
            
        except Exception as e:
            logger.error(f"Error retrieving relevant knowledge: {e}")
            return []
            
    async def synthesize_insights(self, topic: str, context: Dict[str, Any] = None) -> List[Insight]:
        """Combine related knowledge to form new understanding"""
        
        try:
            # Retrieve relevant knowledge
            relevant_nodes = await self.retrieve_relevant_knowledge(topic, limit=20)
            
            if len(relevant_nodes) < 2:
                return []
                
            # Group nodes by similarity
            insight_clusters = await self._cluster_related_nodes(relevant_nodes)
            
            synthesized_insights = []
            
            for cluster in insight_clusters:
                if len(cluster) >= 2:  # Need at least 2 nodes to synthesize
                    insight = await self._synthesize_cluster(cluster, topic, context)
                    if insight:
                        synthesized_insights.append(insight)
                        
            # Store new insights
            for insight in synthesized_insights:
                await self.store_insight(insight, context)
                
            self.stats['insights_generated'] += len(synthesized_insights)
            
            logger.info(f"Synthesized {len(synthesized_insights)} insights for topic: {topic}")
            return synthesized_insights
            
        except Exception as e:
            logger.error(f"Error synthesizing insights: {e}")
            return []
            
    async def consolidate_memories(self) -> ConsolidationEvent:
        """Strengthen important connections, weaken unused ones"""
        
        logger.info("Starting memory consolidation...")
        
        try:
            event_id = self._generate_node_id(f"consolidation_{datetime.utcnow().isoformat()}")
            
            # Get all nodes and relationships
            all_nodes = await self._get_all_nodes()
            all_relationships = await self._get_all_relationships()
            
            strength_changes = {}
            
            # Apply forgetting curve to unused memories
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            for node in all_nodes:
                if node.last_accessed < cutoff_time:
                    # Apply forgetting curve
                    time_since_access = (datetime.utcnow() - node.last_accessed).total_seconds() / 86400  # days
                    decay = math.exp(-self.forgetting_curve_lambda * time_since_access)
                    new_strength = node.strength * decay
                    
                    # Don't let strength go below threshold for important memories
                    if node.access_count > 5:  # Important memory
                        new_strength = max(new_strength, 0.3)
                    else:
                        new_strength = max(new_strength, 0.1)
                        
                    strength_changes[node.node_id] = new_strength - node.strength
                    await self._update_node_strength(node.node_id, new_strength)
                    
            # Strengthen frequently accessed memories
            for node in all_nodes:
                if node.access_count > 3 and node.strength < 0.9:
                    boost = min(0.1, node.access_count * 0.02)
                    new_strength = min(1.0, node.strength + boost)
                    strength_changes[node.node_id] = new_strength - node.strength
                    await self._update_node_strength(node.node_id, new_strength)
                    
            # Strengthen relationships between frequently co-accessed nodes
            await self._strengthen_relationship_patterns()
            
            # Remove very weak memories
            weak_nodes = [node for node in all_nodes if node.strength < 0.05]
            for node in weak_nodes:
                await self._remove_node(node.node_id)
                strength_changes[node.node_id] = -node.strength
                
            consolidation_event = ConsolidationEvent(
                event_id=event_id,
                nodes_affected=list(strength_changes.keys()),
                consolidation_type="periodic",
                strength_changes=strength_changes,
                timestamp=datetime.utcnow()
            )
            
            self.stats['consolidation_events'] += 1
            self.stats['last_consolidation'] = datetime.utcnow().isoformat()
            
            logger.info(f"Memory consolidation complete - {len(strength_changes)} nodes affected")
            return consolidation_event
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            raise
            
    async def find_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify areas where knowledge is sparse or contradictory"""
        
        try:
            gaps = []
            
            # Get all nodes
            all_nodes = await self._get_all_nodes()
            
            # Group by tags/topics
            topic_groups = defaultdict(list)
            for node in all_nodes:
                for tag in node.tags:
                    topic_groups[tag].append(node)
                    
            # Identify sparse topics
            for topic, nodes in topic_groups.items():
                if len(nodes) < 3:  # Sparse knowledge
                    gaps.append({
                        'type': 'sparse_knowledge',
                        'topic': topic,
                        'node_count': len(nodes),
                        'confidence': 1.0 - (len(nodes) / 10)  # Lower confidence for sparse topics
                    })
                    
            # Identify contradictory knowledge
            contradictory_relationships = await self._find_contradictory_relationships()
            for relationship in contradictory_relationships:
                gaps.append({
                    'type': 'contradictory_knowledge',
                    'nodes': [relationship.from_node_id, relationship.to_node_id],
                    'confidence': relationship.strength
                })
                
            logger.info(f"Identified {len(gaps)} knowledge gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Error finding knowledge gaps: {e}")
            return []
            
    # Private helper methods
    
    async def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        
        try:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config['neo4j_uri'],
                auth=(self.config.get('neo4j_user', 'neo4j'), 
                     self.config.get('neo4j_password', 'password'))
            )
            
            # Test connection
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
                
            logger.info("Neo4j connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            # Fallback to SQLite
            await self._initialize_sqlite()
            
    async def _initialize_sqlite(self):
        """Initialize SQLite fallback database"""
        
        db_path = Path(self.config.get('data_dir', '.')) / 'aether_memory.db'
        db_path.parent.mkdir(exist_ok=True)
        
        self.sqlite_conn = sqlite3.connect(str(db_path))
        
        # Create tables
        self.sqlite_conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                node_id TEXT PRIMARY KEY,
                knowledge_type TEXT,
                title TEXT,
                content TEXT,
                metadata TEXT,
                tags TEXT,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER,
                strength REAL,
                embedding TEXT
            )
        ''')
        
        self.sqlite_conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                relationship_id TEXT PRIMARY KEY,
                from_node_id TEXT,
                to_node_id TEXT,
                relationship_type TEXT,
                strength REAL,
                context TEXT,
                created_at TEXT,
                last_reinforced TEXT,
                metadata TEXT
            )
        ''')
        
        self.sqlite_conn.commit()
        logger.info("SQLite database initialized")
        
    async def _store_node(self, node: KnowledgeNode):
        """Store knowledge node in database"""
        
        if self.neo4j_driver:
            await self._store_node_neo4j(node)
        else:
            await self._store_node_sqlite(node)
            
        # Cache the node
        self.node_cache[node.node_id] = node
        
    async def _store_node_sqlite(self, node: KnowledgeNode):
        """Store node in SQLite"""
        
        self.sqlite_conn.execute('''
            INSERT OR REPLACE INTO knowledge_nodes 
            (node_id, knowledge_type, title, content, metadata, tags, created_at, 
             last_accessed, access_count, strength, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node.node_id,
            node.knowledge_type.value,
            node.title,
            node.content,
            json.dumps(node.metadata),
            json.dumps(node.tags),
            node.created_at.isoformat(),
            node.last_accessed.isoformat(),
            node.access_count,
            node.strength,
            json.dumps(node.embedding) if node.embedding else None
        ))
        
        self.sqlite_conn.commit()
        
    async def _get_all_nodes(self) -> List[KnowledgeNode]:
        """Get all knowledge nodes"""
        
        if self.neo4j_driver:
            return await self._get_all_nodes_neo4j()
        else:
            return await self._get_all_nodes_sqlite()
            
    async def _get_all_nodes_sqlite(self) -> List[KnowledgeNode]:
        """Get all nodes from SQLite"""
        
        cursor = self.sqlite_conn.execute('SELECT * FROM knowledge_nodes')
        nodes = []
        
        for row in cursor.fetchall():
            node = KnowledgeNode(
                node_id=row[0],
                knowledge_type=KnowledgeType(row[1]),
                title=row[2],
                content=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
                tags=json.loads(row[5]) if row[5] else [],
                created_at=datetime.fromisoformat(row[6]),
                last_accessed=datetime.fromisoformat(row[7]),
                access_count=row[8],
                strength=row[9],
                embedding=json.loads(row[10]) if row[10] else None
            )
            nodes.append(node)
            
        return nodes
        
    async def _get_all_relationships(self) -> List[KnowledgeRelationship]:
        """Get all relationships"""
        
        if self.neo4j_driver:
            return await self._get_all_relationships_neo4j()
        else:
            return await self._get_all_relationships_sqlite()
            
    async def _get_all_relationships_sqlite(self) -> List[KnowledgeRelationship]:
        """Get all relationships from SQLite"""
        
        cursor = self.sqlite_conn.execute('SELECT * FROM knowledge_relationships')
        relationships = []
        
        for row in cursor.fetchall():
            relationship = KnowledgeRelationship(
                relationship_id=row[0],
                from_node_id=row[1],
                to_node_id=row[2],
                relationship_type=RelationshipType(row[3]),
                strength=row[4],
                context=row[5],
                created_at=datetime.fromisoformat(row[6]),
                last_reinforced=datetime.fromisoformat(row[7]),
                metadata=json.loads(row[8]) if row[8] else {}
            )
            relationships.append(relationship)
            
        return relationships
        
    def _generate_node_id(self, content: str) -> str:
        """Generate unique node ID from content"""
        
        return hashlib.md5(f"{content}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
        
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text content"""
        
        # Simple keyword extraction
        words = re.findall(r'\w+', text.lower())
        
        # Common programming/learning keywords
        relevant_keywords = {
            'python', 'javascript', 'code', 'programming', 'algorithm', 'data',
            'learning', 'ai', 'machine', 'neural', 'network', 'model', 'train',
            'test', 'debug', 'optimize', 'performance', 'security', 'web',
            'api', 'database', 'query', 'function', 'class', 'method', 'variable'
        }
        
        tags = [word for word in words if word in relevant_keywords]
        
        # Limit to 10 most relevant tags
        return list(set(tags))[:10]
        
    async def _create_contextual_relationships(self, node: KnowledgeNode, context: Dict[str, Any]):
        """Create relationships based on context"""
        
        if not context:
            return
            
        # Link to mentor if this came from an interaction
        if 'mentor' in context:
            mentor_nodes = await self._find_nodes_by_tag(f"mentor_{context['mentor']}")
            for mentor_node in mentor_nodes[-5:]:  # Recent interactions
                await self._create_relationship(
                    mentor_node.node_id,
                    node.node_id,
                    RelationshipType.LEARNED_FROM,
                    0.7,
                    f"Learning from {context['mentor']}"
                )
                
        # Link to related concepts based on content similarity
        await self._link_to_existing_knowledge(node)
        
    async def _link_to_existing_knowledge(self, node: KnowledgeNode):
        """Link node to existing similar knowledge"""
        
        if not node.embedding:
            return
            
        # Find similar nodes
        all_nodes = await self._get_all_nodes()
        similarities = []
        
        for existing_node in all_nodes:
            if existing_node.node_id != node.node_id and existing_node.embedding:
                similarity = self.embedding_generator.similarity(node.embedding, existing_node.embedding)
                if similarity > 0.7:  # High similarity threshold
                    similarities.append((similarity, existing_node))
                    
        # Create relationships to most similar nodes
        similarities.sort(reverse=True)
        for similarity, similar_node in similarities[:3]:  # Top 3 similar nodes
            await self._create_relationship(
                node.node_id,
                similar_node.node_id,
                RelationshipType.RELATED_TO,
                similarity,
                "Content similarity"
            )
            
    async def _create_relationship(self, from_node_id: str, to_node_id: str,
                                 relationship_type: RelationshipType, strength: float,
                                 context: str):
        """Create relationship between nodes"""
        
        relationship_id = self._generate_node_id(f"{from_node_id}_{to_node_id}_{relationship_type.value}")
        
        relationship = KnowledgeRelationship(
            relationship_id=relationship_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relationship_type=relationship_type,
            strength=strength,
            context=context,
            created_at=datetime.utcnow(),
            last_reinforced=datetime.utcnow()
        )
        
        if self.neo4j_driver:
            await self._store_relationship_neo4j(relationship)
        else:
            await self._store_relationship_sqlite(relationship)
            
        self.relationship_cache[relationship_id] = relationship
        self.stats['total_relationships'] += 1
        
    async def _store_relationship_sqlite(self, relationship: KnowledgeRelationship):
        """Store relationship in SQLite"""
        
        self.sqlite_conn.execute('''
            INSERT OR REPLACE INTO knowledge_relationships
            (relationship_id, from_node_id, to_node_id, relationship_type, strength,
             context, created_at, last_reinforced, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            relationship.relationship_id,
            relationship.from_node_id,
            relationship.to_node_id,
            relationship.relationship_type.value,
            relationship.strength,
            relationship.context,
            relationship.created_at.isoformat(),
            relationship.last_reinforced.isoformat(),
            json.dumps(relationship.metadata)
        ))
        
        self.sqlite_conn.commit()
        
    async def _update_node_access(self, node_id: str):
        """Update node access statistics"""
        
        if self.neo4j_driver:
            await self._update_node_access_neo4j(node_id)
        else:
            await self._update_node_access_sqlite(node_id)
            
        # Update cache
        if node_id in self.node_cache:
            self.node_cache[node_id].last_accessed = datetime.utcnow()
            self.node_cache[node_id].access_count += 1
            
    async def _update_node_access_sqlite(self, node_id: str):
        """Update node access in SQLite"""
        
        self.sqlite_conn.execute('''
            UPDATE knowledge_nodes 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE node_id = ?
        ''', (datetime.utcnow().isoformat(), node_id))
        
        self.sqlite_conn.commit()
        
    async def _update_node_strength(self, node_id: str, new_strength: float):
        """Update node strength"""
        
        if self.neo4j_driver:
            await self._update_node_strength_neo4j(node_id, new_strength)
        else:
            await self._update_node_strength_sqlite(node_id, new_strength)
            
        # Update cache
        if node_id in self.node_cache:
            self.node_cache[node_id].strength = new_strength
            
    async def _update_node_strength_sqlite(self, node_id: str, new_strength: float):
        """Update node strength in SQLite"""
        
        self.sqlite_conn.execute('''
            UPDATE knowledge_nodes SET strength = ? WHERE node_id = ?
        ''', (new_strength, node_id))
        
        self.sqlite_conn.commit()
        
    async def _cluster_related_nodes(self, nodes: List[KnowledgeNode]) -> List[List[KnowledgeNode]]:
        """Cluster nodes by similarity for insight synthesis"""
        
        if len(nodes) < 2:
            return [nodes] if nodes else []
            
        clusters = []
        used_nodes = set()
        
        for node in nodes:
            if node.node_id in used_nodes:
                continue
                
            cluster = [node]
            used_nodes.add(node.node_id)
            
            # Find similar nodes for this cluster
            for other_node in nodes:
                if (other_node.node_id not in used_nodes and 
                    node.embedding and other_node.embedding):
                    
                    similarity = self.embedding_generator.similarity(
                        node.embedding, other_node.embedding
                    )
                    
                    if similarity > 0.6:  # Cluster threshold
                        cluster.append(other_node)
                        used_nodes.add(other_node.node_id)
                        
            clusters.append(cluster)
            
        return clusters
        
    async def _synthesize_cluster(self, cluster: List[KnowledgeNode], topic: str,
                                context: Dict[str, Any]) -> Optional[Insight]:
        """Synthesize insight from cluster of related nodes"""
        
        if len(cluster) < 2:
            return None
            
        # Combine content from cluster nodes
        combined_content = []
        supporting_nodes = []
        total_confidence = 0.0
        
        for node in cluster:
            combined_content.append(node.content)
            supporting_nodes.append(node.node_id)
            total_confidence += node.strength
            
        # Generate insight content
        insight_content = f"Based on {len(cluster)} related pieces of knowledge about {topic}: "
        insight_content += " ".join(combined_content[:200])  # Limit length
        
        # Calculate confidence
        avg_confidence = total_confidence / len(cluster)
        synthesis_confidence = min(0.9, avg_confidence * 0.8)  # Slightly lower than source confidence
        
        insight = Insight(
            insight_id=self._generate_node_id(insight_content),
            content=insight_content,
            confidence=synthesis_confidence,
            supporting_nodes=supporting_nodes,
            context=context or {},
            created_at=datetime.utcnow(),
            validated=False
        )
        
        return insight
        
    async def _find_nodes_by_tag(self, tag: str) -> List[KnowledgeNode]:
        """Find nodes with specific tag"""
        
        all_nodes = await self._get_all_nodes()
        return [node for node in all_nodes if tag in node.tags]
        
    async def _find_contradictory_relationships(self) -> List[KnowledgeRelationship]:
        """Find relationships marked as contradictory"""
        
        all_relationships = await self._get_all_relationships()
        return [rel for rel in all_relationships if rel.relationship_type == RelationshipType.CONTRADICTS]
        
    async def _strengthen_relationship_patterns(self):
        """Strengthen relationships between frequently co-accessed nodes"""
        
        # This would analyze access patterns and strengthen relationships
        # Placeholder implementation
        pass
        
    async def _remove_node(self, node_id: str):
        """Remove weak node from database"""
        
        if self.neo4j_driver:
            await self._remove_node_neo4j(node_id)
        else:
            await self._remove_node_sqlite(node_id)
            
        # Remove from cache
        if node_id in self.node_cache:
            del self.node_cache[node_id]
            
        self.stats['total_nodes'] -= 1
        
    async def _remove_node_sqlite(self, node_id: str):
        """Remove node from SQLite"""
        
        self.sqlite_conn.execute('DELETE FROM knowledge_nodes WHERE node_id = ?', (node_id,))
        self.sqlite_conn.execute('DELETE FROM knowledge_relationships WHERE from_node_id = ? OR to_node_id = ?', 
                               (node_id, node_id))
        self.sqlite_conn.commit()
        
    async def _consolidation_loop(self):
        """Background loop for memory consolidation"""
        
        consolidation_interval = self.config.get('consolidation_interval_hours', 6) * 3600
        
        while True:
            try:
                await asyncio.sleep(consolidation_interval)
                await self.consolidate_memories()
                
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                await asyncio.sleep(600)  # 10 minute retry
                
    async def _load_statistics(self):
        """Load existing statistics"""
        
        if self.sqlite_conn:
            cursor = self.sqlite_conn.execute('SELECT COUNT(*) FROM knowledge_nodes')
            self.stats['total_nodes'] = cursor.fetchone()[0]
            
            cursor = self.sqlite_conn.execute('SELECT COUNT(*) FROM knowledge_relationships')
            self.stats['total_relationships'] = cursor.fetchone()[0]
            
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        # Update current statistics
        await self._load_statistics()
        
        return {
            'statistics': self.stats,
            'cache_size': len(self.node_cache),
            'recent_insights': len(self.recent_insights),
            'database_type': 'neo4j' if self.neo4j_driver else 'sqlite',
            'consolidation_threshold': self.consolidation_threshold,
            'memory_strength_distribution': await self._get_strength_distribution()
        }
        
    async def _get_strength_distribution(self) -> Dict[str, int]:
        """Get distribution of memory strengths"""
        
        all_nodes = await self._get_all_nodes()
        
        distribution = {
            'weak (0.0-0.3)': 0,
            'moderate (0.3-0.6)': 0,
            'strong (0.6-0.9)': 0,
            'permanent (0.9-1.0)': 0
        }
        
        for node in all_nodes:
            if node.strength <= 0.3:
                distribution['weak (0.0-0.3)'] += 1
            elif node.strength <= 0.6:
                distribution['moderate (0.3-0.6)'] += 1
            elif node.strength <= 0.9:
                distribution['strong (0.6-0.9)'] += 1
            else:
                distribution['permanent (0.9-1.0)'] += 1
                
        return distribution
        
    async def shutdown(self):
        """Gracefully shutdown the memory system"""
        
        logger.info("Shutting down Neuroplastic Memory...")
        
        # Final consolidation
        await self.consolidate_memories()
        
        # Close connections
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.sqlite_conn:
            self.sqlite_conn.close()
            
        logger.info("Neuroplastic Memory shutdown complete")

class ConsolidationScheduler:
    """Manages memory consolidation scheduling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_consolidation = None
        self.consolidation_interval = timedelta(hours=config.get('consolidation_interval_hours', 6))
        
    def should_consolidate(self) -> bool:
        """Check if consolidation should run"""
        
        if self.last_consolidation is None:
            return True
            
        return datetime.utcnow() - self.last_consolidation > self.consolidation_interval
        
    def mark_consolidation_complete(self):
        """Mark consolidation as complete"""
        
        self.last_consolidation = datetime.utcnow()