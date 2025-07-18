"""
GraphManager - Central hub for knowledge graph operations
Manages Neo4j connections, NetworkX fallback, and graph operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import json
import pickle
from pathlib import Path

import networkx as nx
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import redis.asyncio as redis

from ..config import get_config
from .graph_schema import GraphSchema, NodeType, RelationType, SecurityEvent


@dataclass
class GraphNode:
    """Graph node representation"""
    id: str
    type: NodeType
    properties: Dict[str, Any]
    labels: List[str]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties,
            "labels": self.labels,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class GraphEdge:
    """Graph edge representation"""
    id: str
    source: str
    target: str
    type: RelationType
    properties: Dict[str, Any]
    created_at: datetime
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "weight": self.weight
        }


class GraphManager:
    """Central knowledge graph manager with Neo4j and NetworkX support"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = get_config()
        if config_override:
            # Apply configuration overrides
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self.logger = logging.getLogger(__name__)
        self.schema = GraphSchema()
        
        # Database connections
        self.neo4j_driver = None
        self.redis_client = None
        self.networkx_graph = nx.MultiDiGraph()
        
        # Fallback mode
        self.use_fallback = False
        self.fallback_file = Path("graph_fallback.pkl")
        
        # Connection pools
        self.connection_pool = None
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Graph metadata
        self.graph_metadata = {
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "node_count": 0,
            "edge_count": 0,
            "schema_version": "1.0"
        }
        
        # Cache for frequently accessed nodes
        self.node_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance metrics
        self.query_metrics = {
            "total_queries": 0,
            "avg_query_time": 0.0,
            "slow_queries": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize graph database connections"""
        try:
            # Initialize Neo4j connection
            await self._initialize_neo4j()
            
            # Initialize Redis for caching
            await self._initialize_redis()
            
            # Initialize schema
            await self._initialize_schema()
            
            # Load existing graph data if in fallback mode
            if self.use_fallback:
                await self._load_fallback_graph()
            
            self.logger.info("Graph manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize graph manager: {e}")
            return False
    
    async def _initialize_neo4j(self) -> None:
        """Initialize Neo4j connection"""
        try:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config.database.neo4j_uri,
                auth=(
                    self.config.database.neo4j_user,
                    self.config.database.neo4j_password.get_secret_value()
                ),
                max_connection_pool_size=self.config.database.neo4j_pool_size,
                connection_timeout=self.config.database.neo4j_timeout
            )
            
            # Test connection
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            
            self.logger.info("Neo4j connection established")
            
        except (ServiceUnavailable, AuthError) as e:
            self.logger.warning(f"Neo4j unavailable, switching to fallback mode: {e}")
            self.use_fallback = True
            self.neo4j_driver = None
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.database.redis_host,
                port=self.config.database.redis_port,
                db=self.config.database.redis_db,
                password=self.config.database.redis_password.get_secret_value() if self.config.database.redis_password else None,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established")
            
        except Exception as e:
            self.logger.warning(f"Redis unavailable, caching disabled: {e}")
            self.redis_client = None
    
    async def _initialize_schema(self) -> None:
        """Initialize graph schema"""
        if not self.use_fallback and self.neo4j_driver:
            try:
                async with self.neo4j_driver.session() as session:
                    # Create constraints and indexes
                    await self._create_constraints(session)
                    await self._create_indexes(session)
                    
                self.logger.info("Graph schema initialized")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize schema: {e}")
                raise
    
    async def _create_constraints(self, session) -> None:
        """Create database constraints"""
        constraints = [
            "CREATE CONSTRAINT host_id IF NOT EXISTS FOR (n:Host) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT service_id IF NOT EXISTS FOR (n:Service) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT vulnerability_id IF NOT EXISTS FOR (n:Vulnerability) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (n:User) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT credential_id IF NOT EXISTS FOR (n:Credential) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT attack_id IF NOT EXISTS FOR (n:Attack) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT artifact_id IF NOT EXISTS FOR (n:Artifact) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT threat_id IF NOT EXISTS FOR (n:Threat) REQUIRE n.id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                await session.run(constraint)
            except Exception as e:
                self.logger.debug(f"Constraint may already exist: {e}")
    
    async def _create_indexes(self, session) -> None:
        """Create database indexes"""
        indexes = [
            "CREATE INDEX host_ip IF NOT EXISTS FOR (n:Host) ON (n.ip)",
            "CREATE INDEX service_port IF NOT EXISTS FOR (n:Service) ON (n.port)",
            "CREATE INDEX vulnerability_cve IF NOT EXISTS FOR (n:Vulnerability) ON (n.cve_id)",
            "CREATE INDEX vulnerability_severity IF NOT EXISTS FOR (n:Vulnerability) ON (n.severity)",
            "CREATE INDEX attack_timestamp IF NOT EXISTS FOR (n:Attack) ON (n.timestamp)",
            "CREATE INDEX threat_severity IF NOT EXISTS FOR (n:Threat) ON (n.severity_score)",
            "CREATE INDEX node_created_at IF NOT EXISTS FOR (n) ON (n.created_at)",
            "CREATE INDEX node_updated_at IF NOT EXISTS FOR (n) ON (n.updated_at)"
        ]
        
        for index in indexes:
            try:
                await session.run(index)
            except Exception as e:
                self.logger.debug(f"Index may already exist: {e}")
    
    async def _load_fallback_graph(self) -> None:
        """Load NetworkX graph from fallback file"""
        try:
            if self.fallback_file.exists():
                with open(self.fallback_file, 'rb') as f:
                    graph_data = pickle.load(f)
                    self.networkx_graph = graph_data.get('graph', nx.MultiDiGraph())
                    self.graph_metadata = graph_data.get('metadata', self.graph_metadata)
                
                self.logger.info(f"Loaded fallback graph with {self.networkx_graph.number_of_nodes()} nodes")
            
        except Exception as e:
            self.logger.error(f"Failed to load fallback graph: {e}")
            self.networkx_graph = nx.MultiDiGraph()
    
    async def _save_fallback_graph(self) -> None:
        """Save NetworkX graph to fallback file"""
        try:
            graph_data = {
                'graph': self.networkx_graph,
                'metadata': self.graph_metadata
            }
            
            with open(self.fallback_file, 'wb') as f:
                pickle.dump(graph_data, f)
                
            self.logger.debug("Fallback graph saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save fallback graph: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper error handling"""
        if self.use_fallback or not self.neo4j_driver:
            yield None
            return
        
        session = None
        try:
            session = self.neo4j_driver.session()
            yield session
        except Exception as e:
            self.logger.error(f"Session error: {e}")
            raise
        finally:
            if session:
                await session.close()
    
    async def create_node(self, node_type: NodeType, properties: Dict[str, Any], 
                         labels: Optional[List[str]] = None) -> GraphNode:
        """Create a new graph node"""
        node_id = f"{node_type.value}:{properties.get('id', self._generate_id())}"
        now = datetime.now()
        
        if labels is None:
            labels = [node_type.value.title()]
        
        node = GraphNode(
            id=node_id,
            type=node_type,
            properties=properties,
            labels=labels,
            created_at=now,
            updated_at=now
        )
        
        # Add to appropriate backend
        if self.use_fallback:
            await self._create_node_networkx(node)
        else:
            await self._create_node_neo4j(node)
        
        # Update cache
        await self._cache_node(node)
        
        # Update metadata
        self.graph_metadata["node_count"] += 1
        self.graph_metadata["last_updated"] = now
        
        self.logger.debug(f"Created node: {node_id}")
        return node
    
    async def _create_node_neo4j(self, node: GraphNode) -> None:
        """Create node in Neo4j"""
        async with self.get_session() as session:
            if not session:
                return
            
            labels_str = ":".join(node.labels)
            query = f"""
            CREATE (n:{labels_str} {{
                id: $id,
                type: $type,
                created_at: $created_at,
                updated_at: $updated_at
            }})
            SET n += $properties
            RETURN n
            """
            
            await session.run(query, {
                "id": node.id,
                "type": node.type.value,
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
                "properties": node.properties
            })
    
    async def _create_node_networkx(self, node: GraphNode) -> None:
        """Create node in NetworkX"""
        self.networkx_graph.add_node(
            node.id,
            type=node.type.value,
            properties=node.properties,
            labels=node.labels,
            created_at=node.created_at,
            updated_at=node.updated_at
        )
        
        # Save to fallback file periodically
        if self.networkx_graph.number_of_nodes() % 100 == 0:
            await self._save_fallback_graph()
    
    async def create_edge(self, source_id: str, target_id: str, 
                         edge_type: RelationType, properties: Optional[Dict[str, Any]] = None,
                         weight: float = 1.0) -> GraphEdge:
        """Create a new graph edge"""
        edge_id = f"{source_id}-{edge_type.value}-{target_id}"
        now = datetime.now()
        
        if properties is None:
            properties = {}
        
        edge = GraphEdge(
            id=edge_id,
            source=source_id,
            target=target_id,
            type=edge_type,
            properties=properties,
            created_at=now,
            weight=weight
        )
        
        # Add to appropriate backend
        if self.use_fallback:
            await self._create_edge_networkx(edge)
        else:
            await self._create_edge_neo4j(edge)
        
        # Update metadata
        self.graph_metadata["edge_count"] += 1
        self.graph_metadata["last_updated"] = now
        
        self.logger.debug(f"Created edge: {edge_id}")
        return edge
    
    async def _create_edge_neo4j(self, edge: GraphEdge) -> None:
        """Create edge in Neo4j"""
        async with self.get_session() as session:
            if not session:
                return
            
            query = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            CREATE (a)-[r:{edge.type.value.upper()} {{
                id: $id,
                created_at: $created_at,
                weight: $weight
            }}]->(b)
            SET r += $properties
            RETURN r
            """
            
            await session.run(query, {
                "source_id": edge.source,
                "target_id": edge.target,
                "id": edge.id,
                "created_at": edge.created_at.isoformat(),
                "weight": edge.weight,
                "properties": edge.properties
            })
    
    async def _create_edge_networkx(self, edge: GraphEdge) -> None:
        """Create edge in NetworkX"""
        self.networkx_graph.add_edge(
            edge.source,
            edge.target,
            key=edge.id,
            type=edge.type.value,
            properties=edge.properties,
            created_at=edge.created_at,
            weight=edge.weight
        )
    
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID"""
        # Check cache first
        cached_node = await self._get_cached_node(node_id)
        if cached_node:
            return cached_node
        
        # Get from database
        if self.use_fallback:
            node = await self._get_node_networkx(node_id)
        else:
            node = await self._get_node_neo4j(node_id)
        
        if node:
            await self._cache_node(node)
        
        return node
    
    async def _get_node_neo4j(self, node_id: str) -> Optional[GraphNode]:
        """Get node from Neo4j"""
        async with self.get_session() as session:
            if not session:
                return None
            
            query = """
            MATCH (n {id: $node_id})
            RETURN n, labels(n) as labels
            """
            
            result = await session.run(query, {"node_id": node_id})
            record = await result.single()
            
            if not record:
                return None
            
            node_data = record["n"]
            labels = record["labels"]
            
            # Extract properties
            properties = dict(node_data)
            node_type = NodeType(properties.pop("type", "unknown"))
            created_at = datetime.fromisoformat(properties.pop("created_at"))
            updated_at = datetime.fromisoformat(properties.pop("updated_at"))
            properties.pop("id", None)  # Remove id from properties
            
            return GraphNode(
                id=node_id,
                type=node_type,
                properties=properties,
                labels=labels,
                created_at=created_at,
                updated_at=updated_at
            )
    
    async def _get_node_networkx(self, node_id: str) -> Optional[GraphNode]:
        """Get node from NetworkX"""
        if not self.networkx_graph.has_node(node_id):
            return None
        
        node_data = self.networkx_graph.nodes[node_id]
        
        return GraphNode(
            id=node_id,
            type=NodeType(node_data.get("type", "unknown")),
            properties=node_data.get("properties", {}),
            labels=node_data.get("labels", []),
            created_at=node_data.get("created_at", datetime.now()),
            updated_at=node_data.get("updated_at", datetime.now())
        )
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        now = datetime.now()
        
        if self.use_fallback:
            success = await self._update_node_networkx(node_id, properties, now)
        else:
            success = await self._update_node_neo4j(node_id, properties, now)
        
        if success:
            # Update cache
            await self._invalidate_cache(node_id)
            
            # Update metadata
            self.graph_metadata["last_updated"] = now
            
            self.logger.debug(f"Updated node: {node_id}")
        
        return success
    
    async def _update_node_neo4j(self, node_id: str, properties: Dict[str, Any], 
                                updated_at: datetime) -> bool:
        """Update node in Neo4j"""
        async with self.get_session() as session:
            if not session:
                return False
            
            query = """
            MATCH (n {id: $node_id})
            SET n += $properties
            SET n.updated_at = $updated_at
            RETURN n
            """
            
            result = await session.run(query, {
                "node_id": node_id,
                "properties": properties,
                "updated_at": updated_at.isoformat()
            })
            
            return await result.single() is not None
    
    async def _update_node_networkx(self, node_id: str, properties: Dict[str, Any], 
                                   updated_at: datetime) -> bool:
        """Update node in NetworkX"""
        if not self.networkx_graph.has_node(node_id):
            return False
        
        node_data = self.networkx_graph.nodes[node_id]
        node_data["properties"].update(properties)
        node_data["updated_at"] = updated_at
        
        return True
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete node and all its edges"""
        if self.use_fallback:
            success = await self._delete_node_networkx(node_id)
        else:
            success = await self._delete_node_neo4j(node_id)
        
        if success:
            # Update cache
            await self._invalidate_cache(node_id)
            
            # Update metadata
            self.graph_metadata["node_count"] -= 1
            self.graph_metadata["last_updated"] = datetime.now()
            
            self.logger.debug(f"Deleted node: {node_id}")
        
        return success
    
    async def _delete_node_neo4j(self, node_id: str) -> bool:
        """Delete node from Neo4j"""
        async with self.get_session() as session:
            if not session:
                return False
            
            query = """
            MATCH (n {id: $node_id})
            DETACH DELETE n
            RETURN count(n) as deleted
            """
            
            result = await session.run(query, {"node_id": node_id})
            record = await result.single()
            
            return record["deleted"] > 0
    
    async def _delete_node_networkx(self, node_id: str) -> bool:
        """Delete node from NetworkX"""
        if not self.networkx_graph.has_node(node_id):
            return False
        
        self.networkx_graph.remove_node(node_id)
        return True
    
    async def find_nodes(self, node_type: Optional[NodeType] = None, 
                        properties: Optional[Dict[str, Any]] = None,
                        limit: int = 100) -> List[GraphNode]:
        """Find nodes matching criteria"""
        if self.use_fallback:
            return await self._find_nodes_networkx(node_type, properties, limit)
        else:
            return await self._find_nodes_neo4j(node_type, properties, limit)
    
    async def _find_nodes_neo4j(self, node_type: Optional[NodeType], 
                               properties: Optional[Dict[str, Any]], 
                               limit: int) -> List[GraphNode]:
        """Find nodes in Neo4j"""
        async with self.get_session() as session:
            if not session:
                return []
            
            conditions = []
            params = {"limit": limit}
            
            if node_type:
                conditions.append("n.type = $type")
                params["type"] = node_type.value
            
            if properties:
                for key, value in properties.items():
                    param_name = f"prop_{key}"
                    conditions.append(f"n.{key} = ${param_name}")
                    params[param_name] = value
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
            MATCH (n)
            WHERE {where_clause}
            RETURN n, labels(n) as labels
            LIMIT $limit
            """
            
            result = await session.run(query, params)
            nodes = []
            
            async for record in result:
                node_data = record["n"]
                labels = record["labels"]
                
                # Extract properties
                properties = dict(node_data)
                node_type = NodeType(properties.pop("type", "unknown"))
                created_at = datetime.fromisoformat(properties.pop("created_at"))
                updated_at = datetime.fromisoformat(properties.pop("updated_at"))
                node_id = properties.pop("id")
                
                nodes.append(GraphNode(
                    id=node_id,
                    type=node_type,
                    properties=properties,
                    labels=labels,
                    created_at=created_at,
                    updated_at=updated_at
                ))
            
            return nodes
    
    async def _find_nodes_networkx(self, node_type: Optional[NodeType], 
                                  properties: Optional[Dict[str, Any]], 
                                  limit: int) -> List[GraphNode]:
        """Find nodes in NetworkX"""
        nodes = []
        count = 0
        
        for node_id, node_data in self.networkx_graph.nodes(data=True):
            if count >= limit:
                break
            
            # Check type filter
            if node_type and node_data.get("type") != node_type.value:
                continue
            
            # Check properties filter
            if properties:
                node_props = node_data.get("properties", {})
                if not all(node_props.get(k) == v for k, v in properties.items()):
                    continue
            
            nodes.append(GraphNode(
                id=node_id,
                type=NodeType(node_data.get("type", "unknown")),
                properties=node_data.get("properties", {}),
                labels=node_data.get("labels", []),
                created_at=node_data.get("created_at", datetime.now()),
                updated_at=node_data.get("updated_at", datetime.now())
            ))
            
            count += 1
        
        return nodes
    
    async def get_neighbors(self, node_id: str, 
                           direction: str = "both",
                           edge_type: Optional[RelationType] = None,
                           limit: int = 100) -> List[GraphNode]:
        """Get neighboring nodes"""
        if self.use_fallback:
            return await self._get_neighbors_networkx(node_id, direction, edge_type, limit)
        else:
            return await self._get_neighbors_neo4j(node_id, direction, edge_type, limit)
    
    async def _get_neighbors_neo4j(self, node_id: str, direction: str, 
                                  edge_type: Optional[RelationType], 
                                  limit: int) -> List[GraphNode]:
        """Get neighbors from Neo4j"""
        async with self.get_session() as session:
            if not session:
                return []
            
            # Build relationship pattern
            if direction == "incoming":
                rel_pattern = "<-[r]-(neighbor)"
            elif direction == "outgoing":
                rel_pattern = "-[r]->(neighbor)"
            else:  # both
                rel_pattern = "-[r]-(neighbor)"
            
            conditions = []
            params = {"node_id": node_id, "limit": limit}
            
            if edge_type:
                conditions.append(f"type(r) = '{edge_type.value.upper()}'")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
            MATCH (n {{id: $node_id}}){rel_pattern}
            WHERE {where_clause}
            RETURN neighbor, labels(neighbor) as labels
            LIMIT $limit
            """
            
            result = await session.run(query, params)
            nodes = []
            
            async for record in result:
                node_data = record["neighbor"]
                labels = record["labels"]
                
                # Extract properties
                properties = dict(node_data)
                node_type = NodeType(properties.pop("type", "unknown"))
                created_at = datetime.fromisoformat(properties.pop("created_at"))
                updated_at = datetime.fromisoformat(properties.pop("updated_at"))
                neighbor_id = properties.pop("id")
                
                nodes.append(GraphNode(
                    id=neighbor_id,
                    type=node_type,
                    properties=properties,
                    labels=labels,
                    created_at=created_at,
                    updated_at=updated_at
                ))
            
            return nodes
    
    async def _get_neighbors_networkx(self, node_id: str, direction: str, 
                                     edge_type: Optional[RelationType], 
                                     limit: int) -> List[GraphNode]:
        """Get neighbors from NetworkX"""
        if not self.networkx_graph.has_node(node_id):
            return []
        
        neighbors = set()
        
        if direction in ["outgoing", "both"]:
            for neighbor in self.networkx_graph.successors(node_id):
                if len(neighbors) >= limit:
                    break
                
                # Check edge type filter
                if edge_type:
                    edges = self.networkx_graph[node_id][neighbor]
                    if not any(edge.get("type") == edge_type.value for edge in edges.values()):
                        continue
                
                neighbors.add(neighbor)
        
        if direction in ["incoming", "both"]:
            for neighbor in self.networkx_graph.predecessors(node_id):
                if len(neighbors) >= limit:
                    break
                
                # Check edge type filter
                if edge_type:
                    edges = self.networkx_graph[neighbor][node_id]
                    if not any(edge.get("type") == edge_type.value for edge in edges.values()):
                        continue
                
                neighbors.add(neighbor)
        
        # Convert to GraphNode objects
        nodes = []
        for neighbor_id in list(neighbors)[:limit]:
            node_data = self.networkx_graph.nodes[neighbor_id]
            nodes.append(GraphNode(
                id=neighbor_id,
                type=NodeType(node_data.get("type", "unknown")),
                properties=node_data.get("properties", {}),
                labels=node_data.get("labels", []),
                created_at=node_data.get("created_at", datetime.now()),
                updated_at=node_data.get("updated_at", datetime.now())
            ))
        
        return nodes
    
    async def _cache_node(self, node: GraphNode) -> None:
        """Cache node in Redis"""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"node:{node.id}"
            cache_data = json.dumps(node.to_dict())
            await self.redis_client.setex(cache_key, self.cache_ttl, cache_data)
        except Exception as e:
            self.logger.debug(f"Failed to cache node {node.id}: {e}")
    
    async def _get_cached_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node from cache"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"node:{node_id}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                node_dict = json.loads(cached_data)
                return GraphNode(
                    id=node_dict["id"],
                    type=NodeType(node_dict["type"]),
                    properties=node_dict["properties"],
                    labels=node_dict["labels"],
                    created_at=datetime.fromisoformat(node_dict["created_at"]),
                    updated_at=datetime.fromisoformat(node_dict["updated_at"])
                )
        except Exception as e:
            self.logger.debug(f"Failed to get cached node {node_id}: {e}")
        
        return None
    
    async def _invalidate_cache(self, node_id: str) -> None:
        """Invalidate node cache"""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"node:{node_id}"
            await self.redis_client.delete(cache_key)
        except Exception as e:
            self.logger.debug(f"Failed to invalidate cache for {node_id}: {e}")
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if self.use_fallback:
            return {
                "node_count": self.networkx_graph.number_of_nodes(),
                "edge_count": self.networkx_graph.number_of_edges(),
                "backend": "NetworkX",
                "metadata": self.graph_metadata
            }
        else:
            async with self.get_session() as session:
                if not session:
                    return {"error": "Database unavailable"}
                
                # Get node count
                node_result = await session.run("MATCH (n) RETURN count(n) as count")
                node_count = (await node_result.single())["count"]
                
                # Get edge count
                edge_result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                edge_count = (await edge_result.single())["count"]
                
                return {
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "backend": "Neo4j",
                    "metadata": self.graph_metadata
                }
    
    async def close(self) -> None:
        """Close all connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Save fallback graph
        if self.use_fallback:
            await self._save_fallback_graph()
        
        self.logger.info("Graph manager closed")