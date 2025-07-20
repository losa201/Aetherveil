"""
Data persistence layer for knowledge graph
"""

import asyncio
import sqlite3
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPersistence:
    """Handles persistence of knowledge graph data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        
    async def initialize(self):
        """Initialize database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id TEXT PRIMARY KEY,
                content TEXT,
                node_type TEXT,
                confidence REAL,
                relevance REAL,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER,
                source TEXT,
                metadata TEXT
            )
        """)
        
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                strength REAL,
                confidence REAL,
                created_at TEXT,
                evidence TEXT,
                PRIMARY KEY (source_id, target_id)
            )
        """)
        
        self.connection.commit()
        logger.info("Database initialized")
        
    async def save_nodes(self, nodes: List[Any]):
        """Save knowledge nodes"""
        if not self.connection:
            return
            
        for node in nodes:
            self.connection.execute("""
                INSERT OR REPLACE INTO knowledge_nodes 
                (id, content, node_type, confidence, relevance, created_at, last_accessed, access_count, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id, node.content, node.node_type, node.confidence, node.relevance,
                node.created_at.isoformat(), node.last_accessed.isoformat(), 
                node.access_count, node.source, json.dumps(node.metadata)
            ))
        
        self.connection.commit()
        
    async def save_edges(self, edges: List[Any]):
        """Save knowledge edges"""
        if not self.connection:
            return
            
        for edge in edges:
            self.connection.execute("""
                INSERT OR REPLACE INTO knowledge_edges
                (source_id, target_id, relationship_type, strength, confidence, created_at, evidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.source_id, edge.target_id, edge.relationship_type,
                edge.strength, edge.confidence, edge.created_at.isoformat(),
                json.dumps(edge.evidence)
            ))
            
        self.connection.commit()
        
    async def load_nodes(self) -> List[Dict[str, Any]]:
        """Load knowledge nodes"""
        if not self.connection:
            return []
            
        cursor = self.connection.execute("SELECT * FROM knowledge_nodes")
        nodes = []
        
        for row in cursor.fetchall():
            nodes.append({
                "id": row[0],
                "content": row[1],
                "node_type": row[2],
                "confidence": row[3],
                "relevance": row[4],
                "created_at": row[5],
                "last_accessed": row[6],
                "access_count": row[7],
                "source": row[8],
                "metadata": json.loads(row[9]) if row[9] else {}
            })
            
        return nodes
        
    async def load_edges(self) -> List[Dict[str, Any]]:
        """Load knowledge edges"""
        if not self.connection:
            return []
            
        cursor = self.connection.execute("SELECT * FROM knowledge_edges")
        edges = []
        
        for row in cursor.fetchall():
            edges.append({
                "source_id": row[0],
                "target_id": row[1],
                "relationship_type": row[2],
                "strength": row[3],
                "confidence": row[4],
                "created_at": row[5],
                "evidence": json.loads(row[6]) if row[6] else []
            })
            
        return edges
        
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None