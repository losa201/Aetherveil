"""
Graph API Routes - REST endpoints for basic graph operations
Provides CRUD operations for nodes, edges, and graph queries
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from pydantic import BaseModel, Field
from ..graph_manager import GraphManager
from ..graph_schema import NodeType, RelationType, GraphSchema
from ...config import get_config

# Initialize router
graph_router = APIRouter(prefix="/api/graph", tags=["graph"])
logger = logging.getLogger(__name__)

# Global graph manager instance
graph_manager = None

async def get_graph_manager() -> GraphManager:
    """Dependency to get graph manager instance"""
    global graph_manager
    if graph_manager is None:
        graph_manager = GraphManager()
        await graph_manager.initialize()
    return graph_manager

# Pydantic models for request/response

class NodeCreate(BaseModel):
    """Model for creating a new node"""
    type: str = Field(..., description="Node type")
    properties: Dict[str, Any] = Field(..., description="Node properties")
    labels: Optional[List[str]] = Field(None, description="Node labels")

class NodeUpdate(BaseModel):
    """Model for updating a node"""
    properties: Dict[str, Any] = Field(..., description="Properties to update")

class EdgeCreate(BaseModel):
    """Model for creating a new edge"""
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Edge type")
    properties: Optional[Dict[str, Any]] = Field(None, description="Edge properties")
    weight: Optional[float] = Field(1.0, description="Edge weight")

class GraphQuery(BaseModel):
    """Model for graph queries"""
    node_type: Optional[str] = Field(None, description="Filter by node type")
    properties: Optional[Dict[str, Any]] = Field(None, description="Filter by properties")
    limit: Optional[int] = Field(100, description="Maximum number of results")
    offset: Optional[int] = Field(0, description="Offset for pagination")

class NeighborQuery(BaseModel):
    """Model for neighbor queries"""
    node_id: str = Field(..., description="Node ID to find neighbors for")
    direction: Optional[str] = Field("both", description="Direction: incoming, outgoing, or both")
    edge_type: Optional[str] = Field(None, description="Filter by edge type")
    limit: Optional[int] = Field(100, description="Maximum number of results")

class PathQuery(BaseModel):
    """Model for path queries"""
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    max_length: Optional[int] = Field(10, description="Maximum path length")
    algorithm: Optional[str] = Field("shortest", description="Path algorithm")

# Node operations

@graph_router.post("/nodes", response_model=Dict[str, Any])
async def create_node(
    node_data: NodeCreate,
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Create a new node in the graph"""
    try:
        # Validate node type
        try:
            node_type = NodeType(node_data.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid node type: {node_data.type}")
        
        # Validate properties against schema
        schema = GraphSchema()
        validation_errors = schema.validate_node(node_type, node_data.properties)
        if validation_errors:
            raise HTTPException(status_code=400, detail=f"Validation errors: {validation_errors}")
        
        # Create node
        node = await graph_mgr.create_node(
            node_type=node_type,
            properties=node_data.properties,
            labels=node_data.labels
        )
        
        return {
            "success": True,
            "node": node.to_dict(),
            "message": "Node created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating node: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@graph_router.get("/nodes/{node_id}", response_model=Dict[str, Any])
async def get_node(
    node_id: str = Path(..., description="Node ID"),
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Get a node by ID"""
    try:
        node = await graph_mgr.get_node(node_id)
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        return {
            "success": True,
            "node": node.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@graph_router.put("/nodes/{node_id}", response_model=Dict[str, Any])
async def update_node(
    node_id: str = Path(..., description="Node ID"),
    node_data: NodeUpdate = Body(...),
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Update a node's properties"""
    try:
        # Check if node exists
        existing_node = await graph_mgr.get_node(node_id)
        if not existing_node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Update node
        success = await graph_mgr.update_node(node_id, node_data.properties)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update node")
        
        # Get updated node
        updated_node = await graph_mgr.get_node(node_id)
        
        return {
            "success": True,
            "node": updated_node.to_dict() if updated_node else None,
            "message": "Node updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating node: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@graph_router.delete("/nodes/{node_id}", response_model=Dict[str, Any])
async def delete_node(
    node_id: str = Path(..., description="Node ID"),
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Delete a node and all its edges"""
    try:
        # Check if node exists
        existing_node = await graph_mgr.get_node(node_id)
        if not existing_node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Delete node
        success = await graph_mgr.delete_node(node_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to delete node")
        
        return {
            "success": True,
            "message": "Node deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting node: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Edge operations

@graph_router.post("/edges", response_model=Dict[str, Any])
async def create_edge(
    edge_data: EdgeCreate,
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Create a new edge in the graph"""
    try:
        # Validate edge type
        try:
            edge_type = RelationType(edge_data.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid edge type: {edge_data.type}")
        
        # Check if source and target nodes exist
        source_node = await graph_mgr.get_node(edge_data.source_id)
        target_node = await graph_mgr.get_node(edge_data.target_id)
        
        if not source_node:
            raise HTTPException(status_code=404, detail="Source node not found")
        if not target_node:
            raise HTTPException(status_code=404, detail="Target node not found")
        
        # Validate relationship against schema
        schema = GraphSchema()
        validation_errors = schema.validate_relationship(
            edge_type, source_node.type, target_node.type, edge_data.properties or {}
        )
        if validation_errors:
            raise HTTPException(status_code=400, detail=f"Validation errors: {validation_errors}")
        
        # Create edge
        edge = await graph_mgr.create_edge(
            source_id=edge_data.source_id,
            target_id=edge_data.target_id,
            edge_type=edge_type,
            properties=edge_data.properties,
            weight=edge_data.weight
        )
        
        return {
            "success": True,
            "edge": edge.to_dict(),
            "message": "Edge created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating edge: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Query operations

@graph_router.post("/nodes/search", response_model=Dict[str, Any])
async def search_nodes(
    query: GraphQuery,
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Search for nodes based on criteria"""
    try:
        # Parse node type
        node_type = None
        if query.node_type:
            try:
                node_type = NodeType(query.node_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid node type: {query.node_type}")
        
        # Search nodes
        nodes = await graph_mgr.find_nodes(
            node_type=node_type,
            properties=query.properties,
            limit=query.limit
        )
        
        # Apply offset
        if query.offset:
            nodes = nodes[query.offset:]
        
        return {
            "success": True,
            "nodes": [node.to_dict() for node in nodes],
            "count": len(nodes),
            "total_count": len(nodes) + (query.offset or 0)  # Approximation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@graph_router.post("/nodes/neighbors", response_model=Dict[str, Any])
async def get_neighbors(
    query: NeighborQuery,
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Get neighbors of a node"""
    try:
        # Validate edge type
        edge_type = None
        if query.edge_type:
            try:
                edge_type = RelationType(query.edge_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid edge type: {query.edge_type}")
        
        # Validate direction
        if query.direction not in ["incoming", "outgoing", "both"]:
            raise HTTPException(status_code=400, detail="Direction must be 'incoming', 'outgoing', or 'both'")
        
        # Get neighbors
        neighbors = await graph_mgr.get_neighbors(
            node_id=query.node_id,
            direction=query.direction,
            edge_type=edge_type,
            limit=query.limit
        )
        
        return {
            "success": True,
            "node_id": query.node_id,
            "neighbors": [neighbor.to_dict() for neighbor in neighbors],
            "count": len(neighbors)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting neighbors: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@graph_router.post("/paths", response_model=Dict[str, Any])
async def find_paths(
    query: PathQuery,
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Find paths between two nodes"""
    try:
        # This is a placeholder - in a full implementation, you'd integrate with AttackPathAnalyzer
        # For now, return a basic response
        
        # Check if nodes exist
        source_node = await graph_mgr.get_node(query.source_id)
        target_node = await graph_mgr.get_node(query.target_id)
        
        if not source_node:
            raise HTTPException(status_code=404, detail="Source node not found")
        if not target_node:
            raise HTTPException(status_code=404, detail="Target node not found")
        
        # This would be implemented with actual path finding algorithms
        paths = []
        
        return {
            "success": True,
            "source_id": query.source_id,
            "target_id": query.target_id,
            "paths": paths,
            "count": len(paths)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding paths: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Graph statistics

@graph_router.get("/stats", response_model=Dict[str, Any])
async def get_graph_stats(
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Get graph statistics"""
    try:
        stats = await graph_mgr.get_graph_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Schema operations

@graph_router.get("/schema", response_model=Dict[str, Any])
async def get_schema():
    """Get the graph schema"""
    try:
        schema = GraphSchema()
        schema_dict = schema.export_schema()
        
        return {
            "success": True,
            "schema": schema_dict
        }
        
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@graph_router.get("/schema/node_types", response_model=Dict[str, Any])
async def get_node_types():
    """Get all available node types"""
    try:
        node_types = [node_type.value for node_type in NodeType]
        
        return {
            "success": True,
            "node_types": node_types
        }
        
    except Exception as e:
        logger.error(f"Error getting node types: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@graph_router.get("/schema/relationship_types", response_model=Dict[str, Any])
async def get_relationship_types():
    """Get all available relationship types"""
    try:
        relationship_types = [rel_type.value for rel_type in RelationType]
        
        return {
            "success": True,
            "relationship_types": relationship_types
        }
        
    except Exception as e:
        logger.error(f"Error getting relationship types: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Bulk operations

@graph_router.post("/bulk/nodes", response_model=Dict[str, Any])
async def bulk_create_nodes(
    nodes_data: List[NodeCreate],
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Create multiple nodes in bulk"""
    try:
        created_nodes = []
        errors = []
        
        for i, node_data in enumerate(nodes_data):
            try:
                # Validate node type
                node_type = NodeType(node_data.type)
                
                # Validate properties
                schema = GraphSchema()
                validation_errors = schema.validate_node(node_type, node_data.properties)
                if validation_errors:
                    errors.append(f"Node {i}: {validation_errors}")
                    continue
                
                # Create node
                node = await graph_mgr.create_node(
                    node_type=node_type,
                    properties=node_data.properties,
                    labels=node_data.labels
                )
                
                created_nodes.append(node.to_dict())
                
            except Exception as e:
                errors.append(f"Node {i}: {str(e)}")
        
        return {
            "success": len(errors) == 0,
            "created_nodes": created_nodes,
            "errors": errors,
            "created_count": len(created_nodes),
            "error_count": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Error in bulk node creation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@graph_router.post("/bulk/edges", response_model=Dict[str, Any])
async def bulk_create_edges(
    edges_data: List[EdgeCreate],
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Create multiple edges in bulk"""
    try:
        created_edges = []
        errors = []
        
        for i, edge_data in enumerate(edges_data):
            try:
                # Validate edge type
                edge_type = RelationType(edge_data.type)
                
                # Check if nodes exist
                source_node = await graph_mgr.get_node(edge_data.source_id)
                target_node = await graph_mgr.get_node(edge_data.target_id)
                
                if not source_node:
                    errors.append(f"Edge {i}: Source node not found")
                    continue
                if not target_node:
                    errors.append(f"Edge {i}: Target node not found")
                    continue
                
                # Validate relationship
                schema = GraphSchema()
                validation_errors = schema.validate_relationship(
                    edge_type, source_node.type, target_node.type, edge_data.properties or {}
                )
                if validation_errors:
                    errors.append(f"Edge {i}: {validation_errors}")
                    continue
                
                # Create edge
                edge = await graph_mgr.create_edge(
                    source_id=edge_data.source_id,
                    target_id=edge_data.target_id,
                    edge_type=edge_type,
                    properties=edge_data.properties,
                    weight=edge_data.weight
                )
                
                created_edges.append(edge.to_dict())
                
            except Exception as e:
                errors.append(f"Edge {i}: {str(e)}")
        
        return {
            "success": len(errors) == 0,
            "created_edges": created_edges,
            "errors": errors,
            "created_count": len(created_edges),
            "error_count": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Error in bulk edge creation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check

@graph_router.get("/health", response_model=Dict[str, Any])
async def health_check(
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Check graph database health"""
    try:
        stats = await graph_mgr.get_graph_stats()
        
        return {
            "success": True,
            "status": "healthy",
            "backend": stats.get("backend", "unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }