"""
Visualization API Routes - Graph visualization endpoints
Provides endpoints for generating graph visualizations and layouts
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import json
import io
import base64

from pydantic import BaseModel, Field
from ..graph_manager import GraphManager
from ..graph_visualizer import GraphVisualizer
from ..graph_schema import NodeType, RelationType
from ...config import get_config

# Initialize router
visualization_router = APIRouter(prefix="/api/visualization", tags=["visualization"])
logger = logging.getLogger(__name__)

# Global instances
graph_manager = None
graph_visualizer = None

async def get_graph_manager() -> GraphManager:
    """Dependency to get graph manager instance"""
    global graph_manager
    if graph_manager is None:
        graph_manager = GraphManager()
        await graph_manager.initialize()
    return graph_manager

async def get_graph_visualizer(graph_mgr: GraphManager = Depends(get_graph_manager)) -> GraphVisualizer:
    """Dependency to get graph visualizer instance"""
    global graph_visualizer
    if graph_visualizer is None:
        graph_visualizer = GraphVisualizer(graph_mgr)
    return graph_visualizer

# Pydantic models for request/response

class GraphLayoutQuery(BaseModel):
    """Model for graph layout queries"""
    layout_algorithm: Optional[str] = Field("force_directed", description="Layout algorithm")
    node_filter: Optional[Dict[str, Any]] = Field(None, description="Node filter criteria")
    edge_filter: Optional[Dict[str, Any]] = Field(None, description="Edge filter criteria")
    max_nodes: Optional[int] = Field(100, description="Maximum number of nodes")
    include_labels: Optional[bool] = Field(True, description="Include node labels")
    include_properties: Optional[bool] = Field(False, description="Include node properties")

class SubgraphQuery(BaseModel):
    """Model for subgraph queries"""
    center_node_id: str = Field(..., description="Center node ID")
    max_depth: Optional[int] = Field(2, description="Maximum depth from center")
    node_types: Optional[List[str]] = Field(None, description="Filter by node types")
    edge_types: Optional[List[str]] = Field(None, description="Filter by edge types")

class PathVisualizationQuery(BaseModel):
    """Model for path visualization"""
    path_nodes: List[str] = Field(..., description="List of node IDs in path")
    highlight_critical: Optional[bool] = Field(True, description="Highlight critical nodes")
    show_context: Optional[bool] = Field(True, description="Show surrounding context")

class NetworkMapQuery(BaseModel):
    """Model for network map queries"""
    network_id: Optional[str] = Field(None, description="Specific network ID")
    include_vulnerabilities: Optional[bool] = Field(True, description="Include vulnerabilities")
    include_threats: Optional[bool] = Field(True, description="Include threats")
    risk_threshold: Optional[float] = Field(0.5, description="Risk threshold for filtering")

class ExportOptions(BaseModel):
    """Model for export options"""
    format: str = Field("json", description="Export format (json, graphml, gexf, png, svg)")
    include_positions: Optional[bool] = Field(True, description="Include node positions")
    include_styling: Optional[bool] = Field(True, description="Include visual styling")
    image_width: Optional[int] = Field(800, description="Image width for image formats")
    image_height: Optional[int] = Field(600, description="Image height for image formats")

# The actual GraphVisualizer is imported from the module

# Graph layout endpoints

@visualization_router.post("/layout", response_model=Dict[str, Any])
async def generate_graph_layout(
    query: GraphLayoutQuery,
    visualizer: GraphVisualizer = Depends(get_graph_visualizer)
):
    """Generate graph layout for visualization"""
    try:
        layout_data = await visualizer.generate_layout(query)
        
        return {
            "success": True,
            "layout": layout_data,
            "algorithm": query.layout_algorithm,
            "node_count": len(layout_data.get("nodes", [])),
            "edge_count": len(layout_data.get("edges", []))
        }
        
    except Exception as e:
        logger.error(f"Error generating graph layout: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@visualization_router.post("/subgraph", response_model=Dict[str, Any])
async def generate_subgraph(
    query: SubgraphQuery,
    visualizer: GraphVisualizer = Depends(get_graph_visualizer)
):
    """Generate subgraph visualization around a center node"""
    try:
        subgraph_data = await visualizer.generate_subgraph(query)
        
        return {
            "success": True,
            "subgraph": subgraph_data,
            "center_node": query.center_node_id,
            "max_depth": query.max_depth
        }
        
    except Exception as e:
        logger.error(f"Error generating subgraph: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@visualization_router.post("/path", response_model=Dict[str, Any])
async def visualize_path(
    query: PathVisualizationQuery,
    visualizer: GraphVisualizer = Depends(get_graph_visualizer)
):
    """Visualize an attack path"""
    try:
        path_visualization = await visualizer.visualize_path(query)
        
        return {
            "success": True,
            "path_visualization": path_visualization,
            "path_length": len(query.path_nodes)
        }
        
    except Exception as e:
        logger.error(f"Error visualizing path: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@visualization_router.post("/network_map", response_model=Dict[str, Any])
async def generate_network_map(
    query: NetworkMapQuery,
    visualizer: GraphVisualizer = Depends(get_graph_visualizer)
):
    """Generate network topology map"""
    try:
        network_map = await visualizer.generate_network_map(query)
        
        return {
            "success": True,
            "network_map": network_map,
            "include_vulnerabilities": query.include_vulnerabilities,
            "include_threats": query.include_threats
        }
        
    except Exception as e:
        logger.error(f"Error generating network map: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Node and edge styling endpoints

@visualization_router.get("/styling/node_types", response_model=Dict[str, Any])
async def get_node_type_styling():
    """Get default styling for node types"""
    try:
        node_styling = {
            "host": {
                "color": "#4CAF50",
                "shape": "circle",
                "size": 20,
                "label_color": "#000000"
            },
            "service": {
                "color": "#2196F3",
                "shape": "square",
                "size": 15,
                "label_color": "#000000"
            },
            "vulnerability": {
                "color": "#FF9800",
                "shape": "triangle",
                "size": 18,
                "label_color": "#000000"
            },
            "threat": {
                "color": "#F44336",
                "shape": "diamond",
                "size": 25,
                "label_color": "#FFFFFF"
            },
            "user": {
                "color": "#9C27B0",
                "shape": "circle",
                "size": 16,
                "label_color": "#000000"
            },
            "credential": {
                "color": "#FF5722",
                "shape": "hexagon",
                "size": 14,
                "label_color": "#FFFFFF"
            },
            "attack": {
                "color": "#795548",
                "shape": "star",
                "size": 22,
                "label_color": "#FFFFFF"
            }
        }
        
        return {
            "success": True,
            "node_styling": node_styling
        }
        
    except Exception as e:
        logger.error(f"Error getting node styling: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@visualization_router.get("/styling/edge_types", response_model=Dict[str, Any])
async def get_edge_type_styling():
    """Get default styling for edge types"""
    try:
        edge_styling = {
            "affects": {
                "color": "#FF9800",
                "width": 2,
                "style": "solid",
                "arrow": True
            },
            "exploits": {
                "color": "#F44336",
                "width": 3,
                "style": "solid",
                "arrow": True
            },
            "connects_to": {
                "color": "#2196F3",
                "width": 1,
                "style": "dashed",
                "arrow": True
            },
            "uses": {
                "color": "#9C27B0",
                "width": 2,
                "style": "dotted",
                "arrow": True
            },
            "hosts": {
                "color": "#4CAF50",
                "width": 2,
                "style": "solid",
                "arrow": False
            },
            "belongs_to": {
                "color": "#607D8B",
                "width": 1,
                "style": "solid",
                "arrow": True
            }
        }
        
        return {
            "success": True,
            "edge_styling": edge_styling
        }
        
    except Exception as e:
        logger.error(f"Error getting edge styling: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Export endpoints

@visualization_router.post("/export", response_model=Dict[str, Any])
async def export_graph(
    options: ExportOptions,
    visualizer: GraphVisualizer = Depends(get_graph_visualizer)
):
    """Export graph in specified format"""
    try:
        export_data = await visualizer.export_graph(options)
        
        return {
            "success": True,
            "format": options.format,
            "data": export_data,
            "export_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error exporting graph: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Interactive endpoints

@visualization_router.get("/interactive/node/{node_id}", response_model=Dict[str, Any])
async def get_node_details(
    node_id: str = Path(..., description="Node ID"),
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Get detailed information about a node for interactive visualization"""
    try:
        node = await graph_mgr.get_node(node_id)
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Get neighbors
        neighbors = await graph_mgr.get_neighbors(node_id, limit=20)
        
        # Get basic statistics
        neighbor_count = len(neighbors)
        node_types = list(set(neighbor.type.value for neighbor in neighbors))
        
        return {
            "success": True,
            "node": node.to_dict(),
            "neighbors": [neighbor.to_dict() for neighbor in neighbors],
            "statistics": {
                "neighbor_count": neighbor_count,
                "connected_types": node_types
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@visualization_router.get("/interactive/expand/{node_id}", response_model=Dict[str, Any])
async def expand_node(
    node_id: str = Path(..., description="Node ID"),
    depth: int = Query(1, description="Expansion depth"),
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Expand a node to show its connections"""
    try:
        # Get the center node
        center_node = await graph_mgr.get_node(node_id)
        
        if not center_node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Get neighbors at specified depth
        all_nodes = [center_node]
        current_level = [node_id]
        
        for d in range(depth):
            next_level = []
            for current_node_id in current_level:
                neighbors = await graph_mgr.get_neighbors(current_node_id, limit=10)
                for neighbor in neighbors:
                    if neighbor.id not in [n.id for n in all_nodes]:
                        all_nodes.append(neighbor)
                        next_level.append(neighbor.id)
            current_level = next_level
        
        # Simple expansion data
        expansion_data = {
            "center_node": center_node.to_dict(),
            "expanded_nodes": [node.to_dict() for node in all_nodes[1:]],
            "depth": depth,
            "total_nodes": len(all_nodes)
        }
        
        return {
            "success": True,
            "expansion": expansion_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error expanding node: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Layout algorithm endpoints

@visualization_router.get("/layouts/algorithms", response_model=Dict[str, Any])
async def get_layout_algorithms():
    """Get available layout algorithms"""
    try:
        algorithms = {
            "force_directed": {
                "name": "Force Directed",
                "description": "Physics-based layout with attractive and repulsive forces",
                "parameters": {
                    "iterations": {"type": "int", "default": 100, "min": 10, "max": 1000},
                    "spring_strength": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0},
                    "repulsion_strength": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0}
                }
            },
            "hierarchical": {
                "name": "Hierarchical",
                "description": "Tree-like layout with levels",
                "parameters": {
                    "direction": {"type": "string", "default": "top-down", "options": ["top-down", "bottom-up", "left-right", "right-left"]},
                    "level_separation": {"type": "int", "default": 100, "min": 50, "max": 500}
                }
            },
            "circular": {
                "name": "Circular",
                "description": "Arrange nodes in a circle",
                "parameters": {
                    "radius": {"type": "float", "default": 200, "min": 100, "max": 1000}
                }
            },
            "grid": {
                "name": "Grid",
                "description": "Arrange nodes in a grid pattern",
                "parameters": {
                    "spacing": {"type": "int", "default": 50, "min": 20, "max": 200}
                }
            }
        }
        
        return {
            "success": True,
            "algorithms": algorithms
        }
        
    except Exception as e:
        logger.error(f"Error getting layout algorithms: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Filtering endpoints

@visualization_router.post("/filter", response_model=Dict[str, Any])
async def filter_graph(
    filter_criteria: Dict[str, Any] = Body(..., description="Filter criteria"),
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Filter graph based on criteria"""
    try:
        # Extract filter parameters
        node_types = filter_criteria.get("node_types", [])
        edge_types = filter_criteria.get("edge_types", [])
        property_filters = filter_criteria.get("properties", {})
        
        filtered_nodes = []
        
        # Filter nodes by type
        if node_types:
            for node_type_str in node_types:
                try:
                    node_type = NodeType(node_type_str)
                    nodes = await graph_mgr.find_nodes(node_type=node_type, limit=100)
                    filtered_nodes.extend(nodes)
                except ValueError:
                    continue
        else:
            # Get all nodes if no type filter
            filtered_nodes = await graph_mgr.find_nodes(limit=100)
        
        # Apply property filters
        if property_filters:
            final_nodes = []
            for node in filtered_nodes:
                match = True
                for prop_key, prop_value in property_filters.items():
                    if prop_key not in node.properties or node.properties[prop_key] != prop_value:
                        match = False
                        break
                if match:
                    final_nodes.append(node)
            filtered_nodes = final_nodes
        
        return {
            "success": True,
            "filtered_nodes": [node.to_dict() for node in filtered_nodes],
            "count": len(filtered_nodes),
            "filter_criteria": filter_criteria
        }
        
    except Exception as e:
        logger.error(f"Error filtering graph: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Search endpoints

@visualization_router.get("/search", response_model=Dict[str, Any])
async def search_graph(
    query: str = Query(..., description="Search query"),
    node_types: Optional[List[str]] = Query(None, description="Node types to search"),
    limit: int = Query(20, description="Maximum results"),
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Search graph for nodes matching query"""
    try:
        # Simple search implementation
        search_results = []
        
        # Search by node type if specified
        if node_types:
            for node_type_str in node_types:
                try:
                    node_type = NodeType(node_type_str)
                    nodes = await graph_mgr.find_nodes(node_type=node_type, limit=limit)
                    
                    # Filter by query string
                    for node in nodes:
                        if query.lower() in str(node.properties).lower() or query.lower() in node.id.lower():
                            search_results.append(node)
                except ValueError:
                    continue
        else:
            # Search all nodes
            all_nodes = await graph_mgr.find_nodes(limit=limit * 2)
            for node in all_nodes:
                if query.lower() in str(node.properties).lower() or query.lower() in node.id.lower():
                    search_results.append(node)
        
        # Limit results
        search_results = search_results[:limit]
        
        return {
            "success": True,
            "query": query,
            "results": [node.to_dict() for node in search_results],
            "count": len(search_results)
        }
        
    except Exception as e:
        logger.error(f"Error searching graph: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Statistics endpoints

@visualization_router.get("/statistics", response_model=Dict[str, Any])
async def get_visualization_statistics(
    graph_mgr: GraphManager = Depends(get_graph_manager)
):
    """Get statistics for visualization"""
    try:
        stats = await graph_mgr.get_graph_stats()
        
        # Get node type distribution
        node_type_counts = {}
        all_nodes = await graph_mgr.find_nodes(limit=1000)
        for node in all_nodes:
            node_type = node.type.value
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        return {
            "success": True,
            "statistics": {
                "total_nodes": stats.get("node_count", 0),
                "total_edges": stats.get("edge_count", 0),
                "node_type_distribution": node_type_counts,
                "backend": stats.get("backend", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting visualization statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")