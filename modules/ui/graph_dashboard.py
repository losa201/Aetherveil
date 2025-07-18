"""
Advanced Graph UI with Real-time D3.js Dashboards for Threat Intelligence Visualization

This module provides a comprehensive web-based dashboard for visualizing threat intelligence
data using D3.js, with real-time updates and interactive graph exploration capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
import threading
from collections import defaultdict, deque
import time
import hashlib

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, websocket
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import networkx as nx
from neo4j import GraphDatabase
import redis
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder


class NodeType(Enum):
    """Graph node types for threat intelligence visualization"""
    HOST = "host"
    DOMAIN = "domain"
    IP_ADDRESS = "ip_address"
    VULNERABILITY = "vulnerability"
    MALWARE = "malware"
    THREAT_ACTOR = "threat_actor"
    CAMPAIGN = "campaign"
    TECHNIQUE = "technique"
    INDICATOR = "indicator"
    ARTIFACT = "artifact"
    NETWORK = "network"
    GEOLOCATION = "geolocation"


class EdgeType(Enum):
    """Graph edge types for relationships"""
    CONNECTS_TO = "connects_to"
    EXPLOITS = "exploits"
    HOSTS = "hosts"
    BELONGS_TO = "belongs_to"
    USES = "uses"
    COMMUNICATES_WITH = "communicates_with"
    DEPENDS_ON = "depends_on"
    LOCATED_IN = "located_in"
    SIMILAR_TO = "similar_to"
    DERIVED_FROM = "derived_from"
    TARGETS = "targets"
    MITIGATES = "mitigates"


@dataclass
class GraphNode:
    """Represents a node in the threat intelligence graph"""
    id: str
    type: NodeType
    label: str
    properties: Dict[str, Any]
    risk_score: float
    confidence: float
    created_at: datetime
    updated_at: datetime
    connections: Set[str] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'type': self.type.value,
            'label': self.label,
            'properties': self.properties,
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'connections': list(self.connections)
        }


@dataclass
class GraphEdge:
    """Represents an edge in the threat intelligence graph"""
    id: str
    source: str
    target: str
    type: EdgeType
    weight: float
    properties: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'type': self.type.value,
            'weight': self.weight,
            'properties': self.properties,
            'created_at': self.created_at.isoformat()
        }


class ThreatIntelligenceGraph:
    """
    Advanced threat intelligence graph with real-time analytics
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 redis_url: str):
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.redis_client = redis.from_url(redis_url)
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.subscribers: Set[weakref.ref] = set()
        self.analytics_cache = {}
        self.update_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize background tasks
        self.background_tasks = []
        self.start_background_tasks()
    
    def start_background_tasks(self):
        """Start background tasks for real-time updates"""
        self.background_tasks = [
            threading.Thread(target=self._graph_analytics_worker, daemon=True),
            threading.Thread(target=self._threat_correlation_worker, daemon=True),
            threading.Thread(target=self._anomaly_detection_worker, daemon=True)
        ]
        
        for task in self.background_tasks:
            task.start()
    
    def subscribe(self, callback):
        """Subscribe to graph updates"""
        self.subscribers.add(weakref.ref(callback))
    
    def unsubscribe(self, callback):
        """Unsubscribe from graph updates"""
        self.subscribers.discard(weakref.ref(callback))
    
    def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
        """Notify all subscribers of graph updates"""
        dead_refs = set()
        for ref in self.subscribers:
            callback = ref()
            if callback is None:
                dead_refs.add(ref)
            else:
                try:
                    callback(event_type, data)
                except Exception as e:
                    self.logger.error(f"Error notifying subscriber: {e}")
        
        # Clean up dead references
        self.subscribers -= dead_refs
    
    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph"""
        try:
            with self.update_lock:
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **node.to_dict())
                
                # Update Neo4j
                self._update_neo4j_node(node)
                
                # Cache invalidation
                self._invalidate_analytics_cache()
                
                # Notify subscribers
                self._notify_subscribers("node_added", node.to_dict())
                
                return True
        except Exception as e:
            self.logger.error(f"Error adding node {node.id}: {e}")
            return False
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the graph"""
        try:
            with self.update_lock:
                self.edges[edge.id] = edge
                self.graph.add_edge(edge.source, edge.target, key=edge.id, **edge.to_dict())
                
                # Update node connections
                if edge.source in self.nodes:
                    self.nodes[edge.source].connections.add(edge.target)
                if edge.target in self.nodes:
                    self.nodes[edge.target].connections.add(edge.source)
                
                # Update Neo4j
                self._update_neo4j_edge(edge)
                
                # Cache invalidation
                self._invalidate_analytics_cache()
                
                # Notify subscribers
                self._notify_subscribers("edge_added", edge.to_dict())
                
                return True
        except Exception as e:
            self.logger.error(f"Error adding edge {edge.id}: {e}")
            return False
    
    def get_subgraph(self, center_node: str, radius: int = 2, 
                    max_nodes: int = 100) -> Dict[str, Any]:
        """Get a subgraph centered on a specific node"""
        try:
            if center_node not in self.graph:
                return {"nodes": [], "edges": []}
            
            # Get nodes within radius
            subgraph_nodes = set([center_node])
            current_level = {center_node}
            
            for _ in range(radius):
                next_level = set()
                for node in current_level:
                    neighbors = set(self.graph.neighbors(node))
                    neighbors.update(self.graph.predecessors(node))
                    next_level.update(neighbors)
                
                subgraph_nodes.update(next_level)
                current_level = next_level
                
                if len(subgraph_nodes) >= max_nodes:
                    break
            
            # Limit to max_nodes
            if len(subgraph_nodes) > max_nodes:
                # Keep highest risk nodes
                scored_nodes = [(node, self.nodes[node].risk_score) 
                               for node in subgraph_nodes if node in self.nodes]
                scored_nodes.sort(key=lambda x: x[1], reverse=True)
                subgraph_nodes = {node for node, _ in scored_nodes[:max_nodes]}
            
            # Extract subgraph
            subgraph = self.graph.subgraph(subgraph_nodes)
            
            # Format for visualization
            nodes = []
            edges = []
            
            for node_id in subgraph.nodes():
                if node_id in self.nodes:
                    node_data = self.nodes[node_id].to_dict()
                    nodes.append(node_data)
            
            for u, v, key in subgraph.edges(keys=True):
                edge_id = key
                if edge_id in self.edges:
                    edge_data = self.edges[edge_id].to_dict()
                    edges.append(edge_data)
            
            return {
                "nodes": nodes,
                "edges": edges,
                "center": center_node,
                "radius": radius
            }
            
        except Exception as e:
            self.logger.error(f"Error getting subgraph for {center_node}: {e}")
            return {"nodes": [], "edges": []}
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get graph analytics and metrics"""
        cache_key = "graph_analytics"
        
        # Check cache
        cached = self.analytics_cache.get(cache_key)
        if cached and time.time() - cached['timestamp'] < 300:  # 5 minutes
            return cached['data']
        
        try:
            analytics = {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "node_types": self._get_node_type_distribution(),
                "edge_types": self._get_edge_type_distribution(),
                "risk_distribution": self._get_risk_distribution(),
                "centrality_metrics": self._calculate_centrality_metrics(),
                "clustering_coefficient": self._calculate_clustering_coefficient(),
                "connected_components": self._analyze_connected_components(),
                "threat_hotspots": self._identify_threat_hotspots(),
                "temporal_analysis": self._analyze_temporal_patterns(),
                "anomalies": self._detect_graph_anomalies()
            }
            
            # Cache results
            self.analytics_cache[cache_key] = {
                'data': analytics,
                'timestamp': time.time()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error calculating analytics: {e}")
            return {}
    
    def search_nodes(self, query: str, node_types: List[NodeType] = None,
                    limit: int = 50) -> List[Dict[str, Any]]:
        """Search nodes by query string"""
        try:
            results = []
            query_lower = query.lower()
            
            for node in self.nodes.values():
                if node_types and node.type not in node_types:
                    continue
                
                # Search in label and properties
                if (query_lower in node.label.lower() or
                    any(query_lower in str(v).lower() for v in node.properties.values())):
                    results.append(node.to_dict())
                
                if len(results) >= limit:
                    break
            
            # Sort by risk score
            results.sort(key=lambda x: x['risk_score'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching nodes: {e}")
            return []
    
    def get_attack_paths(self, source: str, target: str, 
                        max_paths: int = 10) -> List[Dict[str, Any]]:
        """Find attack paths between two nodes"""
        try:
            if source not in self.graph or target not in self.graph:
                return []
            
            # Find all simple paths
            paths = []
            for path in nx.all_simple_paths(self.graph, source, target, cutoff=6):
                if len(paths) >= max_paths:
                    break
                
                # Calculate path risk score
                path_risk = self._calculate_path_risk(path)
                
                # Get path details
                path_nodes = [self.nodes[node_id].to_dict() for node_id in path
                             if node_id in self.nodes]
                path_edges = []
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = self.graph.get_edge_data(u, v)
                    if edge_data:
                        for key, data in edge_data.items():
                            path_edges.append(data)
                            break
                
                paths.append({
                    "path": path,
                    "nodes": path_nodes,
                    "edges": path_edges,
                    "risk_score": path_risk,
                    "length": len(path)
                })
            
            # Sort by risk score
            paths.sort(key=lambda x: x['risk_score'], reverse=True)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error finding attack paths: {e}")
            return []
    
    def _get_node_type_distribution(self) -> Dict[str, int]:
        """Get distribution of node types"""
        distribution = defaultdict(int)
        for node in self.nodes.values():
            distribution[node.type.value] += 1
        return dict(distribution)
    
    def _get_edge_type_distribution(self) -> Dict[str, int]:
        """Get distribution of edge types"""
        distribution = defaultdict(int)
        for edge in self.edges.values():
            distribution[edge.type.value] += 1
        return dict(distribution)
    
    def _get_risk_distribution(self) -> Dict[str, int]:
        """Get risk score distribution"""
        risk_ranges = {
            "Low (0-3)": 0,
            "Medium (3-6)": 0,
            "High (6-8)": 0,
            "Critical (8-10)": 0
        }
        
        for node in self.nodes.values():
            risk = node.risk_score
            if risk < 3:
                risk_ranges["Low (0-3)"] += 1
            elif risk < 6:
                risk_ranges["Medium (3-6)"] += 1
            elif risk < 8:
                risk_ranges["High (6-8)"] += 1
            else:
                risk_ranges["Critical (8-10)"] += 1
        
        return risk_ranges
    
    def _calculate_centrality_metrics(self) -> Dict[str, Any]:
        """Calculate centrality metrics"""
        try:
            if len(self.graph) == 0:
                return {}
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.graph)
            
            # Betweenness centrality (sample for large graphs)
            if len(self.graph) > 1000:
                betweenness = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
            else:
                betweenness = nx.betweenness_centrality(self.graph)
            
            # Closeness centrality
            closeness = nx.closeness_centrality(self.graph)
            
            # PageRank
            pagerank = nx.pagerank(self.graph)
            
            # Top nodes by each metric
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
            top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "degree_centrality": dict(top_degree),
                "betweenness_centrality": dict(top_betweenness),
                "closeness_centrality": dict(top_closeness),
                "pagerank": dict(top_pagerank)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating centrality metrics: {e}")
            return {}
    
    def _calculate_clustering_coefficient(self) -> float:
        """Calculate average clustering coefficient"""
        try:
            if len(self.graph) == 0:
                return 0.0
            return nx.average_clustering(self.graph.to_undirected())
        except Exception as e:
            self.logger.error(f"Error calculating clustering coefficient: {e}")
            return 0.0
    
    def _analyze_connected_components(self) -> Dict[str, Any]:
        """Analyze connected components"""
        try:
            undirected = self.graph.to_undirected()
            components = list(nx.connected_components(undirected))
            
            component_sizes = [len(comp) for comp in components]
            
            return {
                "count": len(components),
                "largest_size": max(component_sizes) if component_sizes else 0,
                "average_size": np.mean(component_sizes) if component_sizes else 0,
                "size_distribution": component_sizes[:10]  # Top 10 largest
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing connected components: {e}")
            return {}
    
    def _identify_threat_hotspots(self) -> List[Dict[str, Any]]:
        """Identify threat hotspots in the graph"""
        try:
            hotspots = []
            
            # Find nodes with high risk scores and high connectivity
            for node in self.nodes.values():
                if node.risk_score > 6:  # High risk threshold
                    connectivity = len(node.connections)
                    if connectivity > 5:  # High connectivity threshold
                        hotspots.append({
                            "node_id": node.id,
                            "label": node.label,
                            "type": node.type.value,
                            "risk_score": node.risk_score,
                            "connectivity": connectivity,
                            "hotspot_score": node.risk_score * np.log(connectivity + 1)
                        })
            
            # Sort by hotspot score
            hotspots.sort(key=lambda x: x['hotspot_score'], reverse=True)
            
            return hotspots[:20]  # Top 20 hotspots
            
        except Exception as e:
            self.logger.error(f"Error identifying threat hotspots: {e}")
            return []
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in the graph"""
        try:
            now = datetime.now()
            
            # Node creation timeline
            node_timeline = defaultdict(int)
            for node in self.nodes.values():
                day = node.created_at.date()
                node_timeline[day.isoformat()] += 1
            
            # Edge creation timeline
            edge_timeline = defaultdict(int)
            for edge in self.edges.values():
                day = edge.created_at.date()
                edge_timeline[day.isoformat()] += 1
            
            # Recent activity (last 24 hours)
            recent_threshold = now - timedelta(hours=24)
            recent_nodes = sum(1 for node in self.nodes.values() 
                             if node.created_at >= recent_threshold)
            recent_edges = sum(1 for edge in self.edges.values() 
                             if edge.created_at >= recent_threshold)
            
            return {
                "node_timeline": dict(node_timeline),
                "edge_timeline": dict(edge_timeline),
                "recent_nodes": recent_nodes,
                "recent_edges": recent_edges,
                "growth_rate": self._calculate_growth_rate()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {e}")
            return {}
    
    def _detect_graph_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in the graph structure"""
        try:
            anomalies = []
            
            # Detect isolated high-risk nodes
            for node in self.nodes.values():
                if node.risk_score > 7 and len(node.connections) == 0:
                    anomalies.append({
                        "type": "isolated_high_risk",
                        "node_id": node.id,
                        "description": f"High-risk node {node.label} with no connections",
                        "severity": "medium"
                    })
            
            # Detect unusual connectivity patterns
            degrees = [len(node.connections) for node in self.nodes.values()]
            if degrees:
                mean_degree = np.mean(degrees)
                std_degree = np.std(degrees)
                
                for node in self.nodes.values():
                    degree = len(node.connections)
                    if degree > mean_degree + 3 * std_degree:  # 3 sigma rule
                        anomalies.append({
                            "type": "unusual_connectivity",
                            "node_id": node.id,
                            "description": f"Node {node.label} has unusually high connectivity ({degree})",
                            "severity": "low"
                        })
            
            # Detect rapid growth in connections
            recent_threshold = datetime.now() - timedelta(hours=1)
            rapid_growth_nodes = []
            
            for node in self.nodes.values():
                if node.updated_at >= recent_threshold:
                    rapid_growth_nodes.append(node.id)
            
            if len(rapid_growth_nodes) > len(self.nodes) * 0.1:  # 10% threshold
                anomalies.append({
                    "type": "rapid_growth",
                    "description": f"Rapid growth detected: {len(rapid_growth_nodes)} nodes updated recently",
                    "severity": "high"
                })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _calculate_path_risk(self, path: List[str]) -> float:
        """Calculate risk score for a path"""
        try:
            if not path:
                return 0.0
            
            # Average risk of nodes in path
            node_risks = [self.nodes[node_id].risk_score for node_id in path 
                         if node_id in self.nodes]
            avg_risk = np.mean(node_risks) if node_risks else 0.0
            
            # Path length penalty (shorter paths are riskier)
            length_factor = 1.0 / (len(path) ** 0.5)
            
            return avg_risk * length_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating path risk: {e}")
            return 0.0
    
    def _calculate_growth_rate(self) -> Dict[str, float]:
        """Calculate growth rate of nodes and edges"""
        try:
            now = datetime.now()
            
            # Count nodes/edges created in last 24 hours
            last_24h = now - timedelta(hours=24)
            nodes_24h = sum(1 for node in self.nodes.values() 
                           if node.created_at >= last_24h)
            edges_24h = sum(1 for edge in self.edges.values() 
                           if edge.created_at >= last_24h)
            
            # Count nodes/edges created in last 7 days
            last_7d = now - timedelta(days=7)
            nodes_7d = sum(1 for node in self.nodes.values() 
                          if node.created_at >= last_7d)
            edges_7d = sum(1 for edge in self.edges.values() 
                          if edge.created_at >= last_7d)
            
            return {
                "nodes_per_day": nodes_24h,
                "edges_per_day": edges_24h,
                "nodes_per_week": nodes_7d,
                "edges_per_week": edges_7d
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating growth rate: {e}")
            return {}
    
    def _invalidate_analytics_cache(self):
        """Invalidate analytics cache"""
        self.analytics_cache.clear()
    
    def _update_neo4j_node(self, node: GraphNode):
        """Update node in Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                session.write_transaction(self._create_node_tx, node)
        except Exception as e:
            self.logger.error(f"Error updating Neo4j node: {e}")
    
    def _update_neo4j_edge(self, edge: GraphEdge):
        """Update edge in Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                session.write_transaction(self._create_edge_tx, edge)
        except Exception as e:
            self.logger.error(f"Error updating Neo4j edge: {e}")
    
    @staticmethod
    def _create_node_tx(tx, node: GraphNode):
        """Neo4j transaction to create/update node"""
        query = """
        MERGE (n:ThreatNode {id: $id})
        SET n.type = $type,
            n.label = $label,
            n.properties = $properties,
            n.risk_score = $risk_score,
            n.confidence = $confidence,
            n.created_at = $created_at,
            n.updated_at = $updated_at
        """
        tx.run(query, 
               id=node.id,
               type=node.type.value,
               label=node.label,
               properties=json.dumps(node.properties),
               risk_score=node.risk_score,
               confidence=node.confidence,
               created_at=node.created_at.isoformat(),
               updated_at=node.updated_at.isoformat())
    
    @staticmethod
    def _create_edge_tx(tx, edge: GraphEdge):
        """Neo4j transaction to create/update edge"""
        query = """
        MATCH (a:ThreatNode {id: $source}), (b:ThreatNode {id: $target})
        MERGE (a)-[r:THREAT_RELATION {id: $id}]->(b)
        SET r.type = $type,
            r.weight = $weight,
            r.properties = $properties,
            r.created_at = $created_at
        """
        tx.run(query,
               id=edge.id,
               source=edge.source,
               target=edge.target,
               type=edge.type.value,
               weight=edge.weight,
               properties=json.dumps(edge.properties),
               created_at=edge.created_at.isoformat())
    
    def _graph_analytics_worker(self):
        """Background worker for graph analytics"""
        while True:
            try:
                # Refresh analytics every 5 minutes
                time.sleep(300)
                analytics = self.get_analytics()
                
                # Store in Redis for caching
                self.redis_client.setex(
                    "graph_analytics", 
                    300, 
                    json.dumps(analytics, cls=PlotlyJSONEncoder)
                )
                
                # Notify subscribers
                self._notify_subscribers("analytics_updated", analytics)
                
            except Exception as e:
                self.logger.error(f"Error in graph analytics worker: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _threat_correlation_worker(self):
        """Background worker for threat correlation"""
        while True:
            try:
                # Run correlation analysis every 10 minutes
                time.sleep(600)
                
                # Find correlated threats
                correlations = self._find_threat_correlations()
                
                # Notify subscribers
                self._notify_subscribers("correlations_updated", correlations)
                
            except Exception as e:
                self.logger.error(f"Error in threat correlation worker: {e}")
                time.sleep(60)
    
    def _anomaly_detection_worker(self):
        """Background worker for anomaly detection"""
        while True:
            try:
                # Run anomaly detection every 15 minutes
                time.sleep(900)
                
                # Detect anomalies
                anomalies = self._detect_graph_anomalies()
                
                # Notify subscribers
                self._notify_subscribers("anomalies_detected", anomalies)
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection worker: {e}")
                time.sleep(60)
    
    def _find_threat_correlations(self) -> List[Dict[str, Any]]:
        """Find correlated threats in the graph"""
        try:
            correlations = []
            
            # Find nodes with similar properties
            for node1 in self.nodes.values():
                for node2 in self.nodes.values():
                    if node1.id >= node2.id:  # Avoid duplicates
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculate_node_similarity(node1, node2)
                    
                    if similarity > 0.8:  # High similarity threshold
                        correlations.append({
                            "node1": node1.id,
                            "node2": node2.id,
                            "similarity": similarity,
                            "type": "property_similarity"
                        })
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error finding threat correlations: {e}")
            return []
    
    def _calculate_node_similarity(self, node1: GraphNode, node2: GraphNode) -> float:
        """Calculate similarity between two nodes"""
        try:
            # Type similarity
            type_sim = 1.0 if node1.type == node2.type else 0.0
            
            # Property similarity (Jaccard similarity)
            props1 = set(str(v) for v in node1.properties.values())
            props2 = set(str(v) for v in node2.properties.values())
            
            if len(props1) == 0 and len(props2) == 0:
                prop_sim = 1.0
            elif len(props1) == 0 or len(props2) == 0:
                prop_sim = 0.0
            else:
                prop_sim = len(props1 & props2) / len(props1 | props2)
            
            # Risk similarity
            risk_sim = 1.0 - abs(node1.risk_score - node2.risk_score) / 10.0
            
            # Weighted average
            return 0.4 * type_sim + 0.4 * prop_sim + 0.2 * risk_sim
            
        except Exception as e:
            self.logger.error(f"Error calculating node similarity: {e}")
            return 0.0


class GraphDashboard:
    """
    Flask-based web dashboard for graph visualization
    """
    
    def __init__(self, graph: ThreatIntelligenceGraph, host: str = "0.0.0.0", 
                 port: int = 5000):
        self.graph = graph
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'your-secret-key-here'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Enable CORS
        CORS(self.app)
        
        # Subscribe to graph updates
        self.graph.subscribe(self._on_graph_update)
        
        # Setup routes
        self._setup_routes()
        self._setup_websocket_handlers()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('graph_dashboard.html')
        
        @self.app.route('/api/graph/subgraph/<node_id>')
        def get_subgraph(node_id):
            radius = int(request.args.get('radius', 2))
            max_nodes = int(request.args.get('max_nodes', 100))
            
            subgraph = self.graph.get_subgraph(node_id, radius, max_nodes)
            return jsonify(subgraph)
        
        @self.app.route('/api/graph/analytics')
        def get_analytics():
            analytics = self.graph.get_analytics()
            return jsonify(analytics)
        
        @self.app.route('/api/graph/search')
        def search_nodes():
            query = request.args.get('q', '')
            node_types = request.args.getlist('types')
            limit = int(request.args.get('limit', 50))
            
            # Convert string types to NodeType enums
            enum_types = []
            for type_str in node_types:
                try:
                    enum_types.append(NodeType(type_str))
                except ValueError:
                    continue
            
            results = self.graph.search_nodes(query, enum_types, limit)
            return jsonify(results)
        
        @self.app.route('/api/graph/attack-paths')
        def get_attack_paths():
            source = request.args.get('source')
            target = request.args.get('target')
            max_paths = int(request.args.get('max_paths', 10))
            
            if not source or not target:
                return jsonify({"error": "Source and target required"}), 400
            
            paths = self.graph.get_attack_paths(source, target, max_paths)
            return jsonify(paths)
        
        @self.app.route('/api/graph/visualization/<node_id>')
        def get_visualization_data(node_id):
            """Get data formatted for D3.js visualization"""
            subgraph = self.graph.get_subgraph(node_id, 2, 100)
            
            # Format for D3.js
            d3_data = {
                "nodes": [],
                "links": []
            }
            
            # Process nodes
            for node in subgraph["nodes"]:
                d3_node = {
                    "id": node["id"],
                    "label": node["label"],
                    "type": node["type"],
                    "risk_score": node["risk_score"],
                    "confidence": node["confidence"],
                    "properties": node["properties"],
                    "size": max(10, min(50, node["risk_score"] * 5)),
                    "color": self._get_node_color(node["type"], node["risk_score"])
                }
                d3_data["nodes"].append(d3_node)
            
            # Process edges
            for edge in subgraph["edges"]:
                d3_link = {
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": edge["type"],
                    "weight": edge["weight"],
                    "properties": edge["properties"],
                    "color": self._get_edge_color(edge["type"])
                }
                d3_data["links"].append(d3_link)
            
            return jsonify(d3_data)
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers"""
        
        @self.socketio.on('connect')
        def on_connect():
            self.logger.info(f"Client connected: {request.sid}")
            join_room('graph_updates')
        
        @self.socketio.on('disconnect')
        def on_disconnect():
            self.logger.info(f"Client disconnected: {request.sid}")
            leave_room('graph_updates')
        
        @self.socketio.on('subscribe_node')
        def on_subscribe_node(data):
            node_id = data.get('node_id')
            if node_id:
                join_room(f'node_{node_id}')
                self.logger.info(f"Client {request.sid} subscribed to node {node_id}")
        
        @self.socketio.on('unsubscribe_node')
        def on_unsubscribe_node(data):
            node_id = data.get('node_id')
            if node_id:
                leave_room(f'node_{node_id}')
                self.logger.info(f"Client {request.sid} unsubscribed from node {node_id}")
    
    def _on_graph_update(self, event_type: str, data: Dict[str, Any]):
        """Handle graph updates and broadcast to clients"""
        try:
            if event_type in ['node_added', 'node_updated']:
                node_id = data.get('id')
                if node_id:
                    self.socketio.emit('node_update', data, room=f'node_{node_id}')
            
            elif event_type in ['edge_added', 'edge_updated']:
                self.socketio.emit('edge_update', data, room='graph_updates')
            
            elif event_type == 'analytics_updated':
                self.socketio.emit('analytics_update', data, room='graph_updates')
            
            elif event_type == 'anomalies_detected':
                self.socketio.emit('anomalies_update', data, room='graph_updates')
            
            elif event_type == 'correlations_updated':
                self.socketio.emit('correlations_update', data, room='graph_updates')
            
        except Exception as e:
            self.logger.error(f"Error handling graph update: {e}")
    
    def _get_node_color(self, node_type: str, risk_score: float) -> str:
        """Get color for node based on type and risk score"""
        # Base colors by type
        type_colors = {
            "host": "#1f77b4",
            "domain": "#ff7f0e", 
            "ip_address": "#2ca02c",
            "vulnerability": "#d62728",
            "malware": "#9467bd",
            "threat_actor": "#8c564b",
            "campaign": "#e377c2",
            "technique": "#7f7f7f",
            "indicator": "#bcbd22",
            "artifact": "#17becf",
            "network": "#aec7e8",
            "geolocation": "#ffbb78"
        }
        
        base_color = type_colors.get(node_type, "#cccccc")
        
        # Adjust intensity based on risk score
        if risk_score > 8:
            return "#ff0000"  # Critical - red
        elif risk_score > 6:
            return "#ff6600"  # High - orange
        elif risk_score > 3:
            return "#ffcc00"  # Medium - yellow
        else:
            return base_color  # Low - type color
    
    def _get_edge_color(self, edge_type: str) -> str:
        """Get color for edge based on type"""
        type_colors = {
            "connects_to": "#666666",
            "exploits": "#ff0000",
            "hosts": "#0066cc",
            "belongs_to": "#00cc66",
            "uses": "#cc6600",
            "communicates_with": "#6600cc",
            "depends_on": "#cc0066",
            "located_in": "#66cc00",
            "similar_to": "#0066cc",
            "derived_from": "#cc6600",
            "targets": "#ff3333",
            "mitigates": "#33ff33"
        }
        
        return type_colors.get(edge_type, "#999999")
    
    def run(self, debug: bool = False):
        """Run the dashboard server"""
        self.logger.info(f"Starting Graph Dashboard on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)


def create_sample_graph() -> ThreatIntelligenceGraph:
    """Create a sample graph for testing"""
    graph = ThreatIntelligenceGraph(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        redis_url="redis://localhost:6379"
    )
    
    # Add sample nodes
    nodes = [
        GraphNode(
            id="host_1",
            type=NodeType.HOST,
            label="Corporate Server",
            properties={"ip": "192.168.1.100", "os": "Windows Server 2019"},
            risk_score=7.5,
            confidence=0.9,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        GraphNode(
            id="vuln_1",
            type=NodeType.VULNERABILITY,
            label="CVE-2021-44228",
            properties={"cvss": 10.0, "description": "Log4Shell vulnerability"},
            risk_score=9.8,
            confidence=0.95,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        GraphNode(
            id="malware_1",
            type=NodeType.MALWARE,
            label="Cobalt Strike",
            properties={"family": "beacon", "type": "trojan"},
            risk_score=8.5,
            confidence=0.85,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]
    
    # Add sample edges
    edges = [
        GraphEdge(
            id="edge_1",
            source="host_1",
            target="vuln_1",
            type=EdgeType.EXPLOITS,
            weight=0.9,
            properties={"exploit_method": "remote code execution"},
            created_at=datetime.now()
        ),
        GraphEdge(
            id="edge_2",
            source="vuln_1",
            target="malware_1",
            type=EdgeType.USES,
            weight=0.8,
            properties={"deployment_method": "payload injection"},
            created_at=datetime.now()
        )
    ]
    
    # Add to graph
    for node in nodes:
        graph.add_node(node)
    
    for edge in edges:
        graph.add_edge(edge)
    
    return graph


if __name__ == "__main__":
    # Create sample graph and run dashboard
    graph = create_sample_graph()
    dashboard = GraphDashboard(graph)
    dashboard.run(debug=True)