"""
Graph Algorithms - Advanced graph analysis algorithms for security intelligence
Implements community detection, centrality measures, and clustering algorithms
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
from neo4j.exceptions import ServiceUnavailable
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from .graph_manager import GraphManager, GraphNode, GraphEdge
from .graph_schema import NodeType, RelationType


@dataclass
class Community:
    """Represents a community in the graph"""
    community_id: str
    nodes: List[str]
    size: int
    density: float
    modularity: float
    internal_edges: int
    external_edges: int
    dominant_types: List[str]
    risk_score: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "community_id": self.community_id,
            "nodes": self.nodes,
            "size": self.size,
            "density": self.density,
            "modularity": self.modularity,
            "internal_edges": self.internal_edges,
            "external_edges": self.external_edges,
            "dominant_types": self.dominant_types,
            "risk_score": self.risk_score,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class CentralityMeasures:
    """Centrality measures for a node"""
    node_id: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    clustering_coefficient: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_id": self.node_id,
            "degree_centrality": self.degree_centrality,
            "betweenness_centrality": self.betweenness_centrality,
            "closeness_centrality": self.closeness_centrality,
            "eigenvector_centrality": self.eigenvector_centrality,
            "pagerank": self.pagerank,
            "clustering_coefficient": self.clustering_coefficient
        }


@dataclass
class GraphMetrics:
    """Overall graph metrics"""
    nodes_count: int
    edges_count: int
    density: float
    average_clustering: float
    transitivity: float
    diameter: Optional[int]
    average_path_length: Optional[float]
    connected_components: int
    modularity: float
    assortativity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "nodes_count": self.nodes_count,
            "edges_count": self.edges_count,
            "density": self.density,
            "average_clustering": self.average_clustering,
            "transitivity": self.transitivity,
            "diameter": self.diameter,
            "average_path_length": self.average_path_length,
            "connected_components": self.connected_components,
            "modularity": self.modularity,
            "assortativity": self.assortativity
        }


@dataclass
class AnomalyDetection:
    """Anomaly detection results"""
    anomaly_id: str
    node_id: str
    anomaly_type: str
    score: float
    description: str
    features: Dict[str, float]
    detected_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "anomaly_id": self.anomaly_id,
            "node_id": self.node_id,
            "anomaly_type": self.anomaly_type,
            "score": self.score,
            "description": self.description,
            "features": self.features,
            "detected_at": self.detected_at.isoformat()
        }


class GraphAlgorithms:
    """Advanced graph algorithms for security analysis"""
    
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.logger = logging.getLogger(__name__)
        
        # Caches for expensive computations
        self.centrality_cache = {}
        self.community_cache = {}
        self.metrics_cache = {}
        
        # Algorithm parameters
        self.community_resolution = 1.0
        self.centrality_k = None  # For approximate algorithms
        self.anomaly_threshold = 0.8
        self.max_workers = 4
        
        # Node type weights for scoring
        self.node_weights = {
            NodeType.HOST: 1.0,
            NodeType.VULNERABILITY: 2.0,
            NodeType.THREAT: 3.0,
            NodeType.ATTACK: 2.5,
            NodeType.CREDENTIAL: 1.5,
            NodeType.SERVICE: 1.0,
            NodeType.USER: 1.2
        }
    
    async def detect_communities(self, algorithm: str = "louvain", 
                               resolution: float = 1.0) -> List[Community]:
        """Detect communities in the graph using various algorithms"""
        try:
            # Check cache
            cache_key = f"{algorithm}_{resolution}"
            if cache_key in self.community_cache:
                return self.community_cache[cache_key]
            
            # Get graph
            if self.graph_manager.use_fallback:
                graph = self.graph_manager.networkx_graph
            else:
                graph = await self._build_networkx_from_neo4j()
            
            if not graph or graph.number_of_nodes() == 0:
                return []
            
            # Apply community detection algorithm
            if algorithm == "louvain":
                communities = await self._louvain_communities(graph, resolution)
            elif algorithm == "leiden":
                communities = await self._leiden_communities(graph, resolution)
            elif algorithm == "label_propagation":
                communities = await self._label_propagation_communities(graph)
            elif algorithm == "greedy_modularity":
                communities = await self._greedy_modularity_communities(graph)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Cache results
            self.community_cache[cache_key] = communities
            
            return communities
            
        except Exception as e:
            self.logger.error(f"Error detecting communities: {e}")
            return []
    
    async def _build_networkx_from_neo4j(self) -> nx.Graph:
        """Build NetworkX graph from Neo4j for analysis"""
        graph = nx.Graph()
        
        async with self.graph_manager.get_session() as session:
            if not session:
                return graph
            
            # Get all nodes
            nodes_result = await session.run("MATCH (n) RETURN n")
            async for record in nodes_result:
                node = record["n"]
                graph.add_node(node["id"], **dict(node))
            
            # Get all edges
            edges_result = await session.run("MATCH (a)-[r]->(b) RETURN a.id, b.id, r")
            async for record in edges_result:
                source = record["a.id"]
                target = record["b.id"]
                rel = record["r"]
                graph.add_edge(source, target, **dict(rel))
        
        return graph
    
    async def _louvain_communities(self, graph: nx.Graph, resolution: float) -> List[Community]:
        """Apply Louvain algorithm for community detection"""
        try:
            import community as community_louvain
            
            # Convert to undirected graph if needed
            if graph.is_directed():
                graph = graph.to_undirected()
            
            # Apply Louvain algorithm
            partition = community_louvain.best_partition(graph, resolution=resolution)
            
            # Group nodes by community
            communities_dict = defaultdict(list)
            for node, community_id in partition.items():
                communities_dict[community_id].append(node)
            
            # Create Community objects
            communities = []
            for community_id, nodes in communities_dict.items():
                community = await self._create_community(
                    f"louvain_{community_id}",
                    nodes,
                    graph,
                    partition
                )
                communities.append(community)
            
            return communities
            
        except ImportError:
            self.logger.warning("python-louvain not available, using greedy modularity")
            return await self._greedy_modularity_communities(graph)
        except Exception as e:
            self.logger.error(f"Error in Louvain algorithm: {e}")
            return []
    
    async def _leiden_communities(self, graph: nx.Graph, resolution: float) -> List[Community]:
        """Apply Leiden algorithm for community detection"""
        try:
            import leidenalg
            import igraph as ig
            
            # Convert NetworkX to igraph
            ig_graph = ig.Graph.from_networkx(graph)
            
            # Apply Leiden algorithm
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                resolution_parameter=resolution
            )
            
            # Convert back to NetworkX node IDs
            communities = []
            for i, community_nodes in enumerate(partition):
                node_names = [ig_graph.vs[node]["_nx_name"] for node in community_nodes]
                community = await self._create_community(
                    f"leiden_{i}",
                    node_names,
                    graph
                )
                communities.append(community)
            
            return communities
            
        except ImportError:
            self.logger.warning("leidenalg not available, using Louvain fallback")
            return await self._louvain_communities(graph, resolution)
        except Exception as e:
            self.logger.error(f"Error in Leiden algorithm: {e}")
            return []
    
    async def _label_propagation_communities(self, graph: nx.Graph) -> List[Community]:
        """Apply label propagation algorithm"""
        try:
            # Convert to undirected graph if needed
            if graph.is_directed():
                graph = graph.to_undirected()
            
            # Apply label propagation
            communities_gen = nx.algorithms.community.label_propagation_communities(graph)
            
            # Create Community objects
            communities = []
            for i, community_nodes in enumerate(communities_gen):
                community = await self._create_community(
                    f"label_prop_{i}",
                    list(community_nodes),
                    graph
                )
                communities.append(community)
            
            return communities
            
        except Exception as e:
            self.logger.error(f"Error in label propagation: {e}")
            return []
    
    async def _greedy_modularity_communities(self, graph: nx.Graph) -> List[Community]:
        """Apply greedy modularity optimization"""
        try:
            # Convert to undirected graph if needed
            if graph.is_directed():
                graph = graph.to_undirected()
            
            # Apply greedy modularity
            communities_gen = nx.algorithms.community.greedy_modularity_communities(graph)
            
            # Create Community objects
            communities = []
            for i, community_nodes in enumerate(communities_gen):
                community = await self._create_community(
                    f"greedy_mod_{i}",
                    list(community_nodes),
                    graph
                )
                communities.append(community)
            
            return communities
            
        except Exception as e:
            self.logger.error(f"Error in greedy modularity: {e}")
            return []
    
    async def _create_community(self, community_id: str, nodes: List[str], 
                              graph: nx.Graph, partition: Optional[Dict] = None) -> Community:
        """Create Community object with calculated metrics"""
        # Calculate community metrics
        subgraph = graph.subgraph(nodes)
        
        # Basic metrics
        size = len(nodes)
        internal_edges = subgraph.number_of_edges()
        
        # Calculate external edges
        external_edges = 0
        for node in nodes:
            for neighbor in graph.neighbors(node):
                if neighbor not in nodes:
                    external_edges += 1
        
        # Calculate density
        max_edges = size * (size - 1) // 2
        density = internal_edges / max_edges if max_edges > 0 else 0
        
        # Calculate modularity (if partition available)
        modularity = 0.0
        if partition:
            modularity = nx.algorithms.community.modularity(graph, [nodes])
        
        # Get node types
        node_types = []
        for node in nodes:
            node_data = graph.nodes.get(node, {})
            node_type = node_data.get("type", "unknown")
            node_types.append(node_type)
        
        # Find dominant types
        type_counts = Counter(node_types)
        dominant_types = [t for t, c in type_counts.most_common(3)]
        
        # Calculate risk score
        risk_score = await self._calculate_community_risk_score(nodes, graph)
        
        return Community(
            community_id=community_id,
            nodes=nodes,
            size=size,
            density=density,
            modularity=modularity,
            internal_edges=internal_edges,
            external_edges=external_edges,
            dominant_types=dominant_types,
            risk_score=risk_score,
            created_at=datetime.now()
        )
    
    async def _calculate_community_risk_score(self, nodes: List[str], graph: nx.Graph) -> float:
        """Calculate risk score for a community"""
        total_score = 0.0
        
        for node in nodes:
            node_data = graph.nodes.get(node, {})
            node_type_str = node_data.get("type", "unknown")
            
            # Get node type weight
            try:
                node_type = NodeType(node_type_str)
                weight = self.node_weights.get(node_type, 1.0)
            except ValueError:
                weight = 1.0
            
            # Add vulnerability-specific scoring
            if node_type_str == "vulnerability":
                severity = node_data.get("severity", "medium")
                if severity == "critical":
                    weight *= 3.0
                elif severity == "high":
                    weight *= 2.0
                elif severity == "medium":
                    weight *= 1.5
            
            # Add threat-specific scoring
            elif node_type_str == "threat":
                severity_score = node_data.get("severity_score", 50.0)
                weight *= (severity_score / 50.0)
            
            total_score += weight
        
        # Normalize by community size
        return min(100.0, total_score / len(nodes) * 10)
    
    async def calculate_centrality_measures(self, node_ids: Optional[List[str]] = None) -> List[CentralityMeasures]:
        """Calculate centrality measures for nodes"""
        try:
            # Get graph
            if self.graph_manager.use_fallback:
                graph = self.graph_manager.networkx_graph
            else:
                graph = await self._build_networkx_from_neo4j()
            
            if not graph or graph.number_of_nodes() == 0:
                return []
            
            # Convert to undirected for some algorithms
            undirected_graph = graph.to_undirected() if graph.is_directed() else graph
            
            # Calculate centrality measures
            centrality_results = []
            
            # Use ThreadPoolExecutor for parallel computation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit centrality calculations
                degree_future = executor.submit(self._calculate_degree_centrality, undirected_graph)
                betweenness_future = executor.submit(self._calculate_betweenness_centrality, undirected_graph)
                closeness_future = executor.submit(self._calculate_closeness_centrality, undirected_graph)
                eigenvector_future = executor.submit(self._calculate_eigenvector_centrality, undirected_graph)
                pagerank_future = executor.submit(self._calculate_pagerank, graph)
                clustering_future = executor.submit(self._calculate_clustering_coefficient, undirected_graph)
                
                # Get results
                degree_centrality = degree_future.result()
                betweenness_centrality = betweenness_future.result()
                closeness_centrality = closeness_future.result()
                eigenvector_centrality = eigenvector_future.result()
                pagerank = pagerank_future.result()
                clustering_coefficient = clustering_future.result()
            
            # Create CentralityMeasures objects
            target_nodes = node_ids if node_ids else list(graph.nodes())
            
            for node_id in target_nodes:
                if node_id in graph.nodes():
                    measures = CentralityMeasures(
                        node_id=node_id,
                        degree_centrality=degree_centrality.get(node_id, 0.0),
                        betweenness_centrality=betweenness_centrality.get(node_id, 0.0),
                        closeness_centrality=closeness_centrality.get(node_id, 0.0),
                        eigenvector_centrality=eigenvector_centrality.get(node_id, 0.0),
                        pagerank=pagerank.get(node_id, 0.0),
                        clustering_coefficient=clustering_coefficient.get(node_id, 0.0)
                    )
                    centrality_results.append(measures)
            
            return centrality_results
            
        except Exception as e:
            self.logger.error(f"Error calculating centrality measures: {e}")
            return []
    
    def _calculate_degree_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate degree centrality"""
        return nx.degree_centrality(graph)
    
    def _calculate_betweenness_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate betweenness centrality"""
        return nx.betweenness_centrality(graph, k=self.centrality_k)
    
    def _calculate_closeness_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate closeness centrality"""
        return nx.closeness_centrality(graph)
    
    def _calculate_eigenvector_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate eigenvector centrality"""
        try:
            return nx.eigenvector_centrality(graph, max_iter=1000)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            # Fallback to degree centrality if eigenvector fails
            return self._calculate_degree_centrality(graph)
    
    def _calculate_pagerank(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate PageRank"""
        return nx.pagerank(graph)
    
    def _calculate_clustering_coefficient(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate clustering coefficient"""
        return nx.clustering(graph)
    
    async def calculate_graph_metrics(self) -> GraphMetrics:
        """Calculate overall graph metrics"""
        try:
            # Check cache
            if "graph_metrics" in self.metrics_cache:
                return self.metrics_cache["graph_metrics"]
            
            # Get graph
            if self.graph_manager.use_fallback:
                graph = self.graph_manager.networkx_graph
            else:
                graph = await self._build_networkx_from_neo4j()
            
            if not graph or graph.number_of_nodes() == 0:
                return GraphMetrics(0, 0, 0, 0, 0, None, None, 0, 0, 0)
            
            # Convert to undirected for some metrics
            undirected_graph = graph.to_undirected() if graph.is_directed() else graph
            
            # Calculate metrics
            nodes_count = graph.number_of_nodes()
            edges_count = graph.number_of_edges()
            density = nx.density(graph)
            
            # Clustering metrics
            try:
                average_clustering = nx.average_clustering(undirected_graph)
                transitivity = nx.transitivity(undirected_graph)
            except:
                average_clustering = 0.0
                transitivity = 0.0
            
            # Path-based metrics (expensive for large graphs)
            diameter = None
            average_path_length = None
            
            if nodes_count < 1000:  # Only calculate for smaller graphs
                try:
                    if nx.is_connected(undirected_graph):
                        diameter = nx.diameter(undirected_graph)
                        average_path_length = nx.average_shortest_path_length(undirected_graph)
                except:
                    pass
            
            # Component analysis
            connected_components = nx.number_connected_components(undirected_graph)
            
            # Modularity (requires community detection)
            modularity = 0.0
            try:
                communities = nx.algorithms.community.greedy_modularity_communities(undirected_graph)
                modularity = nx.algorithms.community.modularity(undirected_graph, communities)
            except:
                pass
            
            # Assortativity
            assortativity = 0.0
            try:
                assortativity = nx.degree_assortativity_coefficient(undirected_graph)
            except:
                pass
            
            metrics = GraphMetrics(
                nodes_count=nodes_count,
                edges_count=edges_count,
                density=density,
                average_clustering=average_clustering,
                transitivity=transitivity,
                diameter=diameter,
                average_path_length=average_path_length,
                connected_components=connected_components,
                modularity=modularity,
                assortativity=assortativity
            )
            
            # Cache results
            self.metrics_cache["graph_metrics"] = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating graph metrics: {e}")
            return GraphMetrics(0, 0, 0, 0, 0, None, None, 0, 0, 0)
    
    async def detect_anomalies(self, method: str = "isolation_forest", 
                             features: Optional[List[str]] = None) -> List[AnomalyDetection]:
        """Detect anomalous nodes using various methods"""
        try:
            # Get centrality measures for all nodes
            centrality_measures = await self.calculate_centrality_measures()
            
            if not centrality_measures:
                return []
            
            # Prepare feature matrix
            feature_names = features or [
                "degree_centrality",
                "betweenness_centrality",
                "closeness_centrality",
                "eigenvector_centrality",
                "pagerank",
                "clustering_coefficient"
            ]
            
            feature_matrix = []
            node_ids = []
            
            for measure in centrality_measures:
                feature_vector = []
                for feature_name in feature_names:
                    feature_vector.append(getattr(measure, feature_name, 0.0))
                
                feature_matrix.append(feature_vector)
                node_ids.append(measure.node_id)
            
            if not feature_matrix:
                return []
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Apply anomaly detection method
            if method == "isolation_forest":
                anomalies = await self._isolation_forest_anomalies(scaled_features, node_ids, feature_names)
            elif method == "local_outlier_factor":
                anomalies = await self._lof_anomalies(scaled_features, node_ids, feature_names)
            elif method == "statistical":
                anomalies = await self._statistical_anomalies(scaled_features, node_ids, feature_names)
            else:
                raise ValueError(f"Unknown anomaly detection method: {method}")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def _isolation_forest_anomalies(self, features: np.ndarray, node_ids: List[str], 
                                        feature_names: List[str]) -> List[AnomalyDetection]:
        """Detect anomalies using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(features)
            anomaly_scores_normalized = iso_forest.decision_function(features)
            
            # Create anomaly objects
            anomalies = []
            for i, (node_id, score, norm_score) in enumerate(zip(node_ids, anomaly_scores, anomaly_scores_normalized)):
                if score == -1:  # Anomaly detected
                    # Get feature values for this node
                    feature_dict = {name: float(features[i][j]) for j, name in enumerate(feature_names)}
                    
                    anomaly = AnomalyDetection(
                        anomaly_id=f"iso_forest_{node_id}",
                        node_id=node_id,
                        anomaly_type="isolation_forest",
                        score=abs(norm_score),
                        description=f"Node detected as anomaly by Isolation Forest",
                        features=feature_dict,
                        detected_at=datetime.now()
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except ImportError:
            self.logger.warning("scikit-learn not available for Isolation Forest")
            return []
        except Exception as e:
            self.logger.error(f"Error in Isolation Forest anomaly detection: {e}")
            return []
    
    async def _lof_anomalies(self, features: np.ndarray, node_ids: List[str], 
                           feature_names: List[str]) -> List[AnomalyDetection]:
        """Detect anomalies using Local Outlier Factor"""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # Fit LOF
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            anomaly_scores = lof.fit_predict(features)
            negative_outlier_factor = lof.negative_outlier_factor_
            
            # Create anomaly objects
            anomalies = []
            for i, (node_id, score, nof) in enumerate(zip(node_ids, anomaly_scores, negative_outlier_factor)):
                if score == -1:  # Anomaly detected
                    # Get feature values for this node
                    feature_dict = {name: float(features[i][j]) for j, name in enumerate(feature_names)}
                    
                    anomaly = AnomalyDetection(
                        anomaly_id=f"lof_{node_id}",
                        node_id=node_id,
                        anomaly_type="local_outlier_factor",
                        score=abs(nof),
                        description=f"Node detected as anomaly by Local Outlier Factor",
                        features=feature_dict,
                        detected_at=datetime.now()
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except ImportError:
            self.logger.warning("scikit-learn not available for LOF")
            return []
        except Exception as e:
            self.logger.error(f"Error in LOF anomaly detection: {e}")
            return []
    
    async def _statistical_anomalies(self, features: np.ndarray, node_ids: List[str], 
                                   feature_names: List[str]) -> List[AnomalyDetection]:
        """Detect anomalies using statistical methods"""
        try:
            anomalies = []
            
            # Calculate z-scores for each feature
            for j, feature_name in enumerate(feature_names):
                feature_values = features[:, j]
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                
                if std == 0:
                    continue
                
                z_scores = np.abs((feature_values - mean) / std)
                
                # Find anomalies (z-score > 3)
                anomaly_indices = np.where(z_scores > 3)[0]
                
                for idx in anomaly_indices:
                    node_id = node_ids[idx]
                    z_score = z_scores[idx]
                    
                    # Get feature values for this node
                    feature_dict = {name: float(features[idx][k]) for k, name in enumerate(feature_names)}
                    
                    anomaly = AnomalyDetection(
                        anomaly_id=f"statistical_{node_id}_{feature_name}",
                        node_id=node_id,
                        anomaly_type="statistical",
                        score=z_score,
                        description=f"Node has anomalous {feature_name} (z-score: {z_score:.2f})",
                        features=feature_dict,
                        detected_at=datetime.now()
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in statistical anomaly detection: {e}")
            return []
    
    async def cluster_nodes(self, method: str = "kmeans", n_clusters: int = 5) -> Dict[str, List[str]]:
        """Cluster nodes based on their properties"""
        try:
            # Get centrality measures for all nodes
            centrality_measures = await self.calculate_centrality_measures()
            
            if not centrality_measures:
                return {}
            
            # Prepare feature matrix
            feature_matrix = []
            node_ids = []
            
            for measure in centrality_measures:
                feature_vector = [
                    measure.degree_centrality,
                    measure.betweenness_centrality,
                    measure.closeness_centrality,
                    measure.eigenvector_centrality,
                    measure.pagerank,
                    measure.clustering_coefficient
                ]
                
                feature_matrix.append(feature_vector)
                node_ids.append(measure.node_id)
            
            if not feature_matrix:
                return {}
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Apply clustering method
            if method == "kmeans":
                labels = await self._kmeans_clustering(scaled_features, n_clusters)
            elif method == "dbscan":
                labels = await self._dbscan_clustering(scaled_features)
            elif method == "hierarchical":
                labels = await self._hierarchical_clustering(scaled_features, n_clusters)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Group nodes by cluster
            clusters = defaultdict(list)
            for node_id, label in zip(node_ids, labels):
                if label != -1:  # Ignore noise points in DBSCAN
                    clusters[f"cluster_{label}"].append(node_id)
            
            return dict(clusters)
            
        except Exception as e:
            self.logger.error(f"Error clustering nodes: {e}")
            return {}
    
    async def _kmeans_clustering(self, features: np.ndarray, n_clusters: int) -> List[int]:
        """Perform K-means clustering"""
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features)
            return labels.tolist()
        except Exception as e:
            self.logger.error(f"Error in K-means clustering: {e}")
            return []
    
    async def _dbscan_clustering(self, features: np.ndarray) -> List[int]:
        """Perform DBSCAN clustering"""
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(features)
            return labels.tolist()
        except Exception as e:
            self.logger.error(f"Error in DBSCAN clustering: {e}")
            return []
    
    async def _hierarchical_clustering(self, features: np.ndarray, n_clusters: int) -> List[int]:
        """Perform hierarchical clustering"""
        try:
            # Calculate distance matrix
            distances = pdist(features, metric='euclidean')
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distances, method='ward')
            
            # Get cluster labels
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            return (labels - 1).tolist()  # Convert to 0-based indexing
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical clustering: {e}")
            return []
    
    async def find_critical_nodes(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find the most critical nodes based on multiple centrality measures"""
        try:
            # Get centrality measures for all nodes
            centrality_measures = await self.calculate_centrality_measures()
            
            if not centrality_measures:
                return []
            
            # Calculate composite criticality score
            critical_nodes = []
            
            for measure in centrality_measures:
                # Weighted combination of centrality measures
                criticality_score = (
                    measure.degree_centrality * 0.2 +
                    measure.betweenness_centrality * 0.3 +
                    measure.closeness_centrality * 0.2 +
                    measure.eigenvector_centrality * 0.2 +
                    measure.pagerank * 0.1
                )
                
                # Get additional node information
                node = await self.graph_manager.get_node(measure.node_id)
                node_dict = node.to_dict() if node else {"id": measure.node_id}
                
                critical_nodes.append({
                    "node": node_dict,
                    "centrality_measures": measure.to_dict(),
                    "criticality_score": criticality_score
                })
            
            # Sort by criticality score and return top K
            critical_nodes.sort(key=lambda x: x["criticality_score"], reverse=True)
            
            return critical_nodes[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding critical nodes: {e}")
            return []
    
    async def clear_cache(self) -> None:
        """Clear all algorithm caches"""
        self.centrality_cache.clear()
        self.community_cache.clear()
        self.metrics_cache.clear()
        self.logger.info("Graph algorithms cache cleared")