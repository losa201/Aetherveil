"""
Analysis API Routes - Advanced graph analysis endpoints
Provides attack path analysis, vulnerability mapping, and threat intelligence
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from pydantic import BaseModel, Field
from ..graph_manager import GraphManager
from ..attack_path_analyzer import AttackPathAnalyzer
from ..vulnerability_mapper import VulnerabilityMapper
from ..graph_algorithms import GraphAlgorithms
from ..graph_analytics import GraphAnalytics
from ..graph_maintenance import GraphMaintenance
from ..graph_schema import NodeType, RelationType, AttackStage, ThreatType
from ...config import get_config

# Initialize router
analysis_router = APIRouter(prefix="/api/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)

# Global instances
graph_manager = None
attack_analyzer = None
vuln_mapper = None
graph_algorithms = None
graph_analytics = None
graph_maintenance = None

async def get_graph_manager() -> GraphManager:
    """Dependency to get graph manager instance"""
    global graph_manager
    if graph_manager is None:
        graph_manager = GraphManager()
        await graph_manager.initialize()
    return graph_manager

async def get_attack_analyzer(graph_mgr: GraphManager = Depends(get_graph_manager)) -> AttackPathAnalyzer:
    """Dependency to get attack path analyzer instance"""
    global attack_analyzer
    if attack_analyzer is None:
        attack_analyzer = AttackPathAnalyzer(graph_mgr)
    return attack_analyzer

async def get_vulnerability_mapper(graph_mgr: GraphManager = Depends(get_graph_manager)) -> VulnerabilityMapper:
    """Dependency to get vulnerability mapper instance"""
    global vuln_mapper
    if vuln_mapper is None:
        vuln_mapper = VulnerabilityMapper(graph_mgr)
    return vuln_mapper

async def get_graph_algorithms(graph_mgr: GraphManager = Depends(get_graph_manager)) -> GraphAlgorithms:
    """Dependency to get graph algorithms instance"""
    global graph_algorithms
    if graph_algorithms is None:
        graph_algorithms = GraphAlgorithms(graph_mgr)
    return graph_algorithms

async def get_graph_analytics(graph_mgr: GraphManager = Depends(get_graph_manager)) -> GraphAnalytics:
    """Dependency to get graph analytics instance"""
    global graph_analytics
    if graph_analytics is None:
        graph_analytics = GraphAnalytics(graph_mgr)
    return graph_analytics

async def get_graph_maintenance(graph_mgr: GraphManager = Depends(get_graph_manager)) -> GraphMaintenance:
    """Dependency to get graph maintenance instance"""
    global graph_maintenance
    if graph_maintenance is None:
        graph_maintenance = GraphMaintenance(graph_mgr)
    return graph_maintenance

# Pydantic models for request/response

class AttackPathQuery(BaseModel):
    """Model for attack path queries"""
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    max_paths: Optional[int] = Field(5, description="Maximum number of paths to return")
    attack_stages: Optional[List[str]] = Field(None, description="Filter by attack stages")

class AttackSurfaceQuery(BaseModel):
    """Model for attack surface analysis"""
    target_id: str = Field(..., description="Target node ID")
    max_distance: Optional[int] = Field(3, description="Maximum distance for analysis")

class VulnerabilityMappingQuery(BaseModel):
    """Model for vulnerability mapping"""
    scan_results: Optional[Dict[str, Any]] = Field(None, description="Scan results to process")

class ThreatAttributionQuery(BaseModel):
    """Model for threat attribution"""
    indicators: List[str] = Field(..., description="List of indicators to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class CommunityDetectionQuery(BaseModel):
    """Model for community detection"""
    algorithm: Optional[str] = Field("louvain", description="Algorithm to use")
    resolution: Optional[float] = Field(1.0, description="Resolution parameter")

class CentralityQuery(BaseModel):
    """Model for centrality analysis"""
    node_ids: Optional[List[str]] = Field(None, description="Specific nodes to analyze")

class AnomalyDetectionQuery(BaseModel):
    """Model for anomaly detection"""
    method: Optional[str] = Field("isolation_forest", description="Detection method")
    features: Optional[List[str]] = Field(None, description="Features to use")

class ClusteringQuery(BaseModel):
    """Model for node clustering"""
    method: Optional[str] = Field("kmeans", description="Clustering method")
    n_clusters: Optional[int] = Field(5, description="Number of clusters")

# Attack path analysis endpoints

@analysis_router.post("/attack_paths", response_model=Dict[str, Any])
async def find_attack_paths(
    query: AttackPathQuery,
    analyzer: AttackPathAnalyzer = Depends(get_attack_analyzer)
):
    """Find attack paths between two nodes"""
    try:
        # Parse attack stages
        attack_stages = None
        if query.attack_stages:
            attack_stages = []
            for stage_str in query.attack_stages:
                try:
                    attack_stages.append(AttackStage(stage_str))
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid attack stage: {stage_str}")
        
        # Find attack paths
        paths = await analyzer.find_shortest_attack_paths(
            source_node_id=query.source_id,
            target_node_id=query.target_id,
            max_paths=query.max_paths,
            attack_stages=attack_stages
        )
        
        return {
            "success": True,
            "source_id": query.source_id,
            "target_id": query.target_id,
            "paths": [path.to_dict() for path in paths],
            "count": len(paths),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding attack paths: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/attack_surface", response_model=Dict[str, Any])
async def analyze_attack_surface(
    query: AttackSurfaceQuery,
    analyzer: AttackPathAnalyzer = Depends(get_attack_analyzer)
):
    """Analyze attack surface of a target node"""
    try:
        analysis = await analyzer.analyze_attack_surface(
            target_node_id=query.target_id,
            max_distance=query.max_distance
        )
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing attack surface: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/attack_vectors/{node_id}", response_model=Dict[str, Any])
async def get_attack_vectors(
    node_id: str = Path(..., description="Node ID"),
    max_distance: int = Query(2, description="Maximum distance for analysis"),
    analyzer: AttackPathAnalyzer = Depends(get_attack_analyzer)
):
    """Get attack vectors from a given node"""
    try:
        vectors = await analyzer.find_attack_vectors(
            node_id=node_id,
            max_distance=max_distance
        )
        
        return {
            "success": True,
            "node_id": node_id,
            "vectors": [vector.to_dict() for vector in vectors],
            "count": len(vectors)
        }
        
    except Exception as e:
        logger.error(f"Error getting attack vectors: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/critical_paths", response_model=Dict[str, Any])
async def get_critical_paths(
    min_risk_score: float = Query(70.0, description="Minimum risk score"),
    analyzer: AttackPathAnalyzer = Depends(get_attack_analyzer)
):
    """Get all critical attack paths"""
    try:
        paths = await analyzer.get_critical_attack_paths(min_risk_score=min_risk_score)
        
        return {
            "success": True,
            "paths": [path.to_dict() for path in paths],
            "count": len(paths),
            "min_risk_score": min_risk_score
        }
        
    except Exception as e:
        logger.error(f"Error getting critical paths: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Vulnerability mapping endpoints

@analysis_router.post("/vulnerability_mapping", response_model=Dict[str, Any])
async def map_vulnerabilities(
    query: VulnerabilityMappingQuery,
    mapper: VulnerabilityMapper = Depends(get_vulnerability_mapper)
):
    """Map vulnerabilities to assets"""
    try:
        mappings = await mapper.map_vulnerabilities_to_assets(
            scan_results=query.scan_results
        )
        
        return {
            "success": True,
            "mappings": [mapping.to_dict() for mapping in mappings],
            "count": len(mappings)
        }
        
    except Exception as e:
        logger.error(f"Error mapping vulnerabilities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/threat_attribution", response_model=Dict[str, Any])
async def analyze_threat_attribution(
    query: ThreatAttributionQuery,
    mapper: VulnerabilityMapper = Depends(get_vulnerability_mapper)
):
    """Analyze threat attribution based on indicators"""
    try:
        attributions = await mapper.analyze_threat_attribution(
            indicators=query.indicators,
            context=query.context
        )
        
        return {
            "success": True,
            "attributions": [attribution.to_dict() for attribution in attributions],
            "count": len(attributions)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing threat attribution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/vulnerability_intelligence/{cve_id}", response_model=Dict[str, Any])
async def get_vulnerability_intelligence(
    cve_id: str = Path(..., description="CVE ID"),
    mapper: VulnerabilityMapper = Depends(get_vulnerability_mapper)
):
    """Get vulnerability intelligence for a CVE"""
    try:
        intelligence = await mapper.get_vulnerability_intelligence(cve_id)
        
        if not intelligence:
            raise HTTPException(status_code=404, detail="Vulnerability intelligence not found")
        
        return {
            "success": True,
            "intelligence": intelligence.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vulnerability intelligence: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/vulnerability_threat_correlation", response_model=Dict[str, Any])
async def correlate_vulnerabilities_threats(
    limit: int = Query(100, description="Maximum number of results"),
    mapper: VulnerabilityMapper = Depends(get_vulnerability_mapper)
):
    """Correlate vulnerabilities with known threats"""
    try:
        correlations = await mapper.correlate_vulnerabilities_with_threats(limit=limit)
        
        return {
            "success": True,
            "correlations": correlations,
            "count": len(correlations)
        }
        
    except Exception as e:
        logger.error(f"Error correlating vulnerabilities with threats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Graph algorithm endpoints

@analysis_router.post("/communities", response_model=Dict[str, Any])
async def detect_communities(
    query: CommunityDetectionQuery,
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Detect communities in the graph"""
    try:
        communities = await algorithms.detect_communities(
            algorithm=query.algorithm,
            resolution=query.resolution
        )
        
        return {
            "success": True,
            "communities": [community.to_dict() for community in communities],
            "count": len(communities),
            "algorithm": query.algorithm,
            "resolution": query.resolution
        }
        
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/centrality", response_model=Dict[str, Any])
async def calculate_centrality(
    query: CentralityQuery,
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Calculate centrality measures for nodes"""
    try:
        centrality_measures = await algorithms.calculate_centrality_measures(
            node_ids=query.node_ids
        )
        
        return {
            "success": True,
            "centrality_measures": [measure.to_dict() for measure in centrality_measures],
            "count": len(centrality_measures)
        }
        
    except Exception as e:
        logger.error(f"Error calculating centrality: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/metrics", response_model=Dict[str, Any])
async def get_graph_metrics(
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Get overall graph metrics"""
    try:
        metrics = await algorithms.calculate_graph_metrics()
        
        return {
            "success": True,
            "metrics": metrics.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error getting graph metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/anomalies", response_model=Dict[str, Any])
async def detect_anomalies(
    query: AnomalyDetectionQuery,
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Detect anomalous nodes in the graph"""
    try:
        anomalies = await algorithms.detect_anomalies(
            method=query.method,
            features=query.features
        )
        
        return {
            "success": True,
            "anomalies": [anomaly.to_dict() for anomaly in anomalies],
            "count": len(anomalies),
            "method": query.method
        }
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/clustering", response_model=Dict[str, Any])
async def cluster_nodes(
    query: ClusteringQuery,
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Cluster nodes based on their properties"""
    try:
        clusters = await algorithms.cluster_nodes(
            method=query.method,
            n_clusters=query.n_clusters
        )
        
        return {
            "success": True,
            "clusters": clusters,
            "cluster_count": len(clusters),
            "method": query.method
        }
        
    except Exception as e:
        logger.error(f"Error clustering nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/critical_nodes", response_model=Dict[str, Any])
async def find_critical_nodes(
    top_k: int = Query(10, description="Number of top critical nodes to return"),
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Find the most critical nodes in the graph"""
    try:
        critical_nodes = await algorithms.find_critical_nodes(top_k=top_k)
        
        return {
            "success": True,
            "critical_nodes": critical_nodes,
            "count": len(critical_nodes),
            "top_k": top_k
        }
        
    except Exception as e:
        logger.error(f"Error finding critical nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Cache management endpoints

@analysis_router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_analysis_cache(
    analyzer: AttackPathAnalyzer = Depends(get_attack_analyzer),
    mapper: VulnerabilityMapper = Depends(get_vulnerability_mapper),
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Clear all analysis caches"""
    try:
        await analyzer.clear_cache()
        await mapper.clear_cache()
        await algorithms.clear_cache()
        
        return {
            "success": True,
            "message": "All analysis caches cleared"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Batch analysis endpoints

@analysis_router.post("/batch/attack_paths", response_model=Dict[str, Any])
async def batch_attack_paths(
    queries: List[AttackPathQuery],
    analyzer: AttackPathAnalyzer = Depends(get_attack_analyzer)
):
    """Analyze multiple attack paths in batch"""
    try:
        results = []
        errors = []
        
        for i, query in enumerate(queries):
            try:
                # Parse attack stages
                attack_stages = None
                if query.attack_stages:
                    attack_stages = []
                    for stage_str in query.attack_stages:
                        attack_stages.append(AttackStage(stage_str))
                
                # Find attack paths
                paths = await analyzer.find_shortest_attack_paths(
                    source_node_id=query.source_id,
                    target_node_id=query.target_id,
                    max_paths=query.max_paths,
                    attack_stages=attack_stages
                )
                
                results.append({
                    "query_index": i,
                    "source_id": query.source_id,
                    "target_id": query.target_id,
                    "paths": [path.to_dict() for path in paths],
                    "count": len(paths)
                })
                
            except Exception as e:
                errors.append({
                    "query_index": i,
                    "error": str(e)
                })
        
        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors,
            "processed_count": len(results),
            "error_count": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Error in batch attack path analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Advanced analysis endpoints

@analysis_router.get("/risk_assessment", response_model=Dict[str, Any])
async def get_risk_assessment(
    analyzer: AttackPathAnalyzer = Depends(get_attack_analyzer),
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Get comprehensive risk assessment"""
    try:
        # Get critical paths
        critical_paths = await analyzer.get_critical_attack_paths(min_risk_score=70.0)
        
        # Get critical nodes
        critical_nodes = await algorithms.find_critical_nodes(top_k=10)
        
        # Get graph metrics
        metrics = await algorithms.calculate_graph_metrics()
        
        # Calculate overall risk score
        overall_risk = 0.0
        if critical_paths:
            overall_risk = sum(path.risk_score for path in critical_paths) / len(critical_paths)
        
        return {
            "success": True,
            "risk_assessment": {
                "overall_risk_score": overall_risk,
                "critical_paths": [path.to_dict() for path in critical_paths],
                "critical_nodes": critical_nodes,
                "graph_metrics": metrics.to_dict(),
                "assessment_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting risk assessment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/threat_landscape", response_model=Dict[str, Any])
async def get_threat_landscape(
    mapper: VulnerabilityMapper = Depends(get_vulnerability_mapper),
    algorithms: GraphAlgorithms = Depends(get_graph_algorithms)
):
    """Get threat landscape overview"""
    try:
        # Get vulnerability-threat correlations
        correlations = await mapper.correlate_vulnerabilities_with_threats(limit=50)
        
        # Get communities (threat groups)
        communities = await algorithms.detect_communities(algorithm="louvain")
        
        # Get anomalies
        anomalies = await algorithms.detect_anomalies(method="isolation_forest")
        
        return {
            "success": True,
            "threat_landscape": {
                "vulnerability_threat_correlations": correlations,
                "threat_communities": [community.to_dict() for community in communities],
                "anomalous_entities": [anomaly.to_dict() for anomaly in anomalies],
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting threat landscape: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Advanced analytics endpoints

@analysis_router.post("/patterns", response_model=Dict[str, Any])
async def analyze_security_patterns(
    pattern_types: Optional[List[str]] = Query(None, description="Pattern types to analyze"),
    analytics: GraphAnalytics = Depends(get_graph_analytics)
):
    """Analyze security patterns in the graph"""
    try:
        patterns = await analytics.analyze_patterns(pattern_types=pattern_types)
        
        return {
            "success": True,
            "patterns": [pattern.to_dict() for pattern in patterns],
            "count": len(patterns),
            "pattern_types": pattern_types or "all"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/intelligence_correlation", response_model=Dict[str, Any])
async def correlate_intelligence(
    time_window_hours: int = Query(24, description="Time window for correlation in hours"),
    analytics: GraphAnalytics = Depends(get_graph_analytics)
):
    """Correlate intelligence across multiple sources"""
    try:
        correlations = await analytics.correlate_intelligence(time_window_hours=time_window_hours)
        
        return {
            "success": True,
            "correlations": [correlation.to_dict() for correlation in correlations],
            "count": len(correlations),
            "time_window_hours": time_window_hours
        }
        
    except Exception as e:
        logger.error(f"Error correlating intelligence: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/trends", response_model=Dict[str, Any])
async def analyze_trends(
    entity_types: Optional[List[str]] = Query(None, description="Entity types to analyze"),
    time_period_days: int = Query(30, description="Time period for trend analysis"),
    analytics: GraphAnalytics = Depends(get_graph_analytics)
):
    """Analyze trends in graph entities and activities"""
    try:
        trends = await analytics.analyze_trends(
            entity_types=entity_types,
            time_period_days=time_period_days
        )
        
        return {
            "success": True,
            "trends": [trend.to_dict() for trend in trends],
            "count": len(trends),
            "time_period_days": time_period_days
        }
        
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/comprehensive_risk", response_model=Dict[str, Any])
async def calculate_comprehensive_risk(
    entity_ids: Optional[List[str]] = Query(None, description="Specific entity IDs to analyze"),
    analytics: GraphAnalytics = Depends(get_graph_analytics)
):
    """Calculate comprehensive risk scores for entities"""
    try:
        risk_scores = await analytics.calculate_comprehensive_risk_scores(entity_ids=entity_ids)
        
        return {
            "success": True,
            "risk_scores": [risk.to_dict() for risk in risk_scores],
            "count": len(risk_scores)
        }
        
    except Exception as e:
        logger.error(f"Error calculating comprehensive risk: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/analytics_report", response_model=Dict[str, Any])
async def export_analytics_report(
    format: str = Query("json", description="Report format"),
    analytics: GraphAnalytics = Depends(get_graph_analytics)
):
    """Export comprehensive analytics report"""
    try:
        report = await analytics.export_analytics_report(format=format)
        
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error exporting analytics report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Graph maintenance endpoints

@analysis_router.post("/backup", response_model=Dict[str, Any])
async def create_graph_backup(
    backup_type: str = Query("full", description="Backup type: full, incremental, schema_only"),
    compress: bool = Query(True, description="Compress backup file"),
    maintenance: GraphMaintenance = Depends(get_graph_maintenance)
):
    """Create a backup of the knowledge graph"""
    try:
        backup_info = await maintenance.create_backup(backup_type=backup_type, compress=compress)
        
        return {
            "success": True,
            "backup_info": backup_info.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/restore", response_model=Dict[str, Any])
async def restore_graph_backup(
    backup_id: str = Query(..., description="Backup ID to restore"),
    restore_mode: str = Query("replace", description="Restore mode: replace or merge"),
    maintenance: GraphMaintenance = Depends(get_graph_maintenance)
):
    """Restore graph from backup"""
    try:
        success = await maintenance.restore_backup(backup_id=backup_id, restore_mode=restore_mode)
        
        return {
            "success": success,
            "backup_id": backup_id,
            "restore_mode": restore_mode,
            "message": "Restore completed successfully" if success else "Restore failed"
        }
        
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/health", response_model=Dict[str, Any])
async def perform_graph_health_check(
    maintenance: GraphMaintenance = Depends(get_graph_maintenance)
):
    """Perform comprehensive health check"""
    try:
        health_checks = await maintenance.perform_health_check()
        
        return {
            "success": True,
            "health_checks": [check.to_dict() for check in health_checks],
            "overall_status": "healthy" if all(check.status != "critical" for check in health_checks) else "critical"
        }
        
    except Exception as e:
        logger.error(f"Error performing health check: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.post("/optimize", response_model=Dict[str, Any])
async def optimize_graph(
    optimization_types: Optional[List[str]] = Query(None, description="Optimization types"),
    maintenance: GraphMaintenance = Depends(get_graph_maintenance)
):
    """Perform graph optimization"""
    try:
        optimization_results = await maintenance.optimize_graph(optimization_types=optimization_types)
        
        return {
            "success": True,
            "optimization_results": [result.to_dict() for result in optimization_results],
            "count": len(optimization_results)
        }
        
    except Exception as e:
        logger.error(f"Error optimizing graph: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@analysis_router.get("/maintenance_status", response_model=Dict[str, Any])
async def get_maintenance_status(
    maintenance: GraphMaintenance = Depends(get_graph_maintenance)
):
    """Get overall maintenance status"""
    try:
        status = await maintenance.get_maintenance_status()
        
        return {
            "success": True,
            "maintenance_status": status
        }
        
    except Exception as e:
        logger.error(f"Error getting maintenance status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")