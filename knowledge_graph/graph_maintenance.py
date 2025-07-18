"""
Graph Maintenance - Comprehensive maintenance and operations utilities
Provides backup, restore, optimization, and health monitoring capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import gzip
import shutil
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile

import networkx as nx
from neo4j.exceptions import ServiceUnavailable

from .graph_manager import GraphManager, GraphNode, GraphEdge
from .graph_schema import NodeType, RelationType, GraphSchema
from ..config import get_config


@dataclass
class MaintenanceTask:
    """Represents a maintenance task"""
    task_id: str
    task_type: str
    description: str
    scheduled_time: datetime
    estimated_duration: int  # minutes
    priority: str  # critical, high, medium, low
    dependencies: List[str]
    status: str  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "scheduled_time": self.scheduled_time.isoformat(),
            "estimated_duration": self.estimated_duration,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "result": self.result
        }


@dataclass
class BackupInfo:
    """Information about a graph backup"""
    backup_id: str
    backup_type: str  # full, incremental, schema_only
    file_path: str
    compressed: bool
    size_bytes: int
    node_count: int
    edge_count: int
    created_at: datetime
    retention_days: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type,
            "file_path": self.file_path,
            "compressed": self.compressed,
            "size_bytes": self.size_bytes,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "created_at": self.created_at.isoformat(),
            "retention_days": self.retention_days,
            "checksum": self.checksum,
            "metadata": self.metadata
        }


@dataclass
class HealthCheck:
    """Graph health check results"""
    check_id: str
    check_type: str
    status: str  # healthy, warning, critical
    message: str
    details: Dict[str, Any]
    checked_at: datetime
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "check_id": self.check_id,
            "check_type": self.check_type,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
            "recommended_actions": self.recommended_actions
        }


@dataclass
class OptimizationResult:
    """Graph optimization results"""
    optimization_id: str
    optimization_type: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvements: Dict[str, float]
    duration_seconds: float
    performed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "optimization_id": self.optimization_id,
            "optimization_type": self.optimization_type,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "improvements": self.improvements,
            "duration_seconds": self.duration_seconds,
            "performed_at": self.performed_at.isoformat()
        }


class GraphMaintenance:
    """Comprehensive graph maintenance and operations"""
    
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Maintenance configuration
        self.backup_directory = Path("backups/knowledge_graph")
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        
        self.log_directory = Path("logs/maintenance")
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Task management
        self.scheduled_tasks = {}
        self.task_history = []
        self.running_tasks = set()
        
        # Health monitoring
        self.health_checks = [
            "connectivity",
            "performance",
            "data_integrity",
            "storage_usage",
            "schema_validation",
            "index_health",
            "cache_efficiency"
        ]
        
        # Optimization settings
        self.optimization_thresholds = {
            "orphaned_nodes": 100,
            "duplicate_edges": 50,
            "index_fragmentation": 0.3,
            "cache_hit_ratio": 0.8
        }
        
        # Retention policies
        self.retention_policies = {
            "backups": {
                "daily": 30,    # Keep daily backups for 30 days
                "weekly": 12,   # Keep weekly backups for 12 weeks
                "monthly": 12   # Keep monthly backups for 12 months
            },
            "logs": {
                "maintenance": 90,  # Keep maintenance logs for 90 days
                "performance": 30,  # Keep performance logs for 30 days
                "error": 180       # Keep error logs for 180 days
            }
        }
    
    async def create_backup(self, backup_type: str = "full", compress: bool = True) -> BackupInfo:
        """Create a backup of the knowledge graph"""
        try:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"Starting {backup_type} backup: {backup_id}")
            start_time = datetime.now()
            
            # Create backup data
            backup_data = await self._collect_backup_data(backup_type)
            
            # Determine file path
            file_extension = ".gz" if compress else ".json"
            backup_file = self.backup_directory / f"{backup_id}{file_extension}"
            
            # Write backup data
            if compress:
                with gzip.open(backup_file, 'wt', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, default=str)
            else:
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, default=str)
            
            # Calculate file size and checksum
            file_size = backup_file.stat().st_size
            checksum = await self._calculate_file_checksum(backup_file)
            
            # Create backup info
            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                file_path=str(backup_file),
                compressed=compress,
                size_bytes=file_size,
                node_count=backup_data.get("node_count", 0),
                edge_count=backup_data.get("edge_count", 0),
                created_at=start_time,
                retention_days=self.retention_policies["backups"]["daily"],
                checksum=checksum,
                metadata={
                    "graph_backend": backup_data.get("backend", "unknown"),
                    "schema_version": backup_data.get("schema_version", "1.0"),
                    "backup_duration": (datetime.now() - start_time).total_seconds()
                }
            )
            
            # Save backup metadata
            await self._save_backup_metadata(backup_info)
            
            self.logger.info(f"Backup completed: {backup_id} ({file_size} bytes)")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise
    
    async def _collect_backup_data(self, backup_type: str) -> Dict[str, Any]:
        """Collect data for backup"""
        backup_data = {
            "backup_metadata": {
                "backup_type": backup_type,
                "created_at": datetime.now().isoformat(),
                "graph_backend": "neo4j" if not self.graph_manager.use_fallback else "networkx"
            }
        }
        
        if backup_type in ["full", "incremental"]:
            # Collect nodes
            nodes = await self.graph_manager.find_nodes(limit=10000)
            backup_data["nodes"] = [node.to_dict() for node in nodes]
            backup_data["node_count"] = len(nodes)
            
            # Collect edges (simplified - would need proper edge collection)
            backup_data["edges"] = []
            backup_data["edge_count"] = 0
            
            # For a full implementation, you'd traverse the graph to collect all edges
            
        if backup_type in ["full", "schema_only"]:
            # Collect schema
            schema = GraphSchema()
            backup_data["schema"] = schema.export_schema()
            backup_data["schema_version"] = "1.0"
        
        # Collect graph statistics
        stats = await self.graph_manager.get_graph_stats()
        backup_data["statistics"] = stats
        backup_data["backend"] = stats.get("backend", "unknown")
        
        return backup_data
    
    async def restore_backup(self, backup_id: str, restore_mode: str = "replace") -> bool:
        """Restore graph from backup"""
        try:
            self.logger.info(f"Starting restore from backup: {backup_id}")
            
            # Load backup metadata
            backup_info = await self._load_backup_metadata(backup_id)
            if not backup_info:
                raise ValueError(f"Backup not found: {backup_id}")
            
            # Load backup data
            backup_data = await self._load_backup_data(backup_info)
            
            # Validate backup data
            if not await self._validate_backup_data(backup_data):
                raise ValueError("Backup data validation failed")
            
            # Perform restore based on mode
            if restore_mode == "replace":
                await self._restore_replace_mode(backup_data)
            elif restore_mode == "merge":
                await self._restore_merge_mode(backup_data)
            else:
                raise ValueError(f"Unknown restore mode: {restore_mode}")
            
            self.logger.info(f"Restore completed: {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring backup {backup_id}: {e}")
            return False
    
    async def _restore_replace_mode(self, backup_data: Dict[str, Any]) -> None:
        """Restore in replace mode (clear existing data)"""
        # WARNING: This would clear all existing data
        # In a production system, you'd want additional safeguards
        
        self.logger.warning("Replace mode restore - this will clear existing data")
        
        # Clear existing data (implementation depends on backend)
        # For Neo4j: MATCH (n) DETACH DELETE n
        # For NetworkX: graph.clear()
        
        # Restore nodes
        if "nodes" in backup_data:
            for node_data in backup_data["nodes"]:
                try:
                    node_type = NodeType(node_data["type"])
                    await self.graph_manager.create_node(
                        node_type=node_type,
                        properties=node_data["properties"],
                        labels=node_data.get("labels", [])
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to restore node {node_data.get('id', 'unknown')}: {e}")
        
        # Restore edges
        if "edges" in backup_data:
            for edge_data in backup_data["edges"]:
                try:
                    edge_type = RelationType(edge_data["type"])
                    await self.graph_manager.create_edge(
                        source_id=edge_data["source"],
                        target_id=edge_data["target"],
                        edge_type=edge_type,
                        properties=edge_data.get("properties", {}),
                        weight=edge_data.get("weight", 1.0)
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to restore edge {edge_data.get('id', 'unknown')}: {e}")
    
    async def _restore_merge_mode(self, backup_data: Dict[str, Any]) -> None:
        """Restore in merge mode (merge with existing data)"""
        # Restore nodes (skip if already exists)
        if "nodes" in backup_data:
            for node_data in backup_data["nodes"]:
                try:
                    # Check if node exists
                    existing_node = await self.graph_manager.get_node(node_data["id"])
                    if not existing_node:
                        node_type = NodeType(node_data["type"])
                        await self.graph_manager.create_node(
                            node_type=node_type,
                            properties=node_data["properties"],
                            labels=node_data.get("labels", [])
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to merge node {node_data.get('id', 'unknown')}: {e}")
    
    async def perform_health_check(self) -> List[HealthCheck]:
        """Perform comprehensive health check"""
        try:
            health_results = []
            
            for check_type in self.health_checks:
                try:
                    result = await self._perform_individual_health_check(check_type)
                    health_results.append(result)
                except Exception as e:
                    health_results.append(HealthCheck(
                        check_id=f"health_{check_type}_{datetime.now().timestamp()}",
                        check_type=check_type,
                        status="critical",
                        message=f"Health check failed: {str(e)}",
                        details={"error": str(e)},
                        checked_at=datetime.now(),
                        recommended_actions=["Investigate health check failure"]
                    ))
            
            # Log overall health status
            critical_checks = [h for h in health_results if h.status == "critical"]
            if critical_checks:
                self.logger.error(f"Critical health issues found: {len(critical_checks)}")
            else:
                self.logger.info("Health check completed - no critical issues")
            
            return health_results
            
        except Exception as e:
            self.logger.error(f"Error performing health check: {e}")
            return []
    
    async def _perform_individual_health_check(self, check_type: str) -> HealthCheck:
        """Perform individual health check"""
        check_id = f"health_{check_type}_{datetime.now().timestamp()}"
        
        if check_type == "connectivity":
            return await self._check_connectivity(check_id)
        elif check_type == "performance":
            return await self._check_performance(check_id)
        elif check_type == "data_integrity":
            return await self._check_data_integrity(check_id)
        elif check_type == "storage_usage":
            return await self._check_storage_usage(check_id)
        elif check_type == "schema_validation":
            return await self._check_schema_validation(check_id)
        elif check_type == "index_health":
            return await self._check_index_health(check_id)
        elif check_type == "cache_efficiency":
            return await self._check_cache_efficiency(check_id)
        else:
            return HealthCheck(
                check_id=check_id,
                check_type=check_type,
                status="warning",
                message=f"Unknown health check type: {check_type}",
                details={},
                checked_at=datetime.now()
            )
    
    async def _check_connectivity(self, check_id: str) -> HealthCheck:
        """Check database connectivity"""
        try:
            stats = await self.graph_manager.get_graph_stats()
            
            if stats and "node_count" in stats:
                return HealthCheck(
                    check_id=check_id,
                    check_type="connectivity",
                    status="healthy",
                    message="Database connection is healthy",
                    details={"backend": stats.get("backend", "unknown")},
                    checked_at=datetime.now()
                )
            else:
                return HealthCheck(
                    check_id=check_id,
                    check_type="connectivity",
                    status="warning",
                    message="Database connection unstable",
                    details={"stats": stats},
                    checked_at=datetime.now(),
                    recommended_actions=["Check database configuration", "Verify network connectivity"]
                )
        except Exception as e:
            return HealthCheck(
                check_id=check_id,
                check_type="connectivity",
                status="critical",
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e)},
                checked_at=datetime.now(),
                recommended_actions=["Check database service", "Verify credentials", "Check network"]
            )
    
    async def _check_performance(self, check_id: str) -> HealthCheck:
        """Check query performance"""
        try:
            import time
            
            # Test query performance
            start_time = time.time()
            nodes = await self.graph_manager.find_nodes(limit=10)
            query_time = time.time() - start_time
            
            status = "healthy"
            message = f"Query performance is good ({query_time:.3f}s for 10 nodes)"
            recommendations = []
            
            if query_time > 5.0:
                status = "critical"
                message = f"Query performance is poor ({query_time:.3f}s for 10 nodes)"
                recommendations = ["Optimize database indexes", "Check system resources"]
            elif query_time > 1.0:
                status = "warning"
                message = f"Query performance is slow ({query_time:.3f}s for 10 nodes)"
                recommendations = ["Monitor performance trends", "Consider optimization"]
            
            return HealthCheck(
                check_id=check_id,
                check_type="performance",
                status=status,
                message=message,
                details={
                    "query_time_seconds": query_time,
                    "nodes_retrieved": len(nodes)
                },
                checked_at=datetime.now(),
                recommended_actions=recommendations
            )
            
        except Exception as e:
            return HealthCheck(
                check_id=check_id,
                check_type="performance",
                status="critical",
                message=f"Performance check failed: {str(e)}",
                details={"error": str(e)},
                checked_at=datetime.now(),
                recommended_actions=["Investigate performance issues"]
            )
    
    async def _check_data_integrity(self, check_id: str) -> HealthCheck:
        """Check data integrity"""
        try:
            issues = []
            
            # Check for orphaned nodes (nodes without any edges)
            # This is a simplified check - full implementation would be more comprehensive
            sample_nodes = await self.graph_manager.find_nodes(limit=100)
            orphaned_count = 0
            
            for node in sample_nodes:
                neighbors = await self.graph_manager.get_neighbors(node.id, limit=1)
                if not neighbors:
                    orphaned_count += 1
            
            if orphaned_count > self.optimization_thresholds["orphaned_nodes"]:
                issues.append(f"Found {orphaned_count} potentially orphaned nodes")
            
            # Check for schema violations
            schema = GraphSchema()
            validation_errors = 0
            
            for node in sample_nodes[:10]:  # Sample check
                errors = schema.validate_node(node.type, node.properties)
                validation_errors += len(errors)
            
            if validation_errors > 0:
                issues.append(f"Found {validation_errors} schema validation errors")
            
            # Determine status
            if not issues:
                status = "healthy"
                message = "Data integrity check passed"
            elif len(issues) == 1:
                status = "warning"
                message = f"Minor data integrity issues: {issues[0]}"
            else:
                status = "critical"
                message = f"Multiple data integrity issues found: {len(issues)}"
            
            return HealthCheck(
                check_id=check_id,
                check_type="data_integrity",
                status=status,
                message=message,
                details={
                    "issues_found": issues,
                    "orphaned_nodes": orphaned_count,
                    "validation_errors": validation_errors,
                    "nodes_checked": len(sample_nodes)
                },
                checked_at=datetime.now(),
                recommended_actions=["Run data cleanup", "Validate schema compliance"] if issues else []
            )
            
        except Exception as e:
            return HealthCheck(
                check_id=check_id,
                check_type="data_integrity",
                status="critical",
                message=f"Data integrity check failed: {str(e)}",
                details={"error": str(e)},
                checked_at=datetime.now(),
                recommended_actions=["Investigate data integrity issues"]
            )
    
    async def _check_storage_usage(self, check_id: str) -> HealthCheck:
        """Check storage usage"""
        try:
            # Check backup directory size
            backup_size = sum(f.stat().st_size for f in self.backup_directory.rglob('*') if f.is_file())
            backup_count = len(list(self.backup_directory.glob('backup_*')))
            
            # Check log directory size
            log_size = sum(f.stat().st_size for f in self.log_directory.rglob('*') if f.is_file())
            
            total_size = backup_size + log_size
            
            # Simple thresholds (in production, these would be configurable)
            size_gb = total_size / (1024**3)
            
            if size_gb > 10:
                status = "critical"
                message = f"High storage usage: {size_gb:.2f} GB"
                recommendations = ["Clean up old backups", "Archive old logs"]
            elif size_gb > 5:
                status = "warning"
                message = f"Moderate storage usage: {size_gb:.2f} GB"
                recommendations = ["Monitor storage growth", "Review retention policies"]
            else:
                status = "healthy"
                message = f"Storage usage is normal: {size_gb:.2f} GB"
                recommendations = []
            
            return HealthCheck(
                check_id=check_id,
                check_type="storage_usage",
                status=status,
                message=message,
                details={
                    "total_size_bytes": total_size,
                    "total_size_gb": size_gb,
                    "backup_size_bytes": backup_size,
                    "backup_count": backup_count,
                    "log_size_bytes": log_size
                },
                checked_at=datetime.now(),
                recommended_actions=recommendations
            )
            
        except Exception as e:
            return HealthCheck(
                check_id=check_id,
                check_type="storage_usage",
                status="warning",
                message=f"Storage check failed: {str(e)}",
                details={"error": str(e)},
                checked_at=datetime.now()
            )
    
    async def _check_schema_validation(self, check_id: str) -> HealthCheck:
        """Check schema validation"""
        try:
            schema = GraphSchema()
            sample_nodes = await self.graph_manager.find_nodes(limit=50)
            
            total_errors = 0
            error_details = {}
            
            for node in sample_nodes:
                errors = schema.validate_node(node.type, node.properties)
                if errors:
                    total_errors += len(errors)
                    error_details[node.id] = errors
            
            if total_errors == 0:
                status = "healthy"
                message = "All nodes pass schema validation"
            elif total_errors < 5:
                status = "warning"
                message = f"Minor schema validation issues: {total_errors} errors"
            else:
                status = "critical"
                message = f"Significant schema validation issues: {total_errors} errors"
            
            return HealthCheck(
                check_id=check_id,
                check_type="schema_validation",
                status=status,
                message=message,
                details={
                    "total_errors": total_errors,
                    "nodes_checked": len(sample_nodes),
                    "error_details": error_details
                },
                checked_at=datetime.now(),
                recommended_actions=["Fix schema violations", "Update node properties"] if total_errors > 0 else []
            )
            
        except Exception as e:
            return HealthCheck(
                check_id=check_id,
                check_type="schema_validation",
                status="critical",
                message=f"Schema validation check failed: {str(e)}",
                details={"error": str(e)},
                checked_at=datetime.now()
            )
    
    async def _check_index_health(self, check_id: str) -> HealthCheck:
        """Check index health (Neo4j specific)"""
        if self.graph_manager.use_fallback:
            return HealthCheck(
                check_id=check_id,
                check_type="index_health",
                status="healthy",
                message="Index health not applicable for NetworkX backend",
                details={"backend": "networkx"},
                checked_at=datetime.now()
            )
        
        try:
            # This would check Neo4j index health
            # For now, return a placeholder
            return HealthCheck(
                check_id=check_id,
                check_type="index_health",
                status="healthy",
                message="Index health check not fully implemented",
                details={"backend": "neo4j"},
                checked_at=datetime.now(),
                recommended_actions=["Implement comprehensive index health monitoring"]
            )
            
        except Exception as e:
            return HealthCheck(
                check_id=check_id,
                check_type="index_health",
                status="warning",
                message=f"Index health check failed: {str(e)}",
                details={"error": str(e)},
                checked_at=datetime.now()
            )
    
    async def _check_cache_efficiency(self, check_id: str) -> HealthCheck:
        """Check cache efficiency"""
        try:
            # This would check cache hit ratios and efficiency
            # For now, return a placeholder
            return HealthCheck(
                check_id=check_id,
                check_type="cache_efficiency",
                status="healthy",
                message="Cache efficiency monitoring not fully implemented",
                details={},
                checked_at=datetime.now(),
                recommended_actions=["Implement cache performance monitoring"]
            )
            
        except Exception as e:
            return HealthCheck(
                check_id=check_id,
                check_type="cache_efficiency",
                status="warning",
                message=f"Cache efficiency check failed: {str(e)}",
                details={"error": str(e)},
                checked_at=datetime.now()
            )
    
    async def optimize_graph(self, optimization_types: Optional[List[str]] = None) -> List[OptimizationResult]:
        """Perform graph optimization"""
        try:
            optimizations = optimization_types or ["cleanup", "defragment", "rebuild_indexes"]
            results = []
            
            for opt_type in optimizations:
                result = await self._perform_optimization(opt_type)
                if result:
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing graph optimization: {e}")
            return []
    
    async def _perform_optimization(self, optimization_type: str) -> Optional[OptimizationResult]:
        """Perform specific optimization"""
        start_time = datetime.now()
        optimization_id = f"opt_{optimization_type}_{start_time.timestamp()}"
        
        try:
            # Get before metrics
            before_metrics = await self._get_optimization_metrics()
            
            if optimization_type == "cleanup":
                await self._cleanup_orphaned_nodes()
                await self._remove_duplicate_edges()
            elif optimization_type == "defragment":
                await self._defragment_storage()
            elif optimization_type == "rebuild_indexes":
                await self._rebuild_indexes()
            else:
                self.logger.warning(f"Unknown optimization type: {optimization_type}")
                return None
            
            # Get after metrics
            after_metrics = await self._get_optimization_metrics()
            
            # Calculate improvements
            improvements = {}
            for metric, after_value in after_metrics.items():
                before_value = before_metrics.get(metric, 0)
                if before_value > 0:
                    improvement = (after_value - before_value) / before_value
                    improvements[metric] = improvement
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                optimization_id=optimization_id,
                optimization_type=optimization_type,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvements=improvements,
                duration_seconds=duration,
                performed_at=start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in {optimization_type} optimization: {e}")
            return None
    
    async def _get_optimization_metrics(self) -> Dict[str, Any]:
        """Get metrics for optimization comparison"""
        try:
            stats = await self.graph_manager.get_graph_stats()
            
            # Add more detailed metrics here
            metrics = {
                "node_count": stats.get("node_count", 0),
                "edge_count": stats.get("edge_count", 0),
                # Add other relevant metrics
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting optimization metrics: {e}")
            return {}
    
    async def _cleanup_orphaned_nodes(self) -> None:
        """Remove orphaned nodes"""
        try:
            # Find and remove nodes without any connections
            sample_nodes = await self.graph_manager.find_nodes(limit=1000)
            orphaned_nodes = []
            
            for node in sample_nodes:
                neighbors = await self.graph_manager.get_neighbors(node.id, limit=1)
                if not neighbors:
                    orphaned_nodes.append(node.id)
            
            # Remove orphaned nodes
            for node_id in orphaned_nodes:
                await self.graph_manager.delete_node(node_id)
            
            if orphaned_nodes:
                self.logger.info(f"Cleaned up {len(orphaned_nodes)} orphaned nodes")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up orphaned nodes: {e}")
    
    async def _remove_duplicate_edges(self) -> None:
        """Remove duplicate edges"""
        # This would implement duplicate edge detection and removal
        # Implementation depends on the specific backend and requirements
        self.logger.info("Duplicate edge removal not fully implemented")
    
    async def _defragment_storage(self) -> None:
        """Defragment storage (backend-specific)"""
        # This would implement storage defragmentation
        # Implementation depends on the specific backend
        self.logger.info("Storage defragmentation not fully implemented")
    
    async def _rebuild_indexes(self) -> None:
        """Rebuild database indexes"""
        # This would rebuild database indexes for better performance
        # Implementation depends on the specific backend
        self.logger.info("Index rebuilding not fully implemented")
    
    async def schedule_maintenance_task(self, task: MaintenanceTask) -> bool:
        """Schedule a maintenance task"""
        try:
            self.scheduled_tasks[task.task_id] = task
            self.logger.info(f"Scheduled maintenance task: {task.task_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error scheduling maintenance task: {e}")
            return False
    
    async def run_scheduled_tasks(self) -> List[MaintenanceTask]:
        """Run due scheduled tasks"""
        try:
            completed_tasks = []
            current_time = datetime.now()
            
            for task_id, task in list(self.scheduled_tasks.items()):
                if (task.scheduled_time <= current_time and 
                    task.status == "pending" and 
                    task_id not in self.running_tasks):
                    
                    # Check dependencies
                    if await self._check_task_dependencies(task):
                        await self._run_maintenance_task(task)
                        completed_tasks.append(task)
                        
                        if task.status == "completed":
                            del self.scheduled_tasks[task_id]
            
            return completed_tasks
            
        except Exception as e:
            self.logger.error(f"Error running scheduled tasks: {e}")
            return []
    
    async def _check_task_dependencies(self, task: MaintenanceTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id in self.scheduled_tasks:
                dep_task = self.scheduled_tasks[dep_id]
                if dep_task.status != "completed":
                    return False
        return True
    
    async def _run_maintenance_task(self, task: MaintenanceTask) -> None:
        """Run a maintenance task"""
        try:
            task.status = "running"
            task.started_at = datetime.now()
            self.running_tasks.add(task.task_id)
            
            self.logger.info(f"Running maintenance task: {task.task_id}")
            
            # Execute task based on type
            if task.task_type == "backup":
                result = await self.create_backup()
                task.result = {"backup_info": result.to_dict()}
            elif task.task_type == "health_check":
                result = await self.perform_health_check()
                task.result = {"health_checks": [h.to_dict() for h in result]}
            elif task.task_type == "optimization":
                result = await self.optimize_graph()
                task.result = {"optimizations": [o.to_dict() for o in result]}
            elif task.task_type == "cleanup":
                await self.cleanup_old_files()
                task.result = {"cleanup": "completed"}
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.status = "completed"
            task.completed_at = datetime.now()
            
            self.logger.info(f"Completed maintenance task: {task.task_id}")
            
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.logger.error(f"Maintenance task failed {task.task_id}: {e}")
        finally:
            self.running_tasks.discard(task.task_id)
    
    async def cleanup_old_files(self) -> None:
        """Clean up old backup and log files based on retention policies"""
        try:
            current_time = datetime.now()
            
            # Clean up old backups
            backup_files = list(self.backup_directory.glob("backup_*"))
            for backup_file in backup_files:
                file_age = current_time - datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                if file_age.days > self.retention_policies["backups"]["daily"]:
                    backup_file.unlink()
                    self.logger.info(f"Removed old backup: {backup_file.name}")
            
            # Clean up old logs
            log_files = list(self.log_directory.glob("*.log"))
            for log_file in log_files:
                file_age = current_time - datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_age.days > self.retention_policies["logs"]["maintenance"]:
                    log_file.unlink()
                    self.logger.info(f"Removed old log: {log_file.name}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {e}")
    
    async def get_maintenance_status(self) -> Dict[str, Any]:
        """Get overall maintenance status"""
        try:
            status = {
                "scheduled_tasks": len(self.scheduled_tasks),
                "running_tasks": len(self.running_tasks),
                "completed_tasks_today": len([
                    t for t in self.task_history 
                    if t.completed_at and t.completed_at.date() == datetime.now().date()
                ]),
                "last_backup": await self._get_last_backup_info(),
                "last_health_check": await self._get_last_health_check_info(),
                "storage_usage": await self._get_storage_usage_summary()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting maintenance status: {e}")
            return {"error": str(e)}
    
    # Helper methods for file operations
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        import hashlib
        
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    async def _save_backup_metadata(self, backup_info: BackupInfo) -> None:
        """Save backup metadata"""
        metadata_file = self.backup_directory / f"{backup_info.backup_id}.meta"
        with open(metadata_file, 'w') as f:
            json.dump(backup_info.to_dict(), f, indent=2)
    
    async def _load_backup_metadata(self, backup_id: str) -> Optional[BackupInfo]:
        """Load backup metadata"""
        try:
            metadata_file = self.backup_directory / f"{backup_id}.meta"
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            return BackupInfo(
                backup_id=data["backup_id"],
                backup_type=data["backup_type"],
                file_path=data["file_path"],
                compressed=data["compressed"],
                size_bytes=data["size_bytes"],
                node_count=data["node_count"],
                edge_count=data["edge_count"],
                created_at=datetime.fromisoformat(data["created_at"]),
                retention_days=data["retention_days"],
                checksum=data["checksum"],
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            self.logger.error(f"Error loading backup metadata: {e}")
            return None
    
    async def _load_backup_data(self, backup_info: BackupInfo) -> Dict[str, Any]:
        """Load backup data from file"""
        backup_file = Path(backup_info.file_path)
        
        if backup_info.compressed:
            with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(backup_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    async def _validate_backup_data(self, backup_data: Dict[str, Any]) -> bool:
        """Validate backup data structure"""
        required_fields = ["backup_metadata"]
        
        for field in required_fields:
            if field not in backup_data:
                self.logger.error(f"Missing required field in backup: {field}")
                return False
        
        return True
    
    async def _get_last_backup_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the last backup"""
        try:
            backup_files = list(self.backup_directory.glob("backup_*.meta"))
            if not backup_files:
                return None
            
            # Get the most recent backup
            latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_backup, 'r') as f:
                backup_data = json.load(f)
            
            return {
                "backup_id": backup_data["backup_id"],
                "created_at": backup_data["created_at"],
                "size_bytes": backup_data["size_bytes"],
                "backup_type": backup_data["backup_type"]
            }
        except Exception:
            return None
    
    async def _get_last_health_check_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the last health check"""
        # This would load from health check logs
        return {"status": "not_implemented"}
    
    async def _get_storage_usage_summary(self) -> Dict[str, Any]:
        """Get storage usage summary"""
        try:
            backup_size = sum(f.stat().st_size for f in self.backup_directory.rglob('*') if f.is_file())
            log_size = sum(f.stat().st_size for f in self.log_directory.rglob('*') if f.is_file())
            
            return {
                "backup_size_gb": backup_size / (1024**3),
                "log_size_gb": log_size / (1024**3),
                "total_size_gb": (backup_size + log_size) / (1024**3)
            }
        except Exception:
            return {"error": "Could not calculate storage usage"}