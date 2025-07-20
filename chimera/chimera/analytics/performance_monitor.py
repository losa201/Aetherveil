"""
Real-Time Performance Metrics and Optimization System
Continuously monitors Chimera's performance and automatically optimizes operations
"""

import asyncio
import logging
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil

from ..core.events import EventSystem, EventType, EventEmitter

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """A single performance metric measurement"""
    
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any]
    category: str  # system, network, reasoning, execution, etc.
    source_component: str

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    
    metric_name: str
    baseline_value: float
    acceptable_range: Tuple[float, float]  # (min, max) acceptable values
    critical_threshold: float  # Value that triggers immediate action
    measurement_window: int  # Seconds to average over
    last_updated: datetime

@dataclass
class OptimizationAction:
    """An optimization action to improve performance"""
    
    action_id: str
    action_type: str  # parameter_adjustment, resource_reallocation, algorithm_switch
    target_component: str
    description: str
    parameters: Dict[str, Any]
    expected_improvement: float
    risk_level: float  # 0.0 to 1.0
    execution_time: datetime
    result: Optional[Dict[str, Any]] = None

class PerformanceMonitor(EventEmitter):
    """
    Real-Time Performance Monitoring and Optimization System
    
    Features:
    - Real-time metric collection across all components
    - Intelligent baseline establishment and drift detection
    - Automated optimization recommendations and execution
    - Performance trend analysis and prediction
    - Resource utilization monitoring and optimization
    - Bottleneck detection and resolution
    """
    
    def __init__(self, config, event_system: EventSystem):
        super().__init__(event_system, "PerformanceMonitor")
        
        self.config = config
        
        # Metrics storage and tracking
        self.metrics: deque = deque(maxlen=10000)  # Recent metrics
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.optimization_history: List[OptimizationAction] = []
        
        # Performance tracking by component
        self.component_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.component_health: Dict[str, float] = {}
        
        # Optimization parameters
        self.optimization_enabled = config.get("performance.optimization_enabled", True)
        self.baseline_learning_period = config.get("performance.baseline_learning_days", 7)
        self.optimization_aggressiveness = config.get("performance.optimization_aggressiveness", 0.7)
        self.critical_threshold_multiplier = config.get("performance.critical_threshold_multiplier", 2.0)
        
        # System resource monitoring
        self.system_metrics = {
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "disk_io": deque(maxlen=100),
            "network_io": deque(maxlen=100)
        }
        
        # Performance trends and predictions
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        self.bottleneck_detection: Dict[str, List[datetime]] = defaultdict(list)
        
    async def initialize(self):
        """Initialize performance monitoring system"""
        
        await self._initialize_baselines()
        await self._start_system_monitoring()
        
        # Start background monitoring and optimization
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._optimization_loop())
        asyncio.create_task(self._trend_analysis_loop())
        asyncio.create_task(self._system_health_loop())
        
        await self.emit_event(
            EventType.REASONING_START,
            {"message": "Performance monitoring initialized", "optimization_enabled": self.optimization_enabled}
        )
        
        logger.info("Performance monitoring system initialized")
        
    async def record_metric(self, metric_name: str, value: float, unit: str = "",
                          context: Dict[str, Any] = None, category: str = "general",
                          source_component: str = "unknown"):
        """Record a performance metric"""
        
        context = context or {}
        
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            context=context,
            category=category,
            source_component=source_component
        )
        
        self.metrics.append(metric)
        self.component_metrics[source_component].append(metric)
        
        # Update performance trends
        self.performance_trends[metric_name].append(value)
        if len(self.performance_trends[metric_name]) > 100:
            self.performance_trends[metric_name] = self.performance_trends[metric_name][-50:]
            
        # Check against baselines
        await self._check_performance_baseline(metric)
        
        # Detect bottlenecks
        await self._detect_bottlenecks(metric)
        
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard"""
        
        recent_window = datetime.utcnow() - timedelta(minutes=30)
        recent_metrics = [m for m in self.metrics if m.timestamp > recent_window]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
            
        # Overall system health
        system_health = await self._calculate_system_health()
        
        # Component health breakdown
        component_health = {}
        for component, metrics in self.component_metrics.items():
            recent_component_metrics = [m for m in metrics if m.timestamp > recent_window]
            if recent_component_metrics:
                component_health[component] = await self._calculate_component_health(recent_component_metrics)
                
        # Performance trends
        trend_analysis = await self._analyze_performance_trends()
        
        # Resource utilization
        resource_usage = await self._get_resource_utilization()
        
        # Active bottlenecks
        bottlenecks = await self._identify_active_bottlenecks()
        
        # Recent optimizations
        recent_optimizations = [opt for opt in self.optimization_history 
                               if opt.execution_time > recent_window]
        
        dashboard = {
            "system_health": {
                "overall_score": system_health,
                "status": self._get_health_status(system_health),
                "component_breakdown": component_health
            },
            "performance_metrics": {
                "total_metrics_collected": len(self.metrics),
                "metrics_per_minute": len(recent_metrics) / 30,
                "categories": self._get_category_breakdown(recent_metrics),
                "trending_metrics": trend_analysis
            },
            "resource_utilization": resource_usage,
            "bottlenecks": {
                "active_bottlenecks": bottlenecks,
                "bottleneck_history": len([b for b_list in self.bottleneck_detection.values() 
                                         for b in b_list if b > recent_window])
            },
            "optimizations": {
                "recent_optimizations": len(recent_optimizations),
                "optimization_success_rate": self._calculate_optimization_success_rate(),
                "pending_recommendations": await self._get_pending_recommendations()
            },
            "baselines": {
                "established_baselines": len(self.baselines),
                "baseline_violations": await self._count_baseline_violations(recent_metrics)
            }
        }
        
        return dashboard
        
    async def optimize_performance(self, target_component: str = None, 
                                 aggressive: bool = False) -> List[OptimizationAction]:
        """Trigger performance optimization"""
        
        optimization_actions = []
        
        # Analyze current performance
        performance_issues = await self._identify_performance_issues(target_component)
        
        for issue in performance_issues:
            # Generate optimization strategies
            strategies = await self._generate_optimization_strategies(issue)
            
            for strategy in strategies:
                if aggressive or strategy.risk_level <= self.optimization_aggressiveness:
                    # Execute optimization
                    action = await self._execute_optimization(strategy)
                    optimization_actions.append(action)
                    
        await self.emit_event(
            EventType.REASONING_UPDATED,
            {"action": "performance_optimization", "optimizations": len(optimization_actions)}
        )
        
        return optimization_actions
        
    async def predict_performance_issues(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Predict potential performance issues"""
        
        predictions = []
        
        # Analyze trends for each metric
        for metric_name, trend_data in self.performance_trends.items():
            if len(trend_data) < 10:
                continue
                
            # Simple linear regression for trend prediction
            prediction = await self._predict_metric_trend(metric_name, trend_data, hours_ahead)
            
            if prediction["confidence"] > 0.7:
                # Check if predicted value exceeds acceptable thresholds
                baseline = self.baselines.get(metric_name)
                if baseline and (prediction["predicted_value"] > baseline.critical_threshold):
                    predictions.append({
                        "metric": metric_name,
                        "predicted_value": prediction["predicted_value"],
                        "time_to_threshold": prediction["time_to_threshold"],
                        "confidence": prediction["confidence"],
                        "severity": "critical" if prediction["predicted_value"] > baseline.critical_threshold * 1.5 else "warning"
                    })
                    
        return predictions
        
    async def set_performance_baseline(self, metric_name: str, target_value: float,
                                     acceptable_range: Tuple[float, float],
                                     critical_threshold: float = None):
        """Set or update performance baseline"""
        
        critical_threshold = critical_threshold or target_value * self.critical_threshold_multiplier
        
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=target_value,
            acceptable_range=acceptable_range,
            critical_threshold=critical_threshold,
            measurement_window=300,  # 5 minutes
            last_updated=datetime.utcnow()
        )
        
        self.baselines[metric_name] = baseline
        
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {"action": "baseline_updated", "metric": metric_name, "target": target_value}
        )
        
    # Private methods
    
    async def _initialize_baselines(self):
        """Initialize default performance baselines"""
        
        default_baselines = {
            "response_time": {
                "target": 2.0,
                "range": (0.5, 5.0),
                "critical": 10.0,
                "unit": "seconds"
            },
            "success_rate": {
                "target": 0.9,
                "range": (0.8, 1.0),
                "critical": 0.7,
                "unit": "ratio"
            },
            "memory_usage": {
                "target": 0.7,
                "range": (0.3, 0.8),
                "critical": 0.9,
                "unit": "ratio"
            },
            "cpu_usage": {
                "target": 0.6,
                "range": (0.2, 0.8),
                "critical": 0.95,
                "unit": "ratio"
            },
            "error_rate": {
                "target": 0.05,
                "range": (0.0, 0.1),
                "critical": 0.2,
                "unit": "ratio"
            }
        }
        
        for metric_name, config in default_baselines.items():
            await self.set_performance_baseline(
                metric_name=metric_name,
                target_value=config["target"],
                acceptable_range=config["range"],
                critical_threshold=config["critical"]
            )
            
    async def _start_system_monitoring(self):
        """Start system resource monitoring"""
        
        # Initial system metrics collection
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            await self.record_metric("system_cpu_usage", cpu_percent / 100.0, "ratio", 
                                   category="system", source_component="system_monitor")
            await self.record_metric("system_memory_usage", memory.percent / 100.0, "ratio",
                                   category="system", source_component="system_monitor")
            await self.record_metric("system_disk_usage", disk.percent / 100.0, "ratio",
                                   category="system", source_component="system_monitor")
                                   
        except Exception as e:
            logger.warning(f"Could not collect system metrics: {e}")
            
    async def _check_performance_baseline(self, metric: PerformanceMetric):
        """Check metric against established baseline"""
        
        baseline = self.baselines.get(metric.metric_name)
        if not baseline:
            return
            
        # Check if value is outside acceptable range
        min_acceptable, max_acceptable = baseline.acceptable_range
        
        if metric.value < min_acceptable or metric.value > max_acceptable:
            severity = "warning"
            
            # Check if critical threshold exceeded
            if metric.value > baseline.critical_threshold:
                severity = "critical"
                
            await self.emit_event(
                EventType.OPSEC_VIOLATION,
                {
                    "metric": metric.metric_name,
                    "value": metric.value,
                    "baseline": baseline.baseline_value,
                    "severity": severity,
                    "component": metric.source_component
                }
            )
            
    async def _detect_bottlenecks(self, metric: PerformanceMetric):
        """Detect performance bottlenecks"""
        
        # Look for sustained high values
        component = metric.source_component
        recent_metrics = list(self.component_metrics[component])[-10:]
        
        if len(recent_metrics) >= 5:
            # Check for sustained high response times
            if metric.metric_name.endswith("_time") and metric.value > 5.0:
                recent_times = [m.value for m in recent_metrics 
                               if m.metric_name == metric.metric_name]
                if len(recent_times) >= 3 and statistics.mean(recent_times) > 5.0:
                    self.bottleneck_detection[component].append(datetime.utcnow())
                    
            # Check for sustained low success rates
            elif metric.metric_name.endswith("_rate") and metric.value < 0.7:
                recent_rates = [m.value for m in recent_metrics 
                               if m.metric_name == metric.metric_name]
                if len(recent_rates) >= 3 and statistics.mean(recent_rates) < 0.7:
                    self.bottleneck_detection[component].append(datetime.utcnow())
                    
        # Cleanup old bottleneck records
        cutoff = datetime.utcnow() - timedelta(hours=1)
        for component in self.bottleneck_detection:
            self.bottleneck_detection[component] = [
                timestamp for timestamp in self.bottleneck_detection[component]
                if timestamp > cutoff
            ]
            
    async def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        
        health_factors = []
        
        # Component health
        for component, health in self.component_health.items():
            health_factors.append(health)
            
        # Resource utilization health
        recent_window = datetime.utcnow() - timedelta(minutes=10)
        recent_metrics = [m for m in self.metrics if m.timestamp > recent_window]
        
        system_metrics = [m for m in recent_metrics if m.category == "system"]
        if system_metrics:
            avg_cpu = statistics.mean([m.value for m in system_metrics 
                                     if "cpu" in m.metric_name])
            avg_memory = statistics.mean([m.value for m in system_metrics 
                                        if "memory" in m.metric_name])
            
            # Health decreases as utilization approaches 100%
            cpu_health = max(0, 1.0 - (avg_cpu - 0.5) * 2)  # Optimal at 50%
            memory_health = max(0, 1.0 - (avg_memory - 0.7) * 3.33)  # Optimal at 70%
            
            health_factors.extend([cpu_health, memory_health])
            
        # Baseline violation factor
        violation_count = await self._count_baseline_violations(recent_metrics)
        violation_factor = max(0, 1.0 - (violation_count / len(recent_metrics) if recent_metrics else 0))
        health_factors.append(violation_factor)
        
        # Overall health is average of all factors
        overall_health = statistics.mean(health_factors) if health_factors else 0.5
        
        return min(overall_health, 1.0)
        
    async def _calculate_component_health(self, component_metrics: List[PerformanceMetric]) -> float:
        """Calculate health score for a component"""
        
        if not component_metrics:
            return 0.5
            
        health_factors = []
        
        # Response time factor
        response_times = [m.value for m in component_metrics if "time" in m.metric_name]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            time_health = max(0, 1.0 - (avg_response_time / 10.0))  # Health decreases as time increases
            health_factors.append(time_health)
            
        # Success rate factor
        success_rates = [m.value for m in component_metrics if "rate" in m.metric_name or "success" in m.metric_name]
        if success_rates:
            avg_success_rate = statistics.mean(success_rates)
            health_factors.append(avg_success_rate)
            
        # Error rate factor (inverted)
        error_rates = [m.value for m in component_metrics if "error" in m.metric_name]
        if error_rates:
            avg_error_rate = statistics.mean(error_rates)
            error_health = max(0, 1.0 - avg_error_rate)
            health_factors.append(error_health)
            
        component_health = statistics.mean(health_factors) if health_factors else 0.5
        return min(component_health, 1.0)
        
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        
        trend_analysis = {}
        
        for metric_name, trend_data in self.performance_trends.items():
            if len(trend_data) < 5:
                continue
                
            # Calculate trend direction
            recent_data = trend_data[-10:]
            older_data = trend_data[-20:-10] if len(trend_data) >= 20 else []
            
            if older_data:
                recent_avg = statistics.mean(recent_data)
                older_avg = statistics.mean(older_data)
                
                change_percent = ((recent_avg - older_avg) / older_avg) * 100
                
                if abs(change_percent) > 5:  # Significant change threshold
                    trend_analysis[metric_name] = {
                        "direction": "improving" if change_percent < 0 and "time" in metric_name else 
                                   "improving" if change_percent > 0 and "rate" in metric_name else
                                   "degrading",
                        "change_percent": abs(change_percent),
                        "recent_average": recent_avg,
                        "trend_strength": min(abs(change_percent) / 20.0, 1.0)
                    }
                    
        return trend_analysis
        
    async def _get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization"""
        
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "current": cpu_percent,
                    "status": "high" if cpu_percent > 80 else "normal" if cpu_percent > 50 else "low"
                },
                "memory": {
                    "current": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "status": "high" if memory.percent > 85 else "normal" if memory.percent > 70 else "low"
                },
                "disk": {
                    "current": disk.percent,
                    "free_gb": disk.free / (1024**3),
                    "status": "high" if disk.percent > 90 else "normal" if disk.percent > 70 else "low"
                }
            }
        except Exception as e:
            logger.warning(f"Could not get resource utilization: {e}")
            return {"error": "Resource monitoring unavailable"}
            
    async def _identify_active_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify currently active bottlenecks"""
        
        active_bottlenecks = []
        recent_window = datetime.utcnow() - timedelta(minutes=15)
        
        for component, bottleneck_times in self.bottleneck_detection.items():
            recent_bottlenecks = [t for t in bottleneck_times if t > recent_window]
            
            if len(recent_bottlenecks) >= 2:  # Multiple recent bottlenecks
                active_bottlenecks.append({
                    "component": component,
                    "frequency": len(recent_bottlenecks),
                    "last_occurrence": max(recent_bottlenecks).isoformat(),
                    "severity": "high" if len(recent_bottlenecks) > 5 else "medium"
                })
                
        return active_bottlenecks
        
    def _get_health_status(self, health_score: float) -> str:
        """Get health status string from score"""
        
        if health_score >= 0.9:
            return "excellent"
        elif health_score >= 0.7:
            return "good"
        elif health_score >= 0.5:
            return "fair"
        elif health_score >= 0.3:
            return "poor"
        else:
            return "critical"
            
    def _get_category_breakdown(self, metrics: List[PerformanceMetric]) -> Dict[str, int]:
        """Get breakdown of metrics by category"""
        
        categories = defaultdict(int)
        for metric in metrics:
            categories[metric.category] += 1
            
        return dict(categories)
        
    def _calculate_optimization_success_rate(self) -> float:
        """Calculate success rate of optimizations"""
        
        if not self.optimization_history:
            return 0.0
            
        successful_optimizations = [opt for opt in self.optimization_history 
                                   if opt.result and opt.result.get("success", False)]
        
        return len(successful_optimizations) / len(self.optimization_history)
        
    async def _get_pending_recommendations(self) -> List[str]:
        """Get pending optimization recommendations"""
        
        recommendations = []
        
        # Check for high resource usage
        recent_window = datetime.utcnow() - timedelta(minutes=10)
        recent_metrics = [m for m in self.metrics if m.timestamp > recent_window]
        
        system_metrics = [m for m in recent_metrics if m.category == "system"]
        
        for metric in system_metrics:
            if "cpu" in metric.metric_name and metric.value > 0.8:
                recommendations.append("Consider reducing CPU-intensive operations")
            elif "memory" in metric.metric_name and metric.value > 0.85:
                recommendations.append("Memory usage is high - consider cleanup or optimization")
                
        # Check for performance degradation
        trend_analysis = await self._analyze_performance_trends()
        
        for metric_name, trend in trend_analysis.items():
            if trend["direction"] == "degrading" and trend["trend_strength"] > 0.5:
                recommendations.append(f"Performance degradation detected in {metric_name}")
                
        return recommendations[:5]  # Top 5 recommendations
        
    async def _count_baseline_violations(self, metrics: List[PerformanceMetric]) -> int:
        """Count baseline violations in metrics"""
        
        violations = 0
        
        for metric in metrics:
            baseline = self.baselines.get(metric.metric_name)
            if baseline:
                min_acceptable, max_acceptable = baseline.acceptable_range
                if metric.value < min_acceptable or metric.value > max_acceptable:
                    violations += 1
                    
        return violations
        
    async def _identify_performance_issues(self, target_component: str = None) -> List[Dict[str, Any]]:
        """Identify current performance issues"""
        
        issues = []
        recent_window = datetime.utcnow() - timedelta(minutes=30)
        
        # Analyze by component
        components_to_check = [target_component] if target_component else self.component_metrics.keys()
        
        for component in components_to_check:
            component_metrics = [m for m in self.component_metrics[component] 
                               if m.timestamp > recent_window]
            
            if not component_metrics:
                continue
                
            # Check for high response times
            response_times = [m.value for m in component_metrics if "time" in m.metric_name]
            if response_times and statistics.mean(response_times) > 5.0:
                issues.append({
                    "type": "high_response_time",
                    "component": component,
                    "severity": "high",
                    "avg_value": statistics.mean(response_times),
                    "threshold": 5.0
                })
                
            # Check for low success rates
            success_rates = [m.value for m in component_metrics if "rate" in m.metric_name]
            if success_rates and statistics.mean(success_rates) < 0.8:
                issues.append({
                    "type": "low_success_rate",
                    "component": component,
                    "severity": "medium",
                    "avg_value": statistics.mean(success_rates),
                    "threshold": 0.8
                })
                
        return issues
        
    async def _generate_optimization_strategies(self, issue: Dict[str, Any]) -> List[OptimizationAction]:
        """Generate optimization strategies for an issue"""
        
        strategies = []
        
        if issue["type"] == "high_response_time":
            strategies.append(OptimizationAction(
                action_id=f"optimize_timeout_{issue['component']}_{int(time.time())}",
                action_type="parameter_adjustment",
                target_component=issue["component"],
                description="Increase timeout thresholds and implement caching",
                parameters={"timeout_multiplier": 1.5, "enable_caching": True},
                expected_improvement=0.3,
                risk_level=0.2,
                execution_time=datetime.utcnow()
            ))
            
        elif issue["type"] == "low_success_rate":
            strategies.append(OptimizationAction(
                action_id=f"retry_strategy_{issue['component']}_{int(time.time())}",
                action_type="algorithm_switch",
                target_component=issue["component"],
                description="Implement intelligent retry strategy with backoff",
                parameters={"max_retries": 3, "backoff_factor": 1.5},
                expected_improvement=0.2,
                risk_level=0.3,
                execution_time=datetime.utcnow()
            ))
            
        return strategies
        
    async def _execute_optimization(self, strategy: OptimizationAction) -> OptimizationAction:
        """Execute an optimization strategy"""
        
        try:
            # Simulate optimization execution
            # In real implementation, this would apply actual optimizations
            
            await asyncio.sleep(0.1)  # Simulate execution time
            
            # Record successful execution
            strategy.result = {
                "success": True,
                "execution_time": 0.1,
                "message": f"Applied {strategy.action_type} to {strategy.target_component}"
            }
            
            logger.info(f"Executed optimization: {strategy.description}")
            
        except Exception as e:
            strategy.result = {
                "success": False,
                "error": str(e),
                "message": f"Failed to apply optimization: {e}"
            }
            
            logger.error(f"Optimization failed: {strategy.action_id} - {e}")
            
        self.optimization_history.append(strategy)
        
        return strategy
        
    async def _predict_metric_trend(self, metric_name: str, trend_data: List[float], 
                                  hours_ahead: int) -> Dict[str, Any]:
        """Predict metric trend using linear regression"""
        
        if len(trend_data) < 10:
            return {"confidence": 0.0}
            
        # Simple linear regression
        x_values = list(range(len(trend_data)))
        y_values = trend_data
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict future value
        future_x = len(trend_data) + (hours_ahead * 6)  # Assuming 10-minute intervals
        predicted_value = slope * future_x + intercept
        
        # Calculate confidence based on R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum((y_values[i] - (slope * x_values[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Estimate time to critical threshold
        baseline = self.baselines.get(metric_name)
        time_to_threshold = None
        
        if baseline and slope != 0:
            threshold_x = (baseline.critical_threshold - intercept) / slope
            time_to_threshold = max(0, (threshold_x - len(trend_data)) / 6)  # Convert to hours
            
        return {
            "predicted_value": predicted_value,
            "confidence": max(0, min(r_squared, 1.0)),
            "slope": slope,
            "time_to_threshold": time_to_threshold
        }
        
    # Background monitoring loops
    
    async def _metrics_collection_loop(self):
        """Background metrics collection"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Collect system metrics
                await self._start_system_monitoring()
                
                # Update component health scores
                for component, metrics in self.component_metrics.items():
                    recent_metrics = [m for m in metrics 
                                    if m.timestamp > datetime.utcnow() - timedelta(minutes=10)]
                    self.component_health[component] = await self._calculate_component_health(recent_metrics)
                    
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)
                
    async def _optimization_loop(self):
        """Background optimization loop"""
        
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                if self.optimization_enabled:
                    # Check for optimization opportunities
                    await self.optimize_performance(aggressive=False)
                    
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)
                
    async def _trend_analysis_loop(self):
        """Background trend analysis"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Analyze performance trends
                trend_analysis = await self._analyze_performance_trends()
                
                # Update baselines based on learned performance
                await self._update_learned_baselines()
                
                # Predict performance issues
                predictions = await self.predict_performance_issues(24)
                
                if predictions:
                    await self.emit_event(
                        EventType.REASONING_UPDATED,
                        {"action": "performance_predictions", "predictions": len(predictions)}
                    )
                    
            except Exception as e:
                logger.error(f"Error in trend analysis loop: {e}")
                await asyncio.sleep(300)
                
    async def _system_health_loop(self):
        """Background system health monitoring"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Calculate overall system health
                system_health = await self._calculate_system_health()
                
                # Record system health as metric
                await self.record_metric("system_health_score", system_health, "score",
                                       category="system", source_component="health_monitor")
                
                # Trigger alerts for critical health
                if system_health < 0.3:
                    await self.emit_event(
                        EventType.OPSEC_VIOLATION,
                        {"type": "critical_system_health", "health_score": system_health}
                    )
                    
            except Exception as e:
                logger.error(f"Error in system health loop: {e}")
                await asyncio.sleep(60)
                
    async def _update_learned_baselines(self):
        """Update baselines based on learned performance patterns"""
        
        # Look for stable performance patterns over the learning period
        cutoff_date = datetime.utcnow() - timedelta(days=self.baseline_learning_period)
        learning_metrics = [m for m in self.metrics if m.timestamp > cutoff_date]
        
        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in learning_metrics:
            metric_groups[metric.metric_name].append(metric.value)
            
        # Update baselines for metrics with sufficient data
        for metric_name, values in metric_groups.items():
            if len(values) > 100:  # Need sufficient sample size
                
                # Calculate stable performance characteristics
                median_value = statistics.median(values)
                std_dev = statistics.stdev(values)
                
                # Update baseline if current one is significantly off
                current_baseline = self.baselines.get(metric_name)
                if current_baseline:
                    baseline_diff = abs(current_baseline.baseline_value - median_value)
                    
                    if baseline_diff > std_dev:  # Significant difference
                        # Update baseline with learned values
                        new_range = (median_value - 2*std_dev, median_value + 2*std_dev)
                        new_critical = median_value + 3*std_dev
                        
                        await self.set_performance_baseline(
                            metric_name=metric_name,
                            target_value=median_value,
                            acceptable_range=new_range,
                            critical_threshold=new_critical
                        )
                        
                        logger.info(f"Updated learned baseline for {metric_name}: {median_value}")
                        
    async def shutdown(self):
        """Shutdown performance monitoring system"""
        logger.info("Performance monitoring system shutdown complete")