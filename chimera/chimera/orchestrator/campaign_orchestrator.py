"""
Intelligent Campaign Orchestration System
Coordinates multiple campaigns, optimizes resource allocation, and manages campaign priorities
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import statistics

from ..core.events import EventSystem, EventType, EventEmitter

logger = logging.getLogger(__name__)

class CampaignPriority(Enum):
    """Campaign priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class CampaignStatus(Enum):
    """Campaign status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class CampaignRequest:
    """A request for a campaign to be executed"""
    
    campaign_id: str
    target: str
    scope_file: str
    priority: CampaignPriority
    requested_by: str
    created_at: datetime
    deadline: Optional[datetime] = None
    resource_requirements: Dict[str, float] = None  # CPU, memory, network
    tags: List[str] = None
    dependencies: List[str] = None  # Other campaign IDs this depends on
    
    def __post_init__(self):
        if self.resource_requirements is None:
            self.resource_requirements = {"cpu": 0.5, "memory": 0.3, "network": 0.2}
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class CampaignExecution:
    """A campaign being executed"""
    
    request: CampaignRequest
    status: CampaignStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    current_phase: str = ""
    allocated_resources: Dict[str, float] = None
    estimated_completion: Optional[datetime] = None
    findings_count: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.allocated_resources is None:
            self.allocated_resources = {}

@dataclass
class ResourcePool:
    """Available resource pool for campaigns"""
    
    total_cpu: float
    total_memory: float
    total_network: float
    available_cpu: float
    available_memory: float
    available_network: float
    reserved_resources: Dict[str, Dict[str, float]]  # campaign_id -> resources
    
    def __post_init__(self):
        if self.reserved_resources is None:
            self.reserved_resources = {}

class CampaignOrchestrator(EventEmitter):
    """
    Intelligent Campaign Orchestration System
    
    Features:
    - Multi-campaign priority-based scheduling
    - Resource allocation optimization
    - Dependency management
    - Progress tracking and estimation
    - Adaptive campaign parallelization
    - Load balancing across resources
    - Campaign failure recovery
    """
    
    def __init__(self, config, event_system: EventSystem):
        super().__init__(event_system, "CampaignOrchestrator")
        
        self.config = config
        
        # Campaign management
        self.pending_campaigns: deque = deque()
        self.running_campaigns: Dict[str, CampaignExecution] = {}
        self.completed_campaigns: Dict[str, CampaignExecution] = {}
        
        # Resource management
        self.resource_pool = ResourcePool(
            total_cpu=config.get("orchestrator.total_cpu", 4.0),
            total_memory=config.get("orchestrator.total_memory", 8.0),
            total_network=config.get("orchestrator.total_network", 2.0),
            available_cpu=config.get("orchestrator.total_cpu", 4.0),
            available_memory=config.get("orchestrator.total_memory", 8.0),
            available_network=config.get("orchestrator.total_network", 2.0)
        )
        
        # Orchestration parameters
        self.max_concurrent_campaigns = config.get("orchestrator.max_concurrent", 3)
        self.min_resource_threshold = config.get("orchestrator.min_resource_threshold", 0.1)
        self.campaign_timeout = config.get("orchestrator.campaign_timeout_hours", 24)
        self.dependency_timeout = config.get("orchestrator.dependency_timeout_hours", 48)
        
        # Performance tracking
        self.campaign_statistics: Dict[str, List[float]] = defaultdict(list)
        self.resource_utilization_history: deque = deque(maxlen=1000)
        self.scheduling_decisions: List[Dict[str, Any]] = []
        
        # Optimization parameters
        self.optimization_enabled = config.get("orchestrator.optimization_enabled", True)
        self.adaptive_scaling = config.get("orchestrator.adaptive_scaling", True)
        self.predictive_scheduling = config.get("orchestrator.predictive_scheduling", True)
        
    async def initialize(self):
        """Initialize campaign orchestrator"""
        
        # Start background orchestration loops
        asyncio.create_task(self._orchestration_loop())
        asyncio.create_task(self._resource_monitoring_loop())
        asyncio.create_task(self._dependency_resolution_loop())
        asyncio.create_task(self._optimization_loop())
        
        await self.emit_event(
            EventType.REASONING_START,
            {"message": "Campaign orchestrator initialized", "max_concurrent": self.max_concurrent_campaigns}
        )
        
        logger.info("Campaign orchestrator initialized")
        
    async def submit_campaign(self, target: str, scope_file: str, 
                            priority: CampaignPriority = CampaignPriority.MEDIUM,
                            requested_by: str = "user",
                            deadline: Optional[datetime] = None,
                            resource_requirements: Dict[str, float] = None,
                            tags: List[str] = None,
                            dependencies: List[str] = None) -> str:
        """
        Submit a campaign for execution
        
        Returns:
            Campaign ID
        """
        
        campaign_id = str(uuid.uuid4())
        
        campaign_request = CampaignRequest(
            campaign_id=campaign_id,
            target=target,
            scope_file=scope_file,
            priority=priority,
            requested_by=requested_by,
            created_at=datetime.utcnow(),
            deadline=deadline,
            resource_requirements=resource_requirements,
            tags=tags or [],
            dependencies=dependencies or []
        )
        
        # Add to pending queue
        self.pending_campaigns.append(campaign_request)
        
        await self.emit_event(
            EventType.TASK_START,
            {
                "campaign_id": campaign_id,
                "target": target,
                "priority": priority.name,
                "queue_position": len(self.pending_campaigns)
            }
        )
        
        logger.info(f"Campaign {campaign_id} submitted for target {target} with priority {priority.name}")
        
        return campaign_id
        
    async def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific campaign"""
        
        # Check running campaigns
        if campaign_id in self.running_campaigns:
            execution = self.running_campaigns[campaign_id]
            return {
                "campaign_id": campaign_id,
                "status": execution.status.value,
                "progress": execution.progress,
                "current_phase": execution.current_phase,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "estimated_completion": execution.estimated_completion.isoformat() if execution.estimated_completion else None,
                "findings_count": execution.findings_count,
                "allocated_resources": execution.allocated_resources
            }
            
        # Check completed campaigns
        if campaign_id in self.completed_campaigns:
            execution = self.completed_campaigns[campaign_id]
            return {
                "campaign_id": campaign_id,
                "status": execution.status.value,
                "progress": execution.progress,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "findings_count": execution.findings_count,
                "error_message": execution.error_message
            }
            
        # Check pending campaigns
        for i, request in enumerate(self.pending_campaigns):
            if request.campaign_id == campaign_id:
                return {
                    "campaign_id": campaign_id,
                    "status": "pending",
                    "queue_position": i + 1,
                    "priority": request.priority.name,
                    "estimated_start": await self._estimate_start_time(i)
                }
                
        return None
        
    async def cancel_campaign(self, campaign_id: str) -> bool:
        """Cancel a campaign"""
        
        # Remove from pending queue
        for i, request in enumerate(self.pending_campaigns):
            if request.campaign_id == campaign_id:
                del self.pending_campaigns[i]
                logger.info(f"Cancelled pending campaign {campaign_id}")
                return True
                
        # Cancel running campaign
        if campaign_id in self.running_campaigns:
            execution = self.running_campaigns[campaign_id]
            execution.status = CampaignStatus.CANCELLED
            execution.completed_at = datetime.utcnow()
            
            # Free resources
            await self._free_campaign_resources(campaign_id)
            
            # Move to completed
            self.completed_campaigns[campaign_id] = execution
            del self.running_campaigns[campaign_id]
            
            await self.emit_event(
                EventType.TASK_CANCELLED,
                {"campaign_id": campaign_id}
            )
            
            logger.info(f"Cancelled running campaign {campaign_id}")
            return True
            
        return False
        
    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause a running campaign"""
        
        if campaign_id in self.running_campaigns:
            execution = self.running_campaigns[campaign_id]
            if execution.status == CampaignStatus.RUNNING:
                execution.status = CampaignStatus.PAUSED
                
                await self.emit_event(
                    EventType.TASK_PAUSED,
                    {"campaign_id": campaign_id}
                )
                
                logger.info(f"Paused campaign {campaign_id}")
                return True
                
        return False
        
    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume a paused campaign"""
        
        if campaign_id in self.running_campaigns:
            execution = self.running_campaigns[campaign_id]
            if execution.status == CampaignStatus.PAUSED:
                execution.status = CampaignStatus.RUNNING
                
                await self.emit_event(
                    EventType.TASK_RESUMED,
                    {"campaign_id": campaign_id}
                )
                
                logger.info(f"Resumed campaign {campaign_id}")
                return True
                
        return False
        
    async def get_orchestration_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive orchestration dashboard"""
        
        # Campaign statistics
        total_campaigns = len(self.running_campaigns) + len(self.completed_campaigns) + len(self.pending_campaigns)
        completed_campaigns = len([c for c in self.completed_campaigns.values() 
                                 if c.status == CampaignStatus.COMPLETED])
        failed_campaigns = len([c for c in self.completed_campaigns.values() 
                              if c.status == CampaignStatus.FAILED])
        
        # Resource utilization
        cpu_utilization = (self.resource_pool.total_cpu - self.resource_pool.available_cpu) / self.resource_pool.total_cpu
        memory_utilization = (self.resource_pool.total_memory - self.resource_pool.available_memory) / self.resource_pool.total_memory
        network_utilization = (self.resource_pool.total_network - self.resource_pool.available_network) / self.resource_pool.total_network
        
        # Performance metrics
        completion_times = [stat for stat_list in self.campaign_statistics.values() for stat in stat_list]
        avg_completion_time = statistics.mean(completion_times) if completion_times else 0.0
        
        # Queue analysis
        queue_by_priority = defaultdict(int)
        for request in self.pending_campaigns:
            queue_by_priority[request.priority.name] += 1
            
        return {
            "campaign_statistics": {
                "total_campaigns": total_campaigns,
                "running_campaigns": len(self.running_campaigns),
                "pending_campaigns": len(self.pending_campaigns),
                "completed_campaigns": completed_campaigns,
                "failed_campaigns": failed_campaigns,
                "success_rate": completed_campaigns / max(completed_campaigns + failed_campaigns, 1)
            },
            "resource_utilization": {
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "network_utilization": network_utilization,
                "available_slots": self.max_concurrent_campaigns - len(self.running_campaigns)
            },
            "performance_metrics": {
                "average_completion_time": avg_completion_time,
                "concurrent_campaigns": len(self.running_campaigns),
                "max_concurrent": self.max_concurrent_campaigns,
                "queue_depth": len(self.pending_campaigns)
            },
            "queue_analysis": {
                "queue_by_priority": dict(queue_by_priority),
                "oldest_pending": min([r.created_at for r in self.pending_campaigns], default=None),
                "estimated_queue_time": await self._estimate_queue_time()
            },
            "optimization_status": {
                "optimization_enabled": self.optimization_enabled,
                "adaptive_scaling": self.adaptive_scaling,
                "predictive_scheduling": self.predictive_scheduling,
                "recent_decisions": len(self.scheduling_decisions[-10:])
            }
        }
        
    async def optimize_scheduling(self) -> Dict[str, Any]:
        """Optimize campaign scheduling and resource allocation"""
        
        optimization_results = {
            "actions_taken": [],
            "performance_improvements": {},
            "resource_reallocation": {}
        }
        
        # Analyze current scheduling efficiency
        efficiency_score = await self._calculate_scheduling_efficiency()
        
        # Reorder pending campaigns for optimal execution
        if len(self.pending_campaigns) > 1:
            original_order = list(self.pending_campaigns)
            optimized_order = await self._optimize_campaign_order()
            
            if optimized_order != original_order:
                self.pending_campaigns = deque(optimized_order)
                optimization_results["actions_taken"].append("Reordered pending campaign queue for optimal execution")
                
        # Optimize resource allocation for running campaigns
        resource_optimizations = await self._optimize_resource_allocation()
        optimization_results["resource_reallocation"] = resource_optimizations
        
        # Adaptive scaling recommendations
        if self.adaptive_scaling:
            scaling_recommendations = await self._generate_scaling_recommendations()
            optimization_results["scaling_recommendations"] = scaling_recommendations
            
        # Update scheduling parameters based on performance
        parameter_updates = await self._update_scheduling_parameters()
        optimization_results["parameter_updates"] = parameter_updates
        
        new_efficiency_score = await self._calculate_scheduling_efficiency()
        optimization_results["performance_improvements"]["efficiency_improvement"] = new_efficiency_score - efficiency_score
        
        return optimization_results
        
    # Private methods
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check for campaigns to start
                await self._start_pending_campaigns()
                
                # Update campaign progress
                await self._update_campaign_progress()
                
                # Check for completed campaigns
                await self._check_completed_campaigns()
                
                # Handle campaign timeouts
                await self._handle_campaign_timeouts()
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(30)
                
    async def _start_pending_campaigns(self):
        """Start pending campaigns based on available resources and priorities"""
        
        if not self.pending_campaigns:
            return
            
        # Check if we can start more campaigns
        if len(self.running_campaigns) >= self.max_concurrent_campaigns:
            return
            
        # Sort pending campaigns by priority and creation time
        sorted_pending = sorted(self.pending_campaigns, 
                               key=lambda x: (x.priority.value, x.created_at), 
                               reverse=True)
        
        for request in sorted_pending:
            # Check dependencies
            if not await self._dependencies_satisfied(request):
                continue
                
            # Check resource availability
            if await self._can_allocate_resources(request.resource_requirements):
                # Remove from pending queue
                self.pending_campaigns.remove(request)
                
                # Allocate resources and start campaign
                allocated_resources = await self._allocate_resources(request.resource_requirements)
                
                execution = CampaignExecution(
                    request=request,
                    status=CampaignStatus.RUNNING,
                    started_at=datetime.utcnow(),
                    allocated_resources=allocated_resources,
                    estimated_completion=await self._estimate_completion_time(request)
                )
                
                self.running_campaigns[request.campaign_id] = execution
                
                # Record scheduling decision
                self.scheduling_decisions.append({
                    "timestamp": datetime.utcnow(),
                    "campaign_id": request.campaign_id,
                    "action": "started",
                    "resources_allocated": allocated_resources,
                    "queue_position": len(sorted_pending) - sorted_pending.index(request)
                })
                
                await self.emit_event(
                    EventType.TASK_START,
                    {
                        "campaign_id": request.campaign_id,
                        "target": request.target,
                        "allocated_resources": allocated_resources
                    }
                )
                
                logger.info(f"Started campaign {request.campaign_id} for target {request.target}")
                
                # Check if we've reached capacity
                if len(self.running_campaigns) >= self.max_concurrent_campaigns:
                    break
                    
    async def _dependencies_satisfied(self, request: CampaignRequest) -> bool:
        """Check if campaign dependencies are satisfied"""
        
        for dep_id in request.dependencies:
            # Check if dependency is completed successfully
            if dep_id in self.completed_campaigns:
                dep_execution = self.completed_campaigns[dep_id]
                if dep_execution.status != CampaignStatus.COMPLETED:
                    return False
            else:
                # Dependency not completed yet
                return False
                
        return True
        
    async def _can_allocate_resources(self, requirements: Dict[str, float]) -> bool:
        """Check if required resources can be allocated"""
        
        required_cpu = requirements.get("cpu", 0.0)
        required_memory = requirements.get("memory", 0.0)
        required_network = requirements.get("network", 0.0)
        
        return (self.resource_pool.available_cpu >= required_cpu and
                self.resource_pool.available_memory >= required_memory and
                self.resource_pool.available_network >= required_network)
                
    async def _allocate_resources(self, requirements: Dict[str, float]) -> Dict[str, float]:
        """Allocate resources for a campaign"""
        
        allocated = {
            "cpu": requirements.get("cpu", 0.0),
            "memory": requirements.get("memory", 0.0),
            "network": requirements.get("network", 0.0)
        }
        
        # Deduct from available pool
        self.resource_pool.available_cpu -= allocated["cpu"]
        self.resource_pool.available_memory -= allocated["memory"]
        self.resource_pool.available_network -= allocated["network"]
        
        return allocated
        
    async def _free_campaign_resources(self, campaign_id: str):
        """Free resources allocated to a campaign"""
        
        if campaign_id in self.running_campaigns:
            execution = self.running_campaigns[campaign_id]
            resources = execution.allocated_resources
            
            # Return resources to pool
            self.resource_pool.available_cpu += resources.get("cpu", 0.0)
            self.resource_pool.available_memory += resources.get("memory", 0.0)
            self.resource_pool.available_network += resources.get("network", 0.0)
            
            # Ensure we don't exceed total capacity
            self.resource_pool.available_cpu = min(self.resource_pool.available_cpu, self.resource_pool.total_cpu)
            self.resource_pool.available_memory = min(self.resource_pool.available_memory, self.resource_pool.total_memory)
            self.resource_pool.available_network = min(self.resource_pool.available_network, self.resource_pool.total_network)
            
    async def _update_campaign_progress(self):
        """Update progress of running campaigns"""
        
        for campaign_id, execution in self.running_campaigns.items():
            if execution.status == CampaignStatus.RUNNING:
                # Simulate progress based on time elapsed
                if execution.started_at:
                    elapsed = (datetime.utcnow() - execution.started_at).total_seconds()
                    estimated_duration = 3600  # 1 hour default
                    
                    # Update progress (capped at 95% until actually complete)
                    execution.progress = min(elapsed / estimated_duration * 0.95, 0.95)
                    
                    # Update current phase based on progress
                    if execution.progress < 0.2:
                        execution.current_phase = "reconnaissance"
                    elif execution.progress < 0.4:
                        execution.current_phase = "assessment"
                    elif execution.progress < 0.6:
                        execution.current_phase = "planning"
                    elif execution.progress < 0.8:
                        execution.current_phase = "exploitation"
                    else:
                        execution.current_phase = "reporting"
                        
    async def _check_completed_campaigns(self):
        """Check for completed campaigns and handle completion"""
        
        completed_campaign_ids = []
        
        for campaign_id, execution in self.running_campaigns.items():
            # Simulate completion based on progress or actual completion signal
            if execution.progress >= 0.95 and execution.status == CampaignStatus.RUNNING:
                # Mark as completed
                execution.status = CampaignStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.progress = 1.0
                execution.findings_count = 5  # Simulated findings
                
                completed_campaign_ids.append(campaign_id)
                
        # Move completed campaigns
        for campaign_id in completed_campaign_ids:
            execution = self.running_campaigns[campaign_id]
            
            # Free resources
            await self._free_campaign_resources(campaign_id)
            
            # Record completion statistics
            if execution.started_at and execution.completed_at:
                duration = (execution.completed_at - execution.started_at).total_seconds() / 3600  # hours
                self.campaign_statistics[execution.request.target].append(duration)
                
            # Move to completed
            self.completed_campaigns[campaign_id] = execution
            del self.running_campaigns[campaign_id]
            
            await self.emit_event(
                EventType.TASK_COMPLETE,
                {
                    "campaign_id": campaign_id,
                    "findings_count": execution.findings_count,
                    "duration": duration if execution.started_at and execution.completed_at else 0
                }
            )
            
            logger.info(f"Campaign {campaign_id} completed with {execution.findings_count} findings")
            
    async def _handle_campaign_timeouts(self):
        """Handle campaign timeouts"""
        
        timeout_threshold = datetime.utcnow() - timedelta(hours=self.campaign_timeout)
        timed_out_campaigns = []
        
        for campaign_id, execution in self.running_campaigns.items():
            if execution.started_at and execution.started_at < timeout_threshold:
                timed_out_campaigns.append(campaign_id)
                
        for campaign_id in timed_out_campaigns:
            execution = self.running_campaigns[campaign_id]
            execution.status = CampaignStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.error_message = "Campaign timed out"
            
            # Free resources
            await self._free_campaign_resources(campaign_id)
            
            # Move to completed
            self.completed_campaigns[campaign_id] = execution
            del self.running_campaigns[campaign_id]
            
            await self.emit_event(
                EventType.TASK_FAILED,
                {"campaign_id": campaign_id, "error": "timeout"}
            )
            
            logger.warning(f"Campaign {campaign_id} timed out after {self.campaign_timeout} hours")
            
    async def _estimate_start_time(self, queue_position: int) -> Optional[str]:
        """Estimate when a pending campaign will start"""
        
        if queue_position == 0:
            return datetime.utcnow().isoformat()
            
        # Estimate based on average completion time and available slots
        avg_completion_time = 2.0  # Default 2 hours
        if self.campaign_statistics:
            all_times = [t for times in self.campaign_statistics.values() for t in times]
            if all_times:
                avg_completion_time = statistics.mean(all_times)
                
        available_slots = max(1, self.max_concurrent_campaigns - len(self.running_campaigns))
        estimated_hours = (queue_position / available_slots) * avg_completion_time
        
        estimated_start = datetime.utcnow() + timedelta(hours=estimated_hours)
        return estimated_start.isoformat()
        
    async def _estimate_completion_time(self, request: CampaignRequest) -> datetime:
        """Estimate completion time for a campaign"""
        
        # Base estimate on historical data for similar targets
        base_estimate = 2.0  # Default 2 hours
        
        if request.target in self.campaign_statistics:
            target_times = self.campaign_statistics[request.target]
            if target_times:
                base_estimate = statistics.mean(target_times)
                
        # Adjust based on resource allocation
        resource_factor = sum(request.resource_requirements.values()) / 3.0  # Average resource usage
        adjusted_estimate = base_estimate / max(resource_factor, 0.1)
        
        return datetime.utcnow() + timedelta(hours=adjusted_estimate)
        
    async def _estimate_queue_time(self) -> float:
        """Estimate total time to clear pending queue"""
        
        if not self.pending_campaigns:
            return 0.0
            
        avg_completion_time = 2.0  # Default 2 hours
        if self.campaign_statistics:
            all_times = [t for times in self.campaign_statistics.values() for t in times]
            if all_times:
                avg_completion_time = statistics.mean(all_times)
                
        available_slots = max(1, self.max_concurrent_campaigns - len(self.running_campaigns))
        queue_time = (len(self.pending_campaigns) / available_slots) * avg_completion_time
        
        return queue_time
        
    async def _optimize_campaign_order(self) -> List[CampaignRequest]:
        """Optimize the order of pending campaigns"""
        
        # Score campaigns based on multiple factors
        scored_campaigns = []
        
        for request in self.pending_campaigns:
            score = 0.0
            
            # Priority factor (higher priority = higher score)
            score += request.priority.value * 10
            
            # Age factor (older campaigns get priority)
            age_hours = (datetime.utcnow() - request.created_at).total_seconds() / 3600
            score += min(age_hours * 0.1, 5.0)  # Cap age bonus at 5 points
            
            # Deadline urgency
            if request.deadline:
                time_to_deadline = (request.deadline - datetime.utcnow()).total_seconds() / 3600
                if time_to_deadline > 0:
                    urgency = max(0, 10 - time_to_deadline)  # More urgent = higher score
                    score += urgency
                    
            # Resource efficiency (prefer campaigns that use resources efficiently)
            resource_efficiency = 1.0 / max(sum(request.resource_requirements.values()), 0.1)
            score += resource_efficiency
            
            scored_campaigns.append((score, request))
            
        # Sort by score (highest first)
        scored_campaigns.sort(key=lambda x: x[0], reverse=True)
        
        return [request for score, request in scored_campaigns]
        
    async def _optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation for running campaigns"""
        
        optimizations = {}
        
        # Analyze resource usage patterns
        for campaign_id, execution in self.running_campaigns.items():
            current_allocation = execution.allocated_resources
            
            # Simulate optimal allocation based on campaign phase
            optimal_allocation = await self._calculate_optimal_allocation(execution)
            
            # Check if reallocation would be beneficial
            if self._should_reallocate(current_allocation, optimal_allocation):
                optimizations[campaign_id] = {
                    "current": current_allocation,
                    "optimal": optimal_allocation,
                    "improvement": self._calculate_allocation_improvement(current_allocation, optimal_allocation)
                }
                
        return optimizations
        
    async def _calculate_optimal_allocation(self, execution: CampaignExecution) -> Dict[str, float]:
        """Calculate optimal resource allocation for a campaign"""
        
        base_allocation = execution.request.resource_requirements.copy()
        
        # Adjust based on current phase
        if execution.current_phase == "reconnaissance":
            # Network-heavy phase
            base_allocation["network"] *= 1.5
            base_allocation["cpu"] *= 0.8
        elif execution.current_phase == "exploitation":
            # CPU-heavy phase
            base_allocation["cpu"] *= 1.3
            base_allocation["memory"] *= 1.2
        elif execution.current_phase == "reporting":
            # Memory-heavy phase
            base_allocation["memory"] *= 1.4
            base_allocation["network"] *= 0.7
            
        return base_allocation
        
    def _should_reallocate(self, current: Dict[str, float], optimal: Dict[str, float]) -> bool:
        """Check if reallocation would be beneficial"""
        
        # Calculate total difference
        total_diff = sum(abs(optimal.get(k, 0) - current.get(k, 0)) for k in ["cpu", "memory", "network"])
        
        # Only reallocate if difference is significant
        return total_diff > 0.2
        
    def _calculate_allocation_improvement(self, current: Dict[str, float], optimal: Dict[str, float]) -> float:
        """Calculate improvement score from reallocation"""
        
        # Simple efficiency improvement calculation
        current_efficiency = sum(current.values())
        optimal_efficiency = sum(optimal.values())
        
        return (optimal_efficiency - current_efficiency) / max(current_efficiency, 0.1)
        
    async def _generate_scaling_recommendations(self) -> List[str]:
        """Generate adaptive scaling recommendations"""
        
        recommendations = []
        
        # Analyze resource utilization trends
        cpu_utilization = (self.resource_pool.total_cpu - self.resource_pool.available_cpu) / self.resource_pool.total_cpu
        memory_utilization = (self.resource_pool.total_memory - self.resource_pool.available_memory) / self.resource_pool.total_memory
        
        # Check if we should scale up
        if cpu_utilization > 0.8 and len(self.pending_campaigns) > 0:
            recommendations.append("Consider increasing CPU resources to handle pending campaigns")
            
        if memory_utilization > 0.8:
            recommendations.append("Consider increasing memory allocation for better performance")
            
        # Check if we should scale down
        if cpu_utilization < 0.3 and len(self.pending_campaigns) == 0:
            recommendations.append("Consider reducing resource allocation during low usage periods")
            
        # Check concurrent campaign limits
        if len(self.pending_campaigns) > self.max_concurrent_campaigns * 2:
            recommendations.append("Consider increasing maximum concurrent campaigns to reduce queue time")
            
        return recommendations
        
    async def _update_scheduling_parameters(self) -> Dict[str, Any]:
        """Update scheduling parameters based on performance"""
        
        updates = {}
        
        # Analyze recent scheduling efficiency
        recent_decisions = self.scheduling_decisions[-20:] if len(self.scheduling_decisions) >= 20 else self.scheduling_decisions
        
        if recent_decisions:
            # Calculate average queue time reduction
            avg_queue_positions = statistics.mean([d.get("queue_position", 0) for d in recent_decisions])
            
            # Adjust max concurrent based on performance
            if avg_queue_positions > 5 and self.resource_pool.available_cpu > 0.5:
                new_max_concurrent = min(self.max_concurrent_campaigns + 1, 8)
                if new_max_concurrent != self.max_concurrent_campaigns:
                    self.max_concurrent_campaigns = new_max_concurrent
                    updates["max_concurrent_campaigns"] = new_max_concurrent
                    
        return updates
        
    async def _calculate_scheduling_efficiency(self) -> float:
        """Calculate current scheduling efficiency"""
        
        if not self.running_campaigns and not self.completed_campaigns:
            return 0.5  # Neutral score
            
        efficiency_factors = []
        
        # Resource utilization efficiency
        total_utilization = (
            (self.resource_pool.total_cpu - self.resource_pool.available_cpu) / self.resource_pool.total_cpu +
            (self.resource_pool.total_memory - self.resource_pool.available_memory) / self.resource_pool.total_memory +
            (self.resource_pool.total_network - self.resource_pool.available_network) / self.resource_pool.total_network
        ) / 3.0
        
        efficiency_factors.append(total_utilization)
        
        # Queue management efficiency
        if self.pending_campaigns:
            avg_queue_age = statistics.mean([
                (datetime.utcnow() - r.created_at).total_seconds() / 3600 
                for r in self.pending_campaigns
            ])
            queue_efficiency = max(0, 1.0 - (avg_queue_age / 24.0))  # Efficiency decreases with age
            efficiency_factors.append(queue_efficiency)
        else:
            efficiency_factors.append(1.0)  # Perfect if no queue
            
        # Completion rate efficiency
        if self.completed_campaigns:
            completed_count = len([c for c in self.completed_campaigns.values() 
                                 if c.status == CampaignStatus.COMPLETED])
            completion_rate = completed_count / len(self.completed_campaigns)
            efficiency_factors.append(completion_rate)
        else:
            efficiency_factors.append(0.5)  # Neutral if no completed campaigns
            
        return statistics.mean(efficiency_factors)
        
    # Background monitoring loops
    
    async def _resource_monitoring_loop(self):
        """Background resource monitoring"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Record resource utilization
                utilization = {
                    "timestamp": datetime.utcnow(),
                    "cpu_utilization": (self.resource_pool.total_cpu - self.resource_pool.available_cpu) / self.resource_pool.total_cpu,
                    "memory_utilization": (self.resource_pool.total_memory - self.resource_pool.available_memory) / self.resource_pool.total_memory,
                    "network_utilization": (self.resource_pool.total_network - self.resource_pool.available_network) / self.resource_pool.total_network,
                    "active_campaigns": len(self.running_campaigns),
                    "pending_campaigns": len(self.pending_campaigns)
                }
                
                self.resource_utilization_history.append(utilization)
                
                # Check for resource exhaustion
                if utilization["cpu_utilization"] > 0.95:
                    await self.emit_event(
                        EventType.OPSEC_VIOLATION,
                        {"type": "resource_exhaustion", "resource": "cpu", "utilization": utilization["cpu_utilization"]}
                    )
                    
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(30)
                
    async def _dependency_resolution_loop(self):
        """Background dependency resolution"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Check for dependency timeouts
                timeout_threshold = datetime.utcnow() - timedelta(hours=self.dependency_timeout)
                
                timed_out_campaigns = []
                for request in self.pending_campaigns:
                    if request.dependencies and request.created_at < timeout_threshold:
                        # Check if any dependencies are still not satisfied
                        for dep_id in request.dependencies:
                            if dep_id not in self.completed_campaigns:
                                timed_out_campaigns.append(request)
                                break
                                
                # Handle timed out dependencies
                for request in timed_out_campaigns:
                    self.pending_campaigns.remove(request)
                    
                    # Create a failed execution record
                    execution = CampaignExecution(
                        request=request,
                        status=CampaignStatus.FAILED,
                        error_message="Dependency timeout"
                    )
                    execution.completed_at = datetime.utcnow()
                    
                    self.completed_campaigns[request.campaign_id] = execution
                    
                    logger.warning(f"Campaign {request.campaign_id} failed due to dependency timeout")
                    
            except Exception as e:
                logger.error(f"Error in dependency resolution loop: {e}")
                await asyncio.sleep(60)
                
    async def _optimization_loop(self):
        """Background optimization loop"""
        
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                if self.optimization_enabled:
                    # Run scheduling optimization
                    optimization_results = await self.optimize_scheduling()
                    
                    if optimization_results["actions_taken"]:
                        logger.info(f"Applied {len(optimization_results['actions_taken'])} scheduling optimizations")
                        
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)
                
    async def shutdown(self):
        """Shutdown campaign orchestrator"""
        
        # Cancel all running campaigns
        for campaign_id in list(self.running_campaigns.keys()):
            await self.cancel_campaign(campaign_id)
            
        logger.info("Campaign orchestrator shutdown complete")