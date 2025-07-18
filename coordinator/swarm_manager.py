"""
Enhanced Swarm Manager for Aetherveil Sentinel
Manages deployment, coordination, monitoring, auto-scaling, and load balancing of swarm agents
"""
import asyncio
import json
import logging
import uuid
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import zmq
import zmq.asyncio

from config.config import config
from coordinator.models import Agent, Task, AgentType, TaskStatus
from coordinator.security import security_manager

logger = logging.getLogger(__name__)

class LoadBalancer:
    """Load balancer for distributing tasks among agents"""
    
    def __init__(self):
        self.algorithms = {
            "round_robin": self._round_robin,
            "least_connections": self._least_connections,
            "weighted_round_robin": self._weighted_round_robin,
            "resource_aware": self._resource_aware,
            "capability_based": self._capability_based
        }
        self.current_algorithm = "resource_aware"
        self.round_robin_counters = {}
        
    def select_agent(self, agents: List[Agent], task_type: str, task_priority: int = 5) -> Optional[Agent]:
        """Select best agent for task using current algorithm"""
        try:
            if not agents:
                return None
                
            algorithm = self.algorithms.get(self.current_algorithm, self._round_robin)
            return algorithm(agents, task_type, task_priority)
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            return agents[0] if agents else None
    
    def _round_robin(self, agents: List[Agent], task_type: str, task_priority: int) -> Agent:
        """Round robin selection"""
        key = f"{task_type}_{task_priority}"
        if key not in self.round_robin_counters:
            self.round_robin_counters[key] = 0
        
        selected_agent = agents[self.round_robin_counters[key] % len(agents)]
        self.round_robin_counters[key] += 1
        
        return selected_agent
    
    def _least_connections(self, agents: List[Agent], task_type: str, task_priority: int) -> Agent:
        """Select agent with least active tasks"""
        return min(agents, key=lambda a: len(a.metrics.get("active_tasks", [])))
    
    def _weighted_round_robin(self, agents: List[Agent], task_type: str, task_priority: int) -> Agent:
        """Weighted round robin based on agent capacity"""
        weights = []
        for agent in agents:
            cpu_usage = agent.metrics.get("cpu_usage", 0.5)
            memory_usage = agent.metrics.get("memory_usage", 0.5)
            # Higher weight for agents with lower resource usage
            weight = max(0.1, 1.0 - (cpu_usage + memory_usage) / 2)
            weights.append(weight)
        
        # Weighted random selection
        import random
        return random.choices(agents, weights=weights)[0]
    
    def _resource_aware(self, agents: List[Agent], task_type: str, task_priority: int) -> Agent:
        """Select agent based on resource availability and task requirements"""
        scored_agents = []
        
        for agent in agents:
            score = self._calculate_agent_score(agent, task_type, task_priority)
            scored_agents.append((agent, score))
        
        # Sort by score (higher is better)
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0]
    
    def _capability_based(self, agents: List[Agent], task_type: str, task_priority: int) -> Agent:
        """Select agent based on capability match and performance"""
        best_agent = None
        best_score = -1
        
        for agent in agents:
            if task_type not in agent.capabilities:
                continue
                
            # Calculate capability score
            capability_score = agent.metrics.get("task_success_rate", 0.5)
            performance_score = 1.0 - agent.metrics.get("avg_response_time", 0.5)
            load_score = 1.0 - agent.metrics.get("cpu_usage", 0.5)
            
            total_score = (capability_score * 0.4 + performance_score * 0.3 + load_score * 0.3)
            
            if total_score > best_score:
                best_score = total_score
                best_agent = agent
        
        return best_agent or agents[0]
    
    def _calculate_agent_score(self, agent: Agent, task_type: str, task_priority: int) -> float:
        """Calculate agent score for task assignment"""
        try:
            # Base score factors
            cpu_score = 1.0 - agent.metrics.get("cpu_usage", 0.5)
            memory_score = 1.0 - agent.metrics.get("memory_usage", 0.5)
            load_score = 1.0 - (len(agent.metrics.get("active_tasks", [])) / 10.0)
            
            # Performance factors
            success_rate = agent.metrics.get("task_success_rate", 0.5)
            avg_response_time = agent.metrics.get("avg_response_time", 1.0)
            response_score = max(0.1, 1.0 - (avg_response_time / 60.0))  # Normalize to 60 seconds
            
            # Capability match
            capability_score = 1.0 if task_type in agent.capabilities else 0.0
            
            # Priority adjustment
            priority_weight = task_priority / 10.0
            
            # Calculate weighted score
            score = (
                cpu_score * 0.2 +
                memory_score * 0.2 +
                load_score * 0.15 +
                success_rate * 0.25 +
                response_score * 0.1 +
                capability_score * 0.1
            ) * priority_weight
            
            return score
            
        except Exception as e:
            logger.error(f"Agent score calculation failed: {e}")
            return 0.0

class AutoScaler:
    """Auto-scaling manager for agent swarm"""
    
    def __init__(self, swarm_manager):
        self.swarm_manager = swarm_manager
        self.scaling_policies = {
            AgentType.RECONNAISSANCE: {"min": 1, "max": 10, "target_cpu": 0.7, "target_memory": 0.8},
            AgentType.SCANNER: {"min": 1, "max": 8, "target_cpu": 0.6, "target_memory": 0.7},
            AgentType.EXPLOITER: {"min": 1, "max": 5, "target_cpu": 0.5, "target_memory": 0.6},
            AgentType.OSINT: {"min": 1, "max": 6, "target_cpu": 0.6, "target_memory": 0.7},
            AgentType.STEALTH: {"min": 1, "max": 4, "target_cpu": 0.5, "target_memory": 0.6}
        }
        self.scaling_history = {}
        self.cooldown_period = 300  # 5 minutes between scaling actions
        
    async def evaluate_scaling(self):
        """Evaluate if scaling is needed"""
        try:
            for agent_type in AgentType:
                scaling_decision = await self._evaluate_agent_type_scaling(agent_type)
                
                if scaling_decision["action"] != "none":
                    await self._execute_scaling_action(agent_type, scaling_decision)
                    
        except Exception as e:
            logger.error(f"Scaling evaluation failed: {e}")
    
    async def _evaluate_agent_type_scaling(self, agent_type: AgentType) -> Dict[str, Any]:
        """Evaluate scaling for specific agent type"""
        try:
            agents = [a for a in self.swarm_manager.agents.values() if a.type == agent_type and a.status == "running"]
            policy = self.scaling_policies[agent_type]
            
            if not agents:
                return {"action": "scale_up", "target_count": policy["min"], "reason": "no_agents"}
            
            # Calculate metrics
            avg_cpu = statistics.mean([a.metrics.get("cpu_usage", 0) for a in agents])
            avg_memory = statistics.mean([a.metrics.get("memory_usage", 0) for a in agents])
            avg_load = statistics.mean([len(a.metrics.get("active_tasks", [])) for a in agents])
            
            # Pending tasks for this agent type
            pending_tasks = len([
                t for t in self.swarm_manager.tasks.values() 
                if t.status == TaskStatus.PENDING and t.task_type in self.swarm_manager.agent_capabilities[agent_type]
            ])
            
            current_count = len(agents)
            decision = {"action": "none", "current_count": current_count, "metrics": {
                "avg_cpu": avg_cpu,
                "avg_memory": avg_memory,
                "avg_load": avg_load,
                "pending_tasks": pending_tasks
            }}
            
            # Check if cooldown period has passed
            last_scaling = self.scaling_history.get(agent_type, datetime.min)
            if (datetime.utcnow() - last_scaling).total_seconds() < self.cooldown_period:
                decision["reason"] = "cooldown_active"
                return decision
            
            # Scale up conditions
            if (avg_cpu > policy["target_cpu"] or 
                avg_memory > policy["target_memory"] or 
                avg_load > 5 or 
                pending_tasks > current_count * 2):
                
                if current_count < policy["max"]:
                    target_count = min(policy["max"], current_count + max(1, pending_tasks // 3))
                    decision.update({
                        "action": "scale_up",
                        "target_count": target_count,
                        "reason": "high_load"
                    })
            
            # Scale down conditions
            elif (avg_cpu < policy["target_cpu"] * 0.3 and 
                  avg_memory < policy["target_memory"] * 0.3 and 
                  avg_load < 2 and 
                  pending_tasks == 0):
                
                if current_count > policy["min"]:
                    target_count = max(policy["min"], current_count - 1)
                    decision.update({
                        "action": "scale_down",
                        "target_count": target_count,
                        "reason": "low_load"
                    })
            
            return decision
            
        except Exception as e:
            logger.error(f"Agent type scaling evaluation failed: {e}")
            return {"action": "none", "error": str(e)}
    
    async def _execute_scaling_action(self, agent_type: AgentType, decision: Dict[str, Any]):
        """Execute scaling action"""
        try:
            action = decision["action"]
            target_count = decision["target_count"]
            current_count = decision["current_count"]
            
            if action == "scale_up":
                for _ in range(target_count - current_count):
                    await self.swarm_manager.deploy_agent(agent_type, {
                        "auto_scaled": True,
                        "reason": decision["reason"]
                    })
                    
            elif action == "scale_down":
                agents_to_remove = [
                    a for a in self.swarm_manager.agents.values()
                    if a.type == agent_type and a.status == "running" and a.current_task is None
                ][:current_count - target_count]
                
                for agent in agents_to_remove:
                    await self.swarm_manager.shutdown_agent(agent.id)
            
            # Update scaling history
            self.scaling_history[agent_type] = datetime.utcnow()
            
            logger.info(f"Executed {action} for {agent_type.value}: {current_count} -> {target_count}")
            
        except Exception as e:
            logger.error(f"Scaling action execution failed: {e}")

class MetricsCollector:
    """Metrics collection and analysis for swarm performance"""
    
    def __init__(self, swarm_manager):
        self.swarm_manager = swarm_manager
        self.metrics_history = []
        self.collection_interval = 60  # seconds
        
    async def start_collection(self):
        """Start metrics collection loop"""
        while True:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current swarm metrics"""
        try:
            timestamp = datetime.utcnow()
            agents = list(self.swarm_manager.agents.values())
            tasks = list(self.swarm_manager.tasks.values())
            
            # Agent metrics
            total_agents = len(agents)
            active_agents = len([a for a in agents if a.status == "running"])
            idle_agents = len([a for a in agents if a.status == "running" and a.current_task is None])
            
            # Task metrics
            total_tasks = len(tasks)
            pending_tasks = len([t for t in tasks if t.status == TaskStatus.PENDING])
            running_tasks = len([t for t in tasks if t.status == TaskStatus.RUNNING])
            completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
            failed_tasks = len([t for t in tasks if t.status == TaskStatus.FAILED])
            
            # Performance metrics
            if active_agents > 0:
                avg_cpu = statistics.mean([a.metrics.get("cpu_usage", 0) for a in agents if a.status == "running"])
                avg_memory = statistics.mean([a.metrics.get("memory_usage", 0) for a in agents if a.status == "running"])
                avg_response_time = statistics.mean([a.metrics.get("avg_response_time", 0) for a in agents if a.status == "running"])
                agent_utilization = (active_agents - idle_agents) / active_agents if active_agents > 0 else 0
            else:
                avg_cpu = avg_memory = avg_response_time = agent_utilization = 0
            
            # Task completion rate
            recent_completed = len([t for t in tasks if t.status == TaskStatus.COMPLETED and 
                                  t.completed_at and (timestamp - t.completed_at).total_seconds() < 60])
            tasks_per_second = recent_completed / 60.0
            
            # Error rate
            recent_failed = len([t for t in tasks if t.status == TaskStatus.FAILED and 
                               t.completed_at and (timestamp - t.completed_at).total_seconds() < 60])
            error_rate = recent_failed / max(1, recent_completed + recent_failed)
            
            return {
                "timestamp": timestamp,
                "agents": {
                    "total": total_agents,
                    "active": active_agents,
                    "idle": idle_agents,
                    "utilization": agent_utilization,
                    "avg_cpu": avg_cpu,
                    "avg_memory": avg_memory,
                    "avg_response_time": avg_response_time
                },
                "tasks": {
                    "total": total_tasks,
                    "pending": pending_tasks,
                    "running": running_tasks,
                    "completed": completed_tasks,
                    "failed": failed_tasks,
                    "tasks_per_second": tasks_per_second,
                    "error_rate": error_rate
                },
                "performance": {
                    "tasks_per_second": tasks_per_second,
                    "average_response_time": avg_response_time,
                    "error_rate": error_rate,
                    "agent_utilization": agent_utilization
                }
            }
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {"timestamp": datetime.utcnow(), "error": str(e)}
    
    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified duration"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
            recent_metrics = [m for m in self.metrics_history if m.get("timestamp", datetime.min) > cutoff_time]
            
            if not recent_metrics:
                return {"error": "No metrics data available"}
            
            # Calculate aggregated metrics
            avg_tasks_per_second = statistics.mean([m["tasks"]["tasks_per_second"] for m in recent_metrics])
            avg_response_time = statistics.mean([m["agents"]["avg_response_time"] for m in recent_metrics])
            avg_error_rate = statistics.mean([m["tasks"]["error_rate"] for m in recent_metrics])
            avg_utilization = statistics.mean([m["agents"]["utilization"] for m in recent_metrics])
            
            return {
                "duration_minutes": duration_minutes,
                "data_points": len(recent_metrics),
                "summary": {
                    "avg_tasks_per_second": avg_tasks_per_second,
                    "avg_response_time": avg_response_time,
                    "avg_error_rate": avg_error_rate,
                    "avg_utilization": avg_utilization
                }
            }
            
        except Exception as e:
            logger.error(f"Metrics summary failed: {e}")
            return {"error": str(e)}

class HealthMonitor:
    """Health monitoring for swarm components"""
    
    def __init__(self, swarm_manager):
        self.swarm_manager = swarm_manager
        self.health_checks = {
            "agents": self._check_agent_health,
            "tasks": self._check_task_health,
            "communication": self._check_communication_health,
            "resources": self._check_resource_health
        }
        self.health_history = []
        
    async def start_monitoring(self):
        """Start health monitoring loop"""
        while True:
            try:
                health_report = await self._perform_health_check()
                self.health_history.append(health_report)
                
                # Keep only last 100 health reports
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                # Handle unhealthy conditions
                await self._handle_health_issues(health_report)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health monitoring failed: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            timestamp = datetime.utcnow()
            health_report = {
                "timestamp": timestamp,
                "overall_status": "healthy",
                "components": {}
            }
            
            # Run all health checks
            for component, check_func in self.health_checks.items():
                try:
                    component_health = await check_func()
                    health_report["components"][component] = component_health
                    
                    if component_health.get("status") != "healthy":
                        health_report["overall_status"] = "unhealthy"
                        
                except Exception as e:
                    health_report["components"][component] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_report["overall_status"] = "unhealthy"
            
            return health_report
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "timestamp": datetime.utcnow(),
                "overall_status": "error",
                "error": str(e)
            }
    
    async def _check_agent_health(self) -> Dict[str, Any]:
        """Check agent health"""
        try:
            agents = list(self.swarm_manager.agents.values())
            
            if not agents:
                return {"status": "warning", "message": "No agents deployed"}
            
            # Check agent status distribution
            status_counts = {}
            for agent in agents:
                status = agent.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Check for unhealthy agents
            unhealthy_agents = [a for a in agents if a.status in ["error", "timeout", "failed"]]
            unresponsive_agents = [a for a in agents if a.last_heartbeat and 
                                 (datetime.utcnow() - a.last_heartbeat).total_seconds() > 300]
            
            if len(unhealthy_agents) > len(agents) * 0.3:  # More than 30% unhealthy
                return {
                    "status": "unhealthy",
                    "message": f"{len(unhealthy_agents)} agents are unhealthy",
                    "details": {
                        "total_agents": len(agents),
                        "unhealthy_agents": len(unhealthy_agents),
                        "unresponsive_agents": len(unresponsive_agents),
                        "status_counts": status_counts
                    }
                }
            
            return {
                "status": "healthy",
                "message": f"{len(agents)} agents operational",
                "details": {
                    "total_agents": len(agents),
                    "unhealthy_agents": len(unhealthy_agents),
                    "unresponsive_agents": len(unresponsive_agents),
                    "status_counts": status_counts
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_task_health(self) -> Dict[str, Any]:
        """Check task health"""
        try:
            tasks = list(self.swarm_manager.tasks.values())
            
            if not tasks:
                return {"status": "healthy", "message": "No tasks in queue"}
            
            # Check task status distribution
            status_counts = {}
            for task in tasks:
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Check for stuck tasks
            stuck_tasks = []
            for task in tasks:
                if task.status == TaskStatus.RUNNING and task.started_at:
                    runtime = (datetime.utcnow() - task.started_at).total_seconds()
                    if runtime > 3600:  # Task running for more than 1 hour
                        stuck_tasks.append(task)
            
            # Check error rate
            recent_tasks = [t for t in tasks if t.completed_at and 
                          (datetime.utcnow() - t.completed_at).total_seconds() < 3600]
            if recent_tasks:
                error_rate = len([t for t in recent_tasks if t.status == TaskStatus.FAILED]) / len(recent_tasks)
                if error_rate > 0.5:  # More than 50% error rate
                    return {
                        "status": "unhealthy",
                        "message": f"High error rate: {error_rate:.2%}",
                        "details": {
                            "total_tasks": len(tasks),
                            "stuck_tasks": len(stuck_tasks),
                            "error_rate": error_rate,
                            "status_counts": status_counts
                        }
                    }
            
            return {
                "status": "healthy",
                "message": f"{len(tasks)} tasks managed",
                "details": {
                    "total_tasks": len(tasks),
                    "stuck_tasks": len(stuck_tasks),
                    "status_counts": status_counts
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_communication_health(self) -> Dict[str, Any]:
        """Check communication health"""
        try:
            # Check ZMQ socket status
            if not self.swarm_manager.command_socket:
                return {
                    "status": "unhealthy",
                    "message": "Command socket not initialized"
                }
            
            # Check for recent communication
            active_agents = [a for a in self.swarm_manager.agents.values() 
                           if a.status == "running" and a.last_heartbeat]
            
            if active_agents:
                recent_communication = [
                    a for a in active_agents 
                    if (datetime.utcnow() - a.last_heartbeat).total_seconds() < 120
                ]
                
                communication_health = len(recent_communication) / len(active_agents)
                
                if communication_health < 0.8:  # Less than 80% agents communicating
                    return {
                        "status": "unhealthy",
                        "message": f"Poor communication: {communication_health:.2%}",
                        "details": {
                            "active_agents": len(active_agents),
                            "communicating_agents": len(recent_communication),
                            "communication_health": communication_health
                        }
                    }
            
            return {
                "status": "healthy",
                "message": "Communication channels operational",
                "details": {
                    "active_agents": len(active_agents),
                    "communicating_agents": len(recent_communication) if active_agents else 0
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_resource_health(self) -> Dict[str, Any]:
        """Check resource health"""
        try:
            agents = [a for a in self.swarm_manager.agents.values() if a.status == "running"]
            
            if not agents:
                return {"status": "healthy", "message": "No agents using resources"}
            
            # Check resource usage
            cpu_usage = [a.metrics.get("cpu_usage", 0) for a in agents]
            memory_usage = [a.metrics.get("memory_usage", 0) for a in agents]
            
            avg_cpu = statistics.mean(cpu_usage)
            avg_memory = statistics.mean(memory_usage)
            max_cpu = max(cpu_usage)
            max_memory = max(memory_usage)
            
            # Check for resource exhaustion
            if avg_cpu > 0.9 or avg_memory > 0.9:
                return {
                    "status": "unhealthy",
                    "message": f"High resource usage: CPU {avg_cpu:.2%}, Memory {avg_memory:.2%}",
                    "details": {
                        "avg_cpu": avg_cpu,
                        "avg_memory": avg_memory,
                        "max_cpu": max_cpu,
                        "max_memory": max_memory
                    }
                }
            
            return {
                "status": "healthy",
                "message": f"Resource usage normal: CPU {avg_cpu:.2%}, Memory {avg_memory:.2%}",
                "details": {
                    "avg_cpu": avg_cpu,
                    "avg_memory": avg_memory,
                    "max_cpu": max_cpu,
                    "max_memory": max_memory
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _handle_health_issues(self, health_report: Dict[str, Any]):
        """Handle identified health issues"""
        try:
            if health_report.get("overall_status") == "unhealthy":
                logger.warning(f"Swarm health issue detected: {health_report}")
                
                # Trigger auto-scaling if needed
                await self.swarm_manager.auto_scaler.evaluate_scaling()
                
                # Handle specific component issues
                components = health_report.get("components", {})
                
                if components.get("agents", {}).get("status") == "unhealthy":
                    await self._handle_agent_health_issues(components["agents"])
                
                if components.get("tasks", {}).get("status") == "unhealthy":
                    await self._handle_task_health_issues(components["tasks"])
                    
        except Exception as e:
            logger.error(f"Health issue handling failed: {e}")
    
    async def _handle_agent_health_issues(self, agent_health: Dict[str, Any]):
        """Handle agent health issues"""
        try:
            # Remove unhealthy agents
            unhealthy_agents = [
                a for a in self.swarm_manager.agents.values() 
                if a.status in ["error", "timeout", "failed"]
            ]
            
            for agent in unhealthy_agents:
                logger.warning(f"Removing unhealthy agent: {agent.id}")
                await self.swarm_manager.shutdown_agent(agent.id)
                
        except Exception as e:
            logger.error(f"Agent health issue handling failed: {e}")
    
    async def _handle_task_health_issues(self, task_health: Dict[str, Any]):
        """Handle task health issues"""
        try:
            # Cancel stuck tasks
            stuck_tasks = [
                t for t in self.swarm_manager.tasks.values()
                if t.status == TaskStatus.RUNNING and t.started_at and
                (datetime.utcnow() - t.started_at).total_seconds() > 3600
            ]
            
            for task in stuck_tasks:
                logger.warning(f"Cancelling stuck task: {task.id}")
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.utcnow()
                task.error = "Task cancelled due to timeout"
                
                # Free up agent
                if task.agent_id and task.agent_id in self.swarm_manager.agents:
                    self.swarm_manager.agents[task.agent_id].current_task = None
                    
        except Exception as e:
            logger.error(f"Task health issue handling failed: {e}")

class SwarmManager:
    """Enhanced swarm manager with auto-scaling and load balancing"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.agent_capabilities = {
            AgentType.RECONNAISSANCE: [
                "dns_enumeration",
                "port_scanning",
                "service_detection",
                "subdomain_discovery",
                "whois_lookup",
                "banner_grabbing",
                "zone_transfer",
                "reverse_dns"
            ],
            AgentType.SCANNER: [
                "vulnerability_scanning",
                "web_scanning",
                "network_scanning",
                "ssl_analysis",
                "configuration_analysis",
                "cve_scanning",
                "web_crawler",
                "directory_bruteforce",
                "injection_testing",
                "authentication_testing"
            ],
            AgentType.EXPLOITER: [
                "exploit_execution",
                "payload_generation",
                "privilege_escalation",
                "persistence_establishment",
                "lateral_movement",
                "backdoor_deployment",
                "reverse_shell",
                "buffer_overflow",
                "sql_injection_exploit",
                "web_shell_upload",
                "service_exploitation"
            ],
            AgentType.OSINT: [
                "social_media_intelligence",
                "threat_intelligence",
                "dark_web_monitoring",
                "breach_data_analysis",
                "reputation_analysis",
                "email_intelligence",
                "domain_intelligence",
                "person_intelligence",
                "company_intelligence",
                "ip_intelligence",
                "certificate_intelligence",
                "paste_monitoring"
            ],
            AgentType.STEALTH: [
                "traffic_obfuscation",
                "proxy_management",
                "evasion_techniques",
                "anti_detection",
                "behavior_mimicry",
                "tor_routing",
                "vpn_management",
                "packet_crafting",
                "timing_manipulation",
                "fingerprint_spoofing",
                "decoy_traffic",
                "session_management"
            ]
        }
        
        self.zmq_context = zmq.asyncio.Context()
        self.command_socket = None
        self.heartbeat_interval = 30  # seconds
        self.agent_timeout = 180  # seconds
        
        # Enhanced components
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(self)
        self.metrics_collector = MetricsCollector(self)
        self.health_monitor = HealthMonitor(self)
        
        # Performance tracking
        self.performance_metrics = {
            "tasks_per_second": 0,
            "average_response_time": 0,
            "error_rate": 0,
            "agent_utilization": 0
        }
        
    async def initialize(self):
        """Initialize enhanced swarm manager"""
        try:
            # Setup ZMQ socket for agent commands
            self.command_socket = self.zmq_context.socket(zmq.PUSH)
            self.command_socket.bind(f"tcp://*:{config.network.zmq_port + 1}")
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._auto_scaling_loop())
            asyncio.create_task(self.metrics_collector.start_collection())
            asyncio.create_task(self.health_monitor.start_monitoring())
            
            logger.info("Enhanced swarm manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize swarm manager: {e}")
            raise
    
    async def cleanup(self):
        """Clean up swarm manager"""
        try:
            if self.command_socket:
                self.command_socket.close()
            
            # Shutdown all agents
            for agent_id in list(self.agents.keys()):
                await self.shutdown_agent(agent_id)
                
            logger.info("Swarm manager cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during swarm manager cleanup: {e}")
    
    async def deploy_agent(self, agent_type: AgentType, configuration: Dict[str, Any]) -> str:
        """Deploy a new swarm agent"""
        try:
            agent_id = str(uuid.uuid4())
            
            # Create agent configuration
            agent_config = {
                "agent_id": agent_id,
                "agent_type": agent_type.value,
                "capabilities": self.agent_capabilities[agent_type],
                "configuration": configuration,
                "zmq_coordinator": f"tcp://coordinator:{config.network.zmq_port}",
                "zmq_commands": f"tcp://coordinator:{config.network.zmq_port + 1}",
                "encryption_key": security_manager.encrypt_data(config.security.encryption_key)
            }
            
            # Create agent record
            agent = Agent(
                id=agent_id,
                type=agent_type,
                status="deploying",
                created_at=datetime.utcnow(),
                capabilities=self.agent_capabilities[agent_type],
                configuration=agent_config
            )
            
            self.agents[agent_id] = agent
            
            # Deploy agent container (in production, use Kubernetes/Cloud Run)
            await self._deploy_agent_container(agent_config)
            
            logger.info(f"Agent {agent_id} deployed successfully")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to deploy agent: {e}")
            raise
    
    async def _deploy_agent_container(self, agent_config: Dict[str, Any]):
        """Deploy agent container (placeholder for actual deployment)"""
        # In production, this would deploy to Kubernetes or Cloud Run
        # For now, we'll simulate deployment
        await asyncio.sleep(2)  # Simulate deployment time
        
        agent_id = agent_config["agent_id"]
        if agent_id in self.agents:
            self.agents[agent_id].status = "running"
            self.agents[agent_id].last_heartbeat = datetime.utcnow()
    
    async def shutdown_agent(self, agent_id: str) -> bool:
        """Shutdown specific agent"""
        try:
            if agent_id not in self.agents:
                return False
            
            # Send shutdown command
            await self._send_agent_command(agent_id, "shutdown")
            
            # Remove from active agents
            del self.agents[agent_id]
            
            logger.info(f"Agent {agent_id} shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown agent {agent_id}: {e}")
            return False
    
    async def assign_task(self, task: Task, agent_id: Optional[str] = None) -> bool:
        """Assign task to agent using load balancer or specific agent"""
        try:
            if agent_id:
                # Assign to specific agent
                if agent_id not in self.agents:
                    logger.error(f"Agent {agent_id} not found")
                    return False
                
                agent = self.agents[agent_id]
                
                # Check if agent has required capability
                if task.task_type not in agent.capabilities:
                    logger.error(f"Agent {agent_id} doesn't have capability {task.task_type}")
                    return False
                
                # Check if agent is available
                if agent.status != "running" or agent.current_task is not None:
                    logger.error(f"Agent {agent_id} is not available")
                    return False
                
                selected_agent = agent
                
            else:
                # Use load balancer to select best agent
                available_agents = await self.get_available_agents(task.task_type)
                
                if not available_agents:
                    logger.error(f"No available agents for task type {task.task_type}")
                    return False
                
                # Get task priority from parameters
                task_priority = task.parameters.get("priority", 5)
                
                # Select agent using load balancer
                selected_agent = self.load_balancer.select_agent(available_agents, task.task_type, task_priority)
                
                if not selected_agent:
                    logger.error(f"Load balancer failed to select agent for task {task.id}")
                    return False
            
            # Update agent and task status
            selected_agent.current_task = task.id
            task.agent_id = selected_agent.id
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            self.tasks[task.id] = task
            
            # Update agent metrics
            if "active_tasks" not in selected_agent.metrics:
                selected_agent.metrics["active_tasks"] = []
            selected_agent.metrics["active_tasks"].append(task.id)
            
            # Send task to agent
            task_command = {
                "command": "execute_task",
                "task_id": task.id,
                "task_type": task.task_type,
                "target": task.target,
                "parameters": task.parameters
            }
            
            await self._send_agent_command(selected_agent.id, task_command)
            
            logger.info(f"Task {task.id} assigned to agent {selected_agent.id} via {'direct' if agent_id else 'load_balancer'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign task {task.id}: {e}")
            return False
    
    async def _send_agent_command(self, agent_id: str, command: Any):
        """Send command to specific agent"""
        try:
            message = {
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "command": command
            }
            
            # Encrypt message
            encrypted_message = security_manager.encrypt_data(json.dumps(message))
            
            await self.command_socket.send_string(encrypted_message)
            
        except Exception as e:
            logger.error(f"Failed to send command to agent {agent_id}: {e}")
    
    async def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_heartbeat = datetime.utcnow()
    
    async def update_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None):
        """Update task status with enhanced metrics tracking"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            old_status = task.status
            task.status = TaskStatus(status)
            
            if status == "completed":
                task.completed_at = datetime.utcnow()
                task.result = result
                
                # Calculate task execution time
                if task.started_at:
                    execution_time = (task.completed_at - task.started_at).total_seconds()
                    task.execution_time = execution_time
                
                # Update agent metrics
                if task.agent_id and task.agent_id in self.agents:
                    agent = self.agents[task.agent_id]
                    agent.current_task = None
                    
                    # Update agent performance metrics
                    if "tasks_completed" not in agent.metrics:
                        agent.metrics["tasks_completed"] = 0
                    agent.metrics["tasks_completed"] += 1
                    
                    # Update response time metrics
                    if task.started_at:
                        response_times = agent.metrics.get("response_times", [])
                        response_times.append(execution_time)
                        # Keep only last 10 response times
                        agent.metrics["response_times"] = response_times[-10:]
                        agent.metrics["avg_response_time"] = sum(response_times) / len(response_times)
                    
                    # Update success rate
                    completed = agent.metrics.get("tasks_completed", 0)
                    failed = agent.metrics.get("tasks_failed", 0)
                    total = completed + failed
                    agent.metrics["task_success_rate"] = completed / total if total > 0 else 1.0
                    
                    # Remove from active tasks
                    if "active_tasks" in agent.metrics and task_id in agent.metrics["active_tasks"]:
                        agent.metrics["active_tasks"].remove(task_id)
            
            elif status == "failed":
                task.completed_at = datetime.utcnow()
                task.error = result.get("error") if result else "Unknown error"
                
                # Update agent metrics
                if task.agent_id and task.agent_id in self.agents:
                    agent = self.agents[task.agent_id]
                    agent.current_task = None
                    
                    # Update agent failure metrics
                    if "tasks_failed" not in agent.metrics:
                        agent.metrics["tasks_failed"] = 0
                    agent.metrics["tasks_failed"] += 1
                    
                    # Update success rate
                    completed = agent.metrics.get("tasks_completed", 0)
                    failed = agent.metrics.get("tasks_failed", 0)
                    total = completed + failed
                    agent.metrics["task_success_rate"] = completed / total if total > 0 else 1.0
                    
                    # Remove from active tasks
                    if "active_tasks" in agent.metrics and task_id in agent.metrics["active_tasks"]:
                        agent.metrics["active_tasks"].remove(task_id)
            
            # Log status change
            logger.info(f"Task {task_id} status changed from {old_status} to {status}")
            
            # Update performance metrics
            await self._update_performance_metrics()
    
    async def get_available_agents(self, capability: str = None) -> List[Agent]:
        """Get available agents with optional capability filter"""
        available = []
        
        for agent in self.agents.values():
            if agent.status == "running" and agent.current_task is None:
                if capability is None or capability in agent.capabilities:
                    available.append(agent)
        
        return available
    
    async def get_agents(self) -> List[Agent]:
        """Get all agents"""
        return list(self.agents.values())
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get specific agent"""
        return self.agents.get(agent_id)
    
    async def get_agent_count(self) -> int:
        """Get total agent count"""
        return len(self.agents)
    
    async def get_active_agent_count(self) -> int:
        """Get active agent count"""
        return len([a for a in self.agents.values() if a.status == "running"])
    
    async def scale_agents(self, agent_type: AgentType, target_count: int):
        """Scale agents of specific type to target count"""
        try:
            current_count = len([
                a for a in self.agents.values() 
                if a.type == agent_type and a.status == "running"
            ])
            
            if current_count < target_count:
                # Scale up
                for _ in range(target_count - current_count):
                    await self.deploy_agent(agent_type, {})
            
            elif current_count > target_count:
                # Scale down
                agents_to_remove = [
                    a for a in self.agents.values()
                    if a.type == agent_type and a.status == "running" and a.current_task is None
                ][:current_count - target_count]
                
                for agent in agents_to_remove:
                    await self.shutdown_agent(agent.id)
            
            logger.info(f"Scaled {agent_type} agents to {target_count}")
            
        except Exception as e:
            logger.error(f"Failed to scale agents: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while True:
            try:
                now = datetime.utcnow()
                timeout_threshold = now - timedelta(seconds=self.agent_timeout)
                
                # Check for timed out agents
                timed_out_agents = [
                    agent_id for agent_id, agent in self.agents.items()
                    if agent.last_heartbeat and agent.last_heartbeat < timeout_threshold
                ]
                
                for agent_id in timed_out_agents:
                    logger.warning(f"Agent {agent_id} timed out")
                    self.agents[agent_id].status = "timeout"
                    
                    # Reassign tasks if any
                    if self.agents[agent_id].current_task:
                        await self._reassign_task(self.agents[agent_id].current_task)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _reassign_task(self, task_id: str):
        """Reassign task to another agent"""
        try:
            if task_id not in self.tasks:
                return
            
            task = self.tasks[task_id]
            
            # Find available agent with required capability
            available_agents = await self.get_available_agents(task.task_type)
            
            if available_agents:
                # Reset task status
                task.status = TaskStatus.PENDING
                task.agent_id = None
                task.started_at = None
                
                # Assign to new agent
                await self.assign_task(available_agents[0].id, task)
                logger.info(f"Task {task_id} reassigned to agent {available_agents[0].id}")
            else:
                # No available agents, mark as failed
                task.status = TaskStatus.FAILED
                task.error = "No available agents"
                task.completed_at = datetime.utcnow()
                logger.error(f"Failed to reassign task {task_id}: No available agents")
                
        except Exception as e:
            logger.error(f"Error reassigning task {task_id}: {e}")
    
    async def _auto_scaling_loop(self):
        """Auto-scaling background loop"""
        while True:
            try:
                await self.auto_scaler.evaluate_scaling()
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(120)
    
    async def _update_performance_metrics(self):
        """Update swarm performance metrics"""
        try:
            agents = list(self.agents.values())
            tasks = list(self.tasks.values())
            
            # Calculate tasks per second
            recent_completed = len([
                t for t in tasks 
                if t.status == TaskStatus.COMPLETED and t.completed_at and 
                (datetime.utcnow() - t.completed_at).total_seconds() < 60
            ])
            self.performance_metrics["tasks_per_second"] = recent_completed / 60.0
            
            # Calculate average response time
            if agents:
                response_times = [a.metrics.get("avg_response_time", 0) for a in agents if a.status == "running"]
                self.performance_metrics["average_response_time"] = sum(response_times) / len(response_times) if response_times else 0
            
            # Calculate error rate
            recent_failed = len([
                t for t in tasks 
                if t.status == TaskStatus.FAILED and t.completed_at and 
                (datetime.utcnow() - t.completed_at).total_seconds() < 60
            ])
            total_recent = recent_completed + recent_failed
            self.performance_metrics["error_rate"] = recent_failed / total_recent if total_recent > 0 else 0
            
            # Calculate agent utilization
            active_agents = len([a for a in agents if a.status == "running"])
            busy_agents = len([a for a in agents if a.status == "running" and a.current_task])
            self.performance_metrics["agent_utilization"] = busy_agents / active_agents if active_agents > 0 else 0
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def set_load_balancer_algorithm(self, algorithm: str) -> bool:
        """Set load balancer algorithm"""
        try:
            if algorithm in self.load_balancer.algorithms:
                self.load_balancer.current_algorithm = algorithm
                logger.info(f"Load balancer algorithm changed to {algorithm}")
                return True
            else:
                logger.error(f"Unknown load balancer algorithm: {algorithm}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set load balancer algorithm: {e}")
            return False
    
    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        try:
            return {
                "current_algorithm": self.load_balancer.current_algorithm,
                "available_algorithms": list(self.load_balancer.algorithms.keys()),
                "round_robin_counters": self.load_balancer.round_robin_counters.copy()
            }
            
        except Exception as e:
            logger.error(f"Failed to get load balancer stats: {e}")
            return {"error": str(e)}
    
    async def get_scaling_policies(self) -> Dict[str, Any]:
        """Get auto-scaling policies"""
        try:
            return {
                "policies": {
                    agent_type.value: policy 
                    for agent_type, policy in self.auto_scaler.scaling_policies.items()
                },
                "scaling_history": {
                    agent_type.value: timestamp.isoformat() if timestamp != datetime.min else None
                    for agent_type, timestamp in self.auto_scaler.scaling_history.items()
                },
                "cooldown_period": self.auto_scaler.cooldown_period
            }
            
        except Exception as e:
            logger.error(f"Failed to get scaling policies: {e}")
            return {"error": str(e)}
    
    async def update_scaling_policy(self, agent_type: str, policy_updates: Dict[str, Any]) -> bool:
        """Update scaling policy for agent type"""
        try:
            agent_type_enum = AgentType(agent_type)
            
            if agent_type_enum in self.auto_scaler.scaling_policies:
                current_policy = self.auto_scaler.scaling_policies[agent_type_enum]
                
                # Update policy with new values
                for key, value in policy_updates.items():
                    if key in current_policy:
                        current_policy[key] = value
                
                logger.info(f"Updated scaling policy for {agent_type}: {policy_updates}")
                return True
            else:
                logger.error(f"Unknown agent type: {agent_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update scaling policy: {e}")
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        try:
            if self.health_monitor.health_history:
                return self.health_monitor.health_history[-1]
            else:
                return {"status": "unknown", "message": "No health data available"}
                
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            return {
                "current_metrics": self.performance_metrics.copy(),
                "historical_summary": self.metrics_collector.get_metrics_summary()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    async def force_scaling_evaluation(self) -> Dict[str, Any]:
        """Force immediate scaling evaluation"""
        try:
            scaling_results = {}
            
            for agent_type in AgentType:
                decision = await self.auto_scaler._evaluate_agent_type_scaling(agent_type)
                scaling_results[agent_type.value] = decision
                
                if decision["action"] != "none":
                    await self.auto_scaler._execute_scaling_action(agent_type, decision)
            
            return {
                "forced_evaluation": True,
                "scaling_results": scaling_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to force scaling evaluation: {e}")
            return {"error": str(e)}
    
    async def get_agent_performance_ranking(self) -> List[Dict[str, Any]]:
        """Get agents ranked by performance"""
        try:
            agents = [a for a in self.agents.values() if a.status == "running"]
            
            ranked_agents = []
            for agent in agents:
                score = self.load_balancer._calculate_agent_score(agent, "general", 5)
                ranked_agents.append({
                    "agent_id": agent.id,
                    "agent_type": agent.type.value,
                    "performance_score": score,
                    "metrics": agent.metrics
                })
            
            # Sort by performance score (descending)
            ranked_agents.sort(key=lambda x: x["performance_score"], reverse=True)
            
            return ranked_agents
            
        except Exception as e:
            logger.error(f"Failed to get agent performance ranking: {e}")
            return []
    
    def is_healthy(self) -> bool:
        """Check if swarm manager is healthy"""
        try:
            # Check if we have active agents
            active_count = len([a for a in self.agents.values() if a.status == "running"])
            
            # Check recent health report
            if self.health_monitor.health_history:
                latest_health = self.health_monitor.health_history[-1]
                return (active_count > 0 or len(self.agents) == 0) and latest_health.get("overall_status") != "error"
            
            return active_count > 0 or len(self.agents) == 0
            
        except Exception:
            return False
    
    async def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get swarm metrics"""
        try:
            total_agents = len(self.agents)
            active_agents = len([a for a in self.agents.values() if a.status == "running"])
            busy_agents = len([a for a in self.agents.values() if a.current_task is not None])
            
            total_tasks = len(self.tasks)
            pending_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
            running_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
            completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
            failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
            
            return {
                "agents": {
                    "total": total_agents,
                    "active": active_agents,
                    "busy": busy_agents,
                    "idle": active_agents - busy_agents,
                    "by_type": {
                        agent_type.value: len([
                            a for a in self.agents.values() 
                            if a.type == agent_type and a.status == "running"
                        ])
                        for agent_type in AgentType
                    }
                },
                "tasks": {
                    "total": total_tasks,
                    "pending": pending_tasks,
                    "running": running_tasks,
                    "completed": completed_tasks,
                    "failed": failed_tasks
                },
                "health": {
                    "overall": self.is_healthy(),
                    "agent_timeout_threshold": self.agent_timeout,
                    "last_heartbeat_check": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting swarm metrics: {e}")
            return {}