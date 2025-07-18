"""
Distributed Agent Registry for Aetherveil Sentinel
Implements service discovery, capability registration, and agent lifecycle management
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import aioredis
from contextlib import asynccontextmanager
import hashlib

from config.config import config
from coordinator.models import Agent, AgentType, TaskStatus
from coordinator.distributed_coordinator import DistributedCoordinator

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent states in the registry"""
    REGISTERING = "registering"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAINING = "draining"
    TERMINATED = "terminated"
    UNHEALTHY = "unhealthy"

class ServiceType(Enum):
    """Service types for discovery"""
    AGENT = "agent"
    COORDINATOR = "coordinator"
    MONITOR = "monitor"
    GATEWAY = "gateway"

@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    agent_type: AgentType
    capabilities: List[str]
    endpoint: str
    metadata: Dict[str, Any]
    state: AgentState
    registered_at: datetime
    last_heartbeat: datetime
    health_status: str
    load_metrics: Dict[str, float]
    tags: List[str]
    version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
            "state": self.state.value,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "health_status": self.health_status,
            "load_metrics": self.load_metrics,
            "tags": self.tags,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentRegistration':
        """Create from dictionary"""
        return cls(
            agent_id=data["agent_id"],
            agent_type=AgentType(data["agent_type"]),
            capabilities=data["capabilities"],
            endpoint=data["endpoint"],
            metadata=data["metadata"],
            state=AgentState(data["state"]),
            registered_at=datetime.fromisoformat(data["registered_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            health_status=data["health_status"],
            load_metrics=data["load_metrics"],
            tags=data["tags"],
            version=data["version"]
        )

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    service_id: str
    service_type: ServiceType
    name: str
    host: str
    port: int
    protocol: str
    metadata: Dict[str, Any]
    health_check: Optional[str]
    registered_at: datetime
    last_seen: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "service_id": self.service_id,
            "service_type": self.service_type.value,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "metadata": self.metadata,
            "health_check": self.health_check,
            "registered_at": self.registered_at.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceEndpoint':
        """Create from dictionary"""
        return cls(
            service_id=data["service_id"],
            service_type=ServiceType(data["service_type"]),
            name=data["name"],
            host=data["host"],
            port=data["port"],
            protocol=data["protocol"],
            metadata=data["metadata"],
            health_check=data.get("health_check"),
            registered_at=datetime.fromisoformat(data["registered_at"]),
            last_seen=datetime.fromisoformat(data["last_seen"])
        )

class HealthChecker:
    """Health checker for registered services"""
    
    def __init__(self, registry: 'AgentRegistry'):
        self.registry = registry
        self.check_interval = 30  # seconds
        self.unhealthy_threshold = 3  # failed checks
        self.running = False
        
    async def start(self):
        """Start health checking"""
        self.running = True
        asyncio.create_task(self._health_check_loop())
        logger.info("Health checker started")
    
    async def stop(self):
        """Stop health checking"""
        self.running = False
        logger.info("Health checker stopped")
    
    async def _health_check_loop(self):
        """Main health check loop"""
        while self.running:
            try:
                await self._check_all_agents()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_agents(self):
        """Check health of all agents"""
        try:
            agents = await self.registry.get_all_agents()
            
            for agent in agents:
                try:
                    await self._check_agent_health(agent)
                except Exception as e:
                    logger.error(f"Health check failed for agent {agent.agent_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to check all agents: {e}")
    
    async def _check_agent_health(self, agent: AgentRegistration):
        """Check health of specific agent"""
        try:
            # Check heartbeat timeout
            heartbeat_timeout = timedelta(seconds=120)  # 2 minutes
            if datetime.utcnow() - agent.last_heartbeat > heartbeat_timeout:
                await self._mark_agent_unhealthy(agent, "heartbeat_timeout")
                return
            
            # Check agent-specific health endpoint if available
            if agent.metadata.get("health_endpoint"):
                health_status = await self._check_health_endpoint(agent)
                if not health_status:
                    await self._mark_agent_unhealthy(agent, "health_check_failed")
                    return
            
            # Check resource thresholds
            if agent.load_metrics:
                cpu_usage = agent.load_metrics.get("cpu_usage", 0)
                memory_usage = agent.load_metrics.get("memory_usage", 0)
                
                if cpu_usage > 0.95 or memory_usage > 0.95:
                    await self._mark_agent_unhealthy(agent, "resource_exhaustion")
                    return
            
            # Mark as healthy if all checks pass
            if agent.state == AgentState.UNHEALTHY:
                await self.registry.update_agent_state(agent.agent_id, AgentState.ACTIVE)
                
        except Exception as e:
            logger.error(f"Health check failed for agent {agent.agent_id}: {e}")
    
    async def _check_health_endpoint(self, agent: AgentRegistration) -> bool:
        """Check agent health endpoint"""
        try:
            import aiohttp
            
            health_endpoint = agent.metadata.get("health_endpoint")
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_endpoint) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"Health endpoint check failed for {agent.agent_id}: {e}")
            return False
    
    async def _mark_agent_unhealthy(self, agent: AgentRegistration, reason: str):
        """Mark agent as unhealthy"""
        try:
            await self.registry.update_agent_state(agent.agent_id, AgentState.UNHEALTHY)
            await self.registry.update_agent_metadata(
                agent.agent_id, 
                {"unhealthy_reason": reason, "unhealthy_at": datetime.utcnow().isoformat()}
            )
            
            logger.warning(f"Agent {agent.agent_id} marked as unhealthy: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to mark agent {agent.agent_id} as unhealthy: {e}")

class LoadBalancer:
    """Advanced load balancer for agent selection"""
    
    def __init__(self, registry: 'AgentRegistry'):
        self.registry = registry
        self.algorithms = {
            "round_robin": self._round_robin,
            "least_connections": self._least_connections,
            "weighted_round_robin": self._weighted_round_robin,
            "resource_aware": self._resource_aware,
            "capability_based": self._capability_based,
            "location_aware": self._location_aware,
            "consistent_hash": self._consistent_hash
        }
        self.current_algorithm = "resource_aware"
        self.state = {}
        
    async def select_agent(self, capability: str, tags: List[str] = None, 
                          metadata: Dict[str, Any] = None) -> Optional[AgentRegistration]:
        """Select best agent for capability"""
        try:
            # Get available agents with capability
            agents = await self.registry.get_agents_by_capability(capability)
            
            # Filter by tags if specified
            if tags:
                agents = [a for a in agents if any(tag in a.tags for tag in tags)]
            
            # Filter by metadata if specified
            if metadata:
                agents = [a for a in agents if self._matches_metadata(a, metadata)]
            
            # Filter active agents only
            agents = [a for a in agents if a.state == AgentState.ACTIVE]
            
            if not agents:
                return None
            
            # Apply load balancing algorithm
            algorithm = self.algorithms.get(self.current_algorithm, self._round_robin)
            return await algorithm(agents, capability, tags, metadata)
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            return None
    
    def _matches_metadata(self, agent: AgentRegistration, metadata: Dict[str, Any]) -> bool:
        """Check if agent matches metadata criteria"""
        for key, value in metadata.items():
            if key not in agent.metadata or agent.metadata[key] != value:
                return False
        return True
    
    async def _round_robin(self, agents: List[AgentRegistration], capability: str, 
                          tags: List[str], metadata: Dict[str, Any]) -> AgentRegistration:
        """Round robin selection"""
        key = f"rr_{capability}"
        current_index = self.state.get(key, 0)
        selected_agent = agents[current_index % len(agents)]
        self.state[key] = (current_index + 1) % len(agents)
        return selected_agent
    
    async def _least_connections(self, agents: List[AgentRegistration], capability: str,
                                tags: List[str], metadata: Dict[str, Any]) -> AgentRegistration:
        """Least connections selection"""
        return min(agents, key=lambda a: len(a.load_metrics.get("active_tasks", [])))
    
    async def _weighted_round_robin(self, agents: List[AgentRegistration], capability: str,
                                   tags: List[str], metadata: Dict[str, Any]) -> AgentRegistration:
        """Weighted round robin based on capacity"""
        import random
        
        weights = []
        for agent in agents:
            cpu_usage = agent.load_metrics.get("cpu_usage", 0.5)
            memory_usage = agent.load_metrics.get("memory_usage", 0.5)
            weight = max(0.1, 1.0 - (cpu_usage + memory_usage) / 2)
            weights.append(weight)
        
        return random.choices(agents, weights=weights)[0]
    
    async def _resource_aware(self, agents: List[AgentRegistration], capability: str,
                             tags: List[str], metadata: Dict[str, Any]) -> AgentRegistration:
        """Resource-aware selection"""
        scored_agents = []
        
        for agent in agents:
            score = await self._calculate_agent_score(agent, capability)
            scored_agents.append((agent, score))
        
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0]
    
    async def _capability_based(self, agents: List[AgentRegistration], capability: str,
                               tags: List[str], metadata: Dict[str, Any]) -> AgentRegistration:
        """Capability-based selection"""
        # Prefer agents with more specific capabilities
        scored_agents = []
        
        for agent in agents:
            capability_score = self._calculate_capability_score(agent, capability)
            performance_score = agent.load_metrics.get("task_success_rate", 0.5)
            total_score = capability_score * 0.6 + performance_score * 0.4
            scored_agents.append((agent, total_score))
        
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0]
    
    async def _location_aware(self, agents: List[AgentRegistration], capability: str,
                             tags: List[str], metadata: Dict[str, Any]) -> AgentRegistration:
        """Location-aware selection"""
        # Prefer agents in the same region/zone
        preferred_location = metadata.get("preferred_location") if metadata else None
        
        if preferred_location:
            local_agents = [a for a in agents if a.metadata.get("location") == preferred_location]
            if local_agents:
                return await self._resource_aware(local_agents, capability, tags, metadata)
        
        return await self._resource_aware(agents, capability, tags, metadata)
    
    async def _consistent_hash(self, agents: List[AgentRegistration], capability: str,
                              tags: List[str], metadata: Dict[str, Any]) -> AgentRegistration:
        """Consistent hash selection"""
        if not agents:
            return None
        
        # Use task identifier or other consistent key
        key = metadata.get("task_id", "") if metadata else ""
        hash_value = hashlib.md5(key.encode()).hexdigest()
        index = int(hash_value, 16) % len(agents)
        return agents[index]
    
    async def _calculate_agent_score(self, agent: AgentRegistration, capability: str) -> float:
        """Calculate agent score for selection"""
        try:
            # Resource availability
            cpu_score = 1.0 - agent.load_metrics.get("cpu_usage", 0.5)
            memory_score = 1.0 - agent.load_metrics.get("memory_usage", 0.5)
            
            # Task load
            active_tasks = len(agent.load_metrics.get("active_tasks", []))
            load_score = max(0, 1.0 - (active_tasks / 10.0))
            
            # Performance metrics
            success_rate = agent.load_metrics.get("task_success_rate", 0.5)
            avg_response_time = agent.load_metrics.get("avg_response_time", 1.0)
            response_score = max(0.1, 1.0 - (avg_response_time / 60.0))
            
            # Capability match
            capability_score = 1.0 if capability in agent.capabilities else 0.0
            
            # Age factor (prefer newer agents slightly)
            age_hours = (datetime.utcnow() - agent.registered_at).total_seconds() / 3600
            age_score = max(0.5, 1.0 - (age_hours / 24.0))
            
            # Weighted score
            total_score = (
                cpu_score * 0.25 +
                memory_score * 0.25 +
                load_score * 0.20 +
                success_rate * 0.15 +
                response_score * 0.10 +
                capability_score * 0.05
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Score calculation failed for agent {agent.agent_id}: {e}")
            return 0.0
    
    def _calculate_capability_score(self, agent: AgentRegistration, capability: str) -> float:
        """Calculate capability-specific score"""
        # Check if agent has the exact capability
        if capability in agent.capabilities:
            # Check for specialized capabilities
            specialized_count = sum(1 for cap in agent.capabilities if cap.startswith(capability))
            return min(1.0, 0.5 + (specialized_count * 0.1))
        
        return 0.0

class AgentRegistry:
    """Distributed agent registry with service discovery"""
    
    def __init__(self, distributed_coordinator: DistributedCoordinator):
        self.coordinator = distributed_coordinator
        self.redis: Optional[aioredis.Redis] = None
        self.health_checker: Optional[HealthChecker] = None
        self.load_balancer: Optional[LoadBalancer] = None
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.running = False
        
    async def initialize(self):
        """Initialize agent registry"""
        try:
            self.redis = self.coordinator.redis
            self.health_checker = HealthChecker(self)
            self.load_balancer = LoadBalancer(self)
            
            # Start health checker
            await self.health_checker.start()
            
            # Start event processor
            asyncio.create_task(self._event_processor())
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_task())
            
            self.running = True
            logger.info("Agent registry initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent registry: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown agent registry"""
        try:
            self.running = False
            
            if self.health_checker:
                await self.health_checker.stop()
                
            logger.info("Agent registry shutdown")
            
        except Exception as e:
            logger.error(f"Error shutting down agent registry: {e}")
    
    async def register_agent(self, agent_id: str, agent_type: AgentType, 
                           capabilities: List[str], endpoint: str,
                           metadata: Dict[str, Any] = None, tags: List[str] = None,
                           version: str = "1.0.0") -> bool:
        """Register new agent"""
        try:
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                endpoint=endpoint,
                metadata=metadata or {},
                state=AgentState.REGISTERING,
                registered_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                health_status="healthy",
                load_metrics={},
                tags=tags or [],
                version=version
            )
            
            # Store in Redis
            await self.redis.set(
                f"agent:{agent_id}",
                json.dumps(registration.to_dict()),
                ex=3600  # 1 hour TTL
            )
            
            # Add to capability indexes
            for capability in capabilities:
                await self.redis.sadd(f"capability:{capability}", agent_id)
            
            # Add to type index
            await self.redis.sadd(f"type:{agent_type.value}", agent_id)
            
            # Add to tags indexes
            for tag in tags or []:
                await self.redis.sadd(f"tag:{tag}", agent_id)
            
            # Update state to active
            await self.update_agent_state(agent_id, AgentState.ACTIVE)
            
            # Emit event
            await self._emit_event("agent_registered", {
                "agent_id": agent_id,
                "agent_type": agent_type.value,
                "capabilities": capabilities
            })
            
            logger.info(f"Agent {agent_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister agent"""
        try:
            # Get agent info
            agent = await self.get_agent(agent_id)
            if not agent:
                return False
            
            # Remove from indexes
            for capability in agent.capabilities:
                await self.redis.srem(f"capability:{capability}", agent_id)
            
            await self.redis.srem(f"type:{agent.agent_type.value}", agent_id)
            
            for tag in agent.tags:
                await self.redis.srem(f"tag:{tag}", agent_id)
            
            # Remove agent record
            await self.redis.delete(f"agent:{agent_id}")
            
            # Emit event
            await self._emit_event("agent_deregistered", {
                "agent_id": agent_id,
                "agent_type": agent.agent_type.value
            })
            
            logger.info(f"Agent {agent_id} deregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister agent {agent_id}: {e}")
            return False
    
    async def update_agent_heartbeat(self, agent_id: str, load_metrics: Dict[str, float] = None):
        """Update agent heartbeat"""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return False
            
            agent.last_heartbeat = datetime.utcnow()
            if load_metrics:
                agent.load_metrics.update(load_metrics)
            
            await self.redis.set(
                f"agent:{agent_id}",
                json.dumps(agent.to_dict()),
                ex=3600
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for agent {agent_id}: {e}")
            return False
    
    async def update_agent_state(self, agent_id: str, state: AgentState) -> bool:
        """Update agent state"""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return False
            
            old_state = agent.state
            agent.state = state
            
            await self.redis.set(
                f"agent:{agent_id}",
                json.dumps(agent.to_dict()),
                ex=3600
            )
            
            # Emit event
            await self._emit_event("agent_state_changed", {
                "agent_id": agent_id,
                "old_state": old_state.value,
                "new_state": state.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update state for agent {agent_id}: {e}")
            return False
    
    async def update_agent_metadata(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        """Update agent metadata"""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return False
            
            agent.metadata.update(metadata)
            
            await self.redis.set(
                f"agent:{agent_id}",
                json.dumps(agent.to_dict()),
                ex=3600
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata for agent {agent_id}: {e}")
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent by ID"""
        try:
            data = await self.redis.get(f"agent:{agent_id}")
            if data:
                return AgentRegistration.from_dict(json.loads(data))
            return None
            
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    async def get_all_agents(self) -> List[AgentRegistration]:
        """Get all registered agents"""
        try:
            agents = []
            keys = await self.redis.keys("agent:*")
            
            for key in keys:
                data = await self.redis.get(key)
                if data:
                    agents.append(AgentRegistration.from_dict(json.loads(data)))
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to get all agents: {e}")
            return []
    
    async def get_agents_by_type(self, agent_type: AgentType) -> List[AgentRegistration]:
        """Get agents by type"""
        try:
            agent_ids = await self.redis.smembers(f"type:{agent_type.value}")
            agents = []
            
            for agent_id in agent_ids:
                agent = await self.get_agent(agent_id)
                if agent:
                    agents.append(agent)
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to get agents by type {agent_type}: {e}")
            return []
    
    async def get_agents_by_capability(self, capability: str) -> List[AgentRegistration]:
        """Get agents by capability"""
        try:
            agent_ids = await self.redis.smembers(f"capability:{capability}")
            agents = []
            
            for agent_id in agent_ids:
                agent = await self.get_agent(agent_id)
                if agent:
                    agents.append(agent)
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to get agents by capability {capability}: {e}")
            return []
    
    async def get_agents_by_tag(self, tag: str) -> List[AgentRegistration]:
        """Get agents by tag"""
        try:
            agent_ids = await self.redis.smembers(f"tag:{tag}")
            agents = []
            
            for agent_id in agent_ids:
                agent = await self.get_agent(agent_id)
                if agent:
                    agents.append(agent)
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to get agents by tag {tag}: {e}")
            return []
    
    async def select_agent(self, capability: str, tags: List[str] = None,
                          metadata: Dict[str, Any] = None) -> Optional[AgentRegistration]:
        """Select best agent for capability"""
        return await self.load_balancer.select_agent(capability, tags, metadata)
    
    async def register_service(self, service_id: str, service_type: ServiceType,
                              name: str, host: str, port: int, protocol: str = "tcp",
                              metadata: Dict[str, Any] = None, health_check: str = None) -> bool:
        """Register service endpoint"""
        try:
            endpoint = ServiceEndpoint(
                service_id=service_id,
                service_type=service_type,
                name=name,
                host=host,
                port=port,
                protocol=protocol,
                metadata=metadata or {},
                health_check=health_check,
                registered_at=datetime.utcnow(),
                last_seen=datetime.utcnow()
            )
            
            await self.redis.set(
                f"service:{service_id}",
                json.dumps(endpoint.to_dict()),
                ex=3600
            )
            
            # Add to type index
            await self.redis.sadd(f"service_type:{service_type.value}", service_id)
            
            logger.info(f"Service {service_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service_id}: {e}")
            return False
    
    async def discover_services(self, service_type: ServiceType) -> List[ServiceEndpoint]:
        """Discover services by type"""
        try:
            service_ids = await self.redis.smembers(f"service_type:{service_type.value}")
            services = []
            
            for service_id in service_ids:
                data = await self.redis.get(f"service:{service_id}")
                if data:
                    services.append(ServiceEndpoint.from_dict(json.loads(data)))
            
            return services
            
        except Exception as e:
            logger.error(f"Failed to discover services of type {service_type}: {e}")
            return []
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to handlers"""
        try:
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        await handler(data)
                    except Exception as e:
                        logger.error(f"Event handler error: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to emit event {event_type}: {e}")
    
    async def _event_processor(self):
        """Process registry events"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe("registry:events")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self._emit_event(data['event_type'], data['data'])
                    except Exception as e:
                        logger.error(f"Failed to process event: {e}")
                        
        except Exception as e:
            logger.error(f"Event processor error: {e}")
    
    async def _cleanup_task(self):
        """Cleanup expired registrations"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                # Clean up expired agents
                agents = await self.get_all_agents()
                for agent in agents:
                    if agent.state == AgentState.TERMINATED:
                        await self.deregister_agent(agent.agent_id)
                        
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            agents = await self.get_all_agents()
            
            stats = {
                "total_agents": len(agents),
                "by_type": {},
                "by_state": {},
                "by_capability": {},
                "healthy_agents": 0,
                "unhealthy_agents": 0
            }
            
            for agent in agents:
                # Count by type
                agent_type = agent.agent_type.value
                stats["by_type"][agent_type] = stats["by_type"].get(agent_type, 0) + 1
                
                # Count by state
                state = agent.state.value
                stats["by_state"][state] = stats["by_state"].get(state, 0) + 1
                
                # Count by capability
                for capability in agent.capabilities:
                    stats["by_capability"][capability] = stats["by_capability"].get(capability, 0) + 1
                
                # Count health
                if agent.health_status == "healthy":
                    stats["healthy_agents"] += 1
                else:
                    stats["unhealthy_agents"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get registry stats: {e}")
            return {"error": str(e)}
    
    @asynccontextmanager
    async def agent_lock(self, agent_id: str):
        """Context manager for agent locking"""
        lock = self.coordinator.get_distributed_lock(f"agent:{agent_id}")
        try:
            await lock.acquire()
            yield
        finally:
            await lock.release()