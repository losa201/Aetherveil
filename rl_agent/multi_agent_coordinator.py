"""
Multi-Agent Coordination Capabilities

This module implements sophisticated multi-agent coordination for distributed
cybersecurity operations, including swarm intelligence, cooperative learning,
and tactical coordination protocols.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import time
import threading
import queue
import asyncio
from abc import ABC, abstractmethod
import logging
import json
from collections import defaultdict, deque


class CoordinationProtocol(Enum):
    """Types of coordination protocols"""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    SWARM = "swarm"
    DEMOCRATIC = "democratic"
    MARKET_BASED = "market_based"


class AgentRole(Enum):
    """Roles for specialized agents"""
    SCOUT = "scout"
    EXPLOITER = "exploiter"
    LATERAL_MOVER = "lateral_mover"
    STEALTH_OPERATOR = "stealth_operator"
    DATA_EXFILTRATOR = "data_exfiltrator"
    PERSISTENCE_MANAGER = "persistence_manager"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class CommunicationChannel(Enum):
    """Communication channels between agents"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    SECURE = "secure"
    STEGANOGRAPHIC = "steganographic"
    DEAD_DROP = "dead_drop"


@dataclass
class AgentMessage:
    """Message between agents"""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 1
    encryption_level: int = 0
    ttl: int = 100  # Time to live
    

@dataclass
class CoordinationTask:
    """Task requiring coordination between agents"""
    task_id: str
    task_type: str
    target: str
    required_agents: int
    required_roles: List[AgentRole]
    estimated_duration: int
    priority: int
    dependencies: List[str] = field(default_factory=list)
    assigned_agents: Set[str] = field(default_factory=set)
    status: str = "pending"
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentCapability:
    """Capabilities of an agent"""
    agent_id: str
    role: AgentRole
    skills: Dict[str, float]
    current_load: float
    max_load: float
    position: Optional[str] = None
    status: str = "available"
    last_update: float = field(default_factory=time.time)


class CommunicationManager:
    """Manages communication between agents"""
    
    def __init__(self, max_message_queue: int = 1000):
        self.max_message_queue = max_message_queue
        self.message_queues = defaultdict(lambda: deque(maxlen=max_message_queue))
        self.broadcast_queue = deque(maxlen=max_message_queue)
        self.message_history = deque(maxlen=10000)
        self.encryption_keys = {}
        self.communication_stats = defaultdict(int)
        
    def send_message(self, message: AgentMessage) -> bool:
        """Send message to target agent"""
        try:
            if message.receiver_id == "ALL":
                self.broadcast_queue.append(message)
                self.communication_stats["broadcast_sent"] += 1
            else:
                self.message_queues[message.receiver_id].append(message)
                self.communication_stats["direct_sent"] += 1
                
            self.message_history.append(message)
            return True
            
        except Exception as e:
            self.communication_stats["send_failures"] += 1
            return False
            
    def receive_messages(self, agent_id: str) -> List[AgentMessage]:
        """Receive messages for an agent"""
        messages = []
        
        # Get direct messages
        while self.message_queues[agent_id]:
            messages.append(self.message_queues[agent_id].popleft())
            
        # Get broadcast messages
        while self.broadcast_queue:
            broadcast_msg = self.broadcast_queue.popleft()
            if broadcast_msg.sender_id != agent_id:  # Don't receive own broadcasts
                messages.append(broadcast_msg)
                
        # Filter expired messages
        current_time = time.time()
        valid_messages = [
            msg for msg in messages 
            if (current_time - msg.timestamp) < msg.ttl
        ]
        
        self.communication_stats["messages_received"] += len(valid_messages)
        return valid_messages
        
    def create_secure_channel(self, agent1_id: str, agent2_id: str) -> str:
        """Create secure communication channel between two agents"""
        channel_id = f"secure_{agent1_id}_{agent2_id}"
        # In real implementation, would generate actual encryption keys
        self.encryption_keys[channel_id] = f"key_{random.randint(1000, 9999)}"
        return channel_id
        
    def encrypt_message(self, message: AgentMessage, channel_id: str) -> AgentMessage:
        """Encrypt message for secure channel"""
        # Simplified encryption (in reality would use proper crypto)
        encrypted_message = message
        encrypted_message.encryption_level = 1
        return encrypted_message
        
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return dict(self.communication_stats)


class TaskAllocator:
    """Allocates tasks to agents based on capabilities and availability"""
    
    def __init__(self):
        self.pending_tasks = {}
        self.active_tasks = {}
        self.completed_tasks = {}
        self.agent_capabilities = {}
        self.allocation_history = []
        
    def register_agent(self, capability: AgentCapability):
        """Register agent capabilities"""
        self.agent_capabilities[capability.agent_id] = capability
        
    def submit_task(self, task: CoordinationTask):
        """Submit task for allocation"""
        self.pending_tasks[task.task_id] = task
        
    def allocate_tasks(self) -> Dict[str, List[CoordinationTask]]:
        """Allocate pending tasks to available agents"""
        allocations = defaultdict(list)
        
        # Sort tasks by priority
        sorted_tasks = sorted(
            self.pending_tasks.values(),
            key=lambda t: (t.priority, t.created_at),
            reverse=True
        )
        
        for task in sorted_tasks:
            allocated_agents = self._allocate_single_task(task)
            if allocated_agents:
                task.assigned_agents = set(allocated_agents)
                task.status = "allocated"
                
                for agent_id in allocated_agents:
                    allocations[agent_id].append(task)
                    
                # Move task to active
                self.active_tasks[task.task_id] = task
                del self.pending_tasks[task.task_id]
                
        return dict(allocations)
        
    def _allocate_single_task(self, task: CoordinationTask) -> List[str]:
        """Allocate a single task to best available agents"""
        available_agents = [
            cap for cap in self.agent_capabilities.values()
            if cap.status == "available" and cap.current_load < cap.max_load
        ]
        
        if len(available_agents) < task.required_agents:
            return []
            
        # Score agents for this task
        agent_scores = []
        for agent in available_agents:
            score = self._calculate_agent_score(agent, task)
            agent_scores.append((agent.agent_id, score))
            
        # Select best agents
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected_agents = [
            agent_id for agent_id, _ in agent_scores[:task.required_agents]
        ]
        
        # Check role requirements
        if task.required_roles:
            selected_roles = [
                self.agent_capabilities[agent_id].role 
                for agent_id in selected_agents
            ]
            if not all(role in selected_roles for role in task.required_roles):
                return []  # Cannot satisfy role requirements
                
        return selected_agents
        
    def _calculate_agent_score(self, agent: AgentCapability, task: CoordinationTask) -> float:
        """Calculate agent suitability score for task"""
        score = 0.0
        
        # Role match bonus
        if agent.role in task.required_roles:
            score += 2.0
            
        # Skill match
        task_skills = {
            "reconnaissance": 0.5,
            "exploitation": 0.5,
            "stealth": 0.3,
            "lateral_movement": 0.3
        }
        
        if task.task_type in task_skills:
            required_skill_level = task_skills[task.task_type]
            agent_skill_level = agent.skills.get(task.task_type, 0.0)
            score += min(2.0, agent_skill_level / required_skill_level)
            
        # Load penalty
        load_penalty = agent.current_load / agent.max_load
        score -= load_penalty
        
        # Position bonus (if agent is already in target area)
        if agent.position and task.target:
            if agent.position == task.target:
                score += 1.0
                
        return max(0.0, score)
        
    def complete_task(self, task_id: str, success: bool, results: Dict[str, Any]):
        """Mark task as completed"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "completed" if success else "failed"
            
            # Record allocation history
            self.allocation_history.append({
                "task_id": task_id,
                "assigned_agents": list(task.assigned_agents),
                "success": success,
                "duration": time.time() - task.created_at,
                "results": results
            })
            
            # Move to completed
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            # Update agent availability
            for agent_id in task.assigned_agents:
                if agent_id in self.agent_capabilities:
                    self.agent_capabilities[agent_id].current_load -= 0.3
                    self.agent_capabilities[agent_id].current_load = max(
                        0.0, self.agent_capabilities[agent_id].current_load
                    )
                    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get task allocation statistics"""
        total_tasks = len(self.allocation_history)
        if total_tasks == 0:
            return {"status": "no_data"}
            
        successful_tasks = sum(1 for task in self.allocation_history if task["success"])
        avg_duration = np.mean([task["duration"] for task in self.allocation_history])
        
        return {
            "total_tasks": total_tasks,
            "success_rate": successful_tasks / total_tasks,
            "average_duration": avg_duration,
            "pending_tasks": len(self.pending_tasks),
            "active_tasks": len(self.active_tasks),
            "agent_utilization": {
                agent_id: cap.current_load / cap.max_load
                for agent_id, cap in self.agent_capabilities.items()
            }
        }


class SwarmIntelligence:
    """Implements swarm intelligence for coordinated decision making"""
    
    def __init__(self, pheromone_decay: float = 0.95, learning_rate: float = 0.1):
        self.pheromone_decay = pheromone_decay
        self.learning_rate = learning_rate
        self.pheromone_map = defaultdict(float)
        self.path_success_rates = defaultdict(list)
        self.collective_memory = deque(maxlen=1000)
        
    def update_pheromones(self, path: List[str], success: bool, reward: float):
        """Update pheromone trails based on path success"""
        pheromone_strength = reward if success else -abs(reward) * 0.5
        
        for i in range(len(path) - 1):
            edge = f"{path[i]}->{path[i+1]}"
            current_pheromone = self.pheromone_map[edge]
            
            # Update pheromone with decay
            self.pheromone_map[edge] = (
                current_pheromone * self.pheromone_decay +
                self.learning_rate * pheromone_strength
            )
            
            # Track success rates
            self.path_success_rates[edge].append(float(success))
            if len(self.path_success_rates[edge]) > 100:
                self.path_success_rates[edge].pop(0)
                
    def get_path_probability(self, current_state: str, possible_next_states: List[str]) -> Dict[str, float]:
        """Get probability distribution for next state selection"""
        probabilities = {}
        total_pheromone = 0.0
        
        for next_state in possible_next_states:
            edge = f"{current_state}->{next_state}"
            pheromone = max(0.1, self.pheromone_map[edge])  # Minimum exploration
            probabilities[next_state] = pheromone
            total_pheromone += pheromone
            
        # Normalize to probabilities
        if total_pheromone > 0:
            for state in probabilities:
                probabilities[state] /= total_pheromone
        else:
            # Uniform distribution if no pheromones
            prob = 1.0 / len(possible_next_states)
            probabilities = {state: prob for state in possible_next_states}
            
        return probabilities
        
    def get_collective_recommendation(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendation based on collective swarm experience"""
        # Find similar situations in collective memory
        similar_situations = [
            memory for memory in self.collective_memory
            if self._calculate_situation_similarity(situation, memory["situation"]) > 0.7
        ]
        
        if not similar_situations:
            return {"confidence": 0.0, "recommendation": "explore"}
            
        # Aggregate recommendations
        recommendations = defaultdict(list)
        for memory in similar_situations:
            action = memory["action"]
            success = memory["success"]
            recommendations[action].append(success)
            
        # Calculate confidence scores
        best_action = None
        best_confidence = 0.0
        
        for action, outcomes in recommendations.items():
            success_rate = np.mean(outcomes)
            confidence = len(outcomes) / len(similar_situations)
            
            if success_rate * confidence > best_confidence:
                best_action = action
                best_confidence = success_rate * confidence
                
        return {
            "recommendation": best_action,
            "confidence": best_confidence,
            "sample_size": len(similar_situations)
        }
        
    def _calculate_situation_similarity(self, situation1: Dict, situation2: Dict) -> float:
        """Calculate similarity between two situations"""
        common_keys = set(situation1.keys()) & set(situation2.keys())
        if not common_keys:
            return 0.0
            
        similarity = 0.0
        for key in common_keys:
            val1, val2 = situation1[key], situation2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1)
                similarity += 1.0 - abs(val1 - val2) / max_val
            elif val1 == val2:
                # Exact match
                similarity += 1.0
                
        return similarity / len(common_keys)
        
    def record_collective_experience(self, situation: Dict, action: str, success: bool, reward: float):
        """Record experience in collective memory"""
        experience = {
            "situation": situation,
            "action": action,
            "success": success,
            "reward": reward,
            "timestamp": time.time()
        }
        self.collective_memory.append(experience)


class MultiAgentCoordinator:
    """
    Comprehensive multi-agent coordination system
    
    Orchestrates multiple RL agents for distributed cybersecurity operations
    with sophisticated coordination protocols and swarm intelligence.
    """
    
    def __init__(
        self,
        coordination_protocol: CoordinationProtocol = CoordinationProtocol.HIERARCHICAL,
        max_agents: int = 10,
        communication_delay: float = 0.1
    ):
        """
        Initialize multi-agent coordinator
        
        Args:
            coordination_protocol: Type of coordination protocol to use
            max_agents: Maximum number of agents to coordinate
            communication_delay: Simulated communication delay
        """
        self.coordination_protocol = coordination_protocol
        self.max_agents = max_agents
        self.communication_delay = communication_delay
        
        # Core components
        self.communication_manager = CommunicationManager()
        self.task_allocator = TaskAllocator()
        self.swarm_intelligence = SwarmIntelligence()
        
        # Agent management
        self.agents = {}
        self.agent_capabilities = {}
        self.coordination_state = "idle"
        
        # Coordination metrics
        self.coordination_history = []
        self.performance_metrics = defaultdict(list)
        
        # Logging
        self.logger = logging.getLogger("MultiAgentCoordinator")
        
    def register_agent(
        self,
        agent_id: str,
        role: AgentRole,
        skills: Dict[str, float],
        max_load: float = 1.0
    ):
        """Register a new agent with the coordinator"""
        capability = AgentCapability(
            agent_id=agent_id,
            role=role,
            skills=skills,
            current_load=0.0,
            max_load=max_load
        )
        
        self.agent_capabilities[agent_id] = capability
        self.task_allocator.register_agent(capability)
        
        self.logger.info(f"Registered agent {agent_id} with role {role.value}")
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agent_capabilities:
            del self.agent_capabilities[agent_id]
            self.logger.info(f"Unregistered agent {agent_id}")
            
    def submit_coordination_task(
        self,
        task_type: str,
        target: str,
        required_agents: int = 1,
        required_roles: List[AgentRole] = None,
        priority: int = 1
    ) -> str:
        """Submit a task requiring coordination"""
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        
        task = CoordinationTask(
            task_id=task_id,
            task_type=task_type,
            target=target,
            required_agents=required_agents,
            required_roles=required_roles or [],
            estimated_duration=60,  # Default duration
            priority=priority
        )
        
        self.task_allocator.submit_task(task)
        self.logger.info(f"Submitted coordination task {task_id}")
        
        return task_id
        
    def coordinate_agents(self) -> Dict[str, Any]:
        """Main coordination loop"""
        coordination_results = {
            "allocated_tasks": {},
            "messages_sent": 0,
            "coordination_effectiveness": 0.0
        }
        
        # Allocate tasks to agents
        task_allocations = self.task_allocator.allocate_tasks()
        coordination_results["allocated_tasks"] = task_allocations
        
        # Send coordination messages
        messages_sent = self._send_coordination_messages(task_allocations)
        coordination_results["messages_sent"] = messages_sent
        
        # Calculate coordination effectiveness
        effectiveness = self._calculate_coordination_effectiveness()
        coordination_results["coordination_effectiveness"] = effectiveness
        
        # Update coordination history
        self.coordination_history.append({
            "timestamp": time.time(),
            "tasks_allocated": len(task_allocations),
            "agents_involved": len(set().union(*[tasks for tasks in task_allocations.values()])),
            "effectiveness": effectiveness
        })
        
        return coordination_results
        
    def _send_coordination_messages(self, task_allocations: Dict[str, List[CoordinationTask]]) -> int:
        """Send coordination messages to agents"""
        messages_sent = 0
        
        for agent_id, tasks in task_allocations.items():
            for task in tasks:
                # Send task assignment message
                message = AgentMessage(
                    sender_id="coordinator",
                    receiver_id=agent_id,
                    message_type="task_assignment",
                    content={
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "target": task.target,
                        "collaborators": list(task.assigned_agents - {agent_id}),
                        "priority": task.priority
                    },
                    priority=task.priority
                )
                
                if self.communication_manager.send_message(message):
                    messages_sent += 1
                    
                # Send collaboration messages to other agents
                for collaborator_id in task.assigned_agents:
                    if collaborator_id != agent_id:
                        collab_message = AgentMessage(
                            sender_id="coordinator",
                            receiver_id=collaborator_id,
                            message_type="collaboration_request",
                            content={
                                "task_id": task.task_id,
                                "primary_agent": agent_id,
                                "role_in_task": "collaborator"
                            }
                        )
                        
                        if self.communication_manager.send_message(collab_message):
                            messages_sent += 1
                            
        return messages_sent
        
    def _calculate_coordination_effectiveness(self) -> float:
        """Calculate current coordination effectiveness"""
        if not self.coordination_history:
            return 0.5  # Neutral starting point
            
        recent_history = self.coordination_history[-10:]  # Last 10 coordination cycles
        
        # Factors: task completion rate, agent utilization, communication efficiency
        task_stats = self.task_allocator.get_allocation_stats()
        success_rate = task_stats.get("success_rate", 0.0)
        
        agent_utilization = np.mean(list(task_stats.get("agent_utilization", {0: 0}).values()))
        
        communication_stats = self.communication_manager.get_communication_stats()
        communication_efficiency = 1.0 - (
            communication_stats.get("send_failures", 0) /
            max(1, communication_stats.get("direct_sent", 1))
        )
        
        effectiveness = (success_rate * 0.5 + agent_utilization * 0.3 + communication_efficiency * 0.2)
        return min(1.0, max(0.0, effectiveness))
        
    def process_agent_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Process messages for a specific agent"""
        messages = self.communication_manager.receive_messages(agent_id)
        processed_messages = []
        
        for message in messages:
            processed_msg = {
                "type": message.message_type,
                "content": message.content,
                "sender": message.sender_id,
                "timestamp": message.timestamp,
                "priority": message.priority
            }
            processed_messages.append(processed_msg)
            
        return processed_messages
        
    def report_task_completion(
        self,
        agent_id: str,
        task_id: str,
        success: bool,
        results: Dict[str, Any]
    ):
        """Report task completion from an agent"""
        self.task_allocator.complete_task(task_id, success, results)
        
        # Update swarm intelligence
        if "path" in results:
            path = results["path"]
            reward = results.get("reward", 0.0)
            self.swarm_intelligence.update_pheromones(path, success, reward)
            
        # Record in collective memory
        situation = results.get("situation", {})
        action = results.get("primary_action", "unknown")
        reward = results.get("reward", 0.0)
        self.swarm_intelligence.record_collective_experience(situation, action, success, reward)
        
        # Update agent capability
        if agent_id in self.agent_capabilities:
            capability = self.agent_capabilities[agent_id]
            capability.current_load -= 0.5  # Reduce load after task completion
            capability.current_load = max(0.0, capability.current_load)
            
            # Update skills based on performance
            if success and "skills_used" in results:
                for skill in results["skills_used"]:
                    current_skill = capability.skills.get(skill, 0.0)
                    capability.skills[skill] = min(1.0, current_skill + 0.01)  # Small improvement
                    
        self.logger.info(f"Task {task_id} completed by {agent_id}: {'success' if success else 'failure'}")
        
    def get_coordination_advice(self, agent_id: str, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Provide coordination advice to an agent"""
        advice = {
            "swarm_recommendation": None,
            "collaboration_opportunities": [],
            "strategic_guidance": []
        }
        
        # Get swarm intelligence recommendation
        swarm_rec = self.swarm_intelligence.get_collective_recommendation(situation)
        advice["swarm_recommendation"] = swarm_rec
        
        # Find collaboration opportunities
        agent_capability = self.agent_capabilities.get(agent_id)
        if agent_capability:
            collaborators = self._find_collaboration_opportunities(agent_capability, situation)
            advice["collaboration_opportunities"] = collaborators
            
        # Provide strategic guidance
        guidance = self._generate_strategic_guidance(agent_id, situation)
        advice["strategic_guidance"] = guidance
        
        return advice
        
    def _find_collaboration_opportunities(
        self,
        agent_capability: AgentCapability,
        situation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find potential collaboration opportunities"""
        opportunities = []
        
        for other_agent_id, other_capability in self.agent_capabilities.items():
            if other_agent_id == agent_capability.agent_id:
                continue
                
            if other_capability.status == "available" and other_capability.current_load < 0.8:
                # Calculate collaboration potential
                potential = self._calculate_collaboration_potential(
                    agent_capability, other_capability, situation
                )
                
                if potential > 0.6:
                    opportunities.append({
                        "agent_id": other_agent_id,
                        "role": other_capability.role.value,
                        "potential": potential,
                        "complementary_skills": self._get_complementary_skills(
                            agent_capability, other_capability
                        )
                    })
                    
        # Sort by potential
        opportunities.sort(key=lambda x: x["potential"], reverse=True)
        return opportunities[:3]  # Return top 3 opportunities
        
    def _calculate_collaboration_potential(
        self,
        agent1: AgentCapability,
        agent2: AgentCapability,
        situation: Dict[str, Any]
    ) -> float:
        """Calculate collaboration potential between two agents"""
        potential = 0.0
        
        # Role complementarity
        complementary_roles = {
            AgentRole.SCOUT: [AgentRole.EXPLOITER, AgentRole.STEALTH_OPERATOR],
            AgentRole.EXPLOITER: [AgentRole.LATERAL_MOVER, AgentRole.PERSISTENCE_MANAGER],
            AgentRole.LATERAL_MOVER: [AgentRole.DATA_EXFILTRATOR, AgentRole.STEALTH_OPERATOR],
            AgentRole.STEALTH_OPERATOR: [AgentRole.SCOUT, AgentRole.LATERAL_MOVER],
            AgentRole.DATA_EXFILTRATOR: [AgentRole.STEALTH_OPERATOR, AgentRole.PERSISTENCE_MANAGER]
        }
        
        if agent2.role in complementary_roles.get(agent1.role, []):
            potential += 0.4
            
        # Skill complementarity
        skill_complement = self._calculate_skill_complementarity(agent1.skills, agent2.skills)
        potential += skill_complement * 0.3
        
        # Availability factor
        availability = (2.0 - agent1.current_load - agent2.current_load) / 2.0
        potential += availability * 0.3
        
        return min(1.0, potential)
        
    def _calculate_skill_complementarity(self, skills1: Dict[str, float], skills2: Dict[str, float]) -> float:
        """Calculate how well two skill sets complement each other"""
        all_skills = set(skills1.keys()) | set(skills2.keys())
        if not all_skills:
            return 0.0
            
        complementarity = 0.0
        for skill in all_skills:
            skill1_level = skills1.get(skill, 0.0)
            skill2_level = skills2.get(skill, 0.0)
            
            # High complementarity when one agent is strong where the other is weak
            if skill1_level > 0.7 and skill2_level < 0.3:
                complementarity += 1.0
            elif skill2_level > 0.7 and skill1_level < 0.3:
                complementarity += 1.0
            elif abs(skill1_level - skill2_level) < 0.2:  # Similar skill levels
                complementarity += 0.5
                
        return complementarity / len(all_skills)
        
    def _get_complementary_skills(self, agent1: AgentCapability, agent2: AgentCapability) -> List[str]:
        """Get list of complementary skills between two agents"""
        complementary = []
        
        for skill in set(agent1.skills.keys()) | set(agent2.skills.keys()):
            skill1_level = agent1.skills.get(skill, 0.0)
            skill2_level = agent2.skills.get(skill, 0.0)
            
            if (skill1_level > 0.7 and skill2_level < 0.3) or (skill2_level > 0.7 and skill1_level < 0.3):
                complementary.append(skill)
                
        return complementary
        
    def _generate_strategic_guidance(self, agent_id: str, situation: Dict[str, Any]) -> List[str]:
        """Generate strategic guidance for an agent"""
        guidance = []
        
        agent_capability = self.agent_capabilities.get(agent_id)
        if not agent_capability:
            return guidance
            
        # Role-specific guidance
        if agent_capability.role == AgentRole.SCOUT:
            guidance.append("Focus on reconnaissance and intelligence gathering")
            guidance.append("Share discovered information with exploiter agents")
        elif agent_capability.role == AgentRole.EXPLOITER:
            guidance.append("Prioritize high-value targets identified by scouts")
            guidance.append("Coordinate with stealth operators for covert operations")
        elif agent_capability.role == AgentRole.STEALTH_OPERATOR:
            guidance.append("Minimize detection while maintaining operational capability")
            guidance.append("Support other agents with evasion techniques")
            
        # Situation-specific guidance
        detection_level = situation.get("detection_level", 0.0)
        if detection_level > 0.6:
            guidance.append("High detection risk - consider defensive posture")
            guidance.append("Coordinate with team for synchronized withdrawal if needed")
            
        compromised_hosts = situation.get("compromised_hosts", 0)
        if compromised_hosts > 5:
            guidance.append("Good penetration achieved - consider lateral movement")
            guidance.append("Prepare for data exfiltration operations")
            
        return guidance
        
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status"""
        task_stats = self.task_allocator.get_allocation_stats()
        comm_stats = self.communication_manager.get_communication_stats()
        
        return {
            "coordination_protocol": self.coordination_protocol.value,
            "registered_agents": len(self.agent_capabilities),
            "coordination_effectiveness": self._calculate_coordination_effectiveness(),
            "task_allocation": task_stats,
            "communication": comm_stats,
            "agent_status": {
                agent_id: {
                    "role": cap.role.value,
                    "current_load": cap.current_load,
                    "status": cap.status,
                    "position": cap.position
                }
                for agent_id, cap in self.agent_capabilities.items()
            },
            "recent_coordination_cycles": len(self.coordination_history)
        }
        
    def optimize_coordination(self):
        """Optimize coordination protocols based on performance"""
        if len(self.coordination_history) < 10:
            return
            
        recent_effectiveness = [
            cycle["effectiveness"] for cycle in self.coordination_history[-10:]
        ]
        avg_effectiveness = np.mean(recent_effectiveness)
        
        # Adapt coordination parameters based on performance
        if avg_effectiveness < 0.6:
            # Poor performance - increase communication frequency
            self.communication_delay = max(0.05, self.communication_delay * 0.9)
            self.logger.info("Optimizing coordination: Increased communication frequency")
        elif avg_effectiveness > 0.8:
            # Good performance - can reduce communication overhead
            self.communication_delay = min(0.5, self.communication_delay * 1.1)
            self.logger.info("Optimizing coordination: Reduced communication overhead")
            
    def shutdown(self):
        """Shutdown coordinator and cleanup resources"""
        self.coordination_state = "shutdown"
        self.logger.info("Multi-agent coordinator shutdown completed")