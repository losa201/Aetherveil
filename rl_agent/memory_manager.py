"""
Experience Replay and Memory Management

This module implements sophisticated memory management and experience replay
systems for RL agents, including prioritized experience replay, episodic memory,
and strategic knowledge storage.
"""

import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import pickle
import json
import time
import heapq
from abc import ABC, abstractmethod
import threading
import queue
from pathlib import Path


@dataclass
class Experience:
    """Single experience tuple for replay buffer"""
    state: np.ndarray
    action: Union[int, np.ndarray, Dict]
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    episode_id: Optional[str] = None
    agent_id: Optional[str] = None
    priority: float = 1.0
    td_error: float = 0.0


@dataclass
class Episode:
    """Complete episode for episodic memory"""
    episode_id: str
    experiences: List[Experience]
    total_reward: float
    episode_length: int
    success: bool
    objectives_completed: List[str]
    attack_chain: List[Dict[str, Any]]
    network_topology: str
    difficulty_level: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class StrategicKnowledge:
    """Strategic knowledge extracted from experiences"""
    knowledge_id: str
    knowledge_type: str  # "attack_pattern", "defense_counter", "vulnerability_exploit"
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    success_rate: float
    usage_count: int = 0
    effectiveness_score: float = 0.0
    last_updated: float = field(default_factory=time.time)


class BaseMemoryBuffer(ABC):
    """Base class for memory buffers"""
    
    def __init__(self, capacity: int, name: str = "base_buffer"):
        self.capacity = capacity
        self.name = name
        self.size = 0
        
    @abstractmethod
    def add(self, experience: Experience):
        """Add experience to buffer"""
        pass
        
    @abstractmethod
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        pass
        
    @abstractmethod
    def clear(self):
        """Clear buffer"""
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Get buffer information"""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "size": self.size,
            "utilization": self.size / self.capacity if self.capacity > 0 else 0
        }


class UniformReplayBuffer(BaseMemoryBuffer):
    """Standard uniform experience replay buffer"""
    
    def __init__(self, capacity: int = 100000):
        super().__init__(capacity, "uniform_replay")
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.size = len(self.buffer)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        if self.size < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
        
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.size = 0


class PrioritizedReplayBuffer(BaseMemoryBuffer):
    """Prioritized Experience Replay buffer"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity, "prioritized_replay")
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
        self.beta_increment = 0.001
        
    def add(self, experience: Experience):
        """Add experience with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            # Replace oldest experience
            idx = self.size % self.capacity
            self.buffer[idx] = experience
            self.priorities[idx] = self.max_priority
            
        self.size = len(self.buffer)
        
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        if self.size == 0:
            return [], np.array([]), np.array([])
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:self.size])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, min(batch_size, self.size), p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Update beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
        
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.priorities.clear()
        self.size = 0


class EpisodicMemory:
    """Episodic memory for storing and retrieving complete episodes"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.episode_index = {}  # For fast lookup
        self.success_episodes = []
        self.failure_episodes = []
        
    def add_episode(self, episode: Episode):
        """Add complete episode to memory"""
        if len(self.episodes) >= self.capacity:
            # Remove oldest episode from index
            old_episode = self.episodes[0]
            if old_episode.episode_id in self.episode_index:
                del self.episode_index[old_episode.episode_id]
                
        self.episodes.append(episode)
        self.episode_index[episode.episode_id] = episode
        
        # Categorize episode
        if episode.success:
            self.success_episodes.append(episode.episode_id)
        else:
            self.failure_episodes.append(episode.episode_id)
            
        # Maintain category lists
        self._maintain_category_lists()
        
    def _maintain_category_lists(self):
        """Maintain success/failure episode lists within capacity"""
        max_category_size = self.capacity // 2
        
        if len(self.success_episodes) > max_category_size:
            self.success_episodes = self.success_episodes[-max_category_size:]
            
        if len(self.failure_episodes) > max_category_size:
            self.failure_episodes = self.failure_episodes[-max_category_size:]
            
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get specific episode by ID"""
        return self.episode_index.get(episode_id)
        
    def get_similar_episodes(
        self,
        current_state: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_episodes: int = 10
    ) -> List[Episode]:
        """Get episodes similar to current state"""
        similar_episodes = []
        
        for episode in self.episodes:
            similarity = self._calculate_similarity(current_state, episode)
            if similarity >= similarity_threshold:
                similar_episodes.append((episode, similarity))
                
        # Sort by similarity and return top episodes
        similar_episodes.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in similar_episodes[:max_episodes]]
        
    def _calculate_similarity(self, state: Dict[str, Any], episode: Episode) -> float:
        """Calculate similarity between current state and episode"""
        similarity = 0.0
        factors = 0
        
        # Network topology similarity
        if state.get("network_topology") == episode.network_topology:
            similarity += 0.3
        factors += 1
        
        # Network size similarity
        state_size = state.get("network_size", 20)
        episode_size = len(episode.experiences[0].state) if episode.experiences else 20
        size_similarity = 1.0 - abs(state_size - episode_size) / max(state_size, episode_size)
        similarity += size_similarity * 0.2
        factors += 1
        
        # Objectives similarity
        state_objectives = set(state.get("objectives", []))
        episode_objectives = set(episode.objectives_completed)
        if state_objectives and episode_objectives:
            obj_similarity = len(state_objectives & episode_objectives) / len(state_objectives | episode_objectives)
            similarity += obj_similarity * 0.3
        factors += 1
        
        # Difficulty level similarity
        state_difficulty = state.get("difficulty_level", 0.5)
        diff_similarity = 1.0 - abs(state_difficulty - episode.difficulty_level)
        similarity += diff_similarity * 0.2
        factors += 1
        
        return similarity / factors if factors > 0 else 0.0
        
    def get_successful_strategies(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get successful attack strategies from similar contexts"""
        strategies = []
        
        for episode_id in self.success_episodes[-50:]:  # Recent successful episodes
            episode = self.episode_index.get(episode_id)
            if episode and self._calculate_similarity(context, episode) > 0.5:
                strategy = {
                    "episode_id": episode_id,
                    "attack_chain": episode.attack_chain,
                    "total_reward": episode.total_reward,
                    "success_rate": 1.0,  # This episode was successful
                    "objectives_completed": episode.objectives_completed
                }
                strategies.append(strategy)
                
        return strategies
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        if not self.episodes:
            return {"status": "empty"}
            
        total_episodes = len(self.episodes)
        successful_episodes = len(self.success_episodes)
        
        total_rewards = [ep.total_reward for ep in self.episodes]
        episode_lengths = [ep.episode_length for ep in self.episodes]
        
        return {
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": successful_episodes / total_episodes if total_episodes > 0 else 0,
            "mean_reward": np.mean(total_rewards),
            "mean_length": np.mean(episode_lengths),
            "unique_topologies": len(set(ep.network_topology for ep in self.episodes)),
            "memory_utilization": total_episodes / self.capacity
        }


class StrategicKnowledgeBase:
    """Knowledge base for storing and retrieving strategic insights"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.knowledge = {}
        self.knowledge_index = defaultdict(list)  # Index by type
        self.usage_stats = defaultdict(int)
        
    def add_knowledge(self, knowledge: StrategicKnowledge):
        """Add strategic knowledge"""
        if len(self.knowledge) >= self.capacity:
            # Remove least used knowledge
            self._remove_least_used()
            
        self.knowledge[knowledge.knowledge_id] = knowledge
        self.knowledge_index[knowledge.knowledge_type].append(knowledge.knowledge_id)
        
    def _remove_least_used(self):
        """Remove least used knowledge to make space"""
        if not self.knowledge:
            return
            
        # Find least used knowledge
        least_used_id = min(self.knowledge.keys(), 
                           key=lambda k: self.knowledge[k].usage_count)
        
        knowledge = self.knowledge[least_used_id]
        del self.knowledge[least_used_id]
        
        # Remove from index
        if least_used_id in self.knowledge_index[knowledge.knowledge_type]:
            self.knowledge_index[knowledge.knowledge_type].remove(least_used_id)
            
    def query_knowledge(
        self,
        knowledge_type: str,
        conditions: Dict[str, Any],
        min_success_rate: float = 0.6
    ) -> List[StrategicKnowledge]:
        """Query knowledge base for relevant strategies"""
        relevant_knowledge = []
        
        for knowledge_id in self.knowledge_index.get(knowledge_type, []):
            knowledge = self.knowledge[knowledge_id]
            
            if knowledge.success_rate >= min_success_rate:
                # Check if conditions match
                if self._conditions_match(conditions, knowledge.conditions):
                    knowledge.usage_count += 1
                    relevant_knowledge.append(knowledge)
                    
        # Sort by effectiveness score
        relevant_knowledge.sort(key=lambda k: k.effectiveness_score, reverse=True)
        return relevant_knowledge
        
    def _conditions_match(self, query_conditions: Dict, stored_conditions: Dict) -> bool:
        """Check if query conditions match stored conditions"""
        match_score = 0
        total_conditions = len(stored_conditions)
        
        if total_conditions == 0:
            return True
            
        for key, value in stored_conditions.items():
            if key in query_conditions:
                if isinstance(value, (int, float)) and isinstance(query_conditions[key], (int, float)):
                    # Numerical comparison with tolerance
                    if abs(value - query_conditions[key]) <= 0.1:
                        match_score += 1
                elif value == query_conditions[key]:
                    match_score += 1
                    
        return (match_score / total_conditions) >= 0.7  # 70% match threshold
        
    def update_knowledge_effectiveness(self, knowledge_id: str, success: bool, reward: float):
        """Update knowledge effectiveness based on usage results"""
        if knowledge_id in self.knowledge:
            knowledge = self.knowledge[knowledge_id]
            
            # Update effectiveness score with exponential moving average
            alpha = 0.1
            new_score = 1.0 if success else 0.0
            if reward > 0:
                new_score *= min(2.0, reward / 5.0)  # Normalize reward impact
                
            knowledge.effectiveness_score = (
                alpha * new_score + (1 - alpha) * knowledge.effectiveness_score
            )
            knowledge.last_updated = time.time()
            
    def extract_knowledge_from_episodes(self, episodes: List[Episode]):
        """Extract strategic knowledge from successful episodes"""
        for episode in episodes:
            if episode.success and episode.total_reward > 5.0:  # High-reward episodes
                self._extract_attack_patterns(episode)
                self._extract_vulnerability_exploits(episode)
                
    def _extract_attack_patterns(self, episode: Episode):
        """Extract attack patterns from successful episode"""
        attack_chain = episode.attack_chain
        if len(attack_chain) < 3:
            return
            
        # Look for common patterns in attack sequences
        for i in range(len(attack_chain) - 2):
            pattern = attack_chain[i:i+3]
            
            # Create knowledge entry for this pattern
            knowledge_id = f"pattern_{episode.episode_id}_{i}"
            conditions = {
                "network_topology": episode.network_topology,
                "difficulty_level": episode.difficulty_level
            }
            
            actions = [
                {
                    "action_type": step.get("action_type"),
                    "technique": step.get("technique"),
                    "target_type": step.get("target_type")
                }
                for step in pattern
            ]
            
            knowledge = StrategicKnowledge(
                knowledge_id=knowledge_id,
                knowledge_type="attack_pattern",
                conditions=conditions,
                actions=actions,
                success_rate=1.0,  # From successful episode
                effectiveness_score=episode.total_reward / episode.episode_length
            )
            
            self.add_knowledge(knowledge)
            
    def _extract_vulnerability_exploits(self, episode: Episode):
        """Extract successful vulnerability exploitation techniques"""
        for step in episode.attack_chain:
            if step.get("action_type") == "exploit" and step.get("success"):
                knowledge_id = f"exploit_{step.get('technique')}_{episode.episode_id}"
                
                conditions = {
                    "vulnerability_type": step.get("vulnerability_type"),
                    "target_type": step.get("target_type"),
                    "defense_level": step.get("defense_level", 0.5)
                }
                
                actions = [{
                    "action_type": "exploit",
                    "technique": step.get("technique"),
                    "parameters": step.get("parameters", {})
                }]
                
                knowledge = StrategicKnowledge(
                    knowledge_id=knowledge_id,
                    knowledge_type="vulnerability_exploit",
                    conditions=conditions,
                    actions=actions,
                    success_rate=1.0,
                    effectiveness_score=step.get("value_gained", 1.0)
                )
                
                self.add_knowledge(knowledge)


class ExperienceReplayManager:
    """
    Comprehensive experience replay and memory management system
    
    Coordinates multiple memory systems for optimal learning.
    """
    
    def __init__(
        self,
        replay_buffer_type: str = "prioritized",
        buffer_capacity: int = 100000,
        episodic_capacity: int = 1000,
        knowledge_capacity: int = 10000,
        save_dir: Optional[str] = None
    ):
        """
        Initialize experience replay manager
        
        Args:
            replay_buffer_type: Type of replay buffer ("uniform" or "prioritized")
            buffer_capacity: Capacity of replay buffer
            episodic_capacity: Capacity of episodic memory
            knowledge_capacity: Capacity of knowledge base
            save_dir: Directory to save memory data
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize memory systems
        if replay_buffer_type == "uniform":
            self.replay_buffer = UniformReplayBuffer(buffer_capacity)
        elif replay_buffer_type == "prioritized":
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
        else:
            raise ValueError(f"Unknown replay buffer type: {replay_buffer_type}")
            
        self.episodic_memory = EpisodicMemory(episodic_capacity)
        self.knowledge_base = StrategicKnowledgeBase(knowledge_capacity)
        
        # Current episode tracking
        self.current_episode_experiences = []
        self.current_episode_id = None
        self.episode_counter = 0
        
        # Statistics
        self.total_experiences = 0
        self.total_episodes = 0
        
        # Background processing
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.start_background_processing()
        
    def start_background_processing(self):
        """Start background thread for memory processing"""
        self.processing_thread = threading.Thread(
            target=self._background_processor,
            daemon=True
        )
        self.processing_thread.start()
        
    def _background_processor(self):
        """Background processor for memory maintenance"""
        while True:
            try:
                task = self.processing_queue.get(timeout=1.0)
                task_type = task.get("type")
                
                if task_type == "extract_knowledge":
                    episodes = task.get("episodes", [])
                    self.knowledge_base.extract_knowledge_from_episodes(episodes)
                elif task_type == "save_memory":
                    self._save_memory_to_disk()
                    
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Background processing error: {e}")
                
    def start_episode(self, episode_id: Optional[str] = None) -> str:
        """Start tracking a new episode"""
        if episode_id is None:
            episode_id = f"episode_{self.episode_counter}_{int(time.time())}"
            
        self.current_episode_id = episode_id
        self.current_episode_experiences = []
        self.episode_counter += 1
        
        return episode_id
        
    def add_experience(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray, Dict],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict[str, Any] = None,
        agent_id: Optional[str] = None
    ):
        """Add experience to replay buffer and current episode"""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
            episode_id=self.current_episode_id,
            agent_id=agent_id
        )
        
        # Add to replay buffer
        self.replay_buffer.add(experience)
        
        # Add to current episode
        self.current_episode_experiences.append(experience)
        
        self.total_experiences += 1
        
    def end_episode(
        self,
        success: bool,
        objectives_completed: List[str] = None,
        network_topology: str = "unknown",
        difficulty_level: float = 0.5
    ):
        """End current episode and store in episodic memory"""
        if not self.current_episode_experiences:
            return
            
        total_reward = sum(exp.reward for exp in self.current_episode_experiences)
        episode_length = len(self.current_episode_experiences)
        
        # Extract attack chain from experiences
        attack_chain = []
        for exp in self.current_episode_experiences:
            step_info = exp.info.copy()
            step_info.update({
                "action": exp.action,
                "reward": exp.reward,
                "timestamp": exp.timestamp
            })
            attack_chain.append(step_info)
            
        # Create episode
        episode = Episode(
            episode_id=self.current_episode_id,
            experiences=self.current_episode_experiences.copy(),
            total_reward=total_reward,
            episode_length=episode_length,
            success=success,
            objectives_completed=objectives_completed or [],
            attack_chain=attack_chain,
            network_topology=network_topology,
            difficulty_level=difficulty_level
        )
        
        # Add to episodic memory
        self.episodic_memory.add_episode(episode)
        
        # Queue knowledge extraction for successful episodes
        if success and total_reward > 3.0:
            self.processing_queue.put({
                "type": "extract_knowledge",
                "episodes": [episode]
            })
            
        self.total_episodes += 1
        
        # Reset current episode
        self.current_episode_experiences = []
        self.current_episode_id = None
        
    def sample_batch(self, batch_size: int) -> Tuple[List[Experience], Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample batch from replay buffer"""
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            return self.replay_buffer.sample(batch_size)
        else:
            experiences = self.replay_buffer.sample(batch_size)
            return experiences, None, None
            
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for prioritized replay"""
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priorities(indices, td_errors)
            
    def get_strategic_advice(
        self,
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get strategic advice based on stored knowledge"""
        advice = {
            "similar_episodes": [],
            "successful_strategies": [],
            "relevant_knowledge": [],
            "recommendations": []
        }
        
        # Get similar episodes
        similar_episodes = self.episodic_memory.get_similar_episodes(current_state)
        advice["similar_episodes"] = [
            {
                "episode_id": ep.episode_id,
                "success": ep.success,
                "total_reward": ep.total_reward,
                "objectives_completed": ep.objectives_completed
            }
            for ep in similar_episodes[:5]
        ]
        
        # Get successful strategies
        successful_strategies = self.episodic_memory.get_successful_strategies(context)
        advice["successful_strategies"] = successful_strategies[:3]
        
        # Query knowledge base
        for knowledge_type in ["attack_pattern", "vulnerability_exploit"]:
            relevant_knowledge = self.knowledge_base.query_knowledge(
                knowledge_type, context
            )
            advice["relevant_knowledge"].extend([
                {
                    "type": k.knowledge_type,
                    "actions": k.actions,
                    "success_rate": k.success_rate,
                    "effectiveness": k.effectiveness_score
                }
                for k in relevant_knowledge[:2]
            ])
            
        # Generate recommendations
        advice["recommendations"] = self._generate_recommendations(advice)
        
        return advice
        
    def _generate_recommendations(self, advice: Dict[str, Any]) -> List[str]:
        """Generate tactical recommendations based on advice"""
        recommendations = []
        
        # Based on similar episodes
        if advice["similar_episodes"]:
            successful_eps = [ep for ep in advice["similar_episodes"] if ep["success"]]
            if successful_eps:
                recommendations.append(
                    f"Similar successful episodes found. Consider following "
                    f"strategies that achieved {max(ep['total_reward'] for ep in successful_eps):.1f} reward."
                )
                
        # Based on knowledge base
        if advice["relevant_knowledge"]:
            high_success_knowledge = [
                k for k in advice["relevant_knowledge"] 
                if k["success_rate"] > 0.8
            ]
            if high_success_knowledge:
                recommendations.append(
                    f"High-success rate techniques available. "
                    f"Consider {high_success_knowledge[0]['type']} strategies."
                )
                
        # Based on successful strategies
        if advice["successful_strategies"]:
            recommendations.append(
                f"Found {len(advice['successful_strategies'])} successful strategies "
                f"for similar scenarios."
            )
            
        return recommendations
        
    def save_memory(self, filepath: Optional[str] = None):
        """Save memory to disk"""
        if filepath is None and self.save_dir is None:
            return
            
        save_path = Path(filepath) if filepath else self.save_dir / "memory.pkl"
        
        memory_data = {
            "replay_buffer": self.replay_buffer,
            "episodic_memory": self.episodic_memory,
            "knowledge_base": self.knowledge_base,
            "statistics": self.get_statistics()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(memory_data, f)
            
    def load_memory(self, filepath: str):
        """Load memory from disk"""
        with open(filepath, 'rb') as f:
            memory_data = pickle.load(f)
            
        self.replay_buffer = memory_data["replay_buffer"]
        self.episodic_memory = memory_data["episodic_memory"]
        self.knowledge_base = memory_data["knowledge_base"]
        
    def _save_memory_to_disk(self):
        """Background memory saving"""
        if self.save_dir:
            try:
                self.save_memory()
            except Exception as e:
                print(f"Failed to save memory: {e}")
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        return {
            "total_experiences": self.total_experiences,
            "total_episodes": self.total_episodes,
            "replay_buffer": self.replay_buffer.get_info(),
            "episodic_memory": self.episodic_memory.get_statistics(),
            "knowledge_base": {
                "total_knowledge": len(self.knowledge_base.knowledge),
                "knowledge_types": len(self.knowledge_base.knowledge_index),
                "most_used_knowledge": max(
                    self.knowledge_base.knowledge.values(),
                    key=lambda k: k.usage_count,
                    default=None
                )
            }
        }
        
    def clear_all_memory(self):
        """Clear all memory systems"""
        self.replay_buffer.clear()
        self.episodic_memory = EpisodicMemory(self.episodic_memory.capacity)
        self.knowledge_base = StrategicKnowledgeBase(self.knowledge_base.capacity)
        self.total_experiences = 0
        self.total_episodes = 0