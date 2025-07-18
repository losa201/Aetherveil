"""
Curriculum Learning for Progressive Skill Development

This module implements comprehensive curriculum learning strategies to progressively
train RL agents from basic to advanced cybersecurity skills.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import math
import time
from abc import ABC, abstractmethod
import json
from pathlib import Path


class SkillLevel(Enum):
    """Skill levels for curriculum progression"""
    BEGINNER = "beginner"
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class CurriculumStage(Enum):
    """Curriculum stages with increasing complexity"""
    BASIC_RECONNAISSANCE = "basic_reconnaissance"
    SIMPLE_EXPLOITATION = "simple_exploitation"
    NETWORK_DISCOVERY = "network_discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    STEALTH_OPERATIONS = "stealth_operations"
    ADVANCED_PERSISTENCE = "advanced_persistence"
    MULTI_VECTOR_ATTACKS = "multi_vector_attacks"
    ADAPTIVE_EVASION = "adaptive_evasion"
    COORDINATION_TACTICS = "coordination_tactics"


@dataclass
class CurriculumRequirement:
    """Requirements for progressing to next curriculum stage"""
    min_success_rate: float = 0.7
    min_average_reward: float = 5.0
    min_episodes: int = 50
    max_detection_rate: float = 0.3
    required_objectives: List[str] = field(default_factory=list)
    skill_demonstrations: List[str] = field(default_factory=list)


@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress"""
    success_rate: float = 0.0
    average_reward: float = 0.0
    average_episode_length: float = 0.0
    detection_rate: float = 0.0
    objectives_completion_rate: Dict[str, float] = field(default_factory=dict)
    skill_mastery_scores: Dict[str, float] = field(default_factory=dict)
    episodes_completed: int = 0
    total_training_time: float = 0.0


class CurriculumTask(ABC):
    """Base class for curriculum tasks"""
    
    def __init__(self, name: str, difficulty: float, requirements: List[str] = None):
        self.name = name
        self.difficulty = difficulty  # 0.0 to 1.0
        self.requirements = requirements or []
        self.completion_count = 0
        self.success_count = 0
        
    @abstractmethod
    def generate_environment_config(self) -> Dict[str, Any]:
        """Generate environment configuration for this task"""
        pass
        
    @abstractmethod
    def evaluate_performance(self, episode_results: Dict[str, Any]) -> float:
        """Evaluate agent performance on this task"""
        pass
        
    def get_success_rate(self) -> float:
        """Get success rate for this task"""
        return self.success_count / max(1, self.completion_count)
        
    def record_attempt(self, success: bool):
        """Record task attempt"""
        self.completion_count += 1
        if success:
            self.success_count += 1


class ReconnaissanceTask(CurriculumTask):
    """Basic reconnaissance task"""
    
    def __init__(self):
        super().__init__("reconnaissance", 0.2, ["basic_scanning", "host_discovery"])
        
    def generate_environment_config(self) -> Dict[str, Any]:
        return {
            "network_size": random.randint(5, 10),
            "topology": "flat",
            "vulnerability_density": 0.5,
            "defense_strength": 0.3,
            "objectives": ["discover_hosts"],
            "episode_length": 50
        }
        
    def evaluate_performance(self, episode_results: Dict[str, Any]) -> float:
        discovered_ratio = len(episode_results.get("discovered_hosts", [])) / episode_results.get("network_size", 1)
        return discovered_ratio


class ExploitationTask(CurriculumTask):
    """Basic exploitation task"""
    
    def __init__(self):
        super().__init__("exploitation", 0.4, ["vulnerability_scanning", "basic_exploits"])
        
    def generate_environment_config(self) -> Dict[str, Any]:
        return {
            "network_size": random.randint(8, 15),
            "topology": "flat",
            "vulnerability_density": 0.6,
            "defense_strength": 0.4,
            "objectives": ["compromise_hosts"],
            "episode_length": 75
        }
        
    def evaluate_performance(self, episode_results: Dict[str, Any]) -> float:
        compromised_ratio = len(episode_results.get("compromised_hosts", [])) / episode_results.get("network_size", 1)
        return compromised_ratio


class LateralMovementTask(CurriculumTask):
    """Lateral movement task"""
    
    def __init__(self):
        super().__init__("lateral_movement", 0.6, ["network_navigation", "host_pivoting"])
        
    def generate_environment_config(self) -> Dict[str, Any]:
        return {
            "network_size": random.randint(15, 25),
            "topology": "hierarchical",
            "vulnerability_density": 0.4,
            "defense_strength": 0.5,
            "objectives": ["reach_secure_segment"],
            "episode_length": 100
        }
        
    def evaluate_performance(self, episode_results: Dict[str, Any]) -> float:
        segments_penetrated = len(set(
            host_info.get("segment", "unknown") 
            for host_info in episode_results.get("compromised_hosts_info", {}).values()
        ))
        return min(1.0, segments_penetrated / 3.0)  # Assume 3 main segments


class StealthTask(CurriculumTask):
    """Stealth operations task"""
    
    def __init__(self):
        super().__init__("stealth_operations", 0.7, ["evasion_techniques", "detection_avoidance"])
        
    def generate_environment_config(self) -> Dict[str, Any]:
        return {
            "network_size": random.randint(20, 30),
            "topology": "dmz",
            "vulnerability_density": 0.3,
            "defense_strength": 0.7,
            "detection_threshold": 0.6,  # Lower threshold for harder detection
            "objectives": ["stealth_compromise"],
            "episode_length": 150
        }
        
    def evaluate_performance(self, episode_results: Dict[str, Any]) -> float:
        detection_penalty = episode_results.get("detection_level", 1.0)
        success_bonus = 1.0 if episode_results.get("objectives_completed", 0) > 0 else 0.0
        return max(0.0, success_bonus - detection_penalty)


class AdvancedCoordinationTask(CurriculumTask):
    """Advanced multi-agent coordination task"""
    
    def __init__(self):
        super().__init__("coordination", 0.9, ["multi_agent_tactics", "distributed_attacks"])
        
    def generate_environment_config(self) -> Dict[str, Any]:
        return {
            "network_size": random.randint(30, 50),
            "topology": "cloud_hybrid",
            "vulnerability_density": 0.25,
            "defense_strength": 0.8,
            "detection_threshold": 0.5,
            "objectives": ["coordinated_compromise", "data_exfiltration"],
            "episode_length": 200,
            "multi_agent": True,
            "num_agents": random.randint(2, 4)
        }
        
    def evaluate_performance(self, episode_results: Dict[str, Any]) -> float:
        coordination_score = episode_results.get("coordination_effectiveness", 0.0)
        objective_score = len(episode_results.get("objectives_completed", [])) / 2.0
        return (coordination_score + objective_score) / 2.0


class AdaptiveCurriculumScheduler:
    """Adaptive scheduler for curriculum progression"""
    
    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.difficulty_adjustments = {}
        
    def should_progress(self, current_performance: LearningMetrics, requirements: CurriculumRequirement) -> bool:
        """Determine if agent should progress to next stage"""
        conditions = [
            current_performance.success_rate >= requirements.min_success_rate,
            current_performance.average_reward >= requirements.min_average_reward,
            current_performance.episodes_completed >= requirements.min_episodes,
            current_performance.detection_rate <= requirements.max_detection_rate
        ]
        
        # Check objective completion
        if requirements.required_objectives:
            obj_completion = all(
                current_performance.objectives_completion_rate.get(obj, 0.0) >= 0.7
                for obj in requirements.required_objectives
            )
            conditions.append(obj_completion)
            
        return all(conditions)
        
    def should_regress(self, current_performance: LearningMetrics) -> bool:
        """Determine if agent should regress to easier stage"""
        if len(self.performance_history) < 10:
            return False
            
        # Check for significant performance degradation
        recent_performance = self.performance_history[-10:]
        avg_recent_reward = np.mean([p.average_reward for p in recent_performance])
        
        if len(self.performance_history) >= 20:
            older_performance = self.performance_history[-20:-10]
            avg_older_reward = np.mean([p.average_reward for p in older_performance])
            
            # Regress if recent performance is significantly worse
            if avg_recent_reward < avg_older_reward * 0.7:
                return True
                
        return False
        
    def adapt_difficulty(self, task_name: str, performance: float):
        """Adapt task difficulty based on performance"""
        if task_name not in self.difficulty_adjustments:
            self.difficulty_adjustments[task_name] = 0.0
            
        if performance > 0.8:
            # Increase difficulty if too easy
            self.difficulty_adjustments[task_name] += self.adaptation_rate
        elif performance < 0.3:
            # Decrease difficulty if too hard
            self.difficulty_adjustments[task_name] -= self.adaptation_rate
            
        # Clamp adjustments
        self.difficulty_adjustments[task_name] = np.clip(
            self.difficulty_adjustments[task_name], -0.5, 0.5
        )
        
    def get_adjusted_difficulty(self, task_name: str, base_difficulty: float) -> float:
        """Get adjusted difficulty for a task"""
        adjustment = self.difficulty_adjustments.get(task_name, 0.0)
        return np.clip(base_difficulty + adjustment, 0.1, 1.0)


class CurriculumManager:
    """
    Comprehensive curriculum learning manager
    
    Orchestrates progressive skill development for RL agents through
    structured task sequences and adaptive difficulty adjustment.
    """
    
    def __init__(
        self,
        initial_stage: CurriculumStage = CurriculumStage.BASIC_RECONNAISSANCE,
        adaptive_scheduling: bool = True,
        save_dir: Optional[str] = None
    ):
        """
        Initialize curriculum manager
        
        Args:
            initial_stage: Starting curriculum stage
            adaptive_scheduling: Enable adaptive difficulty scheduling
            save_dir: Directory to save curriculum progress
        """
        self.current_stage = initial_stage
        self.adaptive_scheduling = adaptive_scheduling
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize curriculum components
        self.tasks = self._initialize_tasks()
        self.requirements = self._initialize_requirements()
        self.scheduler = AdaptiveCurriculumScheduler() if adaptive_scheduling else None
        
        # Progress tracking
        self.current_metrics = LearningMetrics()
        self.stage_history = [initial_stage]
        self.performance_history = []
        self.skill_progression = {skill.value: 0.0 for skill in SkillLevel}
        
        # Task selection
        self.current_task_weights = self._initialize_task_weights()
        self.task_performance = {task.name: [] for task in self.tasks.values()}
        
    def _initialize_tasks(self) -> Dict[CurriculumStage, CurriculumTask]:
        """Initialize curriculum tasks for each stage"""
        return {
            CurriculumStage.BASIC_RECONNAISSANCE: ReconnaissanceTask(),
            CurriculumStage.SIMPLE_EXPLOITATION: ExploitationTask(),
            CurriculumStage.LATERAL_MOVEMENT: LateralMovementTask(),
            CurriculumStage.STEALTH_OPERATIONS: StealthTask(),
            CurriculumStage.COORDINATION_TACTICS: AdvancedCoordinationTask(),
            # Add more tasks as needed
        }
        
    def _initialize_requirements(self) -> Dict[CurriculumStage, CurriculumRequirement]:
        """Initialize progression requirements for each stage"""
        return {
            CurriculumStage.BASIC_RECONNAISSANCE: CurriculumRequirement(
                min_success_rate=0.6,
                min_average_reward=3.0,
                min_episodes=30,
                max_detection_rate=0.5,
                required_objectives=["discover_hosts"]
            ),
            CurriculumStage.SIMPLE_EXPLOITATION: CurriculumRequirement(
                min_success_rate=0.7,
                min_average_reward=5.0,
                min_episodes=50,
                max_detection_rate=0.4,
                required_objectives=["compromise_hosts"]
            ),
            CurriculumStage.LATERAL_MOVEMENT: CurriculumRequirement(
                min_success_rate=0.65,
                min_average_reward=7.0,
                min_episodes=75,
                max_detection_rate=0.35,
                required_objectives=["reach_secure_segment"]
            ),
            CurriculumStage.STEALTH_OPERATIONS: CurriculumRequirement(
                min_success_rate=0.6,
                min_average_reward=8.0,
                min_episodes=100,
                max_detection_rate=0.25,
                required_objectives=["stealth_compromise"]
            ),
            CurriculumStage.COORDINATION_TACTICS: CurriculumRequirement(
                min_success_rate=0.55,
                min_average_reward=10.0,
                min_episodes=150,
                max_detection_rate=0.2,
                required_objectives=["coordinated_compromise", "data_exfiltration"]
            ),
        }
        
    def _initialize_task_weights(self) -> Dict[str, float]:
        """Initialize task selection weights"""
        current_task = self.tasks.get(self.current_stage)
        if current_task:
            return {current_task.name: 1.0}
        return {}
        
    def get_next_task(self) -> Tuple[CurriculumTask, Dict[str, Any]]:
        """Get next task and environment configuration"""
        # Select task based on current stage and weights
        if self.current_stage in self.tasks:
            task = self.tasks[self.current_stage]
            
            # Generate environment config
            env_config = task.generate_environment_config()
            
            # Apply adaptive difficulty if enabled
            if self.scheduler:
                base_difficulty = task.difficulty
                adjusted_difficulty = self.scheduler.get_adjusted_difficulty(
                    task.name, base_difficulty
                )
                env_config = self._adjust_config_difficulty(env_config, adjusted_difficulty)
                
            return task, env_config
        else:
            # Fallback to reconnaissance task
            task = self.tasks[CurriculumStage.BASIC_RECONNAISSANCE]
            return task, task.generate_environment_config()
            
    def _adjust_config_difficulty(self, config: Dict[str, Any], difficulty: float) -> Dict[str, Any]:
        """Adjust environment configuration based on difficulty"""
        adjusted_config = config.copy()
        
        # Adjust network size
        base_size = config.get("network_size", 20)
        size_adjustment = int((difficulty - 0.5) * base_size * 0.5)
        adjusted_config["network_size"] = max(5, base_size + size_adjustment)
        
        # Adjust defense strength
        base_defense = config.get("defense_strength", 0.5)
        defense_adjustment = (difficulty - 0.5) * 0.3
        adjusted_config["defense_strength"] = np.clip(base_defense + defense_adjustment, 0.1, 0.9)
        
        # Adjust vulnerability density (inverse relationship)
        base_vuln = config.get("vulnerability_density", 0.5)
        vuln_adjustment = -(difficulty - 0.5) * 0.3
        adjusted_config["vulnerability_density"] = np.clip(base_vuln + vuln_adjustment, 0.1, 0.9)
        
        # Adjust episode length
        base_length = config.get("episode_length", 100)
        length_adjustment = int((difficulty - 0.5) * base_length * 0.3)
        adjusted_config["episode_length"] = max(50, base_length + length_adjustment)
        
        return adjusted_config
        
    def record_episode_result(self, episode_results: Dict[str, Any]):
        """Record results from completed episode"""
        # Update current metrics
        self._update_metrics(episode_results)
        
        # Record task performance
        current_task = self.tasks.get(self.current_stage)
        if current_task:
            task_performance = current_task.evaluate_performance(episode_results)
            self.task_performance[current_task.name].append(task_performance)
            
            # Record task attempt
            success = episode_results.get("success", False)
            current_task.record_attempt(success)
            
            # Adaptive difficulty adjustment
            if self.scheduler:
                self.scheduler.adapt_difficulty(current_task.name, task_performance)
                
        # Check for stage progression
        self._check_progression()
        
    def _update_metrics(self, episode_results: Dict[str, Any]):
        """Update learning metrics with episode results"""
        # Exponential moving average for metrics
        alpha = 0.1
        
        episode_reward = episode_results.get("total_reward", 0.0)
        episode_success = episode_results.get("success", False)
        episode_length = episode_results.get("episode_length", 0)
        detection_level = episode_results.get("detection_level", 0.0)
        
        # Update metrics
        self.current_metrics.average_reward = (
            alpha * episode_reward + 
            (1 - alpha) * self.current_metrics.average_reward
        )
        
        self.current_metrics.success_rate = (
            alpha * float(episode_success) + 
            (1 - alpha) * self.current_metrics.success_rate
        )
        
        self.current_metrics.average_episode_length = (
            alpha * episode_length + 
            (1 - alpha) * self.current_metrics.average_episode_length
        )
        
        self.current_metrics.detection_rate = (
            alpha * detection_level + 
            (1 - alpha) * self.current_metrics.detection_rate
        )
        
        # Update objective completion rates
        objectives_completed = episode_results.get("objectives_completed", [])
        for objective in objectives_completed:
            current_rate = self.current_metrics.objectives_completion_rate.get(objective, 0.0)
            self.current_metrics.objectives_completion_rate[objective] = (
                alpha * 1.0 + (1 - alpha) * current_rate
            )
            
        self.current_metrics.episodes_completed += 1
        
        # Record performance snapshot
        self.performance_history.append(LearningMetrics(
            success_rate=self.current_metrics.success_rate,
            average_reward=self.current_metrics.average_reward,
            average_episode_length=self.current_metrics.average_episode_length,
            detection_rate=self.current_metrics.detection_rate,
            objectives_completion_rate=self.current_metrics.objectives_completion_rate.copy(),
            episodes_completed=self.current_metrics.episodes_completed
        ))
        
    def _check_progression(self):
        """Check if agent should progress to next stage"""
        current_requirements = self.requirements.get(self.current_stage)
        if not current_requirements:
            return
            
        # Check for progression
        if self.scheduler and self.scheduler.should_progress(self.current_metrics, current_requirements):
            self._progress_stage()
        elif self.scheduler and self.scheduler.should_regress(self.current_metrics):
            self._regress_stage()
            
    def _progress_stage(self):
        """Progress to next curriculum stage"""
        stage_order = list(CurriculumStage)
        current_idx = stage_order.index(self.current_stage)
        
        if current_idx < len(stage_order) - 1:
            next_stage = stage_order[current_idx + 1]
            self.current_stage = next_stage
            self.stage_history.append(next_stage)
            
            # Reset some metrics for new stage
            self.current_metrics.episodes_completed = 0
            
            print(f"Progressed to curriculum stage: {next_stage.value}")
            
    def _regress_stage(self):
        """Regress to previous curriculum stage"""
        stage_order = list(CurriculumStage)
        current_idx = stage_order.index(self.current_stage)
        
        if current_idx > 0:
            prev_stage = stage_order[current_idx - 1]
            self.current_stage = prev_stage
            self.stage_history.append(prev_stage)
            
            print(f"Regressed to curriculum stage: {prev_stage.value}")
            
    def get_skill_level(self) -> SkillLevel:
        """Determine current skill level based on progress"""
        stage_progress = len(set(self.stage_history)) / len(CurriculumStage)
        avg_performance = self.current_metrics.average_reward / 10.0  # Normalize
        
        overall_skill = (stage_progress + avg_performance) / 2.0
        
        if overall_skill < 0.2:
            return SkillLevel.BEGINNER
        elif overall_skill < 0.4:
            return SkillLevel.NOVICE
        elif overall_skill < 0.6:
            return SkillLevel.INTERMEDIATE
        elif overall_skill < 0.8:
            return SkillLevel.ADVANCED
        elif overall_skill < 0.95:
            return SkillLevel.EXPERT
        else:
            return SkillLevel.MASTER
            
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get comprehensive curriculum status"""
        return {
            "current_stage": self.current_stage.value,
            "skill_level": self.get_skill_level().value,
            "current_metrics": {
                "success_rate": self.current_metrics.success_rate,
                "average_reward": self.current_metrics.average_reward,
                "detection_rate": self.current_metrics.detection_rate,
                "episodes_completed": self.current_metrics.episodes_completed
            },
            "stage_history": [stage.value for stage in self.stage_history],
            "task_performance": {
                name: {
                    "success_rate": task.get_success_rate(),
                    "completion_count": task.completion_count,
                    "recent_performance": perf[-10:] if perf else []
                }
                for name, task in self.tasks.items()
                for perf in [self.task_performance.get(task.name, [])]
            },
            "progression_requirements": {
                req_name: {
                    "min_success_rate": req.min_success_rate,
                    "min_average_reward": req.min_average_reward,
                    "current_success_rate": self.current_metrics.success_rate,
                    "current_average_reward": self.current_metrics.average_reward,
                    "progress_ratio": min(1.0, self.current_metrics.success_rate / req.min_success_rate)
                }
                for req_name, req in [(self.current_stage.value, self.requirements.get(self.current_stage))]
                if req is not None
            }
        }
        
    def save_progress(self, filepath: Optional[str] = None):
        """Save curriculum progress to disk"""
        if filepath is None and self.save_dir is None:
            return
            
        save_path = Path(filepath) if filepath else self.save_dir / "curriculum_progress.json"
        
        progress_data = {
            "current_stage": self.current_stage.value,
            "stage_history": [stage.value for stage in self.stage_history],
            "current_metrics": self.current_metrics.__dict__,
            "performance_history": [metrics.__dict__ for metrics in self.performance_history],
            "task_performance": self.task_performance,
            "skill_progression": self.skill_progression
        }
        
        with open(save_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    def load_progress(self, filepath: str):
        """Load curriculum progress from disk"""
        with open(filepath, 'r') as f:
            progress_data = json.load(f)
            
        # Restore state
        self.current_stage = CurriculumStage(progress_data["current_stage"])
        self.stage_history = [CurriculumStage(stage) for stage in progress_data["stage_history"]]
        
        # Restore metrics
        metrics_data = progress_data["current_metrics"]
        self.current_metrics = LearningMetrics(**metrics_data)
        
        # Restore performance history
        self.performance_history = [
            LearningMetrics(**metrics_dict) 
            for metrics_dict in progress_data["performance_history"]
        ]
        
        self.task_performance = progress_data.get("task_performance", {})
        self.skill_progression = progress_data.get("skill_progression", {})
        
    def reset_curriculum(self, initial_stage: CurriculumStage = CurriculumStage.BASIC_RECONNAISSANCE):
        """Reset curriculum to initial state"""
        self.current_stage = initial_stage
        self.stage_history = [initial_stage]
        self.current_metrics = LearningMetrics()
        self.performance_history = []
        self.task_performance = {task.name: [] for task in self.tasks.values()}
        
        # Reset task statistics
        for task in self.tasks.values():
            task.completion_count = 0
            task.success_count = 0
            
    def get_training_recommendations(self) -> List[str]:
        """Get recommendations for improving training"""
        recommendations = []
        
        # Check success rate
        if self.current_metrics.success_rate < 0.5:
            recommendations.append(
                "Low success rate detected. Consider reducing difficulty or "
                "providing more guidance in early stages."
            )
            
        # Check detection rate
        if self.current_metrics.detection_rate > 0.6:
            recommendations.append(
                "High detection rate. Focus on stealth training and "
                "detection avoidance strategies."
            )
            
        # Check episode length
        if self.current_metrics.average_episode_length > 150:
            recommendations.append(
                "Episodes are taking too long. Consider efficiency training "
                "or reducing task complexity."
            )
            
        # Check stage progression
        if len(set(self.stage_history)) == 1 and self.current_metrics.episodes_completed > 200:
            recommendations.append(
                "Agent stuck on current stage. Consider adjusting progression "
                "requirements or providing additional training resources."
            )
            
        return recommendations