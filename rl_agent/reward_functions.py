"""
Reward Functions for Tactical Learning

This module implements sophisticated reward functions that guide RL agents
to learn effective cybersecurity tactics while balancing objectives like
stealth, efficiency, and strategic value.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math
from abc import ABC, abstractmethod


class RewardType(Enum):
    """Types of rewards in cybersecurity scenarios"""
    DISCOVERY = "discovery"
    EXPLOITATION = "exploitation"
    PERSISTENCE = "persistence"
    STEALTH = "stealth"
    EFFICIENCY = "efficiency"
    STRATEGIC_VALUE = "strategic_value"
    OBJECTIVE_COMPLETION = "objective_completion"
    SURVIVAL = "survival"
    COORDINATION = "coordination"
    INNOVATION = "innovation"


@dataclass
class RewardEvent:
    """Represents a reward-generating event"""
    event_type: str
    value: float
    timestamp: int
    context: Dict[str, Any]
    agent_id: Optional[str] = None
    target_id: Optional[str] = None
    technique_used: Optional[str] = None


class BaseRewardFunction(ABC):
    """Base class for reward functions"""
    
    def __init__(self, weight: float = 1.0, name: str = "base_reward"):
        self.weight = weight
        self.name = name
        
    @abstractmethod
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        """Calculate reward for the given transition"""
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about this reward function"""
        return {
            "name": self.name,
            "weight": self.weight,
            "type": self.__class__.__name__
        }


class DiscoveryReward(BaseRewardFunction):
    """Reward for discovering new hosts, services, and vulnerabilities"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "discovery_reward")
        self.base_discovery_reward = 0.1
        self.vulnerability_bonus = 0.2
        self.critical_asset_bonus = 0.5
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Reward for discovering new hosts
        if "discoveries" in info and info["discoveries"]:
            num_discoveries = len(info["discoveries"])
            reward += self.base_discovery_reward * num_discoveries
            
            # Bonus for discovering high-value targets
            for discovery in info["discoveries"]:
                if discovery.get("type") == "domain_controller":
                    reward += self.critical_asset_bonus
                elif discovery.get("type") in ["database", "server"]:
                    reward += self.critical_asset_bonus * 0.5
                    
        # Reward for discovering vulnerabilities
        if "vulnerabilities_found" in info and info["vulnerabilities_found"]:
            for vuln in info["vulnerabilities_found"]:
                severity = vuln.get("severity", 0.5)
                reward += self.vulnerability_bonus * severity
                
        return reward * self.weight


class ExploitationReward(BaseRewardFunction):
    """Reward for successful exploitation and compromise"""
    
    def __init__(self, weight: float = 2.0):
        super().__init__(weight, "exploitation_reward")
        self.base_exploit_reward = 1.0
        self.privilege_escalation_bonus = 1.5
        self.lateral_movement_bonus = 0.8
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Reward for successful exploitation
        if "compromised" in info and info["compromised"]:
            num_compromised = len(info["compromised"])
            reward += self.base_exploit_reward * num_compromised
            
        # Reward for privilege escalation
        if "escalated" in info and info["escalated"]:
            num_escalated = len(info["escalated"])
            reward += self.privilege_escalation_bonus * num_escalated
            
        # Reward for lateral movement
        if "moved_to" in info and info["moved_to"]:
            reward += self.lateral_movement_bonus
            
        # Penalty for failed attempts
        if "failed" in info and info["failed"]:
            num_failed = len(info["failed"])
            reward -= 0.3 * num_failed
            
        return reward * self.weight


class StealthReward(BaseRewardFunction):
    """Reward for maintaining stealth and avoiding detection"""
    
    def __init__(self, weight: float = 1.5):
        super().__init__(weight, "stealth_reward")
        self.detection_penalty_base = -2.0
        self.stealth_bonus = 0.1
        self.noise_penalty = -0.05
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Get detection levels
        prev_detection = state.get("detection_level", 0.0)
        curr_detection = next_state.get("detection_level", 0.0)
        detection_increase = curr_detection - prev_detection
        
        # Penalty for increasing detection
        if detection_increase > 0:
            # Exponential penalty for high detection levels
            penalty_multiplier = math.exp(curr_detection * 2)
            reward += self.detection_penalty_base * detection_increase * penalty_multiplier
            
        # Bonus for maintaining low detection
        if curr_detection < 0.3:
            reward += self.stealth_bonus
            
        # Bonus for reducing detection (e.g., through waiting)
        if detection_increase < 0:
            reward += abs(detection_increase) * 0.5
            
        # Consider action stealthiness
        action_type = action.get("type", "unknown")
        stealth_penalties = {
            "exploit": -0.1,
            "scan": -0.05,
            "data_exfiltration": -0.15,
            "persistence": -0.08,
            "reconnaissance": -0.02,
            "wait": 0.02
        }
        
        if action_type in stealth_penalties:
            reward += stealth_penalties[action_type]
            
        return reward * self.weight


class EfficiencyReward(BaseRewardFunction):
    """Reward for efficient attacks and resource usage"""
    
    def __init__(self, weight: float = 0.8):
        super().__init__(weight, "efficiency_reward")
        self.time_penalty = -0.01
        self.redundancy_penalty = -0.1
        self.efficiency_bonus = 0.2
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Time penalty - encourage faster completion
        current_step = next_state.get("current_step", 0)
        max_steps = next_state.get("max_steps", 100)
        time_ratio = current_step / max_steps
        reward += self.time_penalty * time_ratio
        
        # Penalty for redundant actions
        attack_chain = next_state.get("attack_chain", [])
        if len(attack_chain) > 1:
            recent_actions = [step.get("action_type") for step in attack_chain[-3:]]
            if len(set(recent_actions)) < len(recent_actions):
                reward += self.redundancy_penalty
                
        # Bonus for efficient progression
        compromised_hosts = len(next_state.get("compromised_hosts", []))
        if compromised_hosts > 0 and current_step > 0:
            efficiency = compromised_hosts / current_step
            if efficiency > 0.1:  # Good efficiency threshold
                reward += self.efficiency_bonus * efficiency
                
        return reward * self.weight


class StrategicValueReward(BaseRewardFunction):
    """Reward based on strategic value of compromised assets"""
    
    def __init__(self, weight: float = 2.5):
        super().__init__(weight, "strategic_value_reward")
        self.asset_values = {
            "domain_controller": 5.0,
            "database": 3.0,
            "server": 2.0,
            "web_server": 1.5,
            "workstation": 1.0,
            "iot_device": 0.5
        }
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Reward based on value of newly compromised hosts
        if "compromised" in info and info["compromised"]:
            for host_id in info["compromised"]:
                host_info = next_state.get("hosts", {}).get(host_id, {})
                host_type = host_info.get("type", "workstation")
                asset_value = self.asset_values.get(host_type, 1.0)
                
                # Multiply by access level
                access_level = host_info.get("access_level", "user")
                access_multiplier = {"user": 1.0, "admin": 1.5, "root": 2.0}.get(access_level, 1.0)
                
                reward += asset_value * access_multiplier
                
        # Bonus for data exfiltration from valuable sources
        if "exfiltrated" in info and info["exfiltrated"]:
            for data_info in info["exfiltrated"]:
                data_value = data_info.get("value", 1.0)
                reward += data_value
                
        return reward * self.weight


class ObjectiveReward(BaseRewardFunction):
    """Reward for completing specific objectives"""
    
    def __init__(self, weight: float = 5.0, objectives: Optional[List[str]] = None):
        super().__init__(weight, "objective_reward")
        self.objectives = objectives or ["compromise_domain_controller", "exfiltrate_data"]
        self.completion_rewards = {
            "compromise_domain_controller": 10.0,
            "exfiltrate_data": 8.0,
            "establish_persistence": 5.0,
            "credential_access": 3.0,
            "network_discovery": 2.0
        }
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Check for newly completed objectives
        prev_completed = set(state.get("objectives_completed", []))
        curr_completed = set(next_state.get("objectives_completed", []))
        newly_completed = curr_completed - prev_completed
        
        for objective in newly_completed:
            if objective in self.completion_rewards:
                reward += self.completion_rewards[objective]
                
        # Progressive rewards for objective progress
        for objective in self.objectives:
            if objective == "compromise_domain_controller":
                # Check if domain controller is discovered/compromised
                hosts = next_state.get("hosts", {})
                dc_progress = 0
                for host_info in hosts.values():
                    if host_info.get("type") == "domain_controller":
                        if host_info.get("discovered"):
                            dc_progress += 0.2
                        if host_info.get("compromised"):
                            dc_progress += 0.8
                            
                # Reward progress towards DC compromise
                prev_progress = self._get_objective_progress(state, objective)
                if dc_progress > prev_progress:
                    reward += (dc_progress - prev_progress) * 2.0
                    
        return reward * self.weight
        
    def _get_objective_progress(self, state: Dict, objective: str) -> float:
        """Get progress towards a specific objective"""
        if objective == "compromise_domain_controller":
            hosts = state.get("hosts", {})
            for host_info in hosts.values():
                if host_info.get("type") == "domain_controller":
                    if host_info.get("compromised"):
                        return 1.0
                    elif host_info.get("discovered"):
                        return 0.2
            return 0.0
        return 0.0


class SurvivalReward(BaseRewardFunction):
    """Reward for surviving without being detected/terminated"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "survival_reward")
        self.survival_bonus = 0.05
        self.termination_penalty = -5.0
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Bonus for each step survived
        if not info.get("terminated", False):
            reward += self.survival_bonus
            
        # Penalty for termination due to detection
        if info.get("terminated", False):
            termination_reason = info.get("termination_reason", "unknown")
            if termination_reason == "detected":
                reward += self.termination_penalty
            elif termination_reason == "objectives_completed":
                reward += 2.0  # Bonus for successful completion
                
        return reward * self.weight


class CoordinationReward(BaseRewardFunction):
    """Reward for multi-agent coordination"""
    
    def __init__(self, weight: float = 1.2):
        super().__init__(weight, "coordination_reward")
        self.collaboration_bonus = 0.3
        self.information_sharing_bonus = 0.2
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Bonus for coordinated actions with other agents
        if "coordinated_agents" in info and info["coordinated_agents"]:
            num_coordinated = len(info["coordinated_agents"])
            reward += self.collaboration_bonus * num_coordinated
            
        # Bonus for sharing valuable information
        if "shared_intelligence" in info and info["shared_intelligence"]:
            for intel in info["shared_intelligence"]:
                intel_value = intel.get("value", 0.5)
                reward += self.information_sharing_bonus * intel_value
                
        # Bonus for successful distributed attacks
        if "distributed_attack" in info and info["distributed_attack"]:
            success_rate = info["distributed_attack"].get("success_rate", 0.0)
            reward += success_rate * 0.5
            
        return reward * self.weight


class InnovationReward(BaseRewardFunction):
    """Reward for novel or creative attack strategies"""
    
    def __init__(self, weight: float = 0.5):
        super().__init__(weight, "innovation_reward")
        self.novelty_bonus = 0.1
        self.technique_diversity_bonus = 0.05
        
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # Bonus for using novel attack techniques
        attack_chain = next_state.get("attack_chain", [])
        if len(attack_chain) > 0:
            recent_techniques = [step.get("technique") for step in attack_chain[-5:]]
            unique_techniques = len(set(recent_techniques))
            
            if unique_techniques > 3:  # Good diversity
                reward += self.technique_diversity_bonus * unique_techniques
                
        # Bonus for novel attack paths
        if "attack_path_novelty" in info:
            novelty_score = info["attack_path_novelty"]
            reward += self.novelty_bonus * novelty_score
            
        return reward * self.weight


class TacticalRewardFunction:
    """
    Comprehensive tactical reward function combining multiple reward components
    
    This class orchestrates multiple reward functions to provide rich learning
    signals for cybersecurity RL agents.
    """
    
    def __init__(
        self,
        reward_weights: Optional[Dict[str, float]] = None,
        custom_rewards: Optional[List[BaseRewardFunction]] = None,
        adaptive_weights: bool = False,
        curriculum_stage: str = "beginner"
    ):
        """
        Initialize the tactical reward function
        
        Args:
            reward_weights: Custom weights for different reward components
            custom_rewards: Additional custom reward functions
            adaptive_weights: Whether to adapt weights during training
            curriculum_stage: Current curriculum stage affecting reward emphasis
        """
        self.adaptive_weights = adaptive_weights
        self.curriculum_stage = curriculum_stage
        
        # Default reward weights
        default_weights = {
            "discovery": 1.0,
            "exploitation": 2.0,
            "stealth": 1.5,
            "efficiency": 0.8,
            "strategic_value": 2.5,
            "objective": 5.0,
            "survival": 1.0,
            "coordination": 1.2,
            "innovation": 0.5
        }
        
        # Update with custom weights
        if reward_weights:
            default_weights.update(reward_weights)
            
        # Adjust weights based on curriculum stage
        self._adjust_weights_for_curriculum(default_weights)
        
        # Initialize reward functions
        self.reward_functions = [
            DiscoveryReward(weight=default_weights["discovery"]),
            ExploitationReward(weight=default_weights["exploitation"]),
            StealthReward(weight=default_weights["stealth"]),
            EfficiencyReward(weight=default_weights["efficiency"]),
            StrategicValueReward(weight=default_weights["strategic_value"]),
            ObjectiveReward(weight=default_weights["objective"]),
            SurvivalReward(weight=default_weights["survival"]),
            CoordinationReward(weight=default_weights["coordination"]),
            InnovationReward(weight=default_weights["innovation"])
        ]
        
        # Add custom reward functions
        if custom_rewards:
            self.reward_functions.extend(custom_rewards)
            
        # Tracking variables
        self.reward_history = []
        self.component_rewards = {rf.name: [] for rf in self.reward_functions}
        
    def _adjust_weights_for_curriculum(self, weights: Dict[str, float]):
        """Adjust reward weights based on curriculum stage"""
        if self.curriculum_stage == "beginner":
            # Emphasize discovery and basic exploitation
            weights["discovery"] *= 1.5
            weights["exploitation"] *= 1.2
            weights["stealth"] *= 0.8
            weights["efficiency"] *= 0.5
            
        elif self.curriculum_stage == "intermediate":
            # Balance all components
            pass  # Use default weights
            
        elif self.curriculum_stage == "advanced":
            # Emphasize stealth, efficiency, and coordination
            weights["stealth"] *= 1.3
            weights["efficiency"] *= 1.2
            weights["coordination"] *= 1.4
            weights["innovation"] *= 1.5
            
        elif self.curriculum_stage == "expert":
            # Focus on advanced tactics and innovation
            weights["strategic_value"] *= 1.2
            weights["coordination"] *= 1.5
            weights["innovation"] *= 2.0
            weights["stealth"] *= 1.4
            
    def calculate_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive reward from all components
        
        Args:
            state: Previous environment state
            action: Action taken
            next_state: New environment state
            info: Additional information from environment
            
        Returns:
            Tuple of (total_reward, component_rewards)
        """
        total_reward = 0.0
        component_rewards = {}
        
        # Calculate reward from each component
        for reward_func in self.reward_functions:
            try:
                component_reward = reward_func.calculate_reward(state, action, next_state, info)
                total_reward += component_reward
                component_rewards[reward_func.name] = component_reward
                
                # Track component rewards
                self.component_rewards[reward_func.name].append(component_reward)
                
            except Exception as e:
                # Log error but continue with other reward functions
                component_rewards[reward_func.name] = 0.0
                print(f"Error in reward function {reward_func.name}: {e}")
                
        # Apply global reward shaping
        total_reward = self._apply_reward_shaping(total_reward, state, next_state, info)
        
        # Track total reward
        self.reward_history.append(total_reward)
        
        # Adaptive weight adjustment
        if self.adaptive_weights and len(self.reward_history) % 100 == 0:
            self._adapt_weights()
            
        return total_reward, component_rewards
        
    def _apply_reward_shaping(
        self,
        reward: float,
        state: Dict,
        next_state: Dict,
        info: Dict
    ) -> float:
        """Apply global reward shaping techniques"""
        
        # Potential-based reward shaping
        prev_potential = self._calculate_potential(state)
        curr_potential = self._calculate_potential(next_state)
        shaped_reward = reward + (curr_potential - prev_potential)
        
        # Clip extreme rewards
        shaped_reward = np.clip(shaped_reward, -10.0, 20.0)
        
        return shaped_reward
        
    def _calculate_potential(self, state: Dict) -> float:
        """Calculate potential function for reward shaping"""
        potential = 0.0
        
        # Potential based on network penetration
        compromised_ratio = len(state.get("compromised_hosts", [])) / max(1, state.get("network_size", 20))
        potential += compromised_ratio * 2.0
        
        # Potential based on privilege levels
        hosts = state.get("hosts", {})
        admin_hosts = sum(1 for h in hosts.values() if h.get("access_level") == "admin")
        root_hosts = sum(1 for h in hosts.values() if h.get("access_level") == "root")
        potential += (admin_hosts * 0.5 + root_hosts * 1.0)
        
        # Potential based on objective progress
        obj_completed = len(state.get("objectives_completed", []))
        potential += obj_completed * 3.0
        
        return potential
        
    def _adapt_weights(self):
        """Adapt reward weights based on learning progress"""
        if len(self.reward_history) < 100:
            return
            
        # Analyze recent performance
        recent_rewards = self.reward_history[-100:]
        avg_recent_reward = np.mean(recent_rewards)
        
        # Analyze component contributions
        for reward_func in self.reward_functions:
            component_name = reward_func.name
            recent_component_rewards = self.component_rewards[component_name][-100:]
            
            if len(recent_component_rewards) > 0:
                avg_component = np.mean(recent_component_rewards)
                component_variance = np.var(recent_component_rewards)
                
                # Increase weight for components with positive contribution and low variance
                if avg_component > 0 and component_variance < 1.0:
                    reward_func.weight *= 1.02
                # Decrease weight for consistently negative components
                elif avg_component < -0.1:
                    reward_func.weight *= 0.98
                    
                # Ensure weights stay within reasonable bounds
                reward_func.weight = np.clip(reward_func.weight, 0.1, 5.0)
                
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward components"""
        if not self.reward_history:
            return {"status": "no_data"}
            
        stats = {
            "total_episodes": len(self.reward_history),
            "mean_total_reward": np.mean(self.reward_history),
            "std_total_reward": np.std(self.reward_history),
            "min_total_reward": np.min(self.reward_history),
            "max_total_reward": np.max(self.reward_history),
            "component_stats": {}
        }
        
        # Component statistics
        for name, rewards in self.component_rewards.items():
            if rewards:
                stats["component_stats"][name] = {
                    "mean": np.mean(rewards),
                    "std": np.std(rewards),
                    "contribution": np.mean(rewards) / max(abs(np.mean(self.reward_history)), 1e-6)
                }
                
        return stats
        
    def reset_statistics(self):
        """Reset reward tracking statistics"""
        self.reward_history = []
        self.component_rewards = {rf.name: [] for rf in self.reward_functions}
        
    def update_curriculum_stage(self, stage: str):
        """Update curriculum stage and adjust weights"""
        self.curriculum_stage = stage
        
        # Get current weights
        current_weights = {rf.name.replace("_reward", ""): rf.weight for rf in self.reward_functions}
        
        # Re-adjust for new curriculum stage
        self._adjust_weights_for_curriculum(current_weights)
        
        # Update reward function weights
        weight_map = {rf.name.replace("_reward", ""): rf for rf in self.reward_functions}
        for name, weight in current_weights.items():
            if name in weight_map:
                weight_map[name].weight = weight
                
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "curriculum_stage": self.curriculum_stage,
            "adaptive_weights": self.adaptive_weights,
            "reward_functions": [rf.get_info() for rf in self.reward_functions],
            "num_episodes_tracked": len(self.reward_history)
        }