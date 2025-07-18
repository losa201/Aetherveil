"""
Enhanced RL System with Continual Learning and Adversarial Training
Implements advanced reinforcement learning with self-play, meta-learning, and adversarial scenarios
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import random
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor
import optuna
from collections import deque, defaultdict
import time
import copy

logger = logging.getLogger(__name__)


@dataclass
class AdversarialScenario:
    """Adversarial training scenario definition"""
    scenario_id: str
    name: str
    description: str
    defender_config: Dict[str, Any]
    attacker_config: Dict[str, Any]
    success_criteria: Dict[str, Any]
    difficulty_level: float
    reward_shaping: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "defender_config": self.defender_config,
            "attacker_config": self.attacker_config,
            "success_criteria": self.success_criteria,
            "difficulty_level": self.difficulty_level,
            "reward_shaping": self.reward_shaping
        }


@dataclass
class MetaLearningConfig:
    """Meta-learning configuration"""
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    meta_batch_size: int = 4
    inner_steps: int = 5
    adaptation_steps: int = 10
    support_set_size: int = 100
    query_set_size: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inner_lr": self.inner_lr,
            "outer_lr": self.outer_lr,
            "meta_batch_size": self.meta_batch_size,
            "inner_steps": self.inner_steps,
            "adaptation_steps": self.adaptation_steps,
            "support_set_size": self.support_set_size,
            "query_set_size": self.query_set_size
        }


class CybersecurityEnvironment(gym.Env):
    """Enhanced cybersecurity environment with adversarial scenarios"""
    
    def __init__(self, scenario: AdversarialScenario, config: Dict[str, Any] = None):
        super().__init__()
        self.scenario = scenario
        self.config = config or {}
        
        # Environment state
        self.current_step = 0
        self.max_steps = self.config.get("max_steps", 200)
        self.network_topology = self._generate_network_topology()
        self.defender_state = self._initialize_defender_state()
        self.attacker_state = self._initialize_attacker_state()
        
        # Action and observation spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        
        # Reward tracking
        self.reward_history = deque(maxlen=1000)
        self.episode_rewards = []
        self.success_rate = 0.0
        
        # Adaptive difficulty
        self.difficulty_controller = DifficultyController(scenario.difficulty_level)
        
        logger.info(f"Initialized cybersecurity environment: {scenario.name}")
    
    def _generate_network_topology(self) -> Dict[str, Any]:
        """Generate realistic network topology"""
        topology = {
            "nodes": [],
            "edges": [],
            "subnets": [],
            "critical_assets": []
        }
        
        # Generate nodes (hosts, servers, network devices)
        num_nodes = random.randint(10, 50)
        for i in range(num_nodes):
            node = {
                "id": f"node_{i}",
                "type": random.choice(["workstation", "server", "router", "switch", "firewall"]),
                "ip": f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}",
                "os": random.choice(["Windows", "Linux", "MacOS", "FreeBSD"]),
                "services": self._generate_services(),
                "vulnerabilities": self._generate_vulnerabilities(),
                "security_level": random.uniform(0.3, 0.9),
                "is_compromised": False,
                "detection_capability": random.uniform(0.1, 0.8)
            }
            topology["nodes"].append(node)
        
        # Generate network connections
        for i in range(len(topology["nodes"])):
            for j in range(i + 1, len(topology["nodes"])):
                if random.random() < 0.3:  # 30% connection probability
                    topology["edges"].append({
                        "source": topology["nodes"][i]["id"],
                        "target": topology["nodes"][j]["id"],
                        "weight": random.uniform(0.1, 1.0)
                    })
        
        # Identify critical assets
        critical_count = max(1, num_nodes // 10)
        critical_nodes = random.sample(topology["nodes"], critical_count)
        for node in critical_nodes:
            node["is_critical"] = True
            topology["critical_assets"].append(node["id"])
        
        return topology
    
    def _generate_services(self) -> List[Dict[str, Any]]:
        """Generate services running on a node"""
        common_services = [
            {"name": "ssh", "port": 22, "version": "OpenSSH 7.4"},
            {"name": "http", "port": 80, "version": "Apache 2.4.6"},
            {"name": "https", "port": 443, "version": "Apache 2.4.6"},
            {"name": "ftp", "port": 21, "version": "vsftpd 3.0.2"},
            {"name": "smtp", "port": 25, "version": "Postfix 3.1.1"},
            {"name": "dns", "port": 53, "version": "BIND 9.11.4"},
            {"name": "mysql", "port": 3306, "version": "MySQL 5.7.25"},
            {"name": "rdp", "port": 3389, "version": "Windows RDP"}
        ]
        
        num_services = random.randint(1, 5)
        return random.sample(common_services, num_services)
    
    def _generate_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Generate vulnerabilities for a node"""
        common_vulns = [
            {"cve": "CVE-2021-34527", "severity": "critical", "exploitable": True},
            {"cve": "CVE-2021-26855", "severity": "critical", "exploitable": True},
            {"cve": "CVE-2020-1472", "severity": "critical", "exploitable": True},
            {"cve": "CVE-2019-0708", "severity": "critical", "exploitable": True},
            {"cve": "CVE-2018-7600", "severity": "critical", "exploitable": True}
        ]
        
        num_vulns = random.randint(0, 3)
        return random.sample(common_vulns, num_vulns)
    
    def _initialize_defender_state(self) -> Dict[str, Any]:
        """Initialize defender state"""
        return {
            "detection_systems": {
                "ids": {"active": True, "accuracy": 0.7, "false_positive_rate": 0.1},
                "antivirus": {"active": True, "detection_rate": 0.8, "updated": True},
                "firewall": {"active": True, "rules": 50, "effectiveness": 0.6},
                "siem": {"active": True, "log_correlation": 0.8, "alert_threshold": 0.7}
            },
            "response_capabilities": {
                "incident_response_time": 15,  # minutes
                "quarantine_capability": True,
                "backup_recovery": True,
                "network_isolation": True
            },
            "awareness_level": random.uniform(0.5, 0.9),
            "budget": 100000,
            "threat_intelligence": {
                "feeds": ["commercial", "open_source"],
                "accuracy": 0.8,
                "coverage": 0.7
            }
        }
    
    def _initialize_attacker_state(self) -> Dict[str, Any]:
        """Initialize attacker state"""
        return {
            "reconnaissance_data": {},
            "compromised_hosts": [],
            "persistence_mechanisms": [],
            "lateral_movement_paths": [],
            "exfiltration_channels": [],
            "techniques_used": [],
            "detection_risk": 0.0,
            "capabilities": {
                "sophistication": random.uniform(0.3, 0.9),
                "resources": random.uniform(0.2, 0.8),
                "patience": random.uniform(0.4, 0.9),
                "stealth": random.uniform(0.3, 0.8)
            }
        }
    
    def _create_action_space(self) -> spaces.Space:
        """Create action space for the environment"""
        # Multi-discrete action space for different attack categories
        return spaces.MultiDiscrete([
            10,  # Reconnaissance actions
            8,   # Initial access actions
            6,   # Persistence actions
            7,   # Privilege escalation actions
            5,   # Defense evasion actions
            8,   # Lateral movement actions
            4,   # Exfiltration actions
            3    # Impact actions
        ])
    
    def _create_observation_space(self) -> spaces.Space:
        """Create observation space for the environment"""
        # High-dimensional observation space
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(256,),  # Network state + attacker state + defender state
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.network_topology = self._generate_network_topology()
        self.defender_state = self._initialize_defender_state()
        self.attacker_state = self._initialize_attacker_state()
        
        # Reset difficulty if needed
        self.difficulty_controller.reset()
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step"""
        self.current_step += 1
        
        # Execute attacker action
        attack_result = self._execute_attack_action(action)
        
        # Execute defender response
        defender_result = self._execute_defender_response(attack_result)
        
        # Calculate reward
        reward = self._calculate_reward(attack_result, defender_result)
        
        # Check if episode is done
        done = self._check_episode_done()
        
        # Update states
        self._update_states(attack_result, defender_result)
        
        # Adapt difficulty
        self.difficulty_controller.update(reward, done)
        
        # Create info dict
        info = {
            "attack_result": attack_result,
            "defender_result": defender_result,
            "detection_risk": self.attacker_state["detection_risk"],
            "compromised_hosts": len(self.attacker_state["compromised_hosts"]),
            "episode_step": self.current_step,
            "difficulty_level": self.difficulty_controller.current_difficulty
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_attack_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Execute attack action and return results"""
        attack_categories = [
            "reconnaissance", "initial_access", "persistence",
            "privilege_escalation", "defense_evasion", "lateral_movement",
            "exfiltration", "impact"
        ]
        
        results = {}
        
        for i, category in enumerate(attack_categories):
            action_id = action[i]
            if action_id > 0:  # 0 is no-op
                result = self._execute_specific_attack(category, action_id)
                results[category] = result
        
        return results
    
    def _execute_specific_attack(self, category: str, action_id: int) -> Dict[str, Any]:
        """Execute specific attack based on category and action ID"""
        success_prob = self._calculate_success_probability(category, action_id)
        success = random.random() < success_prob
        
        result = {
            "category": category,
            "action_id": action_id,
            "success": success,
            "detection_risk": self._calculate_detection_risk(category, action_id),
            "impact": self._calculate_impact(category, action_id, success)
        }
        
        if success:
            self._apply_attack_effects(category, action_id, result)
        
        return result
    
    def _calculate_success_probability(self, category: str, action_id: int) -> float:
        """Calculate probability of attack success"""
        base_prob = 0.3  # Base success probability
        
        # Modify based on attacker capabilities
        capability_bonus = self.attacker_state["capabilities"]["sophistication"] * 0.3
        
        # Modify based on defender strength
        defender_penalty = self.defender_state["awareness_level"] * 0.2
        
        # Modify based on network security
        avg_security = np.mean([node["security_level"] for node in self.network_topology["nodes"]])
        security_penalty = avg_security * 0.2
        
        # Modify based on previous detections
        detection_penalty = self.attacker_state["detection_risk"] * 0.3
        
        final_prob = base_prob + capability_bonus - defender_penalty - security_penalty - detection_penalty
        
        return max(0.05, min(0.95, final_prob))
    
    def _calculate_detection_risk(self, category: str, action_id: int) -> float:
        """Calculate detection risk for an action"""
        base_risk = 0.1
        
        # Different attack types have different detection risks
        risk_multipliers = {
            "reconnaissance": 0.2,
            "initial_access": 0.4,
            "persistence": 0.3,
            "privilege_escalation": 0.6,
            "defense_evasion": 0.1,
            "lateral_movement": 0.5,
            "exfiltration": 0.8,
            "impact": 0.9
        }
        
        category_risk = base_risk * risk_multipliers.get(category, 0.5)
        
        # Modify based on defender capabilities
        ids_effectiveness = self.defender_state["detection_systems"]["ids"]["accuracy"]
        siem_effectiveness = self.defender_state["detection_systems"]["siem"]["log_correlation"]
        
        detection_capability = (ids_effectiveness + siem_effectiveness) / 2
        
        return min(1.0, category_risk * detection_capability)
    
    def _calculate_impact(self, category: str, action_id: int, success: bool) -> float:
        """Calculate impact of an attack action"""
        if not success:
            return 0.0
        
        # Different attack types have different impact levels
        impact_values = {
            "reconnaissance": 0.1,
            "initial_access": 0.3,
            "persistence": 0.2,
            "privilege_escalation": 0.4,
            "defense_evasion": 0.1,
            "lateral_movement": 0.3,
            "exfiltration": 0.8,
            "impact": 1.0
        }
        
        return impact_values.get(category, 0.2)
    
    def _apply_attack_effects(self, category: str, action_id: int, result: Dict[str, Any]):
        """Apply effects of successful attack"""
        if category == "reconnaissance":
            # Gather information about network
            target_nodes = random.sample(self.network_topology["nodes"], 
                                       min(3, len(self.network_topology["nodes"])))
            for node in target_nodes:
                self.attacker_state["reconnaissance_data"][node["id"]] = {
                    "services": node["services"],
                    "os": node["os"],
                    "vulnerabilities": node["vulnerabilities"]
                }
        
        elif category == "initial_access":
            # Compromise a host
            available_targets = [node for node in self.network_topology["nodes"] 
                               if not node["is_compromised"]]
            if available_targets:
                target = random.choice(available_targets)
                target["is_compromised"] = True
                self.attacker_state["compromised_hosts"].append(target["id"])
        
        elif category == "persistence":
            # Establish persistence on compromised hosts
            if self.attacker_state["compromised_hosts"]:
                host_id = random.choice(self.attacker_state["compromised_hosts"])
                self.attacker_state["persistence_mechanisms"].append({
                    "host_id": host_id,
                    "mechanism": random.choice(["scheduled_task", "registry_key", "service", "startup_folder"])
                })
        
        elif category == "lateral_movement":
            # Move to adjacent hosts
            if self.attacker_state["compromised_hosts"]:
                source_host = random.choice(self.attacker_state["compromised_hosts"])
                # Find connected hosts
                connected_hosts = [edge["target"] for edge in self.network_topology["edges"] 
                                 if edge["source"] == source_host]
                connected_hosts.extend([edge["source"] for edge in self.network_topology["edges"] 
                                      if edge["target"] == source_host])
                
                available_targets = [host_id for host_id in connected_hosts 
                                   if host_id not in self.attacker_state["compromised_hosts"]]
                
                if available_targets:
                    new_target = random.choice(available_targets)
                    # Find the actual node and compromise it
                    for node in self.network_topology["nodes"]:
                        if node["id"] == new_target:
                            node["is_compromised"] = True
                            self.attacker_state["compromised_hosts"].append(new_target)
                            break
        
        # Update detection risk
        self.attacker_state["detection_risk"] += result["detection_risk"]
        self.attacker_state["detection_risk"] = min(1.0, self.attacker_state["detection_risk"])
        
        # Track techniques used
        self.attacker_state["techniques_used"].append({
            "category": category,
            "action_id": action_id,
            "timestamp": self.current_step
        })
    
    def _execute_defender_response(self, attack_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute defender response to attacks"""
        response_actions = []
        
        # Detect attacks based on detection systems
        for category, result in attack_result.items():
            if result["success"]:
                detection_prob = self._calculate_detection_probability(result)
                if random.random() < detection_prob:
                    response_actions.append(self._generate_response_action(category, result))
        
        # Execute response actions
        response_results = []
        for action in response_actions:
            result = self._execute_response_action(action)
            response_results.append(result)
        
        return {
            "detected_attacks": len(response_actions),
            "response_actions": response_results,
            "total_detections": len(response_actions)
        }
    
    def _calculate_detection_probability(self, attack_result: Dict[str, Any]) -> float:
        """Calculate probability of detecting an attack"""
        base_detection = 0.3
        
        # Modify based on detection systems
        ids_effectiveness = self.defender_state["detection_systems"]["ids"]["accuracy"]
        siem_effectiveness = self.defender_state["detection_systems"]["siem"]["log_correlation"]
        
        detection_capability = (ids_effectiveness + siem_effectiveness) / 2
        
        # Modify based on attack detection risk
        detection_risk = attack_result["detection_risk"]
        
        final_prob = base_detection + detection_capability * detection_risk
        
        return min(0.95, max(0.05, final_prob))
    
    def _generate_response_action(self, attack_category: str, attack_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate response action"""
        response_types = {
            "reconnaissance": "monitor",
            "initial_access": "quarantine",
            "persistence": "cleanup",
            "privilege_escalation": "isolate",
            "defense_evasion": "update_rules",
            "lateral_movement": "segment_network",
            "exfiltration": "block_communication",
            "impact": "emergency_response"
        }
        
        return {
            "type": response_types.get(attack_category, "generic_response"),
            "attack_category": attack_category,
            "urgency": self._calculate_response_urgency(attack_result),
            "resources_required": random.randint(1, 5)
        }
    
    def _calculate_response_urgency(self, attack_result: Dict[str, Any]) -> str:
        """Calculate urgency of response"""
        impact = attack_result["impact"]
        
        if impact >= 0.8:
            return "critical"
        elif impact >= 0.5:
            return "high"
        elif impact >= 0.3:
            return "medium"
        else:
            return "low"
    
    def _execute_response_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute defender response action"""
        success_prob = 0.7  # Base success probability for responses
        
        # Modify based on response time and resources
        if action["urgency"] == "critical":
            success_prob += 0.1
        
        success = random.random() < success_prob
        
        result = {
            "action_type": action["type"],
            "success": success,
            "effectiveness": random.uniform(0.3, 0.9) if success else 0.0
        }
        
        if success:
            self._apply_response_effects(action, result)
        
        return result
    
    def _apply_response_effects(self, action: Dict[str, Any], result: Dict[str, Any]):
        """Apply effects of successful response"""
        if action["type"] == "quarantine":
            # Reduce attacker capabilities
            self.attacker_state["detection_risk"] += 0.2
        
        elif action["type"] == "cleanup":
            # Remove some persistence mechanisms
            if self.attacker_state["persistence_mechanisms"]:
                removed_count = min(2, len(self.attacker_state["persistence_mechanisms"]))
                for _ in range(removed_count):
                    self.attacker_state["persistence_mechanisms"].pop()
        
        elif action["type"] == "isolate":
            # Isolate compromised hosts
            if self.attacker_state["compromised_hosts"]:
                # Remove some compromised hosts
                removed_count = min(1, len(self.attacker_state["compromised_hosts"]))
                for _ in range(removed_count):
                    host_id = self.attacker_state["compromised_hosts"].pop()
                    # Mark node as no longer compromised
                    for node in self.network_topology["nodes"]:
                        if node["id"] == host_id:
                            node["is_compromised"] = False
                            break
        
        # Improve defender capabilities
        self.defender_state["awareness_level"] = min(1.0, 
                                                   self.defender_state["awareness_level"] + 0.05)
    
    def _calculate_reward(self, attack_result: Dict[str, Any], 
                         defender_result: Dict[str, Any]) -> float:
        """Calculate reward for the current step"""
        reward = 0.0
        
        # Reward for successful attacks
        for category, result in attack_result.items():
            if result["success"]:
                reward += result["impact"] * 10
        
        # Penalty for detection
        detection_penalty = defender_result["detected_attacks"] * 5
        reward -= detection_penalty
        
        # Bonus for achieving objectives
        if self._check_objective_completion():
            reward += 50
        
        # Penalty for high detection risk
        if self.attacker_state["detection_risk"] > 0.8:
            reward -= 20
        
        # Bonus for stealth
        if self.attacker_state["detection_risk"] < 0.3:
            reward += 5
        
        # Apply reward shaping from scenario
        reward = self._apply_reward_shaping(reward)
        
        return reward
    
    def _apply_reward_shaping(self, base_reward: float) -> float:
        """Apply scenario-specific reward shaping"""
        shaping_config = self.scenario.reward_shaping
        
        # Apply multipliers
        if "success_multiplier" in shaping_config:
            if self._check_objective_completion():
                base_reward *= shaping_config["success_multiplier"]
        
        if "stealth_bonus" in shaping_config:
            if self.attacker_state["detection_risk"] < 0.2:
                base_reward += shaping_config["stealth_bonus"]
        
        if "time_penalty" in shaping_config:
            time_penalty = (self.current_step / self.max_steps) * shaping_config["time_penalty"]
            base_reward -= time_penalty
        
        return base_reward
    
    def _check_objective_completion(self) -> bool:
        """Check if attack objectives are completed"""
        success_criteria = self.scenario.success_criteria
        
        # Check critical asset compromise
        if "critical_assets_compromised" in success_criteria:
            required_count = success_criteria["critical_assets_compromised"]
            compromised_critical = sum(1 for asset_id in self.network_topology["critical_assets"] 
                                     if asset_id in self.attacker_state["compromised_hosts"])
            if compromised_critical >= required_count:
                return True
        
        # Check total compromise percentage
        if "compromise_percentage" in success_criteria:
            required_percentage = success_criteria["compromise_percentage"]
            current_percentage = len(self.attacker_state["compromised_hosts"]) / len(self.network_topology["nodes"])
            if current_percentage >= required_percentage:
                return True
        
        # Check data exfiltration
        if "data_exfiltrated" in success_criteria:
            if len(self.attacker_state["exfiltration_channels"]) > 0:
                return True
        
        return False
    
    def _check_episode_done(self) -> bool:
        """Check if episode should end"""
        # Time limit reached
        if self.current_step >= self.max_steps:
            return True
        
        # Objective completed
        if self._check_objective_completion():
            return True
        
        # Detected and contained
        if self.attacker_state["detection_risk"] >= 1.0:
            return True
        
        return False
    
    def _update_states(self, attack_result: Dict[str, Any], defender_result: Dict[str, Any]):
        """Update environment states"""
        # Decay detection risk over time
        self.attacker_state["detection_risk"] *= 0.95
        
        # Update network security based on defender actions
        if defender_result["detected_attacks"] > 0:
            for node in self.network_topology["nodes"]:
                node["security_level"] = min(1.0, node["security_level"] + 0.02)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Create observation vector
        obs = np.zeros(256, dtype=np.float32)
        
        # Network topology features (0-50)
        obs[0] = len(self.network_topology["nodes"]) / 100.0  # Normalize
        obs[1] = len(self.network_topology["edges"]) / 200.0
        obs[2] = len(self.network_topology["critical_assets"]) / 10.0
        
        # Attacker state features (51-100)
        obs[51] = len(self.attacker_state["compromised_hosts"]) / len(self.network_topology["nodes"])
        obs[52] = len(self.attacker_state["persistence_mechanisms"]) / 10.0
        obs[53] = self.attacker_state["detection_risk"]
        obs[54] = self.attacker_state["capabilities"]["sophistication"]
        obs[55] = self.attacker_state["capabilities"]["resources"]
        
        # Defender state features (101-150)
        obs[101] = self.defender_state["detection_systems"]["ids"]["accuracy"]
        obs[102] = self.defender_state["detection_systems"]["siem"]["log_correlation"]
        obs[103] = self.defender_state["awareness_level"]
        obs[104] = self.defender_state["budget"] / 100000.0
        
        # Network state features (151-200)
        avg_security = np.mean([node["security_level"] for node in self.network_topology["nodes"]])
        obs[151] = avg_security
        
        compromised_ratio = len(self.attacker_state["compromised_hosts"]) / len(self.network_topology["nodes"])
        obs[152] = compromised_ratio
        
        # Temporal features (201-255)
        obs[201] = self.current_step / self.max_steps
        obs[202] = self.difficulty_controller.current_difficulty
        
        return obs
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Compromised hosts: {len(self.attacker_state['compromised_hosts'])}")
            print(f"Detection risk: {self.attacker_state['detection_risk']:.3f}")
            print(f"Objective completed: {self._check_objective_completion()}")
            print("---")


class DifficultyController:
    """Adaptive difficulty controller"""
    
    def __init__(self, initial_difficulty: float = 0.5):
        self.initial_difficulty = initial_difficulty
        self.current_difficulty = initial_difficulty
        self.performance_history = deque(maxlen=100)
        self.adaptation_rate = 0.01
        self.target_success_rate = 0.6
        
    def update(self, reward: float, episode_done: bool):
        """Update difficulty based on performance"""
        if episode_done:
            # Determine if episode was successful
            success = reward > 0
            self.performance_history.append(success)
            
            if len(self.performance_history) >= 10:
                recent_success_rate = np.mean(list(self.performance_history)[-10:])
                
                # Adjust difficulty
                if recent_success_rate > self.target_success_rate + 0.1:
                    # Too easy, increase difficulty
                    self.current_difficulty = min(1.0, self.current_difficulty + self.adaptation_rate)
                elif recent_success_rate < self.target_success_rate - 0.1:
                    # Too hard, decrease difficulty
                    self.current_difficulty = max(0.1, self.current_difficulty - self.adaptation_rate)
    
    def reset(self):
        """Reset difficulty controller"""
        self.current_difficulty = self.initial_difficulty
    
    def get_difficulty(self) -> float:
        """Get current difficulty level"""
        return self.current_difficulty


class MetaLearningAgent:
    """Meta-learning agent using Model-Agnostic Meta-Learning (MAML)"""
    
    def __init__(self, policy_network: nn.Module, config: MetaLearningConfig):
        self.policy_network = policy_network
        self.config = config
        self.meta_optimizer = optim.Adam(policy_network.parameters(), lr=config.outer_lr)
        self.task_distributions = []
        self.adaptation_history = []
        
    def meta_train(self, task_batch: List[AdversarialScenario]) -> Dict[str, float]:
        """Perform meta-training on a batch of tasks"""
        meta_loss = 0.0
        
        for task in task_batch:
            # Create task-specific environment
            env = CybersecurityEnvironment(task)
            
            # Collect support set
            support_trajectories = self._collect_trajectories(env, self.config.support_set_size)
            
            # Adapt policy using support set
            adapted_policy = self._adapt_policy(support_trajectories)
            
            # Collect query set with adapted policy
            query_trajectories = self._collect_trajectories(env, self.config.query_set_size, adapted_policy)
            
            # Compute meta-loss
            task_loss = self._compute_loss(query_trajectories)
            meta_loss += task_loss
        
        # Update meta-parameters
        meta_loss /= len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {
            "meta_loss": meta_loss.item(),
            "num_tasks": len(task_batch)
        }
    
    def _collect_trajectories(self, env: CybersecurityEnvironment, num_samples: int, 
                            policy: Optional[nn.Module] = None) -> List[Dict[str, Any]]:
        """Collect trajectory samples from environment"""
        if policy is None:
            policy = self.policy_network
        
        trajectories = []
        
        for _ in range(num_samples):
            obs = env.reset()
            trajectory = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": []
            }
            
            done = False
            while not done:
                # Get action from policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_probs = policy(obs_tensor)
                    action = torch.multinomial(action_probs, 1).item()
                
                trajectory["observations"].append(obs)
                trajectory["actions"].append(action)
                
                obs, reward, done, _ = env.step(action)
                
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(done)
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _adapt_policy(self, support_trajectories: List[Dict[str, Any]]) -> nn.Module:
        """Adapt policy using support set"""
        # Create a copy of the policy for adaptation
        adapted_policy = copy.deepcopy(self.policy_network)
        
        # Create inner optimizer
        inner_optimizer = optim.SGD(adapted_policy.parameters(), lr=self.config.inner_lr)
        
        # Perform inner loop updates
        for _ in range(self.config.inner_steps):
            loss = self._compute_loss(support_trajectories, adapted_policy)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_policy
    
    def _compute_loss(self, trajectories: List[Dict[str, Any]], 
                     policy: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute policy loss from trajectories"""
        if policy is None:
            policy = self.policy_network
        
        total_loss = 0.0
        
        for trajectory in trajectories:
            observations = torch.FloatTensor(trajectory["observations"])
            actions = torch.LongTensor(trajectory["actions"])
            rewards = torch.FloatTensor(trajectory["rewards"])
            
            # Compute returns
            returns = self._compute_returns(rewards)
            
            # Compute policy loss
            action_probs = policy(observations)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1))
            
            # Policy gradient loss
            loss = -(log_probs * returns).mean()
            total_loss += loss
        
        return total_loss / len(trajectories)
    
    def _compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def adapt_to_new_task(self, task: AdversarialScenario) -> nn.Module:
        """Quickly adapt to a new task"""
        env = CybersecurityEnvironment(task)
        
        # Collect small support set
        support_trajectories = self._collect_trajectories(env, self.config.adaptation_steps)
        
        # Adapt policy
        adapted_policy = self._adapt_policy(support_trajectories)
        
        return adapted_policy


class SelfPlayTrainer:
    """Self-play trainer for adversarial scenarios"""
    
    def __init__(self, attacker_policy: nn.Module, defender_policy: nn.Module):
        self.attacker_policy = attacker_policy
        self.defender_policy = defender_policy
        self.training_history = []
        self.policy_pool = []
        
    def train_self_play(self, num_episodes: int, scenario: AdversarialScenario) -> Dict[str, Any]:
        """Train using self-play"""
        results = {
            "episodes": num_episodes,
            "attacker_wins": 0,
            "defender_wins": 0,
            "average_episode_length": 0.0,
            "learning_curves": []
        }
        
        episode_lengths = []
        
        for episode in range(num_episodes):
            # Create environment
            env = CybersecurityEnvironment(scenario)
            
            # Run episode
            episode_result = self._run_self_play_episode(env)
            
            # Update results
            if episode_result["attacker_won"]:
                results["attacker_wins"] += 1
            else:
                results["defender_wins"] += 1
            
            episode_lengths.append(episode_result["episode_length"])
            
            # Update policies
            self._update_policies(episode_result)
            
            # Add to policy pool occasionally
            if episode % 100 == 0:
                self._add_to_policy_pool()
            
            # Log progress
            if episode % 50 == 0:
                avg_length = np.mean(episode_lengths[-50:])
                results["learning_curves"].append({
                    "episode": episode,
                    "average_length": avg_length,
                    "attacker_win_rate": results["attacker_wins"] / (episode + 1)
                })
        
        results["average_episode_length"] = np.mean(episode_lengths)
        
        return results
    
    def _run_self_play_episode(self, env: CybersecurityEnvironment) -> Dict[str, Any]:
        """Run a single self-play episode"""
        obs = env.reset()
        done = False
        episode_length = 0
        attacker_rewards = []
        defender_rewards = []
        
        while not done:
            # Attacker action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                attacker_action_probs = self.attacker_policy(obs_tensor)
                attacker_action = torch.multinomial(attacker_action_probs, 1).item()
            
            # Execute action
            obs, reward, done, info = env.step(attacker_action)
            
            # Calculate defender reward (opposite of attacker)
            defender_reward = -reward
            
            attacker_rewards.append(reward)
            defender_rewards.append(defender_reward)
            episode_length += 1
        
        # Determine winner
        total_attacker_reward = sum(attacker_rewards)
        attacker_won = total_attacker_reward > 0
        
        return {
            "episode_length": episode_length,
            "attacker_won": attacker_won,
            "attacker_rewards": attacker_rewards,
            "defender_rewards": defender_rewards,
            "total_attacker_reward": total_attacker_reward
        }
    
    def _update_policies(self, episode_result: Dict[str, Any]):
        """Update policies based on episode results"""
        # Simple policy update (in practice, would use more sophisticated methods)
        attacker_rewards = torch.FloatTensor(episode_result["attacker_rewards"])
        defender_rewards = torch.FloatTensor(episode_result["defender_rewards"])
        
        # Update attacker policy
        attacker_optimizer = optim.Adam(self.attacker_policy.parameters(), lr=0.001)
        attacker_loss = -attacker_rewards.mean()  # Maximize rewards
        
        attacker_optimizer.zero_grad()
        attacker_loss.backward()
        attacker_optimizer.step()
        
        # Update defender policy
        defender_optimizer = optim.Adam(self.defender_policy.parameters(), lr=0.001)
        defender_loss = -defender_rewards.mean()  # Maximize rewards
        
        defender_optimizer.zero_grad()
        defender_loss.backward()
        defender_optimizer.step()
    
    def _add_to_policy_pool(self):
        """Add current policies to the policy pool"""
        self.policy_pool.append({
            "attacker_policy": copy.deepcopy(self.attacker_policy),
            "defender_policy": copy.deepcopy(self.defender_policy),
            "timestamp": datetime.now()
        })
        
        # Keep only recent policies
        if len(self.policy_pool) > 10:
            self.policy_pool.pop(0)


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, scenario: AdversarialScenario):
        self.scenario = scenario
        self.study = optuna.create_study(direction='maximize')
        self.best_params = None
        
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters"""
        self.study.optimize(self._objective, n_trials=n_trials)
        
        self.best_params = self.study.best_params
        
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": n_trials
        }
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        # Suggest hyperparameters
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
        entropy_coef = trial.suggest_loguniform('entropy_coef', 1e-8, 1e-1)
        
        # Create environment
        env = CybersecurityEnvironment(self.scenario)
        
        # Create and train model
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef=entropy_coef,
            verbose=0
        )
        
        # Train for a short period
        model.learn(total_timesteps=10000)
        
        # Evaluate
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        
        return total_reward


class EnhancedRLSystem:
    """Enhanced RL system with continual learning and adversarial training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scenarios = []
        self.meta_learner = None
        self.self_play_trainer = None
        self.hyperparameter_optimizer = None
        self.policy_networks = {}
        self.training_history = []
        
        # Initialize components
        self._initialize_scenarios()
        self._initialize_meta_learner()
        self._initialize_self_play()
        
    def _initialize_scenarios(self):
        """Initialize adversarial scenarios"""
        # Basic penetration testing scenario
        basic_scenario = AdversarialScenario(
            scenario_id="basic_pentest",
            name="Basic Penetration Test",
            description="Standard penetration testing scenario",
            defender_config={"awareness_level": 0.5, "budget": 50000},
            attacker_config={"sophistication": 0.6, "resources": 0.5},
            success_criteria={"compromise_percentage": 0.3},
            difficulty_level=0.5,
            reward_shaping={"success_multiplier": 2.0, "stealth_bonus": 10.0}
        )
        
        # Advanced persistent threat scenario
        apt_scenario = AdversarialScenario(
            scenario_id="apt_campaign",
            name="Advanced Persistent Threat",
            description="Long-term stealthy attack campaign",
            defender_config={"awareness_level": 0.8, "budget": 100000},
            attacker_config={"sophistication": 0.9, "resources": 0.8, "patience": 0.9},
            success_criteria={"critical_assets_compromised": 1, "data_exfiltrated": True},
            difficulty_level=0.8,
            reward_shaping={"success_multiplier": 3.0, "stealth_bonus": 20.0, "time_penalty": 0.1}
        )
        
        # Insider threat scenario
        insider_scenario = AdversarialScenario(
            scenario_id="insider_threat",
            name="Malicious Insider",
            description="Insider threat with legitimate access",
            defender_config={"awareness_level": 0.4, "budget": 75000},
            attacker_config={"sophistication": 0.7, "resources": 0.6, "stealth": 0.9},
            success_criteria={"data_exfiltrated": True, "compromise_percentage": 0.2},
            difficulty_level=0.6,
            reward_shaping={"success_multiplier": 2.5, "stealth_bonus": 15.0}
        )
        
        self.scenarios = [basic_scenario, apt_scenario, insider_scenario]
    
    def _initialize_meta_learner(self):
        """Initialize meta-learning components"""
        # Create policy network
        policy_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 8 action categories
            nn.Softmax(dim=-1)
        )
        
        meta_config = MetaLearningConfig()
        self.meta_learner = MetaLearningAgent(policy_network, meta_config)
    
    def _initialize_self_play(self):
        """Initialize self-play components"""
        # Create attacker and defender policies
        attacker_policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Softmax(dim=-1)
        )
        
        defender_policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Softmax(dim=-1)
        )
        
        self.self_play_trainer = SelfPlayTrainer(attacker_policy, defender_policy)
    
    async def train_continual_learning(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Train with continual learning across scenarios"""
        results = {
            "iterations": num_iterations,
            "scenario_results": {},
            "meta_learning_results": [],
            "adaptation_metrics": []
        }
        
        for iteration in range(num_iterations):
            # Select random scenario
            scenario = random.choice(self.scenarios)
            
            # Meta-learning update
            task_batch = random.sample(self.scenarios, min(4, len(self.scenarios)))
            meta_result = self.meta_learner.meta_train(task_batch)
            results["meta_learning_results"].append(meta_result)
            
            # Test adaptation to new scenario
            adapted_policy = self.meta_learner.adapt_to_new_task(scenario)
            adaptation_result = await self._test_adaptation(adapted_policy, scenario)
            results["adaptation_metrics"].append(adaptation_result)
            
            # Self-play training
            if iteration % 10 == 0:
                self_play_result = self.self_play_trainer.train_self_play(50, scenario)
                results["scenario_results"][scenario.scenario_id] = self_play_result
            
            # Hyperparameter optimization
            if iteration % 50 == 0:
                if self.hyperparameter_optimizer is None:
                    self.hyperparameter_optimizer = HyperparameterOptimizer(scenario)
                
                opt_result = self.hyperparameter_optimizer.optimize(n_trials=20)
                results[f"hyperopt_{iteration}"] = opt_result
        
        return results
    
    async def _test_adaptation(self, policy: nn.Module, 
                             scenario: AdversarialScenario) -> Dict[str, Any]:
        """Test adaptation performance on scenario"""
        env = CybersecurityEnvironment(scenario)
        
        # Run evaluation episodes
        num_episodes = 10
        episode_rewards = []
        
        for _ in range(num_episodes):
            obs = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_probs = policy(obs_tensor)
                    action = torch.multinomial(action_probs, 1).item()
                
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        return {
            "scenario_id": scenario.scenario_id,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "success_rate": sum(1 for r in episode_rewards if r > 0) / len(episode_rewards)
        }
    
    async def adversarial_training(self, scenario: AdversarialScenario, 
                                 num_episodes: int = 1000) -> Dict[str, Any]:
        """Perform adversarial training against defensive LLMs"""
        results = {
            "scenario": scenario.scenario_id,
            "episodes": num_episodes,
            "training_curve": [],
            "final_performance": {}
        }
        
        # Create environment
        env = CybersecurityEnvironment(scenario)
        
        # Initialize adversarial components
        attacker_model = PPO('MlpPolicy', env, verbose=0)
        
        # Training loop with adversarial updates
        for episode in range(num_episodes):
            # Train attacker
            attacker_model.learn(total_timesteps=1000)
            
            # Evaluate against current defender
            eval_result = await self._evaluate_adversarial_performance(attacker_model, env)
            
            # Update defender (simulate adaptive defense)
            if episode % 100 == 0:
                env.defender_state["awareness_level"] = min(1.0, 
                                                          env.defender_state["awareness_level"] + 0.1)
                env.defender_state["detection_systems"]["ids"]["accuracy"] = min(1.0,
                    env.defender_state["detection_systems"]["ids"]["accuracy"] + 0.05)
            
            # Record training progress
            if episode % 50 == 0:
                results["training_curve"].append({
                    "episode": episode,
                    "performance": eval_result,
                    "defender_strength": env.defender_state["awareness_level"]
                })
        
        # Final evaluation
        results["final_performance"] = await self._evaluate_adversarial_performance(attacker_model, env)
        
        return results
    
    async def _evaluate_adversarial_performance(self, model, env) -> Dict[str, Any]:
        """Evaluate adversarial performance"""
        num_eval_episodes = 20
        rewards = []
        success_count = 0
        
        for _ in range(num_eval_episodes):
            obs = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            
            rewards.append(total_reward)
            if total_reward > 0:
                success_count += 1
        
        return {
            "mean_reward": np.mean(rewards),
            "success_rate": success_count / num_eval_episodes,
            "std_reward": np.std(rewards)
        }
    
    async def curriculum_learning(self, difficulty_progression: List[float]) -> Dict[str, Any]:
        """Implement curriculum learning with progressive difficulty"""
        results = {
            "difficulty_levels": difficulty_progression,
            "learning_curve": [],
            "performance_metrics": []
        }
        
        for difficulty in difficulty_progression:
            # Adjust scenario difficulty
            current_scenario = copy.deepcopy(self.scenarios[0])
            current_scenario.difficulty_level = difficulty
            
            # Train on current difficulty
            training_result = await self.adversarial_training(current_scenario, num_episodes=500)
            
            # Evaluate performance
            performance = training_result["final_performance"]
            results["learning_curve"].append({
                "difficulty": difficulty,
                "performance": performance
            })
            
            # Check if ready for next level
            if performance["success_rate"] < 0.6:
                # Spend more time on current difficulty
                additional_training = await self.adversarial_training(current_scenario, num_episodes=300)
                performance = additional_training["final_performance"]
            
            results["performance_metrics"].append(performance)
        
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "scenarios": len(self.scenarios),
            "meta_learner_config": self.meta_learner.config.to_dict() if self.meta_learner else None,
            "policy_pool_size": len(self.self_play_trainer.policy_pool) if self.self_play_trainer else 0,
            "training_history": len(self.training_history),
            "available_scenarios": [s.scenario_id for s in self.scenarios]
        }
    
    async def save_system_state(self, filepath: str):
        """Save system state for continual learning"""
        state = {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "meta_learner_state": self.meta_learner.policy_network.state_dict() if self.meta_learner else None,
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    async def load_system_state(self, filepath: str):
        """Load system state for continual learning"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore scenarios
        self.scenarios = [AdversarialScenario(**s) for s in state["scenarios"]]
        
        # Restore meta-learner
        if state["meta_learner_state"] and self.meta_learner:
            self.meta_learner.policy_network.load_state_dict(state["meta_learner_state"])
        
        # Restore training history
        self.training_history = state["training_history"]
        
        logger.info(f"Loaded system state from {filepath}")