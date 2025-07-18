"""
Aetherveil Sentinel Reinforcement Learning Agent Module

This module provides comprehensive RL capabilities for cybersecurity scenarios,
including PPO-based agents, custom environments, and multi-agent coordination.
"""

from .rl_agent import RLAgent
from .cybersecurity_env import CybersecurityEnvironment
from .action_spaces import AttackActionSpace
from .reward_functions import TacticalRewardFunction
from .memory_manager import ExperienceReplayManager
from .curriculum_learning import CurriculumManager
from .multi_agent_coordinator import MultiAgentCoordinator
from .model_manager import ModelCheckpointManager
from .training_monitor import TrainingMonitor
from .online_learner import OnlineLearner

__all__ = [
    'RLAgent',
    'CybersecurityEnvironment',
    'AttackActionSpace',
    'TacticalRewardFunction',
    'ExperienceReplayManager',
    'CurriculumManager',
    'MultiAgentCoordinator',
    'ModelCheckpointManager',
    'TrainingMonitor',
    'OnlineLearner'
]