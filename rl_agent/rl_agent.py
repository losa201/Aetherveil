"""
Main RL Agent Implementation with PPO for Cybersecurity Scenarios

This module implements the core reinforcement learning agent using Proximal Policy
Optimization (PPO) for learning optimal attack strategies and tactics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean
import gymnasium as gym
from pathlib import Path
import json
import time
from datetime import datetime


@dataclass
class RLConfig:
    """Configuration for RL Agent"""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    policy_network_arch: List[int] = None
    value_network_arch: List[int] = None
    device: str = "auto"
    normalize_advantage: bool = True
    
    def __post_init__(self):
        if self.policy_network_arch is None:
            self.policy_network_arch = [256, 256]
        if self.value_network_arch is None:
            self.value_network_arch = [256, 256]


class CustomActorCriticPolicy(ActorCriticPolicy):
    """Custom Actor-Critic Policy with cybersecurity-specific features"""
    
    def __init__(self, *args, **kwargs):
        # Add cybersecurity-specific network architectures
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        """Build the custom MLP feature extractor for cybersecurity scenarios"""
        super()._build_mlp_extractor()
        
        # Add attention mechanism for target prioritization
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.features_extractor.features_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Add cybersecurity context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(self.features_extractor.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )


class RL_TrainingCallback(BaseCallback):
    """Custom callback for monitoring RL training progress"""
    
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Called at each step of training"""
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
            
        # Log training metrics at intervals
        if self.n_calls % self.log_interval == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                
                self.logger.record("train/episode_reward_mean", mean_reward)
                self.logger.record("train/episode_length_mean", mean_length)
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.logger.record("train/best_mean_reward", self.best_mean_reward)
                    
        return True


class RLAgent:
    """
    Main Reinforcement Learning Agent for Cybersecurity Scenarios
    
    Uses PPO (Proximal Policy Optimization) to learn optimal attack strategies,
    adapt to different environments, and coordinate with swarm agents.
    """
    
    def __init__(
        self,
        environment: gym.Env,
        config: Optional[RLConfig] = None,
        agent_id: str = "rl_agent_0",
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs"
    ):
        """
        Initialize the RL Agent
        
        Args:
            environment: The cybersecurity environment to train on
            config: Configuration for the RL agent
            agent_id: Unique identifier for this agent
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
        """
        self.config = config or RLConfig()
        self.agent_id = agent_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.environment = environment
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(f"RLAgent_{agent_id}")
        self.logger.setLevel(logging.INFO)
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
            
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize PPO model
        self.model = None
        self.training_stats = {
            'episodes_trained': 0,
            'total_timesteps': 0,
            'best_reward': -np.inf,
            'training_start_time': None,
            'last_checkpoint': None
        }
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the PPO model with custom policy"""
        # Wrap environment for stable-baselines3
        env = Monitor(self.environment, self.log_dir / f"monitor_{self.agent_id}.csv")
        vec_env = DummyVecEnv([lambda: env])
        
        # Policy kwargs for custom architecture
        policy_kwargs = {
            "net_arch": {
                "pi": self.config.policy_network_arch,
                "vf": self.config.value_network_arch
            },
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True,
            "use_sde": False,
            "sde_sample_freq": -1,
            "features_extractor_class": None,
            "features_extractor_kwargs": {}
        }
        
        # Initialize PPO model
        self.model = PPO(
            policy=CustomActorCriticPolicy,
            env=vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            target_kl=self.config.target_kl,
            tensorboard_log=str(self.log_dir),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device
        )
        
        self.logger.info("PPO model initialized successfully")
        
    def train(
        self,
        total_timesteps: int,
        callback_interval: int = 1000,
        save_interval: int = 10000,
        eval_interval: int = 5000
    ) -> Dict[str, Any]:
        """
        Train the RL agent
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback_interval: Interval for logging callbacks
            save_interval: Interval for saving checkpoints
            eval_interval: Interval for evaluation
            
        Returns:
            Training statistics and metrics
        """
        self.logger.info(f"Starting training for {total_timesteps} timesteps")
        self.training_stats['training_start_time'] = datetime.now()
        
        # Setup callback
        callback = RL_TrainingCallback(log_interval=callback_interval)
        
        try:
            # Train the model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=callback_interval,
                reset_num_timesteps=False
            )
            
            # Update training stats
            self.training_stats['total_timesteps'] += total_timesteps
            self.training_stats['episodes_trained'] = len(callback.episode_rewards)
            if len(callback.episode_rewards) > 0:
                best_reward = max(callback.episode_rewards)
                if best_reward > self.training_stats['best_reward']:
                    self.training_stats['best_reward'] = best_reward
                    
            # Save final checkpoint
            self.save_checkpoint("final")
            
            self.logger.info("Training completed successfully")
            
            return {
                'training_stats': self.training_stats,
                'episode_rewards': callback.episode_rewards,
                'episode_lengths': callback.episode_lengths,
                'best_mean_reward': callback.best_mean_reward
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for given observation
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic policy
            action_masks: Optional action masks for invalid actions
            
        Returns:
            Tuple of (action, state_values)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")
            
        action, state = self.model.predict(
            observation,
            deterministic=deterministic
        )
        
        # Apply action masks if provided
        if action_masks is not None:
            action = self._apply_action_masks(action, action_masks)
            
        return action, state
        
    def _apply_action_masks(
        self,
        action: np.ndarray,
        action_masks: np.ndarray
    ) -> np.ndarray:
        """Apply action masks to prevent invalid actions"""
        # Simple masking - set invalid actions to 0 (no-op)
        if len(action_masks) > 0:
            valid_actions = np.where(action_masks == 1)[0]
            if len(valid_actions) > 0 and action not in valid_actions:
                action = np.random.choice(valid_actions)
        return action
        
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the trained agent
        
        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")
            
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        
        for episode in range(n_episodes):
            obs, _ = self.environment.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.environment.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Track success rate if available in info
            if 'success' in info:
                success_rates.append(float(info['success']))
                
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean(success_rates) if success_rates else 0.0,
            'episodes_evaluated': n_episodes
        }
        
    def save_checkpoint(self, checkpoint_name: str = None) -> str:
        """
        Save model checkpoint
        
        Args:
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{int(time.time())}"
            
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.zip"
        
        # Save model
        self.model.save(checkpoint_path)
        
        # Save training stats
        stats_path = self.checkpoint_dir / f"{checkpoint_name}_stats.json"
        with open(stats_path, 'w') as f:
            # Convert datetime to string for JSON serialization
            stats = self.training_stats.copy()
            if stats['training_start_time']:
                stats['training_start_time'] = stats['training_start_time'].isoformat()
            json.dump(stats, f, indent=2)
            
        self.training_stats['last_checkpoint'] = checkpoint_path
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load model
        self.model = PPO.load(checkpoint_path)
        
        # Load training stats if available
        stats_path = checkpoint_path.parent / f"{checkpoint_path.stem}_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                # Convert string back to datetime
                if stats.get('training_start_time'):
                    stats['training_start_time'] = datetime.fromisoformat(
                        stats['training_start_time']
                    )
                self.training_stats.update(stats)
                
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.model is None:
            return {"status": "not_initialized"}
            
        return {
            "status": "initialized",
            "agent_id": self.agent_id,
            "device": str(self.device),
            "policy_class": str(type(self.model.policy)),
            "training_stats": self.training_stats,
            "config": self.config.__dict__
        }
        
    def set_learning_rate(self, learning_rate: float):
        """Update the learning rate"""
        if self.model is not None:
            self.model.learning_rate = learning_rate
            self.config.learning_rate = learning_rate
            self.logger.info(f"Learning rate updated to: {learning_rate}")
            
    def get_policy_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current policy parameters"""
        if self.model is None:
            return {}
            
        return {name: param.clone() for name, param in self.model.policy.named_parameters()}
        
    def update_policy_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Update policy parameters (for multi-agent coordination)"""
        if self.model is None:
            return
            
        with torch.no_grad():
            for name, param in self.model.policy.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name])
                    
        self.logger.info("Policy parameters updated")