"""
Online Learning and Adaptation System

This module implements online learning capabilities for RL agents to continuously
adapt to new environments, threats, and scenarios without full retraining.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import threading
import queue
import logging
from abc import ABC, abstractmethod
from enum import Enum
import random
import copy
import math


class AdaptationTrigger(Enum):
    """Triggers for adaptation"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ENVIRONMENT_CHANGE = "environment_change"
    NEW_THREAT_DETECTED = "new_threat_detected"
    PERIODIC = "periodic"
    MANUAL = "manual"
    CONCEPT_DRIFT = "concept_drift"


class AdaptationStrategy(Enum):
    """Adaptation strategies"""
    INCREMENTAL_LEARNING = "incremental_learning"
    FINE_TUNING = "fine_tuning"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    ENSEMBLE_UPDATE = "ensemble_update"
    DYNAMIC_ARCHITECTURE = "dynamic_architecture"


@dataclass
class AdaptationEvent:
    """Event that triggers adaptation"""
    trigger: AdaptationTrigger
    timestamp: float
    context: Dict[str, Any]
    severity: float  # 0.0 to 1.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_required: bool = True


@dataclass
class AdaptationResult:
    """Result of an adaptation process"""
    strategy_used: AdaptationStrategy
    success: bool
    performance_improvement: float
    adaptation_time: float
    model_changes: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitors performance for adaptation triggers"""
    
    def __init__(self, window_size: int = 100, degradation_threshold: float = 0.15):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None
        self.last_check_time = time.time()
        
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        # Aggregate key metrics into single performance score
        score = self._calculate_performance_score(metrics)
        self.performance_history.append({
            "score": score,
            "timestamp": time.time(),
            "metrics": metrics.copy()
        })
        
        # Update baseline if needed
        if self.baseline_performance is None and len(self.performance_history) >= 20:
            recent_scores = [p["score"] for p in list(self.performance_history)[-20:]]
            self.baseline_performance = np.mean(recent_scores)
            
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate aggregate performance score"""
        # Weight different metrics
        weights = {
            "success_rate": 0.4,
            "average_reward": 0.3,
            "efficiency": 0.2,
            "detection_avoidance": 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                # Normalize metrics to 0-1 range
                normalized_value = self._normalize_metric(metric, value)
                score += weights[metric] * normalized_value
                total_weight += weights[metric]
                
        return score / total_weight if total_weight > 0 else 0.0
        
    def _normalize_metric(self, metric: str, value: float) -> float:
        """Normalize metric to 0-1 range"""
        if metric == "success_rate":
            return np.clip(value, 0.0, 1.0)
        elif metric == "average_reward":
            return np.clip(value / 20.0, 0.0, 1.0)  # Assume max reward ~20
        elif metric == "efficiency":
            return np.clip(value / 0.5, 0.0, 1.0)  # Assume max efficiency ~0.5
        elif metric == "detection_avoidance":
            return np.clip(1.0 - value, 0.0, 1.0)  # Lower detection is better
        else:
            return np.clip(value, 0.0, 1.0)
            
    def check_degradation(self) -> Optional[AdaptationEvent]:
        """Check for performance degradation"""
        if (len(self.performance_history) < 20 or 
            self.baseline_performance is None or
            time.time() - self.last_check_time < 10.0):  # Check every 10 seconds
            return None
            
        # Calculate recent performance
        recent_scores = [p["score"] for p in list(self.performance_history)[-10:]]
        recent_performance = np.mean(recent_scores)
        
        # Check for degradation
        degradation = (self.baseline_performance - recent_performance) / self.baseline_performance
        
        if degradation > self.degradation_threshold:
            self.last_check_time = time.time()
            
            return AdaptationEvent(
                trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
                timestamp=time.time(),
                context={
                    "baseline_performance": self.baseline_performance,
                    "recent_performance": recent_performance,
                    "degradation": degradation,
                    "threshold": self.degradation_threshold
                },
                severity=min(1.0, degradation / self.degradation_threshold),
                evidence=[
                    {"type": "performance_metrics", "data": p["metrics"]} 
                    for p in list(self.performance_history)[-10:]
                ]
            )
            
        return None


class EnvironmentDetector:
    """Detects changes in the environment"""
    
    def __init__(self, change_threshold: float = 0.3):
        self.change_threshold = change_threshold
        self.environment_signatures = deque(maxlen=50)
        self.baseline_signature = None
        
    def update_environment_signature(self, env_state: Dict[str, Any]):
        """Update environment signature"""
        signature = self._compute_signature(env_state)
        self.environment_signatures.append({
            "signature": signature,
            "timestamp": time.time(),
            "state": env_state.copy()
        })
        
        if self.baseline_signature is None and len(self.environment_signatures) >= 10:
            recent_signatures = [s["signature"] for s in list(self.environment_signatures)[-10:]]
            self.baseline_signature = np.mean(recent_signatures, axis=0)
            
    def _compute_signature(self, env_state: Dict[str, Any]) -> np.ndarray:
        """Compute environment signature vector"""
        features = []
        
        # Network topology features
        features.append(env_state.get("network_size", 20) / 50.0)  # Normalized
        features.append(hash(env_state.get("topology", "unknown")) % 1000 / 1000.0)
        
        # Defense characteristics
        features.append(env_state.get("defense_strength", 0.5))
        features.append(env_state.get("vulnerability_density", 0.5))
        features.append(env_state.get("detection_threshold", 0.8))
        
        # Host distribution
        host_types = env_state.get("host_types", {})
        for host_type in ["workstation", "server", "database", "web_server"]:
            features.append(host_types.get(host_type, 0) / 10.0)
            
        return np.array(features[:10])  # Limit to 10 features
        
    def check_environment_change(self) -> Optional[AdaptationEvent]:
        """Check for environment changes"""
        if (len(self.environment_signatures) < 10 or 
            self.baseline_signature is None):
            return None
            
        # Compare recent signatures to baseline
        recent_signatures = [s["signature"] for s in list(self.environment_signatures)[-5:]]
        recent_signature = np.mean(recent_signatures, axis=0)
        
        # Calculate change magnitude
        change_magnitude = np.linalg.norm(recent_signature - self.baseline_signature)
        
        if change_magnitude > self.change_threshold:
            return AdaptationEvent(
                trigger=AdaptationTrigger.ENVIRONMENT_CHANGE,
                timestamp=time.time(),
                context={
                    "baseline_signature": self.baseline_signature.tolist(),
                    "recent_signature": recent_signature.tolist(),
                    "change_magnitude": change_magnitude,
                    "threshold": self.change_threshold
                },
                severity=min(1.0, change_magnitude / self.change_threshold),
                evidence=[
                    {"type": "environment_state", "data": s["state"]} 
                    for s in list(self.environment_signatures)[-5:]
                ]
            )
            
        return None


class IncrementalLearner:
    """Implements incremental learning for online adaptation"""
    
    def __init__(self, learning_rate: float = 0.001, buffer_size: int = 1000):
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.experience_buffer = deque(maxlen=buffer_size)
        self.adaptation_optimizer = None
        
    def add_experience(self, experience: Dict[str, Any]):
        """Add new experience for incremental learning"""
        self.experience_buffer.append(experience)
        
    def adapt_model(self, model: nn.Module, adaptation_steps: int = 10) -> AdaptationResult:
        """Perform incremental learning adaptation"""
        if len(self.experience_buffer) < 10:
            return AdaptationResult(
                strategy_used=AdaptationStrategy.INCREMENTAL_LEARNING,
                success=False,
                performance_improvement=0.0,
                adaptation_time=0.0,
                model_changes={"reason": "insufficient_data"}
            )
            
        start_time = time.time()
        
        # Setup optimizer if not exists
        if self.adaptation_optimizer is None:
            self.adaptation_optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
        # Sample recent experiences
        recent_experiences = list(self.experience_buffer)[-min(100, len(self.experience_buffer)):]
        
        initial_loss = self._evaluate_model(model, recent_experiences)
        
        # Perform adaptation steps
        for step in range(adaptation_steps):
            batch = random.sample(recent_experiences, min(32, len(recent_experiences)))
            loss = self._compute_loss(model, batch)
            
            self.adaptation_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            self.adaptation_optimizer.step()
            
        final_loss = self._evaluate_model(model, recent_experiences)
        adaptation_time = time.time() - start_time
        
        # Calculate improvement
        improvement = (initial_loss - final_loss) / max(initial_loss, 1e-6)
        
        return AdaptationResult(
            strategy_used=AdaptationStrategy.INCREMENTAL_LEARNING,
            success=improvement > 0.01,
            performance_improvement=improvement,
            adaptation_time=adaptation_time,
            model_changes={
                "adaptation_steps": adaptation_steps,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "experiences_used": len(recent_experiences)
            }
        )
        
    def _compute_loss(self, model: nn.Module, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute loss for a batch of experiences"""
        # This is a simplified implementation
        # In practice, this would depend on the specific model architecture
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        for experience in batch:
            state = torch.tensor(experience.get("state", [0.0]), dtype=torch.float32)
            reward = torch.tensor(experience.get("reward", 0.0), dtype=torch.float32)
            
            # Simple prediction loss
            if hasattr(model, 'forward'):
                try:
                    prediction = model(state.unsqueeze(0))
                    if isinstance(prediction, tuple):
                        prediction = prediction[0]
                    loss = torch.nn.functional.mse_loss(prediction.squeeze(), reward)
                    total_loss = total_loss + loss
                except:
                    # Fallback for incompatible models
                    total_loss = total_loss + torch.tensor(0.1, requires_grad=True)
                    
        return total_loss / len(batch)
        
    def _evaluate_model(self, model: nn.Module, experiences: List[Dict[str, Any]]) -> float:
        """Evaluate model on experiences"""
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for experience in experiences:
                state = torch.tensor(experience.get("state", [0.0]), dtype=torch.float32)
                reward = torch.tensor(experience.get("reward", 0.0), dtype=torch.float32)
                
                try:
                    if hasattr(model, 'forward'):
                        prediction = model(state.unsqueeze(0))
                        if isinstance(prediction, tuple):
                            prediction = prediction[0]
                        loss = torch.nn.functional.mse_loss(prediction.squeeze(), reward)
                        total_loss += loss.item()
                        count += 1
                except:
                    continue
                    
        return total_loss / max(count, 1)


class MetaLearner:
    """Implements meta-learning for rapid adaptation"""
    
    def __init__(self, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = None
        self.task_history = deque(maxlen=100)
        
    def add_task(self, task_data: Dict[str, Any]):
        """Add task for meta-learning"""
        self.task_history.append(task_data)
        
    def meta_adapt(self, model: nn.Module, new_task: Dict[str, Any]) -> AdaptationResult:
        """Perform meta-learning adaptation"""
        start_time = time.time()
        
        if len(self.task_history) < 5:
            return AdaptationResult(
                strategy_used=AdaptationStrategy.META_LEARNING,
                success=False,
                performance_improvement=0.0,
                adaptation_time=0.0,
                model_changes={"reason": "insufficient_meta_data"}
            )
            
        # Setup meta-optimizer
        if self.meta_optimizer is None:
            self.meta_optimizer = optim.Adam(model.parameters(), lr=self.meta_lr)
            
        # Save initial model state
        initial_params = copy.deepcopy(model.state_dict())
        
        # Meta-learning update
        meta_loss = 0.0
        for task in list(self.task_history)[-10:]:  # Use recent tasks
            # Inner loop adaptation
            task_model = copy.deepcopy(model)
            task_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)
            
            # Simulate inner adaptation steps
            for _ in range(3):  # Few-shot adaptation
                loss = self._compute_task_loss(task_model, task)
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()
                
            # Compute meta-loss
            meta_loss += self._compute_task_loss(task_model, new_task)
            
        # Meta-gradient update
        meta_loss /= min(10, len(self.task_history))
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        adaptation_time = time.time() - start_time
        
        # Evaluate improvement
        improvement = 0.1  # Placeholder - would need proper evaluation
        
        return AdaptationResult(
            strategy_used=AdaptationStrategy.META_LEARNING,
            success=True,
            performance_improvement=improvement,
            adaptation_time=adaptation_time,
            model_changes={
                "meta_lr": self.meta_lr,
                "inner_lr": self.inner_lr,
                "tasks_used": min(10, len(self.task_history))
            }
        )
        
    def _compute_task_loss(self, model: nn.Module, task: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a specific task"""
        # Simplified task loss computation
        return torch.tensor(0.1, requires_grad=True)


class OnlineLearner:
    """
    Comprehensive online learning and adaptation system
    
    Provides continuous adaptation capabilities for RL agents
    to handle changing environments and new threats.
    """
    
    def __init__(
        self,
        adaptation_threshold: float = 0.15,
        max_adaptation_frequency: float = 300.0,  # seconds
        enable_meta_learning: bool = True
    ):
        """
        Initialize online learner
        
        Args:
            adaptation_threshold: Threshold for triggering adaptation
            max_adaptation_frequency: Minimum time between adaptations
            enable_meta_learning: Enable meta-learning capabilities
        """
        self.adaptation_threshold = adaptation_threshold
        self.max_adaptation_frequency = max_adaptation_frequency
        self.enable_meta_learning = enable_meta_learning
        
        # Monitoring components
        self.performance_monitor = PerformanceMonitor(degradation_threshold=adaptation_threshold)
        self.environment_detector = EnvironmentDetector()
        
        # Learning components
        self.incremental_learner = IncrementalLearner()
        self.meta_learner = MetaLearner() if enable_meta_learning else None
        
        # State tracking
        self.last_adaptation_time = 0.0
        self.adaptation_history = deque(maxlen=100)
        self.current_model = None
        self.baseline_model = None
        
        # Background processing
        self.adaptation_queue = queue.Queue()
        self.processing_active = False
        self.processing_thread = None
        
        # Logging
        self.logger = logging.getLogger("OnlineLearner")
        
    def start_adaptation(self, model: nn.Module):
        """Start online adaptation for a model"""
        self.current_model = model
        self.baseline_model = copy.deepcopy(model)
        
        # Start background processing
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Started online adaptation")
        
    def stop_adaptation(self):
        """Stop online adaptation"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            
        self.logger.info("Stopped online adaptation")
        
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        self.performance_monitor.update_performance(metrics)
        
        # Add to incremental learner if recent experience
        if hasattr(self, '_last_experience'):
            self._last_experience.update({"performance_metrics": metrics})
            self.incremental_learner.add_experience(self._last_experience)
            
    def update_environment(self, env_state: Dict[str, Any]):
        """Update environment state"""
        self.environment_detector.update_environment_signature(env_state)
        
    def add_experience(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict[str, Any] = None
    ):
        """Add experience for online learning"""
        experience = {
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "action": action,
            "reward": reward,
            "next_state": next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            "done": done,
            "info": info or {},
            "timestamp": time.time()
        }
        
        self._last_experience = experience
        self.incremental_learner.add_experience(experience)
        
        # Add to meta-learner if enabled
        if self.meta_learner:
            task_data = {
                "experience": experience,
                "environment_context": info.get("environment_context", {}),
                "performance_context": info.get("performance_context", {})
            }
            self.meta_learner.add_task(task_data)
            
    def trigger_manual_adaptation(self, context: Dict[str, Any] = None):
        """Manually trigger adaptation"""
        event = AdaptationEvent(
            trigger=AdaptationTrigger.MANUAL,
            timestamp=time.time(),
            context=context or {},
            severity=1.0
        )
        
        self.adaptation_queue.put(event)
        self.logger.info("Manual adaptation triggered")
        
    def _adaptation_loop(self):
        """Background adaptation processing loop"""
        while self.processing_active:
            try:
                # Check for adaptation triggers
                self._check_adaptation_triggers()
                
                # Process queued adaptations
                while not self.adaptation_queue.empty():
                    try:
                        event = self.adaptation_queue.get_nowait()
                        self._process_adaptation_event(event)
                    except queue.Empty:
                        break
                        
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Adaptation loop error: {e}")
                time.sleep(10.0)
                
    def _check_adaptation_triggers(self):
        """Check for adaptation triggers"""
        current_time = time.time()
        
        # Respect minimum adaptation frequency
        if current_time - self.last_adaptation_time < self.max_adaptation_frequency:
            return
            
        # Check performance degradation
        perf_event = self.performance_monitor.check_degradation()
        if perf_event:
            self.adaptation_queue.put(perf_event)
            
        # Check environment changes
        env_event = self.environment_detector.check_environment_change()
        if env_event:
            self.adaptation_queue.put(env_event)
            
    def _process_adaptation_event(self, event: AdaptationEvent):
        """Process an adaptation event"""
        if self.current_model is None:
            return
            
        self.logger.info(f"Processing adaptation event: {event.trigger.value}")
        
        # Select adaptation strategy based on event
        strategy = self._select_adaptation_strategy(event)
        
        # Perform adaptation
        result = self._execute_adaptation(strategy, event)
        
        # Record adaptation
        self.adaptation_history.append({
            "event": event,
            "result": result,
            "timestamp": time.time()
        })
        
        self.last_adaptation_time = time.time()
        self.logger.info(f"Adaptation completed: {result.success}")
        
    def _select_adaptation_strategy(self, event: AdaptationEvent) -> AdaptationStrategy:
        """Select appropriate adaptation strategy"""
        if event.trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
            if event.severity > 0.7:
                return AdaptationStrategy.FINE_TUNING
            else:
                return AdaptationStrategy.INCREMENTAL_LEARNING
                
        elif event.trigger == AdaptationTrigger.ENVIRONMENT_CHANGE:
            if self.enable_meta_learning and event.severity > 0.5:
                return AdaptationStrategy.META_LEARNING
            else:
                return AdaptationStrategy.INCREMENTAL_LEARNING
                
        elif event.trigger == AdaptationTrigger.NEW_THREAT_DETECTED:
            return AdaptationStrategy.TRANSFER_LEARNING
            
        else:
            return AdaptationStrategy.INCREMENTAL_LEARNING
            
    def _execute_adaptation(self, strategy: AdaptationStrategy, event: AdaptationEvent) -> AdaptationResult:
        """Execute the selected adaptation strategy"""
        if strategy == AdaptationStrategy.INCREMENTAL_LEARNING:
            return self.incremental_learner.adapt_model(self.current_model)
            
        elif strategy == AdaptationStrategy.META_LEARNING and self.meta_learner:
            new_task = {
                "context": event.context,
                "evidence": event.evidence
            }
            return self.meta_learner.meta_adapt(self.current_model, new_task)
            
        elif strategy == AdaptationStrategy.FINE_TUNING:
            return self._fine_tune_model(event)
            
        else:
            # Fallback to incremental learning
            return self.incremental_learner.adapt_model(self.current_model)
            
    def _fine_tune_model(self, event: AdaptationEvent) -> AdaptationResult:
        """Perform fine-tuning adaptation"""
        start_time = time.time()
        
        # More aggressive learning for fine-tuning
        fine_tune_learner = IncrementalLearner(learning_rate=0.01, buffer_size=500)
        
        # Copy recent experiences
        recent_experiences = list(self.incremental_learner.experience_buffer)[-200:]
        for exp in recent_experiences:
            fine_tune_learner.add_experience(exp)
            
        # Perform fine-tuning
        result = fine_tune_learner.adapt_model(self.current_model, adaptation_steps=50)
        result.strategy_used = AdaptationStrategy.FINE_TUNING
        result.metadata["fine_tuning"] = True
        
        return result
        
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status"""
        return {
            "active": self.processing_active,
            "last_adaptation": self.last_adaptation_time,
            "adaptations_performed": len(self.adaptation_history),
            "performance_monitor": {
                "baseline_performance": self.performance_monitor.baseline_performance,
                "recent_performance_count": len(self.performance_monitor.performance_history)
            },
            "environment_detector": {
                "baseline_signature": (self.environment_detector.baseline_signature.tolist() 
                                     if self.environment_detector.baseline_signature is not None else None),
                "signatures_collected": len(self.environment_detector.environment_signatures)
            },
            "incremental_learner": {
                "experiences": len(self.incremental_learner.experience_buffer),
                "buffer_size": self.incremental_learner.buffer_size
            },
            "meta_learner": {
                "enabled": self.enable_meta_learning,
                "tasks": len(self.meta_learner.task_history) if self.meta_learner else 0
            }
        }
        
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        if not self.adaptation_history:
            return {"status": "no_adaptations"}
            
        successful_adaptations = [a for a in self.adaptation_history if a["result"].success]
        
        # Strategy usage
        strategy_counts = defaultdict(int)
        for adaptation in self.adaptation_history:
            strategy_counts[adaptation["result"].strategy_used.value] += 1
            
        # Trigger analysis
        trigger_counts = defaultdict(int)
        for adaptation in self.adaptation_history:
            trigger_counts[adaptation["event"].trigger.value] += 1
            
        # Performance improvements
        improvements = [a["result"].performance_improvement for a in successful_adaptations]
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "successful_adaptations": len(successful_adaptations),
            "success_rate": len(successful_adaptations) / len(self.adaptation_history),
            "average_improvement": np.mean(improvements) if improvements else 0.0,
            "strategy_usage": dict(strategy_counts),
            "trigger_frequency": dict(trigger_counts),
            "adaptation_times": [a["result"].adaptation_time for a in self.adaptation_history],
            "recent_adaptations": [
                {
                    "trigger": a["event"].trigger.value,
                    "strategy": a["result"].strategy_used.value,
                    "success": a["result"].success,
                    "improvement": a["result"].performance_improvement,
                    "timestamp": a["timestamp"]
                }
                for a in list(self.adaptation_history)[-10:]
            ]
        }
        
    def reset_adaptation(self):
        """Reset adaptation state"""
        if self.baseline_model and self.current_model:
            self.current_model.load_state_dict(self.baseline_model.state_dict())
            
        self.performance_monitor = PerformanceMonitor(degradation_threshold=self.adaptation_threshold)
        self.environment_detector = EnvironmentDetector()
        self.incremental_learner = IncrementalLearner()
        
        if self.enable_meta_learning:
            self.meta_learner = MetaLearner()
            
        self.adaptation_history.clear()
        self.last_adaptation_time = 0.0
        
        self.logger.info("Adaptation state reset")
        
    def save_adaptation_state(self, filepath: str):
        """Save adaptation state"""
        state = {
            "adaptation_history": [
                {
                    "event": {
                        "trigger": a["event"].trigger.value,
                        "timestamp": a["event"].timestamp,
                        "context": a["event"].context,
                        "severity": a["event"].severity
                    },
                    "result": {
                        "strategy_used": a["result"].strategy_used.value,
                        "success": a["result"].success,
                        "performance_improvement": a["result"].performance_improvement,
                        "adaptation_time": a["result"].adaptation_time
                    },
                    "timestamp": a["timestamp"]
                }
                for a in self.adaptation_history
            ],
            "last_adaptation_time": self.last_adaptation_time,
            "performance_baseline": self.performance_monitor.baseline_performance,
            "environment_baseline": (self.environment_detector.baseline_signature.tolist() 
                                   if self.environment_detector.baseline_signature is not None else None)
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        self.logger.info(f"Saved adaptation state to {filepath}")
        
    def load_adaptation_state(self, filepath: str):
        """Load adaptation state"""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        # Restore basic state
        self.last_adaptation_time = state.get("last_adaptation_time", 0.0)
        self.performance_monitor.baseline_performance = state.get("performance_baseline")
        
        if state.get("environment_baseline"):
            self.environment_detector.baseline_signature = np.array(state["environment_baseline"])
            
        self.logger.info(f"Loaded adaptation state from {filepath}")
        
    def cleanup(self):
        """Cleanup resources"""
        self.stop_adaptation()
        self.logger.info("Online learner cleanup completed")