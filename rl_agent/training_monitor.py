"""
Training Monitoring and Visualization System

This module provides comprehensive monitoring and visualization capabilities
for RL training, including real-time metrics, performance dashboards,
and advanced analytics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import threading
import queue
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class TrainingMetric:
    """Single training metric measurement"""
    name: str
    value: float
    timestamp: float
    episode: int
    step: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSession:
    """Training session information"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_episodes: int = 0
    total_steps: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"


class MetricAggregator:
    """Aggregates metrics for analysis"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        
    def add_metric(self, metric: TrainingMetric):
        """Add a metric measurement"""
        self.metrics[metric.name].append(metric)
        
    def get_recent_values(self, metric_name: str, count: int = None) -> List[float]:
        """Get recent values for a metric"""
        if count is None:
            count = self.window_size
        values = [m.value for m in list(self.metrics[metric_name])[-count:]]
        return values
        
    def get_moving_average(self, metric_name: str, window: int = 10) -> float:
        """Get moving average of a metric"""
        values = self.get_recent_values(metric_name, window)
        return np.mean(values) if values else 0.0
        
    def get_trend(self, metric_name: str, window: int = 20) -> str:
        """Get trend direction of a metric"""
        values = self.get_recent_values(metric_name, window)
        if len(values) < 5:
            return "insufficient_data"
            
        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
            
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get comprehensive statistics for a metric"""
        values = self.get_recent_values(metric_name)
        if not values:
            return {}
            
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "trend": self.get_trend(metric_name),
            "count": len(values)
        }


class AlertSystem:
    """Monitoring alert system"""
    
    def __init__(self):
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning"
    ):
        """Add an alert rule"""
        self.alert_rules[name] = {
            "metric_name": metric_name,
            "condition": condition,  # "greater_than", "less_than", "equal_to"
            "threshold": threshold,
            "severity": severity,
            "triggered_count": 0,
            "last_triggered": None
        }
        
    def check_alerts(self, metrics: Dict[str, TrainingMetric]):
        """Check all alert rules against current metrics"""
        alerts = []
        
        for alert_name, rule in self.alert_rules.items():
            metric_name = rule["metric_name"]
            if metric_name in metrics:
                metric_value = metrics[metric_name].value
                
                triggered = False
                if rule["condition"] == "greater_than" and metric_value > rule["threshold"]:
                    triggered = True
                elif rule["condition"] == "less_than" and metric_value < rule["threshold"]:
                    triggered = True
                elif rule["condition"] == "equal_to" and abs(metric_value - rule["threshold"]) < 1e-6:
                    triggered = True
                    
                if triggered:
                    alert = {
                        "name": alert_name,
                        "severity": rule["severity"],
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "threshold": rule["threshold"],
                        "timestamp": time.time(),
                        "message": f"{alert_name}: {metric_name} = {metric_value:.3f} (threshold: {rule['threshold']:.3f})"
                    }
                    
                    alerts.append(alert)
                    self.active_alerts[alert_name] = alert
                    self.alert_history.append(alert)
                    
                    rule["triggered_count"] += 1
                    rule["last_triggered"] = time.time()
                    
                elif alert_name in self.active_alerts:
                    # Clear alert if condition no longer met
                    del self.active_alerts[alert_name]
                    
        return alerts


class VisualizationEngine:
    """Generates visualizations for training monitoring"""
    
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_curves(
        self,
        metrics_data: Dict[str, List[TrainingMetric]],
        save_path: Optional[str] = None
    ) -> str:
        """Plot training curves for multiple metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Define which metrics to plot in each subplot
        metric_plots = {
            (0, 0): {"metrics": ["episode_reward", "average_reward"], "title": "Reward Progress", "ylabel": "Reward"},
            (0, 1): {"metrics": ["success_rate"], "title": "Success Rate", "ylabel": "Success Rate"},
            (1, 0): {"metrics": ["episode_length"], "title": "Episode Length", "ylabel": "Steps"},
            (1, 1): {"metrics": ["detection_rate"], "title": "Detection Rate", "ylabel": "Detection Level"}
        }
        
        for (row, col), plot_config in metric_plots.items():
            ax = axes[row, col]
            
            for metric_name in plot_config["metrics"]:
                if metric_name in metrics_data:
                    data = metrics_data[metric_name]
                    episodes = [m.episode for m in data]
                    values = [m.value for m in data]
                    
                    ax.plot(episodes, values, label=metric_name.replace('_', ' ').title(), linewidth=2)
                    
                    # Add trend line
                    if len(episodes) > 10:
                        z = np.polyfit(episodes, values, 1)
                        p = np.poly1d(z)
                        ax.plot(episodes, p(episodes), "--", alpha=0.7, linewidth=1)
                        
            ax.set_title(plot_config["title"], fontsize=12, fontweight='bold')
            ax.set_xlabel("Episode")
            ax.set_ylabel(plot_config["ylabel"])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"training_curves_{int(time.time())}.png"
        else:
            save_path = Path(save_path)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
        
    def plot_performance_distribution(
        self,
        metrics_data: Dict[str, List[TrainingMetric]],
        save_path: Optional[str] = None
    ) -> str:
        """Plot performance distribution histograms"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Performance Distributions', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ["episode_reward", "success_rate", "episode_length", "detection_rate"]
        
        for i, metric_name in enumerate(metrics_to_plot):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if metric_name in metrics_data:
                values = [m.value for m in metrics_data[metric_name]]
                
                # Histogram
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {std_val:.2f}')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"performance_distribution_{int(time.time())}.png"
        else:
            save_path = Path(save_path)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
        
    def create_interactive_dashboard(
        self,
        metrics_data: Dict[str, List[TrainingMetric]],
        save_path: Optional[str] = None
    ) -> str:
        """Create interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Reward Progress', 'Success Rate', 'Episode Length', 
                          'Detection Rate', 'Metric Correlations', 'Training Velocity'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Reward progress with moving average
        if "episode_reward" in metrics_data:
            reward_data = metrics_data["episode_reward"]
            episodes = [m.episode for m in reward_data]
            rewards = [m.value for m in reward_data]
            
            # Raw rewards
            fig.add_trace(
                go.Scatter(x=episodes, y=rewards, mode='lines', name='Episode Reward',
                          line=dict(color='blue', width=1), opacity=0.6),
                row=1, col=1
            )
            
            # Moving average
            if len(rewards) > 10:
                window = 20
                moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(x=episodes, y=moving_avg, mode='lines', name=f'MA({window})',
                              line=dict(color='red', width=3)),
                    row=1, col=1
                )
                
        # Success rate
        if "success_rate" in metrics_data:
            success_data = metrics_data["success_rate"]
            episodes = [m.episode for m in success_data]
            success_rates = [m.value for m in success_data]
            
            fig.add_trace(
                go.Scatter(x=episodes, y=success_rates, mode='lines+markers', name='Success Rate',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
            
        # Episode length
        if "episode_length" in metrics_data:
            length_data = metrics_data["episode_length"]
            episodes = [m.episode for m in length_data]
            lengths = [m.value for m in length_data]
            
            fig.add_trace(
                go.Scatter(x=episodes, y=lengths, mode='lines', name='Episode Length',
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            
        # Detection rate
        if "detection_rate" in metrics_data:
            detection_data = metrics_data["detection_rate"]
            episodes = [m.episode for m in detection_data]
            detection_rates = [m.value for m in detection_data]
            
            fig.add_trace(
                go.Scatter(x=episodes, y=detection_rates, mode='lines', name='Detection Rate',
                          line=dict(color='orange', width=2)),
                row=2, col=2
            )
            
        # Correlation heatmap
        if len(metrics_data) >= 2:
            correlation_data = {}
            min_length = min(len(data) for data in metrics_data.values())
            
            for metric_name, data in metrics_data.items():
                correlation_data[metric_name] = [m.value for m in data[:min_length]]
                
            df = pd.DataFrame(correlation_data)
            corr_matrix = df.corr()
            
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0, name='Correlation'),
                row=3, col=1
            )
            
        # Training velocity (episodes per hour)
        if "episode_reward" in metrics_data:
            reward_data = metrics_data["episode_reward"]
            timestamps = [m.timestamp for m in reward_data]
            episodes = [m.episode for m in reward_data]
            
            if len(timestamps) > 10:
                # Calculate velocity over time windows
                window_size = 50
                velocities = []
                velocity_episodes = []
                
                for i in range(window_size, len(timestamps)):
                    time_diff = timestamps[i] - timestamps[i - window_size]
                    episode_diff = episodes[i] - episodes[i - window_size]
                    
                    if time_diff > 0:
                        velocity = episode_diff / (time_diff / 3600)  # episodes per hour
                        velocities.append(velocity)
                        velocity_episodes.append(episodes[i])
                        
                fig.add_trace(
                    go.Scatter(x=velocity_episodes, y=velocities, mode='lines', 
                              name='Training Velocity (eps/hour)',
                              line=dict(color='teal', width=2)),
                    row=3, col=2
                )
                
        # Update layout
        fig.update_layout(
            title="RL Training Dashboard",
            height=900,
            showlegend=True,
            template="plotly_white"
        )
        
        if save_path is None:
            save_path = self.save_dir / f"dashboard_{int(time.time())}.html"
        else:
            save_path = Path(save_path)
            
        fig.write_html(save_path)
        return str(save_path)
        
    def plot_attack_chain_analysis(
        self,
        attack_chains: List[List[Dict[str, Any]]],
        save_path: Optional[str] = None
    ) -> str:
        """Visualize attack chain patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Attack Chain Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        chain_lengths = [len(chain) for chain in attack_chains]
        all_actions = []
        action_sequences = []
        
        for chain in attack_chains:
            chain_actions = [step.get("action_type", "unknown") for step in chain]
            all_actions.extend(chain_actions)
            action_sequences.append(" -> ".join(chain_actions[:5]))  # First 5 actions
            
        # Chain length distribution
        axes[0, 0].hist(chain_lengths, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Attack Chain Lengths')
        axes[0, 0].set_xlabel('Chain Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Action type frequency
        action_counts = pd.Series(all_actions).value_counts()
        axes[0, 1].bar(range(len(action_counts)), action_counts.values)
        axes[0, 1].set_title('Action Type Frequency')
        axes[0, 1].set_xlabel('Action Type')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(range(len(action_counts)))
        axes[0, 1].set_xticklabels(action_counts.index, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Action sequence patterns
        sequence_counts = pd.Series(action_sequences).value_counts().head(10)
        axes[1, 0].barh(range(len(sequence_counts)), sequence_counts.values)
        axes[1, 0].set_title('Top Attack Sequences')
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_yticks(range(len(sequence_counts)))
        axes[1, 0].set_yticklabels(sequence_counts.index, fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate by chain length
        if attack_chains:
            length_success = defaultdict(list)
            for chain in attack_chains:
                length = len(chain)
                # Assume success if chain completed certain objectives
                success = any(step.get("success", False) for step in chain)
                length_success[length].append(success)
                
            lengths = sorted(length_success.keys())
            success_rates = [np.mean(length_success[length]) for length in lengths]
            
            axes[1, 1].plot(lengths, success_rates, 'o-', linewidth=2, markersize=6)
            axes[1, 1].set_title('Success Rate by Chain Length')
            axes[1, 1].set_xlabel('Chain Length')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"attack_chain_analysis_{int(time.time())}.png"
        else:
            save_path = Path(save_path)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)


class TrainingMonitor:
    """
    Comprehensive training monitoring and visualization system
    
    Provides real-time monitoring, alerting, and visualization
    capabilities for RL training processes.
    """
    
    def __init__(
        self,
        log_dir: str = "./training_logs",
        enable_alerts: bool = True,
        enable_visualizations: bool = True,
        update_interval: float = 10.0
    ):
        """
        Initialize training monitor
        
        Args:
            log_dir: Directory for storing logs and visualizations
            enable_alerts: Enable alert system
            enable_visualizations: Enable visualization generation
            update_interval: Interval for background updates (seconds)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_alerts = enable_alerts
        self.enable_visualizations = enable_visualizations
        self.update_interval = update_interval
        
        # Core components
        self.metric_aggregator = MetricAggregator()
        self.alert_system = AlertSystem() if enable_alerts else None
        self.visualization_engine = VisualizationEngine(str(self.log_dir / "visualizations"))
        
        # Data storage
        self.training_sessions = {}
        self.current_session = None
        self.metrics_history = defaultdict(list)
        self.attack_chains = []
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_queue = queue.Queue()
        
        # Callbacks
        self.alert_callbacks = []
        self.visualization_callbacks = []
        
        # Logging
        self.logger = logging.getLogger("TrainingMonitor")
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / "training_monitor.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start_session(self, session_id: str, config: Dict[str, Any] = None) -> str:
        """Start a new training session"""
        session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now(),
            config=config or {}
        )
        
        self.training_sessions[session_id] = session
        self.current_session = session
        
        self.logger.info(f"Started training session: {session_id}")
        
        # Start background monitoring
        if not self.monitoring_active:
            self.start_monitoring()
            
        return session_id
        
    def end_session(self, session_id: Optional[str] = None):
        """End a training session"""
        if session_id is None and self.current_session:
            session_id = self.current_session.session_id
            
        if session_id in self.training_sessions:
            session = self.training_sessions[session_id]
            session.end_time = datetime.now()
            session.status = "completed"
            
            self.logger.info(f"Ended training session: {session_id}")
            
            # Generate final visualizations
            if self.enable_visualizations:
                self._generate_session_summary(session_id)
                
    def log_metric(
        self,
        name: str,
        value: float,
        episode: int = 0,
        step: int = 0,
        metadata: Dict[str, Any] = None
    ):
        """Log a training metric"""
        metric = TrainingMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            episode=episode,
            step=step,
            metadata=metadata or {}
        )
        
        # Add to aggregator
        self.metric_aggregator.add_metric(metric)
        
        # Store in history
        self.metrics_history[name].append(metric)
        
        # Update current session
        if self.current_session:
            self.current_session.total_episodes = max(self.current_session.total_episodes, episode)
            self.current_session.total_steps = max(self.current_session.total_steps, step)
            
        # Queue for background processing
        self.update_queue.put(("metric", metric))
        
    def log_episode_completion(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        success: bool,
        detection_level: float,
        attack_chain: List[Dict[str, Any]] = None,
        additional_metrics: Dict[str, float] = None
    ):
        """Log episode completion with comprehensive metrics"""
        # Log standard metrics
        self.log_metric("episode_reward", total_reward, episode)
        self.log_metric("episode_length", episode_length, episode)
        self.log_metric("success_rate", float(success), episode)
        self.log_metric("detection_rate", detection_level, episode)
        
        # Log additional metrics
        if additional_metrics:
            for metric_name, value in additional_metrics.items():
                self.log_metric(metric_name, value, episode)
                
        # Store attack chain
        if attack_chain:
            self.attack_chains.append(attack_chain)
            
        # Calculate derived metrics
        self._calculate_derived_metrics(episode)
        
    def _calculate_derived_metrics(self, episode: int):
        """Calculate derived metrics from base metrics"""
        # Moving averages
        window = 50
        
        for metric_name in ["episode_reward", "episode_length", "detection_rate"]:
            if metric_name in self.metrics_history:
                moving_avg = self.metric_aggregator.get_moving_average(metric_name, window)
                self.log_metric(f"{metric_name}_ma{window}", moving_avg, episode)
                
        # Success rate over window
        if "success_rate" in self.metrics_history:
            recent_successes = self.metric_aggregator.get_recent_values("success_rate", window)
            success_rate = np.mean(recent_successes) if recent_successes else 0.0
            self.log_metric("success_rate_windowed", success_rate, episode)
            
        # Efficiency metric (reward per step)
        if "episode_reward" in self.metrics_history and "episode_length" in self.metrics_history:
            recent_rewards = self.metric_aggregator.get_recent_values("episode_reward", 1)
            recent_lengths = self.metric_aggregator.get_recent_values("episode_length", 1)
            
            if recent_rewards and recent_lengths and recent_lengths[0] > 0:
                efficiency = recent_rewards[0] / recent_lengths[0]
                self.log_metric("efficiency", efficiency, episode)
                
    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning"
    ):
        """Add an alert rule"""
        if self.alert_system:
            self.alert_system.add_alert_rule(name, metric_name, condition, threshold, severity)
            self.logger.info(f"Added alert rule: {name}")
            
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
        
    def add_visualization_callback(self, callback: Callable[[str], None]):
        """Add callback for visualization updates"""
        self.visualization_callbacks.append(callback)
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started background monitoring")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        self.logger.info("Stopped background monitoring")
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        last_alert_check = 0
        last_visualization_update = 0
        
        while self.monitoring_active:
            try:
                # Process queued updates
                while not self.update_queue.empty():
                    try:
                        update_type, data = self.update_queue.get_nowait()
                        if update_type == "metric":
                            self._process_metric_update(data)
                    except queue.Empty:
                        break
                        
                current_time = time.time()
                
                # Check alerts
                if (self.alert_system and 
                    current_time - last_alert_check > 5.0):  # Check every 5 seconds
                    self._check_alerts()
                    last_alert_check = current_time
                    
                # Update visualizations
                if (self.enable_visualizations and 
                    current_time - last_visualization_update > self.update_interval):
                    self._update_visualizations()
                    last_visualization_update = current_time
                    
                time.sleep(1.0)  # Sleep 1 second between iterations
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Wait before retrying
                
    def _process_metric_update(self, metric: TrainingMetric):
        """Process a metric update"""
        # Could add real-time processing here
        pass
        
    def _check_alerts(self):
        """Check alert conditions"""
        if not self.alert_system:
            return
            
        # Get latest metrics
        latest_metrics = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                latest_metrics[metric_name] = history[-1]
                
        # Check alert rules
        alerts = self.alert_system.check_alerts(latest_metrics)
        
        # Notify callbacks
        for alert in alerts:
            self.logger.warning(f"Alert triggered: {alert['message']}")
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
                    
    def _update_visualizations(self):
        """Update visualizations"""
        try:
            if len(self.metrics_history) > 0:
                # Generate training curves
                curves_path = self.visualization_engine.plot_training_curves(self.metrics_history)
                
                # Notify callbacks
                for callback in self.visualization_callbacks:
                    try:
                        callback(curves_path)
                    except Exception as e:
                        self.logger.error(f"Visualization callback error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Visualization update error: {e}")
            
    def _generate_session_summary(self, session_id: str):
        """Generate comprehensive session summary"""
        session = self.training_sessions.get(session_id)
        if not session:
            return
            
        summary_dir = self.log_dir / f"session_{session_id}_summary"
        summary_dir.mkdir(exist_ok=True)
        
        try:
            # Training curves
            curves_path = self.visualization_engine.plot_training_curves(
                self.metrics_history,
                str(summary_dir / "training_curves.png")
            )
            
            # Performance distribution
            dist_path = self.visualization_engine.plot_performance_distribution(
                self.metrics_history,
                str(summary_dir / "performance_distribution.png")
            )
            
            # Interactive dashboard
            dashboard_path = self.visualization_engine.create_interactive_dashboard(
                self.metrics_history,
                str(summary_dir / "dashboard.html")
            )
            
            # Attack chain analysis
            if self.attack_chains:
                attack_path = self.visualization_engine.plot_attack_chain_analysis(
                    self.attack_chains,
                    str(summary_dir / "attack_chain_analysis.png")
                )
                
            # Session summary report
            self._generate_text_summary(session, summary_dir)
            
            self.logger.info(f"Generated session summary for {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate session summary: {e}")
            
    def _generate_text_summary(self, session: TrainingSession, summary_dir: Path):
        """Generate text summary report"""
        summary_file = summary_dir / "summary_report.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Training Session Summary\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"Session ID: {session.session_id}\n")
            f.write(f"Start Time: {session.start_time}\n")
            f.write(f"End Time: {session.end_time}\n")
            
            if session.end_time:
                duration = session.end_time - session.start_time
                f.write(f"Duration: {duration}\n")
                
            f.write(f"Total Episodes: {session.total_episodes}\n")
            f.write(f"Total Steps: {session.total_steps}\n\n")
            
            # Metric statistics
            f.write("Metric Statistics:\n")
            f.write("-" * 30 + "\n")
            
            for metric_name in ["episode_reward", "success_rate", "episode_length", "detection_rate"]:
                if metric_name in self.metrics_history:
                    stats = self.metric_aggregator.get_statistics(metric_name)
                    f.write(f"\n{metric_name.title()}:\n")
                    f.write(f"  Mean: {stats.get('mean', 0):.3f}\n")
                    f.write(f"  Std: {stats.get('std', 0):.3f}\n")
                    f.write(f"  Min: {stats.get('min', 0):.3f}\n")
                    f.write(f"  Max: {stats.get('max', 0):.3f}\n")
                    f.write(f"  Trend: {stats.get('trend', 'unknown')}\n")
                    
            # Alert summary
            if self.alert_system:
                f.write(f"\nAlert Summary:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Active Alerts: {len(self.alert_system.active_alerts)}\n")
                f.write(f"Total Alerts: {len(self.alert_system.alert_history)}\n")
                
                if self.alert_system.active_alerts:
                    f.write("\nActive Alerts:\n")
                    for alert_name, alert in self.alert_system.active_alerts.items():
                        f.write(f"  - {alert['message']}\n")
                        
    def get_current_statistics(self) -> Dict[str, Any]:
        """Get current training statistics"""
        stats = {
            "session": None,
            "metrics": {},
            "alerts": {},
            "visualizations_generated": 0
        }
        
        if self.current_session:
            stats["session"] = {
                "id": self.current_session.session_id,
                "status": self.current_session.status,
                "episodes": self.current_session.total_episodes,
                "steps": self.current_session.total_steps,
                "duration": str(datetime.now() - self.current_session.start_time)
            }
            
        # Current metric statistics
        for metric_name in self.metrics_history:
            stats["metrics"][metric_name] = self.metric_aggregator.get_statistics(metric_name)
            
        # Alert statistics
        if self.alert_system:
            stats["alerts"] = {
                "active": len(self.alert_system.active_alerts),
                "total": len(self.alert_system.alert_history),
                "rules": len(self.alert_system.alert_rules)
            }
            
        return stats
        
    def export_data(self, export_path: str, format: str = "json"):
        """Export training data"""
        export_path = Path(export_path)
        
        export_data = {
            "sessions": {sid: {
                "session_id": session.session_id,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "total_episodes": session.total_episodes,
                "total_steps": session.total_steps,
                "config": session.config,
                "status": session.status
            } for sid, session in self.training_sessions.items()},
            
            "metrics": {name: [
                {
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "episode": m.episode,
                    "step": m.step,
                    "metadata": m.metadata
                } for m in metrics
            ] for name, metrics in self.metrics_history.items()},
            
            "attack_chains": self.attack_chains
        }
        
        if format == "json":
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        self.logger.info(f"Exported training data to {export_path}")
        
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        
        # Save final data
        if self.current_session:
            self.end_session()
            
        self.logger.info("Training monitor cleanup completed")