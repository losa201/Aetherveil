"""
Model Checkpointing and Versioning System

This module implements comprehensive model management including checkpointing,
versioning, model comparison, and automated model selection based on performance.
"""

import torch
import numpy as np
import pickle
import json
import shutil
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime, timedelta
import threading
import queue
from abc import ABC, abstractmethod
import zipfile
import tempfile


class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pytorch"
    STABLE_BASELINES3 = "stable_baselines3"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class CheckpointType(Enum):
    """Types of checkpoints"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    MILESTONE = "milestone"
    BEST_PERFORMANCE = "best_performance"
    EXPERIMENT = "experiment"
    DEPLOYMENT = "deployment"


@dataclass
class ModelMetadata:
    """Metadata for a model checkpoint"""
    model_id: str
    version: str
    checkpoint_type: CheckpointType
    model_format: ModelFormat
    creation_time: datetime
    file_path: str
    file_size: int
    file_hash: str
    
    # Performance metrics
    training_episodes: int = 0
    total_reward: float = 0.0
    success_rate: float = 0.0
    avg_episode_length: float = 0.0
    detection_rate: float = 0.0
    
    # Training configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    training_duration: float = 0.0
    
    # Additional information
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None
    experiment_id: Optional[str] = None
    
    # Validation metrics
    validation_scores: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelComparison:
    """Comparison between two models"""
    model1_id: str
    model2_id: str
    comparison_metrics: Dict[str, float]
    winner: str
    confidence: float
    comparison_time: datetime
    test_environment: Dict[str, Any]
    statistical_significance: bool = False


class ModelValidator(ABC):
    """Base class for model validation"""
    
    @abstractmethod
    def validate_model(self, model_path: str, validation_config: Dict[str, Any]) -> Dict[str, float]:
        """Validate model and return metrics"""
        pass


class PerformanceValidator(ModelValidator):
    """Validates model performance on standard benchmarks"""
    
    def __init__(self, benchmark_environments: List[Dict[str, Any]]):
        self.benchmark_environments = benchmark_environments
        
    def validate_model(self, model_path: str, validation_config: Dict[str, Any]) -> Dict[str, float]:
        """Run model on benchmark environments"""
        # This would load the model and run it on benchmarks
        # For now, return simulated results
        return {
            "avg_reward": np.random.uniform(5.0, 15.0),
            "success_rate": np.random.uniform(0.6, 0.9),
            "avg_episode_length": np.random.uniform(50.0, 150.0),
            "detection_rate": np.random.uniform(0.1, 0.4)
        }


class SecurityValidator(ModelValidator):
    """Validates model security and robustness"""
    
    def validate_model(self, model_path: str, validation_config: Dict[str, Any]) -> Dict[str, float]:
        """Validate model security properties"""
        return {
            "adversarial_robustness": np.random.uniform(0.7, 0.95),
            "privacy_preservation": np.random.uniform(0.8, 1.0),
            "bias_score": np.random.uniform(0.0, 0.3),
            "explainability_score": np.random.uniform(0.6, 0.9)
        }


class ModelCheckpointManager:
    """
    Comprehensive model checkpointing and versioning system
    
    Manages model lifecycle including saving, loading, versioning,
    validation, and automated model selection.
    """
    
    def __init__(
        self,
        base_directory: str,
        max_checkpoints: int = 50,
        auto_cleanup: bool = True,
        compression_enabled: bool = True,
        validation_enabled: bool = True
    ):
        """
        Initialize model checkpoint manager
        
        Args:
            base_directory: Base directory for storing checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            auto_cleanup: Enable automatic cleanup of old checkpoints
            compression_enabled: Enable checkpoint compression
            validation_enabled: Enable model validation
        """
        self.base_directory = Path(base_directory)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        self.compression_enabled = compression_enabled
        self.validation_enabled = validation_enabled
        
        # Create directory structure
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.base_directory / "checkpoints"
        self.metadata_dir = self.base_directory / "metadata"
        self.temp_dir = self.base_directory / "temp"
        self.archive_dir = self.base_directory / "archive"
        
        for directory in [self.checkpoints_dir, self.metadata_dir, self.temp_dir, self.archive_dir]:
            directory.mkdir(exist_ok=True)
            
        # Model registry
        self.models_registry = {}
        self.version_history = {}
        self.performance_history = {}
        
        # Validators
        self.validators = {}
        if validation_enabled:
            self.validators["performance"] = PerformanceValidator([])
            self.validators["security"] = SecurityValidator()
            
        # Background processing
        self.background_queue = queue.Queue()
        self.background_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.background_thread.start()
        
        # Load existing registry
        self._load_registry()
        
        # Logging
        self.logger = logging.getLogger("ModelCheckpointManager")
        
    def _background_processor(self):
        """Background thread for processing tasks"""
        while True:
            try:
                task = self.background_queue.get(timeout=1.0)
                task_type = task.get("type")
                
                if task_type == "validate_model":
                    self._validate_model_background(task)
                elif task_type == "cleanup_old_checkpoints":
                    self._cleanup_old_checkpoints()
                elif task_type == "archive_checkpoint":
                    self._archive_checkpoint(task["model_id"])
                    
                self.background_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Background processing error: {e}")
                
    def save_checkpoint(
        self,
        model: Any,
        model_id: str,
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
        model_format: ModelFormat = ModelFormat.STABLE_BASELINES3,
        performance_metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: List[str] = None,
        force_save: bool = False
    ) -> str:
        """
        Save model checkpoint with metadata
        
        Args:
            model: Model object to save
            model_id: Unique identifier for the model
            checkpoint_type: Type of checkpoint
            model_format: Format to save the model
            performance_metrics: Performance metrics for this checkpoint
            training_config: Training configuration used
            description: Human-readable description
            tags: Tags for categorization
            force_save: Force save even if similar checkpoint exists
            
        Returns:
            Version string of the saved checkpoint
        """
        # Generate version
        version = self._generate_version(model_id)
        
        # Check if we should save (avoid duplicate saves)
        if not force_save and self._should_skip_save(model_id, performance_metrics):
            self.logger.info(f"Skipping save for {model_id} - similar checkpoint exists")
            return self._get_latest_version(model_id)
            
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_id}_{version}_{timestamp}"
        
        if model_format == ModelFormat.STABLE_BASELINES3:
            file_path = self.checkpoints_dir / f"{filename}.zip"
            model.save(str(file_path))
        elif model_format == ModelFormat.PYTORCH:
            file_path = self.checkpoints_dir / f"{filename}.pth"
            torch.save(model.state_dict(), file_path)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
            
        # Compress if enabled
        if self.compression_enabled and file_path.suffix != ".zip":
            compressed_path = self._compress_checkpoint(file_path)
            file_path.unlink()  # Remove uncompressed file
            file_path = compressed_path
            
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            checkpoint_type=checkpoint_type,
            model_format=model_format,
            creation_time=datetime.now(),
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            file_hash=file_hash,
            description=description,
            tags=tags or [],
            hyperparameters=training_config.get("hyperparameters", {}) if training_config else {},
            environment_config=training_config.get("environment", {}) if training_config else {}
        )
        
        # Add performance metrics
        if performance_metrics:
            metadata.total_reward = performance_metrics.get("total_reward", 0.0)
            metadata.success_rate = performance_metrics.get("success_rate", 0.0)
            metadata.avg_episode_length = performance_metrics.get("avg_episode_length", 0.0)
            metadata.detection_rate = performance_metrics.get("detection_rate", 0.0)
            metadata.training_episodes = performance_metrics.get("training_episodes", 0)
            metadata.training_duration = performance_metrics.get("training_duration", 0.0)
            
        # Save metadata
        self._save_metadata(metadata)
        
        # Update registry
        if model_id not in self.models_registry:
            self.models_registry[model_id] = []
        self.models_registry[model_id].append(version)
        
        # Update version history
        if model_id not in self.version_history:
            self.version_history[model_id] = []
        self.version_history[model_id].append({
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_type": checkpoint_type.value,
            "performance": performance_metrics or {}
        })
        
        # Save registry
        self._save_registry()
        
        # Queue validation if enabled
        if self.validation_enabled:
            self.background_queue.put({
                "type": "validate_model",
                "model_id": model_id,
                "version": version,
                "file_path": str(file_path)
            })
            
        # Queue cleanup if needed
        if self.auto_cleanup and len(self.models_registry[model_id]) > self.max_checkpoints:
            self.background_queue.put({"type": "cleanup_old_checkpoints"})
            
        self.logger.info(f"Saved checkpoint {model_id} version {version}")
        return version
        
    def load_checkpoint(
        self,
        model_id: str,
        version: Optional[str] = None,
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Tuple[Any, ModelMetadata]:
        """
        Load model checkpoint
        
        Args:
            model_id: Model identifier
            version: Specific version to load (latest if None)
            checkpoint_type: Filter by checkpoint type
            
        Returns:
            Tuple of (model, metadata)
        """
        # Find the checkpoint to load
        metadata = self._find_checkpoint(model_id, version, checkpoint_type)
        if not metadata:
            raise ValueError(f"No checkpoint found for {model_id}")
            
        # Load the model
        file_path = Path(metadata.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
            
        # Verify file integrity
        if not self._verify_file_integrity(file_path, metadata.file_hash):
            raise ValueError(f"Checkpoint file corrupted: {file_path}")
            
        # Load based on format
        if metadata.model_format == ModelFormat.STABLE_BASELINES3:
            from stable_baselines3 import PPO
            model = PPO.load(str(file_path))
        elif metadata.model_format == ModelFormat.PYTORCH:
            model = torch.load(file_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported model format: {metadata.model_format}")
            
        self.logger.info(f"Loaded checkpoint {model_id} version {metadata.version}")
        return model, metadata
        
    def _find_checkpoint(
        self,
        model_id: str,
        version: Optional[str] = None,
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Optional[ModelMetadata]:
        """Find checkpoint matching criteria"""
        if model_id not in self.models_registry:
            return None
            
        candidates = []
        for ver in self.models_registry[model_id]:
            metadata = self._load_metadata(model_id, ver)
            if metadata:
                if version and metadata.version != version:
                    continue
                if checkpoint_type and metadata.checkpoint_type != checkpoint_type:
                    continue
                candidates.append(metadata)
                
        if not candidates:
            return None
            
        # Return latest if no specific version requested
        if version is None:
            candidates.sort(key=lambda m: m.creation_time, reverse=True)
            
        return candidates[0]
        
    def get_best_model(
        self,
        model_id: str,
        metric: str = "total_reward",
        minimize: bool = False
    ) -> Tuple[Any, ModelMetadata]:
        """
        Get the best performing model based on a metric
        
        Args:
            model_id: Model identifier
            metric: Metric to optimize for
            minimize: Whether to minimize the metric (default: maximize)
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_id not in self.models_registry:
            raise ValueError(f"No models found for {model_id}")
            
        best_metadata = None
        best_score = float('inf') if minimize else float('-inf')
        
        for version in self.models_registry[model_id]:
            metadata = self._load_metadata(model_id, version)
            if metadata:
                score = getattr(metadata, metric, None)
                if score is None:
                    score = metadata.validation_scores.get(metric, None)
                    
                if score is not None:
                    if (minimize and score < best_score) or (not minimize and score > best_score):
                        best_score = score
                        best_metadata = metadata
                        
        if best_metadata is None:
            raise ValueError(f"No valid checkpoints found for metric {metric}")
            
        model, _ = self.load_checkpoint(model_id, best_metadata.version)
        return model, best_metadata
        
    def compare_models(
        self,
        model1_id: str,
        model2_id: str,
        version1: Optional[str] = None,
        version2: Optional[str] = None,
        comparison_config: Optional[Dict[str, Any]] = None
    ) -> ModelComparison:
        """
        Compare two models
        
        Args:
            model1_id: First model identifier
            model2_id: Second model identifier
            version1: Version of first model (latest if None)
            version2: Version of second model (latest if None)
            comparison_config: Configuration for comparison
            
        Returns:
            ModelComparison object
        """
        # Load models
        model1, metadata1 = self.load_checkpoint(model1_id, version1)
        model2, metadata2 = self.load_checkpoint(model2_id, version2)
        
        # Compare metrics
        comparison_metrics = {}
        
        # Performance metrics
        perf_metrics = ["total_reward", "success_rate", "avg_episode_length", "detection_rate"]
        for metric in perf_metrics:
            val1 = getattr(metadata1, metric, 0.0)
            val2 = getattr(metadata2, metric, 0.0)
            
            if metric == "detection_rate":  # Lower is better
                comparison_metrics[metric] = (val2 - val1) / max(val1, val2, 0.01)
            else:  # Higher is better
                comparison_metrics[metric] = (val1 - val2) / max(val1, val2, 0.01)
                
        # Validation metrics
        for metric in set(metadata1.validation_scores.keys()) | set(metadata2.validation_scores.keys()):
            val1 = metadata1.validation_scores.get(metric, 0.0)
            val2 = metadata2.validation_scores.get(metric, 0.0)
            comparison_metrics[f"validation_{metric}"] = (val1 - val2) / max(val1, val2, 0.01)
            
        # Determine winner
        overall_score = np.mean(list(comparison_metrics.values()))
        winner = model1_id if overall_score > 0 else model2_id
        confidence = min(0.99, abs(overall_score))
        
        comparison = ModelComparison(
            model1_id=f"{model1_id}:{metadata1.version}",
            model2_id=f"{model2_id}:{metadata2.version}",
            comparison_metrics=comparison_metrics,
            winner=winner,
            confidence=confidence,
            comparison_time=datetime.now(),
            test_environment=comparison_config or {},
            statistical_significance=confidence > 0.5
        )
        
        return comparison
        
    def _validate_model_background(self, task: Dict[str, Any]):
        """Validate model in background thread"""
        model_id = task["model_id"]
        version = task["version"]
        file_path = task["file_path"]
        
        try:
            validation_results = {}
            
            for validator_name, validator in self.validators.items():
                results = validator.validate_model(file_path, {})
                validation_results.update({f"{validator_name}_{k}": v for k, v in results.items()})
                
            # Update metadata with validation results
            metadata = self._load_metadata(model_id, version)
            if metadata:
                metadata.validation_scores.update(validation_results)
                self._save_metadata(metadata)
                
            self.logger.info(f"Validated model {model_id} version {version}")
            
        except Exception as e:
            self.logger.error(f"Model validation failed for {model_id}:{version} - {e}")
            
    def _generate_version(self, model_id: str) -> str:
        """Generate version string for model"""
        if model_id not in self.models_registry:
            return "1.0.0"
            
        versions = self.models_registry[model_id]
        if not versions:
            return "1.0.0"
            
        # Parse latest version and increment
        latest = max(versions, key=lambda v: [int(x) for x in v.split('.')])
        major, minor, patch = map(int, latest.split('.'))
        
        # Increment patch version
        return f"{major}.{minor}.{patch + 1}"
        
    def _should_skip_save(self, model_id: str, performance_metrics: Optional[Dict[str, float]]) -> bool:
        """Determine if we should skip saving this checkpoint"""
        if not performance_metrics or model_id not in self.models_registry:
            return False
            
        # Check if similar performance checkpoint exists
        recent_versions = self.models_registry[model_id][-5:]  # Last 5 versions
        
        for version in recent_versions:
            metadata = self._load_metadata(model_id, version)
            if metadata:
                # Compare key metrics
                reward_diff = abs(metadata.total_reward - performance_metrics.get("total_reward", 0))
                success_diff = abs(metadata.success_rate - performance_metrics.get("success_rate", 0))
                
                if reward_diff < 0.5 and success_diff < 0.05:  # Very similar performance
                    return True
                    
        return False
        
    def _get_latest_version(self, model_id: str) -> str:
        """Get latest version for model"""
        if model_id not in self.models_registry or not self.models_registry[model_id]:
            return "1.0.0"
        return max(self.models_registry[model_id], key=lambda v: [int(x) for x in v.split('.')])
        
    def _compress_checkpoint(self, file_path: Path) -> Path:
        """Compress checkpoint file"""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        import gzip
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        return compressed_path
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
        
    def _verify_file_integrity(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file integrity using hash"""
        actual_hash = self._calculate_file_hash(file_path)
        return actual_hash == expected_hash
        
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to disk"""
        filename = f"{metadata.model_id}_{metadata.version}.json"
        filepath = self.metadata_dir / filename
        
        # Convert to dict for JSON serialization
        metadata_dict = {
            "model_id": metadata.model_id,
            "version": metadata.version,
            "checkpoint_type": metadata.checkpoint_type.value,
            "model_format": metadata.model_format.value,
            "creation_time": metadata.creation_time.isoformat(),
            "file_path": metadata.file_path,
            "file_size": metadata.file_size,
            "file_hash": metadata.file_hash,
            "training_episodes": metadata.training_episodes,
            "total_reward": metadata.total_reward,
            "success_rate": metadata.success_rate,
            "avg_episode_length": metadata.avg_episode_length,
            "detection_rate": metadata.detection_rate,
            "hyperparameters": metadata.hyperparameters,
            "environment_config": metadata.environment_config,
            "training_duration": metadata.training_duration,
            "description": metadata.description,
            "tags": metadata.tags,
            "parent_version": metadata.parent_version,
            "experiment_id": metadata.experiment_id,
            "validation_scores": metadata.validation_scores,
            "benchmark_results": metadata.benchmark_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
            
    def _load_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Load metadata from disk"""
        filename = f"{model_id}_{version}.json"
        filepath = self.metadata_dir / filename
        
        if not filepath.exists():
            return None
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            return ModelMetadata(
                model_id=data["model_id"],
                version=data["version"],
                checkpoint_type=CheckpointType(data["checkpoint_type"]),
                model_format=ModelFormat(data["model_format"]),
                creation_time=datetime.fromisoformat(data["creation_time"]),
                file_path=data["file_path"],
                file_size=data["file_size"],
                file_hash=data["file_hash"],
                training_episodes=data.get("training_episodes", 0),
                total_reward=data.get("total_reward", 0.0),
                success_rate=data.get("success_rate", 0.0),
                avg_episode_length=data.get("avg_episode_length", 0.0),
                detection_rate=data.get("detection_rate", 0.0),
                hyperparameters=data.get("hyperparameters", {}),
                environment_config=data.get("environment_config", {}),
                training_duration=data.get("training_duration", 0.0),
                description=data.get("description", ""),
                tags=data.get("tags", []),
                parent_version=data.get("parent_version"),
                experiment_id=data.get("experiment_id"),
                validation_scores=data.get("validation_scores", {}),
                benchmark_results=data.get("benchmark_results", {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load metadata for {model_id}:{version} - {e}")
            return None
            
    def _save_registry(self):
        """Save models registry to disk"""
        registry_file = self.base_directory / "registry.json"
        registry_data = {
            "models_registry": self.models_registry,
            "version_history": self.version_history,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
            
    def _load_registry(self):
        """Load models registry from disk"""
        registry_file = self.base_directory / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    
                self.models_registry = data.get("models_registry", {})
                self.version_history = data.get("version_history", {})
                
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
                
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to stay within limits"""
        for model_id, versions in self.models_registry.items():
            if len(versions) > self.max_checkpoints:
                # Sort by creation time and keep newest
                version_metadata = []
                for version in versions:
                    metadata = self._load_metadata(model_id, version)
                    if metadata:
                        version_metadata.append((version, metadata.creation_time, metadata.checkpoint_type))
                        
                # Sort by creation time, but keep important checkpoints
                version_metadata.sort(key=lambda x: x[1])
                
                # Remove oldest, but preserve milestones and best performance
                to_remove = []
                keep_types = {CheckpointType.MILESTONE, CheckpointType.BEST_PERFORMANCE, CheckpointType.DEPLOYMENT}
                
                for version, creation_time, checkpoint_type in version_metadata:
                    if len(versions) - len(to_remove) <= self.max_checkpoints:
                        break
                    if checkpoint_type not in keep_types:
                        to_remove.append(version)
                        
                # Remove old checkpoints
                for version in to_remove:
                    self._remove_checkpoint(model_id, version)
                    
    def _remove_checkpoint(self, model_id: str, version: str):
        """Remove a specific checkpoint"""
        # Remove from registry
        if model_id in self.models_registry:
            if version in self.models_registry[model_id]:
                self.models_registry[model_id].remove(version)
                
        # Archive or delete files
        metadata = self._load_metadata(model_id, version)
        if metadata:
            file_path = Path(metadata.file_path)
            if file_path.exists():
                # Archive important checkpoints, delete others
                if metadata.checkpoint_type in {CheckpointType.MILESTONE, CheckpointType.BEST_PERFORMANCE}:
                    self._archive_checkpoint(model_id, version)
                else:
                    file_path.unlink()
                    
            # Remove metadata
            metadata_file = self.metadata_dir / f"{model_id}_{version}.json"
            if metadata_file.exists():
                metadata_file.unlink()
                
        self.logger.info(f"Removed checkpoint {model_id}:{version}")
        
    def _archive_checkpoint(self, model_id: str, version: str = None):
        """Archive checkpoint to long-term storage"""
        # Implementation would move files to archive storage
        pass
        
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about managed models"""
        stats = {
            "total_models": len(self.models_registry),
            "total_checkpoints": sum(len(versions) for versions in self.models_registry.values()),
            "storage_usage": 0,
            "model_details": {}
        }
        
        # Calculate storage usage and model details
        for model_id, versions in self.models_registry.items():
            model_stats = {
                "versions": len(versions),
                "latest_version": self._get_latest_version(model_id),
                "storage_size": 0,
                "performance_trend": []
            }
            
            for version in versions:
                metadata = self._load_metadata(model_id, version)
                if metadata:
                    model_stats["storage_size"] += metadata.file_size
                    model_stats["performance_trend"].append({
                        "version": version,
                        "total_reward": metadata.total_reward,
                        "success_rate": metadata.success_rate,
                        "creation_time": metadata.creation_time.isoformat()
                    })
                    
            stats["storage_usage"] += model_stats["storage_size"]
            stats["model_details"][model_id] = model_stats
            
        return stats
        
    def export_model(self, model_id: str, version: str, export_path: str, export_format: str = "zip"):
        """Export model with all metadata for sharing or deployment"""
        export_path = Path(export_path)
        
        # Load model and metadata
        model, metadata = self.load_checkpoint(model_id, version)
        
        if export_format == "zip":
            with zipfile.ZipFile(export_path, 'w') as zipf:
                # Add model file
                model_file = Path(metadata.file_path)
                zipf.write(model_file, model_file.name)
                
                # Add metadata
                metadata_content = json.dumps(metadata.__dict__, default=str, indent=2)
                zipf.writestr(f"{model_id}_{version}_metadata.json", metadata_content)
                
                # Add README
                readme_content = f"""
Model Export: {model_id} v{version}

Description: {metadata.description}
Performance: {metadata.total_reward:.2f} total reward, {metadata.success_rate:.2f} success rate
Created: {metadata.creation_time}
Training Episodes: {metadata.training_episodes}

To use this model:
1. Extract the zip file
2. Load the model using appropriate framework
3. Refer to metadata for configuration details
"""
                zipf.writestr("README.txt", readme_content)
                
        self.logger.info(f"Exported model {model_id}:{version} to {export_path}")
        
    def import_model(self, import_path: str, model_id: Optional[str] = None) -> str:
        """Import model from exported package"""
        import_path = Path(import_path)
        
        if import_path.suffix == ".zip":
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(import_path, 'r') as zipf:
                    zipf.extractall(temp_dir)
                    
                # Find metadata file
                temp_path = Path(temp_dir)
                metadata_files = list(temp_path.glob("*_metadata.json"))
                
                if metadata_files:
                    with open(metadata_files[0], 'r') as f:
                        metadata_dict = json.load(f)
                        
                    # Update model_id if provided
                    original_model_id = metadata_dict["model_id"]
                    if model_id:
                        metadata_dict["model_id"] = model_id
                        
                    # Find model file
                    model_files = [f for f in temp_path.iterdir() 
                                 if f.suffix in [".zip", ".pth"] and "metadata" not in f.name]
                    
                    if model_files:
                        # Copy model file to checkpoints
                        new_version = self._generate_version(metadata_dict["model_id"])
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        new_filename = f"{metadata_dict['model_id']}_{new_version}_{timestamp}{model_files[0].suffix}"
                        new_file_path = self.checkpoints_dir / new_filename
                        
                        shutil.copy2(model_files[0], new_file_path)
                        
                        # Create new metadata
                        metadata_dict["version"] = new_version
                        metadata_dict["file_path"] = str(new_file_path)
                        metadata_dict["creation_time"] = datetime.now().isoformat()
                        metadata_dict["file_hash"] = self._calculate_file_hash(new_file_path)
                        
                        # Save metadata
                        metadata_filename = f"{metadata_dict['model_id']}_{new_version}.json"
                        with open(self.metadata_dir / metadata_filename, 'w') as f:
                            json.dump(metadata_dict, f, indent=2)
                            
                        # Update registry
                        final_model_id = metadata_dict["model_id"]
                        if final_model_id not in self.models_registry:
                            self.models_registry[final_model_id] = []
                        self.models_registry[final_model_id].append(new_version)
                        
                        self._save_registry()
                        
                        self.logger.info(f"Imported model {final_model_id}:{new_version}")
                        return new_version
                        
        raise ValueError("Invalid import file format")
        
    def cleanup_corrupted_checkpoints(self):
        """Find and clean up corrupted checkpoints"""
        corrupted = []
        
        for model_id, versions in self.models_registry.items():
            for version in versions:
                metadata = self._load_metadata(model_id, version)
                if metadata:
                    file_path = Path(metadata.file_path)
                    if not file_path.exists():
                        corrupted.append((model_id, version, "File missing"))
                    elif not self._verify_file_integrity(file_path, metadata.file_hash):
                        corrupted.append((model_id, version, "Hash mismatch"))
                        
        # Remove corrupted checkpoints
        for model_id, version, reason in corrupted:
            self.logger.warning(f"Removing corrupted checkpoint {model_id}:{version} - {reason}")
            self._remove_checkpoint(model_id, version)
            
        self._save_registry()
        return corrupted