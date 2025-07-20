#!/usr/bin/env python3
"""
Knowledge Learner Module - Manages learning and knowledge base updates
"""

import asyncio
import json
import logging
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

from config import Config

class KnowledgeLearner:
    """Manages autonomous learning and knowledge base updates"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.knowledge_db_path = None
        self.learning_stats = {
            "total_updates": 0,
            "successful_patterns": 0,
            "failed_patterns": 0,
            "model_improvements": 0
        }
        
        # Learning thresholds and parameters
        self.confidence_threshold = 0.7
        self.min_samples_for_learning = 5
        self.effectiveness_decay = 0.95  # For aging old patterns
    
    async def initialize(self) -> bool:
        """Initialize knowledge learner"""
        try:
            self.logger.info("🎓 Initializing knowledge learner...")
            
            # Setup knowledge database
            await self._setup_knowledge_database()
            
            # Load existing knowledge
            await self._load_existing_knowledge()
            
            # Initialize learning models
            await self._initialize_learning_models()
            
            self.logger.info("✅ Knowledge learner initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Knowledge learner initialization failed: {e}")
            return False
    
    async def _setup_knowledge_database(self):
        """Setup SQLite database for knowledge storage"""
        self.knowledge_db_path = Path(self.config.storage.results_dir) / "knowledge.db"
        
        conn = sqlite3.connect(self.knowledge_db_path)
        cursor = conn.cursor()
        
        # Knowledge patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                first_learned TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                context TEXT
            )
        """)
        
        # Learning metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT
            )
        """)
        
        # Target intelligence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS target_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_hash TEXT UNIQUE NOT NULL,
                target_info TEXT NOT NULL,
                successful_techniques TEXT,
                failed_techniques TEXT,
                risk_profile TEXT,
                last_assessed TEXT NOT NULL,
                assessment_count INTEGER DEFAULT 1
            )
        """)
        
        # Tool effectiveness table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_effectiveness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                category TEXT NOT NULL,
                target_type TEXT,
                success_rate REAL DEFAULT 0.0,
                avg_duration REAL DEFAULT 0.0,
                false_positive_rate REAL DEFAULT 0.0,
                last_updated TEXT NOT NULL,
                sample_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"🧠 Knowledge database initialized: {self.knowledge_db_path}")
    
    async def _load_existing_knowledge(self):
        """Load existing knowledge patterns from database"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM knowledge_patterns")
            pattern_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM target_intelligence")
            target_count = cursor.fetchone()[0]
            
            conn.close()
            
            self.logger.info(f"📚 Loaded {pattern_count} knowledge patterns, {target_count} target profiles")
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing knowledge: {e}")
    
    async def _initialize_learning_models(self):
        """Initialize learning models and algorithms"""
        # Simple effectiveness tracking for now
        # Can be extended with more sophisticated ML models
        self.category_models = {
            "web": {"accuracy": 0.7, "sample_count": 0},
            "api": {"accuracy": 0.65, "sample_count": 0},
            "cloud": {"accuracy": 0.6, "sample_count": 0},
            "infrastructure": {"accuracy": 0.75, "sample_count": 0},
            "identity": {"accuracy": 0.6, "sample_count": 0},
            "supply_chain": {"accuracy": 0.55, "sample_count": 0}
        }
    
    async def get_current_knowledge(self, category: str) -> Dict:
        """Get current knowledge for a specific category"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            # Get effective patterns
            cursor.execute("""
                SELECT pattern_type, pattern_data, effectiveness_score, confidence
                FROM knowledge_patterns
                WHERE category = ? AND effectiveness_score > 0.5
                ORDER BY effectiveness_score DESC
                LIMIT 20
            """, (category,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    "type": row[0],
                    "data": json.loads(row[1]),
                    "effectiveness": row[2],
                    "confidence": row[3]
                })
            
            # Get tool effectiveness for category
            cursor.execute("""
                SELECT tool_name, success_rate, avg_duration, false_positive_rate
                FROM tool_effectiveness
                WHERE category = ?
                ORDER BY success_rate DESC
            """, (category,))
            
            tool_stats = {}
            for row in cursor.fetchall():
                tool_stats[row[0]] = {
                    "success_rate": row[1],
                    "avg_duration": row[2],
                    "false_positive_rate": row[3]
                }
            
            # Get category model performance
            model_stats = self.category_models.get(category, {})
            
            conn.close()
            
            return {
                "category": category,
                "effective_patterns": patterns,
                "tool_effectiveness": tool_stats,
                "model_performance": model_stats,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current knowledge: {e}")
            return {"category": category, "effective_patterns": [], "tool_effectiveness": {}}
    
    async def process_results(self, results: List[Dict], plan: Dict) -> List[Dict]:
        """Process results and update knowledge base"""
        try:
            learning_updates = []
            
            for result in results:
                # Extract learning signals
                updates = await self._extract_learning_signals(result, plan)
                learning_updates.extend(updates)
            
            # Update knowledge patterns
            await self._update_knowledge_patterns(learning_updates)
            
            # Update tool effectiveness
            await self._update_tool_effectiveness(results)
            
            # Update target intelligence
            await self._update_target_intelligence(results)
            
            # Update category models
            await self._update_category_models(results, plan)
            
            self.learning_stats["total_updates"] += len(learning_updates)
            
            return learning_updates
            
        except Exception as e:
            self.logger.error(f"Result processing failed: {e}")
            return []
    
    async def _extract_learning_signals(self, result: Dict, plan: Dict) -> List[Dict]:
        """Extract learning signals from execution results"""
        signals = []
        
        try:
            task = result.get("task", {})
            execution = result.get("execution", {})
            analysis = result.get("analysis", {})
            
            # Success/failure patterns
            was_successful = (
                result.get("status") == "completed" and
                execution.get("returncode", -1) == 0 and
                len(analysis.get("vulnerabilities", [])) > 0
            )
            
            # Tool + parameter combination
            tool_pattern = {
                "tool": task.get("tool"),
                "parameters": task.get("parameters", {}),
                "target_type": self._classify_target_type(task.get("target", "")),
                "success": was_successful,
                "duration": execution.get("duration", 0),
                "vulnerabilities_found": len(analysis.get("vulnerabilities", []))
            }
            
            signals.append({
                "type": "tool_effectiveness",
                "category": plan.get("category", "unknown"),
                "pattern": tool_pattern,
                "timestamp": datetime.now().isoformat()
            })
            
            # Vulnerability detection patterns
            for vuln in analysis.get("vulnerabilities", []):
                vuln_pattern = {
                    "vulnerability_type": vuln.get("type"),
                    "detection_method": vuln.get("source", "unknown"),
                    "confidence": vuln.get("confidence", 0.5),
                    "severity": vuln.get("severity", "low"),
                    "tool_used": task.get("tool")
                }
                
                signals.append({
                    "type": "vulnerability_detection",
                    "category": plan.get("category", "unknown"),
                    "pattern": vuln_pattern,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Error patterns (for improvement)
            if execution.get("returncode", 0) != 0:
                error_pattern = {
                    "tool": task.get("tool"),
                    "error_type": "execution_failure",
                    "stderr": execution.get("stderr", "")[:500],  # Limit size
                    "parameters": task.get("parameters", {})
                }
                
                signals.append({
                    "type": "error_pattern",
                    "category": plan.get("category", "unknown"),
                    "pattern": error_pattern,
                    "timestamp": datetime.now().isoformat()
                })
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"Signal extraction failed: {e}")
            return []
    
    def _classify_target_type(self, target: str) -> str:
        """Classify target type for learning purposes"""
        if not target:
            return "unknown"
        
        # Simple classification
        if "://" in target:
            return "url"
        elif target.count(".") == 3 and all(part.isdigit() for part in target.split(".")):
            return "ip"
        elif "." in target:
            return "domain"
        else:
            return "other"
    
    async def _update_knowledge_patterns(self, learning_updates: List[Dict]):
        """Update knowledge patterns based on learning signals"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            for update in learning_updates:
                pattern_data = json.dumps(update["pattern"], sort_keys=True)
                category = update["category"]
                pattern_type = update["type"]
                
                # Check if pattern exists
                cursor.execute("""
                    SELECT id, success_count, failure_count, effectiveness_score
                    FROM knowledge_patterns
                    WHERE category = ? AND pattern_type = ? AND pattern_data = ?
                """, (category, pattern_type, pattern_data))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing pattern
                    pattern_id, success_count, failure_count, effectiveness = existing
                    
                    # Determine if this was a success or failure
                    was_success = update["pattern"].get("success", False)
                    
                    if was_success:
                        success_count += 1
                        self.learning_stats["successful_patterns"] += 1
                    else:
                        failure_count += 1
                        self.learning_stats["failed_patterns"] += 1
                    
                    # Calculate new effectiveness score
                    total_attempts = success_count + failure_count
                    new_effectiveness = success_count / total_attempts if total_attempts > 0 else 0
                    
                    # Apply decay to old effectiveness
                    aged_effectiveness = effectiveness * self.effectiveness_decay
                    combined_effectiveness = (new_effectiveness + aged_effectiveness) / 2
                    
                    # Calculate confidence based on sample size
                    confidence = min(1.0, total_attempts / self.min_samples_for_learning)
                    
                    cursor.execute("""
                        UPDATE knowledge_patterns
                        SET success_count = ?, failure_count = ?, 
                            effectiveness_score = ?, confidence = ?,
                            last_updated = ?
                        WHERE id = ?
                    """, (
                        success_count, failure_count, combined_effectiveness,
                        confidence, datetime.now().isoformat(), pattern_id
                    ))
                    
                else:
                    # Insert new pattern
                    was_success = update["pattern"].get("success", False)
                    initial_success = 1 if was_success else 0
                    initial_failure = 0 if was_success else 1
                    
                    cursor.execute("""
                        INSERT INTO knowledge_patterns (
                            category, pattern_type, pattern_data,
                            success_count, failure_count, effectiveness_score,
                            confidence, first_learned, last_updated,
                            context
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        category, pattern_type, pattern_data,
                        initial_success, initial_failure, 0.5,  # Neutral starting score
                        0.1,  # Low initial confidence
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        json.dumps({"initial_context": True})
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Knowledge pattern update failed: {e}")
    
    async def _update_tool_effectiveness(self, results: List[Dict]):
        """Update tool effectiveness metrics"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            # Group results by tool and category
            tool_stats = defaultdict(lambda: {
                "successes": 0, "failures": 0, "total_duration": 0,
                "false_positives": 0, "true_positives": 0
            })
            
            for result in results:
                task = result.get("task", {})
                execution = result.get("execution", {})
                analysis = result.get("analysis", {})
                
                tool = task.get("tool", "unknown")
                category = task.get("type", "unknown")
                key = (tool, category)
                
                # Determine success criteria
                was_successful = (
                    result.get("status") == "completed" and
                    execution.get("returncode", -1) == 0
                )
                
                found_vulnerabilities = len(analysis.get("vulnerabilities", [])) > 0
                
                if was_successful:
                    tool_stats[key]["successes"] += 1
                    if found_vulnerabilities:
                        tool_stats[key]["true_positives"] += 1
                else:
                    tool_stats[key]["failures"] += 1
                
                tool_stats[key]["total_duration"] += execution.get("duration", 0)
            
            # Update database
            for (tool, category), stats in tool_stats.items():
                total_runs = stats["successes"] + stats["failures"]
                success_rate = stats["successes"] / total_runs if total_runs > 0 else 0
                avg_duration = stats["total_duration"] / total_runs if total_runs > 0 else 0
                
                # Check if record exists
                cursor.execute("""
                    SELECT success_rate, sample_count FROM tool_effectiveness
                    WHERE tool_name = ? AND category = ?
                """, (tool, category))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update with weighted average
                    old_success_rate, old_sample_count = existing
                    total_samples = old_sample_count + total_runs
                    
                    # Weighted average
                    new_success_rate = (
                        (old_success_rate * old_sample_count + success_rate * total_runs) /
                        total_samples
                    )
                    
                    cursor.execute("""
                        UPDATE tool_effectiveness
                        SET success_rate = ?, avg_duration = ?,
                            last_updated = ?, sample_count = ?
                        WHERE tool_name = ? AND category = ?
                    """, (
                        new_success_rate, avg_duration,
                        datetime.now().isoformat(), total_samples,
                        tool, category
                    ))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO tool_effectiveness (
                            tool_name, category, success_rate, avg_duration,
                            last_updated, sample_count
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        tool, category, success_rate, avg_duration,
                        datetime.now().isoformat(), total_runs
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Tool effectiveness update failed: {e}")
    
    async def _update_target_intelligence(self, results: List[Dict]):
        """Update target intelligence profiles"""
        try:
            # Group results by target
            target_profiles = defaultdict(lambda: {
                "successful_techniques": [],
                "failed_techniques": [],
                "vulnerabilities": [],
                "risk_indicators": []
            })
            
            for result in results:
                task = result.get("task", {})
                execution = result.get("execution", {})
                analysis = result.get("analysis", {})
                target = task.get("target", "")
                
                if not target or target == "auto":
                    continue
                
                # Create target hash for privacy
                import hashlib
                target_hash = hashlib.sha256(target.encode()).hexdigest()[:16]
                
                technique = {
                    "tool": task.get("tool"),
                    "type": task.get("type"),
                    "parameters": task.get("parameters", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                was_successful = (
                    result.get("status") == "completed" and
                    execution.get("returncode", -1) == 0 and
                    len(analysis.get("vulnerabilities", [])) > 0
                )
                
                if was_successful:
                    target_profiles[target_hash]["successful_techniques"].append(technique)
                    target_profiles[target_hash]["vulnerabilities"].extend(
                        analysis.get("vulnerabilities", [])
                    )
                else:
                    target_profiles[target_hash]["failed_techniques"].append(technique)
            
            # Update database
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            for target_hash, profile in target_profiles.items():
                # Calculate risk profile
                vuln_count = len(profile["vulnerabilities"])
                success_rate = len(profile["successful_techniques"]) / (
                    len(profile["successful_techniques"]) + len(profile["failed_techniques"])
                ) if (len(profile["successful_techniques"]) + len(profile["failed_techniques"])) > 0 else 0
                
                risk_profile = {
                    "vulnerability_count": vuln_count,
                    "success_rate": success_rate,
                    "risk_level": "high" if vuln_count > 3 else "medium" if vuln_count > 1 else "low"
                }
                
                # Check if target exists
                cursor.execute("""
                    SELECT id, assessment_count FROM target_intelligence
                    WHERE target_hash = ?
                """, (target_hash,))
                
                existing = cursor.fetchone()
                
                target_info = {
                    "target_type": self._classify_target_type(target_hash),
                    "last_assessment": datetime.now().isoformat()
                }
                
                if existing:
                    # Update existing
                    target_id, assessment_count = existing
                    cursor.execute("""
                        UPDATE target_intelligence
                        SET successful_techniques = ?, failed_techniques = ?,
                            risk_profile = ?, last_assessed = ?,
                            assessment_count = ?
                        WHERE id = ?
                    """, (
                        json.dumps(profile["successful_techniques"]),
                        json.dumps(profile["failed_techniques"]),
                        json.dumps(risk_profile),
                        datetime.now().isoformat(),
                        assessment_count + 1,
                        target_id
                    ))
                else:
                    # Insert new
                    cursor.execute("""
                        INSERT INTO target_intelligence (
                            target_hash, target_info, successful_techniques,
                            failed_techniques, risk_profile, last_assessed
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        target_hash,
                        json.dumps(target_info),
                        json.dumps(profile["successful_techniques"]),
                        json.dumps(profile["failed_techniques"]),
                        json.dumps(risk_profile),
                        datetime.now().isoformat()
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Target intelligence update failed: {e}")
    
    async def _update_category_models(self, results: List[Dict], plan: Dict):
        """Update category-specific model performance"""
        try:
            category = plan.get("category", "unknown")
            
            if category not in self.category_models:
                return
            
            # Calculate success metrics
            total_tasks = len(results)
            successful_tasks = sum(
                1 for r in results
                if r.get("status") == "completed" and
                r.get("execution", {}).get("returncode", -1) == 0
            )
            
            vulnerabilities_found = sum(
                len(r.get("analysis", {}).get("vulnerabilities", []))
                for r in results
            )
            
            # Update model statistics
            model = self.category_models[category]
            old_sample_count = model["sample_count"]
            old_accuracy = model["accuracy"]
            
            # Calculate new accuracy (weighted average)
            current_accuracy = successful_tasks / total_tasks if total_tasks > 0 else 0
            new_sample_count = old_sample_count + total_tasks
            
            if new_sample_count > 0:
                new_accuracy = (
                    (old_accuracy * old_sample_count + current_accuracy * total_tasks) /
                    new_sample_count
                )
            else:
                new_accuracy = current_accuracy
            
            model["accuracy"] = new_accuracy
            model["sample_count"] = new_sample_count
            
            # Log metrics
            await self._log_learning_metric(
                category, "accuracy", new_accuracy
            )
            await self._log_learning_metric(
                category, "vulnerabilities_per_cycle", vulnerabilities_found
            )
            
        except Exception as e:
            self.logger.error(f"Category model update failed: {e}")
    
    async def _log_learning_metric(self, category: str, metric_name: str, value: float):
        """Log learning metric to database"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO learning_metrics (
                    category, metric_name, metric_value, timestamp
                ) VALUES (?, ?, ?, ?)
            """, (
                category, metric_name, value, datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Metric logging failed: {e}")
    
    async def retrain_models(self):
        """Trigger model retraining based on accumulated knowledge"""
        try:
            self.logger.info("🔄 Triggering model retraining...")
            
            # For now, this is a placeholder for more sophisticated retraining
            # In a full implementation, this would:
            # 1. Export training data from knowledge base
            # 2. Retrain local LLM with LoRA fine-tuning
            # 3. Evaluate model performance
            # 4. Deploy improved model if better
            
            for category, model in self.category_models.items():
                if model["sample_count"] >= self.min_samples_for_learning:
                    self.logger.info(f"📚 Retraining {category} model with {model['sample_count']} samples")
                    # Placeholder for actual retraining logic
                    self.learning_stats["model_improvements"] += 1
            
            await self._log_learning_metric("global", "retrain_cycles", 1)
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    async def sync_with_gcp(self, learning_updates: List[Dict]) -> bool:
        """Sync learning data with GCP (if enabled)"""
        if not self.config.gcp.enabled:
            return False
        
        try:
            # Placeholder for GCP sync
            # In full implementation, would sync to:
            # - BigQuery ML for training data
            # - Vertex AI for model updates
            # - Cloud Storage for model artifacts
            
            self.logger.info(f"📤 Syncing {len(learning_updates)} learning updates to GCP")
            return True
            
        except Exception as e:
            self.logger.error(f"GCP sync failed: {e}")
            return False
    
    async def get_learning_statistics(self) -> Dict:
        """Get current learning statistics"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            # Get pattern counts
            cursor.execute("SELECT COUNT(*) FROM knowledge_patterns")
            total_patterns = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM knowledge_patterns WHERE effectiveness_score > 0.7")
            effective_patterns = cursor.fetchone()[0]
            
            # Get recent learning activity
            since = datetime.now() - timedelta(days=7)
            cursor.execute("""
                SELECT COUNT(*) FROM learning_metrics 
                WHERE timestamp > ?
            """, (since.isoformat(),))
            recent_updates = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "learning_stats": self.learning_stats,
                "knowledge_base": {
                    "total_patterns": total_patterns,
                    "effective_patterns": effective_patterns,
                    "effectiveness_ratio": effective_patterns / total_patterns if total_patterns > 0 else 0
                },
                "category_models": self.category_models,
                "recent_activity": {
                    "updates_last_7_days": recent_updates
                },
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get learning statistics: {e}")
            return {}
    
    async def shutdown(self) -> None:
        """Shutdown the learner"""
        try:
            # Save final statistics
            stats = await self.get_learning_statistics()
            
            # Log final metrics
            for category, model in self.category_models.items():
                await self._log_learning_metric(
                    category, "final_accuracy", model["accuracy"]
                )
            
            self.logger.info(f"📊 Learning session complete: {json.dumps(self.learning_stats, indent=2)}")
            self.logger.info("✅ Knowledge learner shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Learner shutdown error: {e}")