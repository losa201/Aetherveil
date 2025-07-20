#!/usr/bin/env python3
"""
Result Collector Module - Collects, analyzes, and stores pentesting results
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiofiles
from google.cloud import bigquery
from google.cloud import storage

from config import Config

class ResultCollector:
    """Collects and manages pentesting results"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = None
        self.bigquery_client = None
        self.storage_client = None
        
        # Result analysis patterns
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.interesting_patterns = self._load_interesting_patterns()
    
    async def initialize(self) -> bool:
        """Initialize result collector"""
        try:
            self.logger.info("📊 Initializing result collector...")
            
            # Setup local SQLite database
            await self._setup_local_database()
            
            # Initialize GCP clients if enabled
            if self.config.gcp.enabled:
                await self._setup_gcp_clients()
            
            # Create result directories
            await self._setup_directories()
            
            self.logger.info("✅ Result collector initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Result collector initialization failed: {e}")
            return False
    
    async def _setup_local_database(self):
        """Setup local SQLite database for results"""
        self.db_path = Path(self.config.storage.results_dir) / "results.db"
        
        # Create database and tables
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                tool TEXT NOT NULL,
                category TEXT NOT NULL,
                target TEXT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                duration REAL,
                findings_count INTEGER DEFAULT 0,
                vulnerabilities_count INTEGER DEFAULT 0,
                data TEXT NOT NULL,
                hash TEXT UNIQUE,
                synced_to_gcp INTEGER DEFAULT 0
            )
        """)
        
        # Vulnerabilities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER,
                type TEXT NOT NULL,
                severity TEXT NOT NULL,
                target TEXT,
                details TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                confirmed INTEGER DEFAULT 0,
                FOREIGN KEY (result_id) REFERENCES results (id)
            )
        """)
        
        # Learning data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                success_pattern TEXT,
                failure_pattern TEXT,
                effectiveness_score REAL,
                timestamp TEXT NOT NULL,
                context TEXT
            )
        """)
        
        # Plans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                category TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                plan_data TEXT NOT NULL,
                executed INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"📊 Local database initialized: {self.db_path}")
    
    async def _setup_gcp_clients(self):
        """Setup GCP BigQuery and Storage clients"""
        try:
            if self.config.gcp.service_account_path:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.gcp.service_account_path
                )
                self.bigquery_client = bigquery.Client(credentials=credentials)
                self.storage_client = storage.Client(credentials=credentials)
            else:
                # Use default credentials
                self.bigquery_client = bigquery.Client()
                self.storage_client = storage.Client()
            
            self.logger.info("✅ GCP clients initialized")
            
        except Exception as e:
            self.logger.warning(f"GCP client initialization failed: {e}")
            self.config.gcp.enabled = False
    
    async def _setup_directories(self):
        """Setup result storage directories"""
        directories = [
            self.config.storage.results_dir,
            self.config.storage.reports_dir,
            f"{self.config.storage.results_dir}/raw",
            f"{self.config.storage.results_dir}/processed"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_vulnerability_patterns(self) -> Dict[str, List[Dict]]:
        """Load patterns for vulnerability detection"""
        return {
            "sql_injection": [
                {"pattern": "injectable", "confidence": 0.9},
                {"pattern": "mysql error", "confidence": 0.7},
                {"pattern": "oracle error", "confidence": 0.7},
                {"pattern": "postgresql error", "confidence": 0.7}
            ],
            "xss": [
                {"pattern": "<script>", "confidence": 0.8},
                {"pattern": "javascript:", "confidence": 0.6},
                {"pattern": "onerror=", "confidence": 0.7}
            ],
            "directory_traversal": [
                {"pattern": "../", "confidence": 0.6},
                {"pattern": "..\\", "confidence": 0.6},
                {"pattern": "/etc/passwd", "confidence": 0.9}
            ],
            "information_disclosure": [
                {"pattern": "server version", "confidence": 0.5},
                {"pattern": "debug mode", "confidence": 0.7},
                {"pattern": "stack trace", "confidence": 0.6}
            ]
        }
    
    def _load_interesting_patterns(self) -> List[Dict]:
        """Load patterns for interesting findings"""
        return [
            {"pattern": "admin", "score": 0.8},
            {"pattern": "login", "score": 0.7},
            {"pattern": "api", "score": 0.6},
            {"pattern": "config", "score": 0.7},
            {"pattern": "backup", "score": 0.8},
            {"pattern": "test", "score": 0.5},
            {"pattern": "dev", "score": 0.6},
            {"pattern": "upload", "score": 0.7}
        ]
    
    async def analyze_result(self, result: Dict) -> Optional[Dict]:
        """Analyze execution result for vulnerabilities and interesting findings"""
        try:
            if not result:
                return None
            
            # Create analyzed result structure
            analyzed = {
                "original_result": result,
                "analysis": {
                    "timestamp": datetime.now().isoformat(),
                    "vulnerabilities": [],
                    "interesting_findings": [],
                    "confidence_scores": {},
                    "risk_assessment": "low"
                },
                "metadata": {
                    "analyzer_version": "1.0",
                    "analysis_duration": 0
                }
            }
            
            analysis_start = datetime.now()
            
            # Extract existing vulnerabilities from result
            existing_vulns = result.get("vulnerabilities", [])
            analyzed["analysis"]["vulnerabilities"].extend(existing_vulns)
            
            # Analyze findings for additional patterns
            findings = result.get("findings", [])
            for finding in findings:
                await self._analyze_finding(finding, analyzed["analysis"])
            
            # Analyze raw output for patterns
            execution = result.get("execution", {})
            stdout = execution.get("stdout", "")
            stderr = execution.get("stderr", "")
            
            await self._analyze_text_output(stdout, analyzed["analysis"])
            await self._analyze_text_output(stderr, analyzed["analysis"])
            
            # Calculate overall risk assessment
            analyzed["analysis"]["risk_assessment"] = self._calculate_risk_assessment(
                analyzed["analysis"]["vulnerabilities"]
            )
            
            # Calculate analysis duration
            analysis_duration = (datetime.now() - analysis_start).total_seconds()
            analyzed["metadata"]["analysis_duration"] = analysis_duration
            
            return analyzed
            
        except Exception as e:
            self.logger.error(f"Result analysis failed: {e}")
            return result  # Return original if analysis fails
    
    async def _analyze_finding(self, finding: Dict, analysis: Dict):
        """Analyze individual finding for vulnerabilities"""
        try:
            finding_text = json.dumps(finding).lower()
            
            # Check for vulnerability patterns
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern_info in patterns:
                    pattern = pattern_info["pattern"]
                    confidence = pattern_info["confidence"]
                    
                    if pattern in finding_text:
                        vulnerability = {
                            "type": vuln_type,
                            "pattern_matched": pattern,
                            "confidence": confidence,
                            "finding_source": finding.get("type", "unknown"),
                            "severity": self._map_confidence_to_severity(confidence),
                            "details": f"Pattern '{pattern}' matched in {finding.get('type', 'finding')}"
                        }
                        analysis["vulnerabilities"].append(vulnerability)
            
            # Check for interesting patterns
            for pattern_info in self.interesting_patterns:
                pattern = pattern_info["pattern"]
                score = pattern_info["score"]
                
                if pattern in finding_text:
                    interesting = {
                        "pattern": pattern,
                        "score": score,
                        "finding_source": finding.get("type", "unknown"),
                        "context": finding_text[:200]  # Limited context
                    }
                    analysis["interesting_findings"].append(interesting)
            
        except Exception as e:
            self.logger.warning(f"Finding analysis failed: {e}")
    
    async def _analyze_text_output(self, text: str, analysis: Dict):
        """Analyze text output for vulnerability patterns"""
        try:
            text_lower = text.lower()
            
            # Check for vulnerability patterns in output
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern_info in patterns:
                    pattern = pattern_info["pattern"]
                    confidence = pattern_info["confidence"]
                    
                    if pattern in text_lower:
                        vulnerability = {
                            "type": vuln_type,
                            "pattern_matched": pattern,
                            "confidence": confidence,
                            "source": "tool_output",
                            "severity": self._map_confidence_to_severity(confidence),
                            "details": f"Pattern '{pattern}' found in tool output"
                        }
                        analysis["vulnerabilities"].append(vulnerability)
            
        except Exception as e:
            self.logger.warning(f"Text analysis failed: {e}")
    
    def _map_confidence_to_severity(self, confidence: float) -> str:
        """Map confidence score to severity level"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_risk_assessment(self, vulnerabilities: List[Dict]) -> str:
        """Calculate overall risk assessment"""
        if not vulnerabilities:
            return "low"
        
        severity_scores = {"low": 1, "medium": 3, "high": 5, "critical": 8}
        total_score = sum(severity_scores.get(v.get("severity", "low"), 1) for v in vulnerabilities)
        
        if total_score >= 15:
            return "critical"
        elif total_score >= 8:
            return "high"
        elif total_score >= 3:
            return "medium"
        else:
            return "low"
    
    async def store_result(self, result: Dict, session_id: str) -> bool:
        """Store result in local database"""
        try:
            # Generate unique hash for deduplication
            result_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest()
            
            # Extract metadata
            task = result.get("task", {})
            execution = result.get("execution", {})
            analysis = result.get("analysis", {})
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # Insert main result
                cursor.execute("""
                    INSERT OR IGNORE INTO results (
                        session_id, task_id, tool, category, target,
                        timestamp, status, duration, findings_count,
                        vulnerabilities_count, data, hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    task.get("name", "unknown"),
                    task.get("tool", "unknown"),
                    task.get("type", "unknown"),
                    task.get("target", ""),
                    datetime.now().isoformat(),
                    result.get("status", "unknown"),
                    execution.get("duration", 0),
                    len(result.get("findings", [])),
                    len(analysis.get("vulnerabilities", [])),
                    json.dumps(result),
                    result_hash
                ))
                
                result_id = cursor.lastrowid
                
                # Insert vulnerabilities
                for vuln in analysis.get("vulnerabilities", []):
                    cursor.execute("""
                        INSERT INTO vulnerabilities (
                            result_id, type, severity, target, details,
                            first_seen, last_seen
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result_id,
                        vuln.get("type", "unknown"),
                        vuln.get("severity", "low"),
                        task.get("target", ""),
                        json.dumps(vuln),
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                self.logger.debug(f"📊 Stored result with ID: {result_id}")
                return True
                
            except sqlite3.IntegrityError:
                # Duplicate result
                self.logger.debug("Duplicate result, skipping storage")
                return True
                
        except Exception as e:
            self.logger.error(f"Result storage failed: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    async def log_plan(self, plan: Dict, session_id: str) -> bool:
        """Log planning data for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO plans (session_id, category, timestamp, plan_data)
                VALUES (?, ?, ?, ?)
            """, (
                session_id,
                plan.get("category", "unknown"),
                datetime.now().isoformat(),
                json.dumps(plan)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Plan logging failed: {e}")
            return False
    
    async def get_recent_results(self, hours: int = 24) -> List[Dict]:
        """Get recent results for context"""
        try:
            since = datetime.now() - timedelta(hours=hours)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data FROM results 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """, (since.isoformat(),))
            
            results = []
            for row in cursor.fetchall():
                try:
                    result = json.loads(row[0])
                    results.append(result)
                except json.JSONDecodeError:
                    continue
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get recent results: {e}")
            return []
    
    async def get_recent_findings(self, hours: int = 24) -> List[Dict]:
        """Get recent interesting findings"""
        try:
            since = datetime.now() - timedelta(hours=hours)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT type, severity, target, details, first_seen
                FROM vulnerabilities 
                WHERE first_seen > ? 
                ORDER BY first_seen DESC
            """, (since.isoformat(),))
            
            findings = []
            for row in cursor.fetchall():
                findings.append({
                    "type": row[0],
                    "severity": row[1],
                    "target": row[2],
                    "details": json.loads(row[3]) if row[3] else {},
                    "timestamp": row[4]
                })
            
            conn.close()
            return findings
            
        except Exception as e:
            self.logger.error(f"Failed to get recent findings: {e}")
            return []
    
    async def send_to_gcp(self, result: Dict) -> bool:
        """Send result to GCP BigQuery"""
        if not self.config.gcp.enabled or not self.bigquery_client:
            return False
        
        try:
            # Prepare data for BigQuery
            bq_data = {
                "timestamp": datetime.now().isoformat(),
                "session_id": result.get("session_id", "unknown"),
                "tool": result.get("task", {}).get("tool", "unknown"),
                "category": result.get("task", {}).get("type", "unknown"),
                "target": result.get("task", {}).get("target", ""),
                "status": result.get("status", "unknown"),
                "duration": result.get("execution", {}).get("duration", 0),
                "findings_count": len(result.get("findings", [])),
                "vulnerabilities_count": len(result.get("analysis", {}).get("vulnerabilities", [])),
                "risk_assessment": result.get("analysis", {}).get("risk_assessment", "low"),
                "raw_data": json.dumps(result)
            }
            
            # Insert into BigQuery table
            table_id = f"{self.config.gcp.project_id}.{self.config.gcp.dataset_id}.pentesting_results"
            
            errors = self.bigquery_client.insert_rows_json(
                table_id, [bq_data]
            )
            
            if not errors:
                self.logger.debug("📊 Result sent to BigQuery")
                
                # Mark as synced in local database
                await self._mark_as_synced(result)
                return True
            else:
                self.logger.warning(f"BigQuery insert errors: {errors}")
                return False
                
        except Exception as e:
            self.logger.error(f"GCP sync failed: {e}")
            return False
    
    async def _mark_as_synced(self, result: Dict):
        """Mark result as synced to GCP"""
        try:
            result_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE results SET synced_to_gcp = 1 WHERE hash = ?
            """, (result_hash,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to mark as synced: {e}")
    
    async def generate_summary_report(self, session_id: str) -> Dict:
        """Generate summary report for session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get session statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_tasks,
                    SUM(findings_count) as total_findings,
                    SUM(vulnerabilities_count) as total_vulnerabilities,
                    AVG(duration) as avg_duration,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time
                FROM results WHERE session_id = ?
            """, (session_id,))
            
            stats = cursor.fetchone()
            
            # Get vulnerability breakdown
            cursor.execute("""
                SELECT severity, COUNT(*) as count
                FROM vulnerabilities v
                JOIN results r ON v.result_id = r.id
                WHERE r.session_id = ?
                GROUP BY severity
            """, (session_id,))
            
            vuln_breakdown = dict(cursor.fetchall())
            
            # Get tool usage
            cursor.execute("""
                SELECT tool, COUNT(*) as count
                FROM results
                WHERE session_id = ?
                GROUP BY tool
            """, (session_id,))
            
            tool_usage = dict(cursor.fetchall())
            
            conn.close()
            
            report = {
                "session_id": session_id,
                "generated_at": datetime.now().isoformat(),
                "statistics": {
                    "total_tasks": stats[0] or 0,
                    "total_findings": stats[1] or 0,
                    "total_vulnerabilities": stats[2] or 0,
                    "average_duration": stats[3] or 0,
                    "start_time": stats[4],
                    "end_time": stats[5]
                },
                "vulnerability_breakdown": vuln_breakdown,
                "tool_usage": tool_usage
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {}
    
    async def cleanup_old_data(self, days: int = 30):
        """Cleanup old data from local database"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old results
            cursor.execute("""
                DELETE FROM results WHERE timestamp < ?
            """, (cutoff.isoformat(),))
            
            # Delete orphaned vulnerabilities
            cursor.execute("""
                DELETE FROM vulnerabilities 
                WHERE result_id NOT IN (SELECT id FROM results)
            """)
            
            # Delete old plans
            cursor.execute("""
                DELETE FROM plans WHERE timestamp < ?
            """, (cutoff.isoformat(),))
            
            conn.commit()
            
            deleted_count = cursor.rowcount
            conn.close()
            
            self.logger.info(f"🧹 Cleaned up {deleted_count} old records")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the collector"""
        try:
            # Sync any pending results to GCP
            if self.config.gcp.enabled:
                await self._sync_pending_results()
            
            # Cleanup old data
            await self.cleanup_old_data()
            
            self.logger.info("✅ Result collector shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Collector shutdown error: {e}")
    
    async def _sync_pending_results(self):
        """Sync pending results to GCP"""
        try:
            if not self.bigquery_client:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data FROM results 
                WHERE synced_to_gcp = 0 
                LIMIT 100
            """)
            
            pending_results = []
            for row in cursor.fetchall():
                try:
                    result = json.loads(row[0])
                    pending_results.append(result)
                except json.JSONDecodeError:
                    continue
            
            conn.close()
            
            # Sync to GCP
            for result in pending_results:
                await self.send_to_gcp(result)
            
            self.logger.info(f"📊 Synced {len(pending_results)} pending results to GCP")
            
        except Exception as e:
            self.logger.warning(f"Pending sync failed: {e}")