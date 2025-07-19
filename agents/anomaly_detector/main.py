"""
Log Anomaly Detector Agent
BigQuery ML-powered failure prediction and log analysis
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, BackgroundTasks
from google.cloud import bigquery, pubsub_v1, logging as cloud_logging
from pydantic import BaseModel, Field
import re
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.getenv("PROJECT_ID", "tidy-computing-465909-i3")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "pipeline_analytics")
ML_DATASET = os.getenv("ML_DATASET", "ml_models")

# Initialize clients
app = FastAPI(
    title="Log Anomaly Detector Agent",
    description="ML-powered log analysis and failure prediction",
    version="1.0.0"
)
bigquery_client = bigquery.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()
logging_client = cloud_logging.Client(project=PROJECT_ID)

class LogEntry(BaseModel):
    timestamp: datetime
    severity: str
    message: str
    source: str
    labels: Dict[str, str] = Field(default_factory=dict)
    run_id: Optional[str] = None
    job_name: Optional[str] = None

class AnomalyResult(BaseModel):
    is_anomaly: bool
    confidence: float
    anomaly_type: str
    description: str
    suggested_action: Optional[str] = None

class LogAnomalyDetector:
    """Advanced log analysis and anomaly detection"""
    
    def __init__(self):
        self.error_patterns = [
            r"error|ERROR|Error",
            r"fail|FAIL|Fail|failed|FAILED|Failed",
            r"exception|Exception|EXCEPTION",
            r"timeout|Timeout|TIMEOUT",
            r"denied|Denied|DENIED",
            r"not found|Not Found|NOT FOUND|404",
            r"internal server error|Internal Server Error|500",
            r"out of memory|Out of Memory|OOM",
            r"connection refused|Connection refused",
            r"network unreachable|Network unreachable"
        ]
        
        self.warning_patterns = [
            r"warning|WARNING|Warning|warn|WARN|Warn",
            r"deprecated|Deprecated|DEPRECATED",
            r"retry|Retry|RETRY|retrying|Retrying",
            r"slow|Slow|SLOW|performance",
            r"quota|Quota|QUOTA|limit|Limit|LIMIT"
        ]
        
        self.ml_model_created = False
    
    async def analyze_log_entry(self, log_entry: LogEntry) -> AnomalyResult:
        """Analyze a single log entry for anomalies"""
        
        # Pattern-based analysis
        pattern_result = self._analyze_patterns(log_entry)
        
        # ML-based analysis (if model exists)
        ml_result = await self._analyze_with_ml(log_entry)
        
        # Combine results
        return self._combine_analysis_results(pattern_result, ml_result, log_entry)
    
    def _analyze_patterns(self, log_entry: LogEntry) -> Dict:
        """Pattern-based log analysis"""
        message = log_entry.message.lower()
        
        # Check for error patterns
        for pattern in self.error_patterns:
            if re.search(pattern.lower(), message):
                return {
                    "type": "error_pattern",
                    "confidence": 0.8,
                    "pattern": pattern,
                    "severity": "high"
                }
        
        # Check for warning patterns
        for pattern in self.warning_patterns:
            if re.search(pattern.lower(), message):
                return {
                    "type": "warning_pattern",
                    "confidence": 0.6,
                    "pattern": pattern,
                    "severity": "medium"
                }
        
        # Severity-based analysis
        if log_entry.severity in ["ERROR", "CRITICAL"]:
            return {
                "type": "severity_anomaly",
                "confidence": 0.7,
                "pattern": f"severity_{log_entry.severity}",
                "severity": "high"
            }
        
        return {
            "type": "normal",
            "confidence": 0.1,
            "pattern": "none",
            "severity": "low"
        }
    
    async def _analyze_with_ml(self, log_entry: LogEntry) -> Optional[Dict]:
        """ML-based log analysis using BigQuery ML"""
        if not self.ml_model_created:
            await self._ensure_ml_model_exists()
        
        try:
            # Prepare features for ML model
            features = self._extract_log_features(log_entry)
            
            # Query ML model for prediction
            query = f"""
            SELECT 
                predicted_anomaly,
                predicted_anomaly_probs
            FROM ML.PREDICT(
                MODEL `{PROJECT_ID}.{ML_DATASET}.log_anomaly_model`,
                (
                    SELECT 
                        @message_length as message_length,
                        @error_keywords as error_keywords,
                        @severity_score as severity_score,
                        @timestamp_hour as timestamp_hour
                )
            )
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("message_length", "INT64", features["message_length"]),
                    bigquery.ScalarQueryParameter("error_keywords", "INT64", features["error_keywords"]),
                    bigquery.ScalarQueryParameter("severity_score", "INT64", features["severity_score"]),
                    bigquery.ScalarQueryParameter("timestamp_hour", "INT64", features["timestamp_hour"]),
                ]
            )
            
            query_job = bigquery_client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            if results:
                result = results[0]
                return {
                    "type": "ml_prediction",
                    "confidence": float(result.predicted_anomaly_probs[0]["prob"]),
                    "prediction": result.predicted_anomaly,
                    "severity": "high" if result.predicted_anomaly else "low"
                }
                
        except Exception as e:
            logger.warning(f"ML analysis failed: {e}")
            
        return None
    
    def _extract_log_features(self, log_entry: LogEntry) -> Dict:
        """Extract features from log entry for ML model"""
        message = log_entry.message.lower()
        
        # Count error-related keywords
        error_keywords = sum(1 for pattern in self.error_patterns 
                           if re.search(pattern.lower(), message))
        
        # Severity score mapping
        severity_scores = {
            "DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4, "CRITICAL": 5
        }
        
        return {
            "message_length": len(log_entry.message),
            "error_keywords": error_keywords,
            "severity_score": severity_scores.get(log_entry.severity, 2),
            "timestamp_hour": log_entry.timestamp.hour
        }
    
    async def _ensure_ml_model_exists(self):
        """Ensure BigQuery ML model exists, create if not"""
        try:
            # Check if model exists
            model_ref = bigquery_client.get_model(f"{PROJECT_ID}.{ML_DATASET}.log_anomaly_model")
            self.ml_model_created = True
            logger.info("ML model found")
        except Exception:
            # Create model if it doesn't exist
            await self._create_ml_model()
    
    async def _create_ml_model(self):
        """Create BigQuery ML model for log anomaly detection"""
        
        # First, create training data
        create_training_query = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{ML_DATASET}.log_training_data` AS
        SELECT
            LENGTH(message) as message_length,
            REGEXP_CONTAINS(LOWER(message), r'error|fail|exception|timeout') as error_keywords,
            CASE 
                WHEN severity = 'DEBUG' THEN 1
                WHEN severity = 'INFO' THEN 2
                WHEN severity = 'WARNING' THEN 3
                WHEN severity = 'ERROR' THEN 4
                WHEN severity = 'CRITICAL' THEN 5
                ELSE 2
            END as severity_score,
            EXTRACT(HOUR FROM timestamp) as timestamp_hour,
            CASE 
                WHEN severity IN ('ERROR', 'CRITICAL') OR 
                     REGEXP_CONTAINS(LOWER(message), r'error|fail|exception') THEN true
                ELSE false
            END as label
        FROM `{PROJECT_ID}.{BIGQUERY_DATASET}.pipeline_logs`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        """
        
        # Create the ML model
        create_model_query = f"""
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{ML_DATASET}.log_anomaly_model`
        OPTIONS(
            model_type='LOGISTIC_REG',
            input_label_cols=['label']
        ) AS
        SELECT
            message_length,
            error_keywords,
            severity_score,
            timestamp_hour,
            label
        FROM `{PROJECT_ID}.{ML_DATASET}.log_training_data`
        WHERE label IS NOT NULL
        """
        
        try:
            # Execute queries
            bigquery_client.query(create_training_query).result()
            bigquery_client.query(create_model_query).result()
            self.ml_model_created = True
            logger.info("Created ML model for log anomaly detection")
        except Exception as e:
            logger.error(f"Failed to create ML model: {e}")
    
    def _combine_analysis_results(self, pattern_result: Dict, ml_result: Optional[Dict], 
                                 log_entry: LogEntry) -> AnomalyResult:
        """Combine pattern and ML analysis results"""
        
        # Determine if it's an anomaly
        pattern_confidence = pattern_result.get("confidence", 0)
        ml_confidence = ml_result.get("confidence", 0) if ml_result else 0
        
        # Weighted combination
        combined_confidence = (pattern_confidence * 0.6) + (ml_confidence * 0.4)
        is_anomaly = combined_confidence > 0.5 or pattern_result.get("severity") == "high"
        
        # Determine anomaly type
        if pattern_result.get("type") == "error_pattern":
            anomaly_type = "error_pattern"
            description = f"Error pattern detected: {pattern_result.get('pattern')}"
        elif pattern_result.get("type") == "warning_pattern":
            anomaly_type = "warning_pattern"
            description = f"Warning pattern detected: {pattern_result.get('pattern')}"
        elif ml_result and ml_result.get("prediction"):
            anomaly_type = "ml_anomaly"
            description = "ML model detected anomalous log pattern"
        else:
            anomaly_type = "normal"
            description = "Normal log entry"
        
        # Suggest actions
        suggested_action = self._suggest_action(anomaly_type, log_entry)
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=combined_confidence,
            anomaly_type=anomaly_type,
            description=description,
            suggested_action=suggested_action
        )
    
    def _suggest_action(self, anomaly_type: str, log_entry: LogEntry) -> Optional[str]:
        """Suggest remediation actions based on anomaly type"""
        
        suggestions = {
            "error_pattern": "Investigate error cause and consider adding error handling",
            "warning_pattern": "Monitor for escalation and review configuration",
            "ml_anomaly": "Review log context and check for system issues",
            "severity_anomaly": "Immediate investigation required for critical errors"
        }
        
        return suggestions.get(anomaly_type)

# Initialize detector
anomaly_detector = LogAnomalyDetector()

async def process_log_batch(logs: List[LogEntry]) -> List[AnomalyResult]:
    """Process a batch of log entries"""
    results = []
    
    for log_entry in logs:
        try:
            result = await anomaly_detector.analyze_log_entry(log_entry)
            results.append(result)
            
            # Publish anomaly alerts
            if result.is_anomaly and result.confidence > 0.7:
                await publish_anomaly_alert(log_entry, result)
                
        except Exception as e:
            logger.error(f"Error analyzing log entry: {e}")
            
    return results

async def publish_anomaly_alert(log_entry: LogEntry, anomaly: AnomalyResult):
    """Publish anomaly alert to Pub/Sub"""
    topic_path = publisher.topic_path(PROJECT_ID, "aetherveil-anomaly-alerts")
    
    alert_data = {
        "timestamp": log_entry.timestamp.isoformat(),
        "anomaly_type": anomaly.anomaly_type,
        "confidence": anomaly.confidence,
        "description": anomaly.description,
        "source": log_entry.source,
        "severity": log_entry.severity,
        "message": log_entry.message[:500],  # Truncate long messages
        "run_id": log_entry.run_id,
        "job_name": log_entry.job_name,
        "suggested_action": anomaly.suggested_action
    }
    
    try:
        future = publisher.publish(topic_path, json.dumps(alert_data).encode("utf-8"))
        message_id = future.result()
        logger.info(f"Published anomaly alert: {message_id}")
    except Exception as e:
        logger.error(f"Error publishing alert: {e}")

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/analyze/log")
async def analyze_single_log(log_entry: LogEntry):
    """Analyze a single log entry"""
    try:
        result = await anomaly_detector.analyze_log_entry(log_entry)
        return result
    except Exception as e:
        logger.error(f"Error analyzing log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def analyze_log_batch(logs: List[LogEntry], background_tasks: BackgroundTasks):
    """Analyze a batch of log entries"""
    try:
        background_tasks.add_task(process_log_batch, logs)
        return {"status": "processing", "count": len(logs)}
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pubsub/logs")
async def receive_pubsub_logs(request: dict):
    """Receive logs from Pub/Sub"""
    try:
        # Decode Pub/Sub message
        message_data = base64.b64decode(request["message"]["data"]).decode("utf-8")
        log_data = json.loads(message_data)
        
        # Convert to LogEntry
        log_entry = LogEntry(
            timestamp=datetime.fromisoformat(log_data["timestamp"]),
            severity=log_data.get("severity", "INFO"),
            message=log_data["message"],
            source=log_data.get("source", "unknown"),
            labels=log_data.get("labels", {}),
            run_id=log_data.get("run_id"),
            job_name=log_data.get("job_name")
        )
        
        # Analyze log entry
        result = await anomaly_detector.analyze_log_entry(log_entry)
        
        # Publish alert if anomaly detected
        if result.is_anomaly and result.confidence > 0.7:
            await publish_anomaly_alert(log_entry, result)
        
        return {"status": "processed"}
        
    except Exception as e:
        logger.error(f"Error processing Pub/Sub log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    """Get ML model status"""
    return {
        "model_created": anomaly_detector.ml_model_created,
        "model_path": f"{PROJECT_ID}.{ML_DATASET}.log_anomaly_model"
    }

@app.post("/model/retrain")
async def retrain_model():
    """Retrain the ML model"""
    try:
        await anomaly_detector._create_ml_model()
        return {"status": "model retrained successfully"}
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)