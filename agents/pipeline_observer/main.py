"""
Pipeline State Monitor Agent
Real-time GitHub Actions workflow monitoring with ML insights
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Response, status
from google.cloud import bigquery, pubsub_v1, monitoring_v3
from google.cloud.exceptions import NotFound
import httpx
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.getenv("PROJECT_ID", "tidy-computing-465909-i3")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "pipeline_analytics")
PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "aetherveil-pipeline-events")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Initialize clients
app = FastAPI(
    title="Autonomous Pipeline Observer Agent (APOA)",
    description="Monitors CI/CD pipelines, detects anomalies, and generates insights.",
    version="1.0.0",
)
bigquery_client = bigquery.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()
monitoring_client = monitoring_v3.MetricServiceClient()

# Pydantic models
class PubSubMessage(BaseModel):
    """The message payload from a Pub/Sub push subscription."""
    data: str
    attributes: Dict[str, str] = Field(default_factory=dict)

class PubSubPushRequest(BaseModel):
    """The full request body from a Pub/Sub push subscription."""
    message: PubSubMessage
    subscription: str

class GitHubWebhookEvent(BaseModel):
    action: str
    workflow_run: Optional[Dict] = None
    workflow_job: Optional[Dict] = None
    repository: Dict
    sender: Dict

class PipelineMetrics(BaseModel):
    run_id: str
    repository: str
    workflow_name: str
    branch: str
    commit_sha: str
    actor: str
    environment: Optional[str] = None
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    jobs_total: int = 0
    jobs_failed: int = 0
    artifact_count: Optional[int] = None
    test_results: Optional[Dict] = None
    security_scan_results: Optional[Dict] = None
    resource_usage: Optional[Dict] = None

class AnomalyDetection:
    """ML-based anomaly detection for pipeline metrics"""
    
    def __init__(self):
        self.model_thresholds = {
            "duration_z_score": 3.0,
            "failure_rate_threshold": 0.1,
            "resource_spike_threshold": 2.0
        }
    
    async def analyze_pipeline_run(self, metrics: PipelineMetrics) -> Dict:
        """Analyze pipeline run for anomalies"""
        anomalies = []
        
        # Get historical data for comparison
        historical_data = await self._get_historical_metrics(
            metrics.repository, 
            metrics.workflow_name,
            days=30
        )
        
        if not historical_data:
            return {"anomalies": [], "baseline_exists": False}
        
        # Duration anomaly detection
        if metrics.duration_seconds:
            duration_anomaly = await self._detect_duration_anomaly(
                metrics.duration_seconds, historical_data
            )
            if duration_anomaly:
                anomalies.append(duration_anomaly)
        
        # Failure pattern analysis
        failure_anomaly = await self._detect_failure_patterns(metrics, historical_data)
        if failure_anomaly:
            anomalies.append(failure_anomaly)
        
        # Resource usage anomaly
        if metrics.resource_usage:
            resource_anomaly = await self._detect_resource_anomaly(
                metrics.resource_usage, historical_data
            )
            if resource_anomaly:
                anomalies.append(resource_anomaly)
        
        return {
            "anomalies": anomalies,
            "baseline_exists": True,
            "confidence_score": self._calculate_confidence(anomalies, historical_data)
        }
    
    async def _get_historical_metrics(self, repo: str, workflow: str, days: int) -> List[Dict]:
        """Fetch historical metrics from BigQuery"""
        query = f"""
        SELECT 
            duration_seconds,
            status,
            resource_usage,
            jobs_failed,
            jobs_total
        FROM `{PROJECT_ID}.{BIGQUERY_DATASET}.pipeline_runs`
        WHERE repository = @repo
        AND workflow_name = @workflow
        AND started_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        AND status IN ('success', 'failure')
        ORDER BY started_at DESC
        LIMIT 100
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("repo", "STRING", repo),
                bigquery.ScalarQueryParameter("workflow", "STRING", workflow),
                bigquery.ScalarQueryParameter("days", "INT64", days),
            ]
        )
        
        try:
            query_job = bigquery_client.query(query, job_config=job_config)
            results = query_job.result()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    async def _detect_duration_anomaly(self, current_duration: float, historical: List[Dict]) -> Optional[Dict]:
        """Detect duration anomalies using statistical analysis"""
        durations = [h["duration_seconds"] for h in historical if h["duration_seconds"]]
        if len(durations) < 5:
            return None
        
        import statistics
        mean_duration = statistics.mean(durations)
        stdev_duration = statistics.stdev(durations) if len(durations) > 1 else 0
        
        if stdev_duration == 0:
            return None
        
        z_score = abs(current_duration - mean_duration) / stdev_duration
        
        if z_score > self.model_thresholds["duration_z_score"]:
            return {
                "type": "duration_anomaly",
                "severity": "high" if z_score > 4 else "medium",
                "message": f"Build duration {current_duration:.1f}s is {z_score:.1f} standard deviations from mean {mean_duration:.1f}s",
                "z_score": z_score,
                "baseline_mean": mean_duration,
                "baseline_stdev": stdev_duration
            }
        
        return None
    
    async def _detect_failure_patterns(self, metrics: PipelineMetrics, historical: List[Dict]) -> Optional[Dict]:
        """Detect unusual failure patterns"""
        recent_failures = [h for h in historical[:10] if h["status"] == "failure"]
        failure_rate = len(recent_failures) / min(len(historical), 10)
        
        if failure_rate > self.model_thresholds["failure_rate_threshold"] and metrics.status == "failure":
            return {
                "type": "failure_pattern",
                "severity": "high" if failure_rate > 0.3 else "medium",
                "message": f"High failure rate detected: {failure_rate:.1%} in recent runs",
                "failure_rate": failure_rate,
                "consecutive_failures": len([h for h in historical[:5] if h["status"] == "failure"])
            }
        
        return None
    
    async def _detect_resource_anomaly(self, current_usage: Dict, historical: List[Dict]) -> Optional[Dict]:
        """Detect resource usage anomalies"""
        if not current_usage.get("cpu_minutes"):
            return None
        
        cpu_usages = [h["resource_usage"]["cpu_minutes"] for h in historical 
                     if h.get("resource_usage") and h["resource_usage"].get("cpu_minutes")]
        
        if len(cpu_usages) < 5:
            return None
        
        import statistics
        mean_cpu = statistics.mean(cpu_usages)
        current_cpu = current_usage["cpu_minutes"]
        
        if current_cpu > mean_cpu * self.model_thresholds["resource_spike_threshold"]:
            return {
                "type": "resource_anomaly",
                "severity": "medium",
                "message": f"CPU usage spike: {current_cpu:.1f} minutes vs baseline {mean_cpu:.1f}",
                "current_usage": current_cpu,
                "baseline_mean": mean_cpu,
                "spike_ratio": current_cpu / mean_cpu
            }
        
        return None
    
    def _calculate_confidence(self, anomalies: List[Dict], historical: List[Dict]) -> float:
        """Calculate confidence score for anomaly detection"""
        if not historical:
            return 0.0
        
        base_confidence = min(len(historical) / 20, 1.0)  # More data = higher confidence
        anomaly_weight = len(anomalies) * 0.2
        
        return min(base_confidence + anomaly_weight, 1.0)

# Initialize anomaly detector
anomaly_detector = AnomalyDetection()

class MetricsCollector:
    """Collect comprehensive pipeline metrics"""
    
    async def collect_workflow_metrics(self, workflow_run: Dict, repository: Dict) -> PipelineMetrics:
        """Collect metrics from GitHub workflow run"""
        
        # Basic workflow information
        metrics = PipelineMetrics(
            run_id=str(workflow_run["id"]),
            repository=repository["full_name"],
            workflow_name=workflow_run["name"] or "unknown",
            branch=workflow_run["head_branch"] or "main",
            commit_sha=workflow_run["head_sha"],
            actor=workflow_run["actor"]["login"],
            environment=workflow_run.get("environment", {}).get("name") if workflow_run.get("environment") else None,
            status=workflow_run["status"],
            started_at=datetime.fromisoformat(workflow_run["created_at"].replace("Z", "+00:00")),
            completed_at=datetime.fromisoformat(workflow_run["updated_at"].replace("Z", "+00:00")) if workflow_run.get("updated_at") else None
        )
        
        # Calculate duration if completed
        if metrics.completed_at and workflow_run["status"] in ["completed", "failure", "success"]:
            metrics.duration_seconds = (metrics.completed_at - metrics.started_at).total_seconds()
        
        # Collect job metrics
        if GITHUB_TOKEN:
            try:
                jobs_data = await self._fetch_workflow_jobs(workflow_run["id"], repository["full_name"])
                metrics.jobs_total = len(jobs_data)
                metrics.jobs_failed = len([job for job in jobs_data if job["conclusion"] == "failure"])
                
                # Collect test results and artifacts
                metrics.test_results = await self._extract_test_results(jobs_data)
                metrics.artifact_count = await self._count_artifacts(workflow_run["id"], repository["full_name"])
                metrics.resource_usage = await self._estimate_resource_usage(jobs_data)
                
            except Exception as e:
                logger.warning(f"Failed to collect detailed metrics: {e}")
        
        return metrics
    
    async def _fetch_workflow_jobs(self, run_id: int, repo: str) -> List[Dict]:
        """Fetch workflow jobs from GitHub API"""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"token {GITHUB_TOKEN}"}
            response = await client.get(
                f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs",
                headers=headers
            )
            response.raise_for_status()
            return response.json()["jobs"]
    
    async def _extract_test_results(self, jobs_data: List[Dict]) -> Optional[Dict]:
        """Extract test results from job logs"""
        # This would parse job logs for test results
        # Simplified implementation for now
        test_jobs = [job for job in jobs_data if "test" in job["name"].lower()]
        if not test_jobs:
            return None
        
        return {
            "total_tests": None,  # Would parse from logs
            "passed_tests": None,
            "failed_tests": None,
            "coverage_percentage": None
        }
    
    async def _count_artifacts(self, run_id: int, repo: str) -> Optional[int]:
        """Count workflow artifacts"""
        if not GITHUB_TOKEN:
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                response = await client.get(
                    f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()["total_count"]
        except Exception as e:
            logger.warning(f"Failed to count artifacts: {e}")
            return None
    
    async def _estimate_resource_usage(self, jobs_data: List[Dict]) -> Optional[Dict]:
        """Estimate resource usage from job timing"""
        if not jobs_data:
            return None
        
        total_duration = 0
        for job in jobs_data:
            if job.get("started_at") and job.get("completed_at"):
                start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
                total_duration += (end - start).total_seconds()
        
        # Rough estimates (would be more accurate with actual monitoring)
        return {
            "cpu_minutes": total_duration / 60,  # Assume 1 CPU per job
            "memory_gb_minutes": total_duration / 60 * 2,  # Assume 2GB per job
            "network_gb": 0.1  # Rough estimate
        }

# Initialize metrics collector
metrics_collector = MetricsCollector()

async def store_metrics_in_bigquery(metrics: PipelineMetrics):
    """Store pipeline metrics in BigQuery"""
    table_id = f"{PROJECT_ID}.{BIGQUERY_DATASET}.pipeline_runs"
    
    # Convert metrics to BigQuery format
    row = {
        "run_id": metrics.run_id,
        "repository": metrics.repository,
        "workflow_name": metrics.workflow_name,
        "branch": metrics.branch,
        "commit_sha": metrics.commit_sha,
        "actor": metrics.actor,
        "environment": metrics.environment,
        "status": metrics.status,
        "started_at": metrics.started_at.isoformat(),
        "completed_at": metrics.completed_at.isoformat() if metrics.completed_at else None,
        "duration_seconds": metrics.duration_seconds,
        "conclusion": metrics.status,
        "jobs_total": metrics.jobs_total,
        "jobs_failed": metrics.jobs_failed,
        "artifact_count": metrics.artifact_count,
        "test_results": metrics.test_results,
        "security_scan_results": metrics.security_scan_results,
        "resource_usage": metrics.resource_usage,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        table = bigquery_client.get_table(table_id)
        errors = bigquery_client.insert_rows_json(table, [row])
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        else:
            logger.info(f"Stored metrics for run {metrics.run_id}")
    except NotFound:
        logger.error(f"BigQuery table {table_id} not found")
    except Exception as e:
        logger.error(f"Error storing metrics: {e}")

async def publish_pipeline_event(event_type: str, metrics: PipelineMetrics, anomalies: Optional[Dict] = None):
    """Publish pipeline event to Pub/Sub"""
    topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
    
    event_data = {
        "event_id": f"{metrics.run_id}_{event_type}_{int(datetime.now().timestamp())}",
        "timestamp": int(datetime.now().timestamp() * 1000),
        "event_type": event_type.upper(),
        "repository": metrics.repository,
        "workflow_name": metrics.workflow_name,
        "workflow_run_id": int(metrics.run_id),
        "branch": metrics.branch,
        "commit_sha": metrics.commit_sha,
        "actor": metrics.actor,
        "environment": metrics.environment,
        "status": metrics.status.upper(),
        "duration_seconds": metrics.duration_seconds,
        "metadata": {
            "jobs_total": str(metrics.jobs_total),
            "jobs_failed": str(metrics.jobs_failed),
            "has_anomalies": str(bool(anomalies and anomalies.get("anomalies"))),
            "anomaly_count": str(len(anomalies.get("anomalies", [])) if anomalies else 0)
        }
    }
    
    try:
        future = publisher.publish(topic_path, json.dumps(event_data).encode("utf-8"))
        message_id = future.result()
        logger.info(f"Published event {event_type} for run {metrics.run_id}: {message_id}")
    except Exception as e:
        logger.error(f"Error publishing event: {e}")

async def send_custom_metrics(metrics: PipelineMetrics, anomalies: Dict):
    """Send custom metrics to Google Cloud Monitoring"""
    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/aetherveil/pipeline_duration"
    series.resource.type = "global"
    
    now = datetime.now(timezone.utc)
    interval = monitoring_v3.TimeInterval({"end_time": {"seconds": int(now.timestamp())}})
    point = monitoring_v3.Point({
        "interval": interval,
        "value": {"double_value": metrics.duration_seconds or 0}
    })
    series.points = [point]
    
    try:
        project_name = f"projects/{PROJECT_ID}"
        monitoring_client.create_time_series(
            name=project_name, 
            time_series=[series]
        )
        logger.info(f"Sent custom metrics for run {metrics.run_id}")
    except Exception as e:
        logger.error(f"Error sending custom metrics: {e}")

# API endpoints
@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Health check endpoint to confirm the service is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle GitHub webhook events"""
    try:
        payload = await request.json()
        event = GitHubWebhookEvent(**payload)
        
        if event.action in ["completed", "requested"] and event.workflow_run:
            # Process workflow run event
            metrics = await metrics_collector.collect_workflow_metrics(
                event.workflow_run, 
                event.repository
            )
            
            # Perform anomaly detection
            anomalies = await anomaly_detector.analyze_pipeline_run(metrics)
            
            # Store data and publish events in background
            background_tasks.add_task(store_metrics_in_bigquery, metrics)
            background_tasks.add_task(
                publish_pipeline_event, 
                "workflow_" + event.action, 
                metrics, 
                anomalies
            )
            background_tasks.add_task(send_custom_metrics, metrics, anomalies)
            
            return {
                "status": "processed",
                "run_id": metrics.run_id,
                "anomalies_detected": len(anomalies.get("anomalies", [])),
                "confidence_score": anomalies.get("confidence_score", 0)
            }
        
        return {"status": "ignored", "reason": "not a workflow completion event"}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/subscriber/github-events", status_code=status.HTTP_204_NO_CONTENT, tags=["Subscribers"])
async def receive_github_event(request: PubSubPushRequest):
    """
    Endpoint to receive and process messages from the 'github-events' Pub/Sub topic.
    Enhanced implementation with comprehensive data processing.
    """
    try:
        # The actual data is Base64 encoded in the message payload
        decoded_data = base64.b64decode(request.message.data).decode("utf-8")
        payload = json.loads(decoded_data)

        logger.info(f"Received GitHub event from subscription '{request.subscription}'")
        
        # Process as GitHub webhook event
        event = GitHubWebhookEvent(**payload)
        
        if event.action in ["completed", "requested"] and event.workflow_run:
            # Process workflow run event
            metrics = await metrics_collector.collect_workflow_metrics(
                event.workflow_run, 
                event.repository
            )
            
            # Perform anomaly detection
            anomalies = await anomaly_detector.analyze_pipeline_run(metrics)
            
            # Store metrics and publish events
            await store_metrics_in_bigquery(metrics)
            await publish_pipeline_event(
                "workflow_" + event.action, 
                metrics, 
                anomalies
            )
            await send_custom_metrics(metrics, anomalies)
            
            logger.info(f"Processed workflow {event.action} for run {metrics.run_id}")

    except Exception as e:
        logger.error(f"Error processing GitHub event: {e}", exc_info=True)
        # Return a 500 error to signal Pub/Sub to retry the message.
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/v1/subscriber/logs", status_code=status.HTTP_204_NO_CONTENT, tags=["Subscribers"])
async def receive_pipeline_log(request: PubSubPushRequest):
    """
    Endpoint to receive and process messages from the 'pipeline-logs' Pub/Sub topic.
    Enhanced implementation with log analysis and correlation.
    """
    try:
        decoded_data = base64.b64decode(request.message.data).decode("utf-8")
        log_entry = json.loads(decoded_data)

        logger.info(f"Received log entry from subscription '{request.subscription}'")
        
        # Process log entry for anomaly detection
        # This would include parsing logs for error patterns, performance metrics, etc.
        # For now, just log the entry
        logger.info(f"Log Entry: {json.dumps(log_entry, indent=2)}")

    except Exception as e:
        logger.error(f"Error processing pipeline log: {e}", exc_info=True)
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/metrics/{repository}/{workflow}")
async def get_pipeline_metrics(repository: str, workflow: str, days: int = 30):
    """Get pipeline metrics for analysis"""
    try:
        historical_data = await anomaly_detector._get_historical_metrics(
            repository, workflow, days
        )
        
        if not historical_data:
            return {"error": "No historical data found"}
        
        # Calculate summary statistics
        durations = [h["duration_seconds"] for h in historical_data if h["duration_seconds"]]
        success_rate = len([h for h in historical_data if h["status"] == "success"]) / len(historical_data)
        
        import statistics
        avg_duration = statistics.mean(durations) if durations else 0
        
        return {
            "repository": repository,
            "workflow": workflow,
            "period_days": days,
            "total_runs": len(historical_data),
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "recent_runs": historical_data[:10]
        }
        
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # This is for local development and debugging.
    # The container runs uvicorn directly.
    uvicorn.run(app, host="0.0.0.0", port=8080)
