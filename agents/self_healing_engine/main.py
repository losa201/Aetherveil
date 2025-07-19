"""
Self-Healing Workflow Engine
Automated failure recovery and pipeline optimization
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from google.cloud import pubsub_v1, bigquery, functions_v1, firestore
from google.cloud import workflows_v1
import httpx
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.getenv("PROJECT_ID", "tidy-computing-465909-i3")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REGION = os.getenv("REGION", "us-central1")

# Initialize clients
app = FastAPI(
    title="Self-Healing Workflow Engine",
    description="Automated failure recovery and pipeline optimization",
    version="1.0.0"
)
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
bigquery_client = bigquery.Client(project=PROJECT_ID)
workflows_client = workflows_v1.WorkflowsClient()
firestore_client = firestore.Client(project=PROJECT_ID)

class FailureType(str, Enum):
    TRANSIENT_NETWORK = "transient_network"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"
    TEST_FAILURE = "test_failure"
    BUILD_FAILURE = "build_failure"
    TIMEOUT = "timeout"
    AUTHENTICATION_ERROR = "authentication_error"

class HealingAction(str, Enum):
    RETRY = "retry"
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    CLEAR_CACHE = "clear_cache"
    UPDATE_CONFIG = "update_config"
    ROLLBACK = "rollback"
    NOTIFY_TEAM = "notify_team"
    CREATE_ISSUE = "create_issue"

class FailureEvent(BaseModel):
    event_id: str
    timestamp: datetime
    failure_type: FailureType
    source: str  # workflow, job, step
    repository: str
    workflow_name: str
    run_id: str
    error_message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealingPlan(BaseModel):
    failure_event: FailureEvent
    actions: List[HealingAction]
    priority: int  # 1-5, 5 being highest
    estimated_duration: int  # seconds
    success_probability: float  # 0-1
    description: str

class SelfHealingEngine:
    """Core self-healing logic and decision making"""
    
    def __init__(self):
        self.healing_rules = self._initialize_healing_rules()
        self.action_history: List[Dict] = []
    
    def _initialize_healing_rules(self) -> Dict[FailureType, List[HealingAction]]:
        """Initialize healing rules based on failure types"""
        return {
            FailureType.TRANSIENT_NETWORK: [
                HealingAction.RETRY,
                HealingAction.NOTIFY_TEAM
            ],
            FailureType.DEPENDENCY_FAILURE: [
                HealingAction.RETRY,
                HealingAction.RESTART_SERVICE,
                HealingAction.NOTIFY_TEAM
            ],
            FailureType.RESOURCE_EXHAUSTION: [
                HealingAction.SCALE_UP,
                HealingAction.CLEAR_CACHE,
                HealingAction.NOTIFY_TEAM
            ],
            FailureType.CONFIGURATION_ERROR: [
                HealingAction.UPDATE_CONFIG,
                HealingAction.ROLLBACK,
                HealingAction.CREATE_ISSUE
            ],
            FailureType.TEST_FAILURE: [
                HealingAction.RETRY,
                HealingAction.CREATE_ISSUE,
                HealingAction.NOTIFY_TEAM
            ],
            FailureType.BUILD_FAILURE: [
                HealingAction.CLEAR_CACHE,
                HealingAction.RETRY,
                HealingAction.CREATE_ISSUE
            ],
            FailureType.TIMEOUT: [
                HealingAction.SCALE_UP,
                HealingAction.RETRY,
                HealingAction.UPDATE_CONFIG
            ],
            FailureType.AUTHENTICATION_ERROR: [
                HealingAction.UPDATE_CONFIG,
                HealingAction.NOTIFY_TEAM
            ]
        }
    
    async def analyze_failure(self, failure_event: FailureEvent) -> HealingPlan:
        """Analyze failure and create healing plan"""
        
        # Classify failure type if not already classified
        if not failure_event.failure_type:
            failure_event.failure_type = await self._classify_failure(failure_event)
        
        # Get historical context
        historical_context = await self._get_historical_context(failure_event)
        
        # Generate healing actions
        actions = self._generate_healing_actions(failure_event, historical_context)
        
        # Calculate priority and success probability
        priority = self._calculate_priority(failure_event, historical_context)
        success_probability = self._estimate_success_probability(failure_event, actions, historical_context)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(actions)
        
        return HealingPlan(
            failure_event=failure_event,
            actions=actions,
            priority=priority,
            estimated_duration=estimated_duration,
            success_probability=success_probability,
            description=self._generate_description(failure_event, actions)
        )
    
    async def _classify_failure(self, failure_event: FailureEvent) -> FailureType:
        """Classify failure type based on error message and context"""
        error_msg = failure_event.error_message.lower()
        
        # Network-related failures
        if any(keyword in error_msg for keyword in ["network", "connection", "timeout", "unreachable"]):
            return FailureType.TRANSIENT_NETWORK
        
        # Resource exhaustion
        if any(keyword in error_msg for keyword in ["memory", "disk", "quota", "limit", "resource"]):
            return FailureType.RESOURCE_EXHAUSTION
        
        # Authentication/authorization
        if any(keyword in error_msg for keyword in ["auth", "permission", "denied", "unauthorized", "401", "403"]):
            return FailureType.AUTHENTICATION_ERROR
        
        # Build failures
        if any(keyword in error_msg for keyword in ["build", "compile", "syntax", "npm", "pip", "maven"]):
            return FailureType.BUILD_FAILURE
        
        # Test failures
        if any(keyword in error_msg for keyword in ["test", "assertion", "expect", "junit", "pytest"]):
            return FailureType.TEST_FAILURE
        
        # Configuration errors
        if any(keyword in error_msg for keyword in ["config", "environment", "variable", "missing"]):
            return FailureType.CONFIGURATION_ERROR
        
        # Default to dependency failure
        return FailureType.DEPENDENCY_FAILURE
    
    async def _get_historical_context(self, failure_event: FailureEvent) -> Dict:
        """Get historical context for similar failures from Firestore"""
        
        try:
            # Query Firestore for historical healing events
            healing_collection = firestore_client.collection('healing_history')
            
            # Build query for similar failures
            query = healing_collection.where('repository', '==', failure_event.repository) \
                                   .where('workflow_name', '==', failure_event.workflow_name) \
                                   .where('failure_type', '==', failure_event.failure_type.value) \
                                   .order_by('timestamp', direction=firestore.Query.DESCENDING) \
                                   .limit(20)
            
            docs = query.stream()
            historical_actions = []
            
            for doc in docs:
                data = doc.to_dict()
                # Filter recent events (last 30 days)
                if data.get('timestamp') and (datetime.now(timezone.utc) - data['timestamp']).days <= 30:
                    historical_actions.append(data)
            
            # Aggregate results by success rate
            aggregated_results = {}
            for action in historical_actions:
                key = f"{action.get('failure_type', '')}_{json.dumps(action.get('healing_actions', []))}"
                if key not in aggregated_results:
                    aggregated_results[key] = {
                        'failure_type': action.get('failure_type'),
                        'healing_actions': action.get('healing_actions', []),
                        'success_count': 0,
                        'total_count': 0,
                        'avg_duration': 0
                    }
                
                aggregated_results[key]['total_count'] += 1
                if action.get('success', False):
                    aggregated_results[key]['success_count'] += 1
                
                if action.get('duration_seconds'):
                    aggregated_results[key]['avg_duration'] = (
                        aggregated_results[key]['avg_duration'] + action['duration_seconds']
                    ) / aggregated_results[key]['total_count']
            
            # Convert to list and sort by success rate
            results = []
            for key, data in aggregated_results.items():
                data['success_rate'] = data['success_count'] / data['total_count'] if data['total_count'] > 0 else 0
                results.append(data)
            
            results.sort(key=lambda x: x['success_rate'], reverse=True)
            
            return {"historical_actions": results[:10]}
            
        except Exception as e:
            logger.error(f"Error fetching historical context from Firestore: {e}")
            return {"historical_actions": []}
    
    def _generate_healing_actions(self, failure_event: FailureEvent, context: Dict) -> List[HealingAction]:
        """Generate healing actions based on failure type and context"""
        
        # Start with rule-based actions
        base_actions = self.healing_rules.get(failure_event.failure_type, [HealingAction.NOTIFY_TEAM])
        
        # Adjust based on historical success
        historical_actions = context.get("historical_actions", [])
        if historical_actions:
            # Prioritize actions that have been successful historically
            successful_actions = [
                action for entry in historical_actions 
                if entry.get("success", False)
                for action in entry.get("healing_actions", [])
            ]
            
            # Combine and deduplicate
            all_actions = list(set(base_actions + successful_actions))
            return all_actions[:3]  # Limit to top 3 actions
        
        return base_actions[:3]
    
    def _calculate_priority(self, failure_event: FailureEvent, context: Dict) -> int:
        """Calculate healing priority (1-5)"""
        priority = 3  # Default medium priority
        
        # Increase priority for production failures
        if "prod" in failure_event.repository.lower() or "main" in failure_event.metadata.get("branch", ""):
            priority += 1
        
        # Increase priority for repeated failures
        historical_count = len(context.get("historical_actions", []))
        if historical_count > 5:
            priority += 1
        
        # Increase priority for certain failure types
        high_priority_types = [FailureType.RESOURCE_EXHAUSTION, FailureType.AUTHENTICATION_ERROR]
        if failure_event.failure_type in high_priority_types:
            priority += 1
        
        return min(priority, 5)
    
    def _estimate_success_probability(self, failure_event: FailureEvent, actions: List[HealingAction], context: Dict) -> float:
        """Estimate probability of successful healing"""
        
        # Base probability based on failure type
        base_probabilities = {
            FailureType.TRANSIENT_NETWORK: 0.8,
            FailureType.DEPENDENCY_FAILURE: 0.6,
            FailureType.RESOURCE_EXHAUSTION: 0.7,
            FailureType.CONFIGURATION_ERROR: 0.5,
            FailureType.TEST_FAILURE: 0.4,
            FailureType.BUILD_FAILURE: 0.5,
            FailureType.TIMEOUT: 0.6,
            FailureType.AUTHENTICATION_ERROR: 0.3
        }
        
        base_prob = base_probabilities.get(failure_event.failure_type, 0.5)
        
        # Adjust based on historical success
        historical_actions = context.get("historical_actions", [])
        if historical_actions:
            success_rate = sum(1 for entry in historical_actions if entry.get("success", False)) / len(historical_actions)
            base_prob = (base_prob + success_rate) / 2
        
        return min(base_prob, 0.95)
    
    def _estimate_duration(self, actions: List[HealingAction]) -> int:
        """Estimate duration for healing actions in seconds"""
        
        action_durations = {
            HealingAction.RETRY: 300,  # 5 minutes
            HealingAction.RESTART_SERVICE: 180,  # 3 minutes
            HealingAction.SCALE_UP: 600,  # 10 minutes
            HealingAction.CLEAR_CACHE: 120,  # 2 minutes
            HealingAction.UPDATE_CONFIG: 300,  # 5 minutes
            HealingAction.ROLLBACK: 900,  # 15 minutes
            HealingAction.NOTIFY_TEAM: 60,  # 1 minute
            HealingAction.CREATE_ISSUE: 120  # 2 minutes
        }
        
        total_duration = sum(action_durations.get(action, 300) for action in actions)
        return total_duration
    
    def _generate_description(self, failure_event: FailureEvent, actions: List[HealingAction]) -> str:
        """Generate human-readable description of healing plan"""
        
        action_descriptions = {
            HealingAction.RETRY: "retry the failed operation",
            HealingAction.RESTART_SERVICE: "restart the affected service",
            HealingAction.SCALE_UP: "scale up resources",
            HealingAction.CLEAR_CACHE: "clear cache and temporary files",
            HealingAction.UPDATE_CONFIG: "update configuration",
            HealingAction.ROLLBACK: "rollback to previous version",
            HealingAction.NOTIFY_TEAM: "notify the development team",
            HealingAction.CREATE_ISSUE: "create a tracking issue"
        }
        
        action_list = [action_descriptions.get(action, str(action)) for action in actions]
        
        return f"To address {failure_event.failure_type.value} in {failure_event.workflow_name}, will: {', '.join(action_list)}"

class HealingExecutor:
    """Execute healing actions"""
    
    def __init__(self):
        self.github_client = httpx.AsyncClient(
            headers={"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
        )
    
    async def execute_healing_plan(self, plan: HealingPlan) -> Dict[str, Any]:
        """Execute a healing plan and return results"""
        
        results = {
            "plan_id": plan.failure_event.event_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "actions_executed": [],
            "success": False,
            "error": None
        }
        
        try:
            for action in plan.actions:
                action_result = await self._execute_action(action, plan.failure_event)
                results["actions_executed"].append({
                    "action": action.value,
                    "success": action_result.get("success", False),
                    "details": action_result.get("details", ""),
                    "duration": action_result.get("duration", 0)
                })
                
                # If action failed and it's critical, stop execution
                if not action_result.get("success", False) and action in [HealingAction.ROLLBACK]:
                    break
            
            # Determine overall success
            results["success"] = any(result["success"] for result in results["actions_executed"])
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error executing healing plan: {e}")
        
        results["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        # Store results for learning
        await self._store_healing_result(plan, results)
        
        return results
    
    async def _execute_action(self, action: HealingAction, failure_event: FailureEvent) -> Dict[str, Any]:
        """Execute a specific healing action"""
        
        start_time = datetime.now()
        
        try:
            if action == HealingAction.RETRY:
                return await self._retry_workflow(failure_event)
            elif action == HealingAction.RESTART_SERVICE:
                return await self._restart_service(failure_event)
            elif action == HealingAction.SCALE_UP:
                return await self._scale_up_resources(failure_event)
            elif action == HealingAction.CLEAR_CACHE:
                return await self._clear_cache(failure_event)
            elif action == HealingAction.UPDATE_CONFIG:
                return await self._update_config(failure_event)
            elif action == HealingAction.ROLLBACK:
                return await self._rollback_deployment(failure_event)
            elif action == HealingAction.NOTIFY_TEAM:
                return await self._notify_team(failure_event)
            elif action == HealingAction.CREATE_ISSUE:
                return await self._create_issue(failure_event)
            else:
                return {"success": False, "details": f"Unknown action: {action}"}
                
        except Exception as e:
            return {
                "success": False,
                "details": f"Action failed: {str(e)}",
                "duration": (datetime.now() - start_time).total_seconds()
            }
    
    async def _retry_workflow(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Retry a failed GitHub Actions workflow"""
        
        if not GITHUB_TOKEN:
            return {"success": False, "details": "No GitHub token available"}
        
        try:
            url = f"https://api.github.com/repos/{failure_event.repository}/actions/runs/{failure_event.run_id}/rerun"
            response = await self.github_client.post(url)
            
            if response.status_code == 201:
                return {"success": True, "details": "Workflow retry initiated"}
            else:
                return {"success": False, "details": f"Retry failed: {response.text}"}
                
        except Exception as e:
            return {"success": False, "details": f"Retry error: {str(e)}"}
    
    async def _restart_service(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Restart a Cloud Run service"""
        # This would integrate with GCP Cloud Run API
        # For now, return a placeholder
        return {"success": True, "details": "Service restart simulated"}
    
    async def _scale_up_resources(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Scale up cloud resources"""
        # This would integrate with GCP autoscaling
        return {"success": True, "details": "Resource scaling simulated"}
    
    async def _clear_cache(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Clear build cache"""
        # This would clear GitHub Actions cache
        return {"success": True, "details": "Cache clearing simulated"}
    
    async def _update_config(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Update configuration"""
        # This would update configuration files
        return {"success": True, "details": "Configuration update simulated"}
    
    async def _rollback_deployment(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Rollback to previous deployment"""
        # This would trigger a rollback
        return {"success": True, "details": "Rollback simulated"}
    
    async def _notify_team(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Notify the team about the failure"""
        
        # Publish notification to Pub/Sub
        topic_path = publisher.topic_path(PROJECT_ID, "aetherveil-team-notifications")
        
        notification = {
            "type": "failure_notification",
            "failure_event": failure_event.dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "urgency": "high" if failure_event.failure_type in [FailureType.RESOURCE_EXHAUSTION] else "medium"
        }
        
        try:
            future = publisher.publish(topic_path, json.dumps(notification).encode("utf-8"))
            future.result()
            return {"success": True, "details": "Team notification sent"}
        except Exception as e:
            return {"success": False, "details": f"Notification failed: {str(e)}"}
    
    async def _create_issue(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Create a GitHub issue for the failure"""
        
        if not GITHUB_TOKEN:
            return {"success": False, "details": "No GitHub token available"}
        
        issue_data = {
            "title": f"Automated Issue: {failure_event.failure_type.value} in {failure_event.workflow_name}",
            "body": f"""
**Failure Details:**
- Type: {failure_event.failure_type.value}
- Workflow: {failure_event.workflow_name}
- Run ID: {failure_event.run_id}
- Time: {failure_event.timestamp}

**Error Message:**
```
{failure_event.error_message}
```

**Metadata:**
{json.dumps(failure_event.metadata, indent=2)}

This issue was automatically created by the Self-Healing Engine.
            """.strip(),
            "labels": ["bug", "automated", "pipeline-failure"]
        }
        
        try:
            url = f"https://api.github.com/repos/{failure_event.repository}/issues"
            response = await self.github_client.post(url, json=issue_data)
            
            if response.status_code == 201:
                issue_url = response.json().get("html_url", "")
                return {"success": True, "details": f"Issue created: {issue_url}"}
            else:
                return {"success": False, "details": f"Issue creation failed: {response.text}"}
                
        except Exception as e:
            return {"success": False, "details": f"Issue creation error: {str(e)}"}
    
    async def _store_healing_result(self, plan: HealingPlan, results: Dict[str, Any]):
        """Store healing results in Firestore for learning"""
        
        healing_doc = {
            "event_id": plan.failure_event.event_id,
            "timestamp": datetime.now(timezone.utc),
            "repository": plan.failure_event.repository,
            "workflow_name": plan.failure_event.workflow_name,
            "failure_type": plan.failure_event.failure_type.value,
            "healing_actions": [action.value for action in plan.actions],
            "success": results["success"],
            "duration_seconds": (
                datetime.fromisoformat(results["completed_at"]) - 
                datetime.fromisoformat(results["started_at"])
            ).total_seconds(),
            "actions_executed": results["actions_executed"],
            "error_message": results.get("error"),
            "priority": plan.priority,
            "success_probability": plan.success_probability,
            "estimated_duration": plan.estimated_duration
        }
        
        try:
            # Store in Firestore
            doc_ref = firestore_client.collection('healing_history').document(plan.failure_event.event_id)
            doc_ref.set(healing_doc)
            logger.info(f"Stored healing result in Firestore for {plan.failure_event.event_id}")
            
            # Also store summary in BigQuery for analytics
            bq_row = {
                "event_id": plan.failure_event.event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "repository": plan.failure_event.repository,
                "workflow_name": plan.failure_event.workflow_name,
                "failure_type": plan.failure_event.failure_type.value,
                "healing_actions": [action.value for action in plan.actions],
                "success": results["success"],
                "duration_seconds": healing_doc["duration_seconds"],
                "error_message": results.get("error")
            }
            
            table_id = f"{PROJECT_ID}.pipeline_analytics.healing_history"
            table = bigquery_client.get_table(table_id)
            bq_errors = bigquery_client.insert_rows_json(table, [bq_row])
            if not bq_errors:
                logger.info(f"Stored healing summary in BigQuery for {plan.failure_event.event_id}")
                
        except Exception as e:
            logger.error(f"Error storing healing result: {e}")

# Initialize components
healing_engine = SelfHealingEngine()
healing_executor = HealingExecutor()

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/analyze/failure")
async def analyze_failure(failure_event: FailureEvent):
    """Analyze a failure event and create healing plan"""
    try:
        plan = await healing_engine.analyze_failure(failure_event)
        return plan
    except Exception as e:
        logger.error(f"Error analyzing failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute/healing")
async def execute_healing(plan: HealingPlan, background_tasks: BackgroundTasks):
    """Execute a healing plan"""
    try:
        background_tasks.add_task(healing_executor.execute_healing_plan, plan)
        return {"status": "execution_started", "plan_id": plan.failure_event.event_id}
    except Exception as e:
        logger.error(f"Error starting healing execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/heal")
async def heal_failure(failure_event: FailureEvent, background_tasks: BackgroundTasks):
    """Complete healing workflow: analyze and execute"""
    try:
        # Analyze failure
        plan = await healing_engine.analyze_failure(failure_event)
        
        # Execute healing in background
        background_tasks.add_task(healing_executor.execute_healing_plan, plan)
        
        return {
            "status": "healing_started",
            "plan": plan,
            "estimated_duration": plan.estimated_duration,
            "success_probability": plan.success_probability
        }
        
    except Exception as e:
        logger.error(f"Error in healing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_healing_statistics():
    """Get healing statistics"""
    
    query = f"""
    SELECT 
        failure_type,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_healings,
        AVG(duration_seconds) as avg_duration,
        MAX(timestamp) as last_healing
    FROM `{PROJECT_ID}.pipeline_analytics.healing_history`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
    GROUP BY failure_type
    ORDER BY total_attempts DESC
    """
    
    try:
        query_job = bigquery_client.query(query)
        results = [dict(row) for row in query_job.result()]
        
        return {
            "period": "last_30_days",
            "statistics": results,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)