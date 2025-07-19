"""
GitHub Webhook Processor Cloud Function
Processes GitHub webhook events and publishes to Pub/Sub
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

import functions_framework
from google.cloud import pubsub_v1
from flask import Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.getenv("PROJECT_ID")
PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "aetherveil-pipeline-events")

# Initialize Pub/Sub client
publisher = pubsub_v1.PublisherClient()

@functions_framework.http
def process_webhook(request: Request):
    """
    Process GitHub webhook events and publish to Pub/Sub
    """
    
    # Validate request method
    if request.method != 'POST':
        return {'error': 'Only POST requests are allowed'}, 405
    
    try:
        # Get request data
        request_json = request.get_json(silent=True)
        headers = dict(request.headers)
        
        if not request_json:
            return {'error': 'No JSON payload found'}, 400
        
        # Extract GitHub event type
        event_type = headers.get('X-GitHub-Event', 'unknown')
        
        # Process different event types
        if event_type == 'workflow_run':
            return process_workflow_run_event(request_json, headers)
        elif event_type == 'workflow_job':
            return process_workflow_job_event(request_json, headers)
        elif event_type == 'push':
            return process_push_event(request_json, headers)
        elif event_type == 'pull_request':
            return process_pull_request_event(request_json, headers)
        else:
            logger.info(f"Ignoring event type: {event_type}")
            return {'status': 'ignored', 'event_type': event_type}, 200
    
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return {'error': str(e)}, 500

def process_workflow_run_event(payload: Dict[str, Any], headers: Dict[str, str]) -> tuple:
    """Process workflow run events"""
    
    action = payload.get('action')
    workflow_run = payload.get('workflow_run', {})
    repository = payload.get('repository', {})
    
    # Only process completed workflows
    if action not in ['completed', 'requested']:
        return {'status': 'ignored', 'reason': f'action {action} not processed'}, 200
    
    # Prepare event data
    event_data = {
        'event_type': 'WORKFLOW_RUN',
        'action': action,
        'timestamp': datetime.utcnow().isoformat(),
        'repository': repository.get('full_name', ''),
        'workflow_name': workflow_run.get('name', ''),
        'workflow_id': workflow_run.get('id', 0),
        'run_number': workflow_run.get('run_number', 0),
        'status': workflow_run.get('status', ''),
        'conclusion': workflow_run.get('conclusion', ''),
        'branch': workflow_run.get('head_branch', ''),
        'commit_sha': workflow_run.get('head_sha', ''),
        'actor': workflow_run.get('actor', {}).get('login', ''),
        'created_at': workflow_run.get('created_at', ''),
        'updated_at': workflow_run.get('updated_at', ''),
        'html_url': workflow_run.get('html_url', ''),
        'jobs_url': workflow_run.get('jobs_url', ''),
        'logs_url': workflow_run.get('logs_url', ''),
        'artifacts_url': workflow_run.get('artifacts_url', ''),
        'environment': workflow_run.get('environment', {}).get('name', '') if workflow_run.get('environment') else '',
        'pull_requests': [
            {
                'id': pr.get('id'),
                'number': pr.get('number'),
                'head': pr.get('head', {}).get('sha'),
                'base': pr.get('base', {}).get('sha')
            }
            for pr in workflow_run.get('pull_requests', [])
        ]
    }
    
    # Publish to Pub/Sub
    success = publish_to_pubsub(event_data, PUBSUB_TOPIC)
    
    if success:
        return {'status': 'processed', 'event_type': 'workflow_run'}, 200
    else:
        return {'error': 'Failed to publish event'}, 500

def process_workflow_job_event(payload: Dict[str, Any], headers: Dict[str, str]) -> tuple:
    """Process workflow job events"""
    
    action = payload.get('action')
    workflow_job = payload.get('workflow_job', {})
    repository = payload.get('repository', {})
    
    # Process job state changes
    if action not in ['completed', 'started', 'queued']:
        return {'status': 'ignored', 'reason': f'action {action} not processed'}, 200
    
    # Prepare event data
    event_data = {
        'event_type': 'WORKFLOW_JOB',
        'action': action,
        'timestamp': datetime.utcnow().isoformat(),
        'repository': repository.get('full_name', ''),
        'workflow_name': workflow_job.get('workflow_name', ''),
        'job_id': workflow_job.get('id', 0),
        'job_name': workflow_job.get('name', ''),
        'run_id': workflow_job.get('run_id', 0),
        'status': workflow_job.get('status', ''),
        'conclusion': workflow_job.get('conclusion', ''),
        'started_at': workflow_job.get('started_at', ''),
        'completed_at': workflow_job.get('completed_at', ''),
        'runner_name': workflow_job.get('runner_name', ''),
        'runner_group_name': workflow_job.get('runner_group_name', ''),
        'labels': workflow_job.get('labels', []),
        'steps': [
            {
                'name': step.get('name', ''),
                'status': step.get('status', ''),
                'conclusion': step.get('conclusion', ''),
                'number': step.get('number', 0),
                'started_at': step.get('started_at', ''),
                'completed_at': step.get('completed_at', '')
            }
            for step in workflow_job.get('steps', [])
        ]
    }
    
    # Publish to Pub/Sub
    success = publish_to_pubsub(event_data, PUBSUB_TOPIC)
    
    if success:
        return {'status': 'processed', 'event_type': 'workflow_job'}, 200
    else:
        return {'error': 'Failed to publish event'}, 500

def process_push_event(payload: Dict[str, Any], headers: Dict[str, str]) -> tuple:
    """Process push events"""
    
    repository = payload.get('repository', {})
    pusher = payload.get('pusher', {})
    
    # Prepare event data
    event_data = {
        'event_type': 'PUSH',
        'timestamp': datetime.utcnow().isoformat(),
        'repository': repository.get('full_name', ''),
        'ref': payload.get('ref', ''),
        'before': payload.get('before', ''),
        'after': payload.get('after', ''),
        'created': payload.get('created', False),
        'deleted': payload.get('deleted', False),
        'forced': payload.get('forced', False),
        'compare': payload.get('compare', ''),
        'commits': [
            {
                'id': commit.get('id', ''),
                'message': commit.get('message', ''),
                'author': commit.get('author', {}).get('name', ''),
                'timestamp': commit.get('timestamp', ''),
                'url': commit.get('url', ''),
                'added': commit.get('added', []),
                'removed': commit.get('removed', []),
                'modified': commit.get('modified', [])
            }
            for commit in payload.get('commits', [])
        ],
        'pusher': {
            'name': pusher.get('name', ''),
            'email': pusher.get('email', '')
        },
        'head_commit': payload.get('head_commit', {})
    }
    
    # Publish to Pub/Sub
    success = publish_to_pubsub(event_data, 'aetherveil-code-events')
    
    if success:
        return {'status': 'processed', 'event_type': 'push'}, 200
    else:
        return {'error': 'Failed to publish event'}, 500

def process_pull_request_event(payload: Dict[str, Any], headers: Dict[str, str]) -> tuple:
    """Process pull request events"""
    
    action = payload.get('action')
    pull_request = payload.get('pull_request', {})
    repository = payload.get('repository', {})
    
    # Process relevant PR actions
    if action not in ['opened', 'closed', 'synchronize', 'reopened']:
        return {'status': 'ignored', 'reason': f'action {action} not processed'}, 200
    
    # Prepare event data
    event_data = {
        'event_type': 'PULL_REQUEST',
        'action': action,
        'timestamp': datetime.utcnow().isoformat(),
        'repository': repository.get('full_name', ''),
        'pr_number': pull_request.get('number', 0),
        'pr_id': pull_request.get('id', 0),
        'title': pull_request.get('title', ''),
        'state': pull_request.get('state', ''),
        'merged': pull_request.get('merged', False),
        'draft': pull_request.get('draft', False),
        'base_branch': pull_request.get('base', {}).get('ref', ''),
        'head_branch': pull_request.get('head', {}).get('ref', ''),
        'base_sha': pull_request.get('base', {}).get('sha', ''),
        'head_sha': pull_request.get('head', {}).get('sha', ''),
        'author': pull_request.get('user', {}).get('login', ''),
        'created_at': pull_request.get('created_at', ''),
        'updated_at': pull_request.get('updated_at', ''),
        'closed_at': pull_request.get('closed_at', ''),
        'merged_at': pull_request.get('merged_at', ''),
        'html_url': pull_request.get('html_url', ''),
        'labels': [label.get('name', '') for label in pull_request.get('labels', [])],
        'requested_reviewers': [
            reviewer.get('login', '') for reviewer in pull_request.get('requested_reviewers', [])
        ],
        'assignees': [
            assignee.get('login', '') for assignee in pull_request.get('assignees', [])
        ]
    }
    
    # Publish to Pub/Sub
    success = publish_to_pubsub(event_data, 'aetherveil-code-events')
    
    if success:
        return {'status': 'processed', 'event_type': 'pull_request'}, 200
    else:
        return {'error': 'Failed to publish event'}, 500

def publish_to_pubsub(event_data: Dict[str, Any], topic_name: str) -> bool:
    """Publish event data to Pub/Sub topic"""
    
    try:
        topic_path = publisher.topic_path(PROJECT_ID, topic_name)
        
        # Convert to JSON and encode
        message_data = json.dumps(event_data, default=str).encode('utf-8')
        
        # Add message attributes
        attributes = {
            'event_type': event_data.get('event_type', ''),
            'repository': event_data.get('repository', ''),
            'timestamp': event_data.get('timestamp', '')
        }
        
        # Publish message
        future = publisher.publish(topic_path, message_data, **attributes)
        message_id = future.result()
        
        logger.info(f"Published message {message_id} to {topic_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error publishing to Pub/Sub: {e}")
        return False