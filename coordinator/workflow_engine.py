"""
Workflow Engine for Aetherveil Sentinel
Orchestrates complex cybersecurity workflows and task chains
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from config.config import config
from coordinator.models import Workflow, Task, WorkflowType, TaskStatus, AgentType
from coordinator.swarm_manager import SwarmManager

logger = logging.getLogger(__name__)

class WorkflowStatus(str, Enum):
    """Workflow status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class WorkflowEngine:
    """Main workflow orchestration engine"""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.workflow_definitions = self._load_workflow_definitions()
        self.swarm_manager = None  # Will be injected
        self.running_workflows = set()
        
    async def initialize(self):
        """Initialize workflow engine"""
        try:
            # Workflow engine will be given swarm manager reference
            logger.info("Workflow engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize workflow engine: {e}")
            raise
    
    def set_swarm_manager(self, swarm_manager: SwarmManager):
        """Set swarm manager reference"""
        self.swarm_manager = swarm_manager
    
    def _load_workflow_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load workflow definitions"""
        return {
            WorkflowType.STEALTH_EXPLOIT: {
                "name": "Stealth Exploitation",
                "description": "Stealthy exploitation workflow with evasion",
                "steps": [
                    {
                        "name": "stealth_recon",
                        "agent_type": AgentType.STEALTH,
                        "task_type": "stealth_reconnaissance",
                        "timeout": 300,
                        "required": True
                    },
                    {
                        "name": "vulnerability_scan",
                        "agent_type": AgentType.SCANNER,
                        "task_type": "vulnerability_scanning",
                        "timeout": 600,
                        "required": True,
                        "depends_on": ["stealth_recon"]
                    },
                    {
                        "name": "exploit_execution",
                        "agent_type": AgentType.EXPLOITER,
                        "task_type": "exploit_execution",
                        "timeout": 900,
                        "required": True,
                        "depends_on": ["vulnerability_scan"]
                    },
                    {
                        "name": "stealth_cleanup",
                        "agent_type": AgentType.STEALTH,
                        "task_type": "cleanup_traces",
                        "timeout": 180,
                        "required": False,
                        "depends_on": ["exploit_execution"]
                    }
                ]
            },
            
            WorkflowType.PASSIVE_RECON: {
                "name": "Passive Reconnaissance",
                "description": "Comprehensive passive information gathering",
                "steps": [
                    {
                        "name": "osint_gathering",
                        "agent_type": AgentType.OSINT,
                        "task_type": "osint_intelligence",
                        "timeout": 600,
                        "required": True
                    },
                    {
                        "name": "dns_enumeration",
                        "agent_type": AgentType.RECONNAISSANCE,
                        "task_type": "dns_enumeration",
                        "timeout": 300,
                        "required": True
                    },
                    {
                        "name": "subdomain_discovery",
                        "agent_type": AgentType.RECONNAISSANCE,
                        "task_type": "subdomain_discovery",
                        "timeout": 900,
                        "required": True,
                        "depends_on": ["dns_enumeration"]
                    },
                    {
                        "name": "threat_intelligence",
                        "agent_type": AgentType.OSINT,
                        "task_type": "threat_intelligence",
                        "timeout": 400,
                        "required": False,
                        "depends_on": ["osint_gathering"]
                    }
                ]
            },
            
            WorkflowType.ACTIVE_ASSESSMENT: {
                "name": "Active Security Assessment",
                "description": "Active reconnaissance and vulnerability assessment",
                "steps": [
                    {
                        "name": "port_scanning",
                        "agent_type": AgentType.RECONNAISSANCE,
                        "task_type": "port_scanning",
                        "timeout": 300,
                        "required": True
                    },
                    {
                        "name": "service_detection",
                        "agent_type": AgentType.RECONNAISSANCE,
                        "task_type": "service_detection",
                        "timeout": 400,
                        "required": True,
                        "depends_on": ["port_scanning"]
                    },
                    {
                        "name": "vulnerability_assessment",
                        "agent_type": AgentType.SCANNER,
                        "task_type": "vulnerability_scanning",
                        "timeout": 1200,
                        "required": True,
                        "depends_on": ["service_detection"]
                    },
                    {
                        "name": "web_application_scan",
                        "agent_type": AgentType.SCANNER,
                        "task_type": "web_scanning",
                        "timeout": 1800,
                        "required": False,
                        "depends_on": ["service_detection"]
                    }
                ]
            },
            
            WorkflowType.VULNERABILITY_SCAN: {
                "name": "Vulnerability Scanning",
                "description": "Comprehensive vulnerability scanning workflow",
                "steps": [
                    {
                        "name": "network_scan",
                        "agent_type": AgentType.SCANNER,
                        "task_type": "network_scanning",
                        "timeout": 600,
                        "required": True
                    },
                    {
                        "name": "web_vulnerability_scan",
                        "agent_type": AgentType.SCANNER,
                        "task_type": "web_scanning",
                        "timeout": 1200,
                        "required": False
                    },
                    {
                        "name": "ssl_analysis",
                        "agent_type": AgentType.SCANNER,
                        "task_type": "ssl_analysis",
                        "timeout": 300,
                        "required": False
                    }
                ]
            },
            
            WorkflowType.OSINT_GATHERING: {
                "name": "OSINT Intelligence Gathering",
                "description": "Comprehensive open source intelligence gathering",
                "steps": [
                    {
                        "name": "social_media_intel",
                        "agent_type": AgentType.OSINT,
                        "task_type": "social_media_intelligence",
                        "timeout": 900,
                        "required": True
                    },
                    {
                        "name": "breach_data_analysis",
                        "agent_type": AgentType.OSINT,
                        "task_type": "breach_data_analysis",
                        "timeout": 600,
                        "required": False
                    },
                    {
                        "name": "dark_web_monitoring",
                        "agent_type": AgentType.OSINT,
                        "task_type": "dark_web_monitoring",
                        "timeout": 1200,
                        "required": False
                    },
                    {
                        "name": "reputation_analysis",
                        "agent_type": AgentType.OSINT,
                        "task_type": "reputation_analysis",
                        "timeout": 300,
                        "required": True
                    }
                ]
            },
            
            WorkflowType.EXPLOITATION_CHAIN: {
                "name": "Exploitation Chain",
                "description": "End-to-end exploitation workflow",
                "steps": [
                    {
                        "name": "initial_recon",
                        "agent_type": AgentType.RECONNAISSANCE,
                        "task_type": "port_scanning",
                        "timeout": 300,
                        "required": True
                    },
                    {
                        "name": "vulnerability_identification",
                        "agent_type": AgentType.SCANNER,
                        "task_type": "vulnerability_scanning",
                        "timeout": 600,
                        "required": True,
                        "depends_on": ["initial_recon"]
                    },
                    {
                        "name": "exploitation",
                        "agent_type": AgentType.EXPLOITER,
                        "task_type": "exploit_execution",
                        "timeout": 900,
                        "required": True,
                        "depends_on": ["vulnerability_identification"]
                    },
                    {
                        "name": "privilege_escalation",
                        "agent_type": AgentType.EXPLOITER,
                        "task_type": "privilege_escalation",
                        "timeout": 600,
                        "required": False,
                        "depends_on": ["exploitation"]
                    },
                    {
                        "name": "persistence",
                        "agent_type": AgentType.EXPLOITER,
                        "task_type": "persistence_establishment",
                        "timeout": 400,
                        "required": False,
                        "depends_on": ["privilege_escalation"]
                    },
                    {
                        "name": "lateral_movement",
                        "agent_type": AgentType.EXPLOITER,
                        "task_type": "lateral_movement",
                        "timeout": 800,
                        "required": False,
                        "depends_on": ["persistence"]
                    }
                ]
            }
        }
    
    async def start_workflow(self, workflow_type: WorkflowType, target: str, parameters: Dict[str, Any]) -> str:
        """Start a new workflow"""
        try:
            workflow_id = str(uuid.uuid4())
            
            # Create workflow instance
            workflow = Workflow(
                id=workflow_id,
                type=workflow_type,
                target=target,
                status=WorkflowStatus.PENDING,
                created_at=datetime.utcnow(),
                parameters=parameters
            )
            
            self.workflows[workflow_id] = workflow
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow(workflow_id))
            
            logger.info(f"Workflow {workflow_id} started for target {target}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            raise
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute workflow steps"""
        try:
            workflow = self.workflows[workflow_id]
            workflow_def = self.workflow_definitions[workflow.type]
            
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.utcnow()
            self.running_workflows.add(workflow_id)
            
            logger.info(f"Executing workflow {workflow_id} ({workflow.type})")
            
            # Create tasks for all steps
            tasks = []
            for step in workflow_def["steps"]:
                task = Task(
                    id=str(uuid.uuid4()),
                    workflow_id=workflow_id,
                    task_type=step["task_type"],
                    target=workflow.target,
                    parameters={
                        **workflow.parameters,
                        "step_name": step["name"],
                        "timeout": step["timeout"]
                    },
                    status=TaskStatus.PENDING,
                    created_at=datetime.utcnow()
                )
                tasks.append((task, step))
            
            workflow.tasks = [task for task, _ in tasks]
            
            # Execute tasks based on dependencies
            await self._execute_tasks_with_dependencies(tasks)
            
            # Check final status
            if all(task.status == TaskStatus.COMPLETED for task, step in tasks if step["required"]):
                workflow.status = WorkflowStatus.COMPLETED
                workflow.results = await self._collect_workflow_results(workflow_id)
            else:
                workflow.status = WorkflowStatus.FAILED
                workflow.results = {"error": "One or more required tasks failed"}
            
            workflow.completed_at = datetime.utcnow()
            self.running_workflows.discard(workflow_id)
            
            logger.info(f"Workflow {workflow_id} completed with status: {workflow.status}")
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            self.running_workflows.discard(workflow_id)
    
    async def _execute_tasks_with_dependencies(self, tasks: List[Tuple[Task, Dict[str, Any]]]):
        """Execute tasks while respecting dependencies"""
        completed_tasks = set()
        executing_tasks = set()
        
        while len(completed_tasks) < len(tasks):
            # Find tasks that can be executed
            ready_tasks = []
            
            for task, step in tasks:
                if task.id in completed_tasks or task.id in executing_tasks:
                    continue
                
                # Check dependencies
                dependencies = step.get("depends_on", [])
                if all(dep in [s["name"] for t, s in tasks if t.id in completed_tasks] for dep in dependencies):
                    ready_tasks.append((task, step))
            
            if not ready_tasks:
                # No more tasks can be executed
                break
            
            # Execute ready tasks
            execution_futures = []
            for task, step in ready_tasks:
                executing_tasks.add(task.id)
                future = asyncio.create_task(self._execute_single_task(task, step))
                execution_futures.append((future, task.id, step["name"]))
            
            # Wait for tasks to complete
            if execution_futures:
                done, pending = await asyncio.wait(
                    [future for future, _, _ in execution_futures],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for future in done:
                    # Find corresponding task
                    for exec_future, task_id, step_name in execution_futures:
                        if exec_future == future:
                            executing_tasks.discard(task_id)
                            completed_tasks.add(task_id)
                            break
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.1)
    
    async def _execute_single_task(self, task: Task, step: Dict[str, Any]) -> bool:
        """Execute a single task"""
        try:
            # Find available agent
            available_agents = await self.swarm_manager.get_available_agents(task.task_type)
            
            if not available_agents:
                # Try to deploy new agent
                agent_id = await self.swarm_manager.deploy_agent(step["agent_type"], {})
                await asyncio.sleep(5)  # Wait for agent to start
                available_agents = await self.swarm_manager.get_available_agents(task.task_type)
            
            if not available_agents:
                logger.error(f"No agents available for task {task.id}")
                task.status = TaskStatus.FAILED
                task.error = "No agents available"
                return False
            
            # Assign task to agent
            success = await self.swarm_manager.assign_task(available_agents[0].id, task)
            
            if not success:
                logger.error(f"Failed to assign task {task.id}")
                task.status = TaskStatus.FAILED
                task.error = "Failed to assign task"
                return False
            
            # Wait for task completion with timeout
            timeout = step.get("timeout", 600)
            start_time = datetime.utcnow()
            
            while task.status == TaskStatus.RUNNING:
                if (datetime.utcnow() - start_time).total_seconds() > timeout:
                    logger.error(f"Task {task.id} timed out")
                    task.status = TaskStatus.FAILED
                    task.error = "Task timed out"
                    return False
                
                await asyncio.sleep(5)
            
            return task.status == TaskStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return False
    
    async def _collect_workflow_results(self, workflow_id: str) -> Dict[str, Any]:
        """Collect and aggregate workflow results"""
        try:
            workflow = self.workflows[workflow_id]
            results = {
                "workflow_id": workflow_id,
                "target": workflow.target,
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                "tasks": {},
                "summary": {
                    "total_tasks": len(workflow.tasks),
                    "completed_tasks": len([t for t in workflow.tasks if t.status == TaskStatus.COMPLETED]),
                    "failed_tasks": len([t for t in workflow.tasks if t.status == TaskStatus.FAILED]),
                    "vulnerabilities_found": 0,
                    "intelligence_gathered": 0
                }
            }
            
            # Collect individual task results
            for task in workflow.tasks:
                results["tasks"][task.id] = {
                    "task_type": task.task_type,
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error
                }
                
                # Aggregate findings
                if task.result:
                    if "vulnerabilities" in task.result:
                        results["summary"]["vulnerabilities_found"] += len(task.result["vulnerabilities"])
                    if "intelligence" in task.result:
                        results["summary"]["intelligence_gathered"] += len(task.result["intelligence"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error collecting workflow results: {e}")
            return {"error": str(e)}
    
    async def stop_workflow(self, workflow_id: str):
        """Stop running workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.CANCELLED
                workflow.completed_at = datetime.utcnow()
                
                # Cancel running tasks
                for task in workflow.tasks:
                    if task.status == TaskStatus.RUNNING:
                        task.status = TaskStatus.CANCELLED
                        task.completed_at = datetime.utcnow()
                
                self.running_workflows.discard(workflow_id)
                
                logger.info(f"Workflow {workflow_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping workflow {workflow_id}: {e}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            
            return {
                "workflow_id": workflow_id,
                "type": workflow.type.value,
                "target": workflow.target,
                "status": workflow.status.value,
                "created_at": workflow.created_at.isoformat(),
                "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                "parameters": workflow.parameters,
                "tasks": [
                    {
                        "id": task.id,
                        "task_type": task.task_type,
                        "status": task.status.value,
                        "agent_id": task.agent_id,
                        "created_at": task.created_at.isoformat(),
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                        "error": task.error
                    }
                    for task in workflow.tasks
                ],
                "results": workflow.results
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            raise
    
    async def get_active_workflow_count(self) -> int:
        """Get count of active workflows"""
        return len(self.running_workflows)
    
    async def get_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflows"""
        return [
            {
                "id": workflow.id,
                "type": workflow.type.value,
                "target": workflow.target,
                "status": workflow.status.value,
                "created_at": workflow.created_at.isoformat()
            }
            for workflow in self.workflows.values()
        ]
    
    async def trigger_exploitation(self, vulnerability_data: Dict[str, Any]):
        """Trigger exploitation workflow based on vulnerability"""
        try:
            if vulnerability_data.get("severity") == "critical":
                # Start exploitation chain
                await self.start_workflow(
                    WorkflowType.EXPLOITATION_CHAIN,
                    vulnerability_data["target"],
                    {
                        "vulnerability": vulnerability_data,
                        "stealth_level": 8,
                        "priority": 10
                    }
                )
                
        except Exception as e:
            logger.error(f"Error triggering exploitation: {e}")
    
    async def analyze_intelligence_patterns(self, intelligence_data: Dict[str, Any]):
        """Analyze intelligence patterns and trigger workflows"""
        try:
            # Analyze patterns and trigger appropriate workflows
            if intelligence_data.get("threat_level") == "high":
                # Trigger active assessment
                await self.start_workflow(
                    WorkflowType.ACTIVE_ASSESSMENT,
                    intelligence_data["target"],
                    {
                        "intelligence": intelligence_data,
                        "priority": 8
                    }
                )
                
        except Exception as e:
            logger.error(f"Error analyzing intelligence patterns: {e}")
    
    def get_workflow_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get available workflow definitions"""
        return self.workflow_definitions