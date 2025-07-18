"""
Orchestrator API Endpoints

REST API endpoints for workflow management and orchestration including
workflow creation, execution, monitoring, and result correlation.

Security Level: DEFENSIVE_ONLY
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, status
from pydantic import BaseModel, Field, validator

from ...modules import get_module
from ...modules.orchestrator import (
    WorkflowType, WorkflowDefinition, WorkflowExecution, WorkflowStatus,
    ExecutionMode, TaskStatus
)
from . import STANDARD_RESPONSES, APIResponseStatus

router = APIRouter()

# Pydantic models
class WorkflowTypeEnum(str, Enum):
    """Workflow type enumeration"""
    RECONNAISSANCE_ONLY = "reconnaissance_only"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    PENETRATION_TEST = "penetration_test"
    THREAT_HUNTING = "threat_hunting"
    COMPLIANCE_AUDIT = "compliance_audit"
    INCIDENT_RESPONSE = "incident_response"
    CUSTOM_WORKFLOW = "custom_workflow"

class ExecutionModeEnum(str, Enum):
    """Execution mode enumeration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    CONDITIONAL = "conditional"

class WorkflowStatusEnum(str, Enum):
    """Workflow status enumeration"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStatusEnum(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class WorkflowCreateRequest(BaseModel):
    """Workflow creation request model"""
    workflow_type: WorkflowTypeEnum = Field(..., description="Type of workflow to create")
    target: str = Field(..., description="Target for the workflow")
    name: Optional[str] = Field(None, description="Custom workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    execution_mode: ExecutionModeEnum = Field(default=ExecutionModeEnum.SEQUENTIAL, description="Execution mode")
    intensity: str = Field(default="normal", description="Workflow intensity level")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional workflow options")
    
    @validator('target')
    def validate_target(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Target cannot be empty')
        return v.strip()

class WorkflowResponse(BaseModel):
    """Workflow response model"""
    workflow_id: str
    name: str
    type: WorkflowTypeEnum
    task_count: int
    execution_mode: ExecutionModeEnum
    created_at: datetime
    target: str
    description: Optional[str] = None

class WorkflowExecutionResponse(BaseModel):
    """Workflow execution response model"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatusEnum
    progress: float
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    completed_tasks: int
    total_tasks: int
    errors: List[str] = []
    metadata: Dict[str, Any] = {}

class WorkflowOperationResponse(BaseModel):
    """Workflow operation response model"""
    status: APIResponseStatus
    operation_id: str
    workflow_id: str
    execution: WorkflowExecutionResponse
    metadata: Dict[str, Any] = {}

class OrchestratorStatusResponse(BaseModel):
    """Orchestrator module status response"""
    module: str
    status: str
    version: str
    workflow_definitions: int
    active_executions: int
    total_executions: int
    available_modules: List[str]
    workflow_types: List[str]

# Dependency to get orchestrator module
async def get_orchestrator_module():
    """Get orchestrator module instance"""
    module = get_module("orchestrator")
    if not module:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator module not available"
        )
    return module

# Endpoints
@router.get("/status", response_model=OrchestratorStatusResponse, responses=STANDARD_RESPONSES)
async def get_orchestrator_status(module=Depends(get_orchestrator_module)):
    """Get orchestrator module status and statistics"""
    try:
        status_info = await module.get_status()
        return OrchestratorStatusResponse(**status_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )

@router.post("/workflows", response_model=WorkflowResponse, responses=STANDARD_RESPONSES)
async def create_workflow(
    workflow_request: WorkflowCreateRequest,
    module=Depends(get_orchestrator_module)
):
    """Create a new workflow"""
    try:
        # Create workflow
        workflow = module.create_workflow(
            workflow_type=WorkflowType(workflow_request.workflow_type.value),
            target=workflow_request.target,
            intensity=workflow_request.intensity,
            **workflow_request.options
        )
        
        # Update workflow with custom details if provided
        if workflow_request.name:
            workflow.name = workflow_request.name
        if workflow_request.description:
            workflow.description = workflow_request.description
        if workflow_request.execution_mode:
            workflow.execution_mode = ExecutionMode(workflow_request.execution_mode.value)
        
        return WorkflowResponse(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            type=WorkflowTypeEnum(workflow.workflow_type.value),
            task_count=len(workflow.tasks),
            execution_mode=ExecutionModeEnum(workflow.execution_mode.value),
            created_at=datetime.utcnow(),
            target=workflow_request.target,
            description=workflow.description
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow creation failed: {str(e)}"
        )

@router.post("/workflows/{workflow_id}/execute", response_model=WorkflowOperationResponse, responses=STANDARD_RESPONSES)
async def execute_workflow(
    workflow_id: str = Path(..., description="Workflow ID to execute"),
    module=Depends(get_orchestrator_module)
):
    """Execute a workflow"""
    try:
        # Execute workflow
        start_time = datetime.utcnow()
        execution = await module.execute_workflow(workflow_id)
        
        # Convert execution to response format
        execution_response = WorkflowExecutionResponse(
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            status=WorkflowStatusEnum(execution.status.value),
            progress=execution.progress,
            start_time=execution.start_time,
            end_time=execution.end_time,
            duration=execution.end_time.timestamp() - execution.start_time.timestamp() if execution.end_time else None,
            completed_tasks=execution.completed_tasks,
            total_tasks=execution.total_tasks,
            errors=execution.errors,
            metadata=execution.metadata
        )
        
        return WorkflowOperationResponse(
            status=APIResponseStatus.COMPLETED if execution.status == WorkflowStatus.COMPLETED else APIResponseStatus.IN_PROGRESS,
            operation_id=f"workflow_exec_{int(start_time.timestamp())}",
            workflow_id=workflow_id,
            execution=execution_response,
            metadata={
                "workflow_type": execution.metadata.get("workflow_type"),
                "correlations": execution.metadata.get("correlations", {})
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )

@router.get("/workflows", response_model=List[WorkflowResponse], responses=STANDARD_RESPONSES)
async def list_workflows(
    workflow_type: Optional[WorkflowTypeEnum] = Query(None, description="Filter by workflow type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum workflows to return"),
    module=Depends(get_orchestrator_module)
):
    """List all workflows"""
    try:
        workflows = module.list_workflows()
        
        # Filter by type if specified
        if workflow_type:
            workflows = [w for w in workflows if w["type"] == workflow_type.value]
        
        # Limit results
        if limit:
            workflows = workflows[:limit]
        
        # Convert to response format
        workflow_responses = []
        for workflow in workflows:
            workflow_responses.append(WorkflowResponse(
                workflow_id=workflow["workflow_id"],
                name=workflow["name"],
                type=WorkflowTypeEnum(workflow["type"]),
                task_count=workflow["task_count"],
                execution_mode=ExecutionModeEnum(workflow["execution_mode"]),
                created_at=datetime.utcnow(),  # This should come from the workflow definition
                target="",  # This should come from the workflow definition
                description=""  # This should come from the workflow definition
            ))
        
        return workflow_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )

@router.get("/executions", response_model=List[WorkflowExecutionResponse], responses=STANDARD_RESPONSES)
async def list_executions(
    status: Optional[WorkflowStatusEnum] = Query(None, description="Filter by execution status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum executions to return"),
    module=Depends(get_orchestrator_module)
):
    """List all workflow executions"""
    try:
        executions = module.list_executions()
        
        # Filter by status if specified
        if status:
            executions = [e for e in executions if e["status"] == status.value]
        
        # Limit results
        if limit:
            executions = executions[:limit]
        
        # Convert to response format
        execution_responses = []
        for execution in executions:
            execution_responses.append(WorkflowExecutionResponse(
                execution_id=execution["execution_id"],
                workflow_id=execution["workflow_id"],
                status=WorkflowStatusEnum(execution["status"]),
                progress=execution["progress"],
                start_time=datetime.fromisoformat(execution["start_time"]),
                end_time=datetime.fromisoformat(execution["end_time"]) if execution.get("end_time") else None,
                duration=execution.get("duration"),
                completed_tasks=0,  # This should come from the execution data
                total_tasks=0,  # This should come from the execution data
                errors=[],  # This should come from the execution data
                metadata={}
            ))
        
        return execution_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list executions: {str(e)}"
        )

@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse, responses=STANDARD_RESPONSES)
async def get_execution_status(
    execution_id: str = Path(..., description="Execution ID to check"),
    module=Depends(get_orchestrator_module)
):
    """Get workflow execution status"""
    try:
        execution = module.get_workflow_status(execution_id)
        
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found"
            )
        
        return WorkflowExecutionResponse(
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            status=WorkflowStatusEnum(execution.status.value),
            progress=execution.progress,
            start_time=execution.start_time,
            end_time=execution.end_time,
            duration=execution.end_time.timestamp() - execution.start_time.timestamp() if execution.end_time else None,
            completed_tasks=execution.completed_tasks,
            total_tasks=execution.total_tasks,
            errors=execution.errors,
            metadata=execution.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution status: {str(e)}"
        )

@router.post("/executions/{execution_id}/pause", responses=STANDARD_RESPONSES)
async def pause_execution(
    execution_id: str = Path(..., description="Execution ID to pause"),
    module=Depends(get_orchestrator_module)
):
    """Pause workflow execution"""
    try:
        success = module.workflow_engine.pause_execution(execution_id)
        
        if success:
            return {"message": f"Execution {execution_id} paused successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found or cannot be paused"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause execution: {str(e)}"
        )

@router.post("/executions/{execution_id}/resume", responses=STANDARD_RESPONSES)
async def resume_execution(
    execution_id: str = Path(..., description="Execution ID to resume"),
    module=Depends(get_orchestrator_module)
):
    """Resume workflow execution"""
    try:
        success = module.workflow_engine.resume_execution(execution_id)
        
        if success:
            return {"message": f"Execution {execution_id} resumed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found or cannot be resumed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume execution: {str(e)}"
        )

@router.post("/executions/{execution_id}/cancel", responses=STANDARD_RESPONSES)
async def cancel_execution(
    execution_id: str = Path(..., description="Execution ID to cancel"),
    module=Depends(get_orchestrator_module)
):
    """Cancel workflow execution"""
    try:
        success = module.workflow_engine.cancel_execution(execution_id)
        
        if success:
            return {"message": f"Execution {execution_id} cancelled successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found or cannot be cancelled"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel execution: {str(e)}"
        )

@router.get("/executions/{execution_id}/export", responses=STANDARD_RESPONSES)
async def export_execution_results(
    execution_id: str = Path(..., description="Execution ID to export"),
    format: str = Query("json", regex="^(json|csv)$", description="Export format"),
    module=Depends(get_orchestrator_module)
):
    """Export workflow execution results"""
    try:
        exported_data = module.export_workflow_results(execution_id, format)
        
        if not exported_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found"
            )
        
        if format == "json":
            return {"data": exported_data}
        elif format == "csv":
            return {"data": exported_data}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )

@router.get("/workflow-types", responses=STANDARD_RESPONSES)
async def get_workflow_types():
    """Get available workflow types"""
    return {
        "workflow_types": [
            {
                "name": "reconnaissance_only",
                "description": "Comprehensive reconnaissance and information gathering",
                "estimated_duration": "5-15 minutes",
                "modules_used": ["reconnaissance", "osint"]
            },
            {
                "name": "vulnerability_assessment",
                "description": "Full vulnerability assessment and scanning",
                "estimated_duration": "15-60 minutes",
                "modules_used": ["reconnaissance", "scanning"]
            },
            {
                "name": "penetration_test",
                "description": "Comprehensive penetration testing workflow",
                "estimated_duration": "30-120 minutes",
                "modules_used": ["reconnaissance", "scanning", "exploitation", "stealth"]
            },
            {
                "name": "threat_hunting",
                "description": "Proactive threat hunting and detection",
                "estimated_duration": "20-90 minutes",
                "modules_used": ["reconnaissance", "scanning", "osint", "stealth"]
            },
            {
                "name": "compliance_audit",
                "description": "Security compliance assessment",
                "estimated_duration": "30-90 minutes",
                "modules_used": ["scanning", "reporting"]
            }
        ]
    }

@router.get("/templates", responses=STANDARD_RESPONSES)
async def get_workflow_templates():
    """Get available workflow templates"""
    return {
        "templates": [
            {
                "name": "Basic Network Assessment",
                "type": "vulnerability_assessment",
                "description": "Standard network vulnerability assessment",
                "target_types": ["ip_address", "network_range"],
                "estimated_duration": "30 minutes"
            },
            {
                "name": "Web Application Security Test",
                "type": "penetration_test",
                "description": "Comprehensive web application security testing",
                "target_types": ["url", "domain"],
                "estimated_duration": "60 minutes"
            },
            {
                "name": "Infrastructure Reconnaissance",
                "type": "reconnaissance_only",
                "description": "Passive and active infrastructure reconnaissance",
                "target_types": ["domain", "organization"],
                "estimated_duration": "15 minutes"
            },
            {
                "name": "Threat Intelligence Gathering",
                "type": "threat_hunting",
                "description": "Comprehensive threat intelligence collection",
                "target_types": ["domain", "ip_address", "organization"],
                "estimated_duration": "45 minutes"
            }
        ]
    }