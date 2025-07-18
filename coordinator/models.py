"""
Pydantic models for the Coordinator API
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class WorkflowType(str, Enum):
    """Workflow types"""
    STEALTH_EXPLOIT = "stealth_exploit"
    PASSIVE_RECON = "passive_recon"
    ACTIVE_ASSESSMENT = "active_assessment"
    VULNERABILITY_SCAN = "vulnerability_scan"
    OSINT_GATHERING = "osint_gathering"
    EXPLOITATION_CHAIN = "exploitation_chain"

class AgentType(str, Enum):
    """Agent types"""
    RECONNAISSANCE = "reconnaissance"
    SCANNER = "scanner"
    EXPLOITER = "exploiter"
    OSINT = "osint"
    STEALTH = "stealth"

class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ReportType(str, Enum):
    """Report types"""
    VULNERABILITY = "vulnerability"
    PENETRATION_TEST = "penetration_test"
    OSINT = "osint"
    EXECUTIVE_SUMMARY = "executive_summary"

class WorkflowRequest(BaseModel):
    """Workflow start request"""
    workflow_type: WorkflowType
    target: str = Field(..., description="Target identifier (IP, domain, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    stealth_level: int = Field(default=5, ge=1, le=10)

class AgentDeployRequest(BaseModel):
    """Agent deployment request"""
    agent_type: AgentType
    configuration: Dict[str, Any] = Field(default_factory=dict)
    resources: Dict[str, Any] = Field(default_factory=dict)

class GraphQuery(BaseModel):
    """Knowledge graph query request"""
    query: str = Field(..., description="Cypher query")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class TrainRequest(BaseModel):
    """RL training request"""
    episodes: int = Field(default=1000, ge=1)
    curriculum: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

class ReportRequest(BaseModel):
    """Report generation request"""
    report_type: ReportType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    format: str = Field(default="pdf", pattern="^(pdf|json|html)$")

class Task(BaseModel):
    """Task model"""
    id: str
    workflow_id: str
    agent_id: Optional[str] = None
    task_type: str
    target: str
    parameters: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Agent(BaseModel):
    """Agent model"""
    id: str
    type: AgentType
    status: str
    created_at: datetime
    last_heartbeat: Optional[datetime] = None
    current_task: Optional[str] = None
    capabilities: List[str]
    configuration: Dict[str, Any]
    metrics: Dict[str, Any] = Field(default_factory=dict)

class Workflow(BaseModel):
    """Workflow model"""
    id: str
    type: WorkflowType
    target: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks: List[Task] = Field(default_factory=list)
    parameters: Dict[str, Any]
    results: Dict[str, Any] = Field(default_factory=dict)

class Vulnerability(BaseModel):
    """Vulnerability model"""
    id: str
    cve_id: Optional[str] = None
    title: str
    description: str
    severity: str
    cvss_score: Optional[float] = None
    target: str
    service: Optional[str] = None
    port: Optional[int] = None
    exploit_available: bool = False
    discovered_at: datetime
    verified: bool = False
    false_positive: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Intelligence(BaseModel):
    """Intelligence data model"""
    id: str
    source: str
    data_type: str
    target: str
    data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    collected_at: datetime
    verified: bool = False
    tags: List[str] = Field(default_factory=list)

class AttackPath(BaseModel):
    """Attack path model"""
    id: str
    source: str
    target: str
    path: List[str]
    difficulty: int = Field(ge=1, le=10)
    stealth_required: int = Field(ge=1, le=10)
    tools_required: List[str] = Field(default_factory=list)
    estimated_time: int  # in minutes
    success_probability: float = Field(ge=0.0, le=1.0)

class SystemMetrics(BaseModel):
    """System metrics model"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_agents: int
    active_workflows: int
    completed_tasks: int
    failed_tasks: int
    vulnerabilities_found: int

class ApiResponse(BaseModel):
    """Standard API response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str = "1.0.0"

class AuthToken(BaseModel):
    """Authentication token"""
    token: str
    expires_at: datetime
    user_id: str
    permissions: List[str]

class SecurityEvent(BaseModel):
    """Security event model"""
    id: str
    event_type: str
    source: str
    target: str
    severity: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    handled: bool = False

class Configuration(BaseModel):
    """System configuration"""
    id: str
    key: str
    value: Any
    description: Optional[str] = None
    category: str
    updated_at: datetime
    updated_by: str