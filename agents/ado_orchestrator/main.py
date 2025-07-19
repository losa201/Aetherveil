"""
Autonomous DevOps Orchestrator (ADO)
Multi-agent LLM system for intelligent DevOps decision making
Phase 2: Medium-term implementation
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from google.cloud import pubsub_v1, bigquery, aiplatform, firestore
from google.cloud import secretmanager
import httpx
from pydantic import BaseModel, Field
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.getenv("PROJECT_ID", "tidy-computing-465909-i3")
REGION = os.getenv("REGION", "us-central1")

# Initialize clients
app = FastAPI(
    title="Autonomous DevOps Orchestrator (ADO)",
    description="Multi-agent LLM system for intelligent DevOps automation",
    version="2.0.0"
)
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
bigquery_client = bigquery.Client(project=PROJECT_ID)
secret_client = secretmanager.SecretManagerServiceClient()
firestore_client = firestore.Client(project=PROJECT_ID)

class AgentRole(str, Enum):
    BUILD_OPTIMIZER = "build_optimizer"
    SECURITY_GUARDIAN = "security_guardian"
    PERFORMANCE_ANALYST = "performance_analyst"
    RELEASE_STRATEGIST = "release_strategist"
    INFRASTRUCTURE_MANAGER = "infrastructure_manager"

class DecisionContext(BaseModel):
    event_type: str
    source_data: Dict[str, Any]
    historical_context: Dict[str, Any] = Field(default_factory=dict)
    current_state: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)

class AgentDecision(BaseModel):
    agent_role: AgentRole
    decision_type: str
    confidence: float  # 0-1
    rationale: str
    recommended_actions: List[Dict[str, Any]]
    estimated_impact: Dict[str, Any]
    risk_assessment: Dict[str, Any]

class OrchestrationPlan(BaseModel):
    context: DecisionContext
    agent_decisions: List[AgentDecision]
    consensus_decision: Optional[Dict[str, Any]] = None
    execution_plan: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LLMAgent:
    """Base class for LLM-powered agents"""
    
    def __init__(self, role: AgentRole, model_name: str = "gpt-4"):
        self.role = role
        self.model_name = model_name
        self.client = None  # Will be initialized with API key
        self.knowledge_base = {}
        self.decision_history = []
    
    async def initialize(self):
        """Initialize the agent with API credentials"""
        try:
            # Get OpenAI API key from Secret Manager
            secret_name = f"projects/{PROJECT_ID}/secrets/openai-api-key/versions/latest"
            response = secret_client.access_secret_version(request={"name": secret_name})
            api_key = response.payload.data.decode("UTF-8")
            
            openai.api_key = api_key
            self.client = openai
            
            # Load agent-specific knowledge
            await self._load_knowledge_base()
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.role}: {e}")
    
    async def _load_knowledge_base(self):
        """Load agent-specific knowledge from BigQuery"""
        
        knowledge_queries = {
            AgentRole.BUILD_OPTIMIZER: """
                SELECT 
                    workflow_name,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*) as success_rate
                FROM `{}.pipeline_analytics.pipeline_runs`
                WHERE started_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                GROUP BY workflow_name
            """.format(PROJECT_ID),
            
            AgentRole.SECURITY_GUARDIAN: """
                SELECT 
                    repository,
                    security_scan_results,
                    COUNT(*) as scan_count
                FROM `{}.pipeline_analytics.pipeline_runs`
                WHERE security_scan_results IS NOT NULL
                AND started_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                GROUP BY repository, security_scan_results
            """.format(PROJECT_ID),
            
            AgentRole.PERFORMANCE_ANALYST: """
                SELECT 
                    workflow_name,
                    resource_usage,
                    duration_seconds,
                    timestamp
                FROM `{}.pipeline_analytics.pipeline_runs`
                WHERE resource_usage IS NOT NULL
                AND started_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
                ORDER BY timestamp DESC
            """.format(PROJECT_ID)
        }
        
        query = knowledge_queries.get(self.role)
        if query:
            try:
                query_job = bigquery_client.query(query)
                results = [dict(row) for row in query_job.result()]
                self.knowledge_base = {"historical_data": results}
            except Exception as e:
                logger.error(f"Error loading knowledge for {self.role}: {e}")
    
    async def analyze_and_decide(self, context: DecisionContext) -> AgentDecision:
        """Analyze context and make a decision"""
        
        # Prepare prompt based on agent role
        prompt = await self._prepare_prompt(context)
        
        try:
            # Call LLM for analysis
            response = await self._call_llm(prompt)
            
            # Parse and structure the response
            decision = await self._parse_llm_response(response, context)
            
            # Store decision in history
            self.decision_history.append({
                "timestamp": datetime.now(timezone.utc),
                "context": context.dict(),
                "decision": decision.dict()
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in agent decision making for {self.role}: {e}")
            
            # Return fallback decision
            return AgentDecision(
                agent_role=self.role,
                decision_type="fallback",
                confidence=0.1,
                rationale=f"Error in analysis: {str(e)}",
                recommended_actions=[],
                estimated_impact={},
                risk_assessment={"risk_level": "unknown"}
            )
    
    async def _prepare_prompt(self, context: DecisionContext) -> str:
        """Prepare role-specific prompt for LLM"""
        
        base_prompt = f"""
        You are an expert {self.role.value.replace('_', ' ')} AI agent in an autonomous DevOps platform.
        
        Context:
        - Event Type: {context.event_type}
        - Source Data: {json.dumps(context.source_data, indent=2)}
        - Historical Context: {json.dumps(context.historical_context, indent=2)}
        - Current State: {json.dumps(context.current_state, indent=2)}
        - Constraints: {json.dumps(context.constraints, indent=2)}
        
        Your Knowledge Base:
        {json.dumps(self.knowledge_base, indent=2)[:2000]}...
        
        """
        
        role_specific_prompts = {
            AgentRole.BUILD_OPTIMIZER: """
            As a Build Optimizer, analyze the pipeline performance and recommend optimizations.
            Focus on:
            - Build duration improvements
            - Cache optimization strategies
            - Dependency management
            - Parallelization opportunities
            - Resource allocation optimization
            
            Provide specific, actionable recommendations with confidence scores.
            """,
            
            AgentRole.SECURITY_GUARDIAN: """
            As a Security Guardian, analyze security implications and recommend protective measures.
            Focus on:
            - Vulnerability assessment
            - Compliance requirements
            - Security policy enforcement
            - Threat detection and mitigation
            - Access control optimization
            
            Prioritize security without significantly impacting development velocity.
            """,
            
            AgentRole.PERFORMANCE_ANALYST: """
            As a Performance Analyst, analyze system performance and resource utilization.
            Focus on:
            - Resource usage patterns
            - Performance bottlenecks
            - Scaling recommendations
            - Cost optimization opportunities
            - SLA compliance
            
            Provide data-driven insights with quantified impact estimates.
            """,
            
            AgentRole.RELEASE_STRATEGIST: """
            As a Release Strategist, analyze deployment patterns and recommend release strategies.
            Focus on:
            - Deployment risk assessment
            - Release timing optimization
            - Rollback strategies
            - Feature flag management
            - Progressive deployment techniques
            
            Balance speed to market with stability and risk mitigation.
            """,
            
            AgentRole.INFRASTRUCTURE_MANAGER: """
            As an Infrastructure Manager, analyze infrastructure needs and recommend changes.
            Focus on:
            - Infrastructure scaling
            - Resource provisioning
            - Cost optimization
            - Reliability improvements
            - Disaster recovery planning
            
            Ensure infrastructure supports business objectives efficiently.
            """
        }
        
        return base_prompt + role_specific_prompts.get(self.role, "")
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API"""
        
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert DevOps AI agent. Provide structured, actionable analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    async def _parse_llm_response(self, response: str, context: DecisionContext) -> AgentDecision:
        """Parse LLM response into structured decision"""
        
        # For now, create a structured response
        # In production, this would parse the actual LLM response
        
        return AgentDecision(
            agent_role=self.role,
            decision_type="analysis_complete",
            confidence=0.8,  # Would be parsed from response
            rationale=response[:500],  # Truncate for storage
            recommended_actions=[
                {
                    "action": "optimize_build",
                    "priority": "high",
                    "details": "Specific optimization details would be here"
                }
            ],
            estimated_impact={
                "performance_improvement": "15%",
                "cost_reduction": "10%",
                "risk_level": "low"
            },
            risk_assessment={
                "risk_level": "low",
                "mitigation_strategies": ["rollback_plan", "monitoring"]
            }
        )

class ADOOrchestrator:
    """Main orchestrator for coordinating AI agents"""
    
    def __init__(self):
        self.agents: Dict[AgentRole, LLMAgent] = {}
        self.decision_log = []
        self.execution_engine = ExecutionEngine()
    
    async def initialize(self):
        """Initialize all agents"""
        
        agent_roles = [
            AgentRole.BUILD_OPTIMIZER,
            AgentRole.SECURITY_GUARDIAN,
            AgentRole.PERFORMANCE_ANALYST,
            AgentRole.RELEASE_STRATEGIST,
            AgentRole.INFRASTRUCTURE_MANAGER
        ]
        
        for role in agent_roles:
            agent = LLMAgent(role)
            await agent.initialize()
            self.agents[role] = agent
            
        logger.info(f"Initialized {len(self.agents)} AI agents")
    
    async def orchestrate_decision(self, context: DecisionContext) -> OrchestrationPlan:
        """Orchestrate decision making across multiple agents"""
        
        # Get decisions from relevant agents
        agent_decisions = []
        
        # Determine which agents are relevant for this context
        relevant_agents = self._select_relevant_agents(context)
        
        # Collect decisions from relevant agents
        for agent_role in relevant_agents:
            agent = self.agents.get(agent_role)
            if agent:
                try:
                    decision = await agent.analyze_and_decide(context)
                    agent_decisions.append(decision)
                except Exception as e:
                    logger.error(f"Error getting decision from {agent_role}: {e}")
        
        # Build consensus
        consensus_decision = await self._build_consensus(agent_decisions, context)
        
        # Create execution plan
        execution_plan = await self._create_execution_plan(consensus_decision, agent_decisions)
        
        plan = OrchestrationPlan(
            context=context,
            agent_decisions=agent_decisions,
            consensus_decision=consensus_decision,
            execution_plan=execution_plan,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "orchestrator_version": "2.0.0",
                "num_agents_consulted": len(agent_decisions)
            }
        )
        
        # Log decision
        self.decision_log.append(plan)
        
        # Store plan in Firestore
        await self._store_orchestration_plan(plan)
        
        return plan
    
    async def _store_orchestration_plan(self, plan: OrchestrationPlan):
        """Store orchestration plan in Firestore"""
        
        try:
            plan_doc = {
                "context": plan.context.dict(),
                "agent_decisions": [decision.dict() for decision in plan.agent_decisions],
                "consensus_decision": plan.consensus_decision,
                "execution_plan": plan.execution_plan,
                "metadata": plan.metadata,
                "created_at": datetime.now(timezone.utc)
            }
            
            doc_id = f"{plan.context.event_type}_{int(datetime.now().timestamp())}"
            doc_ref = firestore_client.collection('orchestration_plans').document(doc_id)
            doc_ref.set(plan_doc)
            
            logger.info(f"Stored orchestration plan in Firestore: {doc_id}")
            
        except Exception as e:
            logger.error(f"Error storing orchestration plan: {e}")
    
    def _select_relevant_agents(self, context: DecisionContext) -> List[AgentRole]:
        """Select relevant agents based on context"""
        
        event_type = context.event_type.lower()
        
        # Default to all agents for comprehensive analysis
        all_agents = list(AgentRole)
        
        # Context-specific agent selection
        if "build" in event_type or "compile" in event_type:
            return [AgentRole.BUILD_OPTIMIZER, AgentRole.PERFORMANCE_ANALYST]
        elif "security" in event_type or "vulnerability" in event_type:
            return [AgentRole.SECURITY_GUARDIAN, AgentRole.RELEASE_STRATEGIST]
        elif "deploy" in event_type or "release" in event_type:
            return [AgentRole.RELEASE_STRATEGIST, AgentRole.INFRASTRUCTURE_MANAGER]
        elif "performance" in event_type or "resource" in event_type:
            return [AgentRole.PERFORMANCE_ANALYST, AgentRole.INFRASTRUCTURE_MANAGER]
        else:
            # For unknown events, consult all agents
            return all_agents
    
    async def _build_consensus(self, decisions: List[AgentDecision], context: DecisionContext) -> Dict[str, Any]:
        """Build consensus from agent decisions"""
        
        if not decisions:
            return {"consensus": "no_decisions", "confidence": 0.0}
        
        # Calculate weighted consensus based on confidence scores
        total_confidence = sum(d.confidence for d in decisions)
        
        if total_confidence == 0:
            return {"consensus": "low_confidence", "confidence": 0.0}
        
        # Aggregate recommendations
        all_actions = []
        for decision in decisions:
            for action in decision.recommended_actions:
                action["source_agent"] = decision.agent_role.value
                action["confidence"] = decision.confidence
                all_actions.append(action)
        
        # Sort by priority and confidence
        all_actions.sort(key=lambda x: (x.get("priority", "medium"), x["confidence"]), reverse=True)
        
        # Build consensus decision
        consensus = {
            "consensus": "reached",
            "confidence": total_confidence / len(decisions),
            "primary_recommendation": all_actions[0] if all_actions else None,
            "alternative_recommendations": all_actions[1:3],  # Top 2 alternatives
            "risk_level": self._assess_consensus_risk(decisions),
            "execution_priority": self._determine_execution_priority(decisions)
        }
        
        return consensus
    
    def _assess_consensus_risk(self, decisions: List[AgentDecision]) -> str:
        """Assess overall risk level from agent decisions"""
        
        risk_levels = [d.risk_assessment.get("risk_level", "medium") for d in decisions]
        
        if "high" in risk_levels:
            return "high"
        elif "medium" in risk_levels:
            return "medium"
        else:
            return "low"
    
    def _determine_execution_priority(self, decisions: List[AgentDecision]) -> str:
        """Determine execution priority"""
        
        avg_confidence = sum(d.confidence for d in decisions) / len(decisions)
        
        if avg_confidence > 0.8:
            return "high"
        elif avg_confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    async def _create_execution_plan(self, consensus: Dict[str, Any], decisions: List[AgentDecision]) -> List[Dict[str, Any]]:
        """Create detailed execution plan"""
        
        if consensus.get("consensus") != "reached":
            return []
        
        primary_rec = consensus.get("primary_recommendation")
        if not primary_rec:
            return []
        
        # Create step-by-step execution plan
        execution_steps = [
            {
                "step": 1,
                "action": "validate_prerequisites",
                "description": "Validate prerequisites for recommended action",
                "estimated_duration": 300,  # 5 minutes
                "dependencies": []
            },
            {
                "step": 2,
                "action": primary_rec.get("action", "execute_recommendation"),
                "description": primary_rec.get("details", "Execute primary recommendation"),
                "estimated_duration": 900,  # 15 minutes
                "dependencies": ["validate_prerequisites"]
            },
            {
                "step": 3,
                "action": "monitor_execution",
                "description": "Monitor execution and collect metrics",
                "estimated_duration": 600,  # 10 minutes
                "dependencies": ["execute_recommendation"]
            },
            {
                "step": 4,
                "action": "validate_results",
                "description": "Validate execution results and update knowledge base",
                "estimated_duration": 300,  # 5 minutes
                "dependencies": ["monitor_execution"]
            }
        ]
        
        return execution_steps

class ExecutionEngine:
    """Execute orchestration plans"""
    
    async def execute_plan(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """Execute an orchestration plan"""
        
        execution_result = {
            "plan_id": id(plan),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "executing",
            "completed_steps": [],
            "current_step": None,
            "errors": []
        }
        
        try:
            for step in plan.execution_plan:
                execution_result["current_step"] = step
                
                # Execute step
                step_result = await self._execute_step(step, plan.context)
                
                execution_result["completed_steps"].append({
                    "step": step,
                    "result": step_result,
                    "completed_at": datetime.now(timezone.utc).isoformat()
                })
                
                if not step_result.get("success", False):
                    execution_result["status"] = "failed"
                    execution_result["errors"].append(step_result.get("error", "Unknown error"))
                    break
            
            if execution_result["status"] == "executing":
                execution_result["status"] = "completed"
            
        except Exception as e:
            execution_result["status"] = "error"
            execution_result["errors"].append(str(e))
        
        execution_result["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        # Store execution results
        await self._store_execution_results(execution_result)
        
        return execution_result
    
    async def _execute_step(self, step: Dict[str, Any], context: DecisionContext) -> Dict[str, Any]:
        """Execute a single step"""
        
        action = step.get("action", "")
        
        # Simulate step execution
        # In production, this would call appropriate APIs
        
        if action == "validate_prerequisites":
            return {"success": True, "details": "Prerequisites validated"}
        elif action == "monitor_execution":
            return {"success": True, "details": "Monitoring initiated"}
        elif action == "validate_results":
            return {"success": True, "details": "Results validated"}
        else:
            # Generic execution
            await asyncio.sleep(1)  # Simulate work
            return {"success": True, "details": f"Executed {action}"}
    
    async def _store_execution_results(self, results: Dict[str, Any]):
        """Store execution results in Firestore and BigQuery"""
        
        try:
            # Store detailed results in Firestore
            execution_doc = {
                "execution_id": str(results["plan_id"]),
                "timestamp": datetime.fromisoformat(results["started_at"]),
                "status": results["status"],
                "duration_seconds": (
                    datetime.fromisoformat(results["completed_at"]) - 
                    datetime.fromisoformat(results["started_at"])
                ).total_seconds(),
                "steps_completed": len(results["completed_steps"]),
                "errors": results["errors"],
                "completed_steps": results["completed_steps"],
                "current_step": results.get("current_step")
            }
            
            doc_ref = firestore_client.collection('ado_executions').document(str(results["plan_id"]))
            doc_ref.set(execution_doc)
            logger.info(f"Stored execution results in Firestore for {results['plan_id']}")
            
            # Store summary in BigQuery for analytics
            bq_row = {
                "execution_id": str(results["plan_id"]),
                "timestamp": results["started_at"],
                "status": results["status"],
                "duration_seconds": execution_doc["duration_seconds"],
                "steps_completed": len(results["completed_steps"]),
                "errors": json.dumps(results["errors"]),
                "success": results["status"] == "completed"
            }
            
            table_id = f"{PROJECT_ID}.pipeline_analytics.ado_executions"
            table = bigquery_client.get_table(table_id)
            bq_errors = bigquery_client.insert_rows_json(table, [bq_row])
            if not bq_errors:
                logger.info(f"Stored execution summary in BigQuery for {results['plan_id']}")
                
        except Exception as e:
            logger.error(f"Error storing execution results: {e}")

# Initialize orchestrator
orchestrator = ADOOrchestrator()

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize ADO on startup"""
    await orchestrator.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents_initialized": len(orchestrator.agents)
    }

@app.post("/orchestrate")
async def orchestrate_decision(context: DecisionContext, background_tasks: BackgroundTasks):
    """Orchestrate a decision across AI agents"""
    try:
        plan = await orchestrator.orchestrate_decision(context)
        
        # Execute plan in background
        background_tasks.add_task(orchestrator.execution_engine.execute_plan, plan)
        
        return {
            "status": "orchestration_complete",
            "plan": plan,
            "execution_started": True
        }
        
    except Exception as e:
        logger.error(f"Error in orchestration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    
    agent_status = {}
    for role, agent in orchestrator.agents.items():
        agent_status[role.value] = {
            "initialized": agent.client is not None,
            "knowledge_base_size": len(agent.knowledge_base),
            "decisions_made": len(agent.decision_history),
            "last_decision": agent.decision_history[-1]["timestamp"].isoformat() if agent.decision_history else None
        }
    
    return {
        "agents": agent_status,
        "total_agents": len(orchestrator.agents),
        "total_decisions": len(orchestrator.decision_log)
    }

@app.get("/decisions/history")
async def get_decision_history(limit: int = 10):
    """Get recent decision history"""
    
    recent_decisions = orchestrator.decision_log[-limit:]
    
    return {
        "decisions": [
            {
                "timestamp": decision.metadata.get("timestamp"),
                "context_type": decision.context.event_type,
                "agents_consulted": decision.metadata.get("num_agents_consulted"),
                "consensus_reached": decision.consensus_decision.get("consensus") == "reached" if decision.consensus_decision else False,
                "execution_steps": len(decision.execution_plan)
            }
            for decision in recent_decisions
        ],
        "total_decisions": len(orchestrator.decision_log)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)