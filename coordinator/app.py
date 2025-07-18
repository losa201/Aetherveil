"""
Aetherveil Sentinel Coordinator Service
Main orchestration service for the cybersecurity organism
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import zmq
import zmq.asyncio

from config.config import config
from coordinator.models import *
from coordinator.security import SecurityManager
from coordinator.swarm_manager import SwarmManager
from coordinator.workflow_engine import WorkflowEngine
from knowledge_graph.graph_manager import GraphManager
from rl_agent.agent import RLAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
sec_manager = SecurityManager()

# Global managers
swarm_manager = SwarmManager()
workflow_engine = WorkflowEngine()
graph_manager = GraphManager()
rl_agent = RLAgent()

# ZMQ Context
zmq_context = zmq.asyncio.Context()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Aetherveil Sentinel Coordinator...")
    await initialize_services()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Aetherveil Sentinel Coordinator...")
    await cleanup_services()

app = FastAPI(
    title="Aetherveil Sentinel Coordinator",
    description="Autonomous cybersecurity organism coordinator",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_services():
    """Initialize all services"""
    try:
        await graph_manager.initialize()
        await swarm_manager.initialize()
        await workflow_engine.initialize()
        await rl_agent.initialize()
        
        # Start ZMQ listener
        asyncio.create_task(zmq_listener())
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

async def cleanup_services():
    """Clean up all services"""
    try:
        await swarm_manager.cleanup()
        await graph_manager.cleanup()
        zmq_context.term()
        logger.info("All services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

async def zmq_listener():
    """Listen for messages from swarm agents"""
    socket = zmq_context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{config.network.zmq_port}")
    
    logger.info(f"ZMQ listener started on port {config.network.zmq_port}")
    
    try:
        while True:
            try:
                message = await socket.recv_json()
                await process_agent_message(message)
            except Exception as e:
                logger.error(f"Error processing ZMQ message: {e}")
    except asyncio.CancelledError:
        logger.info("ZMQ listener cancelled")
    finally:
        socket.close()

async def process_agent_message(message: Dict):
    """Process incoming message from swarm agent"""
    try:
        message_type = message.get("type")
        agent_id = message.get("agent_id")
        
        if message_type == "task_result":
            await handle_task_result(message)
        elif message_type == "agent_status":
            await handle_agent_status(message)
        elif message_type == "vulnerability_found":
            await handle_vulnerability(message)
        elif message_type == "intelligence_data":
            await handle_intelligence_data(message)
        else:
            logger.warning(f"Unknown message type: {message_type}")
            
    except Exception as e:
        logger.error(f"Error processing agent message: {e}")

async def handle_task_result(message: Dict):
    """Handle task completion from agent"""
    task_id = message.get("task_id")
    result = message.get("result")
    
    # Update task status
    await swarm_manager.update_task_status(task_id, "completed", result)
    
    # Update knowledge graph
    if result.get("findings"):
        await graph_manager.add_findings(result["findings"])
    
    # Train RL agent
    reward = calculate_reward(result)
    await rl_agent.update_experience(task_id, reward)

async def handle_agent_status(message: Dict):
    """Handle agent status update"""
    agent_id = message.get("agent_id")
    status = message.get("status")
    
    await swarm_manager.update_agent_status(agent_id, status)

async def handle_vulnerability(message: Dict):
    """Handle vulnerability discovery"""
    vuln_data = message.get("vulnerability")
    
    # Add to knowledge graph
    await graph_manager.add_vulnerability(vuln_data)
    
    # Trigger exploitation workflow if critical
    if vuln_data.get("severity") == "critical":
        await workflow_engine.trigger_exploitation(vuln_data)

async def handle_intelligence_data(message: Dict):
    """Handle OSINT intelligence data"""
    intel_data = message.get("data")
    
    # Add to knowledge graph
    await graph_manager.add_intelligence(intel_data)
    
    # Analyze patterns
    await workflow_engine.analyze_intelligence_patterns(intel_data)

def calculate_reward(result: Dict) -> float:
    """Calculate reward for RL agent based on task result"""
    base_reward = 1.0
    
    # Bonus for finding vulnerabilities
    if result.get("vulnerabilities_found", 0) > 0:
        base_reward += result["vulnerabilities_found"] * 2.0
    
    # Bonus for stealth
    if result.get("stealth_score", 0) > 0.8:
        base_reward += 1.0
    
    # Penalty for errors
    if result.get("errors", 0) > 0:
        base_reward -= result["errors"] * 0.5
    
    return max(0.0, base_reward)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    return sec_manager.validate_token(credentials.credentials)

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "coordinator": "running",
            "swarm_manager": swarm_manager.is_healthy(),
            "knowledge_graph": await graph_manager.is_healthy(),
            "rl_agent": rl_agent.is_healthy()
        }
    }

@app.post("/workflows/start")
async def start_workflow(
    workflow_request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Start a new workflow"""
    try:
        workflow_id = await workflow_engine.start_workflow(
            workflow_request.workflow_type,
            workflow_request.target,
            workflow_request.parameters
        )
        
        return {"workflow_id": workflow_id, "status": "started"}
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/{workflow_id}")
async def get_workflow_status(
    workflow_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get workflow status"""
    try:
        status = await workflow_engine.get_workflow_status(workflow_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflows/{workflow_id}/stop")
async def stop_workflow(
    workflow_id: str,
    current_user: str = Depends(get_current_user)
):
    """Stop a running workflow"""
    try:
        await workflow_engine.stop_workflow(workflow_id)
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Failed to stop workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/swarm/agents")
async def get_swarm_agents(current_user: str = Depends(get_current_user)):
    """Get swarm agent status"""
    try:
        agents = await swarm_manager.get_agents()
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Failed to get swarm agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/swarm/agents/deploy")
async def deploy_agent(
    agent_request: AgentDeployRequest,
    current_user: str = Depends(get_current_user)
):
    """Deploy a new swarm agent"""
    try:
        agent_id = await swarm_manager.deploy_agent(
            agent_request.agent_type,
            agent_request.configuration
        )
        return {"agent_id": agent_id, "status": "deployed"}
    except Exception as e:
        logger.error(f"Failed to deploy agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/query")
async def query_knowledge_graph(
    query_request: GraphQuery,
    current_user: str = Depends(get_current_user)
):
    """Query the knowledge graph"""
    try:
        results = await graph_manager.execute_query(query_request.query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to query knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/attack-paths")
async def get_attack_paths(
    source: str,
    target: str,
    current_user: str = Depends(get_current_user)
):
    """Get attack paths between two nodes"""
    try:
        paths = await graph_manager.find_attack_paths(source, target)
        return {"paths": paths}
    except Exception as e:
        logger.error(f"Failed to get attack paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rl/train")
async def train_rl_agent(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Train the reinforcement learning agent"""
    try:
        background_tasks.add_task(
            rl_agent.train,
            train_request.episodes,
            train_request.curriculum
        )
        return {"status": "training_started"}
    except Exception as e:
        logger.error(f"Failed to start RL training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rl/stats")
async def get_rl_stats(current_user: str = Depends(get_current_user)):
    """Get RL agent statistics"""
    try:
        stats = await rl_agent.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get RL stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/generate")
async def generate_report(
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Generate a security report"""
    try:
        from coordinator.report_generator import ReportGenerator
        
        report_generator = ReportGenerator()
        report_id = await report_generator.generate_report(
            report_request.report_type,
            report_request.parameters
        )
        
        return {"report_id": report_id, "status": "generating"}
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics(current_user: str = Depends(get_current_user)):
    """Get system metrics"""
    try:
        metrics = {
            "swarm_agents": await swarm_manager.get_agent_count(),
            "active_workflows": await workflow_engine.get_active_workflow_count(),
            "knowledge_graph_nodes": await graph_manager.get_node_count(),
            "vulnerabilities_found": await graph_manager.get_vulnerability_count(),
            "rl_training_episodes": await rl_agent.get_training_episodes()
        }
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "coordinator.app:app",
        host="0.0.0.0",
        port=config.network.coordinator_port,
        log_level="info",
        access_log=True
    )