"""
Base Agent class for Aetherveil Sentinel
Foundation for all swarm agents with common functionality
"""
import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import zmq
import zmq.asyncio

from config.config import config
from coordinator.security import security_manager
from coordinator.models import TaskStatus

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all swarm agents"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = "initializing"
        self.current_task = None
        self.zmq_context = zmq.asyncio.Context()
        self.coordinator_socket = None
        self.command_socket = None
        self.heartbeat_interval = 30  # seconds
        self.running = False
        self.task_handlers = {}
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "uptime": 0,
            "start_time": datetime.utcnow()
        }
        
        # Register default handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register task handlers"""
        # Must be implemented by subclasses
        pass
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register handler for specific task type"""
        self.task_handlers[task_type] = handler
    
    async def initialize(self):
        """Initialize agent"""
        try:
            # Setup ZMQ sockets
            self.coordinator_socket = self.zmq_context.socket(zmq.PUSH)
            self.coordinator_socket.connect(f"tcp://coordinator:{config.network.zmq_port}")
            
            self.command_socket = self.zmq_context.socket(zmq.PULL)
            self.command_socket.connect(f"tcp://coordinator:{config.network.zmq_port + 1}")
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._command_listener())
            
            # Set status
            self.status = "running"
            await self._send_status_update()
            
            logger.info(f"Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown agent"""
        try:
            self.running = False
            self.status = "shutting_down"
            
            # Send final status update
            await self._send_status_update()
            
            # Close sockets
            if self.coordinator_socket:
                self.coordinator_socket.close()
            if self.command_socket:
                self.command_socket.close()
            
            # Terminate context
            self.zmq_context.term()
            
            logger.info(f"Agent {self.agent_id} shutdown successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down agent {self.agent_id}: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat to coordinator"""
        while self.running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self):
        """Send heartbeat message"""
        try:
            self.metrics["uptime"] = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()
            
            message = {
                "type": "agent_status",
                "agent_id": self.agent_id,
                "status": self.status,
                "current_task": self.current_task,
                "metrics": self.metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
    
    async def _send_status_update(self):
        """Send status update to coordinator"""
        try:
            message = {
                "type": "agent_status",
                "agent_id": self.agent_id,
                "status": self.status,
                "agent_type": self.agent_type,
                "capabilities": self.capabilities,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending status update: {e}")
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send encrypted message to coordinator"""
        try:
            # Encrypt message
            encrypted_message = security_manager.encrypt_data(json.dumps(message))
            
            await self.coordinator_socket.send_string(encrypted_message)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def _command_listener(self):
        """Listen for commands from coordinator"""
        while self.running:
            try:
                # Set socket timeout
                await asyncio.wait_for(self._process_commands(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in command listener: {e}")
                await asyncio.sleep(1)
    
    async def _process_commands(self):
        """Process incoming commands"""
        try:
            # Check for messages
            encrypted_message = await self.command_socket.recv_string(zmq.NOBLOCK)
            
            # Decrypt message
            decrypted_message = security_manager.decrypt_data(encrypted_message)
            command = json.loads(decrypted_message)
            
            # Check if command is for this agent
            if command.get("agent_id") != self.agent_id:
                return
            
            command_type = command.get("command")
            
            if command_type == "execute_task":
                await self._handle_task_command(command)
            elif command_type == "shutdown":
                await self.shutdown()
            elif command_type == "status":
                await self._send_status_update()
            else:
                logger.warning(f"Unknown command type: {command_type}")
                
        except zmq.Again:
            # No message available
            pass
        except Exception as e:
            logger.error(f"Error processing commands: {e}")
    
    async def _handle_task_command(self, command: Dict[str, Any]):
        """Handle task execution command"""
        try:
            task_id = command.get("task_id")
            task_type = command.get("task_type")
            target = command.get("target")
            parameters = command.get("parameters", {})
            
            if task_type not in self.task_handlers:
                logger.error(f"No handler for task type: {task_type}")
                await self._send_task_result(task_id, {
                    "status": "failed",
                    "error": f"No handler for task type: {task_type}"
                })
                return
            
            # Update status
            self.current_task = task_id
            self.status = "working"
            await self._send_status_update()
            
            # Execute task
            logger.info(f"Executing task {task_id} of type {task_type}")
            
            try:
                result = await self.task_handlers[task_type](target, parameters)
                
                # Send success result
                await self._send_task_result(task_id, {
                    "status": "completed",
                    "result": result
                })
                
                self.metrics["tasks_completed"] += 1
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                
                # Send failure result
                await self._send_task_result(task_id, {
                    "status": "failed",
                    "error": str(e)
                })
                
                self.metrics["tasks_failed"] += 1
            
            # Reset status
            self.current_task = None
            self.status = "running"
            await self._send_status_update()
            
        except Exception as e:
            logger.error(f"Error handling task command: {e}")
    
    async def _send_task_result(self, task_id: str, result: Dict[str, Any]):
        """Send task result to coordinator"""
        try:
            message = {
                "type": "task_result",
                "agent_id": self.agent_id,
                "task_id": task_id,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending task result: {e}")
    
    async def send_vulnerability_found(self, vulnerability: Dict[str, Any]):
        """Send vulnerability discovery notification"""
        try:
            message = {
                "type": "vulnerability_found",
                "agent_id": self.agent_id,
                "vulnerability": vulnerability,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending vulnerability notification: {e}")
    
    async def send_intelligence_data(self, intelligence: Dict[str, Any]):
        """Send intelligence data to coordinator"""
        try:
            message = {
                "type": "intelligence_data",
                "agent_id": self.agent_id,
                "data": intelligence,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending intelligence data: {e}")
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def execute_primary_function(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute primary agent function"""
        pass
    
    # Common utility methods
    async def sleep_with_jitter(self, base_delay: float, jitter_factor: float = 0.2):
        """Sleep with random jitter for stealth"""
        import random
        jitter = random.uniform(-jitter_factor, jitter_factor)
        actual_delay = base_delay * (1 + jitter)
        await asyncio.sleep(max(0.1, actual_delay))
    
    def generate_random_user_agent(self) -> str:
        """Generate random user agent string"""
        import random
        user_agents = config.stealth.user_agents
        return random.choice(user_agents)
    
    async def make_stealthy_request(self, url: str, method: str = "GET", **kwargs) -> Any:
        """Make HTTP request with stealth measures"""
        import aiohttp
        import random
        
        # Add stealth headers
        headers = kwargs.get("headers", {})
        headers.update({
            "User-Agent": self.generate_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        })
        
        kwargs["headers"] = headers
        
        # Add random delay
        await self.sleep_with_jitter(random.uniform(0.5, 2.0))
        
        # Make request
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                return await response.text()
    
    def log_activity(self, activity_type: str, details: Dict[str, Any]):
        """Log agent activity"""
        log_entry = {
            "agent_id": self.agent_id,
            "activity_type": activity_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Agent activity: {json.dumps(log_entry)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        self.metrics["uptime"] = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()
        return self.metrics.copy()

class AgentError(Exception):
    """Custom exception for agent errors"""
    pass