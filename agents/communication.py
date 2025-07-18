"""
Advanced ZMQ Communication Layer for Aetherveil Sentinel Agents
Handles encrypted communication, message routing, and fault tolerance
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import zmq
import zmq.asyncio
from cryptography.fernet import Fernet
from dataclasses import dataclass, asdict

from config.config import config
from coordinator.security import security_manager

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Standard message format for agent communication"""
    id: str
    type: str
    sender: str
    recipient: str
    payload: Dict[str, Any]
    timestamp: str
    ttl: int = 300  # Time to live in seconds
    priority: int = 5  # Priority level (1-10)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(**data)

class MessageBroker:
    """Advanced message broker for agent communication"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.context = zmq.asyncio.Context()
        self.sockets: Dict[str, zmq.asyncio.Socket] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_messages: Dict[str, Message] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.encryption_key = None
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "connection_errors": 0
        }
        
    async def initialize(self):
        """Initialize message broker"""
        try:
            # Setup encryption
            self.encryption_key = security_manager.get_encryption_key()
            
            # Setup coordinator connection
            await self._setup_coordinator_connection()
            
            # Setup peer-to-peer communication
            await self._setup_p2p_communication()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._heartbeat_sender())
            asyncio.create_task(self._message_cleanup())
            
            logger.info(f"Message broker initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize message broker: {e}")
            raise

    async def _setup_coordinator_connection(self):
        """Setup connection to coordinator"""
        try:
            # Outbound messages to coordinator
            self.sockets['coordinator_out'] = self.context.socket(zmq.PUSH)
            self.sockets['coordinator_out'].connect(f"tcp://coordinator:{config.network.zmq_port}")
            
            # Inbound messages from coordinator
            self.sockets['coordinator_in'] = self.context.socket(zmq.PULL)
            self.sockets['coordinator_in'].connect(f"tcp://coordinator:{config.network.zmq_port + 1}")
            
            # Setup high water mark for flow control
            for socket in self.sockets.values():
                socket.set_hwm(config.network.zmq_hwm)
                socket.set(zmq.LINGER, config.network.zmq_linger)
            
        except Exception as e:
            logger.error(f"Failed to setup coordinator connection: {e}")
            raise

    async def _setup_p2p_communication(self):
        """Setup peer-to-peer communication between agents"""
        try:
            # Agent-to-agent communication
            self.sockets['peer_out'] = self.context.socket(zmq.PUSH)
            self.sockets['peer_in'] = self.context.socket(zmq.PULL)
            
            # Bind to dynamic port for incoming messages
            port = self.sockets['peer_in'].bind_to_random_port("tcp://*")
            
            # Register our endpoint with coordinator
            await self._register_endpoint(port)
            
        except Exception as e:
            logger.error(f"Failed to setup P2P communication: {e}")
            raise

    async def _register_endpoint(self, port: int):
        """Register agent endpoint with coordinator"""
        try:
            registration_msg = Message(
                id=str(uuid.uuid4()),
                type="agent_registration",
                sender=self.agent_id,
                recipient="coordinator",
                payload={
                    "agent_id": self.agent_id,
                    "endpoint": f"tcp://localhost:{port}",
                    "capabilities": []
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
            await self.send_message(registration_msg)
            
        except Exception as e:
            logger.error(f"Failed to register endpoint: {e}")

    async def send_message(self, message: Message) -> bool:
        """Send encrypted message"""
        try:
            # Encrypt message
            encrypted_data = self._encrypt_message(message.to_dict())
            
            # Determine target socket
            if message.recipient == "coordinator":
                socket = self.sockets['coordinator_out']
            else:
                socket = self.sockets['peer_out']
            
            # Send message
            await socket.send_string(encrypted_data)
            
            # Update stats
            self.stats["messages_sent"] += 1
            
            # Store for acknowledgment tracking
            self.pending_messages[message.id] = message
            
            logger.debug(f"Sent message {message.id} to {message.recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.stats["messages_failed"] += 1
            return False

    async def receive_message(self) -> Optional[Message]:
        """Receive and decrypt message"""
        try:
            # Check both coordinator and peer sockets
            for socket_name, socket in [('coordinator_in', self.sockets['coordinator_in']), 
                                       ('peer_in', self.sockets['peer_in'])]:
                try:
                    encrypted_data = await socket.recv_string(zmq.NOBLOCK)
                    
                    # Decrypt message
                    message_dict = self._decrypt_message(encrypted_data)
                    message = Message.from_dict(message_dict)
                    
                    # Check if message is for us
                    if message.recipient == self.agent_id or message.recipient == "broadcast":
                        self.stats["messages_received"] += 1
                        logger.debug(f"Received message {message.id} from {message.sender}")
                        return message
                    
                except zmq.Again:
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None

    async def _message_processor(self):
        """Process incoming messages"""
        while self.running:
            try:
                message = await self.receive_message()
                if message:
                    await self._handle_message(message)
                    
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(1)

    async def _handle_message(self, message: Message):
        """Handle incoming message"""
        try:
            # Check message TTL
            msg_time = datetime.fromisoformat(message.timestamp)
            if (datetime.utcnow() - msg_time).total_seconds() > message.ttl:
                logger.warning(f"Message {message.id} expired")
                return
            
            # Route message to handler
            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(message)
            else:
                logger.warning(f"No handler for message type: {message.type}")
                
        except Exception as e:
            logger.error(f"Error handling message {message.id}: {e}")

    async def _heartbeat_sender(self):
        """Send periodic heartbeat messages"""
        while self.running:
            try:
                heartbeat_msg = Message(
                    id=str(uuid.uuid4()),
                    type="heartbeat",
                    sender=self.agent_id,
                    recipient="coordinator",
                    payload={
                        "agent_id": self.agent_id,
                        "status": "running",
                        "stats": self.stats
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
                await self.send_message(heartbeat_msg)
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)

    async def _message_cleanup(self):
        """Clean up expired messages"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                expired_messages = []
                
                for msg_id, message in self.pending_messages.items():
                    msg_time = datetime.fromisoformat(message.timestamp)
                    if (current_time - msg_time).total_seconds() > message.ttl:
                        expired_messages.append(msg_id)
                
                for msg_id in expired_messages:
                    del self.pending_messages[msg_id]
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in message cleanup: {e}")
                await asyncio.sleep(60)

    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler

    def _encrypt_message(self, message_dict: Dict[str, Any]) -> str:
        """Encrypt message using Fernet encryption"""
        try:
            fernet = Fernet(self.encryption_key)
            message_json = json.dumps(message_dict)
            encrypted_bytes = fernet.encrypt(message_json.encode())
            return encrypted_bytes.decode()
            
        except Exception as e:
            logger.error(f"Failed to encrypt message: {e}")
            raise

    def _decrypt_message(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt message using Fernet encryption"""
        try:
            fernet = Fernet(self.encryption_key)
            decrypted_bytes = fernet.decrypt(encrypted_data.encode())
            message_json = decrypted_bytes.decode()
            return json.loads(message_json)
            
        except Exception as e:
            logger.error(f"Failed to decrypt message: {e}")
            raise

    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]):
        """Broadcast message to all agents"""
        try:
            broadcast_msg = Message(
                id=str(uuid.uuid4()),
                type=message_type,
                sender=self.agent_id,
                recipient="broadcast",
                payload=payload,
                timestamp=datetime.utcnow().isoformat()
            )
            
            await self.send_message(broadcast_msg)
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")

    async def request_response(self, recipient: str, message_type: str, 
                              payload: Dict[str, Any], timeout: int = 30) -> Optional[Message]:
        """Send request and wait for response"""
        try:
            request_id = str(uuid.uuid4())
            
            request_msg = Message(
                id=request_id,
                type=message_type,
                sender=self.agent_id,
                recipient=recipient,
                payload=payload,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Send request
            await self.send_message(request_msg)
            
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < timeout:
                message = await self.receive_message()
                if message and message.payload.get("response_to") == request_id:
                    return message
                await asyncio.sleep(0.1)
            
            logger.warning(f"Request {request_id} timed out")
            return None
            
        except Exception as e:
            logger.error(f"Request-response failed: {e}")
            return None

    async def shutdown(self):
        """Shutdown message broker"""
        try:
            self.running = False
            
            # Close all sockets
            for socket in self.sockets.values():
                socket.close()
            
            # Terminate context
            self.context.term()
            
            logger.info(f"Message broker shutdown for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error shutting down message broker: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            "agent_id": self.agent_id,
            "stats": self.stats,
            "pending_messages": len(self.pending_messages),
            "handlers_registered": len(self.message_handlers),
            "sockets_active": len(self.sockets)
        }

class AgentCommunicator:
    """High-level communication interface for agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.broker = MessageBroker(agent_id)
        self.response_handlers: Dict[str, Callable] = {}
        
    async def initialize(self):
        """Initialize communicator"""
        await self.broker.initialize()
        
        # Register default handlers
        self.broker.register_handler("response", self._handle_response)
        self.broker.register_handler("command", self._handle_command)
        self.broker.register_handler("broadcast", self._handle_broadcast)

    async def _handle_response(self, message: Message):
        """Handle response message"""
        response_to = message.payload.get("response_to")
        if response_to and response_to in self.response_handlers:
            await self.response_handlers[response_to](message)

    async def _handle_command(self, message: Message):
        """Handle command message"""
        # This should be implemented by the specific agent
        pass

    async def _handle_broadcast(self, message: Message):
        """Handle broadcast message"""
        # This should be implemented by the specific agent
        pass

    async def send_task_result(self, task_id: str, result: Dict[str, Any]):
        """Send task result to coordinator"""
        await self.broker.send_message(Message(
            id=str(uuid.uuid4()),
            type="task_result",
            sender=self.agent_id,
            recipient="coordinator",
            payload={
                "task_id": task_id,
                "result": result,
                "agent_id": self.agent_id
            },
            timestamp=datetime.utcnow().isoformat()
        ))

    async def send_vulnerability_alert(self, vulnerability: Dict[str, Any]):
        """Send vulnerability alert"""
        await self.broker.broadcast_message("vulnerability_alert", vulnerability)

    async def request_agent_collaboration(self, target_agent: str, 
                                         task_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Request collaboration from another agent"""
        response = await self.broker.request_response(
            target_agent, 
            "collaboration_request", 
            {
                "task_type": task_type,
                "data": data
            }
        )
        
        return response.payload if response else None

    async def shutdown(self):
        """Shutdown communicator"""
        await self.broker.shutdown()

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return self.broker.get_stats()