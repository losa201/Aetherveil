"""
Distributed Coordination Layer for Aetherveil Sentinel
Implements distributed consensus, leader election, and circuit breaker patterns
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import aioredis
from aioredis.lock import Lock
import hashlib
import random

from config.config import config
from coordinator.models import Agent, Task, AgentType, TaskStatus

logger = logging.getLogger(__name__)

class NodeState(Enum):
    """Node states in the distributed system"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    OBSERVER = "observer"

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class DistributedNode:
    """Distributed node information"""
    id: str
    host: str
    port: int
    state: NodeState
    last_heartbeat: datetime
    term: int
    voted_for: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class ConsensusMessage:
    """Consensus protocol message"""
    type: str
    sender: str
    term: int
    data: Dict[str, Any]
    timestamp: datetime

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise Exception("Circuit breaker HALF_OPEN limit reached")
            self.half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
        self.half_open_calls = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class DistributedLock:
    """Distributed lock implementation using Redis"""
    
    def __init__(self, redis_client: aioredis.Redis, key: str, timeout: int = 30):
        self.redis = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.lock_value = str(uuid.uuid4())
        self._locked = False
    
    async def acquire(self, blocking: bool = True, timeout: Optional[int] = None) -> bool:
        """Acquire distributed lock"""
        try:
            if timeout is None:
                timeout = self.timeout
            
            if blocking:
                end_time = time.time() + timeout
                while time.time() < end_time:
                    if await self._try_acquire():
                        self._locked = True
                        return True
                    await asyncio.sleep(0.1)
                return False
            else:
                result = await self._try_acquire()
                if result:
                    self._locked = True
                return result
        except Exception as e:
            logger.error(f"Failed to acquire lock {self.key}: {e}")
            return False
    
    async def _try_acquire(self) -> bool:
        """Try to acquire lock"""
        result = await self.redis.set(
            self.key, 
            self.lock_value, 
            ex=self.timeout, 
            nx=True
        )
        return result is not None
    
    async def release(self):
        """Release distributed lock"""
        if not self._locked:
            return
        
        try:
            # Use Lua script to ensure atomic release
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            await self.redis.eval(script, 1, self.key, self.lock_value)
            self._locked = False
        except Exception as e:
            logger.error(f"Failed to release lock {self.key}: {e}")
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

class ConsensusProtocol:
    """Raft-like consensus protocol implementation"""
    
    def __init__(self, node_id: str, redis_client: aioredis.Redis):
        self.node_id = node_id
        self.redis = redis_client
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id = None
        self.election_timeout = random.uniform(5, 10)  # seconds
        self.heartbeat_interval = 2  # seconds
        self.last_heartbeat = time.time()
        self.nodes: Dict[str, DistributedNode] = {}
        self.votes_received = set()
        self.running = False
        
    async def start(self):
        """Start consensus protocol"""
        self.running = True
        asyncio.create_task(self._election_timer())
        asyncio.create_task(self._heartbeat_sender())
        asyncio.create_task(self._node_discovery())
        logger.info(f"Consensus protocol started for node {self.node_id}")
    
    async def stop(self):
        """Stop consensus protocol"""
        self.running = False
        logger.info(f"Consensus protocol stopped for node {self.node_id}")
    
    async def _election_timer(self):
        """Handle election timeout"""
        while self.running:
            try:
                await asyncio.sleep(0.5)
                
                if self.state != NodeState.LEADER:
                    if time.time() - self.last_heartbeat > self.election_timeout:
                        await self._start_election()
                
            except Exception as e:
                logger.error(f"Election timer error: {e}")
    
    async def _start_election(self):
        """Start leader election"""
        try:
            logger.info(f"Node {self.node_id} starting election for term {self.current_term + 1}")
            
            # Become candidate
            self.state = NodeState.CANDIDATE
            self.current_term += 1
            self.voted_for = self.node_id
            self.votes_received = {self.node_id}
            self.last_heartbeat = time.time()
            
            # Send vote requests
            await self._send_vote_requests()
            
            # Check if we won
            if len(self.votes_received) > len(self.nodes) // 2:
                await self._become_leader()
            
        except Exception as e:
            logger.error(f"Election failed: {e}")
    
    async def _send_vote_requests(self):
        """Send vote requests to all nodes"""
        try:
            vote_request = ConsensusMessage(
                type="vote_request",
                sender=self.node_id,
                term=self.current_term,
                data={
                    "candidate_id": self.node_id,
                    "last_log_index": len(self.log) - 1,
                    "last_log_term": self.log[-1]["term"] if self.log else 0
                },
                timestamp=datetime.utcnow()
            )
            
            # Broadcast vote request
            await self._broadcast_message(vote_request)
            
        except Exception as e:
            logger.error(f"Failed to send vote requests: {e}")
    
    async def _become_leader(self):
        """Become cluster leader"""
        try:
            logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
            
            self.state = NodeState.LEADER
            self.leader_id = self.node_id
            
            # Send initial heartbeat
            await self._send_heartbeat()
            
            # Register as leader in Redis
            await self.redis.set(
                "cluster:leader",
                json.dumps({
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "timestamp": datetime.utcnow().isoformat()
                }),
                ex=30
            )
            
        except Exception as e:
            logger.error(f"Failed to become leader: {e}")
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeats as leader"""
        while self.running:
            try:
                if self.state == NodeState.LEADER:
                    await self._send_heartbeat()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat sender error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self):
        """Send heartbeat to all nodes"""
        try:
            heartbeat = ConsensusMessage(
                type="heartbeat",
                sender=self.node_id,
                term=self.current_term,
                data={
                    "leader_id": self.node_id,
                    "commit_index": self.commit_index
                },
                timestamp=datetime.utcnow()
            )
            
            await self._broadcast_message(heartbeat)
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    async def _broadcast_message(self, message: ConsensusMessage):
        """Broadcast message to all nodes"""
        try:
            message_data = {
                "type": message.type,
                "sender": message.sender,
                "term": message.term,
                "data": message.data,
                "timestamp": message.timestamp.isoformat()
            }
            
            # Publish to Redis channel
            await self.redis.publish("consensus:messages", json.dumps(message_data))
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
    
    async def _node_discovery(self):
        """Discover other nodes in the cluster"""
        while self.running:
            try:
                # Register this node
                node_data = {
                    "id": self.node_id,
                    "host": "localhost",
                    "port": 8000,
                    "state": self.state.value,
                    "last_heartbeat": datetime.utcnow().isoformat(),
                    "term": self.current_term
                }
                
                await self.redis.set(
                    f"cluster:nodes:{self.node_id}",
                    json.dumps(node_data),
                    ex=60
                )
                
                # Discover other nodes
                keys = await self.redis.keys("cluster:nodes:*")
                discovered_nodes = {}
                
                for key in keys:
                    try:
                        node_data = await self.redis.get(key)
                        if node_data:
                            node_info = json.loads(node_data)
                            node_id = node_info["id"]
                            
                            if node_id != self.node_id:
                                discovered_nodes[node_id] = DistributedNode(
                                    id=node_id,
                                    host=node_info["host"],
                                    port=node_info["port"],
                                    state=NodeState(node_info["state"]),
                                    last_heartbeat=datetime.fromisoformat(node_info["last_heartbeat"]),
                                    term=node_info["term"]
                                )
                    except Exception as e:
                        logger.warning(f"Failed to parse node data: {e}")
                
                self.nodes = discovered_nodes
                
                await asyncio.sleep(30)  # Discovery every 30 seconds
                
            except Exception as e:
                logger.error(f"Node discovery error: {e}")
                await asyncio.sleep(30)
    
    async def handle_message(self, message: ConsensusMessage):
        """Handle incoming consensus message"""
        try:
            # Update term if necessary
            if message.term > self.current_term:
                self.current_term = message.term
                self.voted_for = None
                if self.state != NodeState.FOLLOWER:
                    self.state = NodeState.FOLLOWER
                    self.leader_id = None
            
            if message.type == "vote_request":
                await self._handle_vote_request(message)
            elif message.type == "vote_response":
                await self._handle_vote_response(message)
            elif message.type == "heartbeat":
                await self._handle_heartbeat(message)
            elif message.type == "append_entries":
                await self._handle_append_entries(message)
            
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
    
    async def _handle_vote_request(self, message: ConsensusMessage):
        """Handle vote request"""
        try:
            data = message.data
            
            # Check if we can vote for this candidate
            can_vote = (
                message.term > self.current_term or
                (message.term == self.current_term and 
                 (self.voted_for is None or self.voted_for == data["candidate_id"]))
            )
            
            if can_vote:
                self.voted_for = data["candidate_id"]
                self.last_heartbeat = time.time()
                
                # Send vote response
                response = ConsensusMessage(
                    type="vote_response",
                    sender=self.node_id,
                    term=self.current_term,
                    data={
                        "vote_granted": True,
                        "candidate_id": data["candidate_id"]
                    },
                    timestamp=datetime.utcnow()
                )
                
                await self._broadcast_message(response)
            
        except Exception as e:
            logger.error(f"Failed to handle vote request: {e}")
    
    async def _handle_vote_response(self, message: ConsensusMessage):
        """Handle vote response"""
        try:
            if (self.state == NodeState.CANDIDATE and 
                message.term == self.current_term and
                message.data.get("vote_granted") and
                message.data.get("candidate_id") == self.node_id):
                
                self.votes_received.add(message.sender)
                
                # Check if we have majority
                if len(self.votes_received) > len(self.nodes) // 2:
                    await self._become_leader()
            
        except Exception as e:
            logger.error(f"Failed to handle vote response: {e}")
    
    async def _handle_heartbeat(self, message: ConsensusMessage):
        """Handle heartbeat from leader"""
        try:
            if message.term >= self.current_term:
                self.state = NodeState.FOLLOWER
                self.leader_id = message.sender
                self.last_heartbeat = time.time()
                
                # Update commit index
                self.commit_index = max(self.commit_index, message.data.get("commit_index", 0))
            
        except Exception as e:
            logger.error(f"Failed to handle heartbeat: {e}")
    
    async def _handle_append_entries(self, message: ConsensusMessage):
        """Handle append entries request"""
        try:
            # Implementation for log replication
            # This is a simplified version
            pass
            
        except Exception as e:
            logger.error(f"Failed to handle append entries: {e}")
    
    def is_leader(self) -> bool:
        """Check if this node is the leader"""
        return self.state == NodeState.LEADER
    
    def get_leader_id(self) -> Optional[str]:
        """Get current leader ID"""
        return self.leader_id

class DistributedCoordinator:
    """Distributed coordinator with consensus and fault tolerance"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.redis: Optional[aioredis.Redis] = None
        self.consensus: Optional[ConsensusProtocol] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.distributed_locks: Dict[str, DistributedLock] = {}
        self.running = False
        
    async def initialize(self):
        """Initialize distributed coordinator"""
        try:
            # Connect to Redis
            self.redis = aioredis.from_url(
                config.database.redis_url,
                decode_responses=True
            )
            
            # Initialize consensus protocol
            self.consensus = ConsensusProtocol(self.node_id, self.redis)
            await self.consensus.start()
            
            # Start message handler
            asyncio.create_task(self._message_handler())
            
            self.running = True
            logger.info(f"Distributed coordinator initialized for node {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed coordinator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown distributed coordinator"""
        try:
            self.running = False
            
            if self.consensus:
                await self.consensus.stop()
            
            if self.redis:
                await self.redis.close()
            
            logger.info(f"Distributed coordinator shutdown for node {self.node_id}")
            
        except Exception as e:
            logger.error(f"Error shutting down distributed coordinator: {e}")
    
    async def _message_handler(self):
        """Handle consensus messages"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe("consensus:messages")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        consensus_msg = ConsensusMessage(
                            type=data['type'],
                            sender=data['sender'],
                            term=data['term'],
                            data=data['data'],
                            timestamp=datetime.fromisoformat(data['timestamp'])
                        )
                        
                        if consensus_msg.sender != self.node_id:
                            await self.consensus.handle_message(consensus_msg)
                            
                    except Exception as e:
                        logger.error(f"Failed to handle consensus message: {e}")
                        
        except Exception as e:
            logger.error(f"Message handler error: {e}")
    
    def get_circuit_breaker(self, key: str) -> CircuitBreaker:
        """Get or create circuit breaker for key"""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = CircuitBreaker()
        return self.circuit_breakers[key]
    
    def get_distributed_lock(self, key: str, timeout: int = 30) -> DistributedLock:
        """Get distributed lock for key"""
        return DistributedLock(self.redis, key, timeout)
    
    async def execute_with_circuit_breaker(self, key: str, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        circuit_breaker = self.get_circuit_breaker(key)
        return await circuit_breaker.call(func, *args, **kwargs)
    
    async def set_distributed_data(self, key: str, value: Any, ttl: int = 3600):
        """Set data in distributed storage"""
        try:
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            await self.redis.set(key, serialized_value, ex=ttl)
        except Exception as e:
            logger.error(f"Failed to set distributed data: {e}")
            raise
    
    async def get_distributed_data(self, key: str) -> Optional[Any]:
        """Get data from distributed storage"""
        try:
            value = await self.redis.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Failed to get distributed data: {e}")
            return None
    
    async def delete_distributed_data(self, key: str):
        """Delete data from distributed storage"""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete distributed data: {e}")
            raise
    
    async def increment_counter(self, key: str, increment: int = 1) -> int:
        """Increment distributed counter"""
        try:
            return await self.redis.incr(key, increment)
        except Exception as e:
            logger.error(f"Failed to increment counter: {e}")
            raise
    
    async def set_counter(self, key: str, value: int, ttl: int = 3600):
        """Set distributed counter"""
        try:
            await self.redis.set(key, value, ex=ttl)
        except Exception as e:
            logger.error(f"Failed to set counter: {e}")
            raise
    
    async def get_counter(self, key: str) -> int:
        """Get distributed counter value"""
        try:
            value = await self.redis.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Failed to get counter: {e}")
            return 0
    
    async def add_to_set(self, key: str, *values, ttl: int = 3600):
        """Add values to distributed set"""
        try:
            await self.redis.sadd(key, *values)
            await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"Failed to add to set: {e}")
            raise
    
    async def remove_from_set(self, key: str, *values):
        """Remove values from distributed set"""
        try:
            await self.redis.srem(key, *values)
        except Exception as e:
            logger.error(f"Failed to remove from set: {e}")
            raise
    
    async def get_set_members(self, key: str) -> Set[str]:
        """Get members of distributed set"""
        try:
            members = await self.redis.smembers(key)
            return set(members)
        except Exception as e:
            logger.error(f"Failed to get set members: {e}")
            return set()
    
    async def is_member_of_set(self, key: str, value: str) -> bool:
        """Check if value is member of distributed set"""
        try:
            return await self.redis.sismember(key, value)
        except Exception as e:
            logger.error(f"Failed to check set membership: {e}")
            return False
    
    async def acquire_leader_lock(self, timeout: int = 30) -> bool:
        """Acquire leader lock"""
        try:
            if self.consensus and self.consensus.is_leader():
                lock = self.get_distributed_lock("leader_operations", timeout)
                return await lock.acquire(blocking=False)
            return False
        except Exception as e:
            logger.error(f"Failed to acquire leader lock: {e}")
            return False
    
    def is_leader(self) -> bool:
        """Check if this node is the leader"""
        return self.consensus and self.consensus.is_leader()
    
    def get_leader_id(self) -> Optional[str]:
        """Get current leader ID"""
        return self.consensus.get_leader_id() if self.consensus else None
    
    async def wait_for_leader(self, timeout: int = 30) -> bool:
        """Wait for leader election to complete"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.consensus and self.consensus.get_leader_id():
                    return True
                await asyncio.sleep(0.5)
            return False
        except Exception as e:
            logger.error(f"Failed to wait for leader: {e}")
            return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information"""
        try:
            cluster_info = {
                "node_id": self.node_id,
                "is_leader": self.is_leader(),
                "leader_id": self.get_leader_id(),
                "cluster_size": len(self.consensus.nodes) + 1 if self.consensus else 1,
                "nodes": []
            }
            
            if self.consensus:
                cluster_info["term"] = self.consensus.current_term
                cluster_info["state"] = self.consensus.state.value
                
                for node_id, node in self.consensus.nodes.items():
                    cluster_info["nodes"].append({
                        "id": node_id,
                        "host": node.host,
                        "port": node.port,
                        "state": node.state.value,
                        "last_heartbeat": node.last_heartbeat.isoformat(),
                        "term": node.term
                    })
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {"error": str(e)}