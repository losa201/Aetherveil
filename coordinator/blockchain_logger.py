"""
Tamper-Evident Blockchain-Style Logging System for Aetherveil Sentinel
Implements immutable audit logging with cryptographic verification and integrity checking
"""

import asyncio
import json
import logging
import hashlib
import hmac
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import struct
import pickle
import gzip
import base64
import threading
from collections import deque
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.fernet import Fernet

from coordinator.security_manager import SecurityLevel, ThreatLevel

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    AUDIT = "audit"

class EventType(Enum):
    """Event types for blockchain logging"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_INCIDENT = "security_incident"
    SYSTEM_EVENT = "system_event"
    AGENT_EVENT = "agent_event"
    TASK_EVENT = "task_event"
    NETWORK_EVENT = "network_event"
    ERROR_EVENT = "error_event"

@dataclass
class LogEntry:
    """Individual log entry"""
    entry_id: str
    timestamp: datetime
    level: LogLevel
    event_type: EventType
    source: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    entity_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'event_type': self.event_type.value,
            'source': self.source,
            'message': self.message,
            'data': self.data,
            'security_level': self.security_level.value,
            'entity_id': self.entity_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent
        }

@dataclass
class BlockHeader:
    """Blockchain block header"""
    block_id: str
    block_number: int
    timestamp: datetime
    previous_hash: str
    merkle_root: str
    nonce: int
    difficulty: int
    validator: str
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'block_id': self.block_id,
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'validator': self.validator,
            'signature': self.signature
        }

@dataclass
class LogBlock:
    """Blockchain block containing log entries"""
    header: BlockHeader
    entries: List[LogEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'header': self.header.to_dict(),
            'entries': [entry.to_dict() for entry in self.entries],
            'metadata': self.metadata
        }
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(block_data.encode()).hexdigest()

class MerkleTree:
    """Merkle tree for tamper-evident logging"""
    
    def __init__(self, data: List[str]):
        self.data = data
        self.tree = self._build_tree(data)
    
    def _build_tree(self, data: List[str]) -> List[List[str]]:
        """Build Merkle tree"""
        if not data:
            return []
        
        # Hash all data items
        level = [hashlib.sha256(item.encode()).hexdigest() for item in data]
        tree = [level]
        
        # Build tree bottom-up
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else level[i]
                combined = left + right
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            level = next_level
            tree.append(level)
        
        return tree
    
    def get_root(self) -> str:
        """Get Merkle root"""
        return self.tree[-1][0] if self.tree else ""
    
    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """Get Merkle proof for data at index"""
        if index < 0 or index >= len(self.data):
            return []
        
        proof = []
        current_index = index
        
        for level in self.tree[:-1]:
            if current_index % 2 == 0:
                # Current node is left child
                sibling_index = current_index + 1
                if sibling_index < len(level):
                    proof.append((level[sibling_index], 'right'))
            else:
                # Current node is right child
                sibling_index = current_index - 1
                proof.append((level[sibling_index], 'left'))
            
            current_index //= 2
        
        return proof
    
    def verify_proof(self, data: str, index: int, proof: List[Tuple[str, str]]) -> bool:
        """Verify Merkle proof"""
        current_hash = hashlib.sha256(data.encode()).hexdigest()
        
        for sibling_hash, direction in proof:
            if direction == 'right':
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return current_hash == self.get_root()

class BlockchainLogger:
    """Tamper-evident blockchain-style logger"""
    
    def __init__(self, storage_path: str = "/app/blockchain_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.blocks: List[LogBlock] = []
        self.pending_entries: deque = deque()
        self.current_block_number = 0
        self.block_size = 100  # Maximum entries per block
        self.difficulty = 4  # Proof of work difficulty
        
        # Cryptographic keys
        self.private_key: Optional[rsa.RSAPrivateKey] = None
        self.public_key: Optional[rsa.RSAPublicKey] = None
        self.encryption_key: Optional[bytes] = None
        
        # Threading
        self._lock = threading.Lock()
        self._mining_lock = threading.Lock()
        self._shutdown = False
        
        # Statistics
        self.statistics = {
            'total_blocks': 0,
            'total_entries': 0,
            'pending_entries': 0,
            'mining_time': 0,
            'verification_time': 0,
            'last_block_time': None
        }
        
        # Background tasks
        self._mining_task: Optional[asyncio.Task] = None
        self._persistence_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize blockchain logger"""
        try:
            # Generate cryptographic keys
            await self._generate_keys()
            
            # Load existing blockchain
            await self._load_blockchain()
            
            # Start background tasks
            self._mining_task = asyncio.create_task(self._mining_loop())
            self._persistence_task = asyncio.create_task(self._persistence_loop())
            
            logger.info("Blockchain logger initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain logger: {e}")
            raise
    
    async def _generate_keys(self):
        """Generate cryptographic keys"""
        try:
            # Generate RSA key pair for signing
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
            # Generate encryption key
            self.encryption_key = Fernet.generate_key()
            
            # Save keys
            keys_dir = self.storage_path / "keys"
            keys_dir.mkdir(exist_ok=True)
            
            # Save private key
            with open(keys_dir / "private.pem", "wb") as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            with open(keys_dir / "public.pem", "wb") as f:
                f.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            # Save encryption key
            with open(keys_dir / "encryption.key", "wb") as f:
                f.write(self.encryption_key)
            
            logger.info("Cryptographic keys generated")
            
        except Exception as e:
            logger.error(f"Failed to generate keys: {e}")
            raise
    
    async def _load_blockchain(self):
        """Load existing blockchain from storage"""
        try:
            blockchain_file = self.storage_path / "blockchain.json"
            
            if blockchain_file.exists():
                with open(blockchain_file, "r") as f:
                    data = json.load(f)
                
                # Load blocks
                for block_data in data.get("blocks", []):
                    # Reconstruct block header
                    header_data = block_data["header"]
                    header = BlockHeader(
                        block_id=header_data["block_id"],
                        block_number=header_data["block_number"],
                        timestamp=datetime.fromisoformat(header_data["timestamp"]),
                        previous_hash=header_data["previous_hash"],
                        merkle_root=header_data["merkle_root"],
                        nonce=header_data["nonce"],
                        difficulty=header_data["difficulty"],
                        validator=header_data["validator"],
                        signature=header_data["signature"]
                    )
                    
                    # Reconstruct log entries
                    entries = []
                    for entry_data in block_data["entries"]:
                        entry = LogEntry(
                            entry_id=entry_data["entry_id"],
                            timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                            level=LogLevel(entry_data["level"]),
                            event_type=EventType(entry_data["event_type"]),
                            source=entry_data["source"],
                            message=entry_data["message"],
                            data=entry_data["data"],
                            security_level=SecurityLevel(entry_data["security_level"]),
                            entity_id=entry_data.get("entity_id"),
                            session_id=entry_data.get("session_id"),
                            ip_address=entry_data.get("ip_address"),
                            user_agent=entry_data.get("user_agent")
                        )
                        entries.append(entry)
                    
                    # Create block
                    block = LogBlock(
                        header=header,
                        entries=entries,
                        metadata=block_data.get("metadata", {})
                    )
                    
                    self.blocks.append(block)
                
                # Update statistics
                self.current_block_number = len(self.blocks)
                self.statistics['total_blocks'] = len(self.blocks)
                self.statistics['total_entries'] = sum(len(block.entries) for block in self.blocks)
                
                logger.info(f"Loaded {len(self.blocks)} blocks from storage")
            
            # Verify blockchain integrity
            await self._verify_blockchain()
            
        except Exception as e:
            logger.error(f"Failed to load blockchain: {e}")
            raise
    
    async def _verify_blockchain(self):
        """Verify blockchain integrity"""
        try:
            verification_start = time.time()
            
            for i, block in enumerate(self.blocks):
                # Verify block hash
                calculated_hash = block.calculate_hash()
                
                # Verify previous hash (except for genesis block)
                if i > 0:
                    previous_block = self.blocks[i - 1]
                    if block.header.previous_hash != previous_block.calculate_hash():
                        raise ValueError(f"Invalid previous hash in block {i}")
                
                # Verify Merkle root
                entry_strings = [json.dumps(entry.to_dict(), sort_keys=True) for entry in block.entries]
                merkle_tree = MerkleTree(entry_strings)
                if block.header.merkle_root != merkle_tree.get_root():
                    raise ValueError(f"Invalid Merkle root in block {i}")
                
                # Verify signature
                if not self._verify_signature(block.header.to_dict(), block.header.signature):
                    raise ValueError(f"Invalid signature in block {i}")
            
            verification_time = time.time() - verification_start
            self.statistics['verification_time'] = verification_time
            
            logger.info(f"Blockchain verification completed in {verification_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Blockchain verification failed: {e}")
            raise
    
    def log_entry(self, level: LogLevel, event_type: EventType, source: str, message: str,
                  data: Dict[str, Any] = None, security_level: SecurityLevel = SecurityLevel.MEDIUM,
                  entity_id: str = None, session_id: str = None, ip_address: str = None,
                  user_agent: str = None):
        """Add log entry to blockchain"""
        try:
            entry = LogEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                level=level,
                event_type=event_type,
                source=source,
                message=message,
                data=data or {},
                security_level=security_level,
                entity_id=entity_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            with self._lock:
                self.pending_entries.append(entry)
                self.statistics['pending_entries'] = len(self.pending_entries)
            
            logger.debug(f"Added log entry: {entry.entry_id}")
            
        except Exception as e:
            logger.error(f"Failed to add log entry: {e}")
    
    def log_security_event(self, event_type: EventType, source: str, message: str,
                          threat_level: ThreatLevel = ThreatLevel.MEDIUM,
                          data: Dict[str, Any] = None, entity_id: str = None,
                          session_id: str = None, ip_address: str = None):
        """Log security event"""
        self.log_entry(
            level=LogLevel.SECURITY,
            event_type=event_type,
            source=source,
            message=message,
            data={**(data or {}), 'threat_level': threat_level.value},
            security_level=SecurityLevel.HIGH,
            entity_id=entity_id,
            session_id=session_id,
            ip_address=ip_address
        )
    
    def log_audit_event(self, event_type: EventType, source: str, message: str,
                       data: Dict[str, Any] = None, entity_id: str = None,
                       session_id: str = None, ip_address: str = None):
        """Log audit event"""
        self.log_entry(
            level=LogLevel.AUDIT,
            event_type=event_type,
            source=source,
            message=message,
            data=data,
            security_level=SecurityLevel.HIGH,
            entity_id=entity_id,
            session_id=session_id,
            ip_address=ip_address
        )
    
    async def _mining_loop(self):
        """Background mining loop"""
        while not self._shutdown:
            try:
                # Check if we have enough pending entries
                if len(self.pending_entries) >= self.block_size:
                    await self._mine_block()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in mining loop: {e}")
                await asyncio.sleep(5)
    
    async def _mine_block(self):
        """Mine a new block"""
        try:
            mining_start = time.time()
            
            with self._mining_lock:
                if len(self.pending_entries) == 0:
                    return
                
                # Collect entries for new block
                entries = []
                for _ in range(min(self.block_size, len(self.pending_entries))):
                    entries.append(self.pending_entries.popleft())
                
                # Create block header
                previous_hash = self.blocks[-1].calculate_hash() if self.blocks else "0" * 64
                
                # Calculate Merkle root
                entry_strings = [json.dumps(entry.to_dict(), sort_keys=True) for entry in entries]
                merkle_tree = MerkleTree(entry_strings)
                merkle_root = merkle_tree.get_root()
                
                # Create block header
                header = BlockHeader(
                    block_id=str(uuid.uuid4()),
                    block_number=self.current_block_number,
                    timestamp=datetime.utcnow(),
                    previous_hash=previous_hash,
                    merkle_root=merkle_root,
                    nonce=0,
                    difficulty=self.difficulty,
                    validator="aetherveil-sentinel",
                    signature=""
                )
                
                # Proof of work
                block = LogBlock(header=header, entries=entries)
                await self._proof_of_work(block)
                
                # Sign block
                signature = self._sign_block(block.header.to_dict())
                block.header.signature = signature
                
                # Add to blockchain
                self.blocks.append(block)
                self.current_block_number += 1
                
                # Update statistics
                mining_time = time.time() - mining_start
                self.statistics['total_blocks'] = len(self.blocks)
                self.statistics['total_entries'] += len(entries)
                self.statistics['pending_entries'] = len(self.pending_entries)
                self.statistics['mining_time'] = mining_time
                self.statistics['last_block_time'] = datetime.utcnow()
                
                logger.info(f"Mined block {self.current_block_number - 1} with {len(entries)} entries in {mining_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Failed to mine block: {e}")
    
    async def _proof_of_work(self, block: LogBlock):
        """Perform proof of work"""
        try:
            target = "0" * self.difficulty
            
            while True:
                block_hash = block.calculate_hash()
                
                if block_hash.startswith(target):
                    break
                
                block.header.nonce += 1
                
                # Yield control periodically
                if block.header.nonce % 1000 == 0:
                    await asyncio.sleep(0.001)
            
        except Exception as e:
            logger.error(f"Proof of work failed: {e}")
            raise
    
    def _sign_block(self, block_data: Dict[str, Any]) -> str:
        """Sign block with private key"""
        try:
            block_bytes = json.dumps(block_data, sort_keys=True).encode()
            signature = self.private_key.sign(
                block_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            logger.error(f"Failed to sign block: {e}")
            return ""
    
    def _verify_signature(self, block_data: Dict[str, Any], signature: str) -> bool:
        """Verify block signature"""
        try:
            block_bytes = json.dumps(block_data, sort_keys=True).encode()
            signature_bytes = base64.b64decode(signature)
            
            self.public_key.verify(
                signature_bytes,
                block_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
            
        except Exception:
            return False
    
    async def _persistence_loop(self):
        """Background persistence loop"""
        while not self._shutdown:
            try:
                await self._persist_blockchain()
                await asyncio.sleep(60)  # Persist every minute
                
            except Exception as e:
                logger.error(f"Error in persistence loop: {e}")
                await asyncio.sleep(30)
    
    async def _persist_blockchain(self):
        """Persist blockchain to storage"""
        try:
            blockchain_file = self.storage_path / "blockchain.json"
            temp_file = blockchain_file.with_suffix(".tmp")
            
            # Create backup
            if blockchain_file.exists():
                backup_file = blockchain_file.with_suffix(".bak")
                blockchain_file.rename(backup_file)
            
            # Save blockchain
            data = {
                "blocks": [block.to_dict() for block in self.blocks],
                "metadata": {
                    "current_block_number": self.current_block_number,
                    "total_blocks": len(self.blocks),
                    "total_entries": sum(len(block.entries) for block in self.blocks),
                    "last_updated": datetime.utcnow().isoformat()
                }
            }
            
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(blockchain_file)
            
            logger.debug("Blockchain persisted to storage")
            
        except Exception as e:
            logger.error(f"Failed to persist blockchain: {e}")
    
    def query_logs(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[LogEntry]:
        """Query logs from blockchain"""
        try:
            results = []
            
            for block in reversed(self.blocks):  # Start from newest
                for entry in reversed(block.entries):
                    if self._matches_filter(entry, filters):
                        results.append(entry)
                        if len(results) >= limit:
                            return results
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query logs: {e}")
            return []
    
    def _matches_filter(self, entry: LogEntry, filters: Dict[str, Any]) -> bool:
        """Check if entry matches filter criteria"""
        try:
            if not filters:
                return True
            
            for key, value in filters.items():
                if key == "level" and entry.level != LogLevel(value):
                    return False
                elif key == "event_type" and entry.event_type != EventType(value):
                    return False
                elif key == "source" and entry.source != value:
                    return False
                elif key == "entity_id" and entry.entity_id != value:
                    return False
                elif key == "session_id" and entry.session_id != value:
                    return False
                elif key == "start_time" and entry.timestamp < datetime.fromisoformat(value):
                    return False
                elif key == "end_time" and entry.timestamp > datetime.fromisoformat(value):
                    return False
                elif key == "message_contains" and value.lower() not in entry.message.lower():
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to match filter: {e}")
            return False
    
    def get_block_info(self, block_number: int) -> Optional[Dict[str, Any]]:
        """Get block information"""
        try:
            if 0 <= block_number < len(self.blocks):
                block = self.blocks[block_number]
                return {
                    'block_number': block_number,
                    'block_id': block.header.block_id,
                    'timestamp': block.header.timestamp,
                    'entries_count': len(block.entries),
                    'hash': block.calculate_hash(),
                    'previous_hash': block.header.previous_hash,
                    'merkle_root': block.header.merkle_root,
                    'nonce': block.header.nonce,
                    'difficulty': block.header.difficulty,
                    'validator': block.header.validator
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get block info: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        return {
            **self.statistics,
            'blockchain_height': len(self.blocks),
            'storage_path': str(self.storage_path),
            'block_size': self.block_size,
            'difficulty': self.difficulty
        }
    
    async def shutdown(self):
        """Shutdown blockchain logger"""
        try:
            self._shutdown = True
            
            # Cancel tasks
            if self._mining_task:
                self._mining_task.cancel()
            if self._persistence_task:
                self._persistence_task.cancel()
            
            # Mine remaining entries
            if self.pending_entries:
                await self._mine_block()
            
            # Final persistence
            await self._persist_blockchain()
            
            logger.info("Blockchain logger shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown blockchain logger: {e}")

# Global blockchain logger instance
blockchain_logger = BlockchainLogger()