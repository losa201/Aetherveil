"""
ZeroMQ Encryption Manager for Secure Communication Channels
Implements AES-256-GCM encryption for ZeroMQ messages with key rotation and authentication
"""

import asyncio
import json
import logging
import os
import struct
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import base64
import hashlib
import hmac

import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.fernet import Fernet

from coordinator.security_manager import EncryptionAlgorithm, SecurityLevel

logger = logging.getLogger(__name__)

@dataclass
class ZMQEncryptionConfig:
    """ZeroMQ encryption configuration"""
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_rotation_interval: int = 3600  # 1 hour
    max_message_size: int = 1048576  # 1MB
    enable_compression: bool = True
    security_level: SecurityLevel = SecurityLevel.HIGH

@dataclass
class EncryptedMessage:
    """Encrypted ZeroMQ message"""
    message_id: str
    timestamp: float
    sender_id: str
    recipient_id: str
    key_id: str
    algorithm: str
    iv: bytes
    ciphertext: bytes
    mac: bytes
    metadata: Dict[str, Any]

class ZMQEncryptionManager:
    """ZeroMQ encryption manager for secure communication"""
    
    def __init__(self, config: ZMQEncryptionConfig = None):
        self.config = config or ZMQEncryptionConfig()
        self.keys: Dict[str, bytes] = {}
        self.active_key_id: Optional[str] = None
        self.context = zmq.Context()
        self.authenticator: Optional[ThreadAuthenticator] = None
        self.encryption_stats = {
            'messages_encrypted': 0,
            'messages_decrypted': 0,
            'key_rotations': 0,
            'errors': 0
        }
        self._lock = threading.Lock()
        
    def initialize(self, enable_curve: bool = True):
        """Initialize ZMQ encryption manager"""
        try:
            # Generate initial encryption key
            self.generate_key()
            
            # Setup CURVE authentication if enabled
            if enable_curve:
                self._setup_curve_auth()
            
            logger.info("ZMQ encryption manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ encryption manager: {e}")
            raise
    
    def _setup_curve_auth(self):
        """Setup CURVE authentication for ZMQ"""
        try:
            # Start authenticator
            self.authenticator = ThreadAuthenticator(self.context)
            self.authenticator.start()
            
            # Configure CURVE authentication
            self.authenticator.configure_curve(domain='*', location=zmq.auth.CURVE_ALLOW_ANY)
            
            logger.info("CURVE authentication configured")
            
        except Exception as e:
            logger.error(f"Failed to setup CURVE authentication: {e}")
            raise
    
    def generate_key(self, key_id: str = None) -> str:
        """Generate new encryption key"""
        try:
            with self._lock:
                if key_id is None:
                    key_id = str(uuid.uuid4())
                
                # Generate 256-bit key for AES-256
                key_data = os.urandom(32)
                self.keys[key_id] = key_data
                
                if self.active_key_id is None:
                    self.active_key_id = key_id
                
                logger.info(f"Generated encryption key: {key_id}")
                return key_id
                
        except Exception as e:
            logger.error(f"Failed to generate encryption key: {e}")
            raise
    
    def rotate_key(self) -> str:
        """Rotate encryption key"""
        try:
            with self._lock:
                old_key_id = self.active_key_id
                new_key_id = self.generate_key()
                
                # Set new key as active
                self.active_key_id = new_key_id
                
                # Keep old key for decryption of pending messages
                # Remove old keys after safe period
                if len(self.keys) > 5:  # Keep last 5 keys
                    oldest_key = min(self.keys.keys())
                    del self.keys[oldest_key]
                
                self.encryption_stats['key_rotations'] += 1
                logger.info(f"Rotated encryption key: {old_key_id} -> {new_key_id}")
                
                return new_key_id
                
        except Exception as e:
            logger.error(f"Failed to rotate encryption key: {e}")
            raise
    
    def encrypt_message(self, message: bytes, sender_id: str, recipient_id: str,
                       key_id: str = None, metadata: Dict[str, Any] = None) -> bytes:
        """Encrypt ZMQ message"""
        try:
            if key_id is None:
                key_id = self.active_key_id
            
            if key_id not in self.keys:
                raise ValueError(f"Encryption key not found: {key_id}")
            
            # Generate message ID and timestamp
            message_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().timestamp()
            
            # Compress message if enabled
            if self.config.enable_compression:
                import zlib
                message = zlib.compress(message)
            
            # Encrypt message
            key_data = self.keys[key_id]
            
            if self.config.algorithm == EncryptionAlgorithm.AES_256_GCM:
                aesgcm = AESGCM(key_data)
                iv = os.urandom(12)  # 96-bit IV for GCM
                
                # Additional authenticated data
                aad = json.dumps({
                    'message_id': message_id,
                    'timestamp': timestamp,
                    'sender_id': sender_id,
                    'recipient_id': recipient_id
                }).encode()
                
                ciphertext = aesgcm.encrypt(iv, message, aad)
                
                # Generate MAC for message integrity
                mac = hmac.new(key_data, aad + iv + ciphertext, hashlib.sha256).digest()
                
            else:
                raise ValueError(f"Unsupported encryption algorithm: {self.config.algorithm}")
            
            # Create encrypted message
            encrypted_msg = EncryptedMessage(
                message_id=message_id,
                timestamp=timestamp,
                sender_id=sender_id,
                recipient_id=recipient_id,
                key_id=key_id,
                algorithm=self.config.algorithm.value,
                iv=iv,
                ciphertext=ciphertext,
                mac=mac,
                metadata=metadata or {}
            )
            
            # Serialize encrypted message
            msg_data = {
                'message_id': encrypted_msg.message_id,
                'timestamp': encrypted_msg.timestamp,
                'sender_id': encrypted_msg.sender_id,
                'recipient_id': encrypted_msg.recipient_id,
                'key_id': encrypted_msg.key_id,
                'algorithm': encrypted_msg.algorithm,
                'iv': base64.b64encode(encrypted_msg.iv).decode(),
                'ciphertext': base64.b64encode(encrypted_msg.ciphertext).decode(),
                'mac': base64.b64encode(encrypted_msg.mac).decode(),
                'metadata': encrypted_msg.metadata
            }
            
            serialized = json.dumps(msg_data).encode()
            
            self.encryption_stats['messages_encrypted'] += 1
            return serialized
            
        except Exception as e:
            logger.error(f"Failed to encrypt message: {e}")
            self.encryption_stats['errors'] += 1
            raise
    
    def decrypt_message(self, encrypted_data: bytes, expected_recipient: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """Decrypt ZMQ message"""
        try:
            # Deserialize encrypted message
            msg_data = json.loads(encrypted_data.decode())
            
            encrypted_msg = EncryptedMessage(
                message_id=msg_data['message_id'],
                timestamp=msg_data['timestamp'],
                sender_id=msg_data['sender_id'],
                recipient_id=msg_data['recipient_id'],
                key_id=msg_data['key_id'],
                algorithm=msg_data['algorithm'],
                iv=base64.b64decode(msg_data['iv']),
                ciphertext=base64.b64decode(msg_data['ciphertext']),
                mac=base64.b64decode(msg_data['mac']),
                metadata=msg_data['metadata']
            )
            
            # Verify recipient
            if expected_recipient and encrypted_msg.recipient_id != expected_recipient:
                raise ValueError(f"Message not intended for recipient: {expected_recipient}")
            
            # Get decryption key
            if encrypted_msg.key_id not in self.keys:
                raise ValueError(f"Decryption key not found: {encrypted_msg.key_id}")
            
            key_data = self.keys[encrypted_msg.key_id]
            
            # Verify MAC
            aad = json.dumps({
                'message_id': encrypted_msg.message_id,
                'timestamp': encrypted_msg.timestamp,
                'sender_id': encrypted_msg.sender_id,
                'recipient_id': encrypted_msg.recipient_id
            }).encode()
            
            expected_mac = hmac.new(key_data, aad + encrypted_msg.iv + encrypted_msg.ciphertext, hashlib.sha256).digest()
            if not hmac.compare_digest(encrypted_msg.mac, expected_mac):
                raise ValueError("Message MAC verification failed")
            
            # Decrypt message
            if encrypted_msg.algorithm == EncryptionAlgorithm.AES_256_GCM.value:
                aesgcm = AESGCM(key_data)
                plaintext = aesgcm.decrypt(encrypted_msg.iv, encrypted_msg.ciphertext, aad)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {encrypted_msg.algorithm}")
            
            # Decompress if needed
            if self.config.enable_compression:
                import zlib
                plaintext = zlib.decompress(plaintext)
            
            # Return message and metadata
            metadata = {
                'message_id': encrypted_msg.message_id,
                'timestamp': encrypted_msg.timestamp,
                'sender_id': encrypted_msg.sender_id,
                'recipient_id': encrypted_msg.recipient_id,
                'key_id': encrypted_msg.key_id,
                'algorithm': encrypted_msg.algorithm,
                'metadata': encrypted_msg.metadata
            }
            
            self.encryption_stats['messages_decrypted'] += 1
            return plaintext, metadata
            
        except Exception as e:
            logger.error(f"Failed to decrypt message: {e}")
            self.encryption_stats['errors'] += 1
            raise
    
    def create_encrypted_socket(self, socket_type: int, bind_address: str = None,
                              connect_address: str = None, identity: str = None) -> zmq.Socket:
        """Create encrypted ZMQ socket"""
        try:
            socket = self.context.socket(socket_type)
            
            # Set socket identity
            if identity:
                socket.setsockopt(zmq.IDENTITY, identity.encode())
            
            # Configure CURVE encryption if authenticator is available
            if self.authenticator:
                # Generate key pair for this socket
                public_key, secret_key = zmq.curve_keypair()
                socket.curve_publickey = public_key
                socket.curve_secretkey = secret_key
                
                # For client sockets, set server's public key
                if connect_address:
                    socket.curve_serverkey = public_key  # In production, use actual server key
            
            # Bind or connect
            if bind_address:
                socket.bind(bind_address)
                logger.info(f"Encrypted socket bound to {bind_address}")
            elif connect_address:
                socket.connect(connect_address)
                logger.info(f"Encrypted socket connected to {connect_address}")
            
            return socket
            
        except Exception as e:
            logger.error(f"Failed to create encrypted socket: {e}")
            raise
    
    def send_encrypted(self, socket: zmq.Socket, message: bytes, sender_id: str,
                      recipient_id: str, flags: int = 0) -> bool:
        """Send encrypted message"""
        try:
            encrypted_data = self.encrypt_message(message, sender_id, recipient_id)
            socket.send(encrypted_data, flags)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send encrypted message: {e}")
            return False
    
    def recv_encrypted(self, socket: zmq.Socket, expected_recipient: str = None,
                      flags: int = 0) -> Tuple[bytes, Dict[str, Any]]:
        """Receive and decrypt message"""
        try:
            encrypted_data = socket.recv(flags)
            return self.decrypt_message(encrypted_data, expected_recipient)
            
        except Exception as e:
            logger.error(f"Failed to receive encrypted message: {e}")
            raise
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get encryption statistics"""
        return {
            **self.encryption_stats,
            'active_key_id': self.active_key_id,
            'total_keys': len(self.keys),
            'algorithm': self.config.algorithm.value,
            'security_level': self.config.security_level.value
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.authenticator:
                self.authenticator.stop()
                self.authenticator = None
            
            self.context.term()
            logger.info("ZMQ encryption manager cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup ZMQ encryption manager: {e}")

class SecureZMQCommunicator:
    """Secure ZMQ communicator with encryption and authentication"""
    
    def __init__(self, entity_id: str, config: ZMQEncryptionConfig = None):
        self.entity_id = entity_id
        self.encryption_manager = ZMQEncryptionManager(config)
        self.sockets: Dict[str, zmq.Socket] = {}
        self.message_handlers: Dict[str, callable] = {}
        self.running = False
        self._tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize secure communicator"""
        try:
            self.encryption_manager.initialize()
            logger.info(f"Secure ZMQ communicator initialized for {self.entity_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize secure communicator: {e}")
            raise
    
    def create_server_socket(self, address: str, socket_type: int = zmq.REP) -> str:
        """Create server socket"""
        try:
            socket_id = f"server_{len(self.sockets)}"
            socket = self.encryption_manager.create_encrypted_socket(
                socket_type, bind_address=address, identity=socket_id
            )
            self.sockets[socket_id] = socket
            return socket_id
            
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            raise
    
    def create_client_socket(self, address: str, socket_type: int = zmq.REQ) -> str:
        """Create client socket"""
        try:
            socket_id = f"client_{len(self.sockets)}"
            socket = self.encryption_manager.create_encrypted_socket(
                socket_type, connect_address=address, identity=socket_id
            )
            self.sockets[socket_id] = socket
            return socket_id
            
        except Exception as e:
            logger.error(f"Failed to create client socket: {e}")
            raise
    
    def register_handler(self, message_type: str, handler: callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler
    
    async def send_message(self, socket_id: str, message: Dict[str, Any],
                          recipient_id: str) -> bool:
        """Send encrypted message"""
        try:
            socket = self.sockets.get(socket_id)
            if not socket:
                raise ValueError(f"Socket not found: {socket_id}")
            
            # Serialize message
            message_data = json.dumps(message).encode()
            
            # Send encrypted
            return self.encryption_manager.send_encrypted(
                socket, message_data, self.entity_id, recipient_id
            )
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self, socket_id: str, timeout: int = 1000) -> Optional[Dict[str, Any]]:
        """Receive and decrypt message"""
        try:
            socket = self.sockets.get(socket_id)
            if not socket:
                raise ValueError(f"Socket not found: {socket_id}")
            
            # Set timeout
            socket.setsockopt(zmq.RCVTIMEO, timeout)
            
            # Receive encrypted message
            plaintext, metadata = self.encryption_manager.recv_encrypted(
                socket, expected_recipient=self.entity_id
            )
            
            # Deserialize message
            message = json.loads(plaintext.decode())
            message['_metadata'] = metadata
            
            return message
            
        except zmq.Again:
            # Timeout
            return None
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    async def start_message_loop(self, socket_id: str):
        """Start message processing loop"""
        try:
            self.running = True
            
            while self.running:
                message = await self.receive_message(socket_id)
                if message:
                    # Handle message
                    message_type = message.get('type', 'unknown')
                    handler = self.message_handlers.get(message_type)
                    
                    if handler:
                        try:
                            await handler(message)
                        except Exception as e:
                            logger.error(f"Error in message handler: {e}")
                    else:
                        logger.warning(f"No handler for message type: {message_type}")
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
        except Exception as e:
            logger.error(f"Error in message loop: {e}")
    
    def stop(self):
        """Stop communicator"""
        self.running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Close sockets
        for socket in self.sockets.values():
            socket.close()
        
        # Cleanup encryption manager
        self.encryption_manager.cleanup()
        
        logger.info(f"Secure ZMQ communicator stopped for {self.entity_id}")

# Global encryption manager instance
zmq_encryption_manager = ZMQEncryptionManager()