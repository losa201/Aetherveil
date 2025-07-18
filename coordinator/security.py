"""
Security management for Aetherveil Sentinel
Handles authentication, authorization, encryption, and security monitoring
"""
import hashlib
import hmac
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import logging

from config.config import config

logger = logging.getLogger(__name__)

class SecurityManager:
    """Main security management class"""
    
    def __init__(self):
        self.encryption_key = self._derive_key(config.security.encryption_key)
        self.fernet = Fernet(self.encryption_key)
        self.jwt_secret = config.security.jwt_secret
        self.active_tokens = {}  # In production, use Redis or database
        
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        salt = b'aetherveil_sentinel_salt'  # In production, use random salt per user
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(config.security.api_key_length)
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_bytes(32)
        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return salt.hex() + pwdhash.hex()
    
    def verify_password(self, stored_password: str, provided_password: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt = bytes.fromhex(stored_password[:64])
            stored_hash = stored_password[64:]
            pwdhash = hashlib.pbkdf2_hmac(
                'sha256',
                provided_password.encode('utf-8'),
                salt,
                100000
            )
            return pwdhash.hex() == stored_hash
        except Exception:
            return False
    
    def generate_jwt_token(self, user_id: str, permissions: list = None) -> str:
        """Generate JWT token for user"""
        if permissions is None:
            permissions = ["read", "write"]
        
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow(),
            "iss": "aetherveil_sentinel"
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        self.active_tokens[token] = payload
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token is still active
            if token not in self.active_tokens:
                return None
                
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def revoke_token(self, token: str):
        """Revoke JWT token"""
        if token in self.active_tokens:
            del self.active_tokens[token]
    
    def sign_message(self, message: str) -> str:
        """Sign message with HMAC"""
        return hmac.new(
            self.jwt_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, message: str, signature: str) -> bool:
        """Verify message signature"""
        expected_signature = self.sign_message(message)
        return hmac.compare_digest(signature, expected_signature)
    
    def create_blockchain_entry(self, data: Dict[str, Any], previous_hash: str = None) -> Dict[str, Any]:
        """Create tamper-evident blockchain-style log entry"""
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "timestamp": timestamp,
            "data": data,
            "previous_hash": previous_hash or "genesis"
        }
        
        # Create hash of the entry
        entry_str = f"{timestamp}{data}{previous_hash}"
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["hash"] = entry_hash
        
        # Sign the entry
        entry["signature"] = self.sign_message(entry_str)
        
        return entry
    
    def verify_blockchain_entry(self, entry: Dict[str, Any]) -> bool:
        """Verify blockchain entry integrity"""
        try:
            # Extract components
            timestamp = entry["timestamp"]
            data = entry["data"]
            previous_hash = entry["previous_hash"]
            stored_hash = entry["hash"]
            stored_signature = entry["signature"]
            
            # Verify hash
            entry_str = f"{timestamp}{data}{previous_hash}"
            calculated_hash = hashlib.sha256(entry_str.encode()).hexdigest()
            
            if calculated_hash != stored_hash:
                return False
            
            # Verify signature
            return self.verify_signature(entry_str, stored_signature)
            
        except Exception as e:
            logger.error(f"Error verifying blockchain entry: {e}")
            return False

class RoleBasedAccessControl:
    """Role-based access control system"""
    
    def __init__(self):
        self.roles = {
            "admin": {
                "permissions": ["*"],
                "description": "Full system access"
            },
            "operator": {
                "permissions": [
                    "workflows:start",
                    "workflows:stop",
                    "workflows:view",
                    "agents:deploy",
                    "agents:view",
                    "knowledge:query",
                    "knowledge:view",
                    "reports:generate",
                    "reports:view"
                ],
                "description": "Operational access"
            },
            "analyst": {
                "permissions": [
                    "workflows:view",
                    "agents:view",
                    "knowledge:query",
                    "knowledge:view",
                    "reports:view"
                ],
                "description": "Read-only analysis access"
            },
            "viewer": {
                "permissions": [
                    "workflows:view",
                    "agents:view",
                    "knowledge:view",
                    "reports:view"
                ],
                "description": "Read-only viewing access"
            }
        }
        
        self.user_roles = {}  # In production, store in database
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user"""
        if role not in self.roles:
            return False
        
        self.user_roles[user_id] = role
        return True
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        if user_id not in self.user_roles:
            return False
        
        user_role = self.user_roles[user_id]
        role_permissions = self.roles[user_role]["permissions"]
        
        # Admin has all permissions
        if "*" in role_permissions:
            return True
        
        # Check specific permission
        return permission in role_permissions
    
    def get_user_permissions(self, user_id: str) -> list:
        """Get all permissions for user"""
        if user_id not in self.user_roles:
            return []
        
        user_role = self.user_roles[user_id]
        return self.roles[user_role]["permissions"]

class SecurityMonitor:
    """Security monitoring and alerting"""
    
    def __init__(self):
        self.failed_attempts = {}  # In production, use Redis
        self.security_events = []  # In production, use database
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def log_security_event(self, event_type: str, source: str, details: Dict[str, Any]):
        """Log security event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "source": source,
            "details": details
        }
        
        self.security_events.append(event)
        logger.warning(f"Security event: {event_type} from {source}")
        
        # Trigger alerts for critical events
        if event_type in ["failed_login", "unauthorized_access", "suspicious_activity"]:
            self._trigger_alert(event)
    
    def _trigger_alert(self, event: Dict[str, Any]):
        """Trigger security alert"""
        # In production, send to monitoring system
        logger.critical(f"SECURITY ALERT: {event}")
    
    def record_failed_attempt(self, identifier: str):
        """Record failed authentication attempt"""
        now = datetime.utcnow()
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append(now)
        
        # Remove old attempts
        cutoff = now - timedelta(minutes=15)
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        # Check if lockout threshold reached
        if len(self.failed_attempts[identifier]) >= self.max_failed_attempts:
            self.log_security_event(
                "account_lockout",
                identifier,
                {"attempts": len(self.failed_attempts[identifier])}
            )
            return True
        
        return False
    
    def is_locked_out(self, identifier: str) -> bool:
        """Check if identifier is locked out"""
        if identifier not in self.failed_attempts:
            return False
        
        return len(self.failed_attempts[identifier]) >= self.max_failed_attempts
    
    def clear_failed_attempts(self, identifier: str):
        """Clear failed attempts for identifier"""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]

# Global instances
security_manager = SecurityManager()
rbac = RoleBasedAccessControl()
security_monitor = SecurityMonitor()