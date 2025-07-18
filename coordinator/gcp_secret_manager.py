"""
GCP Secret Manager Integration for Aetherveil Sentinel
Implements secure key storage and management using Google Cloud Secret Manager
"""

import asyncio
import json
import logging
import os
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from pathlib import Path

from google.cloud import secretmanager
from google.cloud import monitoring_v3
from google.api_core import exceptions as gcp_exceptions
from google.oauth2 import service_account
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from coordinator.security_manager import SecurityLevel, EncryptionAlgorithm

logger = logging.getLogger(__name__)

class SecretType(Enum):
    """Types of secrets"""
    ENCRYPTION_KEY = "encryption_key"
    DATABASE_PASSWORD = "database_password"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    JWT_SECRET = "jwt_secret"
    WEBHOOK_SECRET = "webhook_secret"
    OAUTH_CLIENT_SECRET = "oauth_client_secret"
    CUSTOM = "custom"

class SecretStatus(Enum):
    """Secret status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"

@dataclass
class SecretMetadata:
    """Secret metadata"""
    secret_id: str
    name: str
    description: str
    secret_type: SecretType
    security_level: SecurityLevel
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    rotation_interval: Optional[int] = None  # seconds
    last_rotation: Optional[datetime] = None
    auto_rotate: bool = False
    status: SecretStatus = SecretStatus.ACTIVE
    labels: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    version: int = 1

@dataclass
class SecretVersion:
    """Secret version information"""
    version_id: str
    secret_id: str
    created_at: datetime
    status: SecretStatus
    is_current: bool = False
    checksum: str = ""
    size: int = 0

class GCPSecretManager:
    """Google Cloud Secret Manager integration"""
    
    def __init__(self, project_id: str, credentials_path: str = None):
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.client: Optional[secretmanager.SecretManagerServiceClient] = None
        self.monitoring_client: Optional[monitoring_v3.MetricServiceClient] = None
        self.secrets_metadata: Dict[str, SecretMetadata] = {}
        self.cache: Dict[str, Tuple[str, datetime]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.local_encryption_key: Optional[bytes] = None
        self._lock = threading.Lock()
        
        # Statistics
        self.statistics = {
            'secrets_created': 0,
            'secrets_accessed': 0,
            'secrets_updated': 0,
            'secrets_rotated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0
        }
        
        # Background tasks
        self._rotation_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize GCP Secret Manager client"""
        try:
            # Initialize credentials
            if self.credentials_path and os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.client = secretmanager.SecretManagerServiceClient(credentials=credentials)
                self.monitoring_client = monitoring_v3.MetricServiceClient(credentials=credentials)
            else:
                # Use default credentials
                self.client = secretmanager.SecretManagerServiceClient()
                self.monitoring_client = monitoring_v3.MetricServiceClient()
            
            # Generate local encryption key for additional security
            self.local_encryption_key = self._generate_local_encryption_key()
            
            # Load existing secrets metadata
            await self._load_secrets_metadata()
            
            # Start background tasks
            self._rotation_task = asyncio.create_task(self._rotation_loop())
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("GCP Secret Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP Secret Manager: {e}")
            raise
    
    def _generate_local_encryption_key(self) -> bytes:
        """Generate local encryption key"""
        try:
            # Try to load existing key
            key_file = Path("/app/keys/local_encryption.key")
            
            if key_file.exists():
                with open(key_file, "rb") as f:
                    return f.read()
            
            # Generate new key
            key = Fernet.generate_key()
            
            # Save key
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, "wb") as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            return key
            
        except Exception as e:
            logger.error(f"Failed to generate local encryption key: {e}")
            raise
    
    async def _load_secrets_metadata(self):
        """Load secrets metadata from GCP"""
        try:
            parent = f"projects/{self.project_id}"
            
            # List all secrets
            request = secretmanager.ListSecretsRequest(parent=parent)
            secrets = self.client.list_secrets(request=request)
            
            for secret in secrets:
                try:
                    # Parse secret name
                    secret_id = secret.name.split('/')[-1]
                    
                    # Get labels and parse metadata
                    labels = dict(secret.labels) if secret.labels else {}
                    
                    # Create metadata
                    metadata = SecretMetadata(
                        secret_id=secret_id,
                        name=labels.get('name', secret_id),
                        description=labels.get('description', ''),
                        secret_type=SecretType(labels.get('type', 'custom')),
                        security_level=SecurityLevel(labels.get('security_level', 'medium')),
                        created_at=secret.create_time,
                        updated_at=secret.create_time,
                        expires_at=datetime.fromisoformat(labels['expires_at']) if labels.get('expires_at') else None,
                        rotation_interval=int(labels['rotation_interval']) if labels.get('rotation_interval') else None,
                        auto_rotate=labels.get('auto_rotate', 'false').lower() == 'true',
                        status=SecretStatus(labels.get('status', 'active')),
                        labels=labels,
                        tags=labels.get('tags', '').split(',') if labels.get('tags') else [],
                        access_count=int(labels.get('access_count', '0')),
                        last_accessed=datetime.fromisoformat(labels['last_accessed']) if labels.get('last_accessed') else None,
                        version=int(labels.get('version', '1'))
                    )
                    
                    self.secrets_metadata[secret_id] = metadata
                    
                except Exception as e:
                    logger.warning(f"Failed to load metadata for secret {secret.name}: {e}")
            
            logger.info(f"Loaded metadata for {len(self.secrets_metadata)} secrets")
            
        except Exception as e:
            logger.error(f"Failed to load secrets metadata: {e}")
    
    async def create_secret(self, name: str, value: str, secret_type: SecretType = SecretType.CUSTOM,
                           description: str = "", security_level: SecurityLevel = SecurityLevel.MEDIUM,
                           expires_at: datetime = None, rotation_interval: int = None,
                           auto_rotate: bool = False, labels: Dict[str, str] = None,
                           tags: List[str] = None, encrypt_locally: bool = True) -> str:
        """Create new secret"""
        try:
            secret_id = f"aetherveil-{name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
            
            # Encrypt value locally if requested
            if encrypt_locally:
                fernet = Fernet(self.local_encryption_key)
                value = fernet.encrypt(value.encode()).decode()
            
            # Prepare labels
            secret_labels = {
                'name': name,
                'description': description,
                'type': secret_type.value,
                'security_level': security_level.value,
                'created_by': 'aetherveil-sentinel',
                'encrypted_locally': str(encrypt_locally).lower(),
                'version': '1',
                'access_count': '0'
            }
            
            if expires_at:
                secret_labels['expires_at'] = expires_at.isoformat()
            
            if rotation_interval:
                secret_labels['rotation_interval'] = str(rotation_interval)
            
            if auto_rotate:
                secret_labels['auto_rotate'] = 'true'
            
            if labels:
                secret_labels.update(labels)
            
            if tags:
                secret_labels['tags'] = ','.join(tags)
            
            # Create secret in GCP
            parent = f"projects/{self.project_id}"
            
            secret = secretmanager.Secret(
                labels=secret_labels,
                replication=secretmanager.Replication(
                    automatic=secretmanager.Replication.Automatic()
                )
            )
            
            request = secretmanager.CreateSecretRequest(
                parent=parent,
                secret_id=secret_id,
                secret=secret
            )
            
            created_secret = self.client.create_secret(request=request)
            
            # Add secret version
            version_request = secretmanager.AddSecretVersionRequest(
                parent=created_secret.name,
                payload=secretmanager.SecretPayload(data=value.encode())
            )
            
            self.client.add_secret_version(request=version_request)
            
            # Create metadata
            metadata = SecretMetadata(
                secret_id=secret_id,
                name=name,
                description=description,
                secret_type=secret_type,
                security_level=security_level,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                expires_at=expires_at,
                rotation_interval=rotation_interval,
                auto_rotate=auto_rotate,
                status=SecretStatus.ACTIVE,
                labels=secret_labels,
                tags=tags or [],
                version=1
            )
            
            self.secrets_metadata[secret_id] = metadata
            
            # Update statistics
            self.statistics['secrets_created'] += 1
            self.statistics['api_calls'] += 2
            
            logger.info(f"Created secret: {name} ({secret_id})")
            return secret_id
            
        except Exception as e:
            logger.error(f"Failed to create secret: {e}")
            self.statistics['errors'] += 1
            raise
    
    async def get_secret(self, secret_id: str, version: str = "latest", 
                        decrypt_locally: bool = True) -> Optional[str]:
        """Get secret value"""
        try:
            # Check cache first
            cache_key = f"{secret_id}:{version}"
            cached_value = self._get_cached_value(cache_key)
            if cached_value:
                self.statistics['cache_hits'] += 1
                return cached_value
            
            self.statistics['cache_misses'] += 1
            
            # Get secret from GCP
            secret_name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            
            request = secretmanager.AccessSecretVersionRequest(name=secret_name)
            response = self.client.access_secret_version(request=request)
            
            # Decode value
            value = response.payload.data.decode()
            
            # Decrypt locally if needed
            if decrypt_locally:
                try:
                    fernet = Fernet(self.local_encryption_key)
                    value = fernet.decrypt(value.encode()).decode()
                except Exception:
                    # Value might not be encrypted locally
                    pass
            
            # Cache value
            self._cache_value(cache_key, value)
            
            # Update metadata
            if secret_id in self.secrets_metadata:
                metadata = self.secrets_metadata[secret_id]
                metadata.access_count += 1
                metadata.last_accessed = datetime.utcnow()
                
                # Update labels in GCP
                await self._update_secret_labels(secret_id, {
                    'access_count': str(metadata.access_count),
                    'last_accessed': metadata.last_accessed.isoformat()
                })
            
            # Update statistics
            self.statistics['secrets_accessed'] += 1
            self.statistics['api_calls'] += 1
            
            return value
            
        except gcp_exceptions.NotFound:
            logger.warning(f"Secret not found: {secret_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get secret {secret_id}: {e}")
            self.statistics['errors'] += 1
            return None
    
    async def update_secret(self, secret_id: str, value: str, encrypt_locally: bool = True) -> bool:
        """Update secret value"""
        try:
            # Encrypt value locally if requested
            if encrypt_locally:
                fernet = Fernet(self.local_encryption_key)
                value = fernet.encrypt(value.encode()).decode()
            
            # Add new version
            secret_name = f"projects/{self.project_id}/secrets/{secret_id}"
            
            request = secretmanager.AddSecretVersionRequest(
                parent=secret_name,
                payload=secretmanager.SecretPayload(data=value.encode())
            )
            
            version = self.client.add_secret_version(request=request)
            
            # Update metadata
            if secret_id in self.secrets_metadata:
                metadata = self.secrets_metadata[secret_id]
                metadata.updated_at = datetime.utcnow()
                metadata.version += 1
                
                # Update labels
                await self._update_secret_labels(secret_id, {
                    'version': str(metadata.version),
                    'updated_at': metadata.updated_at.isoformat()
                })
            
            # Clear cache
            self._clear_secret_cache(secret_id)
            
            # Update statistics
            self.statistics['secrets_updated'] += 1
            self.statistics['api_calls'] += 1
            
            logger.info(f"Updated secret: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update secret {secret_id}: {e}")
            self.statistics['errors'] += 1
            return False
    
    async def rotate_secret(self, secret_id: str, generator_func: callable = None) -> bool:
        """Rotate secret value"""
        try:
            metadata = self.secrets_metadata.get(secret_id)
            if not metadata:
                logger.warning(f"Secret metadata not found: {secret_id}")
                return False
            
            # Generate new value
            if generator_func:
                new_value = generator_func()
            else:
                new_value = self._generate_default_value(metadata.secret_type)
            
            # Update secret
            success = await self.update_secret(secret_id, new_value)
            
            if success:
                # Update rotation metadata
                metadata.last_rotation = datetime.utcnow()
                
                await self._update_secret_labels(secret_id, {
                    'last_rotation': metadata.last_rotation.isoformat()
                })
                
                # Update statistics
                self.statistics['secrets_rotated'] += 1
                
                logger.info(f"Rotated secret: {secret_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_id}: {e}")
            self.statistics['errors'] += 1
            return False
    
    async def delete_secret(self, secret_id: str) -> bool:
        """Delete secret"""
        try:
            secret_name = f"projects/{self.project_id}/secrets/{secret_id}"
            
            request = secretmanager.DeleteSecretRequest(name=secret_name)
            self.client.delete_secret(request=request)
            
            # Remove from metadata
            if secret_id in self.secrets_metadata:
                del self.secrets_metadata[secret_id]
            
            # Clear cache
            self._clear_secret_cache(secret_id)
            
            # Update statistics
            self.statistics['api_calls'] += 1
            
            logger.info(f"Deleted secret: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            self.statistics['errors'] += 1
            return False
    
    async def list_secrets(self, secret_type: SecretType = None, 
                          status: SecretStatus = None) -> List[SecretMetadata]:
        """List secrets with optional filtering"""
        try:
            secrets = []
            
            for metadata in self.secrets_metadata.values():
                # Apply filters
                if secret_type and metadata.secret_type != secret_type:
                    continue
                
                if status and metadata.status != status:
                    continue
                
                secrets.append(metadata)
            
            return secrets
            
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    async def get_secret_versions(self, secret_id: str) -> List[SecretVersion]:
        """Get all versions of a secret"""
        try:
            secret_name = f"projects/{self.project_id}/secrets/{secret_id}"
            
            request = secretmanager.ListSecretVersionsRequest(parent=secret_name)
            versions = self.client.list_secret_versions(request=request)
            
            version_list = []
            for version in versions:
                version_info = SecretVersion(
                    version_id=version.name.split('/')[-1],
                    secret_id=secret_id,
                    created_at=version.create_time,
                    status=SecretStatus(version.state.name.lower() if hasattr(version, 'state') else 'active'),
                    checksum=getattr(version, 'checksum', ''),
                    size=len(version.payload.data) if hasattr(version, 'payload') else 0
                )
                version_list.append(version_info)
            
            # Mark latest version
            if version_list:
                version_list[0].is_current = True
            
            # Update statistics
            self.statistics['api_calls'] += 1
            
            return version_list
            
        except Exception as e:
            logger.error(f"Failed to get secret versions for {secret_id}: {e}")
            self.statistics['errors'] += 1
            return []
    
    def _get_cached_value(self, cache_key: str) -> Optional[str]:
        """Get cached secret value"""
        try:
            with self._lock:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    value, cached_at = cached_data
                    if (datetime.utcnow() - cached_at).total_seconds() < self.cache_ttl:
                        return value
                    else:
                        del self.cache[cache_key]
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached value: {e}")
            return None
    
    def _cache_value(self, cache_key: str, value: str):
        """Cache secret value"""
        try:
            with self._lock:
                self.cache[cache_key] = (value, datetime.utcnow())
                
                # Limit cache size
                if len(self.cache) > 1000:
                    # Remove oldest entries
                    oldest_entries = sorted(self.cache.items(), key=lambda x: x[1][1])
                    for key, _ in oldest_entries[:100]:
                        del self.cache[key]
                
        except Exception as e:
            logger.error(f"Failed to cache value: {e}")
    
    def _clear_secret_cache(self, secret_id: str):
        """Clear cache for specific secret"""
        try:
            with self._lock:
                keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{secret_id}:")]
                for key in keys_to_remove:
                    del self.cache[key]
                
        except Exception as e:
            logger.error(f"Failed to clear secret cache: {e}")
    
    async def _update_secret_labels(self, secret_id: str, labels: Dict[str, str]):
        """Update secret labels"""
        try:
            secret_name = f"projects/{self.project_id}/secrets/{secret_id}"
            
            # Get current secret
            request = secretmanager.GetSecretRequest(name=secret_name)
            secret = self.client.get_secret(request=request)
            
            # Update labels
            current_labels = dict(secret.labels) if secret.labels else {}
            current_labels.update(labels)
            
            # Update secret
            secret.labels = current_labels
            
            update_request = secretmanager.UpdateSecretRequest(
                secret=secret,
                update_mask={'paths': ['labels']}
            )
            
            self.client.update_secret(request=update_request)
            
            # Update statistics
            self.statistics['api_calls'] += 2
            
        except Exception as e:
            logger.error(f"Failed to update secret labels: {e}")
    
    def _generate_default_value(self, secret_type: SecretType) -> str:
        """Generate default value for secret type"""
        try:
            if secret_type == SecretType.ENCRYPTION_KEY:
                return Fernet.generate_key().decode()
            elif secret_type == SecretType.API_KEY:
                return base64.b64encode(os.urandom(32)).decode()
            elif secret_type == SecretType.JWT_SECRET:
                return base64.b64encode(os.urandom(64)).decode()
            elif secret_type == SecretType.WEBHOOK_SECRET:
                return base64.b64encode(os.urandom(32)).decode()
            else:
                return base64.b64encode(os.urandom(32)).decode()
                
        except Exception as e:
            logger.error(f"Failed to generate default value: {e}")
            return base64.b64encode(os.urandom(32)).decode()
    
    async def _rotation_loop(self):
        """Background secret rotation loop"""
        while True:
            try:
                now = datetime.utcnow()
                
                # Find secrets that need rotation
                for secret_id, metadata in self.secrets_metadata.items():
                    if not metadata.auto_rotate or not metadata.rotation_interval:
                        continue
                    
                    # Check if rotation is needed
                    if metadata.last_rotation:
                        next_rotation = metadata.last_rotation + timedelta(seconds=metadata.rotation_interval)
                        if now >= next_rotation:
                            logger.info(f"Auto-rotating secret: {secret_id}")
                            await self.rotate_secret(secret_id)
                    elif metadata.created_at:
                        next_rotation = metadata.created_at + timedelta(seconds=metadata.rotation_interval)
                        if now >= next_rotation:
                            logger.info(f"Auto-rotating secret: {secret_id}")
                            await self.rotate_secret(secret_id)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in rotation loop: {e}")
                await asyncio.sleep(300)
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Send metrics to GCP Monitoring
                await self._send_metrics()
                
                await asyncio.sleep(300)  # Send metrics every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def _send_metrics(self):
        """Send metrics to GCP Monitoring"""
        try:
            if not self.monitoring_client:
                return
            
            project_name = f"projects/{self.project_id}"
            
            # Create metric descriptors and send data
            metrics = [
                ('secrets_created', self.statistics['secrets_created']),
                ('secrets_accessed', self.statistics['secrets_accessed']),
                ('secrets_updated', self.statistics['secrets_updated']),
                ('secrets_rotated', self.statistics['secrets_rotated']),
                ('cache_hits', self.statistics['cache_hits']),
                ('cache_misses', self.statistics['cache_misses']),
                ('api_calls', self.statistics['api_calls']),
                ('errors', self.statistics['errors']),
                ('total_secrets', len(self.secrets_metadata)),
                ('cache_size', len(self.cache))
            ]
            
            # In a real implementation, you would send these metrics to GCP Monitoring
            # For now, we'll just log them
            logger.debug(f"Metrics: {dict(metrics)}")
            
        except Exception as e:
            logger.error(f"Failed to send metrics: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                # Clean up expired cache entries
                now = datetime.utcnow()
                with self._lock:
                    expired_keys = [
                        key for key, (value, cached_at) in self.cache.items()
                        if (now - cached_at).total_seconds() > self.cache_ttl
                    ]
                    
                    for key in expired_keys:
                        del self.cache[key]
                
                # Check for expired secrets
                for secret_id, metadata in self.secrets_metadata.items():
                    if metadata.expires_at and now >= metadata.expires_at:
                        if metadata.status != SecretStatus.EXPIRED:
                            metadata.status = SecretStatus.EXPIRED
                            logger.warning(f"Secret expired: {secret_id}")
                            
                            # Update labels
                            await self._update_secret_labels(secret_id, {
                                'status': SecretStatus.EXPIRED.value
                            })
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get GCP Secret Manager statistics"""
        return {
            **self.statistics,
            'total_secrets': len(self.secrets_metadata),
            'cache_size': len(self.cache),
            'cache_ttl': self.cache_ttl,
            'project_id': self.project_id,
            'active_secrets': len([s for s in self.secrets_metadata.values() if s.status == SecretStatus.ACTIVE]),
            'expired_secrets': len([s for s in self.secrets_metadata.values() if s.status == SecretStatus.EXPIRED]),
            'auto_rotate_secrets': len([s for s in self.secrets_metadata.values() if s.auto_rotate])
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Try to list secrets
            parent = f"projects/{self.project_id}"
            request = secretmanager.ListSecretsRequest(parent=parent, page_size=1)
            self.client.list_secrets(request=request)
            
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'project_id': self.project_id,
                'secrets_count': len(self.secrets_metadata)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def shutdown(self):
        """Shutdown GCP Secret Manager"""
        try:
            # Cancel background tasks
            if self._rotation_task:
                self._rotation_task.cancel()
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Clear cache
            with self._lock:
                self.cache.clear()
            
            logger.info("GCP Secret Manager shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown GCP Secret Manager: {e}")

# Global GCP Secret Manager instance
gcp_secret_manager: Optional[GCPSecretManager] = None

def initialize_gcp_secret_manager(project_id: str, credentials_path: str = None):
    """Initialize global GCP Secret Manager instance"""
    global gcp_secret_manager
    gcp_secret_manager = GCPSecretManager(project_id, credentials_path)
    return gcp_secret_manager