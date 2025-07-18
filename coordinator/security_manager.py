"""
Advanced Security Manager for Aetherveil Sentinel
Implements mutual TLS, certificate rotation, zero-trust architecture, and secure communication
"""

import asyncio
import json
import logging
import os
import ssl
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, asdict
import socket
from pathlib import Path
import hashlib
import hmac
import secrets
import base64

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import jwt
from cryptography.hazmat.primitives import serialization
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, HashingError
import time
import threading
from collections import defaultdict, deque
import ipaddress
from google.cloud import secretmanager
from google.cloud import monitoring_v3
from google.api_core import exceptions as gcp_exceptions
import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator
import struct
import signal
import weakref

from config.config import config

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    MAXIMUM = "maximum"

class EncryptionAlgorithm(Enum):
    """Encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"

class KeyDerivationMethod(Enum):
    """Key derivation methods"""
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    ARGON2 = "argon2"
    HKDF = "hkdf"

class ThreatLevel(Enum):
    """Threat levels for security monitoring"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class CertificateType(Enum):
    """Certificate types"""
    ROOT_CA = "root_ca"
    INTERMEDIATE_CA = "intermediate_ca"
    SERVER = "server"
    CLIENT = "client"
    AGENT = "agent"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    CERTIFICATE = "certificate"
    JWT = "jwt"
    API_KEY = "api_key"
    MUTUAL_TLS = "mutual_tls"

@dataclass
class SecurityIdentity:
    """Security identity for entities"""
    entity_id: str
    entity_type: str
    certificate_path: str
    private_key_path: str
    public_key_path: str
    security_level: SecurityLevel
    permissions: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "certificate_path": self.certificate_path,
            "private_key_path": self.private_key_path,
            "public_key_path": self.public_key_path,
            "security_level": self.security_level.value,
            "permissions": self.permissions,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }

@dataclass
class SecurityToken:
    """Security token for authentication"""
    token_id: str
    entity_id: str
    token_type: str
    token_value: str
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any]
    refresh_token: Optional[str] = None
    revoked: bool = False

@dataclass
class EncryptionKey:
    """Encryption key with metadata"""
    key_id: str
    key_data: bytes
    algorithm: EncryptionAlgorithm
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]
    salt: Optional[bytes] = None
    iv: Optional[bytes] = None

@dataclass
class SecurityEvent:
    """Security event for monitoring"""
    event_id: str
    event_type: str
    severity: ThreatLevel
    source: str
    target: Optional[str]
    description: str
    metadata: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False

@dataclass
class RateLimitRule:
    """Rate limiting rule"""
    rule_id: str
    entity_pattern: str
    resource_pattern: str
    max_requests: int
    window_seconds: int
    burst_limit: int
    block_duration: int
    metadata: Dict[str, Any]

@dataclass
class BlockchainLogEntry:
    """Blockchain-style log entry"""
    block_id: str
    previous_hash: str
    timestamp: datetime
    events: List[SecurityEvent]
    merkle_root: str
    nonce: int
    hash_value: str
    signature: str

class AdvancedEncryptionManager:
    """Advanced encryption manager with multiple algorithms and key derivation methods"""
    
    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key: Optional[bytes] = None
        self.argon2_hasher = PasswordHasher()
        
    def initialize_master_key(self, password: str, salt: bytes = None) -> bytes:
        """Initialize master key from password"""
        if salt is None:
            salt = os.urandom(32)
        
        # Use Argon2 for master key derivation
        try:
            key_hash = self.argon2_hasher.hash(password, salt=salt)
            self.master_key = base64.b64decode(key_hash.split('$')[-1])[:32]
            return self.master_key
        except Exception as e:
            logger.error(f"Failed to initialize master key: {e}")
            raise
    
    def derive_key(self, password: str, salt: bytes, method: KeyDerivationMethod = KeyDerivationMethod.ARGON2,
                   key_length: int = 32, iterations: int = 100000) -> bytes:
        """Derive key using specified method"""
        try:
            if method == KeyDerivationMethod.ARGON2:
                key_hash = self.argon2_hasher.hash(password, salt=salt)
                return base64.b64decode(key_hash.split('$')[-1])[:key_length]
            
            elif method == KeyDerivationMethod.PBKDF2:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=key_length,
                    salt=salt,
                    iterations=iterations
                )
                return kdf.derive(password.encode())
            
            elif method == KeyDerivationMethod.SCRYPT:
                kdf = Scrypt(
                    algorithm=hashes.SHA256(),
                    length=key_length,
                    salt=salt,
                    n=2**14,
                    r=8,
                    p=1
                )
                return kdf.derive(password.encode())
            
            elif method == KeyDerivationMethod.HKDF:
                kdf = HKDF(
                    algorithm=hashes.SHA256(),
                    length=key_length,
                    salt=salt,
                    info=b'aetherveil-sentinel'
                )
                return kdf.derive(password.encode())
            
            else:
                raise ValueError(f"Unsupported key derivation method: {method}")
                
        except Exception as e:
            logger.error(f"Failed to derive key: {e}")
            raise
    
    def generate_encryption_key(self, algorithm: EncryptionAlgorithm, key_id: str = None,
                              expires_in: int = None) -> EncryptionKey:
        """Generate encryption key"""
        try:
            if key_id is None:
                key_id = str(uuid.uuid4())
            
            created_at = datetime.utcnow()
            expires_at = created_at + timedelta(seconds=expires_in) if expires_in else None
            
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                key_data = os.urandom(32)  # 256-bit key
            elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                key_data = os.urandom(32)  # 256-bit key
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                key_data = os.urandom(32)  # 256-bit key
            elif algorithm == EncryptionAlgorithm.FERNET:
                key_data = Fernet.generate_key()
            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
            
            encryption_key = EncryptionKey(
                key_id=key_id,
                key_data=key_data,
                algorithm=algorithm,
                created_at=created_at,
                expires_at=expires_at,
                metadata={}
            )
            
            self.keys[key_id] = encryption_key
            return encryption_key
            
        except Exception as e:
            logger.error(f"Failed to generate encryption key: {e}")
            raise
    
    def encrypt_data(self, data: bytes, key_id: str, associated_data: bytes = None) -> Tuple[bytes, bytes]:
        """Encrypt data with specified key"""
        try:
            encryption_key = self.keys.get(key_id)
            if not encryption_key:
                raise ValueError(f"Encryption key not found: {key_id}")
            
            if encryption_key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                aesgcm = AESGCM(encryption_key.key_data)
                iv = os.urandom(12)  # 96-bit IV for GCM
                ciphertext = aesgcm.encrypt(iv, data, associated_data)
                return ciphertext, iv
            
            elif encryption_key.algorithm == EncryptionAlgorithm.AES_256_CBC:
                iv = os.urandom(16)  # 128-bit IV for CBC
                cipher = Cipher(algorithms.AES(encryption_key.key_data), modes.CBC(iv))
                encryptor = cipher.encryptor()
                
                # Pad data to block size
                padder = padding.PKCS7(128).padder()
                padded_data = padder.update(data) + padder.finalize()
                
                ciphertext = encryptor.update(padded_data) + encryptor.finalize()
                return ciphertext, iv
            
            elif encryption_key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
                chacha = ChaCha20Poly1305(encryption_key.key_data)
                nonce = os.urandom(12)  # 96-bit nonce
                ciphertext = chacha.encrypt(nonce, data, associated_data)
                return ciphertext, nonce
            
            elif encryption_key.algorithm == EncryptionAlgorithm.FERNET:
                fernet = Fernet(encryption_key.key_data)
                ciphertext = fernet.encrypt(data)
                return ciphertext, b''  # Fernet includes IV internally
            
            else:
                raise ValueError(f"Unsupported encryption algorithm: {encryption_key.algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, ciphertext: bytes, key_id: str, iv: bytes, associated_data: bytes = None) -> bytes:
        """Decrypt data with specified key"""
        try:
            encryption_key = self.keys.get(key_id)
            if not encryption_key:
                raise ValueError(f"Encryption key not found: {key_id}")
            
            if encryption_key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                aesgcm = AESGCM(encryption_key.key_data)
                plaintext = aesgcm.decrypt(iv, ciphertext, associated_data)
                return plaintext
            
            elif encryption_key.algorithm == EncryptionAlgorithm.AES_256_CBC:
                cipher = Cipher(algorithms.AES(encryption_key.key_data), modes.CBC(iv))
                decryptor = cipher.decryptor()
                padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
                
                # Unpad data
                unpadder = padding.PKCS7(128).unpadder()
                plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
                return plaintext
            
            elif encryption_key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
                chacha = ChaCha20Poly1305(encryption_key.key_data)
                plaintext = chacha.decrypt(iv, ciphertext, associated_data)
                return plaintext
            
            elif encryption_key.algorithm == EncryptionAlgorithm.FERNET:
                fernet = Fernet(encryption_key.key_data)
                plaintext = fernet.decrypt(ciphertext)
                return plaintext
            
            else:
                raise ValueError(f"Unsupported encryption algorithm: {encryption_key.algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def rotate_key(self, key_id: str) -> EncryptionKey:
        """Rotate encryption key"""
        try:
            old_key = self.keys.get(key_id)
            if not old_key:
                raise ValueError(f"Key not found: {key_id}")
            
            # Generate new key with same algorithm
            new_key = self.generate_encryption_key(
                algorithm=old_key.algorithm,
                key_id=key_id,
                expires_in=None
            )
            
            # Archive old key
            old_key.metadata['rotated_at'] = datetime.utcnow().isoformat()
            old_key.metadata['status'] = 'archived'
            
            return new_key
            
        except Exception as e:
            logger.error(f"Failed to rotate key {key_id}: {e}")
            raise
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get encryption key by ID"""
        return self.keys.get(key_id)
    
    def list_keys(self) -> List[EncryptionKey]:
        """List all encryption keys"""
        return list(self.keys.values())
    
    def delete_key(self, key_id: str):
        """Delete encryption key"""
        if key_id in self.keys:
            del self.keys[key_id]

class CertificateManager:
    """Certificate management for PKI infrastructure"""
    
    def __init__(self, cert_dir: str = "/app/certs"):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(exist_ok=True)
        self.ca_cert = None
        self.ca_key = None
        self.certificates: Dict[str, SecurityIdentity] = {}
        
    async def initialize(self):
        """Initialize certificate manager"""
        try:
            # Create or load CA certificate
            await self._setup_ca()
            
            # Load existing certificates
            await self._load_certificates()
            
            logger.info("Certificate manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize certificate manager: {e}")
            raise
    
    async def _setup_ca(self):
        """Setup Certificate Authority"""
        try:
            ca_cert_path = self.cert_dir / "ca.crt"
            ca_key_path = self.cert_dir / "ca.key"
            
            if ca_cert_path.exists() and ca_key_path.exists():
                # Load existing CA
                with open(ca_cert_path, 'rb') as f:
                    self.ca_cert = x509.load_pem_x509_certificate(f.read())
                
                with open(ca_key_path, 'rb') as f:
                    self.ca_key = serialization.load_pem_private_key(f.read(), password=None)
                
                logger.info("Loaded existing CA certificate")
            else:
                # Generate new CA
                await self._generate_ca_certificate()
                logger.info("Generated new CA certificate")
                
        except Exception as e:
            logger.error(f"Failed to setup CA: {e}")
            raise
    
    async def _generate_ca_certificate(self):
        """Generate CA certificate"""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
            )
            
            # Create CA certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aetherveil Sentinel"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Aetherveil CA"),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=3650)  # 10 years
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.DNSName("aetherveil-ca"),
                ]),
                critical=False,
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            ).add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).sign(private_key, hashes.SHA256())
            
            # Save certificate and key
            ca_cert_path = self.cert_dir / "ca.crt"
            ca_key_path = self.cert_dir / "ca.key"
            
            with open(ca_cert_path, 'wb') as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            with open(ca_key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Set permissions
            os.chmod(ca_key_path, 0o600)
            
            self.ca_cert = cert
            self.ca_key = private_key
            
        except Exception as e:
            logger.error(f"Failed to generate CA certificate: {e}")
            raise
    
    async def _load_certificates(self):
        """Load existing certificates"""
        try:
            cert_files = list(self.cert_dir.glob("*.crt"))
            
            for cert_file in cert_files:
                if cert_file.name == "ca.crt":
                    continue
                    
                try:
                    with open(cert_file, 'rb') as f:
                        cert = x509.load_pem_x509_certificate(f.read())
                    
                    # Extract entity ID from certificate
                    entity_id = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
                    
                    # Create security identity
                    identity = SecurityIdentity(
                        entity_id=entity_id,
                        entity_type="unknown",
                        certificate_path=str(cert_file),
                        private_key_path=str(cert_file.with_suffix('.key')),
                        public_key_path=str(cert_file.with_suffix('.pub')),
                        security_level=SecurityLevel.MEDIUM,
                        permissions=[],
                        metadata={},
                        created_at=cert.not_valid_before,
                        expires_at=cert.not_valid_after
                    )
                    
                    self.certificates[entity_id] = identity
                    
                except Exception as e:
                    logger.warning(f"Failed to load certificate {cert_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load certificates: {e}")
    
    async def create_certificate(self, entity_id: str, entity_type: str,
                               cert_type: CertificateType = CertificateType.CLIENT,
                               dns_names: List[str] = None,
                               ip_addresses: List[str] = None,
                               validity_days: int = 365) -> SecurityIdentity:
        """Create new certificate"""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Create certificate
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aetherveil Sentinel"),
                x509.NameAttribute(NameOID.COMMON_NAME, entity_id),
            ])
            
            cert_builder = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                self.ca_cert.subject
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            )
            
            # Add extensions based on certificate type
            if cert_type in [CertificateType.SERVER, CertificateType.AGENT]:
                # Add SAN for server certificates
                san_names = []
                if dns_names:
                    san_names.extend([x509.DNSName(name) for name in dns_names])
                if ip_addresses:
                    san_names.extend([x509.IPAddress(ip) for ip in ip_addresses])
                
                if san_names:
                    cert_builder = cert_builder.add_extension(
                        x509.SubjectAlternativeName(san_names),
                        critical=False,
                    )
                
                # Key usage for server certificates
                cert_builder = cert_builder.add_extension(
                    x509.KeyUsage(
                        digital_signature=True,
                        key_encipherment=True,
                        content_commitment=False,
                        data_encipherment=False,
                        key_agreement=False,
                        key_cert_sign=False,
                        crl_sign=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                )
                
                # Extended key usage
                cert_builder = cert_builder.add_extension(
                    x509.ExtendedKeyUsage([
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]),
                    critical=True,
                )
            
            elif cert_type == CertificateType.CLIENT:
                # Key usage for client certificates
                cert_builder = cert_builder.add_extension(
                    x509.KeyUsage(
                        digital_signature=True,
                        key_encipherment=True,
                        content_commitment=False,
                        data_encipherment=False,
                        key_agreement=False,
                        key_cert_sign=False,
                        crl_sign=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                )
                
                # Extended key usage
                cert_builder = cert_builder.add_extension(
                    x509.ExtendedKeyUsage([
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]),
                    critical=True,
                )
            
            # Sign certificate
            cert = cert_builder.sign(self.ca_key, hashes.SHA256())
            
            # Save certificate and keys
            cert_path = self.cert_dir / f"{entity_id}.crt"
            key_path = self.cert_dir / f"{entity_id}.key"
            pub_path = self.cert_dir / f"{entity_id}.pub"
            
            with open(cert_path, 'wb') as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            with open(key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            with open(pub_path, 'wb') as f:
                f.write(private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            # Set permissions
            os.chmod(key_path, 0o600)
            
            # Create security identity
            identity = SecurityIdentity(
                entity_id=entity_id,
                entity_type=entity_type,
                certificate_path=str(cert_path),
                private_key_path=str(key_path),
                public_key_path=str(pub_path),
                security_level=SecurityLevel.MEDIUM,
                permissions=[],
                metadata={
                    "cert_type": cert_type.value,
                    "dns_names": dns_names or [],
                    "ip_addresses": ip_addresses or []
                },
                created_at=cert.not_valid_before,
                expires_at=cert.not_valid_after
            )
            
            self.certificates[entity_id] = identity
            
            logger.info(f"Created certificate for {entity_id}")
            return identity
            
        except Exception as e:
            logger.error(f"Failed to create certificate for {entity_id}: {e}")
            raise
    
    async def rotate_certificate(self, entity_id: str) -> SecurityIdentity:
        """Rotate certificate for entity"""
        try:
            old_identity = self.certificates.get(entity_id)
            if not old_identity:
                raise ValueError(f"No certificate found for entity {entity_id}")
            
            # Create new certificate with same parameters
            dns_names = old_identity.metadata.get("dns_names", [])
            ip_addresses = old_identity.metadata.get("ip_addresses", [])
            cert_type = CertificateType(old_identity.metadata.get("cert_type", "client"))
            
            new_identity = await self.create_certificate(
                entity_id=entity_id,
                entity_type=old_identity.entity_type,
                cert_type=cert_type,
                dns_names=dns_names,
                ip_addresses=ip_addresses
            )
            
            # Archive old certificate
            old_cert_path = Path(old_identity.certificate_path)
            old_key_path = Path(old_identity.private_key_path)
            
            archive_dir = self.cert_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            old_cert_path.rename(archive_dir / f"{entity_id}_{timestamp}.crt")
            old_key_path.rename(archive_dir / f"{entity_id}_{timestamp}.key")
            
            logger.info(f"Rotated certificate for {entity_id}")
            return new_identity
            
        except Exception as e:
            logger.error(f"Failed to rotate certificate for {entity_id}: {e}")
            raise
    
    async def revoke_certificate(self, entity_id: str):
        """Revoke certificate"""
        try:
            if entity_id in self.certificates:
                identity = self.certificates[entity_id]
                
                # Move to revoked directory
                revoked_dir = self.cert_dir / "revoked"
                revoked_dir.mkdir(exist_ok=True)
                
                cert_path = Path(identity.certificate_path)
                key_path = Path(identity.private_key_path)
                
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                cert_path.rename(revoked_dir / f"{entity_id}_{timestamp}.crt")
                key_path.rename(revoked_dir / f"{entity_id}_{timestamp}.key")
                
                del self.certificates[entity_id]
                
                logger.info(f"Revoked certificate for {entity_id}")
                
        except Exception as e:
            logger.error(f"Failed to revoke certificate for {entity_id}: {e}")
            raise
    
    async def verify_certificate(self, cert_data: bytes) -> bool:
        """Verify certificate against CA"""
        try:
            cert = x509.load_pem_x509_certificate(cert_data)
            
            # Check if certificate is signed by our CA
            try:
                self.ca_cert.public_key().verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    cert.signature_hash_algorithm
                )
                return True
            except Exception:
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify certificate: {e}")
            return False
    
    def get_certificate(self, entity_id: str) -> Optional[SecurityIdentity]:
        """Get certificate for entity"""
        return self.certificates.get(entity_id)
    
    def create_ssl_context(self, entity_id: str, server_side: bool = False, 
                          security_level: SecurityLevel = SecurityLevel.HIGH,
                          enable_mtls: bool = True, allowed_ciphers: List[str] = None) -> ssl.SSLContext:
        """Create advanced SSL context for entity"""
        try:
            identity = self.certificates.get(entity_id)
            if not identity:
                raise ValueError(f"No certificate found for entity {entity_id}")
            
            # Create SSL context based on security level
            if security_level == SecurityLevel.CRITICAL:
                # Maximum security - TLS 1.3 only
                context = ssl.SSLContext(ssl.PROTOCOL_TLS)
                context.minimum_version = ssl.TLSVersion.TLSv1_3
                context.maximum_version = ssl.TLSVersion.TLSv1_3
            elif security_level == SecurityLevel.HIGH:
                # High security - TLS 1.2 and 1.3
                context = ssl.SSLContext(ssl.PROTOCOL_TLS)
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                context.maximum_version = ssl.TLSVersion.TLSv1_3
            else:
                # Default security
                context = ssl.create_default_context(
                    ssl.Purpose.CLIENT_AUTH if server_side else ssl.Purpose.SERVER_AUTH
                )
            
            # Load certificate chain
            context.load_cert_chain(identity.certificate_path, identity.private_key_path)
            
            # Load CA certificate
            context.load_verify_locations(str(self.cert_dir / "ca.crt"))
            
            # Configure mutual TLS if enabled
            if enable_mtls:
                context.verify_mode = ssl.CERT_REQUIRED
                if not server_side:
                    context.check_hostname = False
            else:
                context.verify_mode = ssl.CERT_NONE
            
            # Configure cipher suites based on security level
            if security_level == SecurityLevel.CRITICAL:
                # Only allow the most secure ciphers
                context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
            elif security_level == SecurityLevel.HIGH:
                # Allow secure ciphers
                context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:ECDHE+AES256:DHE+AES256:!aNULL:!MD5:!DSS')
            
            # Set custom ciphers if provided
            if allowed_ciphers:
                context.set_ciphers(':'.join(allowed_ciphers))
            
            # Additional security settings
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1
            context.options |= ssl.OP_SINGLE_DH_USE
            context.options |= ssl.OP_SINGLE_ECDH_USE
            context.options |= ssl.OP_NO_COMPRESSION
            
            # Enable OCSP stapling for servers
            if server_side:
                try:
                    context.set_servername_callback(self._servername_callback)
                except AttributeError:
                    pass  # Not available in all Python versions
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to create SSL context for {entity_id}: {e}")
            raise
    
    def _servername_callback(self, socket, server_name, context):
        """SNI callback for SSL context"""
        try:
            # In a production environment, you would select the appropriate
            # certificate based on the server name
            logger.debug(f"SNI callback for server name: {server_name}")
            return None
        except Exception as e:
            logger.error(f"SNI callback error: {e}")
            return None
    
    def create_secure_server_config(self, entity_id: str, bind_address: str, 
                                   port: int, security_level: SecurityLevel = SecurityLevel.HIGH) -> Dict[str, Any]:
        """Create secure server configuration"""
        try:
            identity = self.certificates.get(entity_id)
            if not identity:
                raise ValueError(f"No certificate found for entity {entity_id}")
            
            # Create SSL context
            ssl_context = self.create_ssl_context(entity_id, server_side=True, security_level=security_level)
            
            # Server configuration
            config = {
                'bind_address': bind_address,
                'port': port,
                'ssl_context': ssl_context,
                'ssl_version': ssl.PROTOCOL_TLS,
                'certfile': identity.certificate_path,
                'keyfile': identity.private_key_path,
                'ca_certs': str(self.cert_dir / "ca.crt"),
                'cert_reqs': ssl.CERT_REQUIRED,
                'ssl_options': {
                    'do_handshake_on_connect': True,
                    'suppress_ragged_eofs': True,
                    'ciphers': 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
                },
                'security_level': security_level.value,
                'mtls_enabled': True
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create secure server config: {e}")
            raise
    
    def create_secure_client_config(self, entity_id: str, server_hostname: str,
                                   security_level: SecurityLevel = SecurityLevel.HIGH) -> Dict[str, Any]:
        """Create secure client configuration"""
        try:
            identity = self.certificates.get(entity_id)
            if not identity:
                raise ValueError(f"No certificate found for entity {entity_id}")
            
            # Create SSL context
            ssl_context = self.create_ssl_context(entity_id, server_side=False, security_level=security_level)
            
            # Client configuration
            config = {
                'server_hostname': server_hostname,
                'ssl_context': ssl_context,
                'ssl_version': ssl.PROTOCOL_TLS,
                'certfile': identity.certificate_path,
                'keyfile': identity.private_key_path,
                'ca_certs': str(self.cert_dir / "ca.crt"),
                'cert_reqs': ssl.CERT_REQUIRED,
                'check_hostname': True,
                'ssl_options': {
                    'do_handshake_on_connect': True,
                    'suppress_ragged_eofs': True,
                    'ciphers': 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
                },
                'security_level': security_level.value,
                'mtls_enabled': True
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create secure client config: {e}")
            raise
    
    def validate_certificate_chain(self, entity_id: str) -> Dict[str, Any]:
        """Validate certificate chain"""
        try:
            identity = self.certificates.get(entity_id)
            if not identity:
                raise ValueError(f"No certificate found for entity {entity_id}")
            
            # Load certificate
            with open(identity.certificate_path, 'rb') as f:
                cert_data = f.read()
                cert = x509.load_pem_x509_certificate(cert_data)
            
            # Validation results
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'certificate_info': {
                    'subject': cert.subject.rfc4514_string(),
                    'issuer': cert.issuer.rfc4514_string(),
                    'serial_number': str(cert.serial_number),
                    'not_before': cert.not_valid_before.isoformat(),
                    'not_after': cert.not_valid_after.isoformat(),
                    'signature_algorithm': cert.signature_algorithm_oid._name,
                    'key_size': cert.public_key().key_size if hasattr(cert.public_key(), 'key_size') else 'unknown'
                }
            }
            
            # Check expiration
            now = datetime.utcnow()
            if now > cert.not_valid_after:
                validation_results['valid'] = False
                validation_results['errors'].append('Certificate has expired')
            elif (cert.not_valid_after - now).days < 30:
                validation_results['warnings'].append('Certificate expires within 30 days')
            
            # Check if certificate is valid yet
            if now < cert.not_valid_before:
                validation_results['valid'] = False
                validation_results['errors'].append('Certificate is not yet valid')
            
            # Verify signature against CA
            try:
                self.ca_cert.public_key().verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    cert.signature_hash_algorithm
                )
            except Exception as e:
                validation_results['valid'] = False
                validation_results['errors'].append(f'Certificate signature verification failed: {e}')
            
            # Check key usage
            try:
                key_usage = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.KEY_USAGE).value
                if not key_usage.digital_signature:
                    validation_results['warnings'].append('Certificate does not allow digital signatures')
            except x509.ExtensionNotFound:
                validation_results['warnings'].append('Certificate does not have key usage extension')
            
            # Check extended key usage
            try:
                ext_key_usage = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.EXTENDED_KEY_USAGE).value
                if x509.oid.ExtendedKeyUsageOID.SERVER_AUTH not in ext_key_usage:
                    validation_results['warnings'].append('Certificate may not be suitable for server authentication')
            except x509.ExtensionNotFound:
                validation_results['warnings'].append('Certificate does not have extended key usage extension')
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate certificate chain: {e}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'certificate_info': {}
            }
    
    def get_certificate_fingerprint(self, entity_id: str, algorithm: str = 'sha256') -> Optional[str]:
        """Get certificate fingerprint"""
        try:
            identity = self.certificates.get(entity_id)
            if not identity:
                return None
            
            with open(identity.certificate_path, 'rb') as f:
                cert_data = f.read()
                cert = x509.load_pem_x509_certificate(cert_data)
            
            if algorithm == 'sha256':
                fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
            elif algorithm == 'sha1':
                fingerprint = hashlib.sha1(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
            elif algorithm == 'md5':
                fingerprint = hashlib.md5(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Format as colon-separated hex
            return ':'.join(fingerprint[i:i+2] for i in range(0, len(fingerprint), 2))
            
        except Exception as e:
            logger.error(f"Failed to get certificate fingerprint: {e}")
            return None
    
    def export_certificate_bundle(self, entity_id: str, include_private_key: bool = False) -> Optional[str]:
        """Export certificate bundle"""
        try:
            identity = self.certificates.get(entity_id)
            if not identity:
                return None
            
            bundle_parts = []
            
            # Add certificate
            with open(identity.certificate_path, 'r') as f:
                bundle_parts.append(f.read())
            
            # Add CA certificate
            with open(self.cert_dir / "ca.crt", 'r') as f:
                bundle_parts.append(f.read())
            
            # Add private key if requested
            if include_private_key:
                with open(identity.private_key_path, 'r') as f:
                    bundle_parts.append(f.read())
            
            return '\n'.join(bundle_parts)
            
        except Exception as e:
            logger.error(f"Failed to export certificate bundle: {e}")
            return None

class TokenManager:
    """JWT token management"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.tokens: Dict[str, SecurityToken] = {}
        
    def generate_token(self, entity_id: str, permissions: List[str],
                      expires_in: int = 3600, metadata: Dict[str, Any] = None) -> SecurityToken:
        """Generate JWT token"""
        try:
            token_id = str(uuid.uuid4())
            issued_at = datetime.utcnow()
            expires_at = issued_at + timedelta(seconds=expires_in)
            
            payload = {
                "token_id": token_id,
                "entity_id": entity_id,
                "permissions": permissions,
                "iat": issued_at,
                "exp": expires_at,
                "metadata": metadata or {}
            }
            
            token_value = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            token = SecurityToken(
                token_id=token_id,
                entity_id=entity_id,
                token_type="jwt",
                token_value=token_value,
                permissions=permissions,
                issued_at=issued_at,
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            self.tokens[token_id] = token
            return token
            
        except Exception as e:
            logger.error(f"Failed to generate token for {entity_id}: {e}")
            raise
    
    def verify_token(self, token_value: str) -> Optional[SecurityToken]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token_value, self.secret_key, algorithms=[self.algorithm])
            token_id = payload.get("token_id")
            
            if token_id in self.tokens:
                token = self.tokens[token_id]
                
                # Check if token is expired
                if datetime.utcnow() > token.expires_at:
                    del self.tokens[token_id]
                    return None
                
                return token
            
            return None
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return None
    
    def revoke_token(self, token_id: str):
        """Revoke token"""
        if token_id in self.tokens:
            del self.tokens[token_id]
    
    def revoke_all_tokens(self, entity_id: str):
        """Revoke all tokens for entity"""
        tokens_to_revoke = [
            token_id for token_id, token in self.tokens.items()
            if token.entity_id == entity_id
        ]
        
        for token_id in tokens_to_revoke:
            del self.tokens[token_id]

class AccessController:
    """Access control and authorization"""
    
    def __init__(self):
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.roles: Dict[str, List[str]] = {}
        self.entity_roles: Dict[str, List[str]] = {}
        
    def define_policy(self, resource: str, actions: List[str], 
                     conditions: Dict[str, Any] = None):
        """Define access policy"""
        self.policies[resource] = {
            "actions": actions,
            "conditions": conditions or {}
        }
    
    def create_role(self, role_name: str, permissions: List[str]):
        """Create role with permissions"""
        self.roles[role_name] = permissions
    
    def assign_role(self, entity_id: str, role_name: str):
        """Assign role to entity"""
        if entity_id not in self.entity_roles:
            self.entity_roles[entity_id] = []
        
        if role_name not in self.entity_roles[entity_id]:
            self.entity_roles[entity_id].append(role_name)
    
    def revoke_role(self, entity_id: str, role_name: str):
        """Revoke role from entity"""
        if entity_id in self.entity_roles:
            if role_name in self.entity_roles[entity_id]:
                self.entity_roles[entity_id].remove(role_name)
    
    def check_permission(self, entity_id: str, resource: str, action: str,
                        context: Dict[str, Any] = None) -> bool:
        """Check if entity has permission"""
        try:
            # Get entity roles
            entity_roles = self.entity_roles.get(entity_id, [])
            
            # Get permissions from roles
            permissions = []
            for role in entity_roles:
                if role in self.roles:
                    permissions.extend(self.roles[role])
            
            # Check if permission exists
            required_permission = f"{resource}:{action}"
            
            for permission in permissions:
                if permission == required_permission or permission == f"{resource}:*" or permission == "*":
                    # Check conditions if any
                    if resource in self.policies:
                        policy = self.policies[resource]
                        conditions = policy.get("conditions", {})
                        
                        if conditions and context:
                            if not self._evaluate_conditions(conditions, context):
                                return False
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check permission: {e}")
            return False
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate access conditions"""
        try:
            for condition_key, condition_value in conditions.items():
                if condition_key not in context:
                    return False
                
                if isinstance(condition_value, dict):
                    # Complex condition
                    operator = condition_value.get("operator", "equals")
                    value = condition_value.get("value")
                    
                    if operator == "equals":
                        if context[condition_key] != value:
                            return False
                    elif operator == "in":
                        if context[condition_key] not in value:
                            return False
                    elif operator == "not_in":
                        if context[condition_key] in value:
                            return False
                    elif operator == "greater_than":
                        if context[condition_key] <= value:
                            return False
                    elif operator == "less_than":
                        if context[condition_key] >= value:
                            return False
                else:
                    # Simple condition
                    if context[condition_key] != condition_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate conditions: {e}")
            return False

class SecurityManager:
    """Main security manager"""
    
    def __init__(self):
        self.certificate_manager = CertificateManager()
        self.token_manager = TokenManager()
        self.access_controller = AccessController()
        self.encryption_key = None
        self.audit_log: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize security manager"""
        try:
            # Initialize certificate manager
            await self.certificate_manager.initialize()
            
            # Generate encryption key
            self.encryption_key = Fernet.generate_key()
            
            # Setup default roles and policies
            await self._setup_default_security()
            
            logger.info("Security manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize security manager: {e}")
            raise
    
    async def _setup_default_security(self):
        """Setup default security policies"""
        try:
            # Create default roles
            self.access_controller.create_role("admin", ["*"])
            self.access_controller.create_role("coordinator", [
                "agents:*", "tasks:*", "metrics:read", "config:read"
            ])
            self.access_controller.create_role("agent", [
                "tasks:read", "tasks:update", "metrics:write", "heartbeat:write"
            ])
            self.access_controller.create_role("monitor", [
                "metrics:read", "health:read", "logs:read"
            ])
            
            # Define policies
            self.access_controller.define_policy("agents", [
                "create", "read", "update", "delete", "list"
            ])
            self.access_controller.define_policy("tasks", [
                "create", "read", "update", "delete", "list", "assign"
            ])
            self.access_controller.define_policy("metrics", [
                "read", "write"
            ])
            self.access_controller.define_policy("config", [
                "read", "write"
            ])
            
        except Exception as e:
            logger.error(f"Failed to setup default security: {e}")
            raise
    
    async def create_entity_identity(self, entity_id: str, entity_type: str,
                                   role: str = None, permissions: List[str] = None) -> SecurityIdentity:
        """Create security identity for entity"""
        try:
            # Create certificate
            cert_type = CertificateType.AGENT if entity_type == "agent" else CertificateType.CLIENT
            identity = await self.certificate_manager.create_certificate(
                entity_id=entity_id,
                entity_type=entity_type,
                cert_type=cert_type,
                dns_names=[entity_id, f"{entity_id}.aetherveil.local"],
                ip_addresses=["127.0.0.1"]
            )
            
            # Assign role if specified
            if role:
                self.access_controller.assign_role(entity_id, role)
            
            # Add specific permissions if provided
            if permissions:
                for permission in permissions:
                    identity.permissions.append(permission)
            
            # Log creation
            await self._audit_log("identity_created", {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "role": role
            })
            
            return identity
            
        except Exception as e:
            logger.error(f"Failed to create entity identity for {entity_id}: {e}")
            raise
    
    async def authenticate_entity(self, entity_id: str, 
                                method: AuthenticationMethod,
                                credentials: Dict[str, Any]) -> Optional[SecurityIdentity]:
        """Authenticate entity"""
        try:
            if method == AuthenticationMethod.CERTIFICATE:
                cert_data = credentials.get("certificate")
                if cert_data and await self.certificate_manager.verify_certificate(cert_data):
                    identity = self.certificate_manager.get_certificate(entity_id)
                    if identity:
                        await self._audit_log("authentication_success", {
                            "entity_id": entity_id,
                            "method": method.value
                        })
                        return identity
            
            elif method == AuthenticationMethod.JWT:
                token_value = credentials.get("token")
                if token_value:
                    token = self.token_manager.verify_token(token_value)
                    if token and token.entity_id == entity_id:
                        identity = self.certificate_manager.get_certificate(entity_id)
                        if identity:
                            await self._audit_log("authentication_success", {
                                "entity_id": entity_id,
                                "method": method.value
                            })
                            return identity
            
            # Authentication failed
            await self._audit_log("authentication_failed", {
                "entity_id": entity_id,
                "method": method.value
            })
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to authenticate entity {entity_id}: {e}")
            return None
    
    async def authorize_action(self, entity_id: str, resource: str, action: str,
                             context: Dict[str, Any] = None) -> bool:
        """Authorize action for entity"""
        try:
            authorized = self.access_controller.check_permission(
                entity_id, resource, action, context
            )
            
            await self._audit_log("authorization_check", {
                "entity_id": entity_id,
                "resource": resource,
                "action": action,
                "authorized": authorized
            })
            
            return authorized
            
        except Exception as e:
            logger.error(f"Failed to authorize action for {entity_id}: {e}")
            return False
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data"""
        try:
            fernet = Fernet(self.encryption_key)
            return fernet.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        try:
            fernet = Fernet(self.encryption_key)
            return fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def get_encryption_key(self) -> bytes:
        """Get encryption key"""
        return self.encryption_key
    
    async def _audit_log(self, event_type: str, data: Dict[str, Any]):
        """Log security event"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "data": data
            }
            
            self.audit_log.append(log_entry)
            
            # Keep only last 10000 entries
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-10000:]
            
            logger.info(f"Security event: {event_type} - {data}")
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    async def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        return self.audit_log[-limit:]
    
    async def rotate_encryption_key(self):
        """Rotate encryption key"""
        try:
            # Generate new key
            new_key = Fernet.generate_key()
            
            # Archive old key
            old_key = self.encryption_key
            self.encryption_key = new_key
            
            await self._audit_log("encryption_key_rotated", {
                "old_key_hash": hashlib.sha256(old_key).hexdigest(),
                "new_key_hash": hashlib.sha256(new_key).hexdigest()
            })
            
            logger.info("Encryption key rotated")
            
        except Exception as e:
            logger.error(f"Failed to rotate encryption key: {e}")
            raise
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get security status"""
        try:
            return {
                "certificates_count": len(self.certificate_manager.certificates),
                "active_tokens": len(self.token_manager.tokens),
                "roles_count": len(self.access_controller.roles),
                "policies_count": len(self.access_controller.policies),
                "audit_log_entries": len(self.audit_log),
                "ca_certificate_expires": self.certificate_manager.ca_cert.not_valid_after.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get security status: {e}")
            return {"error": str(e)}

# Global security manager instance
security_manager = SecurityManager()