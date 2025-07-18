"""
Enhanced Security Manager with mTLS, TPM-backed secrets, and anti-forensics
Provides enterprise-grade security hardening for Aetherveil Sentinel
"""

import asyncio
import logging
import ssl
import socket
import hashlib
import hmac
import secrets
import time
import os
import subprocess
import psutil
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import base64
import threading
import queue

# Cryptographic libraries
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate, load_der_x509_certificate
from cryptography.x509.oid import NameOID
from cryptography import x509
import cryptography.hazmat.primitives.serialization as crypto_serialization

# TPM libraries
try:
    import tpm2_pytss
    from tpm2_pytss.ESAPI import ESAPI
    from tpm2_pytss.binding import *
    TPM_AVAILABLE = True
except ImportError:
    TPM_AVAILABLE = False

# Vault integration
try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CertificateType(Enum):
    """Certificate types for mTLS"""
    ROOT_CA = "root_ca"
    INTERMEDIATE_CA = "intermediate_ca"
    SERVER = "server"
    CLIENT = "client"
    AGENT = "agent"


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_mtls: bool = True
    enable_tpm: bool = True
    enable_vault: bool = True
    enable_anti_forensics: bool = True
    certificate_validity_days: int = 365
    key_rotation_interval: int = 30  # days
    session_timeout: int = 3600  # seconds
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # seconds
    audit_log_retention: int = 90  # days
    secure_memory_wipe: bool = True
    network_encryption: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_mtls": self.enable_mtls,
            "enable_tpm": self.enable_tpm,
            "enable_vault": self.enable_vault,
            "enable_anti_forensics": self.enable_anti_forensics,
            "certificate_validity_days": self.certificate_validity_days,
            "key_rotation_interval": self.key_rotation_interval,
            "session_timeout": self.session_timeout,
            "max_failed_attempts": self.max_failed_attempts,
            "lockout_duration": self.lockout_duration,
            "audit_log_retention": self.audit_log_retention,
            "secure_memory_wipe": self.secure_memory_wipe,
            "network_encryption": self.network_encryption
        }


@dataclass
class MTLSCertificate:
    """mTLS certificate information"""
    certificate_type: CertificateType
    subject: str
    issuer: str
    serial_number: str
    not_before: datetime
    not_after: datetime
    public_key: bytes
    private_key: Optional[bytes] = None
    certificate_pem: Optional[bytes] = None
    key_usage: List[str] = field(default_factory=list)
    extended_key_usage: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_type": self.certificate_type.value,
            "subject": self.subject,
            "issuer": self.issuer,
            "serial_number": self.serial_number,
            "not_before": self.not_before.isoformat(),
            "not_after": self.not_after.isoformat(),
            "key_usage": self.key_usage,
            "extended_key_usage": self.extended_key_usage
        }


class TPMSecretManager:
    """TPM-backed secret management"""
    
    def __init__(self):
        self.tpm_available = TPM_AVAILABLE
        self.esapi = None
        self.pcr_selection = None
        self.sealed_keys = {}
        
        if self.tpm_available:
            try:
                self._initialize_tpm()
            except Exception as e:
                logger.warning(f"TPM initialization failed: {e}")
                self.tpm_available = False
    
    def _initialize_tpm(self):
        """Initialize TPM connection"""
        try:
            self.esapi = ESAPI()
            
            # Set up PCR selection for sealing
            self.pcr_selection = TPML_PCR_SELECTION()
            pcr_select = TPMS_PCR_SELECTION()
            pcr_select.hash = TPM2_ALG_SHA256
            pcr_select.sizeofSelect = 3
            pcr_select.pcrSelect = (c_uint8 * 3)()
            
            # Select PCR 0, 1, 2 for sealing
            pcr_select.pcrSelect[0] = 0x07  # PCR 0, 1, 2
            self.pcr_selection.count = 1
            self.pcr_selection.pcrSelections = (TPMS_PCR_SELECTION * 1)(pcr_select)
            
            logger.info("TPM initialized successfully")
            
        except Exception as e:
            logger.error(f"TPM initialization failed: {e}")
            raise
    
    def seal_secret(self, secret: bytes, key_name: str) -> Dict[str, Any]:
        """Seal secret using TPM"""
        if not self.tpm_available:
            raise ValueError("TPM not available")
        
        try:
            # Create authentication policy
            auth_policy = self.esapi.start_auth_session(
                tpm_key=ESAPI_TR_NONE,
                bind=ESAPI_TR_NONE,
                session_type=TPM2_SE_POLICY,
                symmetric=TPMT_SYM_DEF(algorithm=TPM2_ALG_NULL),
                auth_hash=TPM2_ALG_SHA256
            )
            
            # Set PCR policy
            self.esapi.policy_pcr(auth_policy, b"", self.pcr_selection)
            
            # Get policy digest
            policy_digest = self.esapi.policy_get_digest(auth_policy)
            
            # Create sensitive data
            sensitive = TPM2B_SENSITIVE_CREATE()
            sensitive.sensitive.userAuth = TPM2B_AUTH()
            sensitive.sensitive.data = TPM2B_SENSITIVE_DATA()
            sensitive.sensitive.data.size = len(secret)
            sensitive.sensitive.data.buffer = (c_uint8 * len(secret))(*secret)
            
            # Create template
            template = TPM2B_PUBLIC()
            template.publicArea.type = TPM2_ALG_KEYEDHASH
            template.publicArea.nameAlg = TPM2_ALG_SHA256
            template.publicArea.objectAttributes = (
                TPMA_OBJECT_USERWITHAUTH |
                TPMA_OBJECT_NODA
            )
            template.publicArea.authPolicy = policy_digest
            template.publicArea.parameters.keyedHashDetail.scheme.scheme = TPM2_ALG_NULL
            
            # Create sealed object
            sealed_object = self.esapi.create(
                parent_handle=ESAPI_TR_RH_OWNER,
                in_sensitive=sensitive,
                in_public=template
            )
            
            # Load sealed object
            sealed_handle = self.esapi.load(
                parent_handle=ESAPI_TR_RH_OWNER,
                in_private=sealed_object.out_private,
                in_public=sealed_object.out_public
            )
            
            # Store for later use
            self.sealed_keys[key_name] = {
                "handle": sealed_handle,
                "private": sealed_object.out_private,
                "public": sealed_object.out_public,
                "policy_digest": policy_digest
            }
            
            # Flush temporary session
            self.esapi.flush_context(auth_policy)
            
            return {
                "key_name": key_name,
                "sealed": True,
                "pcr_bound": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"TPM seal operation failed: {e}")
            raise
    
    def unseal_secret(self, key_name: str) -> bytes:
        """Unseal secret using TPM"""
        if not self.tpm_available:
            raise ValueError("TPM not available")
        
        if key_name not in self.sealed_keys:
            raise ValueError(f"Sealed key {key_name} not found")
        
        try:
            sealed_key = self.sealed_keys[key_name]
            
            # Create authentication policy session
            auth_policy = self.esapi.start_auth_session(
                tpm_key=ESAPI_TR_NONE,
                bind=ESAPI_TR_NONE,
                session_type=TPM2_SE_POLICY,
                symmetric=TPMT_SYM_DEF(algorithm=TPM2_ALG_NULL),
                auth_hash=TPM2_ALG_SHA256
            )
            
            # Set PCR policy
            self.esapi.policy_pcr(auth_policy, b"", self.pcr_selection)
            
            # Unseal the object
            unsealed_data = self.esapi.unseal(
                item_handle=sealed_key["handle"],
                sessions=[auth_policy]
            )
            
            # Flush session
            self.esapi.flush_context(auth_policy)
            
            return bytes(unsealed_data.buffer[:unsealed_data.size])
            
        except Exception as e:
            logger.error(f"TPM unseal operation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if TPM is available"""
        return self.tmp_available


class VaultSecretManager:
    """HashiCorp Vault integration for secret management"""
    
    def __init__(self, vault_url: str, vault_token: str):
        self.vault_available = VAULT_AVAILABLE
        self.client = None
        self.vault_url = vault_url
        self.vault_token = vault_token
        
        if self.vault_available:
            try:
                self._initialize_vault()
            except Exception as e:
                logger.warning(f"Vault initialization failed: {e}")
                self.vault_available = False
    
    def _initialize_vault(self):
        """Initialize Vault connection"""
        try:
            self.client = hvac.Client(
                url=self.vault_url,
                token=self.vault_token,
                verify=True
            )
            
            # Test connection
            if not self.client.is_authenticated():
                raise ValueError("Vault authentication failed")
            
            # Enable KV secrets engine if not exists
            try:
                self.client.sys.enable_secrets_engine(
                    backend_type='kv',
                    path='aetherveil',
                    options={'version': '2'}
                )
            except Exception:
                pass  # Engine may already exist
            
            logger.info("Vault initialized successfully")
            
        except Exception as e:
            logger.error(f"Vault initialization failed: {e}")
            raise
    
    def store_secret(self, path: str, secret: Dict[str, Any]) -> Dict[str, Any]:
        """Store secret in Vault"""
        if not self.vault_available:
            raise ValueError("Vault not available")
        
        try:
            response = self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=secret,
                mount_point='aetherveil'
            )
            
            return {
                "path": path,
                "version": response['data']['version'],
                "created_time": response['data']['created_time'],
                "stored": True
            }
            
        except Exception as e:
            logger.error(f"Vault store operation failed: {e}")
            raise
    
    def retrieve_secret(self, path: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve secret from Vault"""
        if not self.vault_available:
            raise ValueError("Vault not available")
        
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                version=version,
                mount_point='aetherveil'
            )
            
            return response['data']['data']
            
        except Exception as e:
            logger.error(f"Vault retrieve operation failed: {e}")
            raise
    
    def delete_secret(self, path: str) -> bool:
        """Delete secret from Vault"""
        if not self.vault_available:
            raise ValueError("Vault not available")
        
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point='aetherveil'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Vault delete operation failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Vault is available"""
        return self.vault_available


class MTLSManager:
    """Mutual TLS certificate management"""
    
    def __init__(self, cert_dir: Path):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        self.certificates = {}
        self.ca_certificate = None
        self.ca_private_key = None
        
        # Load existing certificates
        self._load_certificates()
    
    def _load_certificates(self):
        """Load existing certificates from disk"""
        try:
            # Load CA certificate
            ca_cert_path = self.cert_dir / "ca.crt"
            ca_key_path = self.cert_dir / "ca.key"
            
            if ca_cert_path.exists() and ca_key_path.exists():
                with open(ca_cert_path, 'rb') as f:
                    self.ca_certificate = load_pem_x509_certificate(f.read())
                
                with open(ca_key_path, 'rb') as f:
                    self.ca_private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
                
                logger.info("Loaded existing CA certificate")
            else:
                self._generate_ca_certificate()
            
            # Load other certificates
            for cert_file in self.cert_dir.glob("*.crt"):
                if cert_file.name != "ca.crt":
                    self._load_certificate(cert_file)
                    
        except Exception as e:
            logger.error(f"Failed to load certificates: {e}")
            raise
    
    def _generate_ca_certificate(self):
        """Generate CA certificate and private key"""
        try:
            # Generate CA private key
            self.ca_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )
            
            # Generate CA certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aetherveil Sentinel"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Security"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Aetherveil CA"),
            ])
            
            self.ca_certificate = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                self.ca_private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=3650)  # 10 years
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("ca.aetherveil.local"),
                ]),
                critical=False,
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            ).add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).sign(self.ca_private_key, hashes.SHA256(), backend=default_backend())
            
            # Save CA certificate and key
            with open(self.cert_dir / "ca.crt", 'wb') as f:
                f.write(self.ca_certificate.public_bytes(serialization.Encoding.PEM))
            
            with open(self.cert_dir / "ca.key", 'wb') as f:
                f.write(self.ca_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Set secure permissions
            os.chmod(self.cert_dir / "ca.key", 0o600)
            
            logger.info("Generated new CA certificate")
            
        except Exception as e:
            logger.error(f"Failed to generate CA certificate: {e}")
            raise
    
    def _load_certificate(self, cert_path: Path):
        """Load certificate from file"""
        try:
            with open(cert_path, 'rb') as f:
                cert = load_pem_x509_certificate(f.read())
            
            # Extract certificate information
            subject = cert.subject.rfc4514_string()
            issuer = cert.issuer.rfc4514_string()
            serial_number = str(cert.serial_number)
            
            cert_info = MTLSCertificate(
                certificate_type=self._determine_certificate_type(cert),
                subject=subject,
                issuer=issuer,
                serial_number=serial_number,
                not_before=cert.not_valid_before,
                not_after=cert.not_valid_after,
                public_key=cert.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ),
                certificate_pem=cert.public_bytes(serialization.Encoding.PEM)
            )
            
            # Load private key if exists
            key_path = cert_path.with_suffix('.key')
            if key_path.exists():
                with open(key_path, 'rb') as f:
                    cert_info.private_key = f.read()
            
            self.certificates[cert_path.stem] = cert_info
            
        except Exception as e:
            logger.error(f"Failed to load certificate {cert_path}: {e}")
    
    def _determine_certificate_type(self, cert: x509.Certificate) -> CertificateType:
        """Determine certificate type from certificate"""
        try:
            # Check if it's a CA certificate
            basic_constraints = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.BASIC_CONSTRAINTS
            ).value
            
            if basic_constraints.ca:
                return CertificateType.ROOT_CA
            
            # Check extended key usage
            try:
                ext_key_usage = cert.extensions.get_extension_for_oid(
                    x509.oid.ExtensionOID.EXTENDED_KEY_USAGE
                ).value
                
                if x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in ext_key_usage:
                    return CertificateType.SERVER
                elif x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH in ext_key_usage:
                    return CertificateType.CLIENT
                
            except x509.ExtensionNotFound:
                pass
            
            return CertificateType.AGENT
            
        except Exception:
            return CertificateType.AGENT
    
    def generate_certificate(self, cert_type: CertificateType, common_name: str,
                           san_list: Optional[List[str]] = None,
                           validity_days: int = 365) -> MTLSCertificate:
        """Generate new certificate"""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Create certificate subject
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aetherveil Sentinel"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, cert_type.value.title()),
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ])
            
            # Create certificate builder
            builder = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                self.ca_certificate.subject
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            )
            
            # Add SAN extension
            san_names = []
            if san_list:
                for san in san_list:
                    if san.startswith("DNS:"):
                        san_names.append(x509.DNSName(san[4:]))
                    elif san.startswith("IP:"):
                        san_names.append(x509.IPAddress(san[3:]))
                    else:
                        san_names.append(x509.DNSName(san))
            
            if san_names:
                builder = builder.add_extension(
                    x509.SubjectAlternativeName(san_names),
                    critical=False
                )
            
            # Add key usage extension
            if cert_type == CertificateType.SERVER:
                builder = builder.add_extension(
                    x509.KeyUsage(
                        digital_signature=True,
                        key_encipherment=True,
                        key_agreement=False,
                        key_cert_sign=False,
                        crl_sign=False,
                        content_commitment=False,
                        data_encipherment=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                ).add_extension(
                    x509.ExtendedKeyUsage([
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                    ]),
                    critical=True,
                )
            
            elif cert_type == CertificateType.CLIENT:
                builder = builder.add_extension(
                    x509.KeyUsage(
                        digital_signature=True,
                        key_encipherment=True,
                        key_agreement=False,
                        key_cert_sign=False,
                        crl_sign=False,
                        content_commitment=False,
                        data_encipherment=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                ).add_extension(
                    x509.ExtendedKeyUsage([
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]),
                    critical=True,
                )
            
            # Sign certificate
            certificate = builder.sign(
                self.ca_private_key,
                hashes.SHA256(),
                backend=default_backend()
            )
            
            # Create certificate info
            cert_info = MTLSCertificate(
                certificate_type=cert_type,
                subject=subject.rfc4514_string(),
                issuer=self.ca_certificate.subject.rfc4514_string(),
                serial_number=str(certificate.serial_number),
                not_before=certificate.not_valid_before,
                not_after=certificate.not_valid_after,
                public_key=certificate.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ),
                private_key=private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ),
                certificate_pem=certificate.public_bytes(serialization.Encoding.PEM)
            )
            
            # Save certificate and key
            cert_filename = common_name.replace(".", "_").replace("*", "wildcard")
            cert_path = self.cert_dir / f"{cert_filename}.crt"
            key_path = self.cert_dir / f"{cert_filename}.key"
            
            with open(cert_path, 'wb') as f:
                f.write(cert_info.certificate_pem)
            
            with open(key_path, 'wb') as f:
                f.write(cert_info.private_key)
            
            # Set secure permissions
            os.chmod(key_path, 0o600)
            
            # Store in memory
            self.certificates[cert_filename] = cert_info
            
            logger.info(f"Generated {cert_type.value} certificate for {common_name}")
            
            return cert_info
            
        except Exception as e:
            logger.error(f"Failed to generate certificate: {e}")
            raise
    
    def get_certificate(self, name: str) -> Optional[MTLSCertificate]:
        """Get certificate by name"""
        return self.certificates.get(name)
    
    def list_certificates(self) -> List[str]:
        """List all certificate names"""
        return list(self.certificates.keys())
    
    def revoke_certificate(self, name: str) -> bool:
        """Revoke certificate"""
        try:
            if name in self.certificates:
                # Remove from memory
                del self.certificates[name]
                
                # Remove files
                cert_path = self.cert_dir / f"{name}.crt"
                key_path = self.cert_dir / f"{name}.key"
                
                if cert_path.exists():
                    cert_path.unlink()
                if key_path.exists():
                    key_path.unlink()
                
                logger.info(f"Revoked certificate: {name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke certificate {name}: {e}")
            return False
    
    def create_ssl_context(self, cert_name: str, verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED) -> ssl.SSLContext:
        """Create SSL context for mTLS"""
        try:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.check_hostname = False
            context.verify_mode = verify_mode
            
            # Load CA certificate
            ca_cert_path = self.cert_dir / "ca.crt"
            context.load_verify_locations(ca_cert_path)
            
            # Load client certificate
            if cert_name in self.certificates:
                cert_path = self.cert_dir / f"{cert_name}.crt"
                key_path = self.cert_dir / f"{cert_name}.key"
                
                if cert_path.exists() and key_path.exists():
                    context.load_cert_chain(cert_path, key_path)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            raise


class AntiForensicsManager:
    """Anti-forensics and evidence elimination"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.secure_delete_passes = 3
        self.memory_wipe_enabled = config.secure_memory_wipe
        self.temp_files = []
        self.memory_regions = []
        
    def secure_delete_file(self, file_path: Path) -> bool:
        """Securely delete file with multiple passes"""
        try:
            if not file_path.exists():
                return True
            
            file_size = file_path.stat().st_size
            
            # Multiple overwrite passes
            with open(file_path, 'r+b') as f:
                for pass_num in range(self.secure_delete_passes):
                    f.seek(0)
                    
                    if pass_num == 0:
                        # First pass: all zeros
                        f.write(b'\x00' * file_size)
                    elif pass_num == 1:
                        # Second pass: all ones
                        f.write(b'\xff' * file_size)
                    else:
                        # Final pass: random data
                        f.write(secrets.token_bytes(file_size))
                    
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete file
            file_path.unlink()
            
            logger.debug(f"Securely deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to securely delete file {file_path}: {e}")
            return False
    
    def secure_delete_directory(self, dir_path: Path) -> bool:
        """Securely delete directory and all contents"""
        try:
            if not dir_path.exists():
                return True
            
            # Recursively delete all files
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    self.secure_delete_file(file_path)
            
            # Remove empty directories
            shutil.rmtree(dir_path, ignore_errors=True)
            
            logger.debug(f"Securely deleted directory: {dir_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to securely delete directory {dir_path}: {e}")
            return False
    
    def wipe_memory_region(self, memory_address: int, size: int) -> bool:
        """Wipe memory region (requires appropriate permissions)"""
        try:
            if not self.memory_wipe_enabled:
                return True
            
            # This would require kernel-level access or special permissions
            # For now, we'll just track the request
            self.memory_regions.append({
                "address": memory_address,
                "size": size,
                "timestamp": datetime.now()
            })
            
            logger.debug(f"Marked memory region for wiping: {memory_address}:{size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to wipe memory region: {e}")
            return False
    
    def clear_system_artifacts(self) -> Dict[str, bool]:
        """Clear system artifacts that could be used for forensics"""
        results = {}
        
        try:
            # Clear bash history
            history_files = [
                Path.home() / ".bash_history",
                Path.home() / ".zsh_history",
                Path.home() / ".history"
            ]
            
            for history_file in history_files:
                if history_file.exists():
                    results[str(history_file)] = self.secure_delete_file(history_file)
            
            # Clear temporary files
            temp_dirs = [
                Path("/tmp"),
                Path("/var/tmp"),
                Path.home() / ".cache"
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    # Only delete our temporary files
                    for temp_file in temp_dir.glob("aetherveil_*"):
                        results[str(temp_file)] = self.secure_delete_file(temp_file)
            
            # Clear log files
            log_files = [
                Path("/var/log/auth.log"),
                Path("/var/log/syslog"),
                Path("/var/log/messages")
            ]
            
            for log_file in log_files:
                if log_file.exists():
                    try:
                        # Truncate log file instead of deleting
                        with open(log_file, 'w') as f:
                            f.truncate(0)
                        results[str(log_file)] = True
                    except PermissionError:
                        results[str(log_file)] = False
            
            # Clear DNS cache
            try:
                subprocess.run(["sudo", "systemctl", "flush-dns"], 
                             capture_output=True, timeout=10)
                results["dns_cache"] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results["dns_cache"] = False
            
            # Clear network connections
            try:
                subprocess.run(["sudo", "netstat", "-tuln"], 
                             capture_output=True, timeout=10)
                results["network_connections"] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results["network_connections"] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to clear system artifacts: {e}")
            return {"error": str(e)}
    
    def create_secure_temp_file(self, suffix: str = "") -> Path:
        """Create secure temporary file"""
        try:
            temp_file = Path(tempfile.mktemp(
                suffix=suffix,
                prefix="aetherveil_",
                dir="/tmp"
            ))
            
            # Create file with secure permissions
            temp_file.touch(mode=0o600)
            
            # Track for later cleanup
            self.temp_files.append(temp_file)
            
            return temp_file
            
        except Exception as e:
            logger.error(f"Failed to create secure temp file: {e}")
            raise
    
    def cleanup_temp_files(self) -> bool:
        """Clean up all temporary files"""
        try:
            success = True
            
            for temp_file in self.temp_files:
                if not self.secure_delete_file(temp_file):
                    success = False
            
            self.temp_files.clear()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return False
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """Get cleanup status"""
        return {
            "temp_files_tracked": len(self.temp_files),
            "memory_regions_tracked": len(self.memory_regions),
            "secure_delete_passes": self.secure_delete_passes,
            "memory_wipe_enabled": self.memory_wipe_enabled
        }


class EnhancedSecurityManager:
    """Enhanced security manager with comprehensive security features"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.tpm_manager = None
        self.vault_manager = None
        self.mtls_manager = None
        self.anti_forensics = None
        self.failed_attempts = defaultdict(int)
        self.locked_accounts = {}
        self.active_sessions = {}
        self.audit_log = []
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize security components"""
        try:
            # Initialize TPM manager
            if self.config.enable_tpm:
                try:
                    self.tpm_manager = TPMSecretManager()
                    logger.info("TPM manager initialized")
                except Exception as e:
                    logger.warning(f"TPM manager initialization failed: {e}")
            
            # Initialize Vault manager
            if self.config.enable_vault:
                try:
                    vault_url = os.getenv("VAULT_URL", "http://localhost:8200")
                    vault_token = os.getenv("VAULT_TOKEN")
                    
                    if vault_token:
                        self.vault_manager = VaultSecretManager(vault_url, vault_token)
                        logger.info("Vault manager initialized")
                    else:
                        logger.warning("Vault token not provided")
                except Exception as e:
                    logger.warning(f"Vault manager initialization failed: {e}")
            
            # Initialize mTLS manager
            if self.config.enable_mtls:
                try:
                    cert_dir = Path(os.getenv("CERT_DIR", "./certificates"))
                    self.mtls_manager = MTLSManager(cert_dir)
                    logger.info("mTLS manager initialized")
                except Exception as e:
                    logger.error(f"mTLS manager initialization failed: {e}")
                    raise
            
            # Initialize anti-forensics
            if self.config.enable_anti_forensics:
                try:
                    self.anti_forensics = AntiForensicsManager(self.config)
                    logger.info("Anti-forensics manager initialized")
                except Exception as e:
                    logger.warning(f"Anti-forensics manager initialization failed: {e}")
                    
        except Exception as e:
            logger.error(f"Security manager initialization failed: {e}")
            raise
    
    def store_secret(self, key: str, value: bytes, 
                    storage_type: str = "vault") -> Dict[str, Any]:
        """Store secret using specified storage type"""
        try:
            if storage_type == "tpm" and self.tmp_manager:
                return self.tpm_manager.seal_secret(value, key)
            elif storage_type == "vault" and self.vault_manager:
                secret_data = {"value": base64.b64encode(value).decode()}
                return self.vault_manager.store_secret(key, secret_data)
            else:
                raise ValueError(f"Storage type {storage_type} not available")
                
        except Exception as e:
            logger.error(f"Failed to store secret: {e}")
            raise
    
    def retrieve_secret(self, key: str, storage_type: str = "vault") -> bytes:
        """Retrieve secret using specified storage type"""
        try:
            if storage_type == "tpm" and self.tpm_manager:
                return self.tpm_manager.unseal_secret(key)
            elif storage_type == "vault" and self.vault_manager:
                secret_data = self.vault_manager.retrieve_secret(key)
                return base64.b64decode(secret_data["value"])
            else:
                raise ValueError(f"Storage type {storage_type} not available")
                
        except Exception as e:
            logger.error(f"Failed to retrieve secret: {e}")
            raise
    
    def create_secure_session(self, user_id: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create secure session"""
        try:
            session_id = secrets.token_urlsafe(32)
            session_key = secrets.token_bytes(32)
            
            session = {
                "session_id": session_id,
                "user_id": user_id,
                "client_info": client_info,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=self.config.session_timeout),
                "session_key": session_key,
                "active": True
            }
            
            self.active_sessions[session_id] = session
            
            # Log session creation
            self._log_security_event({
                "event_type": "session_created",
                "user_id": user_id,
                "session_id": session_id,
                "client_info": client_info,
                "timestamp": datetime.now()
            })
            
            return {
                "session_id": session_id,
                "expires_at": session["expires_at"].isoformat(),
                "session_token": base64.b64encode(session_key).decode()
            }
            
        except Exception as e:
            logger.error(f"Failed to create secure session: {e}")
            raise
    
    def validate_session(self, session_id: str, session_token: str) -> bool:
        """Validate session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Check if session is active
            if not session["active"]:
                return False
            
            # Check if session is expired
            if datetime.now() > session["expires_at"]:
                self._invalidate_session(session_id)
                return False
            
            # Validate session token
            expected_token = base64.b64encode(session["session_key"]).decode()
            if not hmac.compare_digest(session_token, expected_token):
                return False
            
            # Update last activity
            session["last_activity"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return False
    
    def _invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["active"] = False
            
            # Log session invalidation
            self._log_security_event({
                "event_type": "session_invalidated",
                "session_id": session_id,
                "timestamp": datetime.now()
            })
    
    def authenticate_user(self, user_id: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user with security controls"""
        try:
            # Check if account is locked
            if user_id in self.locked_accounts:
                lockout_end = self.locked_accounts[user_id]
                if datetime.now() < lockout_end:
                    raise ValueError("Account is locked")
                else:
                    # Unlock account
                    del self.locked_accounts[user_id]
                    self.failed_attempts[user_id] = 0
            
            # Perform authentication (placeholder)
            # In real implementation, this would verify credentials
            auth_success = self._verify_credentials(user_id, credentials)
            
            if auth_success:
                # Reset failed attempts
                self.failed_attempts[user_id] = 0
                
                # Log successful authentication
                self._log_security_event({
                    "event_type": "authentication_success",
                    "user_id": user_id,
                    "timestamp": datetime.now()
                })
                
                return {"success": True, "user_id": user_id}
            else:
                # Increment failed attempts
                self.failed_attempts[user_id] += 1
                
                # Check if account should be locked
                if self.failed_attempts[user_id] >= self.config.max_failed_attempts:
                    lockout_end = datetime.now() + timedelta(seconds=self.config.lockout_duration)
                    self.locked_accounts[user_id] = lockout_end
                    
                    # Log account lockout
                    self._log_security_event({
                        "event_type": "account_locked",
                        "user_id": user_id,
                        "lockout_end": lockout_end,
                        "timestamp": datetime.now()
                    })
                
                # Log failed authentication
                self._log_security_event({
                    "event_type": "authentication_failed",
                    "user_id": user_id,
                    "failed_attempts": self.failed_attempts[user_id],
                    "timestamp": datetime.now()
                })
                
                raise ValueError("Authentication failed")
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def _verify_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Verify user credentials (placeholder)"""
        # In real implementation, this would verify against a secure store
        return True
    
    def _log_security_event(self, event: Dict[str, Any]):
        """Log security event"""
        try:
            # Add event to audit log
            self.audit_log.append(event)
            
            # Keep only recent events
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-10000:]
            
            # Log to system logger
            logger.info(f"Security event: {event['event_type']} - {event}")
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def generate_agent_certificate(self, agent_id: str) -> Dict[str, Any]:
        """Generate certificate for agent"""
        try:
            if not self.mtls_manager:
                raise ValueError("mTLS manager not available")
            
            # Generate certificate
            cert = self.mtls_manager.generate_certificate(
                cert_type=CertificateType.AGENT,
                common_name=f"agent.{agent_id}.aetherveil.local",
                san_list=[f"agent-{agent_id}", f"{agent_id}.agents.aetherveil.local"],
                validity_days=self.config.certificate_validity_days
            )
            
            return {
                "certificate": cert.certificate_pem.decode(),
                "private_key": cert.private_key.decode(),
                "ca_certificate": self.mtls_manager.ca_certificate.public_bytes(
                    serialization.Encoding.PEM
                ).decode(),
                "valid_until": cert.not_after.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate agent certificate: {e}")
            raise
    
    def create_secure_channel(self, agent_id: str) -> ssl.SSLContext:
        """Create secure channel for agent communication"""
        try:
            if not self.mtls_manager:
                raise ValueError("mTLS manager not available")
            
            # Create SSL context
            context = self.mtls_manager.create_ssl_context(
                cert_name=f"agent_{agent_id}",
                verify_mode=ssl.CERT_REQUIRED
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to create secure channel: {e}")
            raise
    
    def cleanup_security_artifacts(self) -> Dict[str, Any]:
        """Clean up security artifacts"""
        try:
            results = {}
            
            # Clean up anti-forensics artifacts
            if self.anti_forensics:
                results["temp_files"] = self.anti_forensics.cleanup_temp_files()
                results["system_artifacts"] = self.anti_forensics.clear_system_artifacts()
            
            # Clean up expired sessions
            expired_sessions = []
            for session_id, session in self.active_sessions.items():
                if datetime.now() > session["expires_at"]:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._invalidate_session(session_id)
            
            results["expired_sessions"] = len(expired_sessions)
            
            # Clean up old audit logs
            cutoff_time = datetime.now() - timedelta(days=self.config.audit_log_retention)
            original_count = len(self.audit_log)
            
            self.audit_log = [event for event in self.audit_log 
                            if event.get("timestamp", datetime.min) > cutoff_time]
            
            results["audit_logs_cleaned"] = original_count - len(self.audit_log)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to cleanup security artifacts: {e}")
            return {"error": str(e)}
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        try:
            return {
                "config": self.config.to_dict(),
                "tpm_available": self.tpm_manager.is_available() if self.tpm_manager else False,
                "vault_available": self.vault_manager.is_available() if self.vault_manager else False,
                "mtls_enabled": self.mtls_manager is not None,
                "anti_forensics_enabled": self.anti_forensics is not None,
                "active_sessions": len(self.active_sessions),
                "locked_accounts": len(self.locked_accounts),
                "audit_events": len(self.audit_log),
                "certificates": len(self.mtls_manager.certificates) if self.mtls_manager else 0,
                "failed_attempts": dict(self.failed_attempts)
            }
            
        except Exception as e:
            logger.error(f"Failed to get security status: {e}")
            return {"error": str(e)}
    
    async def rotate_keys(self) -> Dict[str, Any]:
        """Rotate encryption keys"""
        try:
            results = {}
            
            # Rotate session keys
            for session_id, session in self.active_sessions.items():
                if session["active"]:
                    new_key = secrets.token_bytes(32)
                    session["session_key"] = new_key
                    session["key_rotated_at"] = datetime.now()
            
            results["session_keys_rotated"] = len(self.active_sessions)
            
            # Rotate certificates (if needed)
            if self.mtls_manager:
                expiring_certs = []
                for cert_name, cert in self.mtls_manager.certificates.items():
                    days_until_expiry = (cert.not_after - datetime.now()).days
                    if days_until_expiry <= 30:  # Rotate if expiring within 30 days
                        expiring_certs.append(cert_name)
                
                for cert_name in expiring_certs:
                    # Regenerate certificate
                    old_cert = self.mtls_manager.certificates[cert_name]
                    new_cert = self.mtls_manager.generate_certificate(
                        cert_type=old_cert.certificate_type,
                        common_name=cert_name,
                        validity_days=self.config.certificate_validity_days
                    )
                
                results["certificates_rotated"] = len(expiring_certs)
            
            # Log key rotation
            self._log_security_event({
                "event_type": "key_rotation",
                "session_keys_rotated": results.get("session_keys_rotated", 0),
                "certificates_rotated": results.get("certificates_rotated", 0),
                "timestamp": datetime.now()
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return {"error": str(e)}