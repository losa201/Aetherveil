"""
Advanced JWT Authentication System for Aetherveil Sentinel
Implements comprehensive JWT token management with refresh tokens, blacklisting, and multi-factor authentication
"""

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import base64

import jwt
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from passlib.context import CryptContext
from passlib.hash import argon2

from coordinator.security_manager import SecurityToken, SecurityLevel, ThreatLevel

logger = logging.getLogger(__name__)

class TokenType(Enum):
    """JWT token types"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    TEMPORARY = "temporary"
    SERVICE = "service"

class AuthenticationFlow(Enum):
    """Authentication flow types"""
    STANDARD = "standard"
    MFA = "mfa"
    OAUTH2 = "oauth2"
    CERTIFICATE = "certificate"
    API_KEY = "api_key"

@dataclass
class JWTConfig:
    """JWT configuration"""
    access_token_expire: int = 900  # 15 minutes
    refresh_token_expire: int = 86400  # 24 hours
    api_key_expire: int = 2592000  # 30 days
    algorithm: str = "RS256"
    issuer: str = "aetherveil-sentinel"
    audience: str = "aetherveil-agents"
    enable_refresh_rotation: bool = True
    max_refresh_uses: int = 5
    blacklist_cleanup_interval: int = 3600  # 1 hour

@dataclass
class TokenClaims:
    """JWT token claims"""
    sub: str  # Subject (user/entity ID)
    iat: int  # Issued at
    exp: int  # Expiration time
    iss: str  # Issuer
    aud: str  # Audience
    jti: str  # JWT ID
    token_type: str  # Token type
    permissions: List[str]  # Permissions
    roles: List[str]  # Roles
    security_level: str  # Security level
    session_id: str  # Session ID
    client_id: str  # Client ID
    metadata: Dict[str, Any]  # Additional metadata

@dataclass
class AuthenticationResult:
    """Authentication result"""
    success: bool
    entity_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    token_type: str = "Bearer"
    permissions: List[str] = None
    roles: List[str] = None
    mfa_required: bool = False
    mfa_token: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class RefreshTokenData:
    """Refresh token data"""
    token_id: str
    entity_id: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    use_count: int
    max_uses: int
    revoked: bool = False
    metadata: Dict[str, Any] = None

class JWTManager:
    """Advanced JWT token manager"""
    
    def __init__(self, config: JWTConfig = None):
        self.config = config or JWTConfig()
        self.private_key: Optional[rsa.RSAPrivateKey] = None
        self.public_key: Optional[rsa.RSAPublicKey] = None
        self.refresh_tokens: Dict[str, RefreshTokenData] = {}
        self.blacklisted_tokens: Set[str] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.password_context = CryptContext(schemes=["argon2"], deprecated="auto")
        self._last_cleanup = time.time()
        
    async def initialize(self, private_key_path: str = None, public_key_path: str = None):
        """Initialize JWT manager"""
        try:
            # Load or generate RSA key pair
            if private_key_path and public_key_path:
                await self._load_key_pair(private_key_path, public_key_path)
            else:
                await self._generate_key_pair()
            
            # Start background cleanup task
            asyncio.create_task(self._cleanup_expired_tokens())
            
            logger.info("JWT manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize JWT manager: {e}")
            raise
    
    async def _load_key_pair(self, private_key_path: str, public_key_path: str):
        """Load RSA key pair from files"""
        try:
            # Load private key
            with open(private_key_path, 'rb') as f:
                private_key_data = f.read()
                self.private_key = serialization.load_pem_private_key(
                    private_key_data, password=None
                )
            
            # Load public key
            with open(public_key_path, 'rb') as f:
                public_key_data = f.read()
                self.public_key = serialization.load_pem_public_key(public_key_data)
            
            logger.info("RSA key pair loaded")
            
        except Exception as e:
            logger.error(f"Failed to load key pair: {e}")
            raise
    
    async def _generate_key_pair(self):
        """Generate RSA key pair"""
        try:
            # Generate private key
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            self.public_key = self.private_key.public_key()
            
            # Save keys to files
            os.makedirs('/app/keys', exist_ok=True)
            
            # Save private key
            with open('/app/keys/jwt_private.pem', 'wb') as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            with open('/app/keys/jwt_public.pem', 'wb') as f:
                f.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            # Set permissions
            os.chmod('/app/keys/jwt_private.pem', 0o600)
            
            logger.info("RSA key pair generated")
            
        except Exception as e:
            logger.error(f"Failed to generate key pair: {e}")
            raise
    
    def create_access_token(self, entity_id: str, permissions: List[str], roles: List[str],
                           security_level: SecurityLevel = SecurityLevel.MEDIUM,
                           session_id: str = None, client_id: str = None,
                           metadata: Dict[str, Any] = None) -> str:
        """Create access token"""
        try:
            now = datetime.utcnow()
            expires = now + timedelta(seconds=self.config.access_token_expire)
            
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            # Create token claims
            claims = TokenClaims(
                sub=entity_id,
                iat=int(now.timestamp()),
                exp=int(expires.timestamp()),
                iss=self.config.issuer,
                aud=self.config.audience,
                jti=str(uuid.uuid4()),
                token_type=TokenType.ACCESS.value,
                permissions=permissions,
                roles=roles,
                security_level=security_level.value,
                session_id=session_id,
                client_id=client_id or entity_id,
                metadata=metadata or {}
            )
            
            # Create JWT
            token = jwt.encode(
                asdict(claims),
                self.private_key,
                algorithm=self.config.algorithm
            )
            
            # Track active session
            self.active_sessions[session_id] = {
                'entity_id': entity_id,
                'created_at': now,
                'last_activity': now,
                'permissions': permissions,
                'roles': roles,
                'security_level': security_level.value
            }
            
            logger.info(f"Created access token for {entity_id}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise
    
    def create_refresh_token(self, entity_id: str, session_id: str,
                            max_uses: int = None) -> str:
        """Create refresh token"""
        try:
            now = datetime.utcnow()
            expires = now + timedelta(seconds=self.config.refresh_token_expire)
            token_id = str(uuid.uuid4())
            
            if max_uses is None:
                max_uses = self.config.max_refresh_uses
            
            # Create refresh token data
            refresh_data = RefreshTokenData(
                token_id=token_id,
                entity_id=entity_id,
                session_id=session_id,
                created_at=now,
                expires_at=expires,
                use_count=0,
                max_uses=max_uses
            )
            
            # Store refresh token
            self.refresh_tokens[token_id] = refresh_data
            
            # Create JWT
            claims = {
                'sub': entity_id,
                'iat': int(now.timestamp()),
                'exp': int(expires.timestamp()),
                'iss': self.config.issuer,
                'aud': self.config.audience,
                'jti': token_id,
                'token_type': TokenType.REFRESH.value,
                'session_id': session_id
            }
            
            token = jwt.encode(claims, self.private_key, algorithm=self.config.algorithm)
            
            logger.info(f"Created refresh token for {entity_id}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise
    
    def create_api_key(self, entity_id: str, permissions: List[str], roles: List[str],
                      name: str = None, expires_in: int = None) -> str:
        """Create API key token"""
        try:
            now = datetime.utcnow()
            expires_in = expires_in or self.config.api_key_expire
            expires = now + timedelta(seconds=expires_in)
            
            # Create token claims
            claims = {
                'sub': entity_id,
                'iat': int(now.timestamp()),
                'exp': int(expires.timestamp()),
                'iss': self.config.issuer,
                'aud': self.config.audience,
                'jti': str(uuid.uuid4()),
                'token_type': TokenType.API_KEY.value,
                'permissions': permissions,
                'roles': roles,
                'name': name or f"API Key {entity_id}",
                'security_level': SecurityLevel.HIGH.value
            }
            
            # Create JWT
            token = jwt.encode(claims, self.private_key, algorithm=self.config.algorithm)
            
            logger.info(f"Created API key for {entity_id}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise
    
    def verify_token(self, token: str, token_type: TokenType = None) -> Optional[TokenClaims]:
        """Verify JWT token"""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience
            )
            
            # Check if token is blacklisted
            jti = payload.get('jti')
            if jti and jti in self.blacklisted_tokens:
                logger.warning(f"Token is blacklisted: {jti}")
                return None
            
            # Verify token type if specified
            if token_type and payload.get('token_type') != token_type.value:
                logger.warning(f"Token type mismatch: expected {token_type.value}, got {payload.get('token_type')}")
                return None
            
            # Create token claims
            claims = TokenClaims(
                sub=payload['sub'],
                iat=payload['iat'],
                exp=payload['exp'],
                iss=payload['iss'],
                aud=payload['aud'],
                jti=payload['jti'],
                token_type=payload.get('token_type', TokenType.ACCESS.value),
                permissions=payload.get('permissions', []),
                roles=payload.get('roles', []),
                security_level=payload.get('security_level', SecurityLevel.MEDIUM.value),
                session_id=payload.get('session_id'),
                client_id=payload.get('client_id'),
                metadata=payload.get('metadata', {})
            )
            
            # Update session activity
            if claims.session_id and claims.session_id in self.active_sessions:
                self.active_sessions[claims.session_id]['last_activity'] = datetime.utcnow()
            
            return claims
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            claims = self.verify_token(refresh_token, TokenType.REFRESH)
            if not claims:
                return None
            
            # Get refresh token data
            refresh_data = self.refresh_tokens.get(claims.jti)
            if not refresh_data:
                logger.warning(f"Refresh token data not found: {claims.jti}")
                return None
            
            # Check if refresh token is revoked
            if refresh_data.revoked:
                logger.warning(f"Refresh token is revoked: {claims.jti}")
                return None
            
            # Check if refresh token has exceeded max uses
            if refresh_data.use_count >= refresh_data.max_uses:
                logger.warning(f"Refresh token has exceeded max uses: {claims.jti}")
                refresh_data.revoked = True
                return None
            
            # Increment use count
            refresh_data.use_count += 1
            
            # Get session data
            session_data = self.active_sessions.get(claims.session_id)
            if not session_data:
                logger.warning(f"Session data not found: {claims.session_id}")
                return None
            
            # Create new access token
            new_access_token = self.create_access_token(
                entity_id=claims.sub,
                permissions=session_data['permissions'],
                roles=session_data['roles'],
                security_level=SecurityLevel(session_data['security_level']),
                session_id=claims.session_id,
                client_id=claims.client_id
            )
            
            # Create new refresh token if rotation is enabled
            new_refresh_token = None
            if self.config.enable_refresh_rotation:
                # Revoke old refresh token
                refresh_data.revoked = True
                
                # Create new refresh token
                new_refresh_token = self.create_refresh_token(
                    entity_id=claims.sub,
                    session_id=claims.session_id,
                    max_uses=refresh_data.max_uses - refresh_data.use_count
                )
            else:
                new_refresh_token = refresh_token
            
            logger.info(f"Refreshed access token for {claims.sub}")
            return new_access_token, new_refresh_token
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return None
    
    def revoke_token(self, token: str):
        """Revoke token (add to blacklist)"""
        try:
            # Decode token to get JTI
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False}  # Don't verify expiration for revocation
            )
            
            jti = payload.get('jti')
            if jti:
                self.blacklisted_tokens.add(jti)
                logger.info(f"Token revoked: {jti}")
            
            # If it's a refresh token, revoke it
            if payload.get('token_type') == TokenType.REFRESH.value:
                refresh_data = self.refresh_tokens.get(jti)
                if refresh_data:
                    refresh_data.revoked = True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
    
    def revoke_session(self, session_id: str):
        """Revoke all tokens for a session"""
        try:
            # Remove session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Revoke all refresh tokens for this session
            for token_data in self.refresh_tokens.values():
                if token_data.session_id == session_id:
                    token_data.revoked = True
            
            logger.info(f"Session revoked: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to revoke session: {e}")
    
    def revoke_all_tokens(self, entity_id: str):
        """Revoke all tokens for an entity"""
        try:
            # Remove all sessions for entity
            sessions_to_remove = [
                session_id for session_id, session_data in self.active_sessions.items()
                if session_data['entity_id'] == entity_id
            ]
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
            
            # Revoke all refresh tokens for entity
            for token_data in self.refresh_tokens.values():
                if token_data.entity_id == entity_id:
                    token_data.revoked = True
            
            logger.info(f"All tokens revoked for entity: {entity_id}")
            
        except Exception as e:
            logger.error(f"Failed to revoke all tokens for entity: {e}")
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token information"""
        try:
            claims = self.verify_token(token)
            if not claims:
                return None
            
            return {
                'entity_id': claims.sub,
                'token_type': claims.token_type,
                'permissions': claims.permissions,
                'roles': claims.roles,
                'security_level': claims.security_level,
                'session_id': claims.session_id,
                'client_id': claims.client_id,
                'issued_at': datetime.fromtimestamp(claims.iat),
                'expires_at': datetime.fromtimestamp(claims.exp),
                'metadata': claims.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get token info: {e}")
            return None
    
    def get_active_sessions(self, entity_id: str = None) -> List[Dict[str, Any]]:
        """Get active sessions"""
        try:
            sessions = []
            for session_id, session_data in self.active_sessions.items():
                if entity_id is None or session_data['entity_id'] == entity_id:
                    sessions.append({
                        'session_id': session_id,
                        'entity_id': session_data['entity_id'],
                        'created_at': session_data['created_at'],
                        'last_activity': session_data['last_activity'],
                        'permissions': session_data['permissions'],
                        'roles': session_data['roles'],
                        'security_level': session_data['security_level']
                    })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
    
    async def _cleanup_expired_tokens(self):
        """Clean up expired tokens and sessions"""
        while True:
            try:
                now = datetime.utcnow()
                
                # Clean up expired refresh tokens
                expired_refresh_tokens = [
                    token_id for token_id, token_data in self.refresh_tokens.items()
                    if token_data.expires_at < now or token_data.revoked
                ]
                
                for token_id in expired_refresh_tokens:
                    del self.refresh_tokens[token_id]
                
                # Clean up expired sessions (inactive for more than 24 hours)
                expired_sessions = [
                    session_id for session_id, session_data in self.active_sessions.items()
                    if (now - session_data['last_activity']).total_seconds() > 86400
                ]
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                
                # Clean up blacklisted tokens (remove after 7 days)
                # In production, you might want to persist blacklisted tokens
                
                if expired_refresh_tokens or expired_sessions:
                    logger.info(f"Cleaned up {len(expired_refresh_tokens)} refresh tokens and {len(expired_sessions)} sessions")
                
                await asyncio.sleep(self.config.blacklist_cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error during token cleanup: {e}")
                await asyncio.sleep(60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get JWT manager statistics"""
        return {
            'active_sessions': len(self.active_sessions),
            'refresh_tokens': len(self.refresh_tokens),
            'blacklisted_tokens': len(self.blacklisted_tokens),
            'algorithm': self.config.algorithm,
            'issuer': self.config.issuer,
            'access_token_expire': self.config.access_token_expire,
            'refresh_token_expire': self.config.refresh_token_expire
        }

# Global JWT manager instance
jwt_manager = JWTManager()