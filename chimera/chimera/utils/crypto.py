"""
Cryptographic utilities for secure operations
"""

import os
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class CryptoUtils:
    """Cryptographic utilities for Chimera"""
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate encryption key"""
        return Fernet.generate_key()
        
    @staticmethod
    def encrypt_data(data: str, key: bytes) -> str:
        """Encrypt data with key"""
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
        
    @staticmethod
    def decrypt_data(encrypted_data: str, key: bytes) -> str:
        """Decrypt data with key"""
        f = Fernet(key)
        decoded = base64.b64decode(encrypted_data.encode())
        decrypted = f.decrypt(decoded)
        return decrypted.decode()
        
    @staticmethod
    def hash_string(data: str) -> str:
        """Hash string with SHA256"""
        return hashlib.sha256(data.encode()).hexdigest()
        
    @staticmethod
    def secure_random(length: int) -> str:
        """Generate secure random string"""
        return base64.b64encode(os.urandom(length)).decode()[:length]