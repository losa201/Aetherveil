"""
Property-based tests for cryptographic functions in Aetherveil Sentinel
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.strategies import text, integers, binary
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

from aetherveil_sentinel.coordinator.security import SecurityManager
from aetherveil_sentinel.coordinator.jwt_manager import JWTManager
from aetherveil_sentinel.coordinator.zmq_encryption import ZMQEncryption


class TestCryptographicProperties:
    """Property-based tests for cryptographic functions."""
    
    @given(st.text(min_size=1, max_size=1000))
    def test_hash_consistency(self, plaintext: str):
        """Test that hash functions are consistent."""
        data = plaintext.encode('utf-8')
        
        # SHA-256 should always produce the same hash for the same input
        hash1 = hashlib.sha256(data).hexdigest()
        hash2 = hashlib.sha256(data).hexdigest()
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters
    
    @given(st.text(min_size=1, max_size=1000))
    def test_hash_deterministic(self, plaintext: str):
        """Test that hash functions are deterministic."""
        data = plaintext.encode('utf-8')
        
        # Multiple hash algorithms should be deterministic
        md5_hash = hashlib.md5(data).hexdigest()
        sha1_hash = hashlib.sha1(data).hexdigest()
        sha256_hash = hashlib.sha256(data).hexdigest()
        
        # Same input should always produce same output
        assert md5_hash == hashlib.md5(data).hexdigest()
        assert sha1_hash == hashlib.sha1(data).hexdigest()
        assert sha256_hash == hashlib.sha256(data).hexdigest()
    
    @given(st.text(min_size=1, max_size=1000), st.text(min_size=1, max_size=1000))
    def test_hash_avalanche_effect(self, plaintext1: str, plaintext2: str):
        """Test that small changes in input produce large changes in output."""
        assume(plaintext1 != plaintext2)
        
        data1 = plaintext1.encode('utf-8')
        data2 = plaintext2.encode('utf-8')
        
        hash1 = hashlib.sha256(data1).hexdigest()
        hash2 = hashlib.sha256(data2).hexdigest()
        
        # Different inputs should produce different hashes
        assert hash1 != hash2
    
    @given(st.binary(min_size=1, max_size=1000))
    def test_encryption_decryption_roundtrip(self, plaintext: bytes):
        """Test that encryption and decryption are inverse operations."""
        # Generate a key
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Encrypt and decrypt
        ciphertext = cipher.encrypt(plaintext)
        decrypted = cipher.decrypt(ciphertext)
        
        assert plaintext == decrypted
        assert ciphertext != plaintext  # Ensure encryption actually changed the data
    
    @given(st.binary(min_size=1, max_size=1000))
    def test_encryption_produces_different_ciphertexts(self, plaintext: bytes):
        """Test that encryption produces different ciphertexts for the same plaintext."""
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Encrypt the same plaintext multiple times
        ciphertext1 = cipher.encrypt(plaintext)
        ciphertext2 = cipher.encrypt(plaintext)
        
        # Should produce different ciphertexts (due to random IV/nonce)
        assert ciphertext1 != ciphertext2
        
        # But both should decrypt to the same plaintext
        assert cipher.decrypt(ciphertext1) == plaintext
        assert cipher.decrypt(ciphertext2) == plaintext
    
    @given(st.binary(min_size=1, max_size=1000), st.binary(min_size=16, max_size=16))
    def test_hmac_properties(self, message: bytes, key: bytes):
        """Test HMAC properties."""
        # Generate HMAC
        hmac1 = hmac.new(key, message, hashlib.sha256).hexdigest()
        hmac2 = hmac.new(key, message, hashlib.sha256).hexdigest()
        
        # HMAC should be deterministic
        assert hmac1 == hmac2
        
        # HMAC should be fixed length
        assert len(hmac1) == 64  # SHA-256 HMAC is 64 hex characters
    
    @given(st.binary(min_size=1, max_size=1000), st.binary(min_size=16, max_size=16))
    def test_hmac_key_sensitivity(self, message: bytes, key: bytes):
        """Test that HMAC is sensitive to key changes."""
        # Generate different key
        key2 = bytearray(key)
        key2[0] ^= 1  # Flip one bit
        key2 = bytes(key2)
        
        hmac1 = hmac.new(key, message, hashlib.sha256).hexdigest()
        hmac2 = hmac.new(key2, message, hashlib.sha256).hexdigest()
        
        # Different keys should produce different HMACs
        assert hmac1 != hmac2
    
    @given(st.text(min_size=1, max_size=100))
    def test_base64_encoding_roundtrip(self, plaintext: str):
        """Test that Base64 encoding/decoding is reversible."""
        data = plaintext.encode('utf-8')
        
        # Encode and decode
        encoded = base64.b64encode(data)
        decoded = base64.b64decode(encoded)
        
        assert data == decoded
    
    @given(st.binary(min_size=16, max_size=16))  # 128-bit key
    def test_aes_encryption_properties(self, key: bytes):
        """Test AES encryption properties."""
        # Test data
        plaintext = b"This is a test message for AES encryption testing."
        
        # Pad to block size
        block_size = 16
        padding_length = block_size - (len(plaintext) % block_size)
        padded_plaintext = plaintext + bytes([padding_length] * padding_length)
        
        # Generate random IV
        iv = os.urandom(16)
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
        
        # Decrypt
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding_length = decrypted_padded[-1]
        decrypted = decrypted_padded[:-padding_length]
        
        assert plaintext == decrypted
        assert ciphertext != padded_plaintext
    
    @given(st.binary(min_size=8, max_size=1000))
    def test_pbkdf2_properties(self, password: bytes):
        """Test PBKDF2 key derivation properties."""
        salt = os.urandom(16)
        iterations = 100000
        
        # Derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        key1 = kdf.derive(password)
        
        # Derive again with same parameters
        kdf2 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        key2 = kdf2.derive(password)
        
        # Should produce same key
        assert key1 == key2
        assert len(key1) == 32


class TestSecurityManagerProperties:
    """Property-based tests for SecurityManager."""
    
    @pytest.fixture
    def security_manager(self):
        """Create SecurityManager instance for testing."""
        return SecurityManager(config={
            "jwt_secret": "test-secret-key-for-testing",
            "encryption_key": Fernet.generate_key().decode(),
            "hash_algorithm": "sha256"
        })
    
    @given(st.text(min_size=1, max_size=1000))
    def test_password_hashing_properties(self, security_manager, password: str):
        """Test password hashing properties."""
        # Hash password
        hashed1 = security_manager.hash_password(password)
        hashed2 = security_manager.hash_password(password)
        
        # Same password should produce different hashes (due to salt)
        assert hashed1 != hashed2
        
        # But both should verify correctly
        assert security_manager.verify_password(password, hashed1)
        assert security_manager.verify_password(password, hashed2)
    
    @given(st.text(min_size=1, max_size=1000))
    def test_password_verification_properties(self, security_manager, password: str):
        """Test password verification properties."""
        # Hash password
        hashed = security_manager.hash_password(password)
        
        # Correct password should verify
        assert security_manager.verify_password(password, hashed)
        
        # Wrong password should not verify
        wrong_password = password + "x"
        assert not security_manager.verify_password(wrong_password, hashed)
    
    @given(st.text(min_size=1, max_size=1000))
    def test_data_encryption_properties(self, security_manager, plaintext: str):
        """Test data encryption properties."""
        data = plaintext.encode('utf-8')
        
        # Encrypt data
        encrypted = security_manager.encrypt_data(data)
        
        # Decrypt data
        decrypted = security_manager.decrypt_data(encrypted)
        
        assert data == decrypted
        assert encrypted != data


class TestJWTManagerProperties:
    """Property-based tests for JWT Manager."""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create JWT Manager instance for testing."""
        return JWTManager(secret="test-jwt-secret-key")
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=50),
        st.one_of(st.text(min_size=1, max_size=100), st.integers(), st.booleans()),
        min_size=1,
        max_size=10
    ))
    def test_jwt_token_properties(self, jwt_manager, payload: dict):
        """Test JWT token properties."""
        # Generate token
        token = jwt_manager.generate_token(payload)
        
        # Decode token
        decoded = jwt_manager.decode_token(token)
        
        # Payload should match (excluding JWT standard claims)
        for key, value in payload.items():
            assert decoded[key] == value
    
    @given(st.text(min_size=1, max_size=100))
    def test_jwt_token_expiration_properties(self, jwt_manager, user_id: str):
        """Test JWT token expiration properties."""
        # Generate token with short expiration
        token = jwt_manager.generate_token({"user_id": user_id}, expires_in=1)
        
        # Should be valid immediately
        decoded = jwt_manager.decode_token(token)
        assert decoded["user_id"] == user_id
        
        # Should have expiration claim
        assert "exp" in decoded
        assert decoded["exp"] > decoded["iat"]  # exp should be after iat
    
    @given(st.text(min_size=1, max_size=100))
    def test_jwt_signature_verification(self, jwt_manager, user_id: str):
        """Test JWT signature verification."""
        payload = {"user_id": user_id}
        token = jwt_manager.generate_token(payload)
        
        # Valid token should decode successfully
        decoded = jwt_manager.decode_token(token)
        assert decoded["user_id"] == user_id
        
        # Tampered token should fail verification
        tampered_token = token[:-5] + "XXXXX"  # Change last 5 characters
        
        with pytest.raises(Exception):  # Should raise verification error
            jwt_manager.decode_token(tampered_token)


class TestZMQEncryptionProperties:
    """Property-based tests for ZMQ Encryption."""
    
    @pytest.fixture
    def zmq_encryption(self):
        """Create ZMQ Encryption instance for testing."""
        return ZMQEncryption(key=os.urandom(32))
    
    @given(st.binary(min_size=1, max_size=1000))
    def test_zmq_message_encryption_properties(self, zmq_encryption, message: bytes):
        """Test ZMQ message encryption properties."""
        # Encrypt message
        encrypted = zmq_encryption.encrypt_message(message)
        
        # Decrypt message
        decrypted = zmq_encryption.decrypt_message(encrypted)
        
        assert message == decrypted
        assert encrypted != message
    
    @given(st.binary(min_size=1, max_size=1000))
    def test_zmq_message_authentication(self, zmq_encryption, message: bytes):
        """Test ZMQ message authentication."""
        # Encrypt message
        encrypted = zmq_encryption.encrypt_message(message)
        
        # Tamper with encrypted message
        tampered = bytearray(encrypted)
        tampered[0] ^= 1  # Flip one bit
        tampered = bytes(tampered)
        
        # Decryption should fail for tampered message
        with pytest.raises(Exception):
            zmq_encryption.decrypt_message(tampered)
    
    @given(st.binary(min_size=1, max_size=1000))
    def test_zmq_encryption_randomness(self, zmq_encryption, message: bytes):
        """Test that ZMQ encryption produces different ciphertexts."""
        # Encrypt same message multiple times
        encrypted1 = zmq_encryption.encrypt_message(message)
        encrypted2 = zmq_encryption.encrypt_message(message)
        
        # Should produce different ciphertexts (due to random nonce)
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same message
        assert zmq_encryption.decrypt_message(encrypted1) == message
        assert zmq_encryption.decrypt_message(encrypted2) == message


# Configure Hypothesis for slower tests
@settings(max_examples=50, deadline=5000)
class TestPerformanceProperties:
    """Property-based tests for performance characteristics."""
    
    @given(st.integers(min_value=1, max_value=1000))
    def test_hash_performance_scales_linearly(self, data_size: int):
        """Test that hash performance scales reasonably with input size."""
        import time
        
        # Generate test data
        data = os.urandom(data_size)
        
        # Time hash operation
        start_time = time.time()
        hashlib.sha256(data).hexdigest()
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Hash should complete in reasonable time
        assert duration < 1.0  # Should complete within 1 second
        
        # Performance should scale reasonably (not exponentially)
        performance_ratio = duration / data_size
        assert performance_ratio < 0.001  # Less than 1ms per byte
    
    @given(st.integers(min_value=1, max_value=100))
    def test_encryption_performance_acceptable(self, data_size_kb: int):
        """Test that encryption performance is acceptable."""
        import time
        
        # Generate test data
        data = os.urandom(data_size_kb * 1024)
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Time encryption
        start_time = time.time()
        ciphertext = cipher.encrypt(data)
        end_time = time.time()
        
        encryption_time = end_time - start_time
        
        # Time decryption
        start_time = time.time()
        decrypted = cipher.decrypt(ciphertext)
        end_time = time.time()
        
        decryption_time = end_time - start_time
        
        # Encryption/decryption should complete in reasonable time
        assert encryption_time < 5.0  # Within 5 seconds
        assert decryption_time < 5.0  # Within 5 seconds
        
        # Verify correctness
        assert data == decrypted