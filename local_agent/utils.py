#!/usr/bin/env python3
"""
Utility Functions Module - Common utility functions for the AI pentesting agent
"""

import asyncio
import hashlib
import json
import logging
import os
import psutil
import socket
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import ipaddress
import re
import base64
import random
import string

def setup_logging(config: Optional[Dict] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("aetherveil")
    
    if logger.hasHandlers():
        return logger
    
    # Configure log level
    log_level = logging.INFO
    if config and config.get("debug", False):
        log_level = logging.DEBUG
    
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log directory is specified
    if config and config.get("log_file"):
        log_file = Path(config["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def generate_session_id() -> str:
    """Generate unique session identifier"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"session_{timestamp}_{random_suffix}"

def generate_task_id(tool: str, target: str = "") -> str:
    """Generate unique task identifier"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_hash = hashlib.md5(target.encode()).hexdigest()[:8] if target else "notarget"
    return f"{tool}_{target_hash}_{timestamp}"

def hash_target(target: str) -> str:
    """Generate consistent hash for target (for privacy)"""
    return hashlib.sha256(target.encode()).hexdigest()[:16]

def validate_ip_address(ip: str) -> bool:
    """Validate IP address format"""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def validate_domain(domain: str) -> bool:
    """Validate domain name format"""
    domain_pattern = re.compile(
        r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
    )
    return bool(domain_pattern.match(domain)) and len(domain) <= 253

def validate_url(url: str) -> bool:
    """Validate URL format"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))

def is_safe_target(target: str, allowed_targets: List[str] = None) -> bool:
    """Check if target is safe to test"""
    if not target:
        return False
    
    # Default safe targets
    safe_defaults = ["127.0.0.1", "localhost", "::1", "0.0.0.0"]
    allowed = allowed_targets or safe_defaults
    
    # Check if target is in allowed list
    if target in allowed:
        return True
    
    # Check if target is private IP
    try:
        ip = ipaddress.ip_address(target)
        return ip.is_private or ip.is_loopback
    except ValueError:
        # Not an IP, check if it's a safe domain
        if target in safe_defaults:
            return True
        # Only allow localhost variants for domains
        return target.lower() in ["localhost", "localhost.localdomain"]

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem usage"""
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    # Limit length
    filename = filename[:255]
    # Ensure not empty
    if not filename:
        filename = "unnamed_file"
    return filename

def sanitize_command_arg(arg: str) -> str:
    """Sanitize command line argument"""
    # Block dangerous patterns
    dangerous_patterns = [';', '|', '&', '`', '$', '$(', '&&', '||', '\n', '\r']
    for pattern in dangerous_patterns:
        if pattern in arg:
            raise ValueError(f"Dangerous pattern '{pattern}' found in argument")
    return arg

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    try:
        return {
            "platform": os.uname().sysname,
            "architecture": os.uname().machine,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "used": psutil.disk_usage('/').used
            },
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
            "uptime": time.time() - psutil.boot_time()
        }
    except Exception as e:
        return {"error": str(e)}

def check_port_open(host: str, port: int, timeout: int = 3) -> bool:
    """Check if a port is open on a host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

async def async_check_port_open(host: str, port: int, timeout: int = 3) -> bool:
    """Async version of port check"""
    try:
        future = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(future, timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return True
    except:
        return False

def resolve_hostname(hostname: str) -> Optional[str]:
    """Resolve hostname to IP address"""
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return None

def get_local_ip() -> Optional[str]:
    """Get local IP address"""
    try:
        # Connect to a remote address to get local IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
        sock.close()
        return local_ip
    except Exception:
        return None

def encode_base64(data: Union[str, bytes]) -> str:
    """Encode data to base64"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')

def decode_base64(data: str) -> bytes:
    """Decode base64 data"""
    return base64.b64decode(data)

def safe_json_loads(data: str) -> Optional[Dict]:
    """Safely load JSON data"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None

def safe_json_dumps(data: Any, indent: int = None) -> str:
    """Safely dump data to JSON"""
    try:
        return json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"

def truncate_string(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate string to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text"""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)

def extract_ips_from_text(text: str) -> List[str]:
    """Extract IP addresses from text"""
    ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
    potential_ips = ip_pattern.findall(text)
    # Validate each IP
    valid_ips = []
    for ip in potential_ips:
        if validate_ip_address(ip):
            valid_ips.append(ip)
    return valid_ips

def extract_domains_from_text(text: str) -> List[str]:
    """Extract domain names from text"""
    domain_pattern = re.compile(
        r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}\b'
    )
    potential_domains = domain_pattern.findall(text)
    # Validate each domain
    valid_domains = []
    for domain in potential_domains:
        if validate_domain(domain):
            valid_domains.append(domain)
    return valid_domains

def calculate_entropy(data: str) -> float:
    """Calculate Shannon entropy of string"""
    if not data:
        return 0
    
    from collections import Counter
    import math
    
    # Count character frequencies
    counter = Counter(data)
    length = len(data)
    
    # Calculate entropy
    entropy = 0
    for count in counter.values():
        probability = count / length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy

def is_likely_encoded(text: str, min_entropy: float = 4.5) -> bool:
    """Check if text is likely encoded/encrypted based on entropy"""
    return calculate_entropy(text) > min_entropy

def rate_limit_check(last_request_time: float, min_interval: float) -> bool:
    """Check if enough time has passed for rate limiting"""
    current_time = time.time()
    return (current_time - last_request_time) >= min_interval

class RateLimiter:
    """Simple rate limiter for requests"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_proceed(self) -> bool:
        """Check if request can proceed"""
        current_time = time.time()
        
        # Remove old requests outside time window
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True
        
        return False
    
    def time_until_next_request(self) -> float:
        """Get time until next request is allowed"""
        if len(self.requests) < self.max_requests:
            return 0
        
        oldest_request = min(self.requests)
        current_time = time.time()
        return max(0, self.time_window - (current_time - oldest_request))

async def run_command_safely(command: List[str], timeout: int = 30, 
                           cwd: Optional[str] = None) -> Dict[str, Any]:
    """Run command safely with timeout and validation"""
    try:
        # Validate command arguments
        validated_command = []
        for arg in command:
            try:
                sanitize_command_arg(str(arg))
                validated_command.append(str(arg))
            except ValueError as e:
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Invalid command argument: {e}",
                    "error": str(e)
                }
        
        start_time = time.time()
        
        # Execute command
        process = await asyncio.create_subprocess_exec(
            *validated_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            end_time = time.time()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "duration": end_time - start_time,
                "command": " ".join(validated_command)
            }
            
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "duration": timeout,
                "command": " ".join(validated_command),
                "timeout": True
            }
            
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command execution error: {str(e)}",
            "duration": 0,
            "command": " ".join(command) if command else "",
            "error": str(e)
        }

def clean_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def extract_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> Optional[str]:
    """Calculate file hash"""
    try:
        hash_func = getattr(hashlib, algorithm)()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception:
        return None

def create_directory_safely(path: Union[str, Path], mode: int = 0o755) -> bool:
    """Create directory safely with proper permissions"""
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        return True
    except Exception:
        return False

def cleanup_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24) -> int:
    """Cleanup old temporary files"""
    try:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return 0
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        removed_count = 0
        
        for file_path in temp_path.rglob('*'):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception:
                        continue
        
        return removed_count
        
    except Exception:
        return 0

def get_file_type(file_path: Union[str, Path]) -> str:
    """Detect file type using magic numbers"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        # Common magic numbers
        if header.startswith(b'\x89PNG'):
            return 'png'
        elif header.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif header.startswith(b'GIF8'):
            return 'gif'
        elif header.startswith(b'%PDF'):
            return 'pdf'
        elif header.startswith(b'PK'):
            return 'zip'
        elif header.startswith(b'\x1f\x8b'):
            return 'gzip'
        else:
            return 'unknown'
            
    except Exception:
        return 'unknown'

def mask_sensitive_data(text: str, patterns: List[str] = None) -> str:
    """Mask sensitive data in text"""
    if not patterns:
        patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9]{20,}\b',  # Potential tokens/keys
        ]
    
    masked_text = text
    for pattern in patterns:
        masked_text = re.sub(pattern, '[MASKED]', masked_text)
    
    return masked_text

# Constants for common use
DEFAULT_TIMEOUT = 30
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
DEFAULT_RATE_LIMIT = 10  # requests per minute
COMMON_PORTS = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 993, 995, 1433, 1521, 3306, 3389, 5432, 5985, 8080]