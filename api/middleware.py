"""
Aetherveil Sentinel API Middleware

Custom middleware for security, rate limiting, auditing, and request handling.
Provides comprehensive protection and monitoring for the API endpoints.

Security Level: DEFENSIVE_ONLY
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent API abuse"""
    
    def __init__(self, app: ASGIApp, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.clients: Dict[str, List[float]] = defaultdict(list)
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process rate limiting for requests"""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries()
            self.last_cleanup = current_time
        
        # Check rate limit
        if self._is_rate_limited(client_ip, current_time):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls_per_minute} requests per minute allowed",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self.clients[client_ip].append(current_time)
        
        # Add rate limit headers
        response = await call_next(request)
        remaining = self._get_remaining_requests(client_ip, current_time)
        
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit"""
        if client_ip not in self.clients:
            return False
        
        # Remove requests older than 1 minute
        minute_ago = current_time - 60
        self.clients[client_ip] = [
            timestamp for timestamp in self.clients[client_ip]
            if timestamp > minute_ago
        ]
        
        return len(self.clients[client_ip]) >= self.calls_per_minute
    
    def _get_remaining_requests(self, client_ip: str, current_time: float) -> int:
        """Get remaining requests for client"""
        if client_ip not in self.clients:
            return self.calls_per_minute
        
        minute_ago = current_time - 60
        recent_requests = [
            timestamp for timestamp in self.clients[client_ip]
            if timestamp > minute_ago
        ]
        
        return max(0, self.calls_per_minute - len(recent_requests))
    
    async def _cleanup_old_entries(self):
        """Clean up old rate limit entries"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        for client_ip in list(self.clients.keys()):
            self.clients[client_ip] = [
                timestamp for timestamp in self.clients[client_ip]
                if timestamp > minute_ago
            ]
            
            # Remove empty entries
            if not self.clients[client_ip]:
                del self.clients[client_ip]

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and protection"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.suspicious_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload=",
            r"onerror=",
            r"eval\(",
            r"exec\(",
            r"union.*select",
            r"drop.*table",
            r"insert.*into",
            r"update.*set",
            r"delete.*from"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process security checks for requests"""
        
        # Check for suspicious patterns in URL
        if self._contains_suspicious_patterns(str(request.url)):
            logger.warning(f"Suspicious URL detected: {request.url}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Security violation",
                    "message": "Request contains suspicious patterns"
                }
            )
        
        # Check request headers
        if not self._validate_headers(request):
            logger.warning(f"Invalid headers from {request.client.host}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Invalid request headers",
                    "message": "Request headers contain suspicious content"
                }
            )
        
        # Check request body for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            if body and self._contains_suspicious_patterns(body.decode('utf-8', errors='ignore')):
                logger.warning(f"Suspicious request body from {request.client.host}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "Security violation",
                        "message": "Request body contains suspicious patterns"
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns"""
        import re
        text_lower = text.lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _validate_headers(self, request: Request) -> bool:
        """Validate request headers"""
        # Check for excessively long headers
        for name, value in request.headers.items():
            if len(name) > 100 or len(value) > 8192:
                return False
            
            # Check for suspicious content in headers
            if self._contains_suspicious_patterns(f"{name}: {value}"):
                return False
        
        return True

class AuditMiddleware(BaseHTTPMiddleware):
    """Audit middleware for request logging and monitoring"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_log: List[Dict[str, Any]] = []
        self.max_log_entries = 10000
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process audit logging for requests"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Store request ID for error handling
        request.state.request_id = request_id
        
        # Log request
        client_ip = self._get_client_ip(request)
        
        request_log = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "content_length": request.headers.get("Content-Length", 0),
            "authorization": "Bearer ***" if request.headers.get("Authorization") else None
        }
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            end_time = time.time()
            request_log.update({
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "response_size": response.headers.get("Content-Length", 0)
            })
            
            # Add request ID to response
            response.headers["X-Request-ID"] = request_id
            
        except Exception as e:
            # Log error
            end_time = time.time()
            request_log.update({
                "status_code": 500,
                "error": str(e),
                "response_time": end_time - start_time
            })
            
            logger.error(f"Request {request_id} failed: {e}")
            raise
        
        # Store audit log
        self._store_audit_log(request_log)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _store_audit_log(self, log_entry: Dict[str, Any]):
        """Store audit log entry"""
        self.audit_log.append(log_entry)
        
        # Rotate log if too large
        if len(self.audit_log) > self.max_log_entries:
            self.audit_log = self.audit_log[-self.max_log_entries:]
        
        # Log to file/database in production
        if log_entry.get("status_code", 0) >= 400:
            logger.warning(f"HTTP {log_entry['status_code']}: {log_entry['method']} {log_entry['url']}")
    
    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit logs"""
        return self.audit_log[-limit:]

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Request validation middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_url_length = 2048
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Validate requests"""
        
        # Check URL length
        if len(str(request.url)) > self.max_url_length:
            return JSONResponse(
                status_code=status.HTTP_414_REQUEST_URI_TOO_LONG,
                content={
                    "error": "URL too long",
                    "message": f"URL length exceeds {self.max_url_length} characters"
                }
            )
        
        # Check request size
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request too large",
                    "message": f"Request size exceeds {self.max_request_size} bytes"
                }
            )
        
        # Check for required headers
        if request.method in ["POST", "PUT", "PATCH"]:
            if not request.headers.get("Content-Type"):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "Missing Content-Type header",
                        "message": "Content-Type header is required for this request"
                    }
                )
        
        return await call_next(request)

class CacheMiddleware(BaseHTTPMiddleware):
    """Simple caching middleware for GET requests"""
    
    def __init__(self, app: ASGIApp, cache_ttl: int = 300):
        super().__init__(app)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Handle caching for GET requests"""
        
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            response = Response(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=cached_response["headers"]
            )
            response.headers["X-Cache"] = "HIT"
            return response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            await self._cache_response(cache_key, response)
        
        response.headers["X-Cache"] = "MISS"
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        key_data = f"{request.method}:{request.url}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid"""
        if cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        
        # Check if expired
        if time.time() > cached_item["expires_at"]:
            del self.cache[cache_key]
            return None
        
        return cached_item["response"]
    
    async def _cache_response(self, cache_key: str, response: Response):
        """Cache response"""
        try:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Store in cache
            self.cache[cache_key] = {
                "response": {
                    "content": body,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                },
                "expires_at": time.time() + self.cache_ttl
            }
            
            # Recreate response
            response.body_iterator = iter([body])
            
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

__all__ = [
    "RateLimitMiddleware",
    "SecurityMiddleware",
    "AuditMiddleware",
    "RequestValidationMiddleware",
    "CacheMiddleware"
]