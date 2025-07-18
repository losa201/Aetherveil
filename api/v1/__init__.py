"""
Aetherveil Sentinel API v1

Version 1 of the REST API endpoints for all security modules.
Provides comprehensive programmatic access to security testing capabilities.

Security Level: DEFENSIVE_ONLY
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class APIResponseStatus(Enum):
    """API response status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class APIResponseCode(Enum):
    """Standard API response codes"""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    SERVICE_UNAVAILABLE = 503

# Common response models
STANDARD_RESPONSES = {
    200: {"description": "Success"},
    201: {"description": "Created"},
    202: {"description": "Accepted"},
    400: {"description": "Bad Request"},
    401: {"description": "Unauthorized"},
    403: {"description": "Forbidden"},
    404: {"description": "Not Found"},
    422: {"description": "Validation Error"},
    429: {"description": "Too Many Requests"},
    500: {"description": "Internal Server Error"}
}

__all__ = [
    "APIResponseStatus",
    "APIResponseCode",
    "STANDARD_RESPONSES"
]