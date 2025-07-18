"""
Aetherveil Sentinel API Module

REST API endpoints for all security modules providing programmatic access
to reconnaissance, scanning, exploitation, stealth, OSINT, orchestration,
and reporting capabilities.

Security Level: DEFENSIVE_ONLY
"""

from typing import Dict, Any, List, Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class APIVersion(Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"

class APIStatus(Enum):
    """API status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"

# API Configuration
API_CONFIG = {
    "title": "Aetherveil Sentinel API",
    "description": "Comprehensive cybersecurity platform API for defensive operations",
    "version": "1.0.0",
    "current_version": APIVersion.V1,
    "status": APIStatus.ACTIVE,
    "security_disclaimer": """
    SECURITY NOTICE: This API provides access to security testing capabilities.
    All operations are designed for authorized defensive security purposes only.
    Users must ensure compliance with applicable laws and regulations.
    """
}

__all__ = [
    "APIVersion",
    "APIStatus", 
    "API_CONFIG"
]