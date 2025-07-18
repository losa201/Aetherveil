"""
Aetherveil Sentinel Core Modules

This package contains the core security modules for the Aetherveil Sentinel platform.
Each module is designed to operate independently while integrating seamlessly with
the overall system architecture.

Modules:
    - reconnaissance: Passive and active reconnaissance capabilities
    - scanning: Vulnerability detection and assessment
    - exploitation: Ethical exploitation and testing capabilities
    - stealth: Evasion and stealth techniques
    - osint: Open source intelligence gathering
    - orchestrator: Workflow management and integration
    - reporting: Report generation and analysis
"""

from typing import Dict, Any, List
import logging
from enum import Enum

# Module version
__version__ = "1.0.0"

# Security classification
SECURITY_LEVEL = "DEFENSIVE_ONLY"

class ModuleType(Enum):
    """Enumeration of available module types"""
    RECONNAISSANCE = "reconnaissance"
    SCANNING = "scanning"
    EXPLOITATION = "exploitation"
    STEALTH = "stealth"
    OSINT = "osint"
    ORCHESTRATOR = "orchestrator"
    REPORTING = "reporting"

class ModuleStatus(Enum):
    """Enumeration of module status states"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

# Global module registry
_module_registry: Dict[str, Any] = {}

def register_module(module_name: str, module_instance: Any) -> None:
    """Register a module instance in the global registry"""
    _module_registry[module_name] = module_instance
    logging.info(f"Module {module_name} registered successfully")

def get_module(module_name: str) -> Any:
    """Retrieve a module instance from the registry"""
    return _module_registry.get(module_name)

def list_modules() -> List[str]:
    """List all registered modules"""
    return list(_module_registry.keys())

def get_module_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all registered modules"""
    info = {}
    for name, module in _module_registry.items():
        info[name] = {
            "name": name,
            "type": getattr(module, "module_type", "unknown"),
            "status": getattr(module, "status", "unknown"),
            "version": getattr(module, "version", "unknown")
        }
    return info

# Security disclaimer
SECURITY_DISCLAIMER = """
SECURITY NOTICE: All modules in this package are designed for defensive security
purposes only. Any exploitation capabilities are intended for authorized
penetration testing and vulnerability assessment within legal boundaries.
Users must ensure compliance with all applicable laws and regulations.
"""

__all__ = [
    "ModuleType",
    "ModuleStatus", 
    "register_module",
    "get_module",
    "list_modules",
    "get_module_info",
    "SECURITY_DISCLAIMER"
]