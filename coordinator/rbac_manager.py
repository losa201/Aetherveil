"""
Advanced Role-Based Access Control (RBAC) Manager for Aetherveil Sentinel
Implements comprehensive RBAC with hierarchical roles, dynamic permissions, and attribute-based access control
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import re
import fnmatch
from collections import defaultdict

from coordinator.security_manager import SecurityLevel, ThreatLevel

logger = logging.getLogger(__name__)

class PermissionType(Enum):
    """Permission types"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"

class AccessDecision(Enum):
    """Access decision types"""
    PERMIT = "permit"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"

class ConditionOperator(Enum):
    """Condition operators"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    REGEX = "regex"
    NOT_REGEX = "not_regex"
    TIME_RANGE = "time_range"
    IP_RANGE = "ip_range"

@dataclass
class AccessCondition:
    """Access condition"""
    attribute: str
    operator: ConditionOperator
    value: Any
    description: str = ""

@dataclass
class Permission:
    """Permission definition"""
    permission_id: str
    resource: str
    action: str
    permission_type: PermissionType = PermissionType.ALLOW
    conditions: List[AccessCondition] = field(default_factory=list)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

@dataclass
class Role:
    """Role definition"""
    role_id: str
    name: str
    description: str
    permissions: List[str] = field(default_factory=list)  # Permission IDs
    parent_roles: List[str] = field(default_factory=list)  # Parent role IDs
    child_roles: List[str] = field(default_factory=list)  # Child role IDs
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    active: bool = True

@dataclass
class RoleAssignment:
    """Role assignment to an entity"""
    assignment_id: str
    entity_id: str
    role_id: str
    assigned_by: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    conditions: List[AccessCondition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

@dataclass
class AccessPolicy:
    """Access policy"""
    policy_id: str
    name: str
    description: str
    target_resources: List[str] = field(default_factory=list)
    target_actions: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    target_roles: List[str] = field(default_factory=list)
    rules: List[Dict[str, Any]] = field(default_factory=list)
    effect: PermissionType = PermissionType.ALLOW
    conditions: List[AccessCondition] = field(default_factory=list)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True

@dataclass
class AccessRequest:
    """Access request"""
    request_id: str
    entity_id: str
    resource: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class AccessResult:
    """Access control result"""
    decision: AccessDecision
    reason: str
    applied_permissions: List[str] = field(default_factory=list)
    applied_roles: List[str] = field(default_factory=list)
    applied_policies: List[str] = field(default_factory=list)
    conditions_evaluated: List[Dict[str, Any]] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

class RBACManager:
    """Advanced Role-Based Access Control Manager"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
        self.role_assignments: Dict[str, List[RoleAssignment]] = defaultdict(list)
        self.access_policies: Dict[str, AccessPolicy] = {}
        self.role_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # role_id -> set of inherited roles
        self.access_cache: Dict[str, Tuple[AccessResult, datetime]] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.statistics = {
            'access_requests': 0,
            'access_granted': 0,
            'access_denied': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def initialize_default_roles(self):
        """Initialize default roles and permissions"""
        try:
            # Create default permissions
            self._create_default_permissions()
            
            # Create default roles
            self._create_default_roles()
            
            # Build role hierarchy
            self._build_role_hierarchy()
            
            logger.info("Default roles and permissions initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize default roles: {e}")
            raise
    
    def _create_default_permissions(self):
        """Create default permissions"""
        default_permissions = [
            # System permissions
            ("system.admin", "system", "*", "Full system administration"),
            ("system.read", "system", "read", "Read system information"),
            ("system.monitor", "system", "monitor", "Monitor system status"),
            
            # Agent permissions
            ("agent.create", "agent", "create", "Create agents"),
            ("agent.read", "agent", "read", "Read agent information"),
            ("agent.update", "agent", "update", "Update agent configuration"),
            ("agent.delete", "agent", "delete", "Delete agents"),
            ("agent.control", "agent", "control", "Control agent operations"),
            
            # Task permissions
            ("task.create", "task", "create", "Create tasks"),
            ("task.read", "task", "read", "Read task information"),
            ("task.update", "task", "update", "Update task status"),
            ("task.delete", "task", "delete", "Delete tasks"),
            ("task.assign", "task", "assign", "Assign tasks to agents"),
            
            # Metrics permissions
            ("metrics.read", "metrics", "read", "Read metrics"),
            ("metrics.write", "metrics", "write", "Write metrics"),
            ("metrics.export", "metrics", "export", "Export metrics"),
            
            # Configuration permissions
            ("config.read", "config", "read", "Read configuration"),
            ("config.write", "config", "write", "Write configuration"),
            ("config.deploy", "config", "deploy", "Deploy configuration"),
            
            # Security permissions
            ("security.audit", "security", "audit", "Access security audit logs"),
            ("security.config", "security", "config", "Configure security settings"),
            ("security.monitor", "security", "monitor", "Monitor security events"),
            
            # API permissions
            ("api.read", "api", "read", "Read API endpoints"),
            ("api.write", "api", "write", "Write API endpoints"),
            ("api.admin", "api", "admin", "API administration")
        ]
        
        for perm_id, resource, action, description in default_permissions:
            permission = Permission(
                permission_id=perm_id,
                resource=resource,
                action=action,
                metadata={"description": description}
            )
            self.permissions[perm_id] = permission
    
    def _create_default_roles(self):
        """Create default roles"""
        # Super Admin Role
        super_admin = Role(
            role_id="super_admin",
            name="Super Administrator",
            description="Full system access with all permissions",
            permissions=list(self.permissions.keys()),
            security_level=SecurityLevel.CRITICAL
        )
        self.roles["super_admin"] = super_admin
        
        # Admin Role
        admin = Role(
            role_id="admin",
            name="Administrator",
            description="System administration with limited permissions",
            permissions=[
                "system.read", "system.monitor",
                "agent.create", "agent.read", "agent.update", "agent.delete",
                "task.create", "task.read", "task.update", "task.delete", "task.assign",
                "metrics.read", "metrics.write", "metrics.export",
                "config.read", "config.write",
                "security.audit", "security.monitor",
                "api.read", "api.write"
            ],
            security_level=SecurityLevel.HIGH
        )
        self.roles["admin"] = admin
        
        # Coordinator Role
        coordinator = Role(
            role_id="coordinator",
            name="Coordinator",
            description="Agent coordination and task management",
            permissions=[
                "agent.read", "agent.control",
                "task.create", "task.read", "task.update", "task.assign",
                "metrics.read", "metrics.write",
                "config.read",
                "api.read", "api.write"
            ],
            security_level=SecurityLevel.HIGH
        )
        self.roles["coordinator"] = coordinator
        
        # Agent Role
        agent = Role(
            role_id="agent",
            name="Agent",
            description="Basic agent operations",
            permissions=[
                "task.read", "task.update",
                "metrics.write",
                "config.read",
                "api.read"
            ],
            security_level=SecurityLevel.MEDIUM
        )
        self.roles["agent"] = agent
        
        # Monitor Role
        monitor = Role(
            role_id="monitor",
            name="Monitor",
            description="Read-only monitoring access",
            permissions=[
                "system.read", "system.monitor",
                "agent.read",
                "task.read",
                "metrics.read",
                "config.read",
                "security.monitor",
                "api.read"
            ],
            security_level=SecurityLevel.MEDIUM
        )
        self.roles["monitor"] = monitor
        
        # API User Role
        api_user = Role(
            role_id="api_user",
            name="API User",
            description="Basic API access",
            permissions=[
                "api.read"
            ],
            security_level=SecurityLevel.LOW
        )
        self.roles["api_user"] = api_user
    
    def _build_role_hierarchy(self):
        """Build role hierarchy for inheritance"""
        try:
            # Clear existing hierarchy
            self.role_hierarchy.clear()
            
            # Build hierarchy based on parent-child relationships
            for role_id, role in self.roles.items():
                self.role_hierarchy[role_id] = set()
                
                # Add direct parents
                for parent_id in role.parent_roles:
                    if parent_id in self.roles:
                        self.role_hierarchy[role_id].add(parent_id)
                
                # Add inherited roles (recursive)
                self._collect_inherited_roles(role_id, set())
            
            logger.info("Role hierarchy built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build role hierarchy: {e}")
            raise
    
    def _collect_inherited_roles(self, role_id: str, visited: Set[str]):
        """Recursively collect inherited roles"""
        if role_id in visited:
            return  # Prevent infinite recursion
        
        visited.add(role_id)
        
        role = self.roles.get(role_id)
        if not role:
            return
        
        for parent_id in role.parent_roles:
            if parent_id in self.roles:
                self.role_hierarchy[role_id].add(parent_id)
                self._collect_inherited_roles(parent_id, visited)
                
                # Add all inherited roles from parent
                self.role_hierarchy[role_id].update(self.role_hierarchy[parent_id])
    
    def create_role(self, name: str, description: str, permissions: List[str] = None,
                   parent_roles: List[str] = None, security_level: SecurityLevel = SecurityLevel.MEDIUM,
                   metadata: Dict[str, Any] = None) -> str:
        """Create new role"""
        try:
            role_id = str(uuid.uuid4())
            
            # Validate permissions
            if permissions:
                invalid_permissions = [p for p in permissions if p not in self.permissions]
                if invalid_permissions:
                    raise ValueError(f"Invalid permissions: {invalid_permissions}")
            
            # Validate parent roles
            if parent_roles:
                invalid_parents = [r for r in parent_roles if r not in self.roles]
                if invalid_parents:
                    raise ValueError(f"Invalid parent roles: {invalid_parents}")
            
            role = Role(
                role_id=role_id,
                name=name,
                description=description,
                permissions=permissions or [],
                parent_roles=parent_roles or [],
                security_level=security_level,
                metadata=metadata or {}
            )
            
            self.roles[role_id] = role
            
            # Update parent roles' child relationships
            if parent_roles:
                for parent_id in parent_roles:
                    if parent_id in self.roles:
                        self.roles[parent_id].child_roles.append(role_id)
            
            # Rebuild hierarchy
            self._build_role_hierarchy()
            
            logger.info(f"Created role: {name} ({role_id})")
            return role_id
            
        except Exception as e:
            logger.error(f"Failed to create role: {e}")
            raise
    
    def create_permission(self, resource: str, action: str, permission_type: PermissionType = PermissionType.ALLOW,
                         conditions: List[AccessCondition] = None, priority: int = 0,
                         metadata: Dict[str, Any] = None) -> str:
        """Create new permission"""
        try:
            permission_id = f"{resource}.{action}"
            
            permission = Permission(
                permission_id=permission_id,
                resource=resource,
                action=action,
                permission_type=permission_type,
                conditions=conditions or [],
                priority=priority,
                metadata=metadata or {}
            )
            
            self.permissions[permission_id] = permission
            
            logger.info(f"Created permission: {permission_id}")
            return permission_id
            
        except Exception as e:
            logger.error(f"Failed to create permission: {e}")
            raise
    
    def assign_role(self, entity_id: str, role_id: str, assigned_by: str,
                   expires_at: datetime = None, conditions: List[AccessCondition] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """Assign role to entity"""
        try:
            if role_id not in self.roles:
                raise ValueError(f"Role not found: {role_id}")
            
            assignment_id = str(uuid.uuid4())
            
            assignment = RoleAssignment(
                assignment_id=assignment_id,
                entity_id=entity_id,
                role_id=role_id,
                assigned_by=assigned_by,
                expires_at=expires_at,
                conditions=conditions or [],
                metadata=metadata or {}
            )
            
            self.role_assignments[entity_id].append(assignment)
            
            # Clear cache for this entity
            self._clear_entity_cache(entity_id)
            
            logger.info(f"Assigned role {role_id} to entity {entity_id}")
            return assignment_id
            
        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            raise
    
    def revoke_role(self, entity_id: str, role_id: str) -> bool:
        """Revoke role from entity"""
        try:
            assignments = self.role_assignments.get(entity_id, [])
            
            for assignment in assignments:
                if assignment.role_id == role_id and assignment.active:
                    assignment.active = False
                    
                    # Clear cache for this entity
                    self._clear_entity_cache(entity_id)
                    
                    logger.info(f"Revoked role {role_id} from entity {entity_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke role: {e}")
            return False
    
    def check_permission(self, entity_id: str, resource: str, action: str,
                        context: Dict[str, Any] = None) -> AccessResult:
        """Check if entity has permission for resource and action"""
        try:
            start_time = time.time()
            
            # Create access request
            request = AccessRequest(
                request_id=str(uuid.uuid4()),
                entity_id=entity_id,
                resource=resource,
                action=action,
                context=context or {}
            )
            
            # Check cache first
            cache_key = f"{entity_id}:{resource}:{action}:{hash(str(sorted((context or {}).items())))}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.statistics['cache_hits'] += 1
                return cached_result
            
            self.statistics['cache_misses'] += 1
            
            # Get entity roles
            entity_roles = self._get_entity_roles(entity_id)
            
            # Collect all permissions (direct and inherited)
            all_permissions = self._collect_entity_permissions(entity_roles)
            
            # Evaluate permissions
            result = self._evaluate_permissions(request, all_permissions, entity_roles)
            
            # Apply policies
            result = self._apply_policies(request, result)
            
            # Cache result
            result.processing_time = time.time() - start_time
            self._cache_result(cache_key, result)
            
            # Update statistics
            self.statistics['access_requests'] += 1
            if result.decision == AccessDecision.PERMIT:
                self.statistics['access_granted'] += 1
            else:
                self.statistics['access_denied'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to check permission: {e}")
            return AccessResult(
                decision=AccessDecision.INDETERMINATE,
                reason=f"Error checking permission: {e}",
                processing_time=time.time() - start_time
            )
    
    def _get_entity_roles(self, entity_id: str) -> List[str]:
        """Get active roles for entity"""
        try:
            active_roles = []
            assignments = self.role_assignments.get(entity_id, [])
            
            for assignment in assignments:
                if (assignment.active and 
                    (assignment.expires_at is None or assignment.expires_at > datetime.utcnow())):
                    
                    # Check assignment conditions
                    if self._evaluate_conditions(assignment.conditions, {}):
                        active_roles.append(assignment.role_id)
            
            return active_roles
            
        except Exception as e:
            logger.error(f"Failed to get entity roles: {e}")
            return []
    
    def _collect_entity_permissions(self, entity_roles: List[str]) -> List[str]:
        """Collect all permissions for entity roles (including inherited)"""
        try:
            all_permissions = set()
            
            for role_id in entity_roles:
                # Add direct permissions
                role = self.roles.get(role_id)
                if role and role.active:
                    all_permissions.update(role.permissions)
                
                # Add inherited permissions
                inherited_roles = self.role_hierarchy.get(role_id, set())
                for inherited_role_id in inherited_roles:
                    inherited_role = self.roles.get(inherited_role_id)
                    if inherited_role and inherited_role.active:
                        all_permissions.update(inherited_role.permissions)
            
            return list(all_permissions)
            
        except Exception as e:
            logger.error(f"Failed to collect entity permissions: {e}")
            return []
    
    def _evaluate_permissions(self, request: AccessRequest, permissions: List[str],
                            entity_roles: List[str]) -> AccessResult:
        """Evaluate permissions against request"""
        try:
            # Match permissions
            matching_permissions = []
            
            for perm_id in permissions:
                permission = self.permissions.get(perm_id)
                if not permission:
                    continue
                
                # Check if permission matches request
                if self._permission_matches(permission, request):
                    matching_permissions.append(permission)
            
            # Sort by priority (higher priority first)
            matching_permissions.sort(key=lambda p: p.priority, reverse=True)
            
            # Evaluate conditions and determine result
            for permission in matching_permissions:
                if self._evaluate_conditions(permission.conditions, request.context):
                    if permission.permission_type == PermissionType.ALLOW:
                        return AccessResult(
                            decision=AccessDecision.PERMIT,
                            reason=f"Permission granted: {permission.permission_id}",
                            applied_permissions=[permission.permission_id],
                            applied_roles=entity_roles
                        )
                    elif permission.permission_type == PermissionType.DENY:
                        return AccessResult(
                            decision=AccessDecision.DENY,
                            reason=f"Permission denied: {permission.permission_id}",
                            applied_permissions=[permission.permission_id],
                            applied_roles=entity_roles
                        )
            
            # No matching permissions found
            return AccessResult(
                decision=AccessDecision.DENY,
                reason="No matching permissions found",
                applied_roles=entity_roles
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate permissions: {e}")
            return AccessResult(
                decision=AccessDecision.INDETERMINATE,
                reason=f"Error evaluating permissions: {e}"
            )
    
    def _permission_matches(self, permission: Permission, request: AccessRequest) -> bool:
        """Check if permission matches request"""
        try:
            # Check resource match
            if not fnmatch.fnmatch(request.resource, permission.resource):
                return False
            
            # Check action match
            if permission.action != "*" and permission.action != request.action:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to match permission: {e}")
            return False
    
    def _evaluate_conditions(self, conditions: List[AccessCondition], context: Dict[str, Any]) -> bool:
        """Evaluate access conditions"""
        try:
            if not conditions:
                return True
            
            for condition in conditions:
                if not self._evaluate_single_condition(condition, context):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate conditions: {e}")
            return False
    
    def _evaluate_single_condition(self, condition: AccessCondition, context: Dict[str, Any]) -> bool:
        """Evaluate single access condition"""
        try:
            attribute_value = context.get(condition.attribute)
            
            if condition.operator == ConditionOperator.EQUALS:
                return attribute_value == condition.value
            elif condition.operator == ConditionOperator.NOT_EQUALS:
                return attribute_value != condition.value
            elif condition.operator == ConditionOperator.GREATER_THAN:
                return attribute_value > condition.value
            elif condition.operator == ConditionOperator.LESS_THAN:
                return attribute_value < condition.value
            elif condition.operator == ConditionOperator.GREATER_EQUAL:
                return attribute_value >= condition.value
            elif condition.operator == ConditionOperator.LESS_EQUAL:
                return attribute_value <= condition.value
            elif condition.operator == ConditionOperator.CONTAINS:
                return condition.value in str(attribute_value)
            elif condition.operator == ConditionOperator.NOT_CONTAINS:
                return condition.value not in str(attribute_value)
            elif condition.operator == ConditionOperator.IN:
                return attribute_value in condition.value
            elif condition.operator == ConditionOperator.NOT_IN:
                return attribute_value not in condition.value
            elif condition.operator == ConditionOperator.REGEX:
                return re.match(condition.value, str(attribute_value)) is not None
            elif condition.operator == ConditionOperator.NOT_REGEX:
                return re.match(condition.value, str(attribute_value)) is None
            elif condition.operator == ConditionOperator.TIME_RANGE:
                # condition.value should be [start_time, end_time]
                current_time = datetime.utcnow()
                start_time, end_time = condition.value
                return start_time <= current_time <= end_time
            elif condition.operator == ConditionOperator.IP_RANGE:
                # condition.value should be IP range/CIDR
                import ipaddress
                client_ip = context.get('client_ip')
                if client_ip:
                    return ipaddress.ip_address(client_ip) in ipaddress.ip_network(condition.value)
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate single condition: {e}")
            return False
    
    def _apply_policies(self, request: AccessRequest, result: AccessResult) -> AccessResult:
        """Apply access policies"""
        try:
            # For now, just return the result as-is
            # In a more advanced implementation, you would apply additional policies here
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply policies: {e}")
            return result
    
    def _get_cached_result(self, cache_key: str) -> Optional[AccessResult]:
        """Get cached access result"""
        try:
            cached_data = self.access_cache.get(cache_key)
            if cached_data:
                result, cached_time = cached_data
                if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl:
                    return result
                else:
                    del self.access_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached result: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result: AccessResult):
        """Cache access result"""
        try:
            self.access_cache[cache_key] = (result, datetime.utcnow())
            
            # Clean up old cache entries
            if len(self.access_cache) > 10000:  # Keep cache size manageable
                old_entries = sorted(self.access_cache.items(), key=lambda x: x[1][1])
                for key, _ in old_entries[:1000]:  # Remove oldest 1000 entries
                    del self.access_cache[key]
            
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    def _clear_entity_cache(self, entity_id: str):
        """Clear cache entries for entity"""
        try:
            keys_to_remove = [key for key in self.access_cache.keys() if key.startswith(f"{entity_id}:")]
            for key in keys_to_remove:
                del self.access_cache[key]
            
        except Exception as e:
            logger.error(f"Failed to clear entity cache: {e}")
    
    def get_entity_permissions(self, entity_id: str) -> List[str]:
        """Get all permissions for entity"""
        try:
            entity_roles = self._get_entity_roles(entity_id)
            return self._collect_entity_permissions(entity_roles)
            
        except Exception as e:
            logger.error(f"Failed to get entity permissions: {e}")
            return []
    
    def get_entity_roles(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get roles assigned to entity"""
        try:
            assignments = self.role_assignments.get(entity_id, [])
            active_assignments = [
                {
                    'assignment_id': assignment.assignment_id,
                    'role_id': assignment.role_id,
                    'role_name': self.roles.get(assignment.role_id, {}).name,
                    'assigned_by': assignment.assigned_by,
                    'assigned_at': assignment.assigned_at,
                    'expires_at': assignment.expires_at,
                    'active': assignment.active
                }
                for assignment in assignments
                if assignment.active and (assignment.expires_at is None or assignment.expires_at > datetime.utcnow())
            ]
            
            return active_assignments
            
        except Exception as e:
            logger.error(f"Failed to get entity roles: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RBAC statistics"""
        return {
            **self.statistics,
            'total_roles': len(self.roles),
            'total_permissions': len(self.permissions),
            'total_assignments': sum(len(assignments) for assignments in self.role_assignments.values()),
            'active_assignments': sum(
                len([a for a in assignments if a.active])
                for assignments in self.role_assignments.values()
            ),
            'cache_entries': len(self.access_cache)
        }

# Global RBAC manager instance
rbac_manager = RBACManager()