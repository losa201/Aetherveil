"""
Graph Schema - Defines the structure and relationships for the knowledge graph
Contains node types, relationship types, and validation rules
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


class NodeType(Enum):
    """Enumeration of all node types in the knowledge graph"""
    HOST = "host"
    SERVICE = "service"
    VULNERABILITY = "vulnerability"
    USER = "user"
    CREDENTIAL = "credential"
    ATTACK = "attack"
    ARTIFACT = "artifact"
    THREAT = "threat"
    NETWORK = "network"
    DOMAIN = "domain"
    CERTIFICATE = "certificate"
    MALWARE = "malware"
    EXPLOIT = "exploit"
    PAYLOAD = "payload"
    TECHNIQUE = "technique"
    TACTIC = "tactic"
    INTELLIGENCE = "intelligence"
    ASSET = "asset"
    ORGANIZATION = "organization"
    LOCATION = "location"


class RelationType(Enum):
    """Enumeration of all relationship types in the knowledge graph"""
    HOSTS = "hosts"
    RUNS_ON = "runs_on"
    CONNECTS_TO = "connects_to"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    AFFECTS = "affects"
    EXPLOITS = "exploits"
    USES = "uses"
    AUTHENTICATES = "authenticates"
    ACCESSES = "accesses"
    COMMUNICATES = "communicates"
    BELONGS_TO = "belongs_to"
    OWNS = "owns"
    MITIGATES = "mitigates"
    GENERATES = "generates"
    CORRELATES = "correlates"
    TRIGGERS = "triggers"
    IMPLEMENTS = "implements"
    REFERENCES = "references"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    AUTHORED_BY = "authored_by"
    TARGETS = "targets"
    BYPASSES = "bypasses"
    ENABLES = "enables"
    REQUIRES = "requires"


class SeverityLevel(Enum):
    """Severity levels for vulnerabilities and threats"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AttackStage(Enum):
    """ATT&CK framework stages"""
    RECONNAISSANCE = "reconnaissance"
    WEAPONIZATION = "weaponization"
    DELIVERY = "delivery"
    EXPLOITATION = "exploitation"
    INSTALLATION = "installation"
    COMMAND_CONTROL = "command_control"
    ACTIONS_OBJECTIVES = "actions_objectives"


class ThreatType(Enum):
    """Types of threats"""
    APT = "apt"
    MALWARE = "malware"
    PHISHING = "phishing"
    RANSOMWARE = "ransomware"
    BOTNET = "botnet"
    INSIDER = "insider"
    DDOS = "ddos"
    DATA_BREACH = "data_breach"
    ZERO_DAY = "zero_day"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: str
    timestamp: datetime
    severity: SeverityLevel
    source: str
    target: Optional[str] = None
    description: str = ""
    indicators: List[str] = field(default_factory=list)
    mitigation: Optional[str] = None
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "source": self.source,
            "target": self.target,
            "description": self.description,
            "indicators": self.indicators,
            "mitigation": self.mitigation,
            "confidence": self.confidence,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class NodeSchema:
    """Schema definition for graph nodes"""
    node_type: NodeType
    required_properties: List[str]
    optional_properties: List[str]
    property_types: Dict[str, type]
    validation_rules: Dict[str, Any]
    default_values: Dict[str, Any] = field(default_factory=dict)
    
    def validate_properties(self, properties: Dict[str, Any]) -> List[str]:
        """Validate node properties against schema"""
        errors = []
        
        # Check required properties
        for prop in self.required_properties:
            if prop not in properties:
                errors.append(f"Missing required property: {prop}")
        
        # Check property types
        for prop, value in properties.items():
            if prop in self.property_types:
                expected_type = self.property_types[prop]
                if not isinstance(value, expected_type):
                    errors.append(f"Property {prop} must be of type {expected_type.__name__}")
        
        # Apply validation rules
        for prop, rule in self.validation_rules.items():
            if prop in properties:
                value = properties[prop]
                if "min_length" in rule and len(str(value)) < rule["min_length"]:
                    errors.append(f"Property {prop} must be at least {rule['min_length']} characters")
                if "max_length" in rule and len(str(value)) > rule["max_length"]:
                    errors.append(f"Property {prop} must be at most {rule['max_length']} characters")
                if "pattern" in rule:
                    import re
                    if not re.match(rule["pattern"], str(value)):
                        errors.append(f"Property {prop} does not match required pattern")
                if "enum" in rule and value not in rule["enum"]:
                    errors.append(f"Property {prop} must be one of {rule['enum']}")
        
        return errors


@dataclass
class RelationshipSchema:
    """Schema definition for graph relationships"""
    relationship_type: RelationType
    source_types: List[NodeType]
    target_types: List[NodeType]
    required_properties: List[str]
    optional_properties: List[str]
    property_types: Dict[str, type]
    bidirectional: bool = False
    
    def validate_relationship(self, source_type: NodeType, target_type: NodeType, 
                            properties: Dict[str, Any]) -> List[str]:
        """Validate relationship against schema"""
        errors = []
        
        # Check source and target types
        if source_type not in self.source_types:
            errors.append(f"Invalid source type {source_type.value} for relationship {self.relationship_type.value}")
        
        if target_type not in self.target_types:
            errors.append(f"Invalid target type {target_type.value} for relationship {self.relationship_type.value}")
        
        # Check required properties
        for prop in self.required_properties:
            if prop not in properties:
                errors.append(f"Missing required property: {prop}")
        
        # Check property types
        for prop, value in properties.items():
            if prop in self.property_types:
                expected_type = self.property_types[prop]
                if not isinstance(value, expected_type):
                    errors.append(f"Property {prop} must be of type {expected_type.__name__}")
        
        return errors


class GraphSchema:
    """Complete graph schema with validation"""
    
    def __init__(self):
        self.node_schemas = self._initialize_node_schemas()
        self.relationship_schemas = self._initialize_relationship_schemas()
        self.mitre_techniques = self._initialize_mitre_techniques()
        self.vulnerability_mappings = self._initialize_vulnerability_mappings()
    
    def _initialize_node_schemas(self) -> Dict[NodeType, NodeSchema]:
        """Initialize all node schemas"""
        return {
            NodeType.HOST: NodeSchema(
                node_type=NodeType.HOST,
                required_properties=["ip", "hostname"],
                optional_properties=["os", "os_version", "domain", "mac_address", "subnet", "ports", "services"],
                property_types={
                    "ip": str,
                    "hostname": str,
                    "os": str,
                    "os_version": str,
                    "domain": str,
                    "mac_address": str,
                    "subnet": str,
                    "ports": list,
                    "services": list
                },
                validation_rules={
                    "ip": {"pattern": r"^(\d{1,3}\.){3}\d{1,3}$"},
                    "hostname": {"min_length": 1, "max_length": 255},
                    "os": {"enum": ["windows", "linux", "macos", "unix", "other"]},
                    "mac_address": {"pattern": r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$"}
                }
            ),
            
            NodeType.SERVICE: NodeSchema(
                node_type=NodeType.SERVICE,
                required_properties=["name", "port", "protocol"],
                optional_properties=["version", "banner", "state", "product", "extra_info"],
                property_types={
                    "name": str,
                    "port": int,
                    "protocol": str,
                    "version": str,
                    "banner": str,
                    "state": str,
                    "product": str,
                    "extra_info": str
                },
                validation_rules={
                    "port": {"min": 1, "max": 65535},
                    "protocol": {"enum": ["tcp", "udp", "icmp", "other"]},
                    "state": {"enum": ["open", "closed", "filtered", "unfiltered", "open|filtered", "closed|filtered"]}
                }
            ),
            
            NodeType.VULNERABILITY: NodeSchema(
                node_type=NodeType.VULNERABILITY,
                required_properties=["cve_id", "severity", "description"],
                optional_properties=["cvss_score", "cvss_vector", "cwe_id", "published_date", "modified_date", 
                                   "exploitable", "exploit_available", "patch_available", "references"],
                property_types={
                    "cve_id": str,
                    "severity": str,
                    "description": str,
                    "cvss_score": float,
                    "cvss_vector": str,
                    "cwe_id": str,
                    "published_date": str,
                    "modified_date": str,
                    "exploitable": bool,
                    "exploit_available": bool,
                    "patch_available": bool,
                    "references": list
                },
                validation_rules={
                    "cve_id": {"pattern": r"^CVE-\d{4}-\d{4,}$"},
                    "severity": {"enum": ["critical", "high", "medium", "low", "info"]},
                    "cvss_score": {"min": 0.0, "max": 10.0},
                    "cwe_id": {"pattern": r"^CWE-\d+$"}
                }
            ),
            
            NodeType.USER: NodeSchema(
                node_type=NodeType.USER,
                required_properties=["username"],
                optional_properties=["domain", "groups", "privileges", "last_login", "password_hash", 
                                   "password_policy", "locked", "disabled", "email"],
                property_types={
                    "username": str,
                    "domain": str,
                    "groups": list,
                    "privileges": list,
                    "last_login": str,
                    "password_hash": str,
                    "password_policy": dict,
                    "locked": bool,
                    "disabled": bool,
                    "email": str
                },
                validation_rules={
                    "username": {"min_length": 1, "max_length": 64},
                    "email": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
                }
            ),
            
            NodeType.CREDENTIAL: NodeSchema(
                node_type=NodeType.CREDENTIAL,
                required_properties=["type", "value"],
                optional_properties=["username", "domain", "hash_type", "cracked", "strength_score", 
                                   "source", "last_used"],
                property_types={
                    "type": str,
                    "value": str,
                    "username": str,
                    "domain": str,
                    "hash_type": str,
                    "cracked": bool,
                    "strength_score": float,
                    "source": str,
                    "last_used": str
                },
                validation_rules={
                    "type": {"enum": ["password", "hash", "token", "key", "certificate"]},
                    "strength_score": {"min": 0.0, "max": 100.0}
                }
            ),
            
            NodeType.ATTACK: NodeSchema(
                node_type=NodeType.ATTACK,
                required_properties=["technique", "stage", "timestamp"],
                optional_properties=["mitre_id", "success", "tools", "indicators", "impact", "persistence", 
                                   "lateral_movement", "data_exfiltration"],
                property_types={
                    "technique": str,
                    "stage": str,
                    "timestamp": str,
                    "mitre_id": str,
                    "success": bool,
                    "tools": list,
                    "indicators": list,
                    "impact": str,
                    "persistence": bool,
                    "lateral_movement": bool,
                    "data_exfiltration": bool
                },
                validation_rules={
                    "stage": {"enum": ["reconnaissance", "weaponization", "delivery", "exploitation", 
                                     "installation", "command_control", "actions_objectives"]},
                    "mitre_id": {"pattern": r"^T\d{4}(\.\d{3})?$"},
                    "impact": {"enum": ["none", "low", "medium", "high", "critical"]}
                }
            ),
            
            NodeType.THREAT: NodeSchema(
                node_type=NodeType.THREAT,
                required_properties=["name", "type", "severity_score"],
                optional_properties=["description", "indicators", "ttps", "attribution", "first_seen", 
                                   "last_seen", "campaign", "malware_families", "targets"],
                property_types={
                    "name": str,
                    "type": str,
                    "severity_score": float,
                    "description": str,
                    "indicators": list,
                    "ttps": list,
                    "attribution": str,
                    "first_seen": str,
                    "last_seen": str,
                    "campaign": str,
                    "malware_families": list,
                    "targets": list
                },
                validation_rules={
                    "type": {"enum": ["apt", "malware", "phishing", "ransomware", "botnet", "insider", 
                                    "ddos", "data_breach", "zero_day"]},
                    "severity_score": {"min": 0.0, "max": 100.0}
                }
            ),
            
            NodeType.NETWORK: NodeSchema(
                node_type=NodeType.NETWORK,
                required_properties=["cidr", "type"],
                optional_properties=["name", "description", "vlan", "gateway", "dns_servers", "dhcp_range"],
                property_types={
                    "cidr": str,
                    "type": str,
                    "name": str,
                    "description": str,
                    "vlan": int,
                    "gateway": str,
                    "dns_servers": list,
                    "dhcp_range": str
                },
                validation_rules={
                    "cidr": {"pattern": r"^(\d{1,3}\.){3}\d{1,3}/\d{1,2}$"},
                    "type": {"enum": ["internal", "external", "dmz", "management", "guest"]},
                    "vlan": {"min": 1, "max": 4094}
                }
            ),
            
            NodeType.DOMAIN: NodeSchema(
                node_type=NodeType.DOMAIN,
                required_properties=["name"],
                optional_properties=["registrar", "creation_date", "expiration_date", "nameservers", 
                                   "whois_info", "subdomains", "mx_records", "txt_records"],
                property_types={
                    "name": str,
                    "registrar": str,
                    "creation_date": str,
                    "expiration_date": str,
                    "nameservers": list,
                    "whois_info": dict,
                    "subdomains": list,
                    "mx_records": list,
                    "txt_records": list
                },
                validation_rules={
                    "name": {"pattern": r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
                }
            )
        }
    
    def _initialize_relationship_schemas(self) -> Dict[RelationType, RelationshipSchema]:
        """Initialize all relationship schemas"""
        return {
            RelationType.HOSTS: RelationshipSchema(
                relationship_type=RelationType.HOSTS,
                source_types=[NodeType.HOST],
                target_types=[NodeType.SERVICE],
                required_properties=[],
                optional_properties=["port", "protocol", "state"],
                property_types={"port": int, "protocol": str, "state": str}
            ),
            
            RelationType.RUNS_ON: RelationshipSchema(
                relationship_type=RelationType.RUNS_ON,
                source_types=[NodeType.SERVICE],
                target_types=[NodeType.HOST],
                required_properties=[],
                optional_properties=["process_id", "user", "command_line"],
                property_types={"process_id": int, "user": str, "command_line": str}
            ),
            
            RelationType.AFFECTS: RelationshipSchema(
                relationship_type=RelationType.AFFECTS,
                source_types=[NodeType.VULNERABILITY],
                target_types=[NodeType.HOST, NodeType.SERVICE],
                required_properties=[],
                optional_properties=["exploitable", "severity", "impact"],
                property_types={"exploitable": bool, "severity": str, "impact": str}
            ),
            
            RelationType.EXPLOITS: RelationshipSchema(
                relationship_type=RelationType.EXPLOITS,
                source_types=[NodeType.ATTACK],
                target_types=[NodeType.VULNERABILITY],
                required_properties=[],
                optional_properties=["success", "method", "payload"],
                property_types={"success": bool, "method": str, "payload": str}
            ),
            
            RelationType.USES: RelationshipSchema(
                relationship_type=RelationType.USES,
                source_types=[NodeType.ATTACK, NodeType.USER],
                target_types=[NodeType.CREDENTIAL, NodeType.ARTIFACT],
                required_properties=[],
                optional_properties=["frequency", "success_rate", "last_used"],
                property_types={"frequency": int, "success_rate": float, "last_used": str}
            ),
            
            RelationType.CONNECTS_TO: RelationshipSchema(
                relationship_type=RelationType.CONNECTS_TO,
                source_types=[NodeType.HOST],
                target_types=[NodeType.HOST, NodeType.SERVICE],
                required_properties=[],
                optional_properties=["protocol", "port", "frequency", "data_volume"],
                property_types={"protocol": str, "port": int, "frequency": int, "data_volume": int}
            ),
            
            RelationType.BELONGS_TO: RelationshipSchema(
                relationship_type=RelationType.BELONGS_TO,
                source_types=[NodeType.HOST, NodeType.USER],
                target_types=[NodeType.NETWORK, NodeType.DOMAIN, NodeType.ORGANIZATION],
                required_properties=[],
                optional_properties=["role", "permissions", "access_level"],
                property_types={"role": str, "permissions": list, "access_level": str}
            ),
            
            RelationType.TARGETS: RelationshipSchema(
                relationship_type=RelationType.TARGETS,
                source_types=[NodeType.ATTACK, NodeType.THREAT],
                target_types=[NodeType.HOST, NodeType.SERVICE, NodeType.USER, NodeType.ORGANIZATION],
                required_properties=[],
                optional_properties=["success", "impact", "method"],
                property_types={"success": bool, "impact": str, "method": str}
            ),
            
            RelationType.CORRELATES: RelationshipSchema(
                relationship_type=RelationType.CORRELATES,
                source_types=[NodeType.ATTACK, NodeType.THREAT, NodeType.VULNERABILITY],
                target_types=[NodeType.ATTACK, NodeType.THREAT, NodeType.VULNERABILITY],
                required_properties=[],
                optional_properties=["correlation_score", "confidence", "method"],
                property_types={"correlation_score": float, "confidence": float, "method": str},
                bidirectional=True
            ),
            
            RelationType.MITIGATES: RelationshipSchema(
                relationship_type=RelationType.MITIGATES,
                source_types=[NodeType.ARTIFACT],
                target_types=[NodeType.VULNERABILITY, NodeType.THREAT],
                required_properties=[],
                optional_properties=["effectiveness", "coverage", "implementation_difficulty"],
                property_types={"effectiveness": float, "coverage": float, "implementation_difficulty": str}
            )
        }
    
    def _initialize_mitre_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Initialize MITRE ATT&CK technique mappings"""
        return {
            "T1001": {
                "name": "Data Obfuscation",
                "tactic": "Command and Control",
                "description": "Adversaries may obfuscate command and control traffic to make it more difficult to detect.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor and analyze traffic flows and packet contents for obfuscation techniques."
            },
            "T1003": {
                "name": "OS Credential Dumping",
                "tactic": "Credential Access",
                "description": "Adversaries may attempt to dump credentials to obtain account login information.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor for unusual processes accessing LSASS memory or registry keys."
            },
            "T1005": {
                "name": "Data from Local System",
                "tactic": "Collection",
                "description": "Adversaries may search local system sources for sensitive information.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor for processes that access sensitive files or directories."
            },
            "T1012": {
                "name": "Query Registry",
                "tactic": "Discovery",
                "description": "Adversaries may interact with the Windows Registry to gather information.",
                "platforms": ["Windows"],
                "detection": "Monitor for registry access patterns and unusual queries."
            },
            "T1018": {
                "name": "Remote System Discovery",
                "tactic": "Discovery",
                "description": "Adversaries may attempt to get a listing of other systems by IP address, hostname, or other logical identifier.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor for network scanning activities and unusual network connections."
            },
            "T1021": {
                "name": "Remote Services",
                "tactic": "Lateral Movement",
                "description": "Adversaries may use valid accounts to log into a service specifically designed to accept remote connections.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor for unusual remote login attempts and service usage."
            },
            "T1055": {
                "name": "Process Injection",
                "tactic": "Defense Evasion",
                "description": "Adversaries may inject code into processes to evade process-based defenses.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor for process injection techniques and unusual process behavior."
            },
            "T1059": {
                "name": "Command and Scripting Interpreter",
                "tactic": "Execution",
                "description": "Adversaries may abuse command and script interpreters to execute commands.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor for unusual command line activity and script execution."
            },
            "T1068": {
                "name": "Exploitation for Privilege Escalation",
                "tactic": "Privilege Escalation",
                "description": "Adversaries may exploit software vulnerabilities to escalate privileges.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor for exploit attempts and unusual privilege escalation activities."
            },
            "T1071": {
                "name": "Application Layer Protocol",
                "tactic": "Command and Control",
                "description": "Adversaries may use application layer protocols to avoid detection.",
                "platforms": ["Linux", "macOS", "Windows"],
                "detection": "Monitor for unusual application protocol usage and traffic patterns."
            }
        }
    
    def _initialize_vulnerability_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize vulnerability classification mappings"""
        return {
            "CWE-79": {
                "name": "Cross-site Scripting",
                "category": "Input Validation",
                "description": "Improper neutralization of input during web page generation",
                "severity": "medium",
                "common_cves": ["CVE-2021-44228", "CVE-2022-22965"],
                "mitigations": ["Input validation", "Output encoding", "Content Security Policy"]
            },
            "CWE-89": {
                "name": "SQL Injection",
                "category": "Input Validation",
                "description": "Improper neutralization of SQL commands in dynamic queries",
                "severity": "high",
                "common_cves": ["CVE-2021-34527", "CVE-2022-30190"],
                "mitigations": ["Parameterized queries", "Input validation", "Least privilege"]
            },
            "CWE-200": {
                "name": "Information Exposure",
                "category": "Information Disclosure",
                "description": "Exposure of sensitive information to unauthorized actors",
                "severity": "medium",
                "common_cves": ["CVE-2021-44228", "CVE-2022-26134"],
                "mitigations": ["Access controls", "Data encryption", "Secure configuration"]
            },
            "CWE-287": {
                "name": "Authentication Bypass",
                "category": "Authentication",
                "description": "Improper authentication mechanisms",
                "severity": "high",
                "common_cves": ["CVE-2021-34473", "CVE-2022-21907"],
                "mitigations": ["Multi-factor authentication", "Strong authentication", "Session management"]
            },
            "CWE-352": {
                "name": "Cross-Site Request Forgery",
                "category": "Session Management",
                "description": "Forced execution of unwanted actions on behalf of authenticated users",
                "severity": "medium",
                "common_cves": ["CVE-2021-26855", "CVE-2022-30190"],
                "mitigations": ["CSRF tokens", "SameSite cookies", "Referrer validation"]
            }
        }
    
    def get_node_schema(self, node_type: NodeType) -> Optional[NodeSchema]:
        """Get schema for a specific node type"""
        return self.node_schemas.get(node_type)
    
    def get_relationship_schema(self, relationship_type: RelationType) -> Optional[RelationshipSchema]:
        """Get schema for a specific relationship type"""
        return self.relationship_schemas.get(relationship_type)
    
    def validate_node(self, node_type: NodeType, properties: Dict[str, Any]) -> List[str]:
        """Validate node against schema"""
        schema = self.get_node_schema(node_type)
        if not schema:
            return [f"Unknown node type: {node_type.value}"]
        
        return schema.validate_properties(properties)
    
    def validate_relationship(self, relationship_type: RelationType, 
                            source_type: NodeType, target_type: NodeType,
                            properties: Dict[str, Any]) -> List[str]:
        """Validate relationship against schema"""
        schema = self.get_relationship_schema(relationship_type)
        if not schema:
            return [f"Unknown relationship type: {relationship_type.value}"]
        
        return schema.validate_relationship(source_type, target_type, properties)
    
    def get_mitre_technique(self, technique_id: str) -> Optional[Dict[str, Any]]:
        """Get MITRE ATT&CK technique information"""
        return self.mitre_techniques.get(technique_id)
    
    def get_vulnerability_mapping(self, cwe_id: str) -> Optional[Dict[str, Any]]:
        """Get vulnerability classification mapping"""
        return self.vulnerability_mappings.get(cwe_id)
    
    def get_valid_relationships(self, source_type: NodeType, target_type: NodeType) -> List[RelationType]:
        """Get valid relationships between two node types"""
        valid_relationships = []
        
        for rel_type, schema in self.relationship_schemas.items():
            if source_type in schema.source_types and target_type in schema.target_types:
                valid_relationships.append(rel_type)
            elif schema.bidirectional and source_type in schema.target_types and target_type in schema.source_types:
                valid_relationships.append(rel_type)
        
        return valid_relationships
    
    def export_schema(self) -> Dict[str, Any]:
        """Export complete schema as dictionary"""
        return {
            "node_types": {nt.value: {
                "required_properties": schema.required_properties,
                "optional_properties": schema.optional_properties,
                "property_types": {k: v.__name__ for k, v in schema.property_types.items()},
                "validation_rules": schema.validation_rules
            } for nt, schema in self.node_schemas.items()},
            "relationship_types": {rt.value: {
                "source_types": [st.value for st in schema.source_types],
                "target_types": [tt.value for tt in schema.target_types],
                "required_properties": schema.required_properties,
                "optional_properties": schema.optional_properties,
                "property_types": {k: v.__name__ for k, v in schema.property_types.items()},
                "bidirectional": schema.bidirectional
            } for rt, schema in self.relationship_schemas.items()},
            "mitre_techniques": self.mitre_techniques,
            "vulnerability_mappings": self.vulnerability_mappings
        }
    
    def import_schema(self, schema_data: Dict[str, Any]) -> None:
        """Import schema from dictionary"""
        # This would be implemented to update the schema from external data
        pass