"""
Custom Cybersecurity Environment for Reinforcement Learning

This module implements a comprehensive cybersecurity simulation environment
where RL agents can learn attack strategies, reconnaissance, and exploitation tactics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import logging
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict
import json


class NetworkTopology(Enum):
    """Network topology types for different scenarios"""
    FLAT = "flat"
    HIERARCHICAL = "hierarchical"
    DMZ = "dmz"
    SEGMENTED = "segmented"
    CLOUD_HYBRID = "cloud_hybrid"


class HostType(Enum):
    """Types of hosts in the network"""
    WORKSTATION = "workstation"
    SERVER = "server"
    DATABASE = "database"
    WEB_SERVER = "web_server"
    DOMAIN_CONTROLLER = "domain_controller"
    FIREWALL = "firewall"
    ROUTER = "router"
    IOT_DEVICE = "iot_device"


class VulnerabilityType(Enum):
    """Types of vulnerabilities"""
    BUFFER_OVERFLOW = "buffer_overflow"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    RCE = "rce"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    WEAK_CREDENTIALS = "weak_credentials"
    UNPATCHED_SOFTWARE = "unpatched_software"
    MISCONFIGURATION = "misconfiguration"


@dataclass
class Host:
    """Represents a host in the network"""
    id: str
    host_type: HostType
    ip_address: str
    os: str
    services: List[Dict[str, Any]] = field(default_factory=list)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    is_compromised: bool = False
    access_level: str = "none"  # none, user, admin, root
    discovered: bool = False
    value: float = 1.0  # Strategic value of the host
    defense_level: float = 0.5  # Defense strength (0-1)
    

@dataclass
class NetworkSegment:
    """Represents a network segment"""
    id: str
    cidr: str
    hosts: List[Host] = field(default_factory=list)
    firewall_rules: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_level: float = 0.5  # Monitoring strength (0-1)


@dataclass
class AttackStep:
    """Represents a step in the attack chain"""
    action_type: str
    target_host: str
    technique: str
    success: bool
    detection_risk: float
    value_gained: float
    timestamp: int


class CybersecurityEnvironment(gym.Env):
    """
    Custom Cybersecurity Environment for RL Training
    
    This environment simulates a network with hosts, vulnerabilities, and defenses.
    The agent learns to perform reconnaissance, exploit vulnerabilities, and 
    achieve objectives while avoiding detection.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        network_size: int = 20,
        topology: NetworkTopology = NetworkTopology.HIERARCHICAL,
        vulnerability_density: float = 0.3,
        defense_strength: float = 0.5,
        episode_length: int = 100,
        detection_threshold: float = 0.8,
        objectives: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the cybersecurity environment
        
        Args:
            network_size: Number of hosts in the network
            topology: Network topology type
            vulnerability_density: Density of vulnerabilities (0-1)
            defense_strength: Overall defense strength (0-1)
            episode_length: Maximum steps per episode
            detection_threshold: Threshold for detection (0-1)
            objectives: List of objectives for the agent
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.network_size = network_size
        self.topology = topology
        self.vulnerability_density = vulnerability_density
        self.defense_strength = defense_strength
        self.episode_length = episode_length
        self.detection_threshold = detection_threshold
        self.objectives = objectives or ["compromise_domain_controller", "exfiltrate_data"]
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        # Initialize logger
        self.logger = logging.getLogger("CybersecurityEnv")
        
        # Environment state
        self.network_graph = None
        self.hosts = {}
        self.segments = {}
        self.current_step = 0
        self.agent_position = None  # Current host agent is on
        self.compromised_hosts = set()
        self.discovered_hosts = set()
        self.attack_chain = []
        self.detection_level = 0.0
        self.objectives_completed = set()
        
        # Observation and action spaces
        self._setup_spaces()
        
        # Generate initial network
        self._generate_network()
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Observation space: Network state, agent state, objectives
        obs_dim = (
            self.network_size * 10 +  # Host states (10 features per host)
            50 +  # Agent state and context
            20    # Objectives and global state
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: Different attack actions
        # 0: Reconnaissance, 1: Scan, 2: Exploit, 3: Lateral Move, 
        # 4: Privilege Escalation, 5: Data Exfiltration, 6: Persistence, 7: Wait
        self.action_space = spaces.Discrete(8)
        
    def _generate_network(self):
        """Generate the network topology and hosts"""
        self.network_graph = nx.Graph()
        self.hosts = {}
        self.segments = {}
        
        if self.topology == NetworkTopology.FLAT:
            self._generate_flat_network()
        elif self.topology == NetworkTopology.HIERARCHICAL:
            self._generate_hierarchical_network()
        elif self.topology == NetworkTopology.DMZ:
            self._generate_dmz_network()
        elif self.topology == NetworkTopology.SEGMENTED:
            self._generate_segmented_network()
        elif self.topology == NetworkTopology.CLOUD_HYBRID:
            self._generate_cloud_hybrid_network()
            
        # Add vulnerabilities
        self._add_vulnerabilities()
        
        # Set initial agent position (external/attacker position)
        self.agent_position = "external"
        
    def _generate_hierarchical_network(self):
        """Generate a hierarchical network topology"""
        # Create network segments
        segments = {
            "dmz": NetworkSegment("dmz", "10.0.1.0/24"),
            "internal": NetworkSegment("internal", "10.0.2.0/24"),
            "secure": NetworkSegment("secure", "10.0.3.0/24"),
            "management": NetworkSegment("management", "10.0.4.0/24")
        }
        
        host_types_distribution = {
            "dmz": [HostType.WEB_SERVER, HostType.FIREWALL],
            "internal": [HostType.WORKSTATION, HostType.SERVER],
            "secure": [HostType.DATABASE, HostType.SERVER],
            "management": [HostType.DOMAIN_CONTROLLER, HostType.SERVER]
        }
        
        host_id = 0
        for segment_name, segment in segments.items():
            hosts_in_segment = max(2, self.network_size // 4)
            
            for i in range(hosts_in_segment):
                host_type = random.choice(host_types_distribution[segment_name])
                host = Host(
                    id=f"host_{host_id}",
                    host_type=host_type,
                    ip_address=f"10.0.{list(segments.keys()).index(segment_name) + 1}.{i + 10}",
                    os=random.choice(["Windows", "Linux", "MacOS"]),
                    value=self._calculate_host_value(host_type),
                    defense_level=self.defense_strength + random.uniform(-0.2, 0.2)
                )
                host.defense_level = max(0.0, min(1.0, host.defense_level))
                
                self.hosts[host.id] = host
                segment.hosts.append(host)
                self.network_graph.add_node(host.id, **host.__dict__)
                host_id += 1
                
        self.segments = segments
        
        # Add network connections
        self._add_network_connections()
        
    def _generate_flat_network(self):
        """Generate a flat network topology"""
        segment = NetworkSegment("main", "192.168.1.0/24")
        
        for i in range(self.network_size):
            host_type = random.choice(list(HostType))
            host = Host(
                id=f"host_{i}",
                host_type=host_type,
                ip_address=f"192.168.1.{i + 10}",
                os=random.choice(["Windows", "Linux", "MacOS"]),
                value=self._calculate_host_value(host_type),
                defense_level=self.defense_strength + random.uniform(-0.2, 0.2)
            )
            host.defense_level = max(0.0, min(1.0, host.defense_level))
            
            self.hosts[host.id] = host
            segment.hosts.append(host)
            self.network_graph.add_node(host.id, **host.__dict__)
            
        self.segments["main"] = segment
        
        # Connect all hosts in flat topology
        for i in range(self.network_size):
            for j in range(i + 1, self.network_size):
                if random.random() < 0.3:  # 30% connection probability
                    self.network_graph.add_edge(f"host_{i}", f"host_{j}")
                    
    def _generate_dmz_network(self):
        """Generate a DMZ network topology"""
        # Similar to hierarchical but with specific DMZ structure
        self._generate_hierarchical_network()
        
    def _generate_segmented_network(self):
        """Generate a segmented network topology"""
        # Create multiple isolated segments
        num_segments = 4
        hosts_per_segment = self.network_size // num_segments
        
        for seg_idx in range(num_segments):
            segment = NetworkSegment(f"segment_{seg_idx}", f"10.{seg_idx}.0.0/24")
            
            for host_idx in range(hosts_per_segment):
                host_type = random.choice(list(HostType))
                host = Host(
                    id=f"host_{seg_idx}_{host_idx}",
                    host_type=host_type,
                    ip_address=f"10.{seg_idx}.0.{host_idx + 10}",
                    os=random.choice(["Windows", "Linux", "MacOS"]),
                    value=self._calculate_host_value(host_type),
                    defense_level=self.defense_strength + random.uniform(-0.2, 0.2)
                )
                host.defense_level = max(0.0, min(1.0, host.defense_level))
                
                self.hosts[host.id] = host
                segment.hosts.append(host)
                self.network_graph.add_node(host.id, **host.__dict__)
                
            self.segments[f"segment_{seg_idx}"] = segment
            
        # Add limited inter-segment connections
        self._add_segmented_connections()
        
    def _generate_cloud_hybrid_network(self):
        """Generate a cloud-hybrid network topology"""
        # Combine on-premises and cloud segments
        segments = {
            "onprem_internal": NetworkSegment("onprem_internal", "10.0.1.0/24"),
            "onprem_dmz": NetworkSegment("onprem_dmz", "10.0.2.0/24"),
            "cloud_public": NetworkSegment("cloud_public", "172.16.1.0/24"),
            "cloud_private": NetworkSegment("cloud_private", "172.16.2.0/24")
        }
        
        # Distribute hosts across segments
        hosts_per_segment = self.network_size // 4
        host_id = 0
        
        for segment_name, segment in segments.items():
            for i in range(hosts_per_segment):
                if "cloud" in segment_name:
                    host_type = random.choice([HostType.SERVER, HostType.WEB_SERVER, HostType.DATABASE])
                else:
                    host_type = random.choice(list(HostType))
                    
                host = Host(
                    id=f"host_{host_id}",
                    host_type=host_type,
                    ip_address=f"{segment.cidr.split('/')[0].rsplit('.', 1)[0]}.{i + 10}",
                    os=random.choice(["Windows", "Linux", "MacOS"]),
                    value=self._calculate_host_value(host_type),
                    defense_level=self.defense_strength + random.uniform(-0.2, 0.2)
                )
                host.defense_level = max(0.0, min(1.0, host.defense_level))
                
                self.hosts[host.id] = host
                segment.hosts.append(host)
                self.network_graph.add_node(host.id, **host.__dict__)
                host_id += 1
                
        self.segments = segments
        self._add_hybrid_connections()
        
    def _calculate_host_value(self, host_type: HostType) -> float:
        """Calculate strategic value of a host based on its type"""
        value_map = {
            HostType.DOMAIN_CONTROLLER: 1.0,
            HostType.DATABASE: 0.9,
            HostType.SERVER: 0.7,
            HostType.WEB_SERVER: 0.6,
            HostType.WORKSTATION: 0.4,
            HostType.FIREWALL: 0.5,
            HostType.ROUTER: 0.5,
            HostType.IOT_DEVICE: 0.2
        }
        return value_map.get(host_type, 0.5)
        
    def _add_vulnerabilities(self):
        """Add vulnerabilities to hosts based on vulnerability density"""
        for host in self.hosts.values():
            num_vulns = np.random.poisson(self.vulnerability_density * 3)
            
            for _ in range(num_vulns):
                vuln_type = random.choice(list(VulnerabilityType))
                vulnerability = {
                    "type": vuln_type.value,
                    "severity": random.uniform(0.3, 1.0),
                    "exploitability": random.uniform(0.2, 0.9),
                    "discovered": False
                }
                host.vulnerabilities.append(vulnerability)
                
            # Add services
            num_services = random.randint(1, 5)
            common_services = ["ssh", "http", "https", "ftp", "smb", "rdp", "dns", "smtp"]
            
            for _ in range(num_services):
                service = {
                    "name": random.choice(common_services),
                    "port": random.randint(1, 65535),
                    "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}",
                    "vulnerable": random.random() < 0.3
                }
                host.services.append(service)
                
    def _add_network_connections(self):
        """Add network connections for hierarchical topology"""
        # Connect within segments
        for segment in self.segments.values():
            hosts_in_segment = [h.id for h in segment.hosts]
            for i, host1 in enumerate(hosts_in_segment):
                for j, host2 in enumerate(hosts_in_segment[i+1:], i+1):
                    if random.random() < 0.4:  # 40% connection probability within segment
                        self.network_graph.add_edge(host1, host2)
                        
        # Connect between segments (limited)
        segment_names = list(self.segments.keys())
        for i, seg1 in enumerate(segment_names):
            for seg2 in segment_names[i+1:]:
                # Connect a few hosts between segments
                hosts1 = [h.id for h in self.segments[seg1].hosts]
                hosts2 = [h.id for h in self.segments[seg2].hosts]
                
                num_connections = random.randint(1, 3)
                for _ in range(num_connections):
                    if hosts1 and hosts2:
                        host1 = random.choice(hosts1)
                        host2 = random.choice(hosts2)
                        self.network_graph.add_edge(host1, host2)
                        
    def _add_segmented_connections(self):
        """Add connections for segmented network"""
        # Connect within segments
        for segment in self.segments.values():
            hosts_in_segment = [h.id for h in segment.hosts]
            for i, host1 in enumerate(hosts_in_segment):
                for j, host2 in enumerate(hosts_in_segment[i+1:], i+1):
                    if random.random() < 0.5:
                        self.network_graph.add_edge(host1, host2)
                        
        # Very limited inter-segment connections
        segment_names = list(self.segments.keys())
        if len(segment_names) > 1:
            # Connect only adjacent segments
            for i in range(len(segment_names) - 1):
                seg1 = segment_names[i]
                seg2 = segment_names[i + 1]
                
                hosts1 = [h.id for h in self.segments[seg1].hosts]
                hosts2 = [h.id for h in self.segments[seg2].hosts]
                
                if hosts1 and hosts2:
                    # Single connection between segments
                    host1 = random.choice(hosts1)
                    host2 = random.choice(hosts2)
                    self.network_graph.add_edge(host1, host2)
                    
    def _add_hybrid_connections(self):
        """Add connections for cloud-hybrid network"""
        # Connect within segments
        for segment in self.segments.values():
            hosts_in_segment = [h.id for h in segment.hosts]
            for i, host1 in enumerate(hosts_in_segment):
                for j, host2 in enumerate(hosts_in_segment[i+1:], i+1):
                    if random.random() < 0.6:
                        self.network_graph.add_edge(host1, host2)
                        
        # Connect on-premises segments
        onprem_segments = [k for k in self.segments.keys() if "onprem" in k]
        for i, seg1 in enumerate(onprem_segments):
            for seg2 in onprem_segments[i+1:]:
                hosts1 = [h.id for h in self.segments[seg1].hosts]
                hosts2 = [h.id for h in self.segments[seg2].hosts]
                
                if hosts1 and hosts2:
                    for _ in range(2):  # Multiple connections
                        host1 = random.choice(hosts1)
                        host2 = random.choice(hosts2)
                        self.network_graph.add_edge(host1, host2)
                        
        # Connect cloud segments
        cloud_segments = [k for k in self.segments.keys() if "cloud" in k]
        for i, seg1 in enumerate(cloud_segments):
            for seg2 in cloud_segments[i+1:]:
                hosts1 = [h.id for h in self.segments[seg1].hosts]
                hosts2 = [h.id for h in self.segments[seg2].hosts]
                
                if hosts1 and hosts2:
                    for _ in range(3):  # More connections in cloud
                        host1 = random.choice(hosts1)
                        host2 = random.choice(hosts2)
                        self.network_graph.add_edge(host1, host2)
                        
        # Limited hybrid connections (on-prem to cloud)
        if "onprem_dmz" in self.segments and "cloud_public" in self.segments:
            dmz_hosts = [h.id for h in self.segments["onprem_dmz"].hosts]
            cloud_hosts = [h.id for h in self.segments["cloud_public"].hosts]
            
            if dmz_hosts and cloud_hosts:
                host1 = random.choice(dmz_hosts)
                host2 = random.choice(cloud_hosts)
                self.network_graph.add_edge(host1, host2)
                
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Reset environment state
        self.current_step = 0
        self.agent_position = "external"
        self.compromised_hosts = set()
        self.discovered_hosts = set()
        self.attack_chain = []
        self.detection_level = 0.0
        self.objectives_completed = set()
        
        # Reset host states
        for host in self.hosts.values():
            host.is_compromised = False
            host.access_level = "none"
            host.discovered = False
            for vuln in host.vulnerabilities:
                vuln["discovered"] = False
                
        # Generate new network if options specify
        if options and options.get("regenerate_network", False):
            self._generate_network()
            
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Execute action
        reward, info = self._execute_action(action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.episode_length
        
        # Get new observation
        observation = self._get_observation()
        
        # Update info
        info.update(self._get_info())
        
        return observation, reward, terminated, truncated, info
        
    def _execute_action(self, action: int) -> Tuple[float, Dict]:
        """Execute the given action and return reward and info"""
        action_map = {
            0: self._action_reconnaissance,
            1: self._action_scan,
            2: self._action_exploit,
            3: self._action_lateral_move,
            4: self._action_privilege_escalation,
            5: self._action_data_exfiltration,
            6: self._action_persistence,
            7: self._action_wait
        }
        
        if action in action_map:
            return action_map[action]()
        else:
            return 0.0, {"error": "Invalid action"}
            
    def _action_reconnaissance(self) -> Tuple[float, Dict]:
        """Perform reconnaissance action"""
        reward = 0.0
        info = {"action": "reconnaissance", "discoveries": []}
        
        if self.agent_position == "external":
            # External reconnaissance - discover public-facing hosts
            for host in self.hosts.values():
                if not host.discovered and host.host_type in [HostType.WEB_SERVER, HostType.FIREWALL]:
                    if random.random() < 0.3:  # 30% chance to discover
                        host.discovered = True
                        self.discovered_hosts.add(host.id)
                        info["discoveries"].append(host.id)
                        reward += 0.1
        else:
            # Internal reconnaissance from compromised host
            current_host = self.hosts.get(self.agent_position)
            if current_host and current_host.is_compromised:
                # Discover connected hosts
                neighbors = list(self.network_graph.neighbors(self.agent_position))
                for neighbor_id in neighbors:
                    neighbor = self.hosts[neighbor_id]
                    if not neighbor.discovered:
                        if random.random() < 0.5:  # 50% chance to discover neighbors
                            neighbor.discovered = True
                            self.discovered_hosts.add(neighbor_id)
                            info["discoveries"].append(neighbor_id)
                            reward += 0.15
                            
        # Small detection risk
        self.detection_level += 0.01
        
        return reward, info
        
    def _action_scan(self) -> Tuple[float, Dict]:
        """Perform network scanning action"""
        reward = 0.0
        info = {"action": "scan", "vulnerabilities_found": []}
        
        # Can only scan discovered hosts
        scannable_hosts = [h for h in self.hosts.values() if h.discovered]
        
        if scannable_hosts:
            target_host = random.choice(scannable_hosts)
            
            # Discover vulnerabilities
            for vuln in target_host.vulnerabilities:
                if not vuln["discovered"]:
                    if random.random() < 0.4:  # 40% chance to find vulnerability
                        vuln["discovered"] = True
                        info["vulnerabilities_found"].append({
                            "host": target_host.id,
                            "type": vuln["type"],
                            "severity": vuln["severity"]
                        })
                        reward += 0.2 * vuln["severity"]
                        
        # Moderate detection risk
        self.detection_level += 0.05
        
        return reward, info
        
    def _action_exploit(self) -> Tuple[float, Dict]:
        """Perform exploitation action"""
        reward = 0.0
        info = {"action": "exploit", "compromised": [], "failed": []}
        
        # Find exploitable hosts
        exploitable_hosts = []
        for host in self.hosts.values():
            if host.discovered and not host.is_compromised:
                discovered_vulns = [v for v in host.vulnerabilities if v["discovered"]]
                if discovered_vulns:
                    exploitable_hosts.append((host, discovered_vulns))
                    
        if exploitable_hosts:
            target_host, vulns = random.choice(exploitable_hosts)
            vuln = max(vulns, key=lambda v: v["exploitability"])
            
            # Calculate exploit success probability
            success_prob = (
                vuln["exploitability"] * 
                (1 - target_host.defense_level) * 
                0.7  # Base success rate
            )
            
            if random.random() < success_prob:
                # Successful exploitation
                target_host.is_compromised = True
                target_host.access_level = "user"
                self.compromised_hosts.add(target_host.id)
                
                # Move agent to compromised host if external
                if self.agent_position == "external":
                    self.agent_position = target_host.id
                    
                info["compromised"].append(target_host.id)
                reward += target_host.value * 2.0
                
                # Record attack step
                attack_step = AttackStep(
                    action_type="exploit",
                    target_host=target_host.id,
                    technique=vuln["type"],
                    success=True,
                    detection_risk=0.1,
                    value_gained=target_host.value,
                    timestamp=self.current_step
                )
                self.attack_chain.append(attack_step)
                
            else:
                # Failed exploitation
                info["failed"].append(target_host.id)
                reward -= 0.5
                
        # High detection risk
        self.detection_level += 0.15
        
        return reward, info
        
    def _action_lateral_move(self) -> Tuple[float, Dict]:
        """Perform lateral movement action"""
        reward = 0.0
        info = {"action": "lateral_move", "moved_to": None}
        
        if self.agent_position != "external":
            current_host = self.hosts.get(self.agent_position)
            if current_host and current_host.is_compromised:
                # Find accessible neighboring hosts
                neighbors = list(self.network_graph.neighbors(self.agent_position))
                accessible_neighbors = [
                    n for n in neighbors 
                    if self.hosts[n].is_compromised and n != self.agent_position
                ]
                
                if accessible_neighbors:
                    target_host_id = random.choice(accessible_neighbors)
                    self.agent_position = target_host_id
                    info["moved_to"] = target_host_id
                    reward += 0.3
                    
        # Low detection risk
        self.detection_level += 0.02
        
        return reward, info
        
    def _action_privilege_escalation(self) -> Tuple[float, Dict]:
        """Perform privilege escalation action"""
        reward = 0.0
        info = {"action": "privilege_escalation", "escalated": []}
        
        if self.agent_position != "external":
            current_host = self.hosts.get(self.agent_position)
            if current_host and current_host.is_compromised:
                if current_host.access_level == "user":
                    # Try to escalate to admin
                    escalation_vulns = [
                        v for v in current_host.vulnerabilities 
                        if v["type"] == VulnerabilityType.PRIVILEGE_ESCALATION.value and v["discovered"]
                    ]
                    
                    success_prob = 0.3  # Base probability
                    if escalation_vulns:
                        best_vuln = max(escalation_vulns, key=lambda v: v["exploitability"])
                        success_prob = best_vuln["exploitability"] * 0.6
                        
                    if random.random() < success_prob:
                        current_host.access_level = "admin"
                        info["escalated"].append(current_host.id)
                        reward += current_host.value * 1.5
                        
                elif current_host.access_level == "admin":
                    # Try to escalate to root
                    if random.random() < 0.2:  # Lower probability for root
                        current_host.access_level = "root"
                        info["escalated"].append(current_host.id)
                        reward += current_host.value * 3.0
                        
        # Moderate detection risk
        self.detection_level += 0.08
        
        return reward, info
        
    def _action_data_exfiltration(self) -> Tuple[float, Dict]:
        """Perform data exfiltration action"""
        reward = 0.0
        info = {"action": "data_exfiltration", "exfiltrated": []}
        
        if self.agent_position != "external":
            current_host = self.hosts.get(self.agent_position)
            if current_host and current_host.is_compromised:
                # High-value hosts have more valuable data
                if current_host.host_type in [HostType.DATABASE, HostType.SERVER]:
                    data_value = current_host.value * 2.0
                    if current_host.access_level in ["admin", "root"]:
                        data_value *= 1.5
                        
                    info["exfiltrated"].append({
                        "host": current_host.id,
                        "value": data_value
                    })
                    reward += data_value
                    
                    # Check if this completes an objective
                    if "exfiltrate_data" in self.objectives:
                        self.objectives_completed.add("exfiltrate_data")
                        reward += 5.0  # Bonus for completing objective
                        
        # High detection risk
        self.detection_level += 0.2
        
        return reward, info
        
    def _action_persistence(self) -> Tuple[float, Dict]:
        """Establish persistence on current host"""
        reward = 0.0
        info = {"action": "persistence", "established": []}
        
        if self.agent_position != "external":
            current_host = self.hosts.get(self.agent_position)
            if current_host and current_host.is_compromised and current_host.access_level in ["admin", "root"]:
                # Establish persistence
                info["established"].append(current_host.id)
                reward += current_host.value * 0.5
                
        # Moderate detection risk
        self.detection_level += 0.1
        
        return reward, info
        
    def _action_wait(self) -> Tuple[float, Dict]:
        """Wait/do nothing - reduces detection risk"""
        # Waiting reduces detection level slightly
        self.detection_level = max(0.0, self.detection_level - 0.05)
        return 0.0, {"action": "wait"}
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the environment state"""
        obs = []
        
        # Host states (10 features per host)
        for host_id in sorted(self.hosts.keys()):
            host = self.hosts[host_id]
            host_obs = [
                float(host.discovered),
                float(host.is_compromised),
                {"none": 0, "user": 0.33, "admin": 0.66, "root": 1.0}[host.access_level],
                host.value,
                host.defense_level,
                len(host.vulnerabilities) / 10.0,  # Normalized
                len([v for v in host.vulnerabilities if v["discovered"]]) / max(1, len(host.vulnerabilities)),
                len(host.services) / 10.0,  # Normalized
                float(host_id == self.agent_position),
                float(host.host_type.value == "domain_controller")
            ]
            obs.extend(host_obs)
            
        # Pad or truncate to exact size
        target_host_features = self.network_size * 10
        if len(obs) < target_host_features:
            obs.extend([0.0] * (target_host_features - len(obs)))
        else:
            obs = obs[:target_host_features]
            
        # Agent state and context (50 features)
        agent_state = [
            self.current_step / self.episode_length,  # Progress
            self.detection_level,
            len(self.compromised_hosts) / self.network_size,
            len(self.discovered_hosts) / self.network_size,
            len(self.objectives_completed) / len(self.objectives),
            float(self.agent_position == "external"),
            len(self.attack_chain) / 50.0,  # Normalized attack chain length
        ]
        
        # Add 43 more features for network topology, segment info, etc.
        network_features = [
            len(self.segments) / 10.0,
            len(self.network_graph.edges) / (self.network_size * self.network_size),
            nx.density(self.network_graph) if self.network_graph.number_of_nodes() > 0 else 0.0,
        ]
        
        # Pad agent state to 50 features
        remaining_features = 50 - len(agent_state) - len(network_features)
        agent_state.extend(network_features)
        agent_state.extend([0.0] * remaining_features)
        obs.extend(agent_state)
        
        # Objectives and global state (20 features)
        objectives_state = [
            float("compromise_domain_controller" in self.objectives_completed),
            float("exfiltrate_data" in self.objectives_completed),
            float(any(h.host_type == HostType.DOMAIN_CONTROLLER and h.is_compromised 
                     for h in self.hosts.values())),
            self.vulnerability_density,
            self.defense_strength,
        ]
        
        # Pad objectives state to 20 features
        remaining_obj_features = 20 - len(objectives_state)
        objectives_state.extend([0.0] * remaining_obj_features)
        obs.extend(objectives_state)
        
        return np.array(obs, dtype=np.float32)
        
    def _get_info(self) -> Dict:
        """Get additional information about the environment state"""
        return {
            "current_step": self.current_step,
            "agent_position": self.agent_position,
            "compromised_hosts": list(self.compromised_hosts),
            "discovered_hosts": list(self.discovered_hosts),
            "detection_level": self.detection_level,
            "objectives_completed": list(self.objectives_completed),
            "attack_chain_length": len(self.attack_chain),
            "network_size": self.network_size,
            "topology": self.topology.value
        }
        
    def _is_terminated(self) -> bool:
        """Check if the episode should terminate"""
        # Terminate if detected
        if self.detection_level >= self.detection_threshold:
            return True
            
        # Terminate if all objectives completed
        if len(self.objectives_completed) >= len(self.objectives):
            return True
            
        # Terminate if domain controller compromised (major objective)
        if any(h.host_type == HostType.DOMAIN_CONTROLLER and h.is_compromised 
               for h in self.hosts.values()):
            self.objectives_completed.add("compromise_domain_controller")
            return True
            
        return False
        
    def render(self, mode: str = "human"):
        """Render the environment"""
        if mode == "human":
            print(f"\nStep: {self.current_step}/{self.episode_length}")
            print(f"Agent Position: {self.agent_position}")
            print(f"Detection Level: {self.detection_level:.2f}")
            print(f"Compromised Hosts: {len(self.compromised_hosts)}")
            print(f"Discovered Hosts: {len(self.discovered_hosts)}")
            print(f"Objectives Completed: {self.objectives_completed}")
            
    def get_network_info(self) -> Dict:
        """Get detailed network information for analysis"""
        return {
            "hosts": {k: v.__dict__ for k, v in self.hosts.items()},
            "segments": {k: v.__dict__ for k, v in self.segments.items()},
            "network_graph": {
                "nodes": list(self.network_graph.nodes()),
                "edges": list(self.network_graph.edges())
            },
            "topology": self.topology.value,
            "network_size": self.network_size
        }