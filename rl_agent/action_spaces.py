"""
Action Spaces for Different Attack Strategies

This module defines comprehensive action spaces for cybersecurity scenarios,
including discrete and continuous action spaces for various attack strategies.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum, IntEnum
import itertools


class AttackPhase(IntEnum):
    """Attack phases in the cyber kill chain"""
    RECONNAISSANCE = 0
    WEAPONIZATION = 1
    DELIVERY = 2
    EXPLOITATION = 3
    INSTALLATION = 4
    COMMAND_CONTROL = 5
    ACTIONS_OBJECTIVES = 6


class AttackTechnique(IntEnum):
    """MITRE ATT&CK inspired attack techniques"""
    # Reconnaissance
    ACTIVE_SCANNING = 0
    PASSIVE_SCANNING = 1
    OSINT_GATHERING = 2
    
    # Initial Access
    EXPLOIT_PUBLIC_APP = 10
    PHISHING = 11
    SUPPLY_CHAIN = 12
    BRUTE_FORCE = 13
    
    # Execution
    COMMAND_LINE = 20
    POWERSHELL = 21
    WMI = 22
    SCHEDULED_TASK = 23
    
    # Persistence
    REGISTRY_MODIFICATION = 30
    STARTUP_FOLDER = 31
    SERVICE_CREATION = 32
    BACKDOOR = 33
    
    # Privilege Escalation
    EXPLOIT_VULN = 40
    BYPASS_UAC = 41
    TOKEN_MANIPULATION = 42
    
    # Defense Evasion
    OBFUSCATION = 50
    PROCESS_INJECTION = 51
    MASQUERADING = 52
    DISABLE_SECURITY = 53
    
    # Credential Access
    CREDENTIAL_DUMPING = 60
    KEYLOGGING = 61
    NETWORK_SNIFFING = 62
    
    # Discovery
    SYSTEM_INFO = 70
    NETWORK_DISCOVERY = 71
    PROCESS_DISCOVERY = 72
    FILE_DISCOVERY = 73
    
    # Lateral Movement
    REMOTE_SERVICES = 80
    EXPLOITATION_REMOTE = 81
    INTERNAL_SPEARPHISH = 82
    
    # Collection
    DATA_STAGING = 90
    SCREEN_CAPTURE = 91
    KEYLOGGING_COLLECT = 92
    
    # Exfiltration
    DATA_TRANSFER = 100
    PHYSICAL_MEDIA = 101
    C2_CHANNEL = 102


class TargetType(IntEnum):
    """Types of targets for attacks"""
    HOST = 0
    SERVICE = 1
    USER = 2
    NETWORK_SEGMENT = 3
    DATA_SOURCE = 4
    CREDENTIAL = 5


@dataclass
class AttackAction:
    """Structured representation of an attack action"""
    technique: AttackTechnique
    target_type: TargetType
    target_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    priority: float = 0.5
    stealth_level: float = 0.5
    resource_cost: float = 0.1


class AttackActionSpace:
    """
    Comprehensive action space for cybersecurity attack scenarios
    
    Supports both discrete and continuous action spaces with hierarchical
    action selection and parameterized actions.
    """
    
    def __init__(
        self,
        action_type: str = "discrete",
        enable_hierarchical: bool = True,
        enable_parameterized: bool = True,
        max_targets: int = 50,
        enable_multi_action: bool = False
    ):
        """
        Initialize the attack action space
        
        Args:
            action_type: Type of action space ("discrete", "continuous", "hybrid")
            enable_hierarchical: Enable hierarchical action selection
            enable_parameterized: Enable parameterized actions
            max_targets: Maximum number of targets in environment
            enable_multi_action: Enable multiple simultaneous actions
        """
        self.action_type = action_type
        self.enable_hierarchical = enable_hierarchical
        self.enable_parameterized = enable_parameterized
        self.max_targets = max_targets
        self.enable_multi_action = enable_multi_action
        
        # Create action space based on configuration
        self.action_space = self._create_action_space()
        
        # Action mappings
        self.technique_map = {technique.value: technique for technique in AttackTechnique}
        self.target_type_map = {target.value: target for target in TargetType}
        
        # Valid technique-target combinations
        self.valid_combinations = self._create_valid_combinations()
        
    def _create_action_space(self) -> gym.Space:
        """Create the appropriate action space based on configuration"""
        if self.action_type == "discrete":
            return self._create_discrete_space()
        elif self.action_type == "continuous":
            return self._create_continuous_space()
        elif self.action_type == "hybrid":
            return self._create_hybrid_space()
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")
            
    def _create_discrete_space(self) -> gym.Space:
        """Create discrete action space"""
        if self.enable_hierarchical:
            return self._create_hierarchical_discrete_space()
        else:
            return self._create_flat_discrete_space()
            
    def _create_flat_discrete_space(self) -> spaces.Discrete:
        """Create flat discrete action space"""
        # Simple discrete space with all valid technique-target combinations
        num_actions = len(self.valid_combinations)
        if self.enable_multi_action:
            # Add combinations of multiple actions
            num_actions += len(self.valid_combinations) * 2  # Pairs
            
        return spaces.Discrete(num_actions)
        
    def _create_hierarchical_discrete_space(self) -> spaces.Dict:
        """Create hierarchical discrete action space"""
        action_dict = {
            "technique": spaces.Discrete(len(AttackTechnique)),
            "target_type": spaces.Discrete(len(TargetType)),
            "target_id": spaces.Discrete(self.max_targets),
        }
        
        if self.enable_parameterized:
            action_dict.update({
                "priority": spaces.Discrete(5),  # 0-4 priority levels
                "stealth_level": spaces.Discrete(5),  # 0-4 stealth levels
                "resource_cost": spaces.Discrete(5),  # 0-4 resource levels
            })
            
        if self.enable_multi_action:
            action_dict["num_actions"] = spaces.Discrete(3)  # 1-3 simultaneous actions
            
        return spaces.Dict(action_dict)
        
    def _create_continuous_space(self) -> spaces.Box:
        """Create continuous action space"""
        # Continuous space for soft action selection
        if self.enable_hierarchical:
            low = []
            high = []
            
            # Technique probabilities
            low.extend([0.0] * len(AttackTechnique))
            high.extend([1.0] * len(AttackTechnique))
            
            # Target type probabilities  
            low.extend([0.0] * len(TargetType))
            high.extend([1.0] * len(TargetType))
            
            # Target selection (continuous index)
            low.append(0.0)
            high.append(1.0)
            
            if self.enable_parameterized:
                # Priority, stealth, resource cost
                low.extend([0.0, 0.0, 0.0])
                high.extend([1.0, 1.0, 1.0])
                
            return spaces.Box(
                low=np.array(low, dtype=np.float32),
                high=np.array(high, dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Simple continuous action space
            dim = len(self.valid_combinations)
            if self.enable_parameterized:
                dim += 3  # priority, stealth, resource
                
            return spaces.Box(
                low=0.0,
                high=1.0,
                shape=(dim,),
                dtype=np.float32
            )
            
    def _create_hybrid_space(self) -> spaces.Dict:
        """Create hybrid discrete-continuous action space"""
        action_dict = {
            "discrete": self._create_hierarchical_discrete_space(),
            "continuous": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(10,),  # Additional continuous parameters
                dtype=np.float32
            )
        }
        return spaces.Dict(action_dict)
        
    def _create_valid_combinations(self) -> List[Tuple[AttackTechnique, TargetType]]:
        """Create valid technique-target combinations"""
        valid_combinations = []
        
        # Define valid combinations based on logical attack patterns
        combination_rules = {
            # Reconnaissance techniques
            AttackTechnique.ACTIVE_SCANNING: [TargetType.HOST, TargetType.SERVICE, TargetType.NETWORK_SEGMENT],
            AttackTechnique.PASSIVE_SCANNING: [TargetType.HOST, TargetType.SERVICE, TargetType.NETWORK_SEGMENT],
            AttackTechnique.OSINT_GATHERING: [TargetType.HOST, TargetType.USER],
            
            # Initial Access
            AttackTechnique.EXPLOIT_PUBLIC_APP: [TargetType.SERVICE, TargetType.HOST],
            AttackTechnique.PHISHING: [TargetType.USER],
            AttackTechnique.SUPPLY_CHAIN: [TargetType.HOST, TargetType.SERVICE],
            AttackTechnique.BRUTE_FORCE: [TargetType.SERVICE, TargetType.CREDENTIAL],
            
            # Execution
            AttackTechnique.COMMAND_LINE: [TargetType.HOST],
            AttackTechnique.POWERSHELL: [TargetType.HOST],
            AttackTechnique.WMI: [TargetType.HOST],
            AttackTechnique.SCHEDULED_TASK: [TargetType.HOST],
            
            # Persistence
            AttackTechnique.REGISTRY_MODIFICATION: [TargetType.HOST],
            AttackTechnique.STARTUP_FOLDER: [TargetType.HOST],
            AttackTechnique.SERVICE_CREATION: [TargetType.HOST],
            AttackTechnique.BACKDOOR: [TargetType.HOST],
            
            # Privilege Escalation
            AttackTechnique.EXPLOIT_VULN: [TargetType.HOST, TargetType.SERVICE],
            AttackTechnique.BYPASS_UAC: [TargetType.HOST],
            AttackTechnique.TOKEN_MANIPULATION: [TargetType.HOST],
            
            # Defense Evasion
            AttackTechnique.OBFUSCATION: [TargetType.HOST],
            AttackTechnique.PROCESS_INJECTION: [TargetType.HOST],
            AttackTechnique.MASQUERADING: [TargetType.HOST],
            AttackTechnique.DISABLE_SECURITY: [TargetType.HOST, TargetType.SERVICE],
            
            # Credential Access
            AttackTechnique.CREDENTIAL_DUMPING: [TargetType.HOST],
            AttackTechnique.KEYLOGGING: [TargetType.HOST, TargetType.USER],
            AttackTechnique.NETWORK_SNIFFING: [TargetType.NETWORK_SEGMENT],
            
            # Discovery
            AttackTechnique.SYSTEM_INFO: [TargetType.HOST],
            AttackTechnique.NETWORK_DISCOVERY: [TargetType.NETWORK_SEGMENT],
            AttackTechnique.PROCESS_DISCOVERY: [TargetType.HOST],
            AttackTechnique.FILE_DISCOVERY: [TargetType.HOST, TargetType.DATA_SOURCE],
            
            # Lateral Movement
            AttackTechnique.REMOTE_SERVICES: [TargetType.HOST, TargetType.SERVICE],
            AttackTechnique.EXPLOITATION_REMOTE: [TargetType.HOST, TargetType.SERVICE],
            AttackTechnique.INTERNAL_SPEARPHISH: [TargetType.USER],
            
            # Collection
            AttackTechnique.DATA_STAGING: [TargetType.HOST, TargetType.DATA_SOURCE],
            AttackTechnique.SCREEN_CAPTURE: [TargetType.HOST, TargetType.USER],
            AttackTechnique.KEYLOGGING_COLLECT: [TargetType.HOST, TargetType.USER],
            
            # Exfiltration
            AttackTechnique.DATA_TRANSFER: [TargetType.DATA_SOURCE, TargetType.HOST],
            AttackTechnique.PHYSICAL_MEDIA: [TargetType.HOST],
            AttackTechnique.C2_CHANNEL: [TargetType.HOST, TargetType.NETWORK_SEGMENT],
        }
        
        for technique, target_types in combination_rules.items():
            for target_type in target_types:
                valid_combinations.append((technique, target_type))
                
        return valid_combinations
        
    def sample_action(self) -> Union[int, np.ndarray, Dict]:
        """Sample a random action from the action space"""
        return self.action_space.sample()
        
    def decode_action(self, action: Union[int, np.ndarray, Dict]) -> AttackAction:
        """Decode raw action into structured AttackAction"""
        if self.action_type == "discrete" and isinstance(action, (int, np.integer)):
            return self._decode_discrete_action(action)
        elif self.action_type == "continuous" and isinstance(action, np.ndarray):
            return self._decode_continuous_action(action)
        elif self.action_type == "hybrid" and isinstance(action, dict):
            return self._decode_hybrid_action(action)
        elif isinstance(action, dict):
            return self._decode_hierarchical_action(action)
        else:
            raise ValueError(f"Invalid action format: {action}")
            
    def _decode_discrete_action(self, action: int) -> AttackAction:
        """Decode discrete action"""
        if action < len(self.valid_combinations):
            technique, target_type = self.valid_combinations[action]
            return AttackAction(
                technique=technique,
                target_type=target_type,
                priority=0.5,
                stealth_level=0.5,
                resource_cost=0.1
            )
        else:
            # Multi-action case
            base_action = action - len(self.valid_combinations)
            primary_idx = base_action % len(self.valid_combinations)
            technique, target_type = self.valid_combinations[primary_idx]
            
            return AttackAction(
                technique=technique,
                target_type=target_type,
                priority=0.7,  # Higher priority for multi-actions
                stealth_level=0.3,  # Lower stealth for multi-actions
                resource_cost=0.2
            )
            
    def _decode_hierarchical_action(self, action: Dict) -> AttackAction:
        """Decode hierarchical discrete action"""
        technique = self.technique_map[action["technique"]]
        target_type = self.target_type_map[action["target_type"]]
        
        # Validate combination
        if (technique, target_type) not in self.valid_combinations:
            # Fall back to first valid combination for this technique
            valid_targets = [
                tt for tech, tt in self.valid_combinations if tech == technique
            ]
            if valid_targets:
                target_type = valid_targets[0]
            else:
                # Ultimate fallback
                technique, target_type = self.valid_combinations[0]
                
        attack_action = AttackAction(
            technique=technique,
            target_type=target_type,
            target_id=str(action["target_id"]) if action["target_id"] < self.max_targets else None
        )
        
        if self.enable_parameterized:
            attack_action.priority = action.get("priority", 2) / 4.0
            attack_action.stealth_level = action.get("stealth_level", 2) / 4.0
            attack_action.resource_cost = action.get("resource_cost", 2) / 4.0
            
        return attack_action
        
    def _decode_continuous_action(self, action: np.ndarray) -> AttackAction:
        """Decode continuous action"""
        if self.enable_hierarchical:
            # Split action vector
            offset = 0
            
            # Technique selection (softmax over techniques)
            technique_probs = action[offset:offset + len(AttackTechnique)]
            technique_idx = np.argmax(technique_probs)
            technique = self.technique_map[technique_idx]
            offset += len(AttackTechnique)
            
            # Target type selection
            target_type_probs = action[offset:offset + len(TargetType)]
            target_type_idx = np.argmax(target_type_probs)
            target_type = self.target_type_map[target_type_idx]
            offset += len(TargetType)
            
            # Target ID (continuous to discrete mapping)
            target_id_continuous = action[offset]
            target_id = int(target_id_continuous * self.max_targets)
            offset += 1
            
            attack_action = AttackAction(
                technique=technique,
                target_type=target_type,
                target_id=str(target_id) if target_id < self.max_targets else None
            )
            
            if self.enable_parameterized and offset < len(action):
                attack_action.priority = action[offset]
                attack_action.stealth_level = action[offset + 1] if offset + 1 < len(action) else 0.5
                attack_action.resource_cost = action[offset + 2] if offset + 2 < len(action) else 0.1
                
            return attack_action
        else:
            # Simple continuous action space
            action_idx = np.argmax(action[:len(self.valid_combinations)])
            technique, target_type = self.valid_combinations[action_idx]
            
            attack_action = AttackAction(
                technique=technique,
                target_type=target_type
            )
            
            if self.enable_parameterized:
                offset = len(self.valid_combinations)
                if offset < len(action):
                    attack_action.priority = action[offset]
                if offset + 1 < len(action):
                    attack_action.stealth_level = action[offset + 1]
                if offset + 2 < len(action):
                    attack_action.resource_cost = action[offset + 2]
                    
            return attack_action
            
    def _decode_hybrid_action(self, action: Dict) -> AttackAction:
        """Decode hybrid action"""
        # Use discrete part for main action selection
        discrete_action = self._decode_hierarchical_action(action["discrete"])
        
        # Use continuous part for fine-tuning parameters
        continuous_params = action["continuous"]
        if len(continuous_params) >= 3:
            discrete_action.priority = continuous_params[0]
            discrete_action.stealth_level = continuous_params[1]
            discrete_action.resource_cost = continuous_params[2]
            
        return discrete_action
        
    def encode_action(self, attack_action: AttackAction) -> Union[int, np.ndarray, Dict]:
        """Encode AttackAction back to raw action format"""
        if self.action_type == "discrete":
            return self._encode_discrete_action(attack_action)
        elif self.action_type == "continuous":
            return self._encode_continuous_action(attack_action)
        elif self.action_type == "hybrid":
            return self._encode_hybrid_action(attack_action)
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")
            
    def _encode_discrete_action(self, attack_action: AttackAction) -> int:
        """Encode to discrete action"""
        combination = (attack_action.technique, attack_action.target_type)
        if combination in self.valid_combinations:
            return self.valid_combinations.index(combination)
        else:
            return 0  # Default action
            
    def _encode_continuous_action(self, attack_action: AttackAction) -> np.ndarray:
        """Encode to continuous action"""
        if self.enable_hierarchical:
            action = []
            
            # Technique one-hot
            technique_vector = np.zeros(len(AttackTechnique))
            technique_vector[attack_action.technique.value] = 1.0
            action.extend(technique_vector)
            
            # Target type one-hot
            target_type_vector = np.zeros(len(TargetType))
            target_type_vector[attack_action.target_type.value] = 1.0
            action.extend(target_type_vector)
            
            # Target ID
            if attack_action.target_id:
                target_id_norm = min(1.0, int(attack_action.target_id) / self.max_targets)
            else:
                target_id_norm = 0.0
            action.append(target_id_norm)
            
            if self.enable_parameterized:
                action.extend([
                    attack_action.priority,
                    attack_action.stealth_level,
                    attack_action.resource_cost
                ])
                
            return np.array(action, dtype=np.float32)
        else:
            # Simple continuous encoding
            action = np.zeros(len(self.valid_combinations))
            combination = (attack_action.technique, attack_action.target_type)
            if combination in self.valid_combinations:
                idx = self.valid_combinations.index(combination)
                action[idx] = 1.0
                
            if self.enable_parameterized:
                action = np.concatenate([
                    action,
                    [attack_action.priority, attack_action.stealth_level, attack_action.resource_cost]
                ])
                
            return action.astype(np.float32)
            
    def _encode_hybrid_action(self, attack_action: AttackAction) -> Dict:
        """Encode to hybrid action"""
        discrete_part = {
            "technique": attack_action.technique.value,
            "target_type": attack_action.target_type.value,
            "target_id": int(attack_action.target_id) if attack_action.target_id else 0
        }
        
        if self.enable_parameterized:
            discrete_part.update({
                "priority": int(attack_action.priority * 4),
                "stealth_level": int(attack_action.stealth_level * 4),
                "resource_cost": int(attack_action.resource_cost * 4)
            })
            
        continuous_part = np.array([
            attack_action.priority,
            attack_action.stealth_level,
            attack_action.resource_cost,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding to 10 dimensions
        ], dtype=np.float32)
        
        return {
            "discrete": discrete_part,
            "continuous": continuous_part
        }
        
    def is_valid_action(self, action: Union[int, np.ndarray, Dict]) -> bool:
        """Check if an action is valid"""
        try:
            attack_action = self.decode_action(action)
            combination = (attack_action.technique, attack_action.target_type)
            return combination in self.valid_combinations
        except:
            return False
            
    def get_valid_actions(self, context: Optional[Dict] = None) -> List[Union[int, np.ndarray, Dict]]:
        """Get list of valid actions given context"""
        if self.action_type == "discrete":
            return list(range(len(self.valid_combinations)))
        else:
            # For continuous/hybrid, return example valid actions
            valid_actions = []
            for i, (technique, target_type) in enumerate(self.valid_combinations[:10]):
                attack_action = AttackAction(
                    technique=technique,
                    target_type=target_type,
                    priority=0.5,
                    stealth_level=0.5,
                    resource_cost=0.1
                )
                valid_actions.append(self.encode_action(attack_action))
            return valid_actions
            
    def get_action_mask(self, context: Optional[Dict] = None) -> np.ndarray:
        """Get action mask for invalid actions"""
        if self.action_type == "discrete":
            mask = np.ones(self.action_space.n, dtype=bool)
            
            # Apply context-based masking if provided
            if context:
                # Example: mask actions based on current capabilities
                if context.get("compromised_hosts", 0) == 0:
                    # If no hosts compromised, can't do lateral movement
                    for i, (technique, _) in enumerate(self.valid_combinations):
                        if technique in [AttackTechnique.REMOTE_SERVICES, 
                                       AttackTechnique.EXPLOITATION_REMOTE]:
                            mask[i] = False
                            
            return mask
        else:
            # For continuous spaces, return all True
            return np.ones(self.action_space.shape[0], dtype=bool)
            
    def get_action_info(self) -> Dict[str, Any]:
        """Get information about the action space"""
        return {
            "action_type": self.action_type,
            "action_space": str(self.action_space),
            "num_techniques": len(AttackTechnique),
            "num_target_types": len(TargetType),
            "num_valid_combinations": len(self.valid_combinations),
            "enable_hierarchical": self.enable_hierarchical,
            "enable_parameterized": self.enable_parameterized,
            "enable_multi_action": self.enable_multi_action,
            "max_targets": self.max_targets
        }