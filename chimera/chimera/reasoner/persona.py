"""
Persona management system for character-driven decision making
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PersonaConfig:
    """Configuration for a specific persona"""
    
    name: str
    description: str
    risk_tolerance: float  # 0.0 (very cautious) to 1.0 (very aggressive)
    stealth_priority: float  # 0.0 (speed over stealth) to 1.0 (stealth over speed)
    creativity_level: float  # 0.0 (conservative) to 1.0 (highly creative)
    collaboration_preference: float  # 0.0 (autonomous) to 1.0 (seeks advice)
    
    # Decision-making preferences
    planning_depth: int  # How many steps ahead to plan
    information_gathering_threshold: float  # When to stop gathering info
    exploitation_aggressiveness: float  # How aggressively to exploit findings
    
    # Behavioral traits
    preferred_tools: list
    avoided_tools: list
    communication_style: str  # formal, casual, technical
    reporting_detail_level: str  # minimal, standard, comprehensive
    
    # Learning preferences
    learning_rate: float
    knowledge_retention: float
    adaptation_speed: float

class PersonaManager:
    """
    Manages different personas and their behavioral characteristics
    Influences decision-making throughout the Chimera system
    """
    
    def __init__(self, config_dir: str = "./configs/personas/"):
        self.config_dir = Path(config_dir)
        self.personas: Dict[str, PersonaConfig] = {}
        self.current_persona: Optional[PersonaConfig] = None
        
    async def initialize(self):
        """Initialize persona manager and load available personas"""
        await self._load_default_personas()
        await self._load_custom_personas()
        logger.info(f"Loaded {len(self.personas)} personas")
        
    async def load_persona(self, persona_name: str) -> bool:
        """Load and activate a specific persona"""
        if persona_name not in self.personas:
            logger.error(f"Persona not found: {persona_name}")
            return False
            
        self.current_persona = self.personas[persona_name]
        logger.info(f"Activated persona: {persona_name}")
        return True
        
    def get_current_persona(self) -> Optional[PersonaConfig]:
        """Get the currently active persona"""
        return self.current_persona
        
    def get_available_personas(self) -> list:
        """Get list of available persona names"""
        return list(self.personas.keys())
        
    def should_take_action(self, risk_level: float, confidence: float) -> bool:
        """
        Decide whether to take an action based on persona's risk tolerance
        """
        if not self.current_persona:
            return confidence > 0.7  # Default threshold
            
        # Adjust threshold based on risk tolerance and confidence
        threshold = (1.0 - self.current_persona.risk_tolerance) * 0.5 + 0.3
        adjusted_confidence = confidence * (1.0 + self.current_persona.creativity_level * 0.2)
        
        return adjusted_confidence > threshold
        
    def get_stealth_preference(self) -> float:
        """Get current persona's stealth preference"""
        if not self.current_persona:
            return 0.7  # Default moderate stealth
        return self.current_persona.stealth_priority
        
    def get_planning_depth(self) -> int:
        """Get how many steps ahead this persona plans"""
        if not self.current_persona:
            return 3  # Default planning depth
        return self.current_persona.planning_depth
        
    def should_seek_collaboration(self, task_complexity: float) -> bool:
        """Decide whether to seek LLM collaboration for a task"""
        if not self.current_persona:
            return task_complexity > 0.6
            
        collaboration_threshold = 1.0 - self.current_persona.collaboration_preference
        return task_complexity > collaboration_threshold
        
    def get_tool_preference(self, available_tools: list) -> list:
        """Get preferred tools based on persona"""
        if not self.current_persona:
            return available_tools
            
        # Prioritize preferred tools
        preferred = [tool for tool in available_tools 
                    if any(pref in tool.lower() 
                          for pref in self.current_persona.preferred_tools)]
        
        # Filter out avoided tools
        avoided = self.current_persona.avoided_tools
        filtered = [tool for tool in available_tools
                   if not any(avoid in tool.lower() for avoid in avoided)]
        
        # Combine: preferred first, then others
        result = preferred + [tool for tool in filtered if tool not in preferred]
        return result
        
    def get_communication_style(self) -> Dict[str, Any]:
        """Get communication preferences for reports and interactions"""
        if not self.current_persona:
            return {
                "style": "technical",
                "detail_level": "standard",
                "formality": "formal"
            }
            
        return {
            "style": self.current_persona.communication_style,
            "detail_level": self.current_persona.reporting_detail_level,
            "formality": "formal" if self.current_persona.risk_tolerance < 0.5 else "casual"
        }
        
    def adapt_learning_rate(self, success_rate: float) -> float:
        """Adapt learning rate based on recent success and persona"""
        if not self.current_persona:
            return 0.1
            
        base_rate = self.current_persona.learning_rate
        adaptation = self.current_persona.adaptation_speed
        
        # Increase learning rate if success rate is low
        if success_rate < 0.5:
            return min(base_rate * (1 + adaptation), 1.0)
        else:
            return max(base_rate * (1 - adaptation * 0.5), 0.01)
            
    async def _load_default_personas(self):
        """Load default built-in personas"""
        default_personas = {
            "cautious": PersonaConfig(
                name="cautious",
                description="Conservative approach prioritizing stealth and safety",
                risk_tolerance=0.2,
                stealth_priority=0.9,
                creativity_level=0.3,
                collaboration_preference=0.8,
                planning_depth=5,
                information_gathering_threshold=0.8,
                exploitation_aggressiveness=0.2,
                preferred_tools=["nmap", "gobuster", "nuclei"],
                avoided_tools=["sqlmap", "metasploit"],
                communication_style="formal",
                reporting_detail_level="comprehensive",
                learning_rate=0.05,
                knowledge_retention=0.95,
                adaptation_speed=0.2
            ),
            
            "balanced": PersonaConfig(
                name="balanced",
                description="Balanced approach between stealth, speed, and effectiveness",
                risk_tolerance=0.5,
                stealth_priority=0.6,
                creativity_level=0.5,
                collaboration_preference=0.5,
                planning_depth=3,
                information_gathering_threshold=0.6,
                exploitation_aggressiveness=0.5,
                preferred_tools=["nmap", "gobuster", "nuclei", "burpsuite"],
                avoided_tools=[],
                communication_style="technical",
                reporting_detail_level="standard",
                learning_rate=0.1,
                knowledge_retention=0.85,
                adaptation_speed=0.5
            ),
            
            "aggressive": PersonaConfig(
                name="aggressive",
                description="Fast and thorough testing with higher risk tolerance",
                risk_tolerance=0.8,
                stealth_priority=0.3,
                creativity_level=0.7,
                collaboration_preference=0.3,
                planning_depth=2,
                information_gathering_threshold=0.4,
                exploitation_aggressiveness=0.8,
                preferred_tools=["sqlmap", "metasploit", "burpsuite", "ffuf"],
                avoided_tools=[],
                communication_style="casual",
                reporting_detail_level="minimal",
                learning_rate=0.15,
                knowledge_retention=0.75,
                adaptation_speed=0.8
            ),
            
            "creative": PersonaConfig(
                name="creative",
                description="Innovative approaches and novel attack vectors",
                risk_tolerance=0.6,
                stealth_priority=0.5,
                creativity_level=0.9,
                collaboration_preference=0.7,
                planning_depth=4,
                information_gathering_threshold=0.5,
                exploitation_aggressiveness=0.6,
                preferred_tools=["custom_scripts", "burpsuite", "nuclei"],
                avoided_tools=["automated_scanners"],
                communication_style="technical",
                reporting_detail_level="comprehensive",
                learning_rate=0.2,
                knowledge_retention=0.8,
                adaptation_speed=0.9
            ),
            
            "stealth_focused": PersonaConfig(
                name="stealth_focused",
                description="Maximum stealth and evasion, minimal footprint",
                risk_tolerance=0.1,
                stealth_priority=1.0,
                creativity_level=0.4,
                collaboration_preference=0.6,
                planning_depth=6,
                information_gathering_threshold=0.9,
                exploitation_aggressiveness=0.1,
                preferred_tools=["passive_recon", "osint"],
                avoided_tools=["automated_scanners", "sqlmap", "metasploit"],
                communication_style="formal",
                reporting_detail_level="comprehensive",
                learning_rate=0.03,
                knowledge_retention=0.98,
                adaptation_speed=0.1
            )
        }
        
        self.personas.update(default_personas)
        
    async def _load_custom_personas(self):
        """Load custom personas from YAML files"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            await self._create_sample_persona_files()
            
        for persona_file in self.config_dir.glob("*.yaml"):
            try:
                with open(persona_file, 'r') as f:
                    data = yaml.safe_load(f)
                    
                persona = PersonaConfig(**data)
                self.personas[persona.name] = persona
                logger.debug(f"Loaded custom persona: {persona.name}")
                
            except Exception as e:
                logger.error(f"Error loading persona file {persona_file}: {e}")
                
    async def _create_sample_persona_files(self):
        """Create sample persona YAML files"""
        sample_personas = ["cautious", "balanced", "aggressive"]
        
        for persona_name in sample_personas:
            if persona_name in self.personas:
                persona = self.personas[persona_name]
                
                # Convert to dict for YAML serialization
                persona_dict = {
                    "name": persona.name,
                    "description": persona.description,
                    "risk_tolerance": persona.risk_tolerance,
                    "stealth_priority": persona.stealth_priority,
                    "creativity_level": persona.creativity_level,
                    "collaboration_preference": persona.collaboration_preference,
                    "planning_depth": persona.planning_depth,
                    "information_gathering_threshold": persona.information_gathering_threshold,
                    "exploitation_aggressiveness": persona.exploitation_aggressiveness,
                    "preferred_tools": persona.preferred_tools,
                    "avoided_tools": persona.avoided_tools,
                    "communication_style": persona.communication_style,
                    "reporting_detail_level": persona.reporting_detail_level,
                    "learning_rate": persona.learning_rate,
                    "knowledge_retention": persona.knowledge_retention,
                    "adaptation_speed": persona.adaptation_speed
                }
                
                # Write to file
                persona_file = self.config_dir / f"{persona_name}.yaml"
                with open(persona_file, 'w') as f:
                    yaml.dump(persona_dict, f, default_flow_style=False, indent=2)
                    
                logger.info(f"Created sample persona file: {persona_file}")