"""
Neuroplastic reasoning engine for Chimera
Combines persona-driven decision making with adaptive learning
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..core.events import EventSystem, EventType, EventEmitter, EventSubscriber
from ..memory.knowledge_graph import KnowledgeGraph
from .persona import PersonaManager, PersonaConfig
from .decision_tree import DecisionEngine, DecisionType, DecisionContext

logger = logging.getLogger(__name__)

@dataclass
class ReasoningResult:
    """Result of a reasoning operation"""
    
    decision: Dict[str, Any]
    confidence: float
    reasoning_path: List[str]
    persona_influence: Dict[str, Any]
    knowledge_used: List[str]
    timestamp: datetime
    
class AnalysisPlan:
    """Plan for target analysis"""
    
    def __init__(self, target: str, phases: List[Dict[str, Any]], estimated_duration: int):
        self.target = target
        self.phases = phases
        self.estimated_duration = estimated_duration
        self.created_at = datetime.utcnow()
        
class CommandInterpretation:
    """Interpretation of a user command"""
    
    def __init__(self, action: str, parameters: Dict[str, Any], confidence: float):
        self.action = action
        self.parameters = parameters
        self.confidence = confidence

class NeuroplasticReasoner(EventEmitter, EventSubscriber):
    """
    Advanced reasoning engine that adapts its decision-making patterns
    based on outcomes and persona characteristics
    """
    
    def __init__(self, config, event_system: EventSystem, knowledge_graph: KnowledgeGraph):
        EventEmitter.__init__(self, event_system, "NeuroplasticReasoner")
        EventSubscriber.__init__(self, event_system)
        
        self.config = config
        self.knowledge_graph = knowledge_graph
        
        # Core components
        self.persona_manager = PersonaManager(
            config.get("persona.persona_config_dir", "./configs/personas/")
        )
        self.decision_engine = DecisionEngine(
            learning_rate=config.get("reasoner.learning_rate", 0.1)
        )
        
        # Reasoning state
        self.reasoning_history: List[ReasoningResult] = []
        self.active_contexts: Dict[str, DecisionContext] = {}
        self.performance_metrics: Dict[str, float] = {}
        
        # Configuration
        self.confidence_threshold = config.get("reasoner.decision_confidence_threshold", 0.7)
        self.max_reasoning_depth = config.get("reasoner.max_reasoning_depth", 5)
        self.knowledge_weight_decay = config.get("reasoner.knowledge_weight_decay", 0.95)
        
        # Learning and adaptation
        self.learning_enabled = True
        self.adaptation_rate = 0.1
        self.recent_success_rate = 0.5
        
    async def initialize(self):
        """Initialize the reasoning engine"""
        try:
            # Initialize persona manager
            await self.persona_manager.initialize()
            
            # Load default persona
            default_persona = self.config.get("persona.default", "balanced")
            await self.persona_manager.load_persona(default_persona)
            
            # Subscribe to outcome events for learning
            await self.subscribe_to_event(EventType.TASK_COMPLETE, self._on_task_complete)
            await self.subscribe_to_event(EventType.TASK_FAILED, self._on_task_failed)
            await self.subscribe_to_event(EventType.KNOWLEDGE_LEARNED, self._on_knowledge_learned)
            
            # Initialize performance tracking
            self._initialize_performance_metrics()
            
            await self.emit_event(
                EventType.REASONING_START,
                {"message": "Neuroplastic reasoner initialized", "persona": default_persona}
            )
            
            logger.info("Neuroplastic reasoner initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize reasoner: {e}")
            raise
            
    async def load_persona(self, persona_name: str) -> bool:
        """Load and activate a specific persona"""
        success = await self.persona_manager.load_persona(persona_name)
        
        if success:
            # Adapt decision engine learning rate based on persona
            persona = self.persona_manager.get_current_persona()
            if persona:
                self.decision_engine.learning_rate = persona.learning_rate
                
            await self.emit_event(
                EventType.REASONING_START,
                {"action": "persona_changed", "persona": persona_name}
            )
            
        return success
        
    async def make_decision(self, decision_type: DecisionType, context_data: Dict[str, Any]) -> ReasoningResult:
        """Make a reasoned decision based on context and persona"""
        
        start_time = datetime.utcnow()
        
        try:
            # Create decision context
            context = await self._create_decision_context(context_data)
            
            # Get persona influence
            persona_influence = await self._calculate_persona_influence(decision_type, context)
            
            # Retrieve relevant knowledge
            relevant_knowledge = await self._retrieve_relevant_knowledge(decision_type, context_data)
            
            # Make the core decision
            decision = self.decision_engine.make_decision(decision_type, context)
            
            # Calculate overall confidence
            base_confidence = decision.get("confidence", 0.5)
            persona_confidence = persona_influence.get("confidence_modifier", 1.0)
            knowledge_confidence = self._calculate_knowledge_confidence(relevant_knowledge)
            
            overall_confidence = min(base_confidence * persona_confidence * knowledge_confidence, 1.0)
            
            # Enhance decision with reasoning
            enhanced_decision = await self._enhance_decision_with_reasoning(
                decision, context, relevant_knowledge, persona_influence
            )
            
            # Create reasoning result
            result = ReasoningResult(
                decision=enhanced_decision,
                confidence=overall_confidence,
                reasoning_path=decision.get("path", []),
                persona_influence=persona_influence,
                knowledge_used=[k.get("id", "") for k in relevant_knowledge],
                timestamp=start_time
            )
            
            # Store reasoning history
            self.reasoning_history.append(result)
            self._prune_reasoning_history()
            
            await self.emit_event(
                EventType.DECISION_MADE,
                {
                    "decision_type": decision_type.value,
                    "confidence": overall_confidence,
                    "reasoning_duration": (datetime.utcnow() - start_time).total_seconds()
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            
            # Return fallback decision
            fallback_decision = {
                "type": decision_type.value,
                "action": "proceed_with_caution",
                "confidence": 0.3,
                "error": str(e)
            }
            
            return ReasoningResult(
                decision=fallback_decision,
                confidence=0.3,
                reasoning_path=["error_fallback"],
                persona_influence={},
                knowledge_used=[],
                timestamp=start_time
            )
            
    async def plan_analysis(self, target: str) -> AnalysisPlan:
        """Plan a comprehensive analysis approach for a target"""
        
        # Get current persona preferences
        persona = self.persona_manager.get_current_persona()
        if not persona:
            raise RuntimeError("No persona loaded")
            
        # Retrieve knowledge about target analysis
        knowledge_query = f"target analysis methodology {target}"
        relevant_knowledge = await self.knowledge_graph.query_knowledge(knowledge_query)
        
        # Create analysis phases based on persona and knowledge
        phases = []
        
        # Phase 1: Information Gathering
        info_gathering_phase = {
            "name": "information_gathering",
            "description": "Passive reconnaissance and OSINT collection",
            "techniques": ["whois_lookup", "dns_enumeration", "search_engine_dorking"],
            "estimated_duration": 30 if persona.stealth_priority > 0.7 else 15,
            "stealth_level": persona.stealth_priority
        }
        
        # Enhance with persona preferences
        if persona.stealth_priority > 0.8:
            info_gathering_phase["techniques"].extend(["social_media_analysis", "breach_database_lookup"])
        
        if persona.creativity_level > 0.7:
            info_gathering_phase["techniques"].extend(["alternative_data_sources", "creative_dorking"])
            
        phases.append(info_gathering_phase)
        
        # Phase 2: Active Reconnaissance
        if persona.risk_tolerance > 0.3:
            active_recon_phase = {
                "name": "active_reconnaissance",
                "description": "Active scanning and enumeration",
                "techniques": ["port_scanning", "service_enumeration", "subdomain_bruteforcing"],
                "estimated_duration": 45 if persona.stealth_priority > 0.6 else 20,
                "stealth_level": max(persona.stealth_priority - 0.2, 0.1)
            }
            
            # Adjust techniques based on persona
            if persona.risk_tolerance > 0.7:
                active_recon_phase["techniques"].extend(["vulnerability_scanning", "web_crawling"])
                
            phases.append(active_recon_phase)
            
        # Phase 3: Vulnerability Assessment
        if persona.risk_tolerance > 0.4:
            vuln_assessment_phase = {
                "name": "vulnerability_assessment",
                "description": "Identify potential security weaknesses",
                "techniques": ["automated_scanning", "manual_testing", "configuration_review"],
                "estimated_duration": 60 if persona.planning_depth > 3 else 30,
                "stealth_level": max(persona.stealth_priority - 0.3, 0.1)
            }
            
            phases.append(vuln_assessment_phase)
            
        # Calculate total estimated duration
        total_duration = sum(phase["estimated_duration"] for phase in phases)
        
        # Add buffer based on persona's planning depth
        buffer_factor = 1 + (persona.planning_depth * 0.1)
        total_duration = int(total_duration * buffer_factor)
        
        plan = AnalysisPlan(target, phases, total_duration)
        
        await self.emit_event(
            EventType.PLAN_CREATED,
            {
                "target": target,
                "phases": len(phases),
                "estimated_duration": total_duration,
                "persona": persona.name
            }
        )
        
        return plan
        
    async def interpret_command(self, command: str) -> CommandInterpretation:
        """Interpret a natural language command"""
        
        command_lower = command.lower().strip()
        
        # Simple command interpretation (could be enhanced with NLP)
        if any(keyword in command_lower for keyword in ["search", "find", "look for"]):
            # Extract search query
            query_start = max(
                command_lower.find("for ") + 4,
                command_lower.find("search ") + 7,
                command_lower.find("find ") + 5
            )
            query = command[query_start:].strip() if query_start > 3 else command
            
            return CommandInterpretation(
                action="search",
                parameters={"query": query},
                confidence=0.8
            )
            
        elif any(keyword in command_lower for keyword in ["analyze", "scan", "test"]):
            # Extract target
            target_keywords = ["analyze", "scan", "test"]
            target_start = 0
            for keyword in target_keywords:
                if keyword in command_lower:
                    target_start = command_lower.find(keyword) + len(keyword)
                    break
                    
            target = command[target_start:].strip()
            
            return CommandInterpretation(
                action="analyze",
                parameters={"target": target},
                confidence=0.9
            )
            
        elif any(keyword in command_lower for keyword in ["learn", "study", "research"]):
            # Extract topic
            topic_start = max(
                command_lower.find("about ") + 6,
                command_lower.find("learn ") + 6,
                command_lower.find("study ") + 6
            )
            topic = command[topic_start:].strip() if topic_start > 5 else command
            
            return CommandInterpretation(
                action="learn",
                parameters={"topic": topic},
                confidence=0.7
            )
            
        elif "persona" in command_lower:
            # Extract persona name
            persona_start = command_lower.find("persona") + 7
            persona_name = command[persona_start:].strip()
            
            return CommandInterpretation(
                action="change_persona",
                parameters={"persona": persona_name},
                confidence=0.9
            )
            
        else:
            # Generic action
            return CommandInterpretation(
                action="general_query",
                parameters={"query": command},
                confidence=0.4
            )
            
    async def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning performance"""
        
        if not self.reasoning_history:
            return {"message": "No reasoning history available"}
            
        recent_decisions = self.reasoning_history[-50:]  # Last 50 decisions
        
        # Calculate confidence distribution
        confidences = [r.confidence for r in recent_decisions]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Calculate decision type distribution
        decision_types = {}
        for result in recent_decisions:
            decision_type = result.decision.get("type", "unknown")
            decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
            
        # Get decision engine statistics
        engine_stats = self.decision_engine.get_learning_statistics()
        
        return {
            "total_decisions": len(self.reasoning_history),
            "recent_decisions": len(recent_decisions),
            "average_confidence": avg_confidence,
            "decision_type_distribution": decision_types,
            "engine_statistics": engine_stats,
            "current_persona": self.persona_manager.current_persona.name if self.persona_manager.current_persona else None,
            "performance_metrics": self.performance_metrics,
            "recent_success_rate": self.recent_success_rate
        }
        
    async def adapt_reasoning(self, feedback: Dict[str, Any]):
        """Adapt reasoning patterns based on feedback"""
        
        if not self.learning_enabled:
            return
            
        success = feedback.get("success", False)
        decision_id = feedback.get("decision_id")
        
        # Update recent success rate
        self._update_success_rate(success)
        
        # Find the relevant reasoning result
        relevant_result = None
        for result in reversed(self.reasoning_history):
            if result.timestamp.isoformat() == decision_id:
                relevant_result = result
                break
                
        if not relevant_result:
            logger.warning(f"Could not find decision with ID: {decision_id}")
            return
            
        # Learn from the outcome
        decision_type_str = relevant_result.decision.get("type", "")
        try:
            decision_type = DecisionType(decision_type_str)
            
            outcome = {
                "success": success,
                "timestamp": datetime.utcnow(),
                "feedback": feedback.get("details", {})
            }
            
            self.decision_engine.learn_from_outcome(
                decision_type,
                relevant_result.reasoning_path,
                outcome
            )
            
            await self.emit_event(
                EventType.KNOWLEDGE_LEARNED,
                {"source": "reasoning_feedback", "success": success, "type": decision_type_str}
            )
            
        except ValueError:
            logger.warning(f"Unknown decision type: {decision_type_str}")
            
        # Adapt persona if needed
        await self._adapt_persona_based_on_performance()
        
    # Private methods
    
    async def _create_decision_context(self, context_data: Dict[str, Any]) -> DecisionContext:
        """Create a decision context from provided data"""
        
        persona = self.persona_manager.get_current_persona()
        persona_prefs = {}
        
        if persona:
            persona_prefs = {
                "risk_tolerance": persona.risk_tolerance,
                "stealth_priority": persona.stealth_priority,
                "creativity_level": persona.creativity_level,
                "collaboration_preference": persona.collaboration_preference,
                "preferred_tools": persona.preferred_tools,
                "avoided_tools": persona.avoided_tools
            }
            
        return DecisionContext(
            target=context_data.get("target", ""),
            current_phase=context_data.get("phase", "reconnaissance"),
            available_tools=context_data.get("available_tools", []),
            previous_results=context_data.get("previous_results", []),
            risk_constraints=context_data.get("risk_constraints", {}),
            persona_preferences=persona_prefs,
            time_constraints=context_data.get("time_constraints"),
            stealth_requirements=context_data.get("stealth_requirements", 0.5)
        )
        
    async def _calculate_persona_influence(self, decision_type: DecisionType, context: DecisionContext) -> Dict[str, Any]:
        """Calculate how the current persona influences the decision"""
        
        persona = self.persona_manager.get_current_persona()
        if not persona:
            return {"confidence_modifier": 1.0, "influences": []}
            
        influences = []
        confidence_modifier = 1.0
        
        # Risk tolerance influence
        if decision_type == DecisionType.RISK_ASSESSMENT:
            if persona.risk_tolerance > 0.7:
                influences.append("High risk tolerance - more willing to take chances")
                confidence_modifier *= 1.1
            elif persona.risk_tolerance < 0.3:
                influences.append("Low risk tolerance - emphasizing caution")
                confidence_modifier *= 0.9
                
        # Stealth priority influence
        if decision_type == DecisionType.STEALTH_LEVEL:
            if persona.stealth_priority > 0.8:
                influences.append("High stealth priority - maximizing operational security")
                confidence_modifier *= 1.2
                
        # Collaboration preference influence
        if decision_type == DecisionType.COLLABORATION_DECISION:
            if persona.collaboration_preference > 0.7:
                influences.append("High collaboration preference - seeking external advice")
                confidence_modifier *= 1.1
                
        return {
            "confidence_modifier": confidence_modifier,
            "influences": influences,
            "persona_name": persona.name
        }
        
    async def _retrieve_relevant_knowledge(self, decision_type: DecisionType, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge from the knowledge graph"""
        
        # Create search query based on decision type and context
        query_terms = [decision_type.value]
        
        if "target" in context_data:
            query_terms.append(context_data["target"])
            
        if "phase" in context_data:
            query_terms.append(context_data["phase"])
            
        query = " ".join(query_terms)
        
        try:
            knowledge_results = await self.knowledge_graph.query_knowledge(query, limit=10)
            return knowledge_results
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []
            
    def _calculate_knowledge_confidence(self, knowledge_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence modifier based on available knowledge"""
        
        if not knowledge_results:
            return 0.8  # Slight penalty for no knowledge
            
        # Calculate confidence based on knowledge quality and relevance
        total_weight = 0
        weighted_confidence = 0
        
        for knowledge in knowledge_results:
            weight = knowledge.get("relevance_score", 0.5)
            confidence = knowledge.get("confidence", 0.5)
            
            total_weight += weight
            weighted_confidence += weight * confidence
            
        if total_weight == 0:
            return 0.8
            
        base_confidence = weighted_confidence / total_weight
        
        # Bonus for having multiple relevant knowledge sources
        count_bonus = min(len(knowledge_results) * 0.05, 0.2)
        
        return min(base_confidence + count_bonus, 1.2)
        
    async def _enhance_decision_with_reasoning(self, decision: Dict[str, Any], context: DecisionContext,
                                             knowledge: List[Dict[str, Any]], persona_influence: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the basic decision with additional reasoning and context"""
        
        enhanced_decision = decision.copy()
        
        # Add reasoning explanation
        reasoning_explanation = []
        
        # Persona reasoning
        if persona_influence.get("influences"):
            reasoning_explanation.extend(persona_influence["influences"])
            
        # Knowledge reasoning
        if knowledge:
            knowledge_summary = f"Based on {len(knowledge)} relevant knowledge sources"
            reasoning_explanation.append(knowledge_summary)
            
        # Context reasoning
        if context.stealth_requirements > 0.7:
            reasoning_explanation.append("High stealth requirements detected - prioritizing OPSEC")
            
        if context.time_constraints and context.time_constraints < 60:
            reasoning_explanation.append("Time constraints detected - optimizing for speed")
            
        enhanced_decision["reasoning"] = reasoning_explanation
        enhanced_decision["context_summary"] = {
            "target": context.target,
            "phase": context.current_phase,
            "stealth_level": context.stealth_requirements,
            "persona": persona_influence.get("persona_name", "unknown")
        }
        
        # Add alternative options if confidence is low
        if decision.get("confidence", 0.5) < self.confidence_threshold:
            enhanced_decision["alternatives"] = await self._generate_alternatives(decision, context)
            
        return enhanced_decision
        
    async def _generate_alternatives(self, decision: Dict[str, Any], context: DecisionContext) -> List[Dict[str, Any]]:
        """Generate alternative options when confidence is low"""
        
        alternatives = []
        
        # Conservative alternative
        conservative_alt = {
            "type": "conservative_approach",
            "description": "More cautious approach with additional safety measures",
            "confidence": 0.7
        }
        alternatives.append(conservative_alt)
        
        # Collaborative alternative
        if self.persona_manager.current_persona and self.persona_manager.current_persona.collaboration_preference > 0.5:
            collaborative_alt = {
                "type": "seek_collaboration",
                "description": "Consult with LLM agents for additional insights",
                "confidence": 0.6
            }
            alternatives.append(collaborative_alt)
            
        # Information gathering alternative
        info_gathering_alt = {
            "type": "gather_more_information",
            "description": "Collect additional information before proceeding",
            "confidence": 0.8
        }
        alternatives.append(info_gathering_alt)
        
        return alternatives
        
    def _initialize_performance_metrics(self):
        """Initialize performance tracking metrics"""
        
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "confidence_accuracy": 0.0,
            "average_decision_time": 0.0,
            "persona_effectiveness": {},
            "decision_type_performance": {}
        }
        
    def _update_success_rate(self, success: bool):
        """Update the recent success rate with exponential moving average"""
        
        # Exponential moving average with alpha = 0.1
        self.recent_success_rate = 0.9 * self.recent_success_rate + 0.1 * (1.0 if success else 0.0)
        
    async def _adapt_persona_based_on_performance(self):
        """Adapt persona parameters based on recent performance"""
        
        if not self.persona_manager.current_persona:
            return
            
        # If success rate is consistently low, consider adapting
        if self.recent_success_rate < 0.3 and len(self.reasoning_history) > 20:
            
            # Suggest persona adaptation
            await self.emit_event(
                EventType.REASONING_START,
                {
                    "action": "persona_adaptation_suggested",
                    "current_success_rate": self.recent_success_rate,
                    "recommendation": "Consider switching to a more conservative persona"
                }
            )
            
    def _prune_reasoning_history(self, max_history: int = 1000):
        """Prune reasoning history to prevent memory growth"""
        
        if len(self.reasoning_history) > max_history:
            self.reasoning_history = self.reasoning_history[-max_history//2:]
            
    # Event handlers
    
    async def _on_task_complete(self, event):
        """Handle task completion events for learning"""
        
        if self.learning_enabled:
            # This could be enhanced to correlate with recent decisions
            self._update_success_rate(True)
            
    async def _on_task_failed(self, event):
        """Handle task failure events for learning"""
        
        if self.learning_enabled:
            self._update_success_rate(False)
            
    async def _on_knowledge_learned(self, event):
        """Handle knowledge learning events"""
        
        # Could adapt reasoning based on new knowledge
        pass
        
    async def shutdown(self):
        """Shutdown the reasoning engine"""
        
        self.unsubscribe_all()
        logger.info("Neuroplastic reasoner shutdown complete")