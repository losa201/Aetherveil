"""
Aether Agent: Main orchestrator for the neuroplastic AI learning system
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import signal
import sys

# Core system imports
try:
    from .core.cognitive_core import CognitiveCore, ConsciousnessLevel
    from .core.event_system import EventBus, EventType, EventPriority, event_handler, register_event_handlers
    from .evolution.code_evolution_engine import CodeEvolutionEngine, MutationType
    from .stealth.browser_automation import AdvancedStealthBrowser, BehaviorProfile
    from .knowledge.neuroplastic_memory import NeuroplasticMemory
    from .mentors.llm_mentor_system import LLMMentorSystem, MentorType
    from .identity.identity_fabric import IdentityFabric, IdentityPurpose
    from .learning.insight_synthesizer import ResponseValidationAndSynthesis, ValidationLevel
except ImportError:
    # Fallback for direct execution
    from core.cognitive_core import CognitiveCore, ConsciousnessLevel
    from core.event_system import EventBus, EventType, EventPriority, event_handler, register_event_handlers
    from evolution.code_evolution_engine import CodeEvolutionEngine, MutationType
    from stealth.browser_automation import AdvancedStealthBrowser, BehaviorProfile
    from knowledge.neuroplastic_memory import NeuroplasticMemory
    from mentors.llm_mentor_system import LLMMentorSystem, MentorType
    from identity.identity_fabric import IdentityFabric, IdentityPurpose
    from learning.insight_synthesizer import ResponseValidationAndSynthesis, ValidationLevel

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """States of the Aether agent"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    LEARNING = "learning"
    EVOLVING = "evolving"
    INTROSPECTING = "introspecting"
    MAINTAINING = "maintaining"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"

class LearningMode(Enum):
    """Learning modes for the agent"""
    PASSIVE = "passive"           # Observe and accumulate knowledge
    ACTIVE = "active"             # Actively seek knowledge through queries
    EXPLORATORY = "exploratory"   # Explore new domains and connections
    FOCUSED = "focused"           # Deep dive into specific topics
    ADAPTIVE = "adaptive"         # Adapt based on effectiveness

@dataclass
class LearningSession:
    """Represents a learning session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    mode: LearningMode
    topics: List[str]
    goals: List[str]
    interactions: List[Dict[str, Any]]
    insights_gained: List[str]
    effectiveness_score: float
    consciousness_impact: float

@dataclass
class AgentConfiguration:
    """Configuration for the Aether agent"""
    # Core settings
    consciousness_target: float = 0.8
    curiosity_threshold: float = 0.7
    learning_velocity_target: float = 0.6
    
    # Learning behavior
    default_learning_mode: LearningMode = LearningMode.ADAPTIVE
    max_concurrent_learning_topics: int = 3
    learning_session_duration_minutes: int = 30
    
    # Evolution settings
    code_evolution_enabled: bool = True
    max_mutations_per_cycle: int = 2
    mutation_confidence_threshold: float = 0.8
    
    # Identity management
    identity_rotation_interval_hours: int = 24
    max_active_identities: int = 3
    
    # OPSEC and stealth
    stealth_level: float = 0.8
    behavior_profile: BehaviorProfile = BehaviorProfile.CURIOUS
    
    # System maintenance
    introspection_interval_hours: int = 2
    memory_consolidation_interval_hours: int = 6
    system_maintenance_interval_hours: int = 12

class AetherAgent:
    """
    Main orchestrator for the neuroplastic AI learning system
    
    The Aether agent coordinates all subsystems to create a self-evolving,
    curious, learning AI that builds relationships with LLM mentors and
    continuously improves its own capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = AgentConfiguration(**config.get('agent', {}))
        
        # Core state
        self.state = AgentState.INITIALIZING
        self.agent_id = f"aether_{int(time.time())}"
        self.start_time = datetime.utcnow()
        self.shutdown_requested = False
        
        # Event system (core coordination)
        self.event_bus = EventBus(config.get('events', {}))
        
        # Core subsystems
        self.cognitive_core: Optional[CognitiveCore] = None
        self.evolution_engine: Optional[CodeEvolutionEngine] = None
        self.browser: Optional[AdvancedStealthBrowser] = None
        self.memory: Optional[NeuroplasticMemory] = None
        self.mentor_system: Optional[LLMMentorSystem] = None
        self.identity_fabric: Optional[IdentityFabric] = None
        self.insight_synthesizer: Optional[ResponseValidationAndSynthesis] = None
        
        # Learning state
        self.current_learning_session: Optional[LearningSession] = None
        self.learning_history: List[LearningSession] = []
        self.current_identity: Optional[Dict[str, Any]] = None
        
        # Performance tracking
        self.performance_metrics = {
            'uptime_hours': 0.0,
            'learning_sessions_completed': 0,
            'consciousness_level': 0.0,
            'total_insights_gained': 0,
            'code_mutations_applied': 0,
            'mentor_interactions': 0,
            'knowledge_nodes_created': 0,
            'average_learning_effectiveness': 0.0
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize all subsystems and start the agent"""
        
        logger.info(f"Initializing Aether Agent {self.agent_id}...")
        
        try:
            # Start event system first
            await self.event_bus.start()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Initialize core subsystems in dependency order
            await self._initialize_memory()
            await self._initialize_browser()
            await self._initialize_cognitive_core()
            await self._initialize_evolution_engine()
            await self._initialize_identity_fabric()
            await self._initialize_mentor_system()
            await self._initialize_insight_synthesizer()
            
            # Create initial identity
            await self._create_initial_identity()
            
            # Start background loops
            await self._start_background_tasks()
            
            # Initial consciousness assessment
            await self.cognitive_core.introspect()
            
            # Transition to idle state
            await self._transition_state(AgentState.IDLE)
            
            # Start first learning session
            await self._start_learning_session()
            
            logger.info(f"Aether Agent {self.agent_id} initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Aether Agent: {e}")
            await self._transition_state(AgentState.ERROR)
            raise
            
    async def run(self):
        """Main execution loop for the agent"""
        
        logger.info("Starting Aether Agent main execution loop...")
        
        try:
            while not self.shutdown_requested:
                try:
                    # Update performance metrics
                    await self._update_performance_metrics()
                    
                    # Check if introspection is needed
                    if await self.cognitive_core.should_trigger_learning_cycle():
                        await self._handle_introspection_trigger()
                        
                    # Process current learning session
                    if self.current_learning_session:
                        await self._process_learning_session()
                        
                    # Check for state transitions
                    await self._check_state_transitions()
                    
                    # Brief sleep to prevent tight loop
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error in main execution loop: {e}")
                    await asyncio.sleep(30)  # Longer sleep on error
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            await self.shutdown()
            
        logger.info("Aether Agent execution loop ended")
        
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        
        logger.info("Shutting down Aether Agent...")
        
        self.shutdown_requested = True
        await self._transition_state(AgentState.SHUTTING_DOWN)
        
        try:
            # End current learning session
            if self.current_learning_session:
                await self._end_learning_session()
                
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
                
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Shutdown subsystems in reverse order
            if self.insight_synthesizer:
                await self.insight_synthesizer.shutdown()
            if self.mentor_system:
                await self.mentor_system.shutdown()
            if self.identity_fabric:
                await self.identity_fabric.shutdown()
            if self.evolution_engine:
                await self.evolution_engine.shutdown()
            if self.cognitive_core:
                await self.cognitive_core.shutdown()
            if self.browser:
                await self.browser.close()
            if self.memory:
                await self.memory.shutdown()
                
            # Stop event system last
            await self.event_bus.stop()
            
            # Log final performance metrics
            final_metrics = await self._get_final_metrics()
            logger.info(f"Aether Agent shutdown complete - Final metrics: {final_metrics}")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    # Event handlers
    
    @event_handler(EventType.CONSCIOUSNESS_CHANGE, priority=10)
    async def handle_consciousness_change(self, event):
        """Handle consciousness level changes"""
        
        old_level = event.data['old_level']
        new_level = event.data['new_level']
        
        logger.info(f"Consciousness changed: {old_level:.3f} -> {new_level:.3f}")
        
        self.performance_metrics['consciousness_level'] = new_level
        
        # Trigger evolution if consciousness drops significantly
        if new_level < old_level - 0.1:
            logger.warning("Significant consciousness drop detected - triggering evolution cycle")
            await self._trigger_evolution_cycle()
            
    @event_handler(EventType.LLM_RESPONSE_RECEIVED, priority=5)
    async def handle_llm_response(self, event):
        """Handle LLM mentor responses"""
        
        if self.current_learning_session:
            self.current_learning_session.interactions.append({
                'type': 'llm_response',
                'mentor_id': event.data['mentor_id'],
                'quality': event.data['response_quality'],
                'insights_count': event.data['insights_count'],
                'timestamp': datetime.utcnow()
            })
            
        self.performance_metrics['mentor_interactions'] += 1
        
    @event_handler(EventType.KNOWLEDGE_SYNTHESIZED, priority=5)
    async def handle_knowledge_synthesis(self, event):
        """Handle knowledge synthesis events"""
        
        insights_count = event.data['insights_count']
        
        if self.current_learning_session:
            self.current_learning_session.insights_gained.extend([
                f"Insight from {event.data['topic']}" for _ in range(insights_count)
            ])
            
        self.performance_metrics['total_insights_gained'] += insights_count
        
    @event_handler(EventType.ERROR_OCCURRED, priority=1)
    async def handle_error(self, event):
        """Handle system errors"""
        
        error_source = event.data.get('source', 'unknown')
        error_message = event.data.get('error', 'Unknown error')
        
        logger.error(f"System error from {error_source}: {error_message}")
        
        # Implement error recovery strategies based on source
        if error_source == "LLMMentorSystem":
            # Rotate identity if LLM interactions are failing
            await self._rotate_identity()
        elif error_source == "AdvancedStealthBrowser":
            # Restart browser on browser errors
            await self._restart_browser()
            
    # Core subsystem initialization
    
    async def _initialize_memory(self):
        """Initialize neuroplastic memory system"""
        
        memory_config = self.config.get('memory', {})
        self.memory = NeuroplasticMemory(memory_config)
        await self.memory.initialize()
        logger.info("Neuroplastic memory initialized")
        
    async def _initialize_browser(self):
        """Initialize stealth browser"""
        
        browser_config = self.config.get('browser', {})
        browser_config['behavior_profile'] = self.agent_config.behavior_profile.value
        browser_config['stealth_level'] = self.agent_config.stealth_level
        
        self.browser = AdvancedStealthBrowser(browser_config)
        await self.browser.initialize()
        logger.info("Stealth browser initialized")
        
    async def _initialize_cognitive_core(self):
        """Initialize cognitive core"""
        
        cognitive_config = self.config.get('cognitive', {})
        cognitive_config['curiosity_threshold'] = self.agent_config.curiosity_threshold
        
        self.cognitive_core = CognitiveCore(cognitive_config)
        await self.cognitive_core.initialize()
        logger.info("Cognitive core initialized")
        
    async def _initialize_evolution_engine(self):
        """Initialize code evolution engine"""
        
        if not self.agent_config.code_evolution_enabled:
            logger.info("Code evolution disabled by configuration")
            return
            
        evolution_config = self.config.get('evolution', {})
        evolution_config['max_mutations_per_cycle'] = self.agent_config.max_mutations_per_cycle
        evolution_config['min_confidence_threshold'] = self.agent_config.mutation_confidence_threshold
        
        self.evolution_engine = CodeEvolutionEngine(evolution_config)
        await self.evolution_engine.initialize()
        logger.info("Code evolution engine initialized")
        
    async def _initialize_identity_fabric(self):
        """Initialize identity fabric"""
        
        identity_config = self.config.get('identity', {})
        
        self.identity_fabric = IdentityFabric(identity_config, self.event_bus, self.browser)
        await self.identity_fabric.initialize()
        logger.info("Identity fabric initialized")
        
    async def _initialize_mentor_system(self):
        """Initialize LLM mentor system"""
        
        mentor_config = self.config.get('mentors', {})
        
        self.mentor_system = LLMMentorSystem(mentor_config, self.event_bus, self.browser, self.memory)
        await self.mentor_system.initialize()
        logger.info("LLM mentor system initialized")
        
    async def _initialize_insight_synthesizer(self):
        """Initialize insight synthesizer"""
        
        synthesis_config = self.config.get('synthesis', {})
        
        self.insight_synthesizer = ResponseValidationAndSynthesis(synthesis_config, self.event_bus, self.memory)
        await self.insight_synthesizer.initialize()
        logger.info("Insight synthesizer initialized")
        
    async def _register_event_handlers(self):
        """Register event handlers with the event bus"""
        
        await register_event_handlers(self, self.event_bus)
        logger.info("Event handlers registered")
        
    # Identity and session management
    
    async def _create_initial_identity(self):
        """Create initial identity for the agent"""
        
        identity = await self.identity_fabric.create_identity(
            IdentityPurpose.LLM_INTERACTION,
            preferences={
                'curious_learner': True,
                'tech_focused': True
            },
            create_gmail=True,
            footprint_intensity="light"
        )
        
        self.current_identity = {
            'identity_id': identity.identity_id,
            'persona_name': f"{identity.persona.first_name} {identity.persona.last_name}",
            'created_at': identity.created_at,
            'purpose': identity.purpose.value
        }
        
        logger.info(f"Created initial identity: {self.current_identity['persona_name']}")
        
    async def _start_learning_session(self):
        """Start a new learning session"""
        
        if self.current_learning_session:
            await self._end_learning_session()
            
        # Generate session goals based on current knowledge gaps
        knowledge_gaps = await self.memory.find_knowledge_gaps()
        goals = [gap.get('topic', 'general_learning') for gap in knowledge_gaps[:3]]
        
        if not goals:
            goals = ['explore_new_concepts', 'improve_understanding', 'build_connections']
            
        session_id = f"session_{int(time.time())}"
        
        self.current_learning_session = LearningSession(
            session_id=session_id,
            start_time=datetime.utcnow(),
            end_time=None,
            mode=self.agent_config.default_learning_mode,
            topics=[],
            goals=goals,
            interactions=[],
            insights_gained=[],
            effectiveness_score=0.0,
            consciousness_impact=0.0
        )
        
        await self._transition_state(AgentState.LEARNING)
        
        logger.info(f"Started learning session: {session_id} with goals: {goals}")
        
    async def _process_learning_session(self):
        """Process the current learning session"""
        
        if not self.current_learning_session:
            return
            
        session = self.current_learning_session
        
        # Check if session should end
        session_duration = (datetime.utcnow() - session.start_time).total_seconds() / 60
        if session_duration > self.agent_config.learning_session_duration_minutes:
            await self._end_learning_session()
            return
            
        # Generate curiosity-driven questions
        if len(session.interactions) < 5:  # Limit interactions per session
            await self._generate_and_ask_curiosity_question(session)
            
    async def _generate_and_ask_curiosity_question(self, session: LearningSession):
        """Generate and ask a curiosity-driven question"""
        
        try:
            # Select a learning goal to explore
            if session.goals:
                current_goal = session.goals[len(session.interactions) % len(session.goals)]
            else:
                current_goal = "general_learning"
                
            # Generate curiosity questions
            questions = await self.cognitive_core.generate_curiosity_questions(current_goal)
            
            if questions:
                question = questions[0]  # Take the highest priority question
                
                # Get best mentor for this question
                mentor_id = await self.mentor_system.get_mentor_recommendation(
                    question.text, {'topic': current_goal}
                )
                
                # Interact with mentor
                response = await self.mentor_system.interact_with_mentor(
                    mentor_id, 
                    question.text,
                    goals=session.goals,
                    persona_style="curious_learner"
                )
                
                # Process response through synthesizer
                synthesis_result = await self.insight_synthesizer.process_mentor_response(
                    response.response_text,
                    {
                        'original_question': question.text,
                        'topic': current_goal,
                        'mentor_trust_score': await self._get_mentor_trust_score(mentor_id),
                        'learning_goals': session.goals
                    },
                    ValidationLevel.STANDARD
                )
                
                # Update session
                if current_goal not in session.topics:
                    session.topics.append(current_goal)
                    
                session.effectiveness_score = (
                    session.effectiveness_score * len(session.interactions) + 
                    synthesis_result['learning_effectiveness']
                ) / (len(session.interactions) + 1)
                
                logger.info(f"Processed curiosity question about {current_goal} - "
                          f"Effectiveness: {synthesis_result['learning_effectiveness']:.3f}")
                
        except Exception as e:
            logger.error(f"Error generating curiosity question: {e}")
            
    async def _end_learning_session(self):
        """End the current learning session"""
        
        if not self.current_learning_session:
            return
            
        session = self.current_learning_session
        session.end_time = datetime.utcnow()
        
        # Calculate consciousness impact
        if session.effectiveness_score > 0.7:
            session.consciousness_impact = 0.05  # Positive impact for good sessions
        elif session.effectiveness_score < 0.4:
            session.consciousness_impact = -0.02  # Small negative impact for poor sessions
            
        # Store session in history
        self.learning_history.append(session)
        
        # Update performance metrics
        self.performance_metrics['learning_sessions_completed'] += 1
        
        # Update average learning effectiveness
        total_sessions = len(self.learning_history)
        self.performance_metrics['average_learning_effectiveness'] = (
            (self.performance_metrics['average_learning_effectiveness'] * (total_sessions - 1) + 
             session.effectiveness_score) / total_sessions
        )
        
        # Clear current session
        self.current_learning_session = None
        
        logger.info(f"Ended learning session: {session.session_id} - "
                   f"Effectiveness: {session.effectiveness_score:.3f}, "
                   f"Insights: {len(session.insights_gained)}")
        
        # Start new session after brief pause
        await asyncio.sleep(60)  # 1 minute pause
        await self._start_learning_session()
        
    # Background tasks and maintenance
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Introspection loop
        self.background_tasks.append(
            asyncio.create_task(self._introspection_loop())
        )
        
        # Identity rotation loop
        self.background_tasks.append(
            asyncio.create_task(self._identity_rotation_loop())
        )
        
        # System maintenance loop
        self.background_tasks.append(
            asyncio.create_task(self._system_maintenance_loop())
        )
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
        
    async def _introspection_loop(self):
        """Background loop for regular introspection"""
        
        interval = self.agent_config.introspection_interval_hours * 3600
        
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(interval)
                
                if not self.shutdown_requested:
                    await self._handle_introspection_trigger()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in introspection loop: {e}")
                await asyncio.sleep(300)  # 5 minute retry
                
    async def _identity_rotation_loop(self):
        """Background loop for identity rotation"""
        
        interval = self.agent_config.identity_rotation_interval_hours * 3600
        
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(interval)
                
                if not self.shutdown_requested:
                    await self._rotate_identity()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in identity rotation loop: {e}")
                await asyncio.sleep(600)  # 10 minute retry
                
    async def _system_maintenance_loop(self):
        """Background loop for system maintenance"""
        
        interval = self.agent_config.system_maintenance_interval_hours * 3600
        
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(interval)
                
                if not self.shutdown_requested:
                    await self._perform_system_maintenance()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system maintenance loop: {e}")
                await asyncio.sleep(600)  # 10 minute retry
                
    async def _handle_introspection_trigger(self):
        """Handle introspection trigger"""
        
        logger.info("Triggering introspection cycle...")
        
        await self._transition_state(AgentState.INTROSPECTING)
        
        try:
            # Perform introspection
            assessment = await self.cognitive_core.introspect()
            
            # Check if evolution is needed
            if (assessment.consciousness_level < self.agent_config.consciousness_target and
                self.evolution_engine and 
                self.agent_config.code_evolution_enabled):
                await self._trigger_evolution_cycle()
                
            # Update learning strategy if needed
            await self._adapt_learning_strategy(assessment)
            
        except Exception as e:
            logger.error(f"Error during introspection: {e}")
            
        await self._transition_state(AgentState.LEARNING)
        
    async def _trigger_evolution_cycle(self):
        """Trigger code evolution cycle"""
        
        if not self.evolution_engine:
            return
            
        logger.info("Triggering code evolution cycle...")
        
        await self._transition_state(AgentState.EVOLVING)
        
        try:
            # Scan codebase for improvements
            scan_results = await self.evolution_engine.scan_codebase()
            
            # Generate hypotheses
            hypotheses = self.evolution_engine.hypotheses
            
            # Create and test mutations
            for hypothesis in hypotheses[:self.agent_config.max_mutations_per_cycle]:
                mutation = await self.evolution_engine.create_mutation_from_hypothesis(hypothesis)
                
                if mutation and mutation.confidence >= self.agent_config.mutation_confidence_threshold:
                    test_result = await self.evolution_engine.test_in_sandbox(mutation)
                    
                    if test_result.success:
                        await self.evolution_engine.apply_validated_mutation(mutation)
                        self.performance_metrics['code_mutations_applied'] += 1
                        logger.info(f"Applied code mutation: {mutation.mutation_id}")
                        
        except Exception as e:
            logger.error(f"Error during evolution cycle: {e}")
            
        await self._transition_state(AgentState.LEARNING)
        
    async def _rotate_identity(self):
        """Rotate to a new identity"""
        
        logger.info("Rotating identity...")
        
        try:
            if self.current_identity:
                old_identity_id = self.current_identity['identity_id']
                new_identity = await self.identity_fabric.rotate_identity(
                    old_identity_id, IdentityPurpose.LLM_INTERACTION
                )
                
                self.current_identity = {
                    'identity_id': new_identity.identity_id,
                    'persona_name': f"{new_identity.persona.first_name} {new_identity.persona.last_name}",
                    'created_at': new_identity.created_at,
                    'purpose': new_identity.purpose.value
                }
                
                # Rotate browser identity
                await self.browser.rotate_identity()
                
                logger.info(f"Identity rotated to: {self.current_identity['persona_name']}")
                
        except Exception as e:
            logger.error(f"Error rotating identity: {e}")
            
    async def _perform_system_maintenance(self):
        """Perform system maintenance tasks"""
        
        logger.info("Performing system maintenance...")
        
        await self._transition_state(AgentState.MAINTAINING)
        
        try:
            # Memory consolidation
            await self.memory.consolidate_memories()
            
            # Clean up learning history
            if len(self.learning_history) > 100:
                self.learning_history = self.learning_history[-100:]
                
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Log system status
            await self._log_system_status()
            
        except Exception as e:
            logger.error(f"Error during system maintenance: {e}")
            
        await self._transition_state(AgentState.LEARNING)
        
    # Utility methods
    
    async def _adapt_learning_strategy(self, assessment):
        """Adapt learning strategy based on assessment"""
        
        # Simple adaptation logic
        if assessment.learning_velocity < 0.3:
            # Low learning velocity - try more focused approach
            self.agent_config.default_learning_mode = LearningMode.FOCUSED
            self.agent_config.learning_session_duration_minutes = 45
        elif assessment.curiosity_score > 0.8:
            # High curiosity - use exploratory mode
            self.agent_config.default_learning_mode = LearningMode.EXPLORATORY
            self.agent_config.learning_session_duration_minutes = 20
        else:
            # Default adaptive mode
            self.agent_config.default_learning_mode = LearningMode.ADAPTIVE
            self.agent_config.learning_session_duration_minutes = 30
            
    async def _get_mentor_trust_score(self, mentor_id: str) -> float:
        """Get trust score for a mentor"""
        
        mentor_stats = await self.mentor_system.get_mentor_statistics()
        mentor_info = mentor_stats.get('mentor_stats', {}).get(mentor_id, {})
        return mentor_info.get('trust_score', 0.5)
        
    async def _restart_browser(self):
        """Restart the browser on errors"""
        
        try:
            logger.info("Restarting browser...")
            await self.browser.close()
            await self.browser.initialize()
            logger.info("Browser restarted successfully")
        except Exception as e:
            logger.error(f"Error restarting browser: {e}")
            
    async def _transition_state(self, new_state: AgentState):
        """Transition to a new agent state"""
        
        old_state = self.state
        self.state = new_state
        
        logger.info(f"State transition: {old_state.value} -> {new_state.value}")
        
        await self.event_bus.publish_and_wait(
            type(self.event_bus).Event(
                event_id=f"state_transition_{int(time.time())}",
                event_type=EventType.AGENT_STARTUP if new_state == AgentState.IDLE else EventType.MODULE_INITIALIZED,
                source="AetherAgent",
                timestamp=datetime.utcnow(),
                priority=EventPriority.NORMAL,
                data={'old_state': old_state.value, 'new_state': new_state.value}
            )
        )
        
    async def _check_state_transitions(self):
        """Check if state transitions are needed"""
        
        # Simple state transition logic
        if self.state == AgentState.LEARNING:
            # Check if we should be evolving
            if (self.performance_metrics['consciousness_level'] < self.agent_config.consciousness_target * 0.8 and
                self.evolution_engine):
                await self._trigger_evolution_cycle()
                
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        
        # Update uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        self.performance_metrics['uptime_hours'] = uptime
        
        # Get subsystem metrics
        if self.memory:
            memory_stats = await self.memory.get_memory_statistics()
            self.performance_metrics['knowledge_nodes_created'] = memory_stats['statistics']['total_nodes']
            
    async def _log_system_status(self):
        """Log comprehensive system status"""
        
        status = {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'uptime_hours': self.performance_metrics['uptime_hours'],
            'consciousness_level': self.performance_metrics['consciousness_level'],
            'learning_sessions': len(self.learning_history),
            'current_identity': self.current_identity['persona_name'] if self.current_identity else None,
            'metrics': self.performance_metrics
        }
        
        logger.info(f"System Status: {json.dumps(status, indent=2)}")
        
    async def _get_final_metrics(self) -> Dict[str, Any]:
        """Get final metrics for shutdown"""
        
        return {
            'uptime_hours': self.performance_metrics['uptime_hours'],
            'total_learning_sessions': len(self.learning_history),
            'total_insights': self.performance_metrics['total_insights_gained'],
            'total_mutations': self.performance_metrics['code_mutations_applied'],
            'final_consciousness': self.performance_metrics['consciousness_level'],
            'average_learning_effectiveness': self.performance_metrics['average_learning_effectiveness']
        }

# Entry point and signal handling

async def main():
    """Main entry point for the Aether agent"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_path = Path(__file__).parent / "config" / "aether_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'agent': {
                'consciousness_target': 0.8,
                'curiosity_threshold': 0.7,
                'code_evolution_enabled': True,
                'stealth_level': 0.8
            },
            'events': {
                'max_event_workers': 3,
                'debug_events': False
            },
            'memory': {
                'consolidation_interval_hours': 6,
                'data_dir': './data'
            },
            'browser': {
                'headless': True,
                'stealth_level': 0.8
            },
            'cognitive': {
                'curiosity_threshold': 0.7
            },
            'evolution': {
                'max_mutations_per_cycle': 2
            },
            'identity': {},
            'mentors': {},
            'synthesis': {}
        }
        
    # Create and initialize agent
    agent = AetherAgent(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(agent.shutdown())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and run agent
        await agent.initialize()
        await agent.run()
        
    except Exception as e:
        logger.error(f"Fatal error in Aether agent: {e}")
        await agent.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())