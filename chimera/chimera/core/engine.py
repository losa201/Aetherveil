"""
Core orchestration engine for Chimera
Coordinates all modules and manages the overall system lifecycle
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .events import EventSystem, EventType, ChimeraEvent, EventEmitter
from ..utils.config import ConfigManager
from ..reasoner.reasoner import NeuroplasticReasoner
from ..memory.knowledge_graph import KnowledgeGraph
from ..web.searcher import StealthWebSearcher
from ..llm.collaborator import LLMCollaborator
from ..planner.planner import TacticalPlanner
from ..executor.executor import TaskExecutor
from ..validator.validator import ModuleValidator
from ..reporter.reporter import ReportGenerator
from ..analytics.performance_monitor import PerformanceMonitor
from ..orchestrator.campaign_orchestrator import CampaignOrchestrator, CampaignPriority
from ..parallel.parallel_worker import ParallelSpecializationWorker

logger = logging.getLogger(__name__)

class CampaignResult:
    """Result of a red-team campaign"""
    
    def __init__(self, campaign_id: str, success: bool, report_path: Optional[str] = None):
        self.campaign_id = campaign_id
        self.success = success
        self.report_path = report_path
        self.timestamp = datetime.utcnow()
        self.findings_count = 0
        self.severity_breakdown = {}

class ChimeraEngine(EventEmitter):
    """
    Main orchestration engine for Chimera
    
    Manages module lifecycle, coordinates between components,
    and provides the main interface for external interactions
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.event_system = EventSystem(
            max_queue_size=config_manager.get("core.event_queue_size", 1000)
        )
        
        # Initialize as event emitter
        super().__init__(self.event_system, "ChimeraEngine")
        
        # Core modules
        self.reasoner: Optional[NeuroplasticReasoner] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.web_searcher: Optional[StealthWebSearcher] = None
        self.llm_collaborator: Optional[LLMCollaborator] = None
        self.planner: Optional[TacticalPlanner] = None
        self.executor: Optional[TaskExecutor] = None
        self.validator: Optional[ModuleValidator] = None
        self.reporter: Optional[ReportGenerator] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.campaign_orchestrator: Optional[CampaignOrchestrator] = None
        
        # State management
        self.initialized = False
        self.current_persona = None
        self.active_campaigns = {}
        self.startup_time = None
        
        # Task management
        self.active_tasks = set()
        self.max_concurrent_tasks = config_manager.get("core.max_concurrent_tasks", 4)
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
    async def initialize(self):
        """Initialize all Chimera modules"""
        if self.initialized:
            return
            
        try:
            await self.emit_event(EventType.SYSTEM_STARTUP, {"message": "Initializing Chimera"})
            
            # Start event system
            await self.event_system.start()
            
            # Initialize knowledge graph first (other modules depend on it)
            self.knowledge_graph = KnowledgeGraph(
                self.config,
                self.event_system
            )
            await self.knowledge_graph.initialize()
            
            # Initialize reasoner with knowledge graph
            self.reasoner = NeuroplasticReasoner(
                self.config,
                self.event_system,
                self.knowledge_graph
            )
            await self.reasoner.initialize()
            
            # Initialize web searcher with stealth capabilities
            self.web_searcher = StealthWebSearcher(
                self.config,
                self.event_system
            )
            await self.web_searcher.initialize()
            
            # Initialize LLM collaborator
            self.llm_collaborator = LLMCollaborator(
                self.config,
                self.event_system
            )
            await self.llm_collaborator.initialize()
            
            # Initialize planner
            self.planner = TacticalPlanner(
                self.config,
                self.event_system,
                self.knowledge_graph
            )
            await self.planner.initialize()
            
            # Initialize executor with OPSEC
            self.executor = TaskExecutor(
                self.config,
                self.event_system,
                self.knowledge_graph
            )
            await self.executor.initialize()
            
            # Initialize validator
            self.validator = ModuleValidator(
                self.config,
                self.event_system
            )
            await self.validator.initialize()
            
            # Initialize reporter
            self.reporter = ReportGenerator(
                self.config,
                self.event_system
            )
            await self.reporter.initialize()
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(
                self.config,
                self.event_system
            )
            await self.performance_monitor.initialize()
            
            # Initialize campaign orchestrator
            self.campaign_orchestrator = CampaignOrchestrator(
                self.config,
                self.event_system
            )
            await self.campaign_orchestrator.initialize()
            
            # Set default persona
            default_persona = self.config.get("persona.default", "balanced")
            await self.set_persona(default_persona)
            
            # Setup event subscriptions
            await self._setup_event_subscriptions()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.initialized = True
            self.startup_time = datetime.utcnow()
            
            await self.emit_event(
                EventType.SYSTEM_STARTUP, 
                {"message": "Chimera initialized successfully", "modules": 8}
            )
            
            logger.info("Chimera engine initialized successfully")
            
        except Exception as e:
            await self.emit_event(
                EventType.SYSTEM_ERROR, 
                {"error": str(e), "phase": "initialization"},
                severity="ERROR"
            )
            logger.error(f"Failed to initialize Chimera: {e}")
            raise
            
    async def set_persona(self, persona_name: str):
        """Set the active persona for decision making"""
        if not self.reasoner:
            raise RuntimeError("Reasoner not initialized")
            
        success = await self.reasoner.load_persona(persona_name)
        if success:
            self.current_persona = persona_name
            await self.emit_event(
                EventType.REASONING_START,
                {"persona": persona_name, "action": "persona_changed"}
            )
            logger.info(f"Persona changed to: {persona_name}")
        else:
            logger.error(f"Failed to load persona: {persona_name}")
            
    async def run_campaign(self, target: str, scope_file: str) -> Optional[CampaignResult]:
        """Run a comprehensive red-team campaign"""
        campaign_id = str(uuid.uuid4())
        campaign_start_time = datetime.utcnow()
        
        try:
            await self.emit_event(
                EventType.TASK_START,
                {"campaign_id": campaign_id, "target": target, "scope_file": scope_file}
            )
            
            # Record campaign start metrics
            if self.performance_monitor:
                await self.performance_monitor.record_metric(
                    "campaign_start", 1.0, "count",
                    context={"target": target, "campaign_id": campaign_id},
                    category="campaign", source_component="engine"
                )
            
            # Load and validate scope
            scope_data = await self._load_scope(scope_file)
            if not scope_data:
                return None
                
            # Create campaign context
            campaign_context = {
                "id": campaign_id,
                "target": target,
                "scope": scope_data,
                "persona": self.current_persona,
                "start_time": campaign_start_time
            }
            
            self.active_campaigns[campaign_id] = campaign_context
            
            # Phase 1: Intelligence gathering and reconnaissance
            phase_start = datetime.utcnow()
            recon_data = await self._run_reconnaissance_phase(campaign_context)
            if self.performance_monitor:
                phase_duration = (datetime.utcnow() - phase_start).total_seconds()
                await self.performance_monitor.record_metric(
                    "reconnaissance_phase_time", phase_duration, "seconds",
                    context={"campaign_id": campaign_id}, category="campaign", 
                    source_component="reconnaissance"
                )
            
            # Phase 2: Vulnerability assessment and planning
            phase_start = datetime.utcnow()
            vulnerabilities = await self._run_assessment_phase(campaign_context, recon_data)
            if self.performance_monitor:
                phase_duration = (datetime.utcnow() - phase_start).total_seconds()
                await self.performance_monitor.record_metric(
                    "assessment_phase_time", phase_duration, "seconds",
                    context={"campaign_id": campaign_id}, category="campaign", 
                    source_component="assessment"
                )
            
            # Phase 3: Exploitation planning
            phase_start = datetime.utcnow()
            exploitation_plan = await self._run_planning_phase(campaign_context, vulnerabilities)
            if self.performance_monitor:
                phase_duration = (datetime.utcnow() - phase_start).total_seconds()
                await self.performance_monitor.record_metric(
                    "planning_phase_time", phase_duration, "seconds",
                    context={"campaign_id": campaign_id}, category="campaign", 
                    source_component="planning"
                )
            
            # Phase 4: Controlled exploitation
            phase_start = datetime.utcnow()
            findings = await self._run_exploitation_phase(campaign_context, exploitation_plan)
            if self.performance_monitor:
                phase_duration = (datetime.utcnow() - phase_start).total_seconds()
                await self.performance_monitor.record_metric(
                    "exploitation_phase_time", phase_duration, "seconds",
                    context={"campaign_id": campaign_id}, category="campaign", 
                    source_component="exploitation"
                )
            
            # Phase 5: Report generation
            phase_start = datetime.utcnow()
            report_path = await self._generate_campaign_report(campaign_context, findings)
            if self.performance_monitor:
                phase_duration = (datetime.utcnow() - phase_start).total_seconds()
                await self.performance_monitor.record_metric(
                    "reporting_phase_time", phase_duration, "seconds",
                    context={"campaign_id": campaign_id}, category="campaign", 
                    source_component="reporting"
                )
            
            # Record overall campaign metrics
            campaign_duration = (datetime.utcnow() - campaign_start_time).total_seconds()
            if self.performance_monitor:
                await self.performance_monitor.record_metric(
                    "campaign_total_time", campaign_duration, "seconds",
                    context={"campaign_id": campaign_id, "findings_count": len(findings)},
                    category="campaign", source_component="engine"
                )
                await self.performance_monitor.record_metric(
                    "campaign_success_rate", 1.0, "ratio",
                    context={"campaign_id": campaign_id}, category="campaign", 
                    source_component="engine"
                )
                await self.performance_monitor.record_metric(
                    "findings_per_campaign", len(findings), "count",
                    context={"campaign_id": campaign_id}, category="campaign", 
                    source_component="engine"
                )
            
            # Clean up campaign
            del self.active_campaigns[campaign_id]
            
            result = CampaignResult(campaign_id, True, report_path)
            result.findings_count = len(findings)
            
            await self.emit_event(
                EventType.TASK_COMPLETE,
                {"campaign_id": campaign_id, "success": True, "findings": len(findings)}
            )
            
            return result
            
        except Exception as e:
            await self.emit_event(
                EventType.TASK_FAILED,
                {"campaign_id": campaign_id, "error": str(e)},
                severity="ERROR"
            )
            
            # Record failure metrics
            campaign_duration = (datetime.utcnow() - campaign_start_time).total_seconds()
            if self.performance_monitor:
                await self.performance_monitor.record_metric(
                    "campaign_total_time", campaign_duration, "seconds",
                    context={"campaign_id": campaign_id, "failed": True},
                    category="campaign", source_component="engine"
                )
                await self.performance_monitor.record_metric(
                    "campaign_success_rate", 0.0, "ratio",
                    context={"campaign_id": campaign_id}, category="campaign", 
                    source_component="engine"
                )
                await self.performance_monitor.record_metric(
                    "campaign_error_rate", 1.0, "ratio",
                    context={"campaign_id": campaign_id, "error": str(e)}, 
                    category="campaign", source_component="engine"
                )
            
            # Clean up on failure
            if campaign_id in self.active_campaigns:
                del self.active_campaigns[campaign_id]
                
            logger.error(f"Campaign {campaign_id} failed: {e}")
            return CampaignResult(campaign_id, False)
            
    async def learn_about(self, topic: str):
        """Learn about a specific topic using web search and LLM collaboration"""
        if not self.web_searcher or not self.llm_collaborator:
            logger.error("Required modules not initialized for learning")
            return
            
        # Search for information
        search_results = await self.web_searcher.search(
            f"cybersecurity {topic} techniques 2024"
        )
        
        # Get LLM insights
        llm_insights = await self.llm_collaborator.get_tactical_advice(
            f"Explain the latest {topic} techniques and how they could be used ethically in red team operations"
        )
        
        # Learn from the information
        if self.knowledge_graph:
            await self.knowledge_graph.learn_from_sources(search_results + [llm_insights])
            
    async def analyze_target(self, target: str):
        """Perform initial target analysis"""
        if not self.reasoner or not self.web_searcher:
            logger.error("Required modules not initialized for analysis")
            return
            
        # Use reasoner to plan analysis approach
        analysis_plan = await self.reasoner.plan_analysis(target)
        
        # Execute reconnaissance
        if self.executor:
            results = await self.executor.execute_reconnaissance(target, analysis_plan)
            return results
            
    async def process_command(self, command: str):
        """Process a command in interactive mode"""
        if not self.reasoner:
            logger.error("Reasoner not initialized")
            return
            
        # Use reasoner to interpret and execute command
        interpretation = await self.reasoner.interpret_command(command)
        
        if interpretation.get("action"):
            await self._execute_interpreted_action(interpretation)
            
    async def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        uptime = None
        if self.startup_time:
            uptime = str(datetime.utcnow() - self.startup_time)
            
        knowledge_stats = {}
        if self.knowledge_graph:
            knowledge_stats = await self.knowledge_graph.get_statistics()
            
        return {
            "initialized": self.initialized,
            "persona": self.current_persona,
            "uptime": uptime,
            "active_campaigns": len(self.active_campaigns),
            "active_tasks": len(self.active_tasks),
            "knowledge_nodes": knowledge_stats.get("node_count", 0),
            "knowledge_edges": knowledge_stats.get("edge_count", 0),
            "event_stats": self.event_system.get_event_stats()
        }
        
    async def continuous_learning(self):
        """Run continuous learning process"""
        while self.initialized:
            try:
                # Learn about new techniques
                topics = ["web application security", "network penetration testing", 
                         "social engineering", "OSINT techniques", "mobile security"]
                
                for topic in topics:
                    await self.learn_about(topic)
                    await asyncio.sleep(60)  # Rate limiting
                    
                # Sleep before next learning cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in continuous learning: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
                
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard"""
        if not self.performance_monitor:
            return {"error": "Performance monitoring not initialized"}
            
        return await self.performance_monitor.get_performance_dashboard()
        
    async def optimize_performance(self, target_component: str = None, aggressive: bool = False):
        """Trigger performance optimization"""
        if not self.performance_monitor:
            logger.warning("Performance monitoring not available for optimization")
            return []
            
        return await self.performance_monitor.optimize_performance(target_component, aggressive)
        
    async def predict_performance_issues(self, hours_ahead: int = 24):
        """Predict potential performance issues"""
        if not self.performance_monitor:
            return []
            
        return await self.performance_monitor.predict_performance_issues(hours_ahead)
        
    async def submit_campaign(self, target: str, scope_file: str, 
                            priority: str = "medium", requested_by: str = "user") -> str:
        """Submit a campaign to the orchestrator"""
        if not self.campaign_orchestrator:
            logger.warning("Campaign orchestrator not available")
            return await self.run_campaign(target, scope_file)  # Fallback to direct execution
            
        # Convert priority string to enum
        priority_map = {
            "low": CampaignPriority.LOW,
            "medium": CampaignPriority.MEDIUM,
            "high": CampaignPriority.HIGH,
            "critical": CampaignPriority.CRITICAL
        }
        
        priority_enum = priority_map.get(priority.lower(), CampaignPriority.MEDIUM)
        
        return await self.campaign_orchestrator.submit_campaign(
            target=target,
            scope_file=scope_file,
            priority=priority_enum,
            requested_by=requested_by
        )
        
    async def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific campaign"""
        if not self.campaign_orchestrator:
            return None
            
        return await self.campaign_orchestrator.get_campaign_status(campaign_id)
        
    async def cancel_campaign(self, campaign_id: str) -> bool:
        """Cancel a campaign"""
        if not self.campaign_orchestrator:
            return False
            
        return await self.campaign_orchestrator.cancel_campaign(campaign_id)
        
    async def get_orchestration_dashboard(self) -> Dict[str, Any]:
        """Get orchestration dashboard"""
        if not self.campaign_orchestrator:
            return {"error": "Campaign orchestrator not initialized"}
            
        return await self.campaign_orchestrator.get_orchestration_dashboard()
        
    async def optimize_campaigns(self) -> Dict[str, Any]:
        """Optimize campaign scheduling and execution"""
        if not self.campaign_orchestrator:
            return {"error": "Campaign orchestrator not initialized"}
            
        return await self.campaign_orchestrator.optimize_scheduling()
                
    async def shutdown(self):
        """Graceful shutdown of all modules"""
        if not self.initialized:
            return
            
        try:
            await self.emit_event(EventType.SYSTEM_SHUTDOWN, {"message": "Shutting down Chimera"})
            
            # Stop background tasks
            for task in self.active_tasks.copy():
                task.cancel()
                
            # Shutdown modules in reverse order
            modules = [
                self.campaign_orchestrator, self.performance_monitor, self.reporter, self.validator, self.executor,
                self.planner, self.llm_collaborator, self.web_searcher,
                self.reasoner, self.knowledge_graph
            ]
            
            for module in modules:
                if module and hasattr(module, 'shutdown'):
                    try:
                        await module.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down module: {e}")
                        
            # Stop event system last
            await self.event_system.stop()
            
            self.initialized = False
            logger.info("Chimera engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    # Private methods
    
    async def _setup_event_subscriptions(self):
        """Setup event subscriptions for cross-module communication"""
        # Subscribe to critical events for monitoring
        await self.event_system.subscribe(EventType.SYSTEM_ERROR, self._handle_system_error)
        await self.event_system.subscribe(EventType.OPSEC_VIOLATION, self._handle_opsec_violation)
        
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Auto-save task
        save_interval = self.config.get("core.auto_save_interval", 300)
        save_task = asyncio.create_task(self._auto_save_loop(save_interval))
        self.active_tasks.add(save_task)
        
    async def _auto_save_loop(self, interval: int):
        """Periodically save system state"""
        while self.initialized:
            try:
                if self.knowledge_graph:
                    await self.knowledge_graph.save_state()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in auto-save: {e}")
                await asyncio.sleep(60)
                
    async def _handle_system_error(self, event: ChimeraEvent):
        """Handle system error events"""
        logger.error(f"System error from {event.source}: {event.data}")
        
    async def _handle_opsec_violation(self, event: ChimeraEvent):
        """Handle OPSEC violation events"""
        logger.warning(f"OPSEC violation detected: {event.data}")
        
        # Could implement automatic response here
        if event.data.get("severity") == "CRITICAL":
            # Pause operations
            pass
            
    async def _load_scope(self, scope_file: str) -> Optional[Dict[str, Any]]:
        """Load and validate bug bounty scope"""
        try:
            scope_path = Path(scope_file)
            if not scope_path.exists():
                logger.error(f"Scope file not found: {scope_file}")
                return None
                
            # Implementation depends on scope file format
            # This is a placeholder
            return {"domains": [], "exclusions": [], "rules": []}
            
        except Exception as e:
            logger.error(f"Error loading scope: {e}")
            return None
            
    async def _run_reconnaissance_phase(self, campaign_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run reconnaissance phase of campaign"""
        # Implementation placeholder
        return {"subdomains": [], "technologies": [], "endpoints": []}
        
    async def _run_assessment_phase(self, campaign_context: Dict[str, Any], recon_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run vulnerability assessment phase"""
        # Implementation placeholder
        return {"vulnerabilities": [], "attack_surface": []}
        
    async def _run_planning_phase(self, campaign_context: Dict[str, Any], vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Run exploitation planning phase"""
        # Implementation placeholder
        return {"attack_chains": [], "priority_targets": []}
        
    async def _run_exploitation_phase(self, campaign_context: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run controlled exploitation phase"""
        # Implementation placeholder
        return []
        
    async def _generate_campaign_report(self, campaign_context: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
        """Generate final campaign report"""
        if self.reporter:
            return await self.reporter.generate_campaign_report(campaign_context, findings)
        return ""
        
    async def _execute_interpreted_action(self, interpretation: Dict[str, Any]):
        """Execute an action interpreted from user command"""
        action = interpretation.get("action")
        params = interpretation.get("parameters", {})
        
        if action == "search":
            if self.web_searcher:
                await self.web_searcher.search(params.get("query", ""))
        elif action == "analyze":
            target = params.get("target")
            if target:
                await self.analyze_target(target)
        # Add more action handlers as needed