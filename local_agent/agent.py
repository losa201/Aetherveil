#!/usr/bin/env python3
"""
Aetherveil 3.0 - Autonomous AI Pentesting Agent
Main orchestrator for hybrid local + GCP AI-driven security testing
"""

import asyncio
import argparse
import logging
import json
import os
import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Local modules
from planner import AIPlanner
from executor import ToolExecutor
from collector import ResultCollector
from learner import KnowledgeLearner
from config import Config
from utils import setup_logging, graceful_shutdown

class AetherveilvAgent:
    """Main autonomous AI pentesting agent orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the autonomous agent"""
        self.config = Config(config_path)
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.current_session_id = None
        
        # Initialize components
        self.planner = AIPlanner(self.config)
        self.executor = ToolExecutor(self.config)
        self.collector = ResultCollector(self.config)
        self.learner = KnowledgeLearner(self.config)
        
        # Performance tracking
        self.stats = {
            "start_time": datetime.now(),
            "tasks_completed": 0,
            "vulnerabilities_found": 0,
            "learning_cycles": 0,
            "cost_spent": 0.0
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def initialize(self) -> bool:
        """Initialize all agent components"""
        try:
            self.logger.info("🚀 Initializing Aetherveil autonomous agent...")
            
            # Initialize local LLM
            if not await self.planner.initialize():
                self.logger.error("Failed to initialize AI planner")
                return False
            
            # Initialize tool executor
            if not await self.executor.initialize():
                self.logger.error("Failed to initialize tool executor")
                return False
            
            # Initialize result collector
            if not await self.collector.initialize():
                self.logger.error("Failed to initialize result collector")
                return False
            
            # Initialize knowledge learner
            if not await self.learner.initialize():
                self.logger.error("Failed to initialize knowledge learner")
                return False
            
            # Create session ID
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info("✅ Agent initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def autonomous_loop(self, category: str = "comprehensive", target: Optional[str] = None) -> None:
        """Main autonomous operation loop"""
        self.running = True
        cycle_count = 0
        
        self.logger.info(f"🤖 Starting autonomous loop for category: {category}")
        
        while self.running:
            try:
                cycle_count += 1
                cycle_start = datetime.now()
                
                self.logger.info(f"🔄 Autonomous cycle {cycle_count} started")
                
                # Phase 1: Planning
                plan = await self._planning_phase(category, target, cycle_count)
                if not plan:
                    await asyncio.sleep(self.config.planning.cycle_delay)
                    continue
                
                # Phase 2: Execution
                results = await self._execution_phase(plan)
                
                # Phase 3: Collection & Analysis
                analyzed_results = await self._collection_phase(results)
                
                # Phase 4: Learning & Knowledge Update
                await self._learning_phase(analyzed_results, plan)
                
                # Phase 5: Progress Reporting
                await self._reporting_phase(cycle_count, cycle_start)
                
                # Adaptive sleep based on findings
                sleep_duration = self._calculate_sleep_duration(analyzed_results)
                await asyncio.sleep(sleep_duration)
                
            except Exception as e:
                self.logger.error(f"Error in autonomous cycle {cycle_count}: {e}")
                await asyncio.sleep(self.config.error_recovery.retry_delay)
    
    async def _planning_phase(self, category: str, target: Optional[str], cycle: int) -> Optional[Dict]:
        """AI-driven planning phase"""
        try:
            self.logger.info("🧠 Planning phase: Generating next actions...")
            
            # Get current knowledge state
            knowledge = await self.learner.get_current_knowledge(category)
            
            # Get recent results for context
            recent_results = await self.collector.get_recent_results(hours=24)
            
            # Generate plan using local LLM
            plan = await self.planner.generate_plan(
                category=category,
                target=target,
                knowledge=knowledge,
                recent_results=recent_results,
                cycle_number=cycle
            )
            
            if plan:
                self.logger.info(f"✅ Generated plan with {len(plan.get('tasks', []))} tasks")
                
                # Log plan to local storage
                await self.collector.log_plan(plan, self.current_session_id)
                
            return plan
            
        except Exception as e:
            self.logger.error(f"Planning phase failed: {e}")
            return None
    
    async def _execution_phase(self, plan: Dict) -> List[Dict]:
        """Execute planned pentesting tasks"""
        results = []
        
        try:
            self.logger.info(f"⚡ Execution phase: Running {len(plan.get('tasks', []))} tasks...")
            
            for task in plan.get('tasks', []):
                if not self.running:
                    break
                
                self.logger.info(f"🔧 Executing task: {task.get('name', 'Unknown')}")
                
                # Execute task with timeout
                result = await self.executor.execute_task(
                    task, 
                    timeout=self.config.execution.task_timeout
                )
                
                if result:
                    results.append(result)
                    self.stats["tasks_completed"] += 1
                    
                    # Real-time vulnerability detection
                    if result.get('vulnerabilities'):
                        self.stats["vulnerabilities_found"] += len(result['vulnerabilities'])
                        self.logger.warning(f"🚨 Found {len(result['vulnerabilities'])} vulnerabilities!")
                
                # Brief pause between tasks
                await asyncio.sleep(self.config.execution.task_delay)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Execution phase failed: {e}")
            return results
    
    async def _collection_phase(self, results: List[Dict]) -> List[Dict]:
        """Collect and analyze execution results"""
        try:
            self.logger.info(f"📊 Collection phase: Analyzing {len(results)} results...")
            
            analyzed_results = []
            
            for result in results:
                # Parse and enrich result data
                analyzed = await self.collector.analyze_result(result)
                
                if analyzed:
                    analyzed_results.append(analyzed)
                    
                    # Store result locally
                    await self.collector.store_result(analyzed, self.current_session_id)
                    
                    # Send to GCP if configured
                    if self.config.gcp.enabled:
                        await self.collector.send_to_gcp(analyzed)
            
            self.logger.info(f"✅ Analyzed {len(analyzed_results)} results")
            return analyzed_results
            
        except Exception as e:
            self.logger.error(f"Collection phase failed: {e}")
            return []
    
    async def _learning_phase(self, results: List[Dict], plan: Dict) -> None:
        """Update knowledge base and improve tactics"""
        try:
            self.logger.info("🎓 Learning phase: Updating knowledge base...")
            
            # Update local knowledge
            learning_updates = await self.learner.process_results(results, plan)
            
            if learning_updates:
                self.stats["learning_cycles"] += 1
                self.logger.info(f"📚 Applied {len(learning_updates)} knowledge updates")
                
                # Sync with GCP if enabled
                if self.config.gcp.enabled:
                    await self.learner.sync_with_gcp(learning_updates)
            
            # Periodic model retraining
            if self.stats["learning_cycles"] % self.config.learning.retrain_frequency == 0:
                await self.learner.retrain_models()
            
        except Exception as e:
            self.logger.error(f"Learning phase failed: {e}")
    
    async def _reporting_phase(self, cycle: int, cycle_start: datetime) -> None:
        """Generate progress reports and metrics"""
        try:
            cycle_duration = datetime.now() - cycle_start
            
            # Update statistics
            self.stats["cycle_duration"] = cycle_duration.total_seconds()
            
            # Log progress
            self.logger.info(f"📈 Cycle {cycle} completed in {cycle_duration.total_seconds():.1f}s")
            self.logger.info(f"📊 Stats: {self.stats['tasks_completed']} tasks, "
                           f"{self.stats['vulnerabilities_found']} vulnerabilities, "
                           f"{self.stats['learning_cycles']} learning cycles")
            
            # Generate detailed report periodically
            if cycle % self.config.reporting.detailed_frequency == 0:
                await self._generate_detailed_report(cycle)
            
        except Exception as e:
            self.logger.error(f"Reporting phase failed: {e}")
    
    def _calculate_sleep_duration(self, results: List[Dict]) -> float:
        """Calculate adaptive sleep duration based on findings"""
        base_sleep = self.config.planning.cycle_delay
        
        # Reduce sleep if vulnerabilities found (more aggressive)
        vuln_count = sum(len(r.get('vulnerabilities', [])) for r in results)
        if vuln_count > 0:
            return max(base_sleep * 0.5, 10)  # Min 10 seconds
        
        # Increase sleep if no interesting findings
        if not any(r.get('interesting_findings') for r in results):
            return base_sleep * 1.5
        
        return base_sleep
    
    async def _generate_detailed_report(self, cycle: int) -> None:
        """Generate comprehensive progress report"""
        try:
            report = {
                "session_id": self.current_session_id,
                "cycle": cycle,
                "timestamp": datetime.now().isoformat(),
                "statistics": self.stats.copy(),
                "uptime": str(datetime.now() - self.stats["start_time"]),
                "recent_findings": await self.collector.get_recent_findings(hours=24)
            }
            
            # Save report locally
            report_path = Path(self.config.storage.reports_dir) / f"report_cycle_{cycle}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"📄 Detailed report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate detailed report: {e}")
    
    async def run_single_category(self, category: str, target: Optional[str] = None) -> Dict:
        """Run focused assessment on specific category"""
        try:
            self.logger.info(f"🎯 Running single category assessment: {category}")
            
            # Generate focused plan
            plan = await self.planner.generate_focused_plan(category, target)
            
            if not plan:
                raise ValueError(f"Failed to generate plan for category: {category}")
            
            # Execute plan
            results = await self._execution_phase(plan)
            analyzed_results = await self._collection_phase(results)
            await self._learning_phase(analyzed_results, plan)
            
            # Generate summary
            summary = {
                "category": category,
                "target": target,
                "session_id": self.current_session_id,
                "tasks_executed": len(results),
                "vulnerabilities_found": sum(len(r.get('vulnerabilities', [])) for r in analyzed_results),
                "findings": analyzed_results
            }
            
            self.logger.info(f"✅ Category assessment complete: {json.dumps(summary, indent=2, default=str)}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Single category run failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Graceful shutdown of all components"""
        try:
            self.logger.info("🔄 Shutting down agent components...")
            
            self.running = False
            
            # Shutdown components
            await self.learner.shutdown()
            await self.collector.shutdown()
            await self.executor.shutdown()
            await self.planner.shutdown()
            
            # Final statistics
            uptime = datetime.now() - self.stats["start_time"]
            self.logger.info(f"📊 Final stats - Uptime: {uptime}, "
                           f"Tasks: {self.stats['tasks_completed']}, "
                           f"Vulnerabilities: {self.stats['vulnerabilities_found']}")
            
            self.logger.info("✅ Agent shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Aetherveil 3.0 Autonomous AI Pentesting Agent")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--category", default="comprehensive", 
                       choices=["web", "api", "cloud", "infrastructure", "identity", "supply_chain", "comprehensive"],
                       help="Pentesting category focus")
    parser.add_argument("--target", help="Specific target to focus on")
    parser.add_argument("--mode", default="autonomous", choices=["autonomous", "single", "daemon"],
                       help="Operation mode")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without actual execution")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize agent
        agent = AetherveilvAgent(args.config)
        
        if not await agent.initialize():
            logger.error("❌ Agent initialization failed")
            sys.exit(1)
        
        logger.info(f"🚀 Starting Aetherveil agent in {args.mode} mode")
        
        # Run based on mode
        if args.mode == "autonomous":
            await agent.autonomous_loop(args.category, args.target)
        elif args.mode == "single":
            result = await agent.run_single_category(args.category, args.target)
            print(json.dumps(result, indent=2, default=str))
        elif args.mode == "daemon":
            # Run as background daemon
            while True:
                await agent.autonomous_loop(args.category, args.target)
                await asyncio.sleep(3600)  # 1 hour between cycles
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        if 'agent' in locals():
            await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())