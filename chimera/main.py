#!/usr/bin/env python3
"""
Chimera: Neuroplastic Autonomous Red-Team Organism
Main entry point and CLI interface
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from chimera.core.engine import ChimeraEngine
from chimera.utils.config import ConfigManager
from chimera.utils.logging import setup_logging

console = Console()

class ChimeraRunner:
    """Main runner for Chimera operations"""
    
    def __init__(self):
        self.engine: Optional[ChimeraEngine] = None
        self.config_manager = ConfigManager()
        self.running = False
        
    async def initialize(self, config_path: str, debug: bool = False):
        """Initialize Chimera with configuration"""
        try:
            # Load configuration
            await self.config_manager.load(config_path)
            
            # Setup logging
            log_level = "DEBUG" if debug else self.config_manager.get("logging.level", "INFO")
            setup_logging(
                level=log_level,
                file_path=self.config_manager.get("logging.file", "./data/logs/chimera.log")
            )
            
            # Initialize core engine
            self.engine = ChimeraEngine(self.config_manager)
            await self.engine.initialize()
            
            console.print(Panel(
                Text("🔥 Chimera Initialized Successfully", style="bold green"),
                title="Neuroplastic Red-Team Organism",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"❌ Failed to initialize Chimera: {e}", style="bold red")
            sys.exit(1)
            
    async def run_campaign(self, target: str, scope_file: str, persona: str):
        """Run a red-team campaign"""
        if not self.engine:
            console.print("❌ Chimera not initialized", style="bold red")
            return
            
        try:
            # Load scope configuration
            scope_path = Path(scope_file)
            if not scope_path.exists():
                console.print(f"❌ Scope file not found: {scope_file}", style="bold red")
                return
                
            # Set persona
            await self.engine.set_persona(persona)
            
            # Start campaign
            console.print(f"🎯 Starting campaign against {target} with persona: {persona}")
            
            self.running = True
            campaign_result = await self.engine.run_campaign(target, scope_file)
            
            if campaign_result:
                console.print("✅ Campaign completed successfully", style="bold green")
                console.print(f"📊 Report generated: {campaign_result.report_path}")
            else:
                console.print("⚠️ Campaign completed with issues", style="bold yellow")
                
        except KeyboardInterrupt:
            console.print("\n🛑 Campaign interrupted by user", style="bold yellow")
            await self.shutdown()
        except Exception as e:
            console.print(f"❌ Campaign failed: {e}", style="bold red")
            
    async def interactive_mode(self):
        """Run Chimera in interactive mode"""
        if not self.engine:
            console.print("❌ Chimera not initialized", style="bold red")
            return
            
        console.print(Panel(
            "🧠 Entering Interactive Mode\nType 'help' for commands, 'exit' to quit",
            title="Chimera Interactive Shell",
            border_style="blue"
        ))
        
        try:
            while True:
                command = await asyncio.to_thread(
                    input, 
                    console.input("chimera> ")
                )
                
                if command.lower() in ['exit', 'quit']:
                    break
                elif command.lower() == 'help':
                    await self.show_help()
                elif command.lower() == 'status':
                    await self.show_status()
                elif command.startswith('persona '):
                    persona = command.split(' ', 1)[1]
                    await self.engine.set_persona(persona)
                elif command.startswith('learn '):
                    topic = command.split(' ', 1)[1]
                    await self.engine.learn_about(topic)
                elif command.startswith('analyze '):
                    target = command.split(' ', 1)[1]
                    await self.engine.analyze_target(target)
                elif command.lower() == 'performance':
                    await self.show_performance()
                elif command.lower() == 'optimize':
                    await self.run_optimization()
                elif command.lower() == 'predict':
                    await self.show_predictions()
                elif command.lower() == 'campaigns':
                    await self.show_campaigns()
                elif command.startswith('submit '):
                    parts = command.split(' ')
                    if len(parts) >= 3:
                        target = parts[1]
                        scope_file = parts[2]
                        priority = parts[3] if len(parts) > 3 else "medium"
                        await self.submit_campaign(target, scope_file, priority)
                    else:
                        console.print("Usage: submit <target> <scope_file> [priority]", style="yellow")
                elif command.startswith('status '):
                    campaign_id = command.split(' ', 1)[1]
                    await self.show_campaign_status(campaign_id)
                elif command.startswith('cancel '):
                    campaign_id = command.split(' ', 1)[1]
                    await self.cancel_campaign(campaign_id)
                elif command.lower() == 'optimize-campaigns':
                    await self.optimize_campaigns()
                else:
                    await self.engine.process_command(command)
                    
        except KeyboardInterrupt:
            console.print("\n👋 Exiting interactive mode", style="bold blue")
            
    async def show_help(self):
        """Show available commands"""
        help_text = """
Available Commands:
  status              - Show current status
  persona <name>      - Switch to persona (cautious/aggressive/creative)
  learn <topic>       - Learn about a specific topic
  analyze <target>    - Analyze a target
  performance         - Show performance dashboard
  optimize            - Run performance optimization
  predict             - Show performance predictions
  campaigns           - Show campaign orchestration dashboard
  submit <target> <scope> [priority] - Submit new campaign
  status <campaign_id> - Show campaign status
  cancel <campaign_id> - Cancel campaign
  optimize-campaigns  - Optimize campaign scheduling
  exit/quit          - Exit interactive mode
        """
        console.print(Panel(help_text, title="Help", border_style="blue"))
        
    async def show_status(self):
        """Show current Chimera status"""
        if self.engine:
            status = await self.engine.get_status()
            console.print(Panel(
                f"Active Persona: {status.get('persona', 'Unknown')}\n"
                f"Knowledge Nodes: {status.get('knowledge_nodes', 0)}\n"
                f"Active Tasks: {status.get('active_tasks', 0)}\n"
                f"Uptime: {status.get('uptime', 'Unknown')}",
                title="Chimera Status",
                border_style="green"
            ))
        else:
            console.print("❌ Engine not initialized", style="bold red")
            
    async def show_performance(self):
        """Show performance dashboard"""
        if not self.engine:
            console.print("❌ Engine not initialized", style="bold red")
            return
            
        try:
            dashboard = await self.engine.get_performance_dashboard()
            
            if "error" in dashboard:
                console.print(f"❌ {dashboard['error']}", style="bold red")
                return
                
            # System health
            health = dashboard.get("system_health", {})
            health_score = health.get("overall_score", 0.0)
            health_status = health.get("status", "unknown")
            
            console.print(Panel(
                f"Overall Health: {health_score:.2f} ({health_status})\n"
                f"Metrics Collected: {dashboard.get('performance_metrics', {}).get('total_metrics_collected', 0)}\n"
                f"Active Bottlenecks: {len(dashboard.get('bottlenecks', {}).get('active_bottlenecks', []))}\n"
                f"Recent Optimizations: {dashboard.get('optimizations', {}).get('recent_optimizations', 0)}",
                title="🚀 Performance Dashboard",
                border_style="cyan"
            ))
            
            # Resource utilization
            resources = dashboard.get("resource_utilization", {})
            if resources and "error" not in resources:
                console.print(Panel(
                    f"CPU: {resources.get('cpu', {}).get('current', 0):.1f}% ({resources.get('cpu', {}).get('status', 'unknown')})\n"
                    f"Memory: {resources.get('memory', {}).get('current', 0):.1f}% ({resources.get('memory', {}).get('status', 'unknown')})\n"
                    f"Disk: {resources.get('disk', {}).get('current', 0):.1f}% ({resources.get('disk', {}).get('status', 'unknown')})",
                    title="💻 Resource Utilization",
                    border_style="blue"
                ))
                
        except Exception as e:
            console.print(f"❌ Error getting performance data: {e}", style="bold red")
            
    async def run_optimization(self):
        """Run performance optimization"""
        if not self.engine:
            console.print("❌ Engine not initialized", style="bold red")
            return
            
        try:
            console.print("🔄 Running performance optimization...", style="bold yellow")
            optimizations = await self.engine.optimize_performance()
            
            if optimizations:
                console.print(f"✅ Applied {len(optimizations)} optimizations:", style="bold green")
                for opt in optimizations:
                    result = opt.result or {}
                    status = "✅" if result.get("success", False) else "❌"
                    console.print(f"  {status} {opt.description}")
            else:
                console.print("ℹ️ No optimizations needed at this time", style="blue")
                
        except Exception as e:
            console.print(f"❌ Error running optimization: {e}", style="bold red")
            
    async def show_predictions(self):
        """Show performance predictions"""
        if not self.engine:
            console.print("❌ Engine not initialized", style="bold red")
            return
            
        try:
            predictions = await self.engine.predict_performance_issues(24)
            
            if predictions:
                console.print("⚠️ Predicted Performance Issues (24h):", style="bold yellow")
                for pred in predictions:
                    severity_style = "red" if pred["severity"] == "critical" else "yellow"
                    console.print(
                        f"  • {pred['metric']}: {pred['predicted_value']:.2f} "
                        f"(confidence: {pred['confidence']:.1%})",
                        style=severity_style
                    )
            else:
                console.print("✅ No performance issues predicted", style="bold green")
                
        except Exception as e:
            console.print(f"❌ Error getting predictions: {e}", style="bold red")
            
    async def show_campaigns(self):
        """Show campaign orchestration dashboard"""
        if not self.engine:
            console.print("❌ Engine not initialized", style="bold red")
            return
            
        try:
            dashboard = await self.engine.get_orchestration_dashboard()
            
            if "error" in dashboard:
                console.print(f"❌ {dashboard['error']}", style="bold red")
                return
                
            # Campaign statistics
            stats = dashboard.get("campaign_statistics", {})
            console.print(Panel(
                f"Total Campaigns: {stats.get('total_campaigns', 0)}\n"
                f"Running: {stats.get('running_campaigns', 0)} | "
                f"Pending: {stats.get('pending_campaigns', 0)} | "
                f"Completed: {stats.get('completed_campaigns', 0)}\n"
                f"Success Rate: {stats.get('success_rate', 0.0):.1%}",
                title="📊 Campaign Statistics",
                border_style="cyan"
            ))
            
            # Resource utilization
            resources = dashboard.get("resource_utilization", {})
            console.print(Panel(
                f"CPU: {resources.get('cpu_utilization', 0.0):.1%} | "
                f"Memory: {resources.get('memory_utilization', 0.0):.1%} | "
                f"Network: {resources.get('network_utilization', 0.0):.1%}\n"
                f"Available Slots: {resources.get('available_slots', 0)}",
                title="💻 Resource Utilization",
                border_style="blue"
            ))
            
            # Queue analysis
            queue = dashboard.get("queue_analysis", {})
            if queue.get("queue_by_priority"):
                priority_info = " | ".join([f"{k}: {v}" for k, v in queue["queue_by_priority"].items()])
                console.print(Panel(
                    f"Queue by Priority: {priority_info}\n"
                    f"Estimated Queue Time: {queue.get('estimated_queue_time', 0.0):.1f}h",
                    title="📋 Queue Analysis",
                    border_style="yellow"
                ))
                
        except Exception as e:
            console.print(f"❌ Error getting campaign data: {e}", style="bold red")
            
    async def submit_campaign(self, target: str, scope_file: str, priority: str):
        """Submit a new campaign"""
        if not self.engine:
            console.print("❌ Engine not initialized", style="bold red")
            return
            
        try:
            console.print(f"🚀 Submitting campaign for {target}...", style="bold yellow")
            campaign_id = await self.engine.submit_campaign(target, scope_file, priority)
            
            console.print(f"✅ Campaign submitted: {campaign_id}", style="bold green")
            console.print(f"Priority: {priority.upper()}", style="blue")
            
        except Exception as e:
            console.print(f"❌ Error submitting campaign: {e}", style="bold red")
            
    async def show_campaign_status(self, campaign_id: str):
        """Show status of a specific campaign"""
        if not self.engine:
            console.print("❌ Engine not initialized", style="bold red")
            return
            
        try:
            status = await self.engine.get_campaign_status(campaign_id)
            
            if not status:
                console.print(f"❌ Campaign {campaign_id} not found", style="bold red")
                return
                
            console.print(Panel(
                f"Campaign ID: {status['campaign_id']}\n"
                f"Status: {status['status'].upper()}\n"
                f"Progress: {status.get('progress', 0.0):.1%}\n"
                f"Phase: {status.get('current_phase', 'Unknown')}\n"
                f"Findings: {status.get('findings_count', 0)}",
                title=f"🎯 Campaign Status",
                border_style="green" if status['status'] == 'completed' else "yellow"
            ))
            
            if status.get('estimated_completion'):
                console.print(f"📅 Estimated completion: {status['estimated_completion']}", style="blue")
                
        except Exception as e:
            console.print(f"❌ Error getting campaign status: {e}", style="bold red")
            
    async def cancel_campaign(self, campaign_id: str):
        """Cancel a campaign"""
        if not self.engine:
            console.print("❌ Engine not initialized", style="bold red")
            return
            
        try:
            success = await self.engine.cancel_campaign(campaign_id)
            
            if success:
                console.print(f"✅ Campaign {campaign_id} cancelled", style="bold green")
            else:
                console.print(f"❌ Could not cancel campaign {campaign_id}", style="bold red")
                
        except Exception as e:
            console.print(f"❌ Error cancelling campaign: {e}", style="bold red")
            
    async def optimize_campaigns(self):
        """Optimize campaign scheduling"""
        if not self.engine:
            console.print("❌ Engine not initialized", style="bold red")
            return
            
        try:
            console.print("🔄 Optimizing campaign scheduling...", style="bold yellow")
            results = await self.engine.optimize_campaigns()
            
            if "error" in results:
                console.print(f"❌ {results['error']}", style="bold red")
                return
                
            actions = results.get("actions_taken", [])
            if actions:
                console.print(f"✅ Applied {len(actions)} optimizations:", style="bold green")
                for action in actions:
                    console.print(f"  • {action}", style="green")
            else:
                console.print("ℹ️ No optimizations needed at this time", style="blue")
                
            # Show performance improvements
            improvements = results.get("performance_improvements", {})
            if improvements:
                console.print("\n📈 Performance Improvements:", style="bold cyan")
                for metric, value in improvements.items():
                    console.print(f"  • {metric}: {value:.2%}", style="cyan")
                    
        except Exception as e:
            console.print(f"❌ Error optimizing campaigns: {e}", style="bold red")
            
    async def shutdown(self):
        """Graceful shutdown"""
        if self.engine:
            console.print("🔄 Shutting down Chimera...", style="bold yellow")
            await self.engine.shutdown()
            console.print("✅ Shutdown complete", style="bold green")
        self.running = False

# CLI Commands
@click.group()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, debug):
    """Chimera: Neuroplastic Autonomous Red-Team Organism"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['debug'] = debug

@cli.command()
@click.argument('target')
@click.option('--scope', required=True, help='Bug bounty scope file')
@click.option('--persona', default='balanced', help='Persona to use')
@click.pass_context
def campaign(ctx, target, scope, persona):
    """Run a red-team campaign against a target"""
    async def run():
        runner = ChimeraRunner()
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            asyncio.create_task(runner.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        await runner.initialize(ctx.obj['config'], ctx.obj['debug'])
        await runner.run_campaign(target, scope, persona)
        await runner.shutdown()
        
    asyncio.run(run())

@cli.command()
@click.pass_context
def interactive(ctx):
    """Run Chimera in interactive mode"""
    async def run():
        runner = ChimeraRunner()
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            asyncio.create_task(runner.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        await runner.initialize(ctx.obj['config'], ctx.obj['debug'])
        await runner.interactive_mode()
        await runner.shutdown()
        
    asyncio.run(run())

@cli.command()
@click.pass_context
def learn(ctx):
    """Learn new techniques from the web"""
    async def run():
        runner = ChimeraRunner()
        await runner.initialize(ctx.obj['config'], ctx.obj['debug'])
        
        if runner.engine:
            await runner.engine.continuous_learning()
            
        await runner.shutdown()
        
    asyncio.run(run())

@cli.command()
@click.pass_context
def status(ctx):
    """Show Chimera status and health"""
    async def run():
        runner = ChimeraRunner()
        await runner.initialize(ctx.obj['config'], ctx.obj['debug'])
        await runner.show_status()
        await runner.shutdown()
        
    asyncio.run(run())

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="bold blue")
        sys.exit(0)
    except Exception as e:
        console.print(f"❌ Fatal error: {e}", style="bold red")
        sys.exit(1)