#!/usr/bin/env python3
"""
Aetherveil AI Pentesting Agent - Main Entry Point
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add local modules to path
sys.path.insert(0, str(Path(__file__).parent))

from agent import AetherVeilAgent
from config import Config
from utils import setup_logging, generate_session_id

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Shutdown signal received. Stopping agent...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Aetherveil AI Pentesting Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --category web --target http://localhost
  %(prog)s --autonomous --category infrastructure 
  %(prog)s --interactive
  %(prog)s --config custom_config.yaml --debug
        """
    )
    
    # Core arguments
    parser.add_argument("--config", "-c", default="config.yaml",
                       help="Configuration file path (default: config.yaml)")
    parser.add_argument("--category", choices=["web", "api", "cloud", "infrastructure", "identity", "supply_chain", "comprehensive"],
                       default="comprehensive", help="Testing category (default: comprehensive)")
    parser.add_argument("--target", "-t", help="Target to test (IP, domain, or URL)")
    
    # Operation modes
    parser.add_argument("--autonomous", "-a", action="store_true",
                       help="Run in autonomous mode (continuous operation)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--single-cycle", "-s", action="store_true",
                       help="Run single planning/execution cycle")
    
    # Configuration overrides
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--safe-mode", action="store_true",
                       help="Enable safe mode (localhost only)")
    parser.add_argument("--max-cycles", type=int, default=0,
                       help="Maximum cycles for autonomous mode (0 = unlimited)")
    parser.add_argument("--cycle-delay", type=int,
                       help="Delay between cycles in seconds")
    
    # GCP integration
    parser.add_argument("--enable-gcp", action="store_true",
                       help="Enable GCP integration")
    parser.add_argument("--gcp-project", help="GCP project ID")
    parser.add_argument("--gcp-service-account", help="GCP service account file path")
    
    # Output options
    parser.add_argument("--output-dir", "-o", help="Output directory for results")
    parser.add_argument("--session-id", help="Custom session ID")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config(args.config)
        print(f"📋 Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.debug:
        config.update_setting("execution", "debug_mode", True)
    
    if args.safe_mode:
        config.update_setting("security", "enable_safe_mode", True)
    
    if args.cycle_delay:
        config.update_setting("planning", "cycle_delay", args.cycle_delay)
    
    if args.enable_gcp:
        config.update_setting("gcp", "enabled", True)
        if args.gcp_project:
            config.update_setting("gcp", "project_id", args.gcp_project)
        if args.gcp_service_account:
            config.update_setting("gcp", "service_account_path", args.gcp_service_account)
    
    if args.output_dir:
        config.update_setting("storage", "base_dir", args.output_dir)
        config.update_setting("storage", "results_dir", f"{args.output_dir}/results")
        config.update_setting("storage", "reports_dir", f"{args.output_dir}/reports")
    
    # Setup logging
    log_config = {
        "debug": args.debug,
        "log_file": f"{config.storage.base_dir}/aetherveil.log" if not args.quiet else None
    }
    logger = setup_logging(log_config)
    
    # Generate session ID
    session_id = args.session_id or generate_session_id()
    
    # Initialize agent
    try:
        agent = AetherVeilAgent(config)
        if not await agent.initialize():
            logger.error("❌ Agent initialization failed")
            sys.exit(1)
        
        logger.info("🚀 Aetherveil AI Pentesting Agent started")
        logger.info(f"📝 Session ID: {session_id}")
        logger.info(f"🎯 Category: {args.category}")
        if args.target:
            logger.info(f"🔍 Target: {args.target}")
        
        # Run based on mode
        if args.interactive:
            await run_interactive_mode(agent, session_id, args.category)
        elif args.autonomous:
            await run_autonomous_mode(agent, session_id, args.category, args.target, args.max_cycles)
        elif args.single_cycle:
            await run_single_cycle(agent, session_id, args.category, args.target)
        else:
            # Default to single cycle
            await run_single_cycle(agent, session_id, args.category, args.target)
        
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            if 'agent' in locals():
                await agent.shutdown()
            logger.info("✅ Agent shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

async def run_interactive_mode(agent, session_id: str, default_category: str):
    """Run agent in interactive mode"""
    print("\n🎮 Interactive Mode - Aetherveil AI Pentesting Agent")
    print("Commands: run, status, results, config, help, quit")
    
    while True:
        try:
            command = input("\naetherveil> ").strip().lower()
            
            if command == "quit" or command == "exit":
                break
            elif command == "help":
                print_help()
            elif command == "status":
                await show_status(agent)
            elif command == "results":
                await show_recent_results(agent)
            elif command == "config":
                show_config(agent)
            elif command.startswith("run"):
                await handle_run_command(agent, session_id, command, default_category)
            elif command == "":
                continue
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_help():
    """Print interactive mode help"""
    print("""
Available commands:
  run [category] [target]  - Run pentesting cycle
  status                   - Show agent status
  results                  - Show recent results
  config                   - Show current configuration
  help                     - Show this help
  quit/exit               - Exit the agent

Categories: web, api, cloud, infrastructure, identity, supply_chain, comprehensive
Examples:
  run web http://localhost
  run infrastructure 192.168.1.1
  run comprehensive
    """)

async def show_status(agent):
    """Show agent status"""
    try:
        # Get system info and stats
        print("\n📊 Agent Status:")
        print(f"  Running: {agent.running}")
        if hasattr(agent, 'current_cycle'):
            print(f"  Current Cycle: {agent.current_cycle}")
        print(f"  Session ID: {getattr(agent, 'session_id', 'N/A')}")
        
        # Component status
        components = ['planner', 'executor', 'collector', 'learner']
        for comp in components:
            if hasattr(agent, comp):
                print(f"  {comp.title()}: ✅ Initialized")
            else:
                print(f"  {comp.title()}: ❌ Not initialized")
        
    except Exception as e:
        print(f"❌ Failed to get status: {e}")

async def show_recent_results(agent):
    """Show recent results"""
    try:
        if hasattr(agent, 'collector'):
            results = await agent.collector.get_recent_results(hours=24)
            findings = await agent.collector.get_recent_findings(hours=24)
            
            print(f"\n📈 Recent Results (last 24h):")
            print(f"  Total Results: {len(results)}")
            print(f"  Vulnerabilities Found: {len(findings)}")
            
            if findings:
                print("  Recent Vulnerabilities:")
                for finding in findings[:5]:  # Show top 5
                    print(f"    - {finding.get('type', 'Unknown')} ({finding.get('severity', 'unknown')})")
        else:
            print("❌ Collector not initialized")
            
    except Exception as e:
        print(f"❌ Failed to get results: {e}")

def show_config(agent):
    """Show current configuration"""
    try:
        config_dict = agent.config.get_config_dict()
        print("\n⚙️ Current Configuration:")
        for section, settings in config_dict.items():
            print(f"  {section}:")
            for key, value in settings.items():
                print(f"    {key}: {value}")
                
    except Exception as e:
        print(f"❌ Failed to get configuration: {e}")

async def handle_run_command(agent, session_id: str, command: str, default_category: str):
    """Handle run command in interactive mode"""
    try:
        parts = command.split()
        category = parts[1] if len(parts) > 1 else default_category
        target = parts[2] if len(parts) > 2 else None
        
        print(f"\n🚀 Running {category} assessment...")
        if target:
            print(f"🎯 Target: {target}")
        
        # Run single cycle
        await agent.run_cycle(category=category, target=target)
        print("✅ Cycle completed")
        
    except Exception as e:
        print(f"❌ Run failed: {e}")

async def run_autonomous_mode(agent, session_id: str, category: str, target: str, max_cycles: int):
    """Run agent in autonomous mode"""
    print(f"\n🤖 Starting autonomous mode - Category: {category}")
    if max_cycles > 0:
        print(f"📊 Maximum cycles: {max_cycles}")
    if target:
        print(f"🎯 Target: {target}")
    
    print("Press Ctrl+C to stop gracefully...")
    
    try:
        await agent.autonomous_loop(category=category, target=target, max_cycles=max_cycles)
    except KeyboardInterrupt:
        print("\n🛑 Stopping autonomous mode...")
        agent.running = False

async def run_single_cycle(agent, session_id: str, category: str, target: str):
    """Run single planning/execution cycle"""
    print(f"\n🔄 Running single cycle - Category: {category}")
    if target:
        print(f"🎯 Target: {target}")
    
    try:
        await agent.run_cycle(category=category, target=target)
        print("✅ Cycle completed successfully")
    except Exception as e:
        print(f"❌ Cycle failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())