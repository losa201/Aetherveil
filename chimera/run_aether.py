#!/usr/bin/env python3
"""
Aether Agent Runner

Simple script to run the Aether neuroplastic AI learning system.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the aether directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "aether"))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('aether.log')
        ]
    )

async def run_aether():
    """Run the Aether agent"""
    try:
        # Import after path setup
        from aether_agent import main
        
        print("🧠 Starting Aether - Neuroplastic AI Learning System...")
        print("=" * 60)
        print("Features:")
        print("• Self-aware cognitive core with curiosity-driven learning")
        print("• Code evolution engine for self-improvement")
        print("• LLM mentor relationship building")
        print("• Advanced stealth browser automation")
        print("• Neuroplastic memory with knowledge synthesis")
        print("• Digital identity management")
        print("=" * 60)
        
        # Run the main agent
        await main()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Note: Some advanced features require additional dependencies:")
        print("- playwright (for browser automation)")
        print("- neo4j (for advanced memory)")
        print("- fake-useragent (for stealth)")
        print("\nRunning in basic mode...")
    except Exception as e:
        print(f"❌ Error running Aether: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    setup_logging()
    
    try:
        asyncio.run(run_aether())
    except KeyboardInterrupt:
        print("\n👋 Aether shutdown complete")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()