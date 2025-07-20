#!/usr/bin/env python3
"""
Full Aether Agent Runner with Dependencies

Runs the complete Aether system with all available dependencies.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the aether directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "aether"))

def check_dependencies():
    """Check which dependencies are available"""
    deps = {
        'playwright': False,
        'fake_useragent': False,
        'neo4j': False,
        'numpy': False,
        'aiofiles': False
    }
    
    try:
        import playwright
        deps['playwright'] = True
    except ImportError:
        pass
        
    try:
        import fake_useragent
        deps['fake_useragent'] = True
    except ImportError:
        pass
        
    try:
        import neo4j
        deps['neo4j'] = True
    except ImportError:
        pass
        
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        pass
        
    try:
        import aiofiles
        deps['aiofiles'] = True
    except ImportError:
        pass
        
    return deps

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('aether_full.log')
        ]
    )

async def run_full_aether():
    """Run the full Aether agent with all capabilities"""
    
    # Check dependencies
    deps = check_dependencies()
    
    print("🧠 Aether - Neuroplastic AI Learning System (Full Version)")
    print("=" * 70)
    print("Dependency Status:")
    
    for dep, available in deps.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {dep:15} {status}")
    
    print("=" * 70)
    
    # Determine capabilities based on available dependencies
    capabilities = []
    
    if deps['playwright']:
        capabilities.append("🌐 Advanced browser automation with stealth")
        capabilities.append("🤖 Real LLM mentor interactions")
        capabilities.append("📧 Gmail account creation")
    else:
        print("⚠️  Browser automation disabled - install playwright")
        
    if deps['neo4j']:
        capabilities.append("🧠 Advanced Neo4j knowledge graph")
    else:
        capabilities.append("💾 SQLite knowledge storage (fallback)")
        
    if deps['fake_useragent']:
        capabilities.append("🥸 Advanced fingerprint randomization")
    else:
        capabilities.append("🔒 Basic stealth capabilities")
        
    if deps['numpy']:
        capabilities.append("🔢 Enhanced numerical processing")
        
    print("Active Capabilities:")
    for cap in capabilities:
        print(f"  {cap}")
    print("=" * 70)
    
    try:
        # Import and run the main agent
        from aether_agent import main
        
        print("🚀 Starting Full Aether Agent...")
        print()
        
        # Run the main agent
        await main()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Note: Running with available dependencies...")
        # Try to run with basic capabilities
        try:
            from simple_aether_demo import main as simple_main
            print("🔄 Falling back to simplified version...")
            await simple_main()
        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")
    except Exception as e:
        print(f"❌ Error running Full Aether: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    setup_logging()
    
    try:
        asyncio.run(run_full_aether())
    except KeyboardInterrupt:
        print("\n👋 Aether shutdown complete")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()