#!/usr/bin/env python3
"""
Chimera Lite: Minimal dependency entry point
Fallback launcher that works with basic Python installation
"""

import sys
import os
import subprocess
from pathlib import Path

def check_and_install_deps():
    """Check for critical dependencies and install if needed"""
    
    critical_deps = [
        'aiohttp',
        'aiofiles', 
        'pyyaml',
        'click',
        'requests'
    ]
    
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("🔧 Installing minimal dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_deps)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please run: pip install aiohttp aiofiles pyyaml click requests")
            return False
    
    return True

def create_minimal_config():
    """Create minimal configuration if none exists"""
    
    config_content = """
# Minimal Chimera Configuration
core:
  max_concurrent_tasks: 2
  debug: false
  
persona:
  default: "balanced"
  
memory:
  max_nodes: 1000
  graph_database: "./data/knowledge/graph.db"
  
web:
  headless: true
  max_search_results: 10
  
logging:
  level: "INFO"
  file: "./data/logs/chimera.log"
"""
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content.strip())
        print("✅ Created minimal config.yaml")
    
    # Create data directories
    for directory in ["data/knowledge", "data/logs", "data/reports"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def launch_chimera():
    """Launch Chimera with minimal configuration"""
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Import and run
        from main import cli
        cli()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running the full deployment script: python deploy.py")
        return False
    except Exception as e:
        print(f"❌ Error launching Chimera: {e}")
        return False
    
    return True

def main():
    """Main entry point for Chimera Lite"""
    
    print("🔥 Chimera Lite Launcher")
    print("=" * 30)
    
    # Check dependencies
    if not check_and_install_deps():
        sys.exit(1)
    
    # Setup minimal config
    create_minimal_config()
    
    # Launch Chimera
    if not launch_chimera():
        sys.exit(1)

if __name__ == "__main__":
    main()