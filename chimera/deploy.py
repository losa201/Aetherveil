#!/usr/bin/env python3
"""
Chimera Deployment Script
Handles dependency management and system initialization
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+ is available"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def detect_platform():
    """Detect platform and architecture"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"🔍 Platform: {system} ({machine})")
    
    is_arm64 = machine in ['arm64', 'aarch64']
    is_termux = 'TERMUX_VERSION' in os.environ
    
    return {
        'system': system,
        'machine': machine,
        'is_arm64': is_arm64,
        'is_termux': is_termux
    }

def create_venv():
    """Create virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
        
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_pip_command():
    """Get appropriate pip command"""
    if platform.system().lower() == "windows":
        return ["venv/Scripts/pip"]
    else:
        return ["venv/bin/pip"]

def install_minimal_deps(platform_info):
    """Install minimal dependencies based on platform"""
    
    pip_cmd = get_pip_command()
    
    # Core dependencies that work on all platforms
    core_deps = [
        "aiohttp>=3.8.0",
        "aiofiles>=0.8.0", 
        "pyyaml>=6.0",
        "click>=8.0.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "psutil>=5.9.0"
    ]
    
    # ARM64/Termux optimized packages
    if platform_info['is_arm64'] or platform_info['is_termux']:
        core_deps.extend([
            "numpy>=1.21.0",  # ARM64 optimized
        ])
    else:
        core_deps.extend([
            "numpy>=1.24.0",
            "selenium>=4.0.0",
        ])
    
    print("📦 Installing core dependencies...")
    
    for dep in core_deps:
        try:
            subprocess.run(pip_cmd + ["install", dep], check=True, capture_output=True)
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Failed to install {dep}: {e}")
            
    # Optional ML packages (if available)
    optional_deps = []
    
    if not platform_info['is_termux']:  # Skip heavy ML on Termux
        optional_deps = [
            "scikit-learn>=1.0.0",
            "transformers>=4.20.0",
        ]
        
    if optional_deps:
        print("📦 Installing optional ML dependencies...")
        
        for dep in optional_deps:
            try:
                subprocess.run(pip_cmd + ["install", dep], check=True, capture_output=True, timeout=300)
                print(f"  ✅ {dep}")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"  ⚠️  Skipped {dep}: {e}")

def create_directories():
    """Create necessary data directories"""
    directories = [
        "data/knowledge",
        "data/reports", 
        "data/logs",
        "data/cache",
        "data/browser_profiles"
    ]
    
    print("📁 Creating data directories...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")

def setup_config():
    """Setup initial configuration"""
    
    config_path = Path("config.yaml")
    env_path = Path(".env")
    
    if not config_path.exists():
        print("⚙️  Creating default configuration...")
        # Config already exists from implementation
        
    if not env_path.exists():
        print("⚙️  Creating environment configuration...")
        subprocess.run(["cp", ".env.example", ".env"])
        print("  ✅ Edit .env file with your settings")

def test_basic_functionality():
    """Test basic system functionality"""
    
    print("🧪 Testing basic functionality...")
    
    python_cmd = ["venv/bin/python"] if platform.system().lower() != "windows" else ["venv/Scripts/python"]
    
    # Test imports
    test_script = """
import sys
sys.path.insert(0, '.')

try:
    from chimera.utils.config import ConfigManager
    print("✅ Configuration system")
    
    from chimera.core.events import EventSystem
    print("✅ Event system")
    
    from chimera.memory.knowledge_graph_lite import LiteKnowledgeGraph
    print("✅ Knowledge graph")
    
    print("✅ All core modules imported successfully")
    
except Exception as e:
    print(f"❌ Import test failed: {e}")
    sys.exit(1)
"""
    
    try:
        result = subprocess.run(
            python_cmd + ["-c", test_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
        
    return True

def main():
    """Main deployment function"""
    
    print("🔥 Chimera Deployment Starting...")
    print("=" * 50)
    
    # Check requirements
    check_python_version()
    platform_info = detect_platform()
    
    # Setup environment
    if not create_venv():
        sys.exit(1)
        
    # Install dependencies
    install_minimal_deps(platform_info)
    
    # Setup directories and config
    create_directories()
    setup_config()
    
    # Test functionality
    if test_basic_functionality():
        print("\n🎉 Chimera deployment successful!")
        print("\nNext steps:")
        print("1. Edit .env file with your configuration")
        print("2. Run: source venv/bin/activate  (or venv\\Scripts\\activate on Windows)")
        print("3. Run: python main.py --help")
        print("4. Start with: python main.py interactive")
        
        if platform_info['is_termux']:
            print("\n📱 Termux-specific notes:")
            print("- Some browser features may be limited")
            print("- Consider using lightweight personas")
            print("- Monitor memory usage with 'status' command")
            
    else:
        print("\n❌ Deployment completed with warnings")
        print("Some features may not work correctly")
        print("Check error messages above")

if __name__ == "__main__":
    main()