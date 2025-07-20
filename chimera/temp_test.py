import sys
sys.path.insert(0, '.')

try:
    from chimera.utils.config import ConfigManager
    print("✅ Configuration system")
    
except Exception as e:
    print(f"❌ Import test failed: {e}")
    sys.exit(1)