# 🚀 Chimera Quick Start Guide

## ⚡ Rapid Deployment (2 Minutes)

### Method 1: Automated Deployment
```bash
cd /root/Aetherveil/chimera
python3 deploy.py
```

The deployment script will:
- ✅ Check Python version (3.8+ required)
- ✅ Create virtual environment  
- ✅ Install platform-optimized dependencies
- ✅ Create data directories
- ✅ Setup configuration files
- ✅ Test basic functionality

### Method 2: Manual Setup
```bash
# Create environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install minimal dependencies
pip install aiohttp aiofiles pyyaml click requests beautifulsoup4 psutil

# Create directories
mkdir -p data/{knowledge,reports,logs,cache}

# Setup config
cp .env.example .env
```

## 🎯 First Run

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Test Installation
```bash
python main.py --help
```

### 3. Interactive Mode
```bash
python main.py interactive
```

### 4. Basic Commands
```bash
# In interactive mode:
chimera> status                    # Check system status
chimera> persona balanced          # Set persona
chimera> learn web security        # Learn about topic
chimera> help                      # Show all commands
```

## 🎭 Choose Your Persona

- **`cautious`** - Maximum stealth, minimal risk (recommended for production)
- **`balanced`** - Good balance of speed and stealth (recommended for learning)
- **`aggressive`** - Fast and thorough (for internal testing only)
- **`creative`** - Innovative approaches (for research)
- **`stealth_focused`** - Maximum OPSEC (for sensitive targets)

## 📋 Sample Bug Bounty Campaign

### 1. Setup Target Scope
```bash
# Edit configs/scopes/example_scope.json with your target
{
  "target": "your-target.com",
  "domains": ["your-target.com", "*.your-target.com"],
  "rules": {
    "max_requests_per_minute": 30,
    "allowed_techniques": ["scanning", "enumeration"]
  }
}
```

### 2. Run Campaign
```bash
python main.py campaign your-target.com \
  --scope configs/scopes/example_scope.json \
  --persona balanced
```

### 3. View Results
Reports are generated in `data/reports/` as Markdown files.

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Core settings
CHIMERA_DEBUG=false
CHIMERA_LOG_LEVEL=INFO

# Proxy settings (optional)
HTTP_PROXY=http://your-proxy:8080
SOCKS_PROXY=socks5://127.0.0.1:9050

# API keys (optional - uses web interfaces if not provided)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### Main Configuration (config.yaml)
Key settings you might want to adjust:
```yaml
core:
  max_concurrent_tasks: 4      # Reduce for low-memory systems
  
web:
  headless: true              # Set to false for debugging
  scraping_delay_range: [2, 8] # Increase for more stealth
  
opsec:
  stealth_mode: true          # Always keep enabled
  fingerprint_randomization: true
```

## 📱 Platform-Specific Notes

### ARM64/Termux
- Optimized for mobile platforms
- Some browser features may be limited
- Uses lightweight implementations
- Monitor memory with `status` command

### Ubuntu/Debian
```bash
# Install system dependencies if needed
sudo apt update
sudo apt install python3-venv python3-pip chromium-browser
```

### Performance Tuning
```bash
# For low-memory systems, edit config.yaml:
core:
  max_concurrent_tasks: 2
memory:
  max_nodes: 1000
  pruning_threshold: 0.2
```

## 🚨 Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements-minimal.txt
```

### Browser Issues
```bash
# Install browser manually
# Ubuntu: sudo apt install chromium-browser
# Mac: brew install chromium
# Check browser path in config
```

### Memory Issues
```bash
# Edit config.yaml to reduce limits:
memory:
  max_nodes: 1000
core:
  max_concurrent_tasks: 2
```

### Permission Errors
```bash
# Fix data directory permissions
chmod -R 755 data/
```

## 🎓 Learning Path

### Beginner
1. Start with `interactive` mode
2. Use `balanced` persona
3. Try `learn` commands for different topics
4. Use `status` to understand the system

### Intermediate  
1. Run test campaigns against practice targets
2. Experiment with different personas
3. Analyze generated reports
4. Customize configuration

### Advanced
1. Develop custom personas
2. Integrate with CI/CD pipelines  
3. Extend knowledge base
4. Contribute new modules

## 🔒 Security Best Practices

1. **Always get authorization** before testing targets
2. **Use appropriate personas** for each situation
3. **Monitor OPSEC metrics** in reports
4. **Review scope boundaries** carefully
5. **Keep system updated** regularly

## 📞 Support

- **Issues**: Check logs in `data/logs/chimera.log`
- **Documentation**: See full README.md
- **Community**: GitHub Discussions
- **Security**: Report via security@your-org.com

---

**🔥 You're ready to unleash Chimera! Start with `python main.py interactive` and let the neuroplastic evolution begin.**