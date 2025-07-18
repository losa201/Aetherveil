# Aetherveil Sentinel RL Agent System

A comprehensive reinforcement learning agent implementation for cybersecurity scenarios, featuring advanced PPO-based learning, multi-agent coordination, and adaptive capabilities.

## Overview

The Aetherveil Sentinel RL Agent System provides a complete framework for training intelligent agents to perform cybersecurity operations. The system includes:

- **PPO-based RL Agent** with custom cybersecurity-specific policy networks
- **Custom Cybersecurity Environment** simulating realistic network scenarios
- **Sophisticated Action Spaces** for different attack strategies and techniques
- **Multi-objective Reward Functions** for tactical learning
- **Experience Replay and Memory Management** with prioritized replay and episodic memory
- **Curriculum Learning** for progressive skill development
- **Multi-Agent Coordination** with swarm intelligence and task allocation
- **Model Checkpointing and Versioning** with automated validation
- **Real-time Training Monitoring** with visualization and alerting
- **Online Learning and Adaptation** for continuous improvement

## Architecture

### Core Components

```
rl_agent/
├── __init__.py                    # Module initialization
├── rl_agent.py                   # Main PPO-based RL agent
├── cybersecurity_env.py          # Custom cybersecurity environment
├── action_spaces.py              # Attack strategy action spaces
├── reward_functions.py           # Tactical reward functions
├── memory_manager.py             # Experience replay and memory
├── curriculum_learning.py        # Progressive skill development
├── multi_agent_coordinator.py    # Multi-agent coordination
├── model_manager.py              # Model checkpointing and versioning
├── training_monitor.py           # Training monitoring and visualization
├── online_learner.py             # Online learning and adaptation
├── demo.py                       # Comprehensive demonstration
└── README.md                     # This file
```

### Key Features

#### 1. RL Agent (`rl_agent.py`)
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Custom Policy**: Cybersecurity-specific actor-critic architecture
- **Features**: Attention mechanisms, context encoding, adaptive learning rates
- **Integration**: Works with stable-baselines3 framework

#### 2. Cybersecurity Environment (`cybersecurity_env.py`)
- **Network Topologies**: Flat, hierarchical, DMZ, segmented, cloud-hybrid
- **Host Types**: Workstations, servers, databases, domain controllers, etc.
- **Vulnerability System**: Dynamic vulnerability generation and exploitation
- **Objectives**: Configurable mission objectives (compromise, exfiltration, persistence)
- **Detection System**: Realistic intrusion detection with stealth mechanics

#### 3. Action Spaces (`action_spaces.py`)
- **Action Types**: Discrete, continuous, and hybrid action spaces
- **Attack Techniques**: MITRE ATT&CK inspired technique taxonomy
- **Parameterized Actions**: Priority, stealth level, resource cost parameters
- **Validation**: Action masking and validity checking

#### 4. Reward Functions (`reward_functions.py`)
- **Multi-objective**: Discovery, exploitation, stealth, efficiency, strategic value
- **Adaptive Weights**: Curriculum-based weight adjustment
- **Potential-based Shaping**: Prevents reward hacking
- **Component Tracking**: Individual reward component analysis

#### 5. Memory Management (`memory_manager.py`)
- **Experience Replay**: Uniform and prioritized experience replay
- **Episodic Memory**: Complete episode storage and retrieval
- **Strategic Knowledge Base**: Extracted patterns and successful strategies
- **Background Processing**: Asynchronous knowledge extraction

#### 6. Curriculum Learning (`curriculum_learning.py`)
- **Progressive Stages**: From basic reconnaissance to advanced coordination
- **Adaptive Scheduling**: Performance-based progression
- **Task Generation**: Dynamic task creation with difficulty adjustment
- **Skill Assessment**: Comprehensive skill level evaluation

#### 7. Multi-Agent Coordination (`multi_agent_coordinator.py`)
- **Coordination Protocols**: Centralized, decentralized, hierarchical, swarm
- **Agent Roles**: Scout, exploiter, stealth operator, data exfiltrator, etc.
- **Task Allocation**: Capability-based task assignment
- **Swarm Intelligence**: Pheromone-based coordination and collective learning

#### 8. Model Management (`model_manager.py`)
- **Checkpointing**: Automatic and manual checkpoint creation
- **Versioning**: Semantic versioning with metadata tracking
- **Validation**: Performance and security validation pipelines
- **Export/Import**: Model sharing and deployment capabilities

#### 9. Training Monitoring (`training_monitor.py`)
- **Real-time Metrics**: Performance tracking and visualization
- **Alert System**: Configurable alerts for training issues
- **Visualizations**: Training curves, distributions, interactive dashboards
- **Attack Analysis**: Attack chain pattern analysis

#### 10. Online Learning (`online_learner.py`)
- **Adaptation Triggers**: Performance degradation, environment changes
- **Learning Strategies**: Incremental learning, meta-learning, fine-tuning
- **Change Detection**: Environment and performance monitoring
- **Continuous Improvement**: Real-time model adaptation

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify installation:
```python
from rl_agent import RLAgent, CybersecurityEnvironment
print("RL Agent system ready!")
```

### Basic Usage

```python
from rl_agent import RLAgent, CybersecurityEnvironment, TacticalRewardFunction

# Create environment
env = CybersecurityEnvironment(
    network_size=20,
    topology="hierarchical",
    objectives=["compromise_domain_controller"]
)

# Create agent
agent = RLAgent(environment=env, agent_id="test_agent")

# Train agent
training_results = agent.train(total_timesteps=10000)

# Use trained agent
obs, _ = env.reset()
action, _ = agent.predict(obs)
```

### Comprehensive Demo

Run the complete demonstration:

```bash
python demo.py --episodes 100 --multi-agent --analysis
```

This will demonstrate all system components working together.

## Configuration

### Environment Configuration

```python
env_config = {
    "network_size": 25,
    "topology": "cloud_hybrid",
    "vulnerability_density": 0.4,
    "defense_strength": 0.6,
    "detection_threshold": 0.7,
    "episode_length": 150,
    "objectives": ["compromise_domain_controller", "exfiltrate_data"]
}
```

### Agent Configuration

```python
from rl_agent import RLConfig

config = RLConfig(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)
```

### Multi-Agent Configuration

```python
from rl_agent import MultiAgentCoordinator, AgentRole

coordinator = MultiAgentCoordinator(
    coordination_protocol="hierarchical",
    max_agents=10
)

# Register agents
coordinator.register_agent("scout_1", AgentRole.SCOUT, {
    "reconnaissance": 0.9,
    "stealth": 0.7
})
```

## Advanced Features

### Curriculum Learning

The system supports progressive skill development through curriculum learning:

```python
from rl_agent import CurriculumManager, CurriculumStage

curriculum = CurriculumManager(
    initial_stage=CurriculumStage.BASIC_RECONNAISSANCE,
    adaptive_scheduling=True
)

# Get next training task
task, env_config = curriculum.get_next_task()
```

### Online Adaptation

Continuous learning and adaptation to new scenarios:

```python
from rl_agent import OnlineLearner

online_learner = OnlineLearner(
    adaptation_threshold=0.15,
    enable_meta_learning=True
)

# Start adaptation
online_learner.start_adaptation(agent.model.policy)

# System automatically adapts during training
```

### Training Monitoring

Comprehensive monitoring with visualization:

```python
from rl_agent import TrainingMonitor

monitor = TrainingMonitor(
    enable_alerts=True,
    enable_visualizations=True
)

# Add custom alerts
monitor.add_alert_rule(
    "low_performance", 
    "success_rate", 
    "less_than", 
    0.3, 
    "warning"
)
```

## Use Cases

### 1. Penetration Testing Automation
Train agents to perform automated penetration testing:
- Network reconnaissance and discovery
- Vulnerability identification and exploitation
- Lateral movement and privilege escalation
- Data exfiltration and persistence

### 2. Red Team Operations
Develop sophisticated red team capabilities:
- Multi-stage attack campaigns
- Coordinated team operations
- Stealth and evasion techniques
- Adaptive strategy development

### 3. Cybersecurity Training
Create realistic training scenarios:
- Progressive skill development
- Scenario-based training
- Performance assessment
- Knowledge transfer

### 4. Threat Intelligence
Understand attack patterns and techniques:
- Attack pattern analysis
- Threat behavior modeling
- Strategy effectiveness evaluation
- Intelligence extraction

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Success Rate**: Percentage of successful objective completions
- **Average Reward**: Mean reward per episode
- **Episode Length**: Steps required to complete objectives
- **Detection Rate**: Level of detection by defensive systems
- **Efficiency**: Reward per step ratio
- **Stealth Score**: Ability to avoid detection
- **Coordination Effectiveness**: Multi-agent collaboration quality

## Model Export and Deployment

### Export Trained Models

```python
# Export model for deployment
model_manager.export_model(
    "production_model", 
    "v1.2.0", 
    "deployment_package.zip"
)
```

### Load and Deploy

```python
# Load for inference
model, metadata = model_manager.load_checkpoint(
    "production_model", 
    version="v1.2.0"
)

# Deploy in production environment
action, _ = model.predict(observation)
```

## Monitoring and Visualization

The system provides rich monitoring capabilities:

### Real-time Dashboards
- Training progress visualization
- Performance metric tracking
- Attack chain analysis
- Multi-agent coordination status

### Alert System
- Performance degradation alerts
- Training anomaly detection
- Resource utilization monitoring
- Custom alert rule configuration

### Comprehensive Logging
- Structured logging with JSON format
- Episode-level detailed tracking
- Component-specific log levels
- Centralized log aggregation

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Check reward function scaling

2. **Poor Performance**
   - Verify environment configuration
   - Check curriculum progression
   - Review reward function weights

3. **Memory Issues**
   - Reduce replay buffer size
   - Enable model compression
   - Use gradient checkpointing

4. **Coordination Problems**
   - Verify agent registration
   - Check communication channels
   - Review task allocation logic

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

Profile training performance:

```python
import cProfile
cProfile.run('agent.train(total_timesteps=1000)')
```

## Contributing

The RL Agent system is designed to be extensible and customizable:

### Adding New Environments
1. Inherit from `CybersecurityEnvironment`
2. Implement custom observation/action spaces
3. Define environment-specific dynamics

### Custom Reward Functions
1. Inherit from `BaseRewardFunction`
2. Implement `calculate_reward` method
3. Add to `TacticalRewardFunction`

### New Coordination Protocols
1. Extend `MultiAgentCoordinator`
2. Implement protocol-specific logic
3. Add communication patterns

## License

This system is part of the Aetherveil Sentinel project and follows the project's licensing terms.

## Citation

If you use this RL agent system in your research or projects, please cite:

```bibtex
@software{aetherveil_rl_agent,
  title={Aetherveil Sentinel RL Agent System},
  author={Aetherveil Sentinel Team},
  year={2024},
  description={Comprehensive reinforcement learning framework for cybersecurity scenarios}
}
```

## Support

For questions, issues, or contributions:
- Check the troubleshooting section
- Review the demo script for usage examples
- Consult the comprehensive docstrings in each module
- Monitor training logs for diagnostic information

---

The Aetherveil Sentinel RL Agent System represents a comprehensive approach to intelligent cybersecurity automation, combining cutting-edge reinforcement learning techniques with domain-specific expertise to create adaptive, coordinated, and effective security agents.