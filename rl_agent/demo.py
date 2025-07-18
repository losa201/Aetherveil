"""
Comprehensive RL Agent Demo

This script demonstrates the complete RL agent system including all components:
- PPO-based RL Agent
- Custom Cybersecurity Environment
- Action Spaces and Reward Functions
- Memory Management and Experience Replay
- Curriculum Learning
- Multi-Agent Coordination
- Model Checkpointing and Versioning
- Training Monitoring and Visualization
- Online Learning and Adaptation
"""

import numpy as np
import torch
import logging
import time
from pathlib import Path
import argparse
import json

# Import all RL agent components
from rl_agent import RLAgent, RLConfig
from cybersecurity_env import CybersecurityEnvironment, NetworkTopology
from action_spaces import AttackActionSpace, AgentRole
from reward_functions import TacticalRewardFunction
from memory_manager import ExperienceReplayManager
from curriculum_learning import CurriculumManager, CurriculumStage
from multi_agent_coordinator import MultiAgentCoordinator, CoordinationProtocol
from model_manager import ModelCheckpointManager, CheckpointType
from training_monitor import TrainingMonitor
from online_learner import OnlineLearner


def setup_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rl_demo.log')
        ]
    )


def create_cybersecurity_environment():
    """Create and configure cybersecurity environment"""
    print("Creating cybersecurity environment...")
    
    env = CybersecurityEnvironment(
        network_size=20,
        topology=NetworkTopology.HIERARCHICAL,
        vulnerability_density=0.4,
        defense_strength=0.6,
        episode_length=100,
        objectives=["compromise_domain_controller", "exfiltrate_data"]
    )
    
    print(f"Environment created with {env.network_size} hosts")
    return env


def create_rl_agent(env):
    """Create and configure RL agent"""
    print("Creating RL agent...")
    
    config = RLConfig(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    agent = RLAgent(
        environment=env,
        config=config,
        agent_id="demo_agent",
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
    )
    
    print("RL agent created with PPO algorithm")
    return agent


def setup_action_spaces():
    """Setup action spaces for attack strategies"""
    print("Setting up action spaces...")
    
    action_space = AttackActionSpace(
        action_type="hierarchical",
        enable_hierarchical=True,
        enable_parameterized=True,
        max_targets=50
    )
    
    print(f"Action space configured with {len(action_space.valid_combinations)} valid combinations")
    return action_space


def setup_reward_functions():
    """Setup tactical reward functions"""
    print("Setting up reward functions...")
    
    reward_function = TacticalRewardFunction(
        curriculum_stage="intermediate",
        adaptive_weights=True
    )
    
    print(f"Reward function created with {len(reward_function.reward_functions)} components")
    return reward_function


def setup_memory_manager():
    """Setup experience replay and memory management"""
    print("Setting up memory management...")
    
    memory_manager = ExperienceReplayManager(
        replay_buffer_type="prioritized",
        buffer_capacity=50000,
        episodic_capacity=500,
        save_dir="./memory"
    )
    
    print("Memory management system initialized")
    return memory_manager


def setup_curriculum_learning():
    """Setup curriculum learning"""
    print("Setting up curriculum learning...")
    
    curriculum_manager = CurriculumManager(
        initial_stage=CurriculumStage.BASIC_RECONNAISSANCE,
        adaptive_scheduling=True,
        save_dir="./curriculum"
    )
    
    print(f"Curriculum learning initialized at stage: {curriculum_manager.current_stage.value}")
    return curriculum_manager


def setup_multi_agent_coordination():
    """Setup multi-agent coordination"""
    print("Setting up multi-agent coordination...")
    
    coordinator = MultiAgentCoordinator(
        coordination_protocol=CoordinationProtocol.HIERARCHICAL,
        max_agents=5
    )
    
    # Register sample agents
    coordinator.register_agent("scout_1", AgentRole.SCOUT, {"reconnaissance": 0.9, "stealth": 0.7})
    coordinator.register_agent("exploiter_1", AgentRole.EXPLOITER, {"exploitation": 0.8, "lateral_movement": 0.6})
    coordinator.register_agent("stealth_1", AgentRole.STEALTH_OPERATOR, {"stealth": 0.9, "evasion": 0.8})
    
    print(f"Multi-agent coordinator initialized with {len(coordinator.agent_capabilities)} agents")
    return coordinator


def setup_model_management():
    """Setup model checkpointing and versioning"""
    print("Setting up model management...")
    
    model_manager = ModelCheckpointManager(
        base_directory="./models",
        max_checkpoints=20,
        auto_cleanup=True,
        validation_enabled=True
    )
    
    print("Model checkpoint manager initialized")
    return model_manager


def setup_training_monitor():
    """Setup training monitoring and visualization"""
    print("Setting up training monitoring...")
    
    monitor = TrainingMonitor(
        log_dir="./training_logs",
        enable_alerts=True,
        enable_visualizations=True
    )
    
    # Add some alert rules
    monitor.add_alert_rule("low_success_rate", "success_rate", "less_than", 0.3, "warning")
    monitor.add_alert_rule("high_detection", "detection_rate", "greater_than", 0.8, "critical")
    
    print("Training monitor initialized with alert system")
    return monitor


def setup_online_learning():
    """Setup online learning and adaptation"""
    print("Setting up online learning...")
    
    online_learner = OnlineLearner(
        adaptation_threshold=0.15,
        enable_meta_learning=True
    )
    
    print("Online learner initialized with meta-learning")
    return online_learner


def run_training_demo(
    agent, env, reward_function, memory_manager, 
    curriculum_manager, model_manager, monitor, online_learner,
    num_episodes=100
):
    """Run comprehensive training demonstration"""
    print(f"\nStarting training demonstration for {num_episodes} episodes...")
    
    # Start monitoring session
    session_id = monitor.start_session("demo_session", {"num_episodes": num_episodes})
    
    # Start online adaptation
    online_learner.start_adaptation(agent.model.policy)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Get curriculum task
        task, env_config = curriculum_manager.get_next_task()
        
        # Reset environment with curriculum configuration
        obs, info = env.reset()
        
        # Start episode in memory manager
        episode_id = memory_manager.start_episode()
        
        episode_reward = 0
        episode_length = 0
        detection_level = 0
        attack_chain = []
        done = False
        
        while not done:
            # Get action from agent
            action, _ = agent.predict(obs, deterministic=False)
            
            # Execute action
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Calculate tactical reward
            state_dict = {"agent_position": env.agent_position, "detection_level": env.detection_level}
            next_state_dict = {"agent_position": env.agent_position, "detection_level": env.detection_level}
            action_dict = {"type": step_info.get("action", "unknown")}
            
            tactical_reward, reward_components = reward_function.calculate_reward(
                state_dict, action_dict, next_state_dict, step_info
            )
            
            # Add experience to memory
            memory_manager.add_experience(
                obs, action, tactical_reward, next_obs, done, step_info
            )
            
            # Add experience to online learner
            online_learner.add_experience(obs, action, tactical_reward, next_obs, done, step_info)
            
            # Update for next step
            obs = next_obs
            episode_reward += tactical_reward
            episode_length += 1
            detection_level = max(detection_level, env.detection_level)
            
            # Record attack step
            if step_info.get("action"):
                attack_chain.append({
                    "action_type": step_info.get("action"),
                    "success": step_info.get("success", False),
                    "target": step_info.get("target"),
                    "step": episode_length
                })
                
        # End episode
        success = len(env.objectives_completed) > 0
        
        # Record in memory manager
        memory_manager.end_episode(
            success=success,
            objectives_completed=list(env.objectives_completed),
            network_topology=env.topology.value
        )
        
        # Record in curriculum manager
        curriculum_manager.record_episode_result({
            "success": success,
            "total_reward": episode_reward,
            "episode_length": episode_length,
            "detection_level": detection_level,
            "objectives_completed": list(env.objectives_completed)
        })
        
        # Log to training monitor
        monitor.log_episode_completion(
            episode=episode,
            total_reward=episode_reward,
            episode_length=episode_length,
            success=success,
            detection_level=detection_level,
            attack_chain=attack_chain,
            additional_metrics={
                "tactical_reward": tactical_reward,
                "curriculum_stage": curriculum_manager.current_stage.value
            }
        )
        
        # Update online learner
        performance_metrics = {
            "success_rate": float(success),
            "average_reward": episode_reward,
            "efficiency": episode_reward / max(episode_length, 1),
            "detection_avoidance": 1.0 - detection_level
        }
        online_learner.update_performance(performance_metrics)
        online_learner.update_environment({"network_topology": env.topology.value, "network_size": env.network_size})
        
        episode_rewards.append(episode_reward)
        
        # Periodic model saving
        if episode % 20 == 0 and episode > 0:
            model_manager.save_checkpoint(
                agent.model,
                "demo_model",
                CheckpointType.AUTOMATIC,
                performance_metrics={
                    "training_episodes": episode,
                    "total_reward": np.mean(episode_rewards[-20:]),
                    "success_rate": np.mean([1 if r > 5 else 0 for r in episode_rewards[-20:]]),
                    "avg_episode_length": episode_length
                }
            )
            
        # Progress update
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                  f"Success = {success}, Stage = {curriculum_manager.current_stage.value}")
            
    # End training session
    monitor.end_session()
    online_learner.stop_adaptation()
    
    print(f"\nTraining completed! Average reward: {np.mean(episode_rewards):.2f}")
    return episode_rewards


def run_multi_agent_demo(coordinator, env, num_tasks=10):
    """Demonstrate multi-agent coordination"""
    print(f"\nRunning multi-agent coordination demo with {num_tasks} tasks...")
    
    # Submit coordination tasks
    task_ids = []
    for i in range(num_tasks):
        task_id = coordinator.submit_coordination_task(
            task_type="reconnaissance" if i % 2 == 0 else "exploitation",
            target=f"host_{i % 5}",
            required_agents=2,
            priority=np.random.randint(1, 4)
        )
        task_ids.append(task_id)
        
    # Run coordination cycles
    for cycle in range(5):
        print(f"Coordination cycle {cycle + 1}")
        
        # Coordinate agents
        results = coordinator.coordinate_agents()
        print(f"  Tasks allocated: {len(results['allocated_tasks'])}")
        print(f"  Messages sent: {results['messages_sent']}")
        print(f"  Effectiveness: {results['coordination_effectiveness']:.2f}")
        
        # Simulate task completion
        for agent_id, tasks in results['allocated_tasks'].items():
            for task in tasks:
                success = np.random.random() > 0.3  # 70% success rate
                coordinator.report_task_completion(
                    agent_id, task.task_id, success, 
                    {"reward": np.random.uniform(1, 5) if success else 0}
                )
                
        time.sleep(1)  # Simulate time passing
        
    status = coordinator.get_coordination_status()
    print(f"Coordination demo completed. Final effectiveness: {status['coordination_effectiveness']:.2f}")


def run_analysis_demo(memory_manager, monitor, model_manager):
    """Demonstrate analysis and visualization capabilities"""
    print("\nRunning analysis and visualization demo...")
    
    # Memory analysis
    print("Memory Analysis:")
    memory_stats = memory_manager.get_statistics()
    print(f"  Total experiences: {memory_stats['total_experiences']}")
    print(f"  Total episodes: {memory_stats['total_episodes']}")
    print(f"  Replay buffer utilization: {memory_stats['replay_buffer']['utilization']:.2f}")
    
    # Get strategic advice
    advice = memory_manager.get_strategic_advice(
        {"network_topology": "hierarchical"},
        {"current_objectives": ["compromise_domain_controller"]}
    )
    print(f"  Strategic recommendations: {len(advice['recommendations'])}")
    
    # Training monitor analysis
    print("\nTraining Monitor Analysis:")
    train_stats = monitor.get_current_statistics()
    if train_stats["session"]:
        print(f"  Session duration: {train_stats['session']['duration']}")
        print(f"  Episodes completed: {train_stats['session']['episodes']}")
        
    # Model management analysis
    print("\nModel Management Analysis:")
    model_stats = model_manager.get_model_statistics()
    print(f"  Total models: {model_stats['total_models']}")
    print(f"  Total checkpoints: {model_stats['total_checkpoints']}")
    print(f"  Storage usage: {model_stats['storage_usage'] / (1024*1024):.1f} MB")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="RL Agent Comprehensive Demo")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--multi-agent", action="store_true", help="Run multi-agent demo")
    parser.add_argument("--analysis", action="store_true", help="Run analysis demo")
    parser.add_argument("--save-dir", type=str, default="./demo_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Aetherveil Sentinel RL Agent Comprehensive Demo")
    print("=" * 60)
    
    try:
        # Initialize all components
        env = create_cybersecurity_environment()
        agent = create_rl_agent(env)
        action_space = setup_action_spaces()
        reward_function = setup_reward_functions()
        memory_manager = setup_memory_manager()
        curriculum_manager = setup_curriculum_learning()
        coordinator = setup_multi_agent_coordination()
        model_manager = setup_model_management()
        monitor = setup_training_monitor()
        online_learner = setup_online_learning()
        
        print(f"\nAll components initialized successfully!")
        print(f"Results will be saved to: {save_dir}")
        
        # Run training demo
        episode_rewards = run_training_demo(
            agent, env, reward_function, memory_manager,
            curriculum_manager, model_manager, monitor, online_learner,
            num_episodes=args.episodes
        )
        
        # Optional demos
        if args.multi_agent:
            run_multi_agent_demo(coordinator, env)
            
        if args.analysis:
            run_analysis_demo(memory_manager, monitor, model_manager)
            
        # Save final results
        results = {
            "episode_rewards": episode_rewards,
            "final_stats": {
                "memory": memory_manager.get_statistics(),
                "curriculum": curriculum_manager.get_curriculum_status(),
                "coordination": coordinator.get_coordination_status(),
                "model_management": model_manager.get_model_statistics(),
                "training_monitor": monitor.get_current_statistics(),
                "online_learning": online_learner.get_adaptation_status()
            }
        }
        
        with open(save_dir / "demo_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\nDemo completed successfully!")
        print(f"Final average reward: {np.mean(episode_rewards):.2f}")
        print(f"Results saved to: {save_dir / 'demo_results.json'}")
        
        # Cleanup
        monitor.cleanup()
        online_learner.cleanup()
        coordinator.shutdown()
        
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        raise
        
    finally:
        print("\nDemo cleanup completed.")


if __name__ == "__main__":
    main()