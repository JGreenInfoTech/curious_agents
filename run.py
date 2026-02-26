"""
Curious Agents — Main Entry Point
==================================

Run this to start training curious agents.

Usage:
    python run.py                    # Default training (3 agents, 5000 episodes)
    python run.py --episodes 1000    # Quick test run
    python run.py --agents 5         # More agents
    python run.py --resume checkpoints/checkpoint_ep500.pt  # Resume from checkpoint
    python run.py --visualize        # Enable periodic visualization
    python run.py --test             # Quick smoke test (50 episodes)
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer, TrainerConfig
from analysis.visualizer import TrainingVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train curious agents')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--agents', type=int, default=3,
                        help='Number of agents')
    parser.add_argument('--steps', type=int, default=100,
                        help='Steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable periodic dashboard visualization')
    parser.add_argument('--viz-freq', type=int, default=200,
                        help='Visualization frequency (episodes)')
    parser.add_argument('--test', action='store_true',
                        help='Quick smoke test (50 episodes)')
    parser.add_argument('--log-dir', type=str, default='logs_phase6',
                        help='Directory for log files (default: logs_phase6)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_phase6',
                        help='Directory for checkpoints (default: checkpoints_phase6)')
    return parser.parse_args()


def smoke_test():
    """Quick test to verify everything works."""
    print("=" * 60)
    print("SMOKE TEST — verifying all components work")
    print("=" * 60)
    
    config = TrainerConfig(
        n_agents=2,
        n_episodes=50,
        steps_per_episode=20,
        stage_1_episodes=20,
        stage_2_episodes=40,
        log_freq=10,
        checkpoint_freq=25,
        log_dir='logs_test',
        checkpoint_dir='checkpoints_test',
    )
    
    trainer = Trainer(config)
    metrics = trainer.train()
    
    # Verify agents learned something
    for agent in trainer.agents:
        report = agent.metacognitive_report()
        print(f"\nAgent {agent.config.agent_id} final state:")
        print(f"  Steps: {report['total_steps']}")
        print(f"  Confidence: {report['confidence']:.4f}")
        print(f"  Avg error: {report['avg_recent_error']:.4f}")
        print(f"  Learning: {report['is_learning']}")
        print(f"  Exploring: {report['is_exploring']}")
    
    print("\n[OK] Smoke test passed -- all components functional")
    return True


def main():
    args = parse_args()
    
    if args.test:
        smoke_test()
        return
    
    # Configure training
    config = TrainerConfig(
        n_agents=args.agents,
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Optional: set up visualization
    viz = None
    if args.visualize:
        viz = TrainingVisualizer(
            world_size=config.world_size,
            n_agents=config.n_agents,
        )
        os.makedirs('viz', exist_ok=True)
    
    # Train
    print(f"\nConfiguration:")
    print(f"  Agents: {config.n_agents}")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Steps/episode: {config.steps_per_episode}")
    print(f"  Seed: {config.seed}")
    print(f"  Log dir: {config.log_dir}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print(f"  Visualization: {'ON' if args.visualize else 'OFF'}")
    
    # Hook visualization into training loop if enabled
    if viz:
        _original_log = trainer.log_metrics
        def log_with_viz(metrics):
            _original_log(metrics)
            ep = metrics['episode']
            if ep % args.viz_freq == 0 and ep > 0:
                agent_positions = {
                    aid: data['position'] 
                    for aid, data in metrics['agents'].items()
                }
                viz.render_dashboard(
                    env_state=trainer.env.get_state(),
                    agent_positions=agent_positions,
                    metrics_history=trainer.episode_metrics,
                    save_path=f'viz/dashboard_ep{ep}.png',
                )
                print(f"  Dashboard saved: viz/dashboard_ep{ep}.png")
        trainer.log_metrics = log_with_viz
    
    metrics = trainer.train(resume_from=args.resume)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL AGENT STATES")
    print("=" * 60)
    for agent in trainer.agents:
        report = agent.metacognitive_report()
        print(f"\nAgent {agent.config.agent_id} ({agent.config.name}):")
        print(f"  Position: ({agent.position[0]:.1f}, {agent.position[1]:.1f})")
        print(f"  Confidence: {report['confidence']:.4f}")
        print(f"  Avg prediction error: {report['avg_recent_error']:.6f}")
        print(f"  Avg learning progress: {report['avg_recent_progress']:.6f}")
        print(f"  Total reward: {agent.total_reward:.4f}")
        print(f"  Vocabulary: {report['vocabulary_size']} words")
        print(f"  Status: {'LEARNING' if report['is_learning'] else 'plateau'}, "
              f"{'exploring' if report['is_exploring'] else 'settled'}")
    
    # Generate final visualization
    if viz:
        agent_positions = {
            a.config.agent_id: tuple(a.position) for a in trainer.agents
        }
        viz.render_dashboard(
            env_state=trainer.env.get_state(),
            agent_positions=agent_positions,
            metrics_history=trainer.episode_metrics,
            save_path='viz/final_dashboard.png',
        )
        
        # Plot trajectories
        for agent in trainer.agents:
            if agent.position_history:
                viz.plot_trajectory(
                    agent.position_history,
                    agent_id=agent.config.agent_id,
                    save_path=f'viz/trajectory_agent{agent.config.agent_id}.png',
                )
        
        print("\nVisualizations saved to viz/")


if __name__ == '__main__':
    main()
