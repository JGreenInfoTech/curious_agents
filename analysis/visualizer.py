"""
Training Visualizer
===================

Matplotlib-based dashboard for observing agent development.
Renders the world state, agent positions, learning curves,
and metacognitive metrics.

Run standalone or call from training loop.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import json
import os
import torch
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainingVisualizer:
    """
    Real-time(ish) visualization of agent training.
    
    Layout:
    ┌──────────────────┬────────────────┐
    │   World Map       │  Prediction    │
    │   (agent+object   │  Error Curves  │
    │    positions)     │                │
    ├──────────────────┼────────────────┤
    │   Learning        │  Confidence &  │
    │   Progress        │  Meta-Cognition│
    └──────────────────┴────────────────┘
    """
    
    def __init__(self, world_size: float = 100.0, n_agents: int = 3):
        self.world_size = world_size
        self.n_agents = n_agents
        self.agent_colors = plt.cm.Set1(np.linspace(0, 1, max(n_agents, 3)))
        
        # Object category colors
        self.category_colors = {
            'fruit': '#FF6B6B',
            'animal': '#4ECDC4',
            'mineral': '#95A5A6',
            'element': '#F39C12',
            'plant': '#2ECC71',
            'toy': '#3498DB',
            'object': '#9B59B6',
        }
    
    def plot_world(self, ax, env_state: Dict, agent_positions: Dict):
        """Render the 2D world with objects and agents."""
        ax.clear()
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect('equal')
        ax.set_title('World Map', fontsize=12, fontweight='bold')
        ax.set_facecolor('#1a1a2e')
        
        # Grid
        ax.grid(True, alpha=0.1, color='white')
        
        # Draw objects
        if 'objects' in env_state:
            for name, obj in env_state['objects'].items():
                pos = obj['position']
                cat = obj.get('category', 'object')
                color = self.category_colors.get(cat, '#FFFFFF')
                
                ax.plot(pos[0], pos[1], 'o', color=color, markersize=10, 
                       markeredgecolor='white', markeredgewidth=0.5, alpha=0.8)
                ax.annotate(name, (pos[0], pos[1]), 
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=7, color='white', alpha=0.7)
        
        # Draw agents
        for aid, pos in agent_positions.items():
            color = self.agent_colors[int(aid) % len(self.agent_colors)]
            ax.plot(pos[0], pos[1], 's', color=color, markersize=12,
                   markeredgecolor='white', markeredgewidth=1.5)
            ax.annotate(f'A{aid}', (pos[0], pos[1]),
                       textcoords="offset points", xytext=(7, 7),
                       fontsize=9, color=color, fontweight='bold')
            
            # Draw perception radius
            circle = plt.Circle((pos[0], pos[1]), 30, fill=False, 
                              color=color, linestyle='--', alpha=0.2)
            ax.add_patch(circle)
    
    def plot_prediction_errors(self, ax, metrics_history: List[Dict]):
        """Plot prediction error curves over time."""
        ax.clear()
        ax.set_title('Prediction Error', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Prediction Error')
        
        if not metrics_history:
            return
        
        episodes = [m['episode'] for m in metrics_history]
        
        for aid in range(self.n_agents):
            errors = []
            for m in metrics_history:
                agent_data = m['agents'].get(aid, m['agents'].get(str(aid), {}))
                errors.append(agent_data.get('avg_error', 0))
            
            color = self.agent_colors[aid % len(self.agent_colors)]
            ax.plot(episodes, errors, color=color, alpha=0.8, 
                   label=f'Agent {aid}', linewidth=1.5)
        
        ax.legend(fontsize=8, loc='upper right')
        ax.set_yscale('log') if max(e for e in errors if e > 0) > 0.1 else None
        ax.grid(True, alpha=0.3)
    
    def plot_learning_progress(self, ax, metrics_history: List[Dict]):
        """Plot learning progress (reward from curiosity)."""
        ax.clear()
        ax.set_title('Learning Progress', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Learning Progress')
        
        if not metrics_history:
            return
        
        episodes = [m['episode'] for m in metrics_history]
        
        for aid in range(self.n_agents):
            progress = []
            for m in metrics_history:
                agent_data = m['agents'].get(aid, m['agents'].get(str(aid), {}))
                progress.append(agent_data.get('avg_progress', 0))
            
            color = self.agent_colors[aid % len(self.agent_colors)]
            # Smooth with moving average
            window = min(20, len(progress))
            if window > 1:
                smoothed = np.convolve(progress, np.ones(window)/window, mode='valid')
                ax.plot(episodes[:len(smoothed)], smoothed, color=color, 
                       alpha=0.8, label=f'Agent {aid}', linewidth=1.5)
            else:
                ax.plot(episodes, progress, color=color, alpha=0.8,
                       label=f'Agent {aid}', linewidth=1.5)
        
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def plot_confidence(self, ax, metrics_history: List[Dict]):
        """Plot confidence and meta-cognitive state."""
        ax.clear()
        ax.set_title('Confidence (Proto-Metacognition)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, 1)
        
        if not metrics_history:
            return
        
        episodes = [m['episode'] for m in metrics_history]
        
        for aid in range(self.n_agents):
            confidence = []
            for m in metrics_history:
                agent_data = m['agents'].get(aid, m['agents'].get(str(aid), {}))
                confidence.append(agent_data.get('confidence', 0.5))
            
            color = self.agent_colors[aid % len(self.agent_colors)]
            ax.plot(episodes, confidence, color=color, alpha=0.8,
                   label=f'Agent {aid}', linewidth=1.5)
        
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, label='Baseline')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    def render_dashboard(self, env_state: Dict, agent_positions: Dict,
                         metrics_history: List[Dict],
                         save_path: Optional[str] = None):
        """
        Render the full 4-panel dashboard.
        
        Call this periodically during training for visual monitoring.
        """
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#0d1117')
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ax_world = fig.add_subplot(gs[0, 0])
        ax_errors = fig.add_subplot(gs[0, 1])
        ax_progress = fig.add_subplot(gs[1, 0])
        ax_confidence = fig.add_subplot(gs[1, 1])
        
        # Style all axes
        for ax in [ax_world, ax_errors, ax_progress, ax_confidence]:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='white', labelsize=8)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
        
        self.plot_world(ax_world, env_state, agent_positions)
        self.plot_prediction_errors(ax_errors, metrics_history)
        self.plot_learning_progress(ax_progress, metrics_history)
        self.plot_confidence(ax_confidence, metrics_history)
        
        # Suptitle
        episode = metrics_history[-1]['episode'] if metrics_history else 0
        stage = metrics_history[-1]['stage'] if metrics_history else 0
        fig.suptitle(f'Curious Agents — Episode {episode} | Stage {stage}',
                    fontsize=14, fontweight='bold', color='white', y=0.98)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_trajectory(self, position_history: List[Tuple[float, float]],
                        agent_id: int = 0, save_path: Optional[str] = None):
        """Plot an agent's movement trajectory over time."""
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        
        positions = np.array(position_history)
        color = self.agent_colors[agent_id % len(self.agent_colors)]
        
        # Color gradient from start (light) to end (dark)
        n = len(positions)
        for i in range(n - 1):
            alpha = 0.2 + 0.8 * (i / n)
            ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                   color=color, alpha=alpha, linewidth=1)
        
        # Start and end markers
        ax.plot(positions[0, 0], positions[0, 1], 'o', color='lime',
               markersize=10, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 's', color='red',
               markersize=10, label='End')
        
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect('equal')
        ax.set_title(f'Agent {agent_id} Trajectory ({n} steps)',
                    color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.1, color='white')
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            plt.close(fig)
        else:
            plt.show()
    
    def plot_from_checkpoint(self, checkpoint_path: str, 
                             metrics_dir: str = "logs",
                             save_path: Optional[str] = None):
        """
        Load a checkpoint and render the dashboard.
        Useful for post-hoc analysis.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        env_state = checkpoint.get('env_state', {'objects': {}})
        agent_positions = {}
        for aid, data in checkpoint['agents'].items():
            agent_positions[aid] = data['position']
        
        # Load metrics if available
        metrics_history = []
        if os.path.exists(metrics_dir):
            for f in sorted(os.listdir(metrics_dir)):
                if f.endswith('.json'):
                    with open(os.path.join(metrics_dir, f)) as fh:
                        metrics_history.extend(json.load(fh))
        
        self.render_dashboard(env_state, agent_positions, 
                             metrics_history, save_path)


def quick_plot(log_dir: str = "logs"):
    """Quick function to plot metrics from log files."""
    metrics = []
    for f in sorted(os.listdir(log_dir)):
        if f.endswith('.json'):
            with open(os.path.join(log_dir, f)) as fh:
                metrics.extend(json.load(fh))
    
    if not metrics:
        print(f"No metrics found in {log_dir}")
        return
    
    viz = TrainingVisualizer(n_agents=len(metrics[0].get('agents', {})))
    viz.render_dashboard(
        env_state={'objects': {}},
        agent_positions={},
        metrics_history=metrics,
    )
