"""
Training Loop
=============

Orchestrates the agent-environment interaction cycle:
  perceive → decide → act → observe outcome → compute reward → learn

Handles multi-agent coordination, logging, checkpointing, and
curriculum progression.

Phase 2 addition: OstensiveTeacher integration for language grounding.
"""

import numpy as np
import torch
import json
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.world import StructuredEnvironment
from agents.curious_agent import CuriousAgent, create_agent
from training.language_grounding import (
    OstensiveTeacher, TeachingConfig, N_OBJECT_CLASSES, ALL_OBJECT_CLASSES
)


@dataclass
class TrainerConfig:
    """Training hyperparameters."""
    n_agents: int = 3
    n_episodes: int = 5000
    steps_per_episode: int = 100
    
    # Curriculum
    stage_1_episodes: int = 500      # Simple environment
    stage_2_episodes: int = 2000     # More objects
    stage_3_episodes: int = 5000     # Dynamic environment
    
    # Learning
    policy_update_freq: int = 10     # Update policy every N steps
    temperature_start: float = 2.0   # High exploration initially
    temperature_end: float = 0.6     # Lower exploration later (floor raised for utterance viability)
    temperature_decay: float = 0.999 # Per-episode decay
    
    # Helping radius: agents must be within this distance to get helping reward
    helping_radius: float = 40.0

    # Communicative reward: approach-only bonus when an agent's utterance aligns with
    # a nearby agent's discrimination-head prediction (shared understanding signal).
    communicative_reward: float = 0.5  # Raised from 0.2: needs to compete with movement at low temp
    
    # Logging
    log_freq: int = 50               # Log every N episodes
    checkpoint_freq: int = 500       # Save every N episodes
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    # Reproducibility
    seed: int = 42
    
    # World
    world_size: float = 100.0
    perception_radius: float = 30.0


class Trainer:
    """
    Multi-agent training loop.
    
    Each step:
    1. All agents perceive their local environment
    2. All agents decide and execute actions
    3. Environment steps (animate objects move)
    4. All agents compute prediction errors
    5. Ostensive teaching: teacher may "point and name" nearby objects
    6. Naming tests: agents try to name nearby objects
    7. Helping rewards + naming rewards calculated
    8. Forward models trained (self-supervised)
    9. Policies updated periodically (REINFORCE)
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        
        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Create environment
        self.env = StructuredEnvironment(
            world_size=config.world_size,
            seed=config.seed
        )
        
        # Create agents — perception_dim includes N_OBJECT_CLASSES utterance slots (Phase 3)
        perception_dim = self.env.get_perception_dim(n_utterance_classes=N_OBJECT_CLASSES)
        self.agents: List[CuriousAgent] = []
        for i in range(config.n_agents):
            agent = create_agent(
                agent_id=i,
                perception_dim=perception_dim,
                n_utterance_classes=N_OBJECT_CLASSES,
                seed=config.seed,
            )
            self.agents.append(agent)

        # Track utterances from the previous step so agents can perceive them
        # when deciding their NEXT action (one step delay — realistic communication).
        self.prev_utterances: Dict[int, Optional[int]] = {}
        
        # Phase 2: Ostensive teacher for language grounding
        self.teaching_config = TeachingConfig()
        self.teacher = OstensiveTeacher(
            config=self.teaching_config,
            seed=config.seed + 7777,  # Separate RNG stream
        )
        
        # Training state
        self.current_episode = 0
        self.current_stage = 0
        self.temperature = config.temperature_start
        
        # Metrics history
        self.episode_metrics: List[Dict] = []
        
        # Setup directories
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        print(f"Trainer initialized:")
        print(f"  {config.n_agents} agents, {perception_dim}D perception "
              f"(129 env + {N_OBJECT_CLASSES} utterance slots)")
        print(f"  Params per agent: {sum(p.numel() for p in self.agents[0].parameters()):,}")
        print(f"  Total params: {sum(p.numel() for a in self.agents for p in a.parameters()):,}")
        print(f"  Language grounding: ENABLED (Phase 2)")
        print(f"    Teaching probs: stage1={self.teaching_config.stage_1_teach_prob}, "
              f"stage2={self.teaching_config.stage_2_teach_prob}, "
              f"stage3={self.teaching_config.stage_3_teach_prob}")
        print(f"    Min exposures to ground: {self.teaching_config.min_exposures_to_ground}")
        print(f"    Vocab refresh interval: {self.teaching_config.refresh_interval_episodes} episodes")
    
    def setup_curriculum(self, episode: int):
        """Progress through environment stages."""
        if episode == 0:
            self.env.setup_stage_1()
            self.current_stage = 1
            print("\n=== Stage 1: Simple environment (4 objects) ===")
        elif episode == self.config.stage_1_episodes and self.current_stage < 2:
            self.env.setup_stage_2()
            self.current_stage = 2
            print("\n=== Stage 2: Richer environment (8 objects + relations) ===")
        elif episode == self.config.stage_2_episodes and self.current_stage < 3:
            self.env.setup_stage_3()
            self.current_stage = 3
            print("\n=== Stage 3: Dynamic environment (objects move) ===")
        
        # Safety: ensure environment is populated for current stage
        # (guards against checkpoint resume with empty env)
        if not self.env.objects and self.current_stage >= 1:
            if self.current_stage >= 3:
                self.env.setup_stage_3()
            elif self.current_stage >= 2:
                self.env.setup_stage_2()
            else:
                self.env.setup_stage_1()
            print(f"  [Safety] Re-initialized environment for stage {self.current_stage}"
                  f" ({len(self.env.objects)} objects)")
    
    def _build_utterance_slots(self, agent: CuriousAgent,
                               utterances: Dict[int, Optional[int]]) -> np.ndarray:
        """
        Build utterance perception slots for one agent.

        Returns a N_OBJECT_CLASSES-dimensional vector where slot[i] = 1.0 if any
        nearby agent uttered word class i this step, 0.0 otherwise.
        Uses helping_radius as the communication range.
        """
        slots = np.zeros(N_OBJECT_CLASSES)
        for other in self.agents:
            if other.config.agent_id == agent.config.agent_id:
                continue
            dist = np.linalg.norm(agent.position - other.position)
            class_idx = utterances.get(other.config.agent_id)
            if (dist <= self.config.helping_radius
                    and class_idx is not None
                    and 0 <= class_idx < N_OBJECT_CLASSES):
                slots[class_idx] = min(1.0, slots[class_idx] + 1.0)
        return slots

    def get_nearby_agents(self, agent: CuriousAgent) -> List[CuriousAgent]:
        """Find agents within helping radius."""
        nearby = []
        for other in self.agents:
            if other.config.agent_id != agent.config.agent_id:
                dist = np.linalg.norm(agent.position - other.position)
                if dist <= self.config.helping_radius:
                    nearby.append(other)
        return nearby
    
    def run_step(self):
        """Execute one simulation step for all agents."""
        # Store pre-step prediction errors for helping reward
        prev_errors = {a.config.agent_id: a.current_prediction_error for a in self.agents}

        actions = {}
        log_probs = {}

        # --- Phase 1: All agents perceive and decide ---
        # Include PREVIOUS step's utterances so agents know what was recently said
        # when choosing their next action (one-step communication delay).
        for agent in self.agents:
            base_perception = self.env.get_flat_perception(
                position=tuple(agent.position),
                perception_radius=self.config.perception_radius,
            )
            utterance_slots = self._build_utterance_slots(agent, self.prev_utterances)
            perception = np.concatenate([base_perception, utterance_slots])
            agent.perceive(perception)
            action, log_prob = agent.decide_action(temperature=self.temperature)
            actions[agent.config.agent_id] = action
            log_probs[agent.config.agent_id] = log_prob

        # --- Phase 2: All agents execute actions ---
        for agent in self.agents:
            agent.execute_action(actions[agent.config.agent_id])

        # Capture THIS step's utterances (set by execute_action above)
        step_utterances: Dict[int, Optional[int]] = {
            a.config.agent_id: a.last_utterance_class for a in self.agents
        }

        # --- Phase 3: Environment steps ---
        self.env.step()

        # --- Phase 4: All agents perceive new state and compute prediction error ---
        # Include THIS step's utterances so agents observe simultaneous communication.
        for agent in self.agents:
            base_perception = self.env.get_flat_perception(
                position=tuple(agent.position),
                perception_radius=self.config.perception_radius,
            )
            utterance_slots = self._build_utterance_slots(agent, step_utterances)
            new_perception = np.concatenate([base_perception, utterance_slots])
            agent.perceive(new_perception)
            agent.compute_prediction_error(actions[agent.config.agent_id])

        # --- Phase 5: Ostensive teaching (language grounding) ---
        # Teacher may "point and name" nearby objects for each agent
        self.teacher.teach_step(
            self.agents, self.env, self.current_stage, self.current_episode
        )

        # --- Phase 6: Naming tests (passive) ---
        self.teacher.test_naming(
            self.agents, self.env, self.current_stage, self.current_episode
        )

        # --- Phase 6.5: Communicative reward ---
        # Agent A said word W → reward A if a nearby agent B's discrimination head
        # prediction agrees with W's class. This is an approach-only shared-understanding
        # signal: A gets rewarded when its utterance is consistent with B's internal state.
        step_comm_rewards: Dict[int, float] = {}
        for agent in self.agents:
            class_idx = step_utterances.get(agent.config.agent_id)
            if class_idx is None or not (0 <= class_idx < N_OBJECT_CLASSES):
                continue
            comm_reward = 0.0
            for other in self.get_nearby_agents(agent):
                if other.internal_state is None:
                    continue
                with torch.no_grad():
                    logits = other.discrimination_head(other.internal_state)
                    other_pred = torch.argmax(logits, dim=-1).item()
                if other_pred == class_idx:
                    comm_reward += self.config.communicative_reward
            if comm_reward > 0.0:
                step_comm_rewards[agent.config.agent_id] = comm_reward

        # --- Phase 7: Compute rewards (curiosity + helping + naming + communicative) ---
        for agent in self.agents:
            nearby = self.get_nearby_agents(agent)

            if nearby:
                others_prev = [prev_errors[o.config.agent_id] for o in nearby]
                others_curr = [o.current_prediction_error for o in nearby]
            else:
                others_prev = None
                others_curr = None

            reward = agent.compute_intrinsic_reward(
                others_prev_errors=others_prev,
                others_curr_errors=others_curr,
            )

            # Naming reward (approach-only: 0.0 if can't name, positive if correct)
            naming_reward = self.teacher.compute_naming_reward(agent, self.env)
            # Communicative reward (approach-only: only fires when shared understanding confirmed)
            comm_reward = step_comm_rewards.get(agent.config.agent_id, 0.0)
            reward += naming_reward + comm_reward

            # Store experience
            agent.store_experience(
                log_prob=log_probs[agent.config.agent_id],
                reward=reward,
                state=agent.internal_state,
                action=actions[agent.config.agent_id],
            )

        # --- Phase 8: Train forward models (every step) ---
        for agent in self.agents:
            agent.train_forward_model(actions[agent.config.agent_id])

        # Update prev_utterances for next step (agents remember what was just said)
        self.prev_utterances = step_utterances
    
    def run_episode(self, episode: int):
        """Run one full episode."""
        self.setup_curriculum(episode)

        # Randomize agent positions each episode to prevent positional ruts
        for agent in self.agents:
            agent.reset_position(seed=episode * 100 + agent.config.agent_id)

        # Clear utterance memory at episode boundary — no cross-episode communication
        self.prev_utterances = {}
        
        for step in range(self.config.steps_per_episode):
            self.run_step()
        
        # Update policies (less frequent than forward model)
        if episode % self.config.policy_update_freq == 0:
            for agent in self.agents:
                agent.update_policy()
        
        # Refresh vocabularies periodically (encoder weights evolve)
        if (episode > 0 
            and episode % self.teaching_config.refresh_interval_episodes == 0):
            self.teacher.refresh_vocabularies(self.agents, self.env, episode)
        
        # Decay temperature
        self.temperature = max(
            self.config.temperature_end,
            self.temperature * self.config.temperature_decay
        )
    
    def collect_metrics(self) -> Dict:
        """Gather current metrics from all agents."""
        metrics = {
            'episode': self.current_episode,
            'stage': self.current_stage,
            'temperature': self.temperature,
            'agents': {}
        }
        
        for agent in self.agents:
            report = agent.metacognitive_report()
            metrics['agents'][agent.config.agent_id] = {
                'position': tuple(agent.position),
                'confidence': report['confidence'],
                'avg_error': report['avg_recent_error'],
                'avg_progress': report['avg_recent_progress'],
                'total_steps': report['total_steps'],
                'vocab_size': report['vocabulary_size'],
                'is_learning': report['is_learning'],
                'is_exploring': report['is_exploring'],
                'total_reward': agent.total_reward,
                'avg_naming_loss': report['avg_naming_loss'],
                'avg_discrimination_loss': report['avg_discrimination_loss'],
            }
        
        # Language grounding metrics
        lang_metrics = self.teacher.get_metrics()
        metrics['language'] = lang_metrics
        
        per_agent_lang = self.teacher.get_per_agent_metrics(self.agents)
        for aid, lang_data in per_agent_lang.items():
            if aid in metrics['agents']:
                metrics['agents'][aid]['words_known'] = lang_data['words_known']
                metrics['agents'][aid]['naming_accuracy'] = lang_data['recent_accuracy']
        
        return metrics
    
    def log_metrics(self, metrics: Dict):
        """Print and save metrics."""
        ep = metrics['episode']
        stage = metrics['stage']
        temp = metrics['temperature']
        
        print(f"\n--- Episode {ep} | Stage {stage} | Temp {temp:.3f} ---")
        for aid, data in metrics['agents'].items():
            learning_str = "LEARNING" if data['is_learning'] else "plateau"
            exploring_str = "exploring" if data['is_exploring'] else "settled"
            words = data.get('words_known', [])
            naming_acc = data.get('naming_accuracy', 0.0)
            print(f"  Agent {aid}: conf={data['confidence']:.3f} "
                  f"err={data['avg_error']:.4f} "
                  f"prog={data['avg_progress']:.4f} "
                  f"reward={data['total_reward']:.2f} "
                  f"[{learning_str}, {exploring_str}]"
                  f"  pos=({data['position'][0]:.0f},{data['position'][1]:.0f})")
            if words:
                print(f"         vocab({len(words)}): {words}  "
                      f"naming_acc={naming_acc:.2f}")
        
        # Language summary
        lang = metrics.get('language', {})
        if lang.get('total_naming_tests', 0) > 0:
            print(f"  Language: {lang['total_teaching_events']} teach events, "
                  f"{lang['total_naming_tests']} tests, "
                  f"accuracy={lang['naming_accuracy']:.2f} "
                  f"(recent={lang['recent_accuracy']:.2f})")
    
    def save_checkpoint(self, episode: int):
        """Save agent states, teacher state, and training metrics."""
        checkpoint = {
            'episode': episode,
            'stage': self.current_stage,
            'temperature': self.temperature,
            'agents': {},
            'env_state': self.env.get_state(),
        }
        
        for agent in self.agents:
            aid = agent.config.agent_id
            checkpoint['agents'][aid] = {
                'state_dict': agent.state_dict(),
                'position': agent.position.tolist(),
                'confidence': agent.confidence,
                'total_steps': agent.total_steps,
                'total_reward': agent.total_reward,
                'vocabulary': {k: v.tolist() for k, v in agent.vocabulary.items()},
                'prediction_error_history': agent.prediction_error_history[-100:],
                'learning_progress_history': agent.learning_progress_history[-100:],
                'confidence_history': agent.confidence_history[-100:],
                'naming_loss_history': agent.naming_loss_history[-100:],
                'discrimination_loss_history': agent.discrimination_loss_history[-100:],
            }
        
        # Save teacher state (word memories, test history)
        teacher_state = {
            'word_memories': {},
            'teaching_events': self.teacher.teaching_events[-200:],
            'naming_tests': self.teacher.naming_tests[-200:],
        }
        for aid, memories in self.teacher.word_memories.items():
            teacher_state['word_memories'][aid] = {}
            for word, mem in memories.items():
                teacher_state['word_memories'][aid][word] = {
                    'word': mem.word,
                    'base_name': mem.base_name,
                    'buffer': [s.tolist() for s in mem.buffer],
                    'write_pos': mem.write_pos,
                    'exposures': mem.exposures,
                    'grounded': mem.grounded,
                    'last_refresh_episode': mem.last_refresh_episode,
                }
        checkpoint['teacher'] = teacher_state
        
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_ep{episode}.pt")
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
        
        # Also save metrics as JSON for easy analysis
        metrics_path = os.path.join(self.config.log_dir, f"metrics_ep{episode}.json")
        # Convert non-serializable types
        metrics_json = []
        for m in self.episode_metrics[-self.config.log_freq:]:
            m_copy = dict(m)
            m_copy['agents'] = {str(k): v for k, v in m_copy['agents'].items()}
            metrics_json.append(m_copy)
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_, np.integer)):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2, cls=NumpyEncoder)
    
    def load_checkpoint(self, path: str):
        """Resume from a checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        
        self.current_episode = checkpoint['episode']
        self.current_stage = checkpoint['stage']
        # Respect the configured floor: if we raised temperature_end after a run, the
        # resumed temperature should climb back up to the new floor, not stay at the old one.
        self.temperature = max(checkpoint['temperature'], self.config.temperature_end)
        
        for agent in self.agents:
            aid = agent.config.agent_id
            if aid in checkpoint['agents']:
                data = checkpoint['agents'][aid]
                # strict=False: handles checkpoints saved before the discrimination_head
                # was added (missing keys get default init). A shape mismatch (e.g.
                # Phase 2→3 encoder dim change 129→139) triggers a RuntimeError which
                # we catch and fall back to structured initialization for that agent.
                try:
                    agent.load_state_dict(data['state_dict'], strict=False)
                except RuntimeError:
                    from agents.curious_agent import apply_structured_initialization
                    print(f"  Warning: checkpoint architecture incompatible for agent {aid} "
                          f"(perception_dim may have changed 129->139 for Phase 3). "
                          f"Re-applying structured initialization.")
                    apply_structured_initialization(agent, seed=self.config.seed + aid * 1000)
                agent.position = np.array(data['position'])
                agent.confidence = data['confidence']
                agent.total_steps = data['total_steps']
                agent.total_reward = data['total_reward']
                agent.vocabulary = {
                    k: np.array(v) for k, v in data['vocabulary'].items()
                }
                agent.naming_loss_history = data.get('naming_loss_history', [])
                agent.discrimination_loss_history = data.get('discrimination_loss_history', [])

        # Restore teacher state if available
        if 'teacher' in checkpoint:
            ts = checkpoint['teacher']
            self.teacher.teaching_events = ts.get('teaching_events', [])
            self.teacher.naming_tests = ts.get('naming_tests', [])

            from training.language_grounding import WordMemory
            for aid_str, memories in ts.get('word_memories', {}).items():
                aid = int(aid_str) if isinstance(aid_str, str) else aid_str
                self.teacher.word_memories[aid] = {}
                for word, mem_data in memories.items():
                    mem = WordMemory(
                        word=mem_data['word'],
                        base_name=mem_data['base_name'],
                        buffer_size=self.teaching_config.vocab_buffer_size,
                    )
                    mem.exposures = mem_data['exposures']
                    mem.grounded = mem_data['grounded']
                    mem.last_refresh_episode = mem_data['last_refresh_episode']

                    # Backward-compatible: new checkpoints have 'buffer',
                    # old ones have 'state_accumulator' (cumulative sum).
                    if 'buffer' in mem_data:
                        mem.buffer = [np.array(s) for s in mem_data['buffer']]
                        mem.write_pos = mem_data.get('write_pos', 0)
                    elif (mem_data.get('state_accumulator') is not None
                          and mem.exposures > 0):
                        # Reconstruct a single-entry buffer from the old average
                        avg = np.array(mem_data['state_accumulator']) / mem.exposures
                        mem.buffer = [avg]
                        mem.write_pos = 1 % mem.buffer_size

                    self.teacher.word_memories[aid][word] = mem
        
        # Restore environment for the current stage
        # (env.__init__ starts with empty objects dict, so we must re-populate)
        if self.current_stage >= 3:
            self.env.setup_stage_3()
        elif self.current_stage >= 2:
            self.env.setup_stage_2()
        elif self.current_stage >= 1:
            self.env.setup_stage_1()
        
        print(f"Resumed from episode {self.current_episode}, stage {self.current_stage}")
        print(f"  Environment restored: {len(self.env.objects)} objects")
        
        # Report vocabulary state
        for agent in self.agents:
            if agent.vocabulary:
                print(f"  Agent {agent.config.agent_id} vocabulary: "
                      f"{list(agent.vocabulary.keys())}")
    
    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        This is where you watch agents develop curiosity, explore,
        and learn to communicate through grounded language.
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        start_ep = self.current_episode
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Starting training: episodes {start_ep} -> {self.config.n_episodes}")
        print(f"{'='*60}\n")
        
        for episode in range(start_ep, self.config.n_episodes):
            self.current_episode = episode
            self.run_episode(episode)
            
            # Collect and store metrics
            metrics = self.collect_metrics()
            self.episode_metrics.append(metrics)
            
            # Periodic logging
            if episode % self.config.log_freq == 0:
                self.log_metrics(metrics)
                elapsed = time.time() - start_time
                eps_per_sec = (episode - start_ep + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{elapsed:.0f}s elapsed, {eps_per_sec:.1f} ep/s]")
            
            # Periodic checkpointing
            if episode % self.config.checkpoint_freq == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        # Final checkpoint
        self.save_checkpoint(self.config.n_episodes)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete: {self.config.n_episodes} episodes in {elapsed:.0f}s")
        print(f"{'='*60}")
        
        return self.episode_metrics
