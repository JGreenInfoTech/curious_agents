"""
Training Loop
=============

Orchestrates the agent-environment interaction cycle:
  perceive → decide → act → observe outcome → compute reward → learn

Handles multi-agent coordination, logging, checkpointing, and
curriculum progression.

Phase 2 addition: OstensiveTeacher integration for language grounding.
"""

import random
import numpy as np
import torch
import json
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.world import StructuredEnvironment, WorldObject
from agents.curious_agent import CuriousAgent, create_agent
from training.language_grounding import (
    OstensiveTeacher, TeachingConfig, N_OBJECT_CLASSES, ALL_OBJECT_CLASSES, WORD_CLASS_MAP,
    N_PROPERTY_CLASSES, ALL_PROPERTY_CLASSES, PROPERTY_WORD_MAP, PROPERTY_DIM_MAP,
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

    property_comm_reward: float = 0.3    # When property utterance matches partner's prediction

    # Phase 4: spatial memory + communication rewards
    referral_reward: float = 0.4         # Reward when agent's utterance helps another discover object
    joint_curiosity_bonus: float = 0.15  # Bonus when two curious agents explore together
    directed_discovery_prob: float = 0.20  # Probability of directed discovery setup per episode

    # Phase 5: property-consequential rewards
    property_approach_penalty: float = -0.2   # per-step penalty near dangerous objects
    property_food_bonus: float = 0.05         # per-step bonus near edible objects
    danger_radius: float = 8.0
    food_radius: float = 8.0

    # Phase 6: food events
    p_event: float = 0.30          # probability a food event fires each episode
    event_duration: int = 20        # steps the event is active
    event_food_bonus: float = 0.5   # per-step bonus during event (replaces baseline 0.05)

    # Stage 4: Reference game
    ref_game_prob: float = 0.35          # Fraction of Stage 4 episodes that are ref games
    ref_game_steps: int = 35             # Steps per ref game episode (vs 100 for normal)
    ref_game_radius: float = 8.0         # Distance threshold: runner "reached" target
    ref_game_success_reward: float = 2.0 # Terminal reward for correct variant
    ref_game_wrong_penalty: float = -1.0 # Terminal reward for wrong variant

    # Logging
    log_freq: int = 50               # Log every N episodes
    checkpoint_freq: int = 500       # Save every N episodes
    log_dir: str = "logs_phase6"
    checkpoint_dir: str = "checkpoints_phase6"
    
    # Reproducibility
    seed: int = 42
    
    # World
    world_size: float = 100.0
    perception_radius: float = 15.0


class Trainer:
    """
    Multi-agent training loop.

    Each step:
    1. All agents perceive their local environment (encoder → GRU → h_t)
    2. All agents decide and execute actions
    3. Environment steps (animate objects move)
    4. All agents compute prediction errors (||predicted_h_{t+1} - actual_h_{t+1}||²)
    5. Ostensive teaching: teacher may "point and name" nearby objects
    6. Naming tests: agents try to name nearby objects
    7. Helping rewards + naming rewards calculated
    8. Forward models trained (self-supervised)
    9. Policies updated periodically (REINFORCE with sequential GRU replay)

    Phase 6: Hidden state management.
    The Trainer holds a per-agent hidden state tensor that persists across
    steps within an episode. At episode start, hidden states are reset to zeros.
    During collection, hidden states are detached before each step so gradients
    don't accumulate through time. During update_policy(), the agent re-runs
    the full trajectory with fresh gradients (BPTT through GRU).
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        
        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Create environment
        self.env = StructuredEnvironment(
            world_size=config.world_size,
            seed=config.seed
        )
        
        # Create agents — perception_dim includes utterance slots (Phase 3) + memory dims (Phase 4)
        # + property utterance slots (Phase 3.5) + goal class token (Stage 4)
        perception_dim = self.env.get_perception_dim(
            n_utterance_classes=N_OBJECT_CLASSES,
            n_memory_classes=N_OBJECT_CLASSES,
            n_property_utterance_classes=N_PROPERTY_CLASSES,
            n_goal_classes=N_OBJECT_CLASSES,
        )
        self.agents: List[CuriousAgent] = []
        for i in range(config.n_agents):
            agent = create_agent(
                agent_id=i,
                perception_dim=perception_dim,
                n_utterance_classes=N_OBJECT_CLASSES,
                seed=config.seed,
            )
            self.agents.append(agent)

        # Phase 6: Per-agent GRU hidden states, managed across steps within an episode.
        # Detached at each collection step; reset to zeros at episode start.
        # Key: agent_id → Tensor of shape (1, internal_state_dim)
        self.hidden_states: Dict[int, torch.Tensor] = {
            i: self.agents[i].reset_hidden() for i in range(config.n_agents)
        }

        # Stage 4: Per-agent goal class vector for reference game runner.
        # 10D one-hot indicating target class. Zeros for scout, free agent,
        # and all agents during normal (non-reference-game) episodes.
        self._goal_class_vecs: Dict[int, np.ndarray] = {
            i: np.zeros(N_OBJECT_CLASSES, dtype=np.float32)
            for i in range(config.n_agents)
        }

        # Stage 4: Reference game state for current episode (for metrics logging).
        self._ref_game_state: Dict = {'active': False}

        # Track utterances from the previous step so agents can perceive them
        # when deciding their NEXT action (one step delay — realistic communication).
        # Format after Phase 3.5: agent_id -> {'class': int|None, 'property': int|None}
        self.prev_utterances: Dict[int, dict] = {}

        # Phase 4: utterance log for referral reward, episode step counter,
        # pre-step salience snapshot for novelty detection
        self.utterance_log: List[Dict] = []
        self.episode_step: int = 0
        self._pre_step_salience: Dict[int, Dict[int, float]] = {}
        # Episode-level reward accumulators for metrics logging
        self.episode_referral_totals: Dict[int, float] = {}
        self.episode_joint_totals: Dict[int, float] = {}
        self.episode_property_comm_totals: Dict[int, float] = {}
        self.episode_property_approach_totals: Dict[int, float] = {}

        # Phase 6: food event state — reset each episode in run_episode()
        self.active_event_object: Optional[str] = None   # dict key of active event object
        self.event_start_step: int = 0
        self.event_end_step: int = 0
        self.episode_event_agent_arrivals: Set[int] = set()  # distinct agent IDs that reached the event object
        self.episode_event_active: bool = False               # whether a food event fired this episode
        
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
              f"(129 env + {N_OBJECT_CLASSES} obj slots + {N_PROPERTY_CLASSES} prop slots "
              f"+ {N_OBJECT_CLASSES} memory dims + {N_OBJECT_CLASSES} goal token)")
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
            self.env.spawn_objects()  # Phase 5: replace with stable 12-object world
            self.current_stage = 1
            print("\n=== Stage 1: Simple environment (4 objects) ===")
        elif episode == self.config.stage_1_episodes and self.current_stage < 2:
            self.env.setup_stage_2()
            self.env.spawn_objects()  # Phase 5: replace with stable 12-object world
            self.current_stage = 2
            print("\n=== Stage 2: Richer environment (8 objects + relations) ===")
        elif episode == self.config.stage_2_episodes and self.current_stage < 3:
            self.env.setup_stage_3()
            self.env.spawn_objects()  # Phase 5: replace with stable 12-object world (no dynamic mode)
            self.current_stage = 3
            print("\n=== Stage 3: Dynamic environment (objects move) ===")
        elif episode == self.config.stage_3_episodes and self.current_stage < 4:
            self.env.setup_stage_4()
            self.current_stage = 4
            print("\n=== Stage 4: Reference game (grammatical communication) ===")

        # Safety: ensure environment is populated for current stage
        # (guards against checkpoint resume with empty env)
        if not self.env.objects and self.current_stage >= 1:
            if self.current_stage >= 4:
                self.env.setup_stage_3()
                self.env.setup_stage_4()
            elif self.current_stage >= 3:
                self.env.setup_stage_3()
            elif self.current_stage >= 2:
                self.env.setup_stage_2()
            else:
                self.env.setup_stage_1()
            self.env.spawn_objects()  # Phase 5: replace with stable 12-object world
            print(f"  [Safety] Re-initialized environment for stage {self.current_stage}"
                  f" ({len(self.env.objects)} objects)")
    
    def _build_utterance_slots(self, agent: CuriousAgent,
                               utterances: Dict[int, dict]) -> np.ndarray:
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
            other_entry = utterances.get(other.config.agent_id)
            # Support both old int format and new dict format
            if isinstance(other_entry, dict):
                class_idx = other_entry.get('class')
            else:
                class_idx = other_entry
            if (dist <= self.config.helping_radius
                    and class_idx is not None
                    and 0 <= class_idx < N_OBJECT_CLASSES):
                slots[class_idx] = min(1.0, slots[class_idx] + 1.0)
        return slots

    def _build_property_utterance_slots(self, agent: CuriousAgent,
                                        utterances: dict) -> np.ndarray:
        """5D slot: index i is set to 1.0 if any nearby agent uttered property i this step."""
        slot = np.zeros(N_PROPERTY_CLASSES, dtype=np.float32)
        for other in self.agents:
            if other.config.agent_id == agent.config.agent_id:
                continue
            prop_idx = utterances.get(other.config.agent_id, {}).get('property')
            if prop_idx is not None:
                dist = np.linalg.norm(agent.position - other.position)
                if dist <= self.config.helping_radius:
                    if 0 <= prop_idx < N_PROPERTY_CLASSES:
                        slot[prop_idx] = 1.0
        return slot

    def get_nearby_agents(self, agent: CuriousAgent) -> List[CuriousAgent]:
        """Find agents within helping radius."""
        nearby = []
        for other in self.agents:
            if other.config.agent_id != agent.config.agent_id:
                dist = np.linalg.norm(agent.position - other.position)
                if dist <= self.config.helping_radius:
                    nearby.append(other)
        return nearby

    def _build_perception(self, agent: CuriousAgent, utterances: Dict) -> np.ndarray:
        """Build full perception vector: env base + utterance slots + property slots + memory dims."""
        base = self.env.get_flat_perception(
            position=tuple(agent.position),
            perception_radius=self.config.perception_radius,
        )
        utterance_slots = self._build_utterance_slots(agent, utterances)
        property_slots = self._build_property_utterance_slots(agent, utterances)
        memory_vec = agent.spatial_memory.to_vec(
            current_episode=self.current_episode,
            n_classes=N_OBJECT_CLASSES,
            decay_horizon=agent.config.memory_decay_horizon,
        )
        goal_vec = self._goal_class_vecs.get(
            agent.config.agent_id,
            np.zeros(N_OBJECT_CLASSES, dtype=np.float32)
        )
        return np.concatenate([base, utterance_slots, property_slots, memory_vec, goal_vec])

    def _get_nearby_object_classes(self, agent: CuriousAgent) -> List[Tuple[int, np.ndarray]]:
        """Return [(class_idx, position), ...] for objects within perception radius."""
        result = []
        for key, obj in self.env.objects.items():
            base = key.split('_')[0] if '_' in key and key.split('_')[-1].isdigit() else key
            class_idx = WORD_CLASS_MAP.get(base, -1)
            if class_idx < 0:
                continue
            dx = abs(obj.position[0] - agent.position[0])
            dy = abs(obj.position[1] - agent.position[1])
            dx = min(dx, self.env.world_size - dx)
            dy = min(dy, self.env.world_size - dy)
            if np.sqrt(dx**2 + dy**2) <= self.config.perception_radius:
                result.append((class_idx, np.array(obj.position)))
        return result

    def _reset_agent_positions(self, episode: int):
        """Space agents evenly around world center for maximum information asymmetry."""
        rng = np.random.RandomState(episode * 137)
        n = len(self.agents)
        for i, agent in enumerate(self.agents):
            angle = (2 * np.pi * i / n) + rng.uniform(-0.3, 0.3)
            radius = 35.0 + rng.uniform(-5.0, 5.0)
            x = (50.0 + radius * np.cos(angle)) % self.env.world_size
            y = (50.0 + radius * np.sin(angle)) % self.env.world_size
            agent.position = np.array([x, y])
            agent.visit_counts.clear()
            agent.position_history = [tuple(agent.position.copy())]

    def _setup_directed_discovery(self, episode: int):
        """20% chance: move one agent near an object its partner lacks memory for."""
        rng = np.random.RandomState(episode * 31)
        if rng.rand() > self.config.directed_discovery_prob:
            return
        for class_idx in range(N_OBJECT_CLASSES):
            saliences = sorted(
                [(a, a.spatial_memory.get_decayed_salience(class_idx, episode))
                 for a in self.agents],
                key=lambda x: x[1], reverse=True
            )
            knower, know_sal = saliences[0]
            learner, learn_sal = saliences[-1]
            if know_sal < 0.3 or learn_sal > 0.1:
                continue
            word = ALL_OBJECT_CLASSES[class_idx]
            for key, obj in self.env.objects.items():
                base = key.split('_')[0] if '_' in key and key.split('_')[-1].isdigit() else key
                if base == word:
                    offset = rng.uniform(-4.0, 4.0, 2)
                    knower.position = (np.array(obj.position) + offset) % self.env.world_size
                    knower.position_history = [tuple(knower.position.copy())]
                    return

    def _place_ref_game_agents(self, scout_id: int, runner_id: int,
                                target_obj: WorldObject, episode: int) -> None:
        """
        Place agents for a reference game episode.

        Geometry (toroidal):
          TARGET <---10u---> SCOUT <---12u---> RUNNER

        - Scout: 10 units from target -> within perception radius (15), sees target
        - Runner: 12 units from scout on far side -> within hearing range of scout (12<15),
                  but 22 units from target (22>15, blind to target)
        - Free agents: random position anywhere in world
        """
        rng = np.random.RandomState(episode * 53)
        target_pos = np.array(target_obj.position, dtype=float)

        # Rejection sampling: try angles until runner is outside perception range of target.
        # Needed because toroidal wrapping can create a short path from runner to target
        # even when the raw Euclidean displacement is 22u (scout 10u + runner 12u away).
        for _ in range(50):
            angle = rng.uniform(0, 2 * np.pi)
            scout_offset = np.array([np.cos(angle), np.sin(angle)]) * 10.0
            scout_pos = (target_pos + scout_offset) % self.env.world_size

            away_angle = angle + np.pi
            runner_offset = np.array([np.cos(away_angle), np.sin(away_angle)]) * 12.0
            runner_pos = (scout_pos + runner_offset) % self.env.world_size

            if self.env.toroidal_distance(tuple(runner_pos), tuple(target_pos)) > self.config.perception_radius:
                break
        # If all 50 attempts fail (extremely unlikely on a 100x100 world), use last attempt.
        # Runner may be closer than ideal but game still runs.

        for agent in self.agents:
            aid = agent.config.agent_id
            if aid == scout_id:
                agent.position = scout_pos.copy()
            elif aid == runner_id:
                agent.position = runner_pos.copy()
            else:
                agent.position = rng.uniform(0, self.env.world_size, 2)
            agent.visit_counts.clear()
            agent.position_history = [tuple(agent.position.copy())]


    def run_reference_game_episode(self, episode: int) -> None:
        """
        35-step reference game episode.

        Scout is placed near the target object (sees it).
        Runner has a goal class token and must navigate to the correct variant.
        Terminal reward fires at episode end: +2.0 correct, -1.0 wrong variant, 0 timeout.
        """
        # --- Role assignment ---
        aid_list = [a.config.agent_id for a in self.agents]
        scout_id, runner_id = random.sample(aid_list, 2)

        # --- Target selection ---
        target_key = random.choice(list(self.env.objects.keys()))
        target_obj = self.env.objects[target_key]

        # Derive base name (strip numeric suffix for variants like apple_2, cat_2)
        parts = target_key.split("_")
        base_name = parts[0] if (len(parts) > 1 and parts[-1].isdigit()) else target_key
        target_class_idx = WORD_CLASS_MAP.get(base_name, -1)

        # Ambiguous: another object shares the same base name
        same_class_keys = [
            k for k in self.env.objects
            if k != target_key
            and (k.split("_")[0] if (len(k.split("_")) > 1
                 and k.split("_")[-1].isdigit()) else k) == base_name
        ]
        target_is_ambiguous = len(same_class_keys) > 0

        # Determine the correct property word for disambiguation (for observability only)
        correct_property_idx = None
        if target_is_ambiguous and same_class_keys:
            canonical_obj = self.env.objects[same_class_keys[0]]
            for prop_word, prop_dim in PROPERTY_DIM_MAP.items():
                if prop_word not in ALL_PROPERTY_CLASSES:
                    continue
                prop_idx = ALL_PROPERTY_CLASSES.index(prop_word)
                if (target_obj.properties[prop_dim] > 0.5
                        and canonical_obj.properties[prop_dim] <= 0.5):
                    correct_property_idx = prop_idx
                    break

        # --- Place agents ---
        self._place_ref_game_agents(scout_id, runner_id, target_obj, episode)

        # --- Set goal class vec for runner (zeros for all others) ---
        for aid in aid_list:
            self._goal_class_vecs[aid] = np.zeros(N_OBJECT_CLASSES, dtype=np.float32)
        if 0 <= target_class_idx < N_OBJECT_CLASSES:
            self._goal_class_vecs[runner_id][target_class_idx] = 1.0

        # --- Initialize tracking state ---
        self._ref_game_state = {
            "active": True,
            "scout_id": scout_id,
            "runner_id": runner_id,
            "target_key": target_key,
            "target_class": base_name,
            "target_is_ambiguous": target_is_ambiguous,
            "outcome": "timeout",
            "runner_min_distance": float("inf"),
            "scout_words_emitted": [],
            "scout_used_property": False,
            "scout_used_correct_property": False,
        }

        # --- Run steps ---
        runner_agent = next(a for a in self.agents if a.config.agent_id == runner_id)
        for _ in range(self.config.ref_game_steps):
            self.run_step()
            dist = self.env.toroidal_distance(
                tuple(runner_agent.position), target_obj.position
            )
            if dist < self._ref_game_state["runner_min_distance"]:
                self._ref_game_state["runner_min_distance"] = dist

        # --- Collect scout utterances from utterance_log ---
        scout_obj_words = []
        scout_prop_words = []
        for entry in self.utterance_log:
            if entry["agent_id"] != scout_id:
                continue
            if entry.get("type") == "object":
                idx = entry.get("class_idx")
                if idx is not None and 0 <= idx < N_OBJECT_CLASSES:
                    scout_obj_words.append(ALL_OBJECT_CLASSES[idx])
            elif entry.get("type") == "property":
                idx = entry.get("property_idx")
                if idx is not None and 0 <= idx < N_PROPERTY_CLASSES:
                    scout_prop_words.append(ALL_PROPERTY_CLASSES[idx])

        self._ref_game_state["scout_words_emitted"] = scout_obj_words + scout_prop_words
        self._ref_game_state["scout_used_property"] = len(scout_prop_words) > 0
        self._ref_game_state["scout_used_correct_property"] = (
            correct_property_idx is not None
            and ALL_PROPERTY_CLASSES[correct_property_idx] in scout_prop_words
        )

        # --- Determine outcome based on runner's final position ---
        dist_correct = self.env.toroidal_distance(
            tuple(runner_agent.position), target_obj.position
        )
        dist_wrong = float("inf")
        for wrong_key in same_class_keys:
            if wrong_key in self.env.objects:
                d = self.env.toroidal_distance(
                    tuple(runner_agent.position),
                    self.env.objects[wrong_key].position
                )
                dist_wrong = min(dist_wrong, d)

        if dist_correct <= self.config.ref_game_radius:
            outcome = "correct"
            terminal_reward = self.config.ref_game_success_reward
        elif dist_wrong <= self.config.ref_game_radius:
            outcome = "wrong_variant"
            terminal_reward = self.config.ref_game_wrong_penalty
        else:
            outcome = "timeout"
            terminal_reward = 0.0

        self._ref_game_state["outcome"] = outcome

        # --- Apply terminal reward to scout and runner's last experience entry ---
        if terminal_reward != 0.0:
            scout_agent = next(a for a in self.agents if a.config.agent_id == scout_id)
            for target_agent in (runner_agent, scout_agent):
                if target_agent.experience_buffer:
                    target_agent.experience_buffer[-1]['reward'] += terminal_reward
                target_agent.total_reward += terminal_reward

        # --- Policy update ---
        if episode % self.config.policy_update_freq == 0:
            for agent in self.agents:
                agent.update_policy(steps_per_episode=self.config.ref_game_steps)

    def run_step(self):
        """Execute one simulation step for all agents."""
        # Store pre-step prediction errors for helping reward
        prev_errors = {a.config.agent_id: a.current_prediction_error for a in self.agents}

        actions = {}
        log_probs = {}
        # Track perceptions this step so we can store them in experience_buffer
        step_perceptions: Dict[int, np.ndarray] = {}

        # --- Phase 1: All agents perceive and decide ---
        # Include PREVIOUS step's utterances so agents know what was recently said
        # when choosing their next action (one-step communication delay).
        # Phase 6: pass hidden state in (detached — no gradient accumulation across steps).
        for agent in self.agents:
            aid = agent.config.agent_id
            perception = self._build_perception(agent, self.prev_utterances)
            step_perceptions[aid] = perception

            # Detach hidden state before passing in — gradients are recomputed during
            # update_policy() via full trajectory replay, not accumulated here.
            h_prev = self.hidden_states[aid].detach()
            h_t = agent.perceive(perception, hidden_state=h_prev)

            # Store new hidden state (detached for next step's collection)
            self.hidden_states[aid] = h_t.detach()

            action, log_prob = agent.decide_action(temperature=self.temperature)
            actions[aid] = action
            log_probs[aid] = log_prob

        # --- Phase 2: All agents execute actions ---
        for agent in self.agents:
            agent.execute_action(actions[agent.config.agent_id])

        # Capture THIS step's utterances (set by execute_action above).
        # Dict format: agent_id -> {'class': int|None, 'property': int|None}
        step_utterances: Dict[int, dict] = {
            a.config.agent_id: {
                'class': a.last_utterance_class,
                'property': a.last_utterance_property,
            }
            for a in self.agents
        }

        # --- Phase 2.5: Snapshot pre-step salience + log utterances ---
        self._pre_step_salience = {
            a.config.agent_id: {
                c: a.spatial_memory.get_decayed_salience(c, self.current_episode)
                for c in range(N_OBJECT_CLASSES)
            }
            for a in self.agents
        }
        for agent in self.agents:
            if agent.last_utterance_class is not None:
                self.utterance_log.append({
                    'agent_id': agent.config.agent_id,
                    'class_idx': agent.last_utterance_class,
                    'step': self.episode_step,
                    'type': 'object',
                })
            if agent.last_utterance_property is not None:
                self.utterance_log.append({
                    'agent_id': agent.config.agent_id,
                    'class_idx': None,
                    'property_idx': agent.last_utterance_property,
                    'step': self.episode_step,
                    'type': 'property',
                })
        self.episode_step += 1

        # --- Phase 3: Environment steps ---
        self.env.step()

        # --- Phase 4: All agents perceive new state and compute prediction error ---
        # Include THIS step's utterances so agents observe simultaneous communication.
        # Phase 6: this second perceive() call within a step is for computing the
        # prediction error only — it reads the GRU output (h_t) but does NOT carry
        # that hidden state forward to the next step. Only the pre-action hidden state
        # transition (Phase 1 above) advances self.hidden_states[aid], keeping
        # collection consistent with update_policy() replay (one GRU step per timestep).
        for agent in self.agents:
            aid = agent.config.agent_id
            new_perception = self._build_perception(agent, step_utterances)
            # Pass the hidden state already stored for this agent (the post-Phase-1 value).
            # Do NOT reassign self.hidden_states[aid] from this result — the next step's
            # Phase 1 perceive() will pick up from the Phase-1 h_t already stored above.
            agent.perceive(new_perception, hidden_state=self.hidden_states[aid].detach())
            agent.compute_prediction_error(actions[aid])

        # --- Phase 4.5: Update spatial memory ---
        for agent in self.agents:
            nearby_classes = self._get_nearby_object_classes(agent)
            agent.update_memory(nearby_classes, agent.current_prediction_error,
                                self.current_episode)

        # --- Phase 5: Ostensive teaching (language grounding) ---
        # Teacher may "point and name" nearby objects for each agent
        self.teacher.teach_step(
            self.agents, self.env, self.current_stage, self.current_episode
        )
        self.teacher.teach_property_step(self.agents, self.env,
                                          self.current_stage, self.current_episode)

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
            class_idx = step_utterances.get(agent.config.agent_id, {}).get('class')
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

        # Property communicative reward
        for agent_a in self.agents:
            prop_idx = step_utterances.get(agent_a.config.agent_id, {}).get('property')
            if prop_idx is None:
                continue
            for agent_b in self.agents:
                if agent_b.config.agent_id == agent_a.config.agent_id:
                    continue
                dist = np.linalg.norm(agent_a.position - agent_b.position)
                if dist > self.config.helping_radius:
                    continue
                if agent_b.internal_state is None:
                    continue
                with torch.no_grad():
                    prop_pred = agent_b.property_discrimination_head(agent_b.internal_state)
                if prop_pred[0, prop_idx].item() > 0.5:
                    aid = agent_a.config.agent_id
                    step_comm_rewards[aid] = step_comm_rewards.get(aid, 0.0) + self.config.property_comm_reward
                    self.episode_property_comm_totals[aid] = self.episode_property_comm_totals.get(aid, 0.0) + self.config.property_comm_reward

        # --- Phase 6.6: Referral reward ---
        # Agent A said word W recently; agent B just found class W it didn't know about.
        referral_rewards: Dict[int, float] = {}
        for agent_b in self.agents:
            err_b = agent_b.current_prediction_error
            if err_b < 0.05:
                continue  # Not a novel find
            nearby_classes = self._get_nearby_object_classes(agent_b)
            for class_idx, _ in nearby_classes:
                prior_sal = self._pre_step_salience.get(
                    agent_b.config.agent_id, {}).get(class_idx, 0.0)
                if prior_sal >= 0.2:
                    continue  # B already knew about this
                # Find a recent utterance of this class by a nearby agent
                for entry in reversed(self.utterance_log):
                    if self.episode_step - entry['step'] > 20:
                        break
                    # Skip property utterance log entries (class_idx is None for those)
                    if entry.get('class_idx') != class_idx:
                        continue
                    if entry['agent_id'] == agent_b.config.agent_id:
                        continue
                    referrer = next(
                        (a for a in self.agents if a.config.agent_id == entry['agent_id']),
                        None
                    )
                    if referrer is None:
                        continue
                    dist = np.linalg.norm(agent_b.position - referrer.position)
                    if dist <= self.config.helping_radius:
                        aid = entry['agent_id']
                        referral_rewards[aid] = (
                            referral_rewards.get(aid, 0.0) + self.config.referral_reward
                        )
                        break

        # --- Phase 6.7: Joint curiosity bonus ---
        # Two agents exploring together in a novel region both get a bonus.
        joint_rewards: Dict[int, float] = {}
        for i, agent_a in enumerate(self.agents):
            for agent_b in self.agents[i+1:]:
                dist = np.linalg.norm(agent_a.position - agent_b.position)
                if (dist <= self.config.helping_radius
                        and agent_a.current_prediction_error > 0.05
                        and agent_b.current_prediction_error > 0.05):
                    aid_a = agent_a.config.agent_id
                    aid_b = agent_b.config.agent_id
                    joint_rewards[aid_a] = joint_rewards.get(aid_a, 0.0) + self.config.joint_curiosity_bonus
                    joint_rewards[aid_b] = joint_rewards.get(aid_b, 0.0) + self.config.joint_curiosity_bonus

        # --- Phase 6.4: property-sensitive approach rewards ---
        # Compute into a dict first so the rewards can be summed into `reward`
        # before store_experience(), reaching the REINFORCE gradient.
        dangerous_dim = PROPERTY_DIM_MAP['dangerous']
        edible_dim = PROPERTY_DIM_MAP['edible']
        property_approach_rewards: Dict[int, float] = {}
        for agent in self.agents:
            aid = agent.config.agent_id
            for obj_key, obj in self.env.objects.items():
                dist = self.env.toroidal_distance(
                    tuple(agent.position), obj.position
                )
                if dist <= self.config.danger_radius:
                    if obj.properties[dangerous_dim] > 0.5:
                        property_approach_rewards[aid] = (
                            property_approach_rewards.get(aid, 0.0)
                            + self.config.property_approach_penalty
                        )
                if dist <= self.config.food_radius:
                    if obj.properties[edible_dim] > 0.5:
                        # Food bonus: use event rate if in event window, else baseline
                        in_event = (
                            self.episode_event_active
                            and obj_key == self.active_event_object
                            and self.event_start_step <= self.episode_step < self.event_end_step
                        )
                        food_r = (
                            self.config.event_food_bonus if in_event
                            else self.config.property_food_bonus
                        )
                        property_approach_rewards[aid] = (
                            property_approach_rewards.get(aid, 0.0) + food_r
                        )

                        # Track event arrivals: record distinct agents near event object during window
                        if in_event:
                            self.episode_event_agent_arrivals.add(aid)

        # --- Phase 7: Compute rewards (curiosity + helping + naming + communicative + referral + joint + property) ---
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
            # Property naming reward (approach-only: fires when property utterance is correct)
            prop_naming_reward = self.teacher.compute_property_naming_reward(agent, self.env)
            # Communicative reward (approach-only: only fires when shared understanding confirmed)
            comm_reward = step_comm_rewards.get(agent.config.agent_id, 0.0)
            reward += naming_reward + prop_naming_reward + comm_reward
            reward += referral_rewards.get(agent.config.agent_id, 0.0)
            reward += joint_rewards.get(agent.config.agent_id, 0.0)
            reward += property_approach_rewards.get(agent.config.agent_id, 0.0)

            # Store experience
            # Phase 6: store raw perception so update_policy() can replay the
            # trajectory through encoder → GRU with fresh gradients (BPTT).
            # The pre-action perception (step_perceptions) is used as the input
            # that produced the action — this is the perception the agent acted on.
            agent.store_experience(
                perception=step_perceptions[agent.config.agent_id],
                action=actions[agent.config.agent_id],
                reward=reward,
            )

        # Accumulate episode totals for metrics logging
        for _agent in self.agents:
            _aid = _agent.config.agent_id
            self.episode_referral_totals[_aid] = self.episode_referral_totals.get(_aid, 0.0) + referral_rewards.get(_aid, 0.0)
            self.episode_joint_totals[_aid] = self.episode_joint_totals.get(_aid, 0.0) + joint_rewards.get(_aid, 0.0)
        for _aid, _r in property_approach_rewards.items():
            self.episode_property_approach_totals[_aid] = (
                self.episode_property_approach_totals.get(_aid, 0.0) + _r
            )

        # --- Phase 8: Train forward models (every step) ---
        for agent in self.agents:
            agent.train_forward_model(actions[agent.config.agent_id])

        # Update prev_utterances for next step (agents remember what was just said)
        self.prev_utterances = step_utterances
    
    def run_episode(self, episode: int):
        """Run one full episode."""
        self.setup_curriculum(episode)

        # Reset per-episode Stage 4 state (zeros goal vecs, clear ref game tracking)
        self._goal_class_vecs = {
            a.config.agent_id: np.zeros(N_OBJECT_CLASSES, dtype=np.float32)
            for a in self.agents
        }
        self._ref_game_state = {'active': False}

        # Space agents apart for information asymmetry, then possibly nudge one
        # toward an object its partner hasn't seen (directed discovery).
        self._reset_agent_positions(episode)
        self._setup_directed_discovery(episode)

        # Clear per-episode communication state
        self.prev_utterances = {}
        self.utterance_log = []
        self.episode_step = 0
        self.episode_referral_totals = {a.config.agent_id: 0.0 for a in self.agents}
        self.episode_joint_totals = {a.config.agent_id: 0.0 for a in self.agents}
        self.episode_property_comm_totals = {a.config.agent_id: 0.0 for a in self.agents}
        self.episode_property_approach_totals = {a.config.agent_id: 0.0 for a in self.agents}

        # Phase 6: food event setup — decide whether a food event fires this episode
        self.episode_event_active = False
        self.episode_event_agent_arrivals = set()
        self.active_event_object = None
        self.event_start_step = 0
        self.event_end_step = 0

        if random.random() < self.config.p_event:
            edible_dim = PROPERTY_DIM_MAP['edible']
            edible_keys = [
                k for k, obj in self.env.objects.items()
                if obj.properties[edible_dim] > 0.5
            ]
            if edible_keys:
                self.active_event_object = random.choice(edible_keys)
                # Assumes steps_per_episode >= event_duration + 15 (default: 100 >= 35)
                event_start = random.randint(
                    10,
                    max(11, self.config.steps_per_episode - self.config.event_duration - 5)
                )
                self.event_start_step = event_start
                self.event_end_step = event_start + self.config.event_duration
                self.episode_event_active = True

        # Phase 6: reset GRU hidden states to zeros at the start of each episode.
        # The GRU should have no temporal context carried over from the previous episode.
        for agent in self.agents:
            self.hidden_states[agent.config.agent_id] = agent.reset_hidden()

        # Stamp current episode onto each agent (used by spatial memory decay)
        for agent in self.agents:
            agent.current_episode = episode
        
        # Stage 4: 35% of episodes are reference game episodes
        if (self.current_stage >= 4
                and random.random() < self.config.ref_game_prob):
            self.run_reference_game_episode(episode)
            # Shared cleanup: vocab refresh + temperature decay
            if (episode > 0
                    and episode % self.teaching_config.refresh_interval_episodes == 0):
                self.teacher.refresh_vocabularies(self.agents, self.env, episode)
            self.temperature = max(
                self.config.temperature_end,
                self.temperature * self.config.temperature_decay
            )
            return

        for step in range(self.config.steps_per_episode):
            self.run_step()
        
        # Update policies (less frequent than forward model).
        # Pass steps_per_episode so update_policy() replays the full episode trajectory.
        if episode % self.config.policy_update_freq == 0:
            for agent in self.agents:
                agent.update_policy(steps_per_episode=self.config.steps_per_episode)
        
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
        
        _episode_steps = (self.config.ref_game_steps
                          if self._ref_game_state.get('active', False)
                          else self.config.steps_per_episode)
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
                'referral_reward': self.episode_referral_totals.get(agent.config.agent_id, 0.0),
                'joint_reward': self.episode_joint_totals.get(agent.config.agent_id, 0.0),
                'property_comm_reward': self.episode_property_comm_totals.get(agent.config.agent_id, 0.0),
                'property_approach_reward': self.episode_property_approach_totals.get(agent.config.agent_id, 0.0),
                'utterance_count': sum(1 for e in self.utterance_log if e['agent_id'] == agent.config.agent_id),
                'utterance_rate': sum(1 for e in self.utterance_log if e['agent_id'] == agent.config.agent_id) / max(1, _episode_steps),
                'memory_entries': len(agent.spatial_memory.entries),
                'memory_avg_salience': float(np.mean([
                    agent.spatial_memory.get_decayed_salience(c, self.current_episode, agent.config.memory_decay_horizon)
                    for c in range(N_OBJECT_CLASSES)
                ])),
                'property_utterance_count': sum(
                    1 for e in self.utterance_log
                    if e['agent_id'] == agent.config.agent_id and e.get('type') == 'property'
                ),
                'property_utterance_rate': sum(
                    1 for e in self.utterance_log
                    if e['agent_id'] == agent.config.agent_id and e.get('type') == 'property'
                ) / max(1, _episode_steps),
                'property_vocab_size': len(agent.property_vocabulary),
            }
        
        # Food event metrics (episode-level)
        metrics['event_active'] = self.episode_event_active
        metrics['event_arrivals'] = len(self.episode_event_agent_arrivals)

        # Stage 4: Reference game metrics (episode-level)
        # runner_min_distance may be float('inf') for timeout � convert to None for JSON
        ref_state = dict(self._ref_game_state)
        if ref_state.get('runner_min_distance') == float('inf'):
            ref_state['runner_min_distance'] = None
        metrics['ref_game'] = ref_state

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
                'property_vocabulary': {k: v.tolist() for k, v in agent.property_vocabulary.items()},
                'prediction_error_history': agent.prediction_error_history[-100:],
                'learning_progress_history': agent.learning_progress_history[-100:],
                'confidence_history': agent.confidence_history[-100:],
                'naming_loss_history': agent.naming_loss_history[-100:],
                'discrimination_loss_history': agent.discrimination_loss_history[-100:],
                'spatial_memory': agent.spatial_memory.serialize(),
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
                          f"(perception_dim may have changed 139->149 for Phase 4, "
                          f"149->154 for Phase 3.5, or 154->164 for Stage 4). "
                          f"Re-applying structured initialization.")
                    apply_structured_initialization(agent, seed=self.config.seed + aid * 1000)
                agent.position = np.array(data['position'])
                agent.confidence = data['confidence']
                agent.total_steps = data['total_steps']
                agent.total_reward = data['total_reward']
                agent.vocabulary = {
                    k: np.array(v) for k, v in data['vocabulary'].items()
                }
                if 'property_vocabulary' in data:
                    agent.property_vocabulary = {
                        k: np.array(v) for k, v in data['property_vocabulary'].items()
                    }
                agent.naming_loss_history = data.get('naming_loss_history', [])
                agent.discrimination_loss_history = data.get('discrimination_loss_history', [])
                if 'spatial_memory' in data:
                    from agents.curious_agent import SpatialMemory
                    agent.spatial_memory = SpatialMemory.deserialize(data['spatial_memory'])
                agent.current_episode = self.current_episode

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
        if self.current_stage >= 4:
            self.env.setup_stage_3()  # Stage 4 uses same 12-object world as Stage 3
            self.env.setup_stage_4()
        elif self.current_stage >= 3:
            self.env.setup_stage_3()
        elif self.current_stage >= 2:
            self.env.setup_stage_2()
        elif self.current_stage >= 1:
            self.env.setup_stage_1()
        self.env.spawn_objects()  # Phase 5: replace with stable 12-object world
        
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
