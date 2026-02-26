"""
Curious Agent - Neural Architecture
====================================

Design philosophy: APPROACH-ONLY motivation.

There is no punishment. There is no negative reward signal.
Prediction error is neutral information — not pain.
The agent is drawn toward interesting things (learning progress),
not fleeing from bad things.

Think of it as designing around the dopaminergic "that's interesting!"
system rather than the amygdala-mediated "that hurts, avoid it" system.

Architecturally this means:
  - Reward = max(0, prev_error - curr_error)  →  only positive
  - No penalty for confusion, failure, or novelty
  - Curiosity = attraction to the zone of learnable complexity
  - Helping = reward when your action increases another's learning

The agent is a small feedforward network (~50K-100K params).
Fits easily on a GTX 1080. Multiple agents run simultaneously.

Phase 6: GRUCell inserted between encoder and all downstream heads.
The GRU maintains a hidden state across steps within an episode,
giving the agent genuine temporal context. The Trainer manages the
hidden state tensor externally and passes it in at each step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import copy


# =============================================================================
# Utility Modules
# =============================================================================

class NormedLinear(nn.Module):
    """Linear layer with layer normalization. Stabilizes small networks."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.norm(self.linear(x))


# =============================================================================
# Spatial Memory
# =============================================================================

@dataclass
class SpatialMemory:
    """
    Per-agent persistent memory of object encounters, decaying over episodes.

    Stores the most recent encounter with each object class: where it was,
    how novel it felt (prediction error = salience), and when. Salience
    decays exponentially across episodes so stale memories don't dominate.

    Persisted in checkpoints — agents remember across training sessions.
    """
    entries: Dict[int, Dict] = field(default_factory=dict)
    # entries[class_idx] = {
    #   'salience':      float  — prediction_error at time of encounter
    #   'episode':       int    — episode last seen
    #   'position':      list   — [x, y] world coordinates
    #   'times_visited': int    — total lifetime encounters with this class
    # }

    def update(self, class_idx: int, salience: float, episode: int,
               position: np.ndarray):
        existing = self.entries.get(class_idx, {'times_visited': 0})
        self.entries[class_idx] = {
            'salience': float(salience),
            'episode': int(episode),
            'position': [float(position[0]), float(position[1])],
            'times_visited': existing['times_visited'] + 1,
        }

    def get_decayed_salience(self, class_idx: int, current_episode: int,
                              decay_horizon: int = 20) -> float:
        if class_idx not in self.entries:
            return 0.0
        entry = self.entries[class_idx]
        delta = max(0, current_episode - entry['episode'])
        return entry['salience'] * float(np.exp(-delta / decay_horizon))

    def to_vec(self, current_episode: int, n_classes: int = 10,
               decay_horizon: int = 20) -> np.ndarray:
        """10D summary vector: decayed salience per object class."""
        vec = np.zeros(n_classes, dtype=np.float32)
        for c in range(n_classes):
            vec[c] = self.get_decayed_salience(c, current_episode, decay_horizon)
        return vec

    def serialize(self) -> Dict:
        return {'entries': {str(k): v for k, v in self.entries.items()}}

    @classmethod
    def deserialize(cls, data: Dict) -> 'SpatialMemory':
        mem = cls()
        mem.entries = {int(k): v for k, v in data.get('entries', {}).items()}
        return mem


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class AgentConfig:
    """All hyperparameters in one place for reproducibility."""

    # Perception
    perception_dim: int = 129       # From environment: 8 objects * (1 + 15) + 1
    max_perceivable_objects: int = 8

    # Internal representation
    hidden_dim: int = 128           # Main processing width
    internal_state_dim: int = 64    # Compressed internal state / GRU hidden size

    # Actions
    n_actions: int = 5              # move_north, move_south, move_east, move_west, stay
    n_utterance_classes: int = 10   # One utterance action per vocabulary word (Phase 3)
    action_names: List[str] = field(default_factory=lambda: [
        'move_north', 'move_south', 'move_east', 'move_west', 'stay'
    ])
    move_step: float = 5.0          # How far each move goes

    # Motivation (APPROACH-ONLY — no negative signals)
    curiosity_weight: float = 1.0   # Reward for learning progress
    helping_weight: float = 0.5     # Reward for helping others learn
    exploration_bonus: float = 0.1  # Small constant reward for movement (anti-stagnation)
    novelty_weight: float = 0.3     # Reward for visiting new grid cells
    visit_grid_resolution: int = 10 # Grid cells per axis for visit tracking (10x10 = 100 cells)

    # Learning
    learning_rate: float = 3e-4
    forward_model_lr: float = 1e-3  # Forward model learns faster (it needs to keep up)
    gamma: float = 0.95             # Discount factor
    entropy_coeff: float = 0.05     # Entropy regularization: penalizes logit concentration,
                                    # keeps utterance actions viable at low temperature
                                    # (raised from 0.02 — gap was closing too slowly at 0.02)

    # Meta-cognition (Phase 1: simple version)
    confidence_smooth: float = 0.95 # EMA smoothing for confidence tracking

    # Language-shaped encoder (Phase 2+)
    # n_object_classes must match N_OBJECT_CLASSES in training/language_grounding.py
    n_object_classes: int = 10          # One class per word in the full vocabulary
    naming_loss_weight: float = 0.1     # Encoder alignment: pull state → word prototype
    discrimination_loss_weight: float = 0.2  # Auxiliary classifier: state → class label
    property_loss_weight: float = 0.2   # Property discrimination head loss weight
    n_property_utterances: int = 5        # Property word emission actions (Phase 3.5)
    n_property_classes: int = 5           # For property discrimination head

    # Spatial Memory (Phase 4)
    memory_decay_horizon: int = 20        # Episodes for salience to decay to ~37%
    memory_salience_threshold: float = 0.01  # Min prediction error to update memory

    # World
    world_size: float = 100.0
    perception_radius: float = 15.0

    # Identity
    agent_id: int = 0
    name: str = "agent_0"


# =============================================================================
# The Agent
# =============================================================================

class CuriousAgent(nn.Module):
    """
    A small agent that:
    1. Perceives structured properties of nearby objects
    2. Builds a forward model (predicts next hidden state from h_t + action)
    3. Is rewarded ONLY for learning progress (prediction error going DOWN)
    4. Can observe other agents and is rewarded for helping them learn
    5. Tracks its own confidence (proto-metacognition)

    Phase 6: A GRUCell sits between the encoder and all downstream heads.
    The encoder maps perception → 64D encoding; the GRU integrates this
    with the previous hidden state to produce h_t. All heads (policy, value,
    naming, discrimination, property) consume h_t instead of raw encoder output.
    The forward model predicts the NEXT h_t: forward_model(h_t, action) → h_{t+1}.

    No punishment. No negative reward. Confusion is just information.
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config

        # ===== PERCEPTION ENCODER =====
        # Takes flat perception vector → compressed encoding (64D)
        # Output feeds into GRUCell, not directly into heads.
        self.encoder = nn.Sequential(
            NormedLinear(config.perception_dim, config.hidden_dim),
            nn.GELU(),  # Smooth activation (no dead neurons like ReLU)
            NormedLinear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            NormedLinear(config.hidden_dim, config.internal_state_dim),
            nn.Tanh(),  # Bounded encoding
        )

        # ===== GRU CELL (Phase 6) =====
        # Integrates new encoder output with previous hidden state.
        # h_t = GRU(encoded_t, h_{t-1})
        # All downstream heads use h_t — this gives the agent temporal context.
        self.gru = nn.GRUCell(
            input_size=config.internal_state_dim,
            hidden_size=config.internal_state_dim,
        )

        # ===== FORWARD MODEL (World Predictor) =====
        # Given current hidden state h_t + action → predict next hidden state h_{t+1}.
        # Curiosity = surprise = ||predicted_h_{t+1} - actual_h_{t+1}||²
        # Input: (h_t || action_5D) = 64 + 5 = 69D
        self.forward_model = nn.Sequential(
            NormedLinear(config.internal_state_dim + config.n_actions, config.hidden_dim),
            nn.GELU(),
            NormedLinear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.internal_state_dim),
            nn.Tanh(),
        )

        # ===== POLICY (Action Selection) =====
        # Given hidden state h_t → action preferences (softmax probabilities)
        # Total action space: n_actions (movement) + n_utterance_classes (word emissions)
        # The forward model only uses movement actions; utterance actions expand policy only.
        _total_actions = config.n_actions + config.n_utterance_classes + config.n_property_utterances
        self.policy = nn.Sequential(
            NormedLinear(config.internal_state_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, _total_actions),
            # No final activation — we'll use softmax at decision time
        )

        # ===== VALUE ESTIMATOR =====
        # Estimates expected future reward (for advantage computation)
        # Input: h_t
        self.value_head = nn.Sequential(
            NormedLinear(config.internal_state_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # ===== DISCRIMINATION HEAD (Auxiliary Object Classifier) =====
        # Takes h_t → object class logits.
        # Trained with cross-entropy as a parallel objective alongside curiosity.
        # Forces the encoder+GRU to produce class-separable representations without
        # replacing or disrupting the forward model / policy gradient flow.
        # Gradients flow through encoder+GRU via a dedicated language_optimizer.
        self.discrimination_head = nn.Sequential(
            nn.Linear(config.internal_state_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.n_object_classes),
        )

        self.property_discrimination_head = nn.Sequential(
            nn.Linear(config.internal_state_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, config.n_property_classes),
            nn.Sigmoid(),   # Binary per property — multiple can be true simultaneously
        )

        # ===== OBSERVABLE STATE (what others can see about me) =====
        # A small "expression" vector that leaks some internal state
        # Think of it as body language / facial expression
        self.expression_projection = nn.Linear(config.internal_state_dim, 8)

        # ===== TRACKING STATE (not learned, just bookkeeping) =====
        self.position = np.array([50.0, 50.0])  # Start at center
        self.internal_state = None                # Current h_t (GRU output)
        self.prev_internal_state = None           # Previous step's h_t
        self.prev_prediction_error = 0.0          # For learning progress calc
        self.current_prediction_error = 0.0

        # Confidence tracker (exponential moving average of prediction accuracy)
        # Starts at 0.5 (uncertain). Rises as predictions improve.
        self.confidence = 0.5

        # Visit tracking for novelty bonus (grid-based)
        self.visit_counts: Dict[Tuple[int, int], int] = {}

        # Experience buffer for learning
        # Phase 6: stores (perception_np, action, reward) tuples for full trajectory replay.
        # Ordered list — one entry per step within the episode.
        # Buffer must hold at least one full episode (100 steps) so update_policy()
        # can replay the complete trajectory. 256 gives room for the current episode
        # plus some history; the replay window is capped at steps_per_episode (100).
        self.experience_buffer: List[Dict] = []
        self.max_buffer_size = 256

        # Lifetime stats
        self.total_steps = 0
        self.total_reward = 0.0
        self.learning_progress_history: List[float] = []
        self.prediction_error_history: List[float] = []
        self.confidence_history: List[float] = []
        self.action_history: List[int] = []
        self.position_history: List[Tuple[float, float]] = []

        # Vocabulary (for language grounding — starts empty)
        self.vocabulary: Dict[str, np.ndarray] = {}

        # Last utterance this step: class index (0-9) or None if agent moved/stayed
        self.last_utterance_class: Optional[int] = None

        # Property vocabulary (Phase 3.5): word → prototype vector
        self.property_vocabulary: Dict[str, np.ndarray] = {}
        # Last property utterance this step: property index (0-4) or None
        self.last_utterance_property: Optional[int] = None

        # Language loss tracking
        self.naming_loss_history: List[float] = []
        self.discrimination_loss_history: List[float] = []

        # Spatial memory (Phase 4): persistent object-encounter memory across episodes
        self.spatial_memory = SpatialMemory()
        self.current_episode: int = 0

        # Set up optimizers
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Three separate optimizers for three separate learning signals.

        Phase 6: GRU params are added to BOTH policy_optimizer and language_optimizer,
        mirroring the encoder sharing pattern. Each optimizer manages its own
        zero_grad/step cycle so they don't interfere with each other.
        """
        # Policy + value learn from intrinsic reward (REINFORCE).
        # Encoder and GRU are shared with language_optimizer.
        policy_params = (
            list(self.encoder.parameters())
            + list(self.gru.parameters())
            + list(self.policy.parameters())
            + list(self.value_head.parameters())
            + list(self.expression_projection.parameters())
        )
        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=self.config.learning_rate
        )

        # Forward model learns from prediction error (self-supervised).
        # It predicts the next h_t, but its own parameters are separate —
        # gradients do NOT flow back through the GRU during forward model training.
        self.forward_model_optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=self.config.forward_model_lr
        )

        # Language losses shape encoder + GRU + discrimination heads via supervised signal.
        # Encoder and GRU params are shared with policy_optimizer — each optimizer manages
        # its own zero_grad/step cycle so they don't interfere with each other.
        language_params = (
            list(self.encoder.parameters())
            + list(self.gru.parameters())
            + list(self.discrimination_head.parameters())
            + list(self.property_discrimination_head.parameters())
        )
        self.language_optimizer = torch.optim.Adam(
            language_params, lr=self.config.learning_rate
        )

    # =========================================================================
    # Phase 6: Hidden State Management
    # =========================================================================

    def reset_hidden(self, device: torch.device = None) -> torch.Tensor:
        """
        Return a zero hidden state tensor for the start of a new episode.

        The Trainer calls this at episode start and holds the returned tensor.
        Shape: (1, internal_state_dim) — batch dim of 1 for GRUCell compatibility.

        Args:
            device: target device for the tensor (default: CPU)
        """
        if device is None:
            device = torch.device('cpu')
        return torch.zeros(1, self.config.internal_state_dim, device=device)

    def gru_step(self, perception: np.ndarray,
                 hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One forward pass through encoder → GRU → h_t.

        This is the primary step API for the Trainer. It replaces the old
        `perceive()` call for the collection path. The Trainer passes the
        current hidden state in and gets the new hidden state back.

        Side-effects (same as old perceive()):
          - self.prev_internal_state ← old self.internal_state
          - self.internal_state ← new h_t

        Args:
            perception: flat numpy perception vector (perception_dim,)
            hidden_state: current GRU hidden state, shape (1, internal_state_dim).
                          Should be detached before calling during collection so
                          gradients don't accumulate across steps.

        Returns:
            h_t: new GRU hidden state, shape (1, internal_state_dim)
            h_t: same tensor (h_t is both the new hidden state and the
                 representation used by downstream heads — GRUCell output IS
                 the new hidden state)
        """
        x = torch.FloatTensor(perception).unsqueeze(0)  # (1, perception_dim)
        encoded = self.encoder(x)                        # (1, internal_state_dim)
        h_t = self.gru(encoded, hidden_state)            # (1, internal_state_dim)

        self.prev_internal_state = self.internal_state
        self.internal_state = h_t

        # Return (new_hidden, h_t) — they are the same tensor for GRUCell
        return h_t, h_t

    # =========================================================================
    # Core Loop: Perceive → Predict → Act → Learn
    # =========================================================================

    def perceive(self, flat_perception: np.ndarray,
                 hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode raw perception into internal state (h_t from GRU).

        Phase 6: hidden_state is now required for full functionality.
        If hidden_state is None, a fresh zero state is used (backward compat
        for any call sites that don't yet pass hidden state).

        Updates self.internal_state to h_t. Returns h_t.
        """
        if hidden_state is None:
            hidden_state = self.reset_hidden()

        x = torch.FloatTensor(flat_perception).unsqueeze(0)  # (1, perception_dim)
        encoded = self.encoder(x)                             # (1, internal_state_dim)
        h_t = self.gru(encoded, hidden_state)                 # (1, internal_state_dim)

        self.prev_internal_state = self.internal_state
        self.internal_state = h_t
        return h_t

    def decide_action(self, temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
        """
        Choose an action based on current hidden state h_t.

        Uses softmax policy with temperature for exploration.
        High temperature = more random. Low = more greedy.

        Returns: (action_index, log_probability)
        """
        assert self.internal_state is not None, "Must perceive before acting"

        logits = self.policy(self.internal_state)  # (1, n_actions)

        # Temperature-scaled softmax
        probs = F.softmax(logits / max(temperature, 0.01), dim=-1)

        # Sample action
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).squeeze()  # Ensure scalar

        return action.item(), log_prob

    def execute_action(self, action: int) -> np.ndarray:
        """
        Execute movement or utterance action. Returns new position.

        Actions 0-3: movement (north/south/east/west)
        Action 4:    stay
        Actions 5+:  utterance — emit word at class index (action - n_actions); no movement
        """
        step = self.config.move_step

        if action < self.config.n_actions:
            # Movement action — clear any previous utterance
            self.last_utterance_class = None
            self.last_utterance_property = None

            if action == 0:    # North
                self.position[1] += step
            elif action == 1:  # South
                self.position[1] -= step
            elif action == 2:  # East
                self.position[0] += step
            elif action == 3:  # West
                self.position[0] -= step
            # action == 4: stay (no position change)

            # Toroidal wrapping — world has no edges
            self.position[0] %= self.config.world_size
            self.position[1] %= self.config.world_size
        else:
            # Utterance action — no movement, emit a word
            action_offset = action - self.config.n_actions
            if action_offset < self.config.n_utterance_classes:
                # Object utterance
                self.last_utterance_class = action_offset
                self.last_utterance_property = None
            else:
                # Property utterance
                self.last_utterance_class = None
                self.last_utterance_property = action_offset - self.config.n_utterance_classes

        self.action_history.append(action)
        self.position_history.append(tuple(self.position.copy()))

        return self.position.copy()

    def reset_position(self, seed: Optional[int] = None):
        """
        Randomize position for a new episode.
        Optionally resets visit counts so novelty bonus refreshes.
        """
        rng = np.random.RandomState(seed)
        self.position = rng.uniform(0, self.config.world_size, size=2)
        self.visit_counts.clear()

    def update_memory(self, nearby_object_classes: List[Tuple[int, np.ndarray]],
                      prediction_error: float, episode: int):
        """Update spatial memory after a prediction error is computed."""
        if prediction_error < self.config.memory_salience_threshold:
            return
        for class_idx, position in nearby_object_classes:
            self.spatial_memory.update(class_idx, prediction_error, episode, position)

    def compute_prediction_error(self, action: int) -> float:
        """
        How wrong was my prediction about what would happen?

        Phase 6: The forward model predicts the next h_t (hidden state).
        We compare forward_model(prev_h_t, action) against actual h_t.

        This is INFORMATION, not pain. It tells the agent where
        its world model is inaccurate.

        Returns: scalar prediction error (MSE between predicted and actual h_t)
        """
        if self.prev_internal_state is None:
            return 0.0

        # Map utterance actions to "stay" for forward model — utterances don't move the agent,
        # so the physical forward model only needs to distinguish movement directions.
        forward_action = action if action < self.config.n_actions else (self.config.n_actions - 1)
        action_onehot = torch.zeros(1, self.config.n_actions)
        action_onehot[0, forward_action] = 1.0

        # What did I predict the next hidden state would be?
        forward_input = torch.cat([self.prev_internal_state.detach(), action_onehot], dim=-1)
        predicted_next_h = self.forward_model(forward_input)

        # What is the actual next hidden state?
        actual_next_h = self.internal_state.detach()

        # Prediction error (MSE between predicted and actual h_t)
        error = F.mse_loss(predicted_next_h, actual_next_h).item()

        # Update tracking
        self.prev_prediction_error = self.current_prediction_error
        self.current_prediction_error = error
        self.prediction_error_history.append(error)

        # Update confidence (EMA of prediction accuracy)
        accuracy = max(0, 1.0 - error)  # Higher accuracy = lower error
        alpha = 1.0 - self.config.confidence_smooth
        self.confidence = self.config.confidence_smooth * self.confidence + alpha * accuracy
        self.confidence_history.append(self.confidence)

        return error

    def compute_intrinsic_reward(self,
                                  others_prev_errors: Optional[List[float]] = None,
                                  others_curr_errors: Optional[List[float]] = None
                                  ) -> float:
        """
        APPROACH-ONLY intrinsic reward.

        Components:
        1. Learning progress: reward when prediction error DECREASES
           - Positive when learning (error goes down)
           - Zero when already mastered (error already low)
           - Zero when incomprehensible (error stays high)
           → Naturally explores at the boundary of competence

        2. Helping bonus: reward when others' prediction errors decrease
           while you're near them (your presence helped them learn)

        3. Exploration bonus: tiny constant reward for moving
           (prevents the agent from just sitting still forever)

        CRITICAL: No negative component. Worst case = 0 reward.
        """
        # --- Curiosity: my own learning progress ---
        learning_progress = max(0.0, self.prev_prediction_error - self.current_prediction_error)
        curiosity_reward = self.config.curiosity_weight * learning_progress

        # --- Helping: others' learning progress ---
        helping_reward = 0.0
        if others_prev_errors is not None and others_curr_errors is not None:
            for prev_e, curr_e in zip(others_prev_errors, others_curr_errors):
                other_progress = max(0.0, prev_e - curr_e)
                helping_reward += other_progress
            helping_reward *= self.config.helping_weight

        # --- Exploration: small bonus for movement ---
        # Only actions 0-3 are movements; 4=stay, 5+=utterance (no physical displacement)
        moved = (len(self.action_history) > 0 and self.action_history[-1] < 4)
        exploration_reward = self.config.exploration_bonus if moved else 0.0

        # --- Novelty: reward for visiting less-seen grid cells ---
        grid_res = self.config.visit_grid_resolution
        cell_size = self.config.world_size / grid_res
        gx = int(self.position[0] // cell_size) % grid_res
        gy = int(self.position[1] // cell_size) % grid_res
        cell = (gx, gy)
        self.visit_counts[cell] = self.visit_counts.get(cell, 0) + 1
        novelty_reward = self.config.novelty_weight / self.visit_counts[cell]

        total_reward = curiosity_reward + helping_reward + exploration_reward + novelty_reward

        self.total_reward += total_reward
        self.learning_progress_history.append(learning_progress)
        self.total_steps += 1

        return total_reward

    # =========================================================================
    # Forward Model Training (Self-Supervised)
    # =========================================================================

    def train_forward_model(self, action: int):
        """
        Train the forward model on (h_t, action) → next_h_t prediction.

        Phase 6: The forward model operates in hidden-state space rather than
        raw encoder space. We use detached prev_internal_state (h_{t-1}) and
        detached internal_state (h_t) so forward model gradients don't flow
        through the GRU or encoder.

        This is self-supervised: no external reward needed.
        The forward model just tries to predict what will happen.
        Better forward model = more accurate prediction error signal
        = better curiosity drive.
        """
        if self.prev_internal_state is None:
            return 0.0

        # Map utterance actions to "stay" — same as in compute_prediction_error
        forward_action = action if action < self.config.n_actions else (self.config.n_actions - 1)
        action_onehot = torch.zeros(1, self.config.n_actions)
        action_onehot[0, forward_action] = 1.0

        # Use detached h_{t-1} and h_t — forward model has its own optimizer
        forward_input = torch.cat([self.prev_internal_state.detach(), action_onehot], dim=-1)
        predicted_next_h = self.forward_model(forward_input)
        actual_next_h = self.internal_state.detach()

        loss = F.mse_loss(predicted_next_h, actual_next_h)

        self.forward_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
        self.forward_model_optimizer.step()

        return loss.item()

    # =========================================================================
    # Language Loss Training (Naming + Discrimination — parallel objectives)
    # =========================================================================

    def train_language_losses(self, word: str, class_idx: int,
                               h_t: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """
        Train naming alignment loss and discrimination loss in a single backward pass.

        Phase 6: Accepts an explicit h_t parameter (the current GRU hidden state).
        If h_t is not provided, falls back to self.internal_state for backward
        compatibility. Callers (Trainer teach_step) should pass the current h_t
        so the language losses flow through GRU → encoder gradients properly.

        Naming loss (MSE):
          Pulls current h_t toward the stored word prototype.
          When the agent sees an apple and is told "apple", the hidden state
          should converge toward the prototype averaged across prior apple
          sightings. This makes representations consistent across contexts.

        Discrimination loss (cross-entropy):
          Auxiliary classifier: h_t → object class prediction.
          Forces the encoder+GRU to produce class-separable representations.
          The discrimination_head is a separate small network — its gradients
          flow through the shared encoder+GRU but the head itself isn't used for
          policy decisions or curiosity computation.

        Both losses share language_optimizer (encoder + GRU + discrimination_head).
        Policy and forward-model optimizers are unaffected.

        Args:
            word: the word that was just taught (must exist in self.vocabulary
                  for naming loss to fire; discrimination loss only needs class_idx)
            class_idx: integer class label (from WORD_CLASS_MAP); -1 to skip discrimination
            h_t: optional explicit hidden state tensor, shape (1, internal_state_dim).
                 If None, uses self.internal_state.

        Returns:
            (naming_loss_value, discrimination_loss_value) as plain floats for logging.
        """
        state = h_t if h_t is not None else self.internal_state
        if state is None:
            return 0.0, 0.0

        has_loss = False
        total_loss = None
        naming_val = 0.0
        disc_val = 0.0

        # --- Naming alignment loss ---
        # Only fires once the word has a stable prototype in vocabulary.
        if word in self.vocabulary:
            prototype = torch.FloatTensor(self.vocabulary[word]).unsqueeze(0)
            naming_loss = F.mse_loss(state, prototype)
            total_loss = self.config.naming_loss_weight * naming_loss
            naming_val = naming_loss.item()
            has_loss = True

        # --- Discrimination loss ---
        # Fires whenever a valid class index is provided.
        if 0 <= class_idx < self.config.n_object_classes:
            logits = self.discrimination_head(state)
            label = torch.tensor([class_idx])
            disc_loss = F.cross_entropy(logits, label)
            weighted = self.config.discrimination_loss_weight * disc_loss
            total_loss = weighted if total_loss is None else total_loss + weighted
            disc_val = disc_loss.item()
            has_loss = True

        if has_loss:
            self.language_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters())
                + list(self.gru.parameters())
                + list(self.discrimination_head.parameters()),
                1.0,
            )
            self.language_optimizer.step()

        self.naming_loss_history.append(naming_val)
        self.discrimination_loss_history.append(disc_val)
        return naming_val, disc_val

    def train_property_losses(self, target_vec: np.ndarray) -> float:
        """
        Train property discrimination head on the current hidden state h_t.

        target_vec: 5D binary numpy array — 1.0 for each property the nearby
                    object qualifies for, 0.0 for the rest. Using the full
                    multi-label target avoids conflicting gradients: training
                    one property at a time incorrectly suppresses co-occurring
                    true properties (e.g. fire is dangerous AND warm AND bright,
                    so all three should be 1.0 simultaneously, not trained as
                    three separate one-hot calls that fight each other).

        Returns BCE loss value (float) for logging.
        """
        if self.internal_state is None:
            return 0.0

        target = torch.tensor(target_vec, dtype=torch.float32).unsqueeze(0)  # (1, 5)

        # Intentionally detach h_t before passing it to the property head.
        # Property losses train only the property_discrimination_head parameters —
        # they do NOT backpropagate through encoder or GRU. This is a deliberate
        # design choice: the encoder/GRU are already shaped by the naming and
        # discrimination losses (train_language_losses); adding a third gradient
        # signal through the same shared params in the same step risks "backward
        # through freed graph" errors when train_language_losses has already
        # released its computation graph, and risks conflicting updates to the
        # shared representation. Detaching keeps the property head as a clean
        # read-only consumer of the current h_t.
        state = self.internal_state.detach()
        pred = self.property_discrimination_head(state)  # (1, 5)
        loss = torch.nn.functional.binary_cross_entropy(pred, target) * self.config.property_loss_weight

        self.language_optimizer.zero_grad()
        loss.backward()
        self.language_optimizer.step()
        return loss.item()

    def get_predicted_class(self, class_names: Optional[List[str]] = None) -> Optional[str]:
        """
        What object class does the discrimination head currently predict?

        Args:
            class_names: optional list mapping class index → name
                         (e.g. ALL_OBJECT_CLASSES from language_grounding).
                         If None, returns the raw index as a string.
        Returns:
            predicted class name/index, or None if no internal state.
        """
        if self.internal_state is None:
            return None
        with torch.no_grad():
            logits = self.discrimination_head(self.internal_state)
            idx = torch.argmax(logits, dim=-1).item()
        if class_names and 0 <= idx < len(class_names):
            return class_names[idx]
        return str(idx)

    # =========================================================================
    # Policy Training (REINFORCE with baseline)
    # =========================================================================

    def store_experience(self, perception: np.ndarray, action: int, reward: float,
                         # Legacy params kept for call-site backward compat — ignored
                         log_prob: Optional[torch.Tensor] = None,
                         state: Optional[torch.Tensor] = None):
        """Buffer an experience for batch policy update.

        Phase 6: Stores raw (perception, action, reward) instead of
        (state, action, reward). During update_policy() the trajectory is
        re-run sequentially through encoder → GRU with fresh gradients,
        so we need the original perception input rather than a detached h_t.

        The log_prob and state params are accepted but ignored (kept so existing
        Trainer call sites don't immediately break — Task 2 will update them).

        Args:
            perception: flat numpy perception vector for this step
            action: integer action taken
            reward: total reward received this step
            log_prob: IGNORED (kept for backward compat)
            state: IGNORED (kept for backward compat)
        """
        self.experience_buffer.append({
            'perception': perception.copy() if isinstance(perception, np.ndarray)
                          else np.array(perception),
            'action': action,
            'reward': float(reward),
        })

        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def update_policy(self, steps_per_episode: int = 100):
        """
        REINFORCE policy gradient update with sequential GRU trajectory replay.

        Phase 6: Re-runs the full episode trajectory (all 100 steps) sequentially
        through encoder → GRU with fresh gradients. This is proper BPTT over the
        episode — every step's h_t is computed from the replayed trajectory rather
        than from a stored detached snapshot.

        The replay window is the most recent `steps_per_episode` experiences, which
        corresponds exactly to the episode that just ended (update_policy is called
        once per episode from run_episode). h starts at zeros — the same starting
        condition as collection — so the replay is consistent.

        Uses value baseline to reduce variance. Only positive rewards, so
        gradients push toward actions that led to learning or helping.
        """
        if len(self.experience_buffer) < steps_per_episode:
            return 0.0

        # Take the most recent full episode of experiences as a sequential window.
        # experience_buffer is chronologically ordered; the last steps_per_episode
        # entries are exactly the episode that just completed.
        window = self.experience_buffer[-steps_per_episode:]

        # Re-run the full episode trajectory through encoder + GRU with fresh gradients.
        # h_0 = zeros, matching the collection-time reset at episode start.
        h = torch.zeros(1, self.config.internal_state_dim)

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for exp in window:
            x = torch.FloatTensor(exp['perception']).unsqueeze(0)  # (1, perception_dim)
            encoded = self.encoder(x)                               # (1, internal_state_dim)
            h = self.gru(encoded, h)                                # (1, internal_state_dim)
            # h now has gradients flowing through encoder and GRU

            # Recompute log_prob from CURRENT policy (fresh graph)
            logits = self.policy(h)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(exp['action'])).squeeze()

            # Value estimate (baseline)
            value = self.value_head(h)
            advantage = exp['reward'] - value.item()

            # Policy gradient: push toward actions with positive advantage
            policy_losses.append(-log_prob * advantage)

            # Value function loss
            value_losses.append(F.mse_loss(value, torch.tensor([[exp['reward']]])))

            # Entropy bonus: subtract entropy from loss to penalize overconfident distributions.
            # Prevents logits from concentrating so heavily on movement that utterance actions
            # become unreachable (the bootstrapping deadlock observed in Phase 3 training).
            entropy_losses.append(-self.config.entropy_coeff * dist.entropy())

        total_loss = (
            torch.stack(policy_losses).sum()
            + 0.5 * torch.stack(value_losses).sum()
            + torch.stack(entropy_losses).sum()
        ) / steps_per_episode

        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.policy_optimizer.step()

        return total_loss.item()

    # =========================================================================
    # Observable State (what others see about me)
    # =========================================================================

    def get_observable_state(self) -> Dict[str, Any]:
        """
        What other agents can perceive about this agent.

        This is the "body language" — partial transparency into internal state.
        Other agents use this for theory of mind (in later phases).
        """
        expression = torch.zeros(8)
        if self.internal_state is not None:
            expression = torch.sigmoid(
                self.expression_projection(self.internal_state.detach())
            ).squeeze()

        return {
            'agent_id': self.config.agent_id,
            'position': tuple(self.position),
            'confidence': self.confidence,
            'recent_prediction_error': self.current_prediction_error,
            'expression': expression.detach().numpy(),
            'last_action': self.action_history[-1] if self.action_history else None,
            'last_utterance_class': self.last_utterance_class,  # None if moved/stayed
            'learning_rate': self.learning_progress_history[-1]
                           if self.learning_progress_history else 0.0,
        }

    # =========================================================================
    # Language Grounding (Starts empty — Phase 2 will expand this)
    # =========================================================================

    def learn_word(self, word: str, internal_state: torch.Tensor):
        """
        Associate a word with the current internal state (h_t).
        Ostensive learning: human points at thing, says word, agent encodes.
        """
        self.vocabulary[word] = internal_state.detach().squeeze().numpy().copy()

    def try_to_name(self, internal_state: torch.Tensor, threshold: float = 0.7) -> Optional[str]:
        """
        Try to find a word for the current experience.
        Returns None if nothing in vocabulary is close enough.
        """
        if not self.vocabulary:
            return None

        state_np = internal_state.detach().squeeze().numpy()

        best_word = None
        best_sim = -1.0

        for word, stored_state in self.vocabulary.items():
            sim = np.dot(state_np, stored_state) / (
                np.linalg.norm(state_np) * np.linalg.norm(stored_state) + 1e-8
            )
            if sim > best_sim:
                best_sim = sim
                best_word = word

        if best_sim >= threshold:
            return best_word
        return None

    # =========================================================================
    # Meta-Cognitive Report (Proto-metacognition for Phase 1)
    # =========================================================================

    def metacognitive_report(self) -> Dict[str, Any]:
        """
        What the agent can report about its own state.

        Phase 1: Simple tracking metrics.
        Phase 4 will add hierarchical self-modeling.
        """
        recent_n = 20
        recent_errors = self.prediction_error_history[-recent_n:] if self.prediction_error_history else [0]
        recent_progress = self.learning_progress_history[-recent_n:] if self.learning_progress_history else [0]

        recent_naming = self.naming_loss_history[-recent_n:] if self.naming_loss_history else [0.0]
        recent_disc = self.discrimination_loss_history[-recent_n:] if self.discrimination_loss_history else [0.0]

        return {
            'confidence': self.confidence,
            'avg_recent_error': np.mean(recent_errors),
            'avg_recent_progress': np.mean(recent_progress),
            'total_steps': self.total_steps,
            'vocabulary_size': len(self.vocabulary),
            'is_learning': np.mean(recent_progress) > 0.001,  # Actively improving?
            'is_exploring': (len(set(self.action_history[-10:])) > 2
                           if len(self.action_history) >= 10 else True),
            'avg_naming_loss': float(np.mean(recent_naming)),
            'avg_discrimination_loss': float(np.mean(recent_disc)),
        }

    def __repr__(self):
        return (f"CuriousAgent(id={self.config.agent_id}, "
                f"steps={self.total_steps}, "
                f"confidence={self.confidence:.3f}, "
                f"vocab={len(self.vocabulary)})")


# =============================================================================
# Structured Initialization (Biological Bootstrap)
# =============================================================================

def apply_structured_initialization(agent: CuriousAgent, seed: int = 42):
    """
    Initialize weights with biological structure rather than random noise.

    Principles:
    - Encoder: slight bias toward detecting large differences in properties
    - Forward model: initialized near identity (predict no change)
    - Policy: slight bias toward exploration (movement actions)
    - Value head: starts at zero (no prior expectations)
    - GRU: default PyTorch init (orthogonal hidden weights, uniform input weights)
      — we leave GRU at default since its recurrent structure is already well-conditioned.

    This gives the agent a "developmental scaffold" without training it.
    """
    torch.manual_seed(seed)

    with torch.no_grad():
        # --- Encoder: Xavier init with slight structure ---
        for layer in agent.encoder:
            if isinstance(layer, NormedLinear):
                nn.init.xavier_normal_(layer.linear.weight, gain=0.8)
                nn.init.zeros_(layer.linear.bias)

        # --- GRU: leave at PyTorch default initialization ---
        # PyTorch initializes GRUCell with kaiming_uniform for input weights
        # and orthogonal for hidden weights — a well-conditioned starting point.
        # No override needed.

        # --- Forward model: near-identity initialization ---
        # The forward model starts by predicting "nothing changes"
        # This means initial prediction errors come from actual changes
        # in the environment, not from random predictions
        for i, layer in enumerate(agent.forward_model):
            if isinstance(layer, NormedLinear):
                nn.init.xavier_normal_(layer.linear.weight, gain=0.5)
                nn.init.zeros_(layer.linear.bias)
            elif isinstance(layer, nn.Linear):
                # Last layer: small weights → output near zero
                # Combined with Tanh, this means "predict similar state"
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

        # --- Policy: slight exploration bias ---
        for layer in agent.policy:
            if isinstance(layer, NormedLinear):
                nn.init.xavier_normal_(layer.linear.weight, gain=0.5)
                nn.init.zeros_(layer.linear.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                # Movement bias: encourage movement (0-3), discourage stay (4),
                # slightly discourage utterances (5+) so agents explore first
                total_actions = (agent.config.n_actions
                                 + agent.config.n_utterance_classes
                                 + agent.config.n_property_utterances)
                movement_bias = torch.tensor([0.1, 0.1, 0.1, 0.1, -0.1])
                utterance_bias = torch.full(
                    (agent.config.n_utterance_classes + agent.config.n_property_utterances,),
                    -0.2,
                )
                full_bias = torch.cat([movement_bias[:agent.config.n_actions],
                                       utterance_bias])[:total_actions]
                layer.bias.copy_(full_bias)

        # --- Value head: start at zero ---
        for layer in agent.value_head:
            if isinstance(layer, NormedLinear):
                nn.init.xavier_normal_(layer.linear.weight, gain=0.3)
                nn.init.zeros_(layer.linear.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        # --- Expression: small random (like subtle facial features) ---
        nn.init.xavier_normal_(agent.expression_projection.weight, gain=0.2)
        nn.init.zeros_(agent.expression_projection.bias)

        # --- Discrimination head: moderate init (will be trained from scratch) ---
        for layer in agent.discrimination_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    return agent


def create_agent(agent_id: int = 0,
                 perception_dim: int = 154,  # 129 base + 10 utterance slots + 10 memory dims + 5 property utterance slots
                 n_utterance_classes: int = 10,
                 seed: int = 42,
                 position: Optional[Tuple[float, float]] = None) -> CuriousAgent:
    """
    Factory: create a new agent with structured initialization.

    Each agent gets a unique seed so they develop differently
    (like siblings with the same genes but different experiences).
    """
    config = AgentConfig(
        agent_id=agent_id,
        name=f"agent_{agent_id}",
        perception_dim=perception_dim,
        n_utterance_classes=n_utterance_classes,
    )

    agent = CuriousAgent(config)
    agent = apply_structured_initialization(agent, seed=seed + agent_id * 1000)

    if position is not None:
        agent.position = np.array(position, dtype=float)
    else:
        # Random starting position, different per agent
        rng = np.random.RandomState(seed + agent_id)
        agent.position = rng.uniform(20, 80, size=2)

    return agent
