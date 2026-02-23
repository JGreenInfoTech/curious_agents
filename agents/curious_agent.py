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
    internal_state_dim: int = 64    # Compressed internal state
    
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

    # World
    world_size: float = 100.0
    perception_radius: float = 30.0

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
    2. Builds a forward model (predicts next perception from state + action)
    3. Is rewarded ONLY for learning progress (prediction error going DOWN)
    4. Can observe other agents and is rewarded for helping them learn
    5. Tracks its own confidence (proto-metacognition)
    
    No punishment. No negative reward. Confusion is just information.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # ===== PERCEPTION ENCODER =====
        # Takes flat perception vector → compressed internal state
        self.encoder = nn.Sequential(
            NormedLinear(config.perception_dim, config.hidden_dim),
            nn.GELU(),  # Smooth activation (no dead neurons like ReLU)
            NormedLinear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            NormedLinear(config.hidden_dim, config.internal_state_dim),
            nn.Tanh(),  # Bounded internal state
        )
        
        # ===== FORWARD MODEL (World Predictor) =====
        # Given current internal state + action → predict next internal state
        # This is the core of curiosity: surprise = prediction error
        self.forward_model = nn.Sequential(
            NormedLinear(config.internal_state_dim + config.n_actions, config.hidden_dim),
            nn.GELU(),
            NormedLinear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.internal_state_dim),
            nn.Tanh(),
        )
        
        # ===== POLICY (Action Selection) =====
        # Given internal state → action preferences (softmax probabilities)
        # Total action space: n_actions (movement) + n_utterance_classes (word emissions)
        # The forward model only uses movement actions; utterance actions expand policy only.
        _total_actions = config.n_actions + config.n_utterance_classes
        self.policy = nn.Sequential(
            NormedLinear(config.internal_state_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, _total_actions),
            # No final activation — we'll use softmax at decision time
        )
        
        # ===== VALUE ESTIMATOR =====
        # Estimates expected future reward (for advantage computation)
        self.value_head = nn.Sequential(
            NormedLinear(config.internal_state_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        
        # ===== DISCRIMINATION HEAD (Auxiliary Object Classifier) =====
        # Takes internal state → object class logits.
        # Trained with cross-entropy as a parallel objective alongside curiosity.
        # Forces the encoder to produce class-separable representations without
        # replacing or disrupting the forward model / policy gradient flow.
        # Gradients flow through encoder via a dedicated language_optimizer.
        self.discrimination_head = nn.Sequential(
            nn.Linear(config.internal_state_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.n_object_classes),
        )

        # ===== OBSERVABLE STATE (what others can see about me) =====
        # A small "expression" vector that leaks some internal state
        # Think of it as body language / facial expression
        self.expression_projection = nn.Linear(config.internal_state_dim, 8)
        
        # ===== TRACKING STATE (not learned, just bookkeeping) =====
        self.position = np.array([50.0, 50.0])  # Start at center
        self.internal_state = None                # Current encoded perception
        self.prev_internal_state = None           # Previous step's state
        self.prev_prediction_error = 0.0          # For learning progress calc
        self.current_prediction_error = 0.0
        
        # Confidence tracker (exponential moving average of prediction accuracy)
        # Starts at 0.5 (uncertain). Rises as predictions improve.
        self.confidence = 0.5
        
        # Visit tracking for novelty bonus (grid-based)
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        
        # Experience buffer for learning
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

        # Language loss tracking
        self.naming_loss_history: List[float] = []
        self.discrimination_loss_history: List[float] = []
        
        # Set up optimizers
        self._setup_optimizers()
    
    def _setup_optimizers(self):
        """Three separate optimizers for three separate learning signals."""
        # Policy + value learn from intrinsic reward (REINFORCE)
        policy_params = list(self.encoder.parameters()) + \
                        list(self.policy.parameters()) + \
                        list(self.value_head.parameters()) + \
                        list(self.expression_projection.parameters())
        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=self.config.learning_rate
        )

        # Forward model learns from prediction error (self-supervised)
        self.forward_model_optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=self.config.forward_model_lr
        )

        # Language losses shape encoder + discrimination head via supervised signal.
        # Encoder params are shared with policy_optimizer — each optimizer manages
        # its own zero_grad/step cycle so they don't interfere with each other.
        language_params = list(self.encoder.parameters()) + \
                          list(self.discrimination_head.parameters())
        self.language_optimizer = torch.optim.Adam(
            language_params, lr=self.config.learning_rate
        )
    
    # =========================================================================
    # Core Loop: Perceive → Predict → Act → Learn
    # =========================================================================
    
    def perceive(self, flat_perception: np.ndarray) -> torch.Tensor:
        """
        Encode raw perception into internal state.
        
        This is what the agent "experiences" — a compressed representation
        of what's around it. Not the raw properties, but a learned encoding.
        """
        x = torch.FloatTensor(flat_perception).unsqueeze(0)  # (1, perception_dim)
        self.prev_internal_state = self.internal_state
        self.internal_state = self.encoder(x)  # (1, internal_state_dim)
        return self.internal_state
    
    def decide_action(self, temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
        """
        Choose an action based on current internal state.
        
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
            word_idx = action - self.config.n_actions
            self.last_utterance_class = (
                word_idx if word_idx < self.config.n_utterance_classes else None
            )

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
    
    def compute_prediction_error(self, action: int) -> float:
        """
        How wrong was my prediction about what would happen?
        
        This is INFORMATION, not pain. It tells the agent where
        its world model is inaccurate.
        
        Returns: scalar prediction error (MSE between predicted and actual)
        """
        if self.prev_internal_state is None:
            return 0.0

        # Map utterance actions to "stay" for forward model — utterances don't move the agent,
        # so the physical forward model only needs to distinguish movement directions.
        forward_action = action if action < self.config.n_actions else (self.config.n_actions - 1)
        action_onehot = torch.zeros(1, self.config.n_actions)
        action_onehot[0, forward_action] = 1.0

        # What did I predict would happen?
        forward_input = torch.cat([self.prev_internal_state, action_onehot], dim=-1)
        predicted_next = self.forward_model(forward_input)

        # What actually happened?
        actual_next = self.internal_state.detach()
        
        # Prediction error (MSE)
        error = F.mse_loss(predicted_next, actual_next).item()
        
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
        Train the forward model on (state, action) → next_state prediction.
        
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

        forward_input = torch.cat([self.prev_internal_state.detach(), action_onehot], dim=-1)
        predicted_next = self.forward_model(forward_input)
        actual_next = self.internal_state.detach()
        
        loss = F.mse_loss(predicted_next, actual_next)
        
        self.forward_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
        self.forward_model_optimizer.step()
        
        return loss.item()
    
    # =========================================================================
    # Language Loss Training (Naming + Discrimination — parallel objectives)
    # =========================================================================

    def train_language_losses(self, word: str, class_idx: int) -> Tuple[float, float]:
        """
        Train naming alignment loss and discrimination loss in a single backward pass.

        Naming loss (MSE):
          Pulls current encoder output toward the stored word prototype.
          When the agent sees an apple and is told "apple", the encoding
          should converge toward the prototype averaged across prior apple
          sightings. This makes representations consistent across contexts.

        Discrimination loss (cross-entropy):
          Auxiliary classifier: internal_state → object class prediction.
          Forces the encoder to produce class-separable representations.
          The discrimination_head is a separate small network — its gradients
          flow through the shared encoder but the head itself isn't used for
          policy decisions or curiosity computation.

        Both losses share language_optimizer (encoder + discrimination_head).
        Policy and forward-model optimizers are unaffected.

        Args:
            word: the word that was just taught (must exist in self.vocabulary
                  for naming loss to fire; discrimination loss only needs class_idx)
            class_idx: integer class label (from WORD_CLASS_MAP); -1 to skip discrimination

        Returns:
            (naming_loss_value, discrimination_loss_value) as plain floats for logging.
        """
        if self.internal_state is None:
            return 0.0, 0.0

        has_loss = False
        total_loss = None
        naming_val = 0.0
        disc_val = 0.0

        # --- Naming alignment loss ---
        # Only fires once the word has a stable prototype in vocabulary.
        if word in self.vocabulary:
            prototype = torch.FloatTensor(self.vocabulary[word]).unsqueeze(0)
            naming_loss = F.mse_loss(self.internal_state, prototype)
            total_loss = self.config.naming_loss_weight * naming_loss
            naming_val = naming_loss.item()
            has_loss = True

        # --- Discrimination loss ---
        # Fires whenever a valid class index is provided.
        if 0 <= class_idx < self.config.n_object_classes:
            logits = self.discrimination_head(self.internal_state)
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
                list(self.encoder.parameters()) + list(self.discrimination_head.parameters()),
                1.0,
            )
            self.language_optimizer.step()

        self.naming_loss_history.append(naming_val)
        self.discrimination_loss_history.append(disc_val)
        return naming_val, disc_val

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
    
    def store_experience(self, log_prob: torch.Tensor, reward: float, 
                         state: torch.Tensor, action: Optional[int] = None):
        """Buffer an experience for batch policy update.
        
        Stores state and action (detached from graph) so we can
        recompute log_prob fresh during updates. This avoids stale
        gradient references when the policy weights change between
        collection and update.
        """
        self.experience_buffer.append({
            'action': action,
            'reward': reward,
            'state': state.detach(),
        })
        
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def update_policy(self, batch_size: int = 32):
        """
        REINFORCE policy gradient update.
        
        Recomputes log_probs from current policy weights to avoid
        stale autograd graph references. Uses value baseline to
        reduce variance. Only positive rewards, so gradients push
        toward actions that led to learning or helping.
        """
        if len(self.experience_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for exp in batch:
            state = exp['state']  # Detached snapshot

            # Recompute log_prob from CURRENT policy (fresh graph)
            logits = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(exp['action'])).squeeze()

            # Value estimate (baseline)
            value = self.value_head(state)
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
        ) / batch_size
        
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
        Associate a word with the current internal state.
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
    
    This gives the agent a "developmental scaffold" without training it.
    """
    torch.manual_seed(seed)
    
    with torch.no_grad():
        # --- Encoder: Xavier init with slight structure ---
        for layer in agent.encoder:
            if isinstance(layer, NormedLinear):
                nn.init.xavier_normal_(layer.linear.weight, gain=0.8)
                nn.init.zeros_(layer.linear.bias)
        
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
                total_actions = agent.config.n_actions + agent.config.n_utterance_classes
                movement_bias = torch.tensor([0.1, 0.1, 0.1, 0.1, -0.1])
                utterance_bias = torch.full((agent.config.n_utterance_classes,), -0.2)
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
                 perception_dim: int = 139,  # 129 base + 10 utterance slots
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
