"""
Language Grounding — Phase 2
============================

Ostensive learning: the system "points at" objects and "says" their names
while agents perceive them. Words get grounded in the agent's internal
representations (encoder outputs), not raw perception vectors.

Design principles:
  - Teaching is APPROACH-ONLY: it's a bonus opportunity, not a requirement
  - Words are grounded through repeated exposure, not one-shot memorization
  - Vocabulary refreshes periodically as the encoder evolves during training
  - Naming accuracy is tested passively (no separate "test phase")
  - Agents can eventually name objects to each other (communication emergence)

Teaching curriculum follows the environment stages:
  Stage 1 (ep 0-500):    Teach 4 basic words frequently
  Stage 2 (ep 500-2000): Add 6 more words, moderate frequency
  Stage 3 (ep 2000+):    Rare teaching, test generalization
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# =============================================================================
# Object Class Registry
# =============================================================================

# All teachable object classes — must stay consistent with OBJECT_LIBRARY in world.py.
# Index position == class label passed to the discrimination head.
ALL_OBJECT_CLASSES: List[str] = [
    'apple', 'rock', 'ball', 'flower',
    'banana', 'cat', 'dog', 'water', 'fire', 'book',
]
N_OBJECT_CLASSES: int = len(ALL_OBJECT_CLASSES)
WORD_CLASS_MAP: Dict[str, int] = {word: idx for idx, word in enumerate(ALL_OBJECT_CLASSES)}


# =============================================================================
# Property Class Registry (Phase 3.5)
# =============================================================================

N_PROPERTY_CLASSES: int = 5
ALL_PROPERTY_CLASSES: List[str] = ['dangerous', 'edible', 'animate', 'warm', 'bright']
PROPERTY_WORD_MAP: Dict[str, int] = {word: idx for idx, word in enumerate(ALL_PROPERTY_CLASSES)}

# Which PROPERTY_SCHEMA dimension each property word corresponds to
PROPERTY_DIM_MAP: Dict[str, int] = {
    'dangerous': 8,
    'edible':    7,
    'animate':   6,
    'warm':      9,
    'bright':    11,
}

# Minimum property value to count as a teaching instance for that word
PROPERTY_THRESHOLDS: Dict[str, float] = {
    'dangerous': 0.5,
    'edible':    0.5,
    'animate':   0.5,
    'warm':      0.6,
    'bright':    0.6,
}


@dataclass
class TeachingConfig:
    """Hyperparameters for the ostensive teaching system."""

    # How often to teach (probability per step per agent)
    stage_1_teach_prob: float = 0.15      # Frequent early teaching
    stage_2_teach_prob: float = 0.08      # Moderate mid-training
    stage_3_teach_prob: float = 0.02      # Rare late-training (test retention)

    # Multi-exposure: how many times agent must see a word
    # before it "sticks" (averaged into the stored representation)
    min_exposures_to_ground: int = 3

    # Vocabulary refresh: re-encode stored words periodically
    # (encoder weights change during training, staling old encodings)
    refresh_interval_episodes: int = 100

    # Circular buffer depth for word memories (exposures per word slot)
    # Each word keeps the most recent `vocab_buffer_size` internal-state snapshots.
    # Averaged into a prototype — more robust to context variation than a single snapshot.
    vocab_buffer_size: int = 8

    # Naming reward: small approach-only bonus when agent names correctly
    naming_reward: float = 0.3

    # Naming test probability per step (passive testing)
    naming_test_prob: float = 0.05

    # Threshold for cosine similarity in naming
    naming_threshold: float = 0.5        # Lower than agent default (0.7) early on

    # Maximum teaching radius: how close agent must be to object
    teaching_radius: float = 15.0

    # Property vocabulary (Phase 3.5)
    property_teach_prob: float = 0.10     # Per step, per agent
    property_naming_reward: float = 0.2   # Approach-only, smaller than object reward


# Which words to teach at each curriculum stage
TEACHING_CURRICULUM = {
    1: ['apple', 'rock', 'ball', 'flower'],
    2: ['apple', 'rock', 'ball', 'flower', 'banana', 'cat', 'dog', 'water', 'fire', 'book'],
    3: ['apple', 'rock', 'ball', 'flower', 'banana', 'cat', 'dog', 'water', 'fire', 'book'],
}


@dataclass
class WordMemory:
    """
    Multi-exposure word memory with circular buffer.

    Maintains a fixed-size ring buffer of the most recent internal-state
    observations. The prototype is the mean over all buffered states.

    Advantages over a simple running sum:
      - Old, stale observations are automatically evicted as new ones arrive
      - Memory footprint is bounded (buffer_size * state_dim floats)
      - Prototype adapts to context variation: seeing "apple" in different
        lighting / positions updates the buffer rather than infinitely
        diluting a cumulative average
    """
    word: str
    base_name: str                              # Object library key (e.g., "apple")
    buffer_size: int = 8                        # Maximum buffered observations
    buffer: List[np.ndarray] = field(default_factory=list)  # Ring buffer entries
    write_pos: int = 0                          # Next write index (cycles mod buffer_size)
    exposures: int = 0                          # Total lifetime exposures (unbounded)
    grounded: bool = False                      # Has reached min_exposures threshold?
    last_refresh_episode: int = 0

    @property
    def prototype(self) -> Optional[np.ndarray]:
        """Mean internal state across all buffered observations."""
        if not self.buffer:
            return None
        return np.mean(self.buffer, axis=0)

    def add_exposure(self, internal_state: np.ndarray):
        """
        Insert a new observation into the ring buffer.

        When the buffer has space, appends. Once full, overwrites the
        oldest entry at write_pos, cycling around the buffer.
        """
        state_copy = internal_state.copy()
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(state_copy)
        else:
            self.buffer[self.write_pos] = state_copy
        self.write_pos = (self.write_pos + 1) % self.buffer_size
        self.exposures += 1

    def reset_for_refresh(self):
        """Clear buffer for re-grounding with updated encoder weights."""
        self.buffer = []
        self.write_pos = 0
        self.exposures = 0
        self.grounded = False


class OstensiveTeacher:
    """
    Automated "teacher" that performs ostensive learning.

    Each step, with some probability, the teacher:
    1. Checks what objects are near each agent
    2. Picks one and "says" its name
    3. The agent's current internal state gets associated with the word

    This simulates a parent pointing at things and naming them.
    """

    def __init__(self, config: TeachingConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Per-agent word memories: agent_id -> {word: WordMemory}
        self.word_memories: Dict[int, Dict[str, WordMemory]] = {}

        # Metrics
        self.teaching_events: List[Dict] = []
        self.naming_tests: List[Dict] = []

    def _get_teach_prob(self, stage: int) -> float:
        """Teaching probability for the current curriculum stage."""
        if stage <= 1:
            return self.config.stage_1_teach_prob
        elif stage <= 2:
            return self.config.stage_2_teach_prob
        else:
            return self.config.stage_3_teach_prob

    def _get_teachable_words(self, stage: int) -> List[str]:
        """Which words are available for teaching at this stage."""
        return TEACHING_CURRICULUM.get(stage, TEACHING_CURRICULUM[3])

    def _find_nearby_objects(self, agent_position, env_objects,
                             world_size: float) -> List[Tuple[str, str, float]]:
        """
        Find objects within teaching radius of agent.

        Returns: [(instance_key, base_name, distance), ...]
        The base_name strips instance suffixes (e.g., "apple_1" -> "apple").
        """
        nearby = []
        for key, obj in env_objects.items():
            # Compute toroidal distance
            dx = abs(obj.position[0] - agent_position[0])
            dy = abs(obj.position[1] - agent_position[1])
            dx = min(dx, world_size - dx)
            dy = min(dy, world_size - dy)
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist <= self.config.teaching_radius:
                # Extract base name (strip _N suffix for duplicates)
                base = key.split('_')[0] if '_' in key and key.split('_')[-1].isdigit() else key
                nearby.append((key, base, dist))
        return nearby

    def teach_step(self, agents, env, stage: int, episode: int):
        """
        Called each training step. May teach one word to each agent.

        Args:
            agents: list of CuriousAgent
            env: StructuredEnvironment
            stage: current curriculum stage
            episode: current episode number
        """
        teach_prob = self._get_teach_prob(stage)
        teachable_words = self._get_teachable_words(stage)

        for agent in agents:
            aid = agent.config.agent_id

            # Initialize memory dict if needed
            if aid not in self.word_memories:
                self.word_memories[aid] = {}

            # Roll the dice: do we teach this step?
            if self.rng.rand() > teach_prob:
                continue

            # Agent must have perceived (has internal state)
            if agent.internal_state is None:
                continue

            # Find nearby objects
            nearby = self._find_nearby_objects(
                agent.position, env.objects, env.world_size
            )
            if not nearby:
                continue

            # Filter to teachable words
            teachable_nearby = [
                (key, base, dist) for key, base, dist in nearby
                if base in teachable_words
            ]
            if not teachable_nearby:
                continue

            # Pick the closest teachable object
            teachable_nearby.sort(key=lambda x: x[2])
            obj_key, base_name, dist = teachable_nearby[0]

            # Get the agent's current internal state as numpy
            internal_state = agent.internal_state.detach().squeeze().numpy().copy()

            # Create or update word memory
            if base_name not in self.word_memories[aid]:
                self.word_memories[aid][base_name] = WordMemory(
                    word=base_name,
                    base_name=base_name,
                    buffer_size=self.config.vocab_buffer_size,
                )

            memory = self.word_memories[aid][base_name]
            memory.add_exposure(internal_state)

            # Once enough exposures accumulate, ground the word and train language losses
            if memory.exposures >= self.config.min_exposures_to_ground:
                if not memory.grounded:
                    memory.grounded = True
                    self.teaching_events.append({
                        'episode': episode,
                        'agent_id': aid,
                        'word': base_name,
                        'exposures': memory.exposures,
                        'event': 'grounded',
                    })

                # Update vocabulary with latest circular-buffer prototype
                agent.vocabulary[base_name] = memory.prototype.copy()

                # --- Language loss training ---
                # Naming loss: pull current encoding toward word prototype.
                # Discrimination loss: train auxiliary classifier (state → class).
                # Both shape the encoder without disrupting curiosity dynamics.
                class_idx = WORD_CLASS_MAP.get(base_name, -1)
                agent.train_language_losses(base_name, class_idx)

    def teach_property_step(self, agents, env, stage: int, episode: int):
        """
        Each step, may teach property words to agents near qualifying objects.

        A property word is taught when agent is within teaching_radius of an
        object whose property value for that dimension exceeds the threshold.
        Uses the same WordMemory / circular-buffer mechanism as object teaching.
        """
        for agent in agents:
            aid = agent.config.agent_id
            if agent.internal_state is None:
                continue
            if self.rng.rand() > self.config.property_teach_prob:
                continue

            if aid not in self.word_memories:
                self.word_memories[aid] = {}

            nearby = self._find_nearby_objects(
                agent.position, env.objects, env.world_size
            )
            if not nearby:
                continue

            internal_state = agent.internal_state.detach().squeeze().numpy().copy()

            # Track which property words have already been taught this step to avoid
            # double-counting the same internal_state when multiple nearby objects
            # qualify for the same property word.
            taught_properties: set = set()

            # For each nearby object, teach any property words it qualifies for
            for obj_key, base_name, dist in nearby:
                obj = env.objects.get(obj_key)
                if obj is None:
                    continue
                for prop_word, dim_idx in PROPERTY_DIM_MAP.items():
                    if obj.properties[dim_idx] <= PROPERTY_THRESHOLDS[prop_word]:
                        continue
                    # Skip if this property word was already taught this step
                    if prop_word in taught_properties:
                        continue
                    taught_properties.add(prop_word)
                    # This object qualifies as a teaching instance for prop_word
                    mem_key = f'prop_{prop_word}'
                    if mem_key not in self.word_memories[aid]:
                        self.word_memories[aid][mem_key] = WordMemory(
                            word=prop_word,
                            base_name=mem_key,
                            buffer_size=self.config.vocab_buffer_size,
                        )
                    memory = self.word_memories[aid][mem_key]
                    memory.add_exposure(internal_state)

                    if memory.exposures >= self.config.min_exposures_to_ground:
                        if not memory.grounded:
                            memory.grounded = True
                            self.teaching_events.append({
                                'episode': episode,
                                'agent_id': aid,
                                'word': prop_word,
                                'type': 'property',
                                'event': 'grounded',
                            })
                        agent.property_vocabulary[prop_word] = memory.prototype.copy()
                        prop_idx = PROPERTY_WORD_MAP[prop_word]
                        agent.train_property_losses(prop_word, prop_idx)

    def test_naming(self, agents, env, stage: int, episode: int) -> List[Dict]:
        """
        Passively test whether agents can name nearby objects.

        Returns list of test results for metrics.
        """
        results = []
        teachable_words = self._get_teachable_words(stage)

        for agent in agents:
            if agent.internal_state is None:
                continue
            if not agent.vocabulary:
                continue

            # Only test with some probability
            if self.rng.rand() > self.config.naming_test_prob:
                continue

            # Find what's nearby
            nearby = self._find_nearby_objects(
                agent.position, env.objects, env.world_size
            )
            if not nearby:
                continue

            # Pick closest object
            nearby.sort(key=lambda x: x[2])
            obj_key, base_name, dist = nearby[0]

            # Agent tries to name it
            guess = agent.try_to_name(
                agent.internal_state,
                threshold=self.config.naming_threshold
            )

            correct = (guess == base_name)
            result = {
                'episode': episode,
                'agent_id': agent.config.agent_id,
                'target': base_name,
                'guess': guess,
                'correct': correct,
                'distance': dist,
                'vocab_size': len(agent.vocabulary),
            }
            results.append(result)
            self.naming_tests.append(result)

        return results

    def compute_naming_reward(self, agent, env) -> float:
        """
        Small approach-only reward when agent correctly names a nearby object.

        Only fires when agent has vocabulary AND is near something nameable.
        Returns 0.0 if agent can't name or names wrong — never negative.
        """
        if agent.internal_state is None or not agent.vocabulary:
            return 0.0

        nearby = self._find_nearby_objects(
            agent.position, env.objects, env.world_size
        )
        if not nearby:
            return 0.0

        # Closest object
        nearby.sort(key=lambda x: x[2])
        obj_key, base_name, dist = nearby[0]

        guess = agent.try_to_name(
            agent.internal_state,
            threshold=self.config.naming_threshold
        )

        if guess == base_name:
            return self.config.naming_reward
        return 0.0

    def compute_property_naming_reward(self, agent, env) -> float:
        """
        Reward when agent's last utterance was a property word that correctly
        describes a nearby object.

        Returns 0.0 if no property utterance this step, or if the property
        doesn't apply to nearby objects. Never negative.
        """
        if agent.last_utterance_property is None:
            return 0.0
        if not agent.property_vocabulary:
            return 0.0

        prop_word = ALL_PROPERTY_CLASSES[agent.last_utterance_property]
        dim_idx = PROPERTY_DIM_MAP.get(prop_word)
        threshold = PROPERTY_THRESHOLDS.get(prop_word, 0.5)
        if dim_idx is None:
            return 0.0

        nearby = self._find_nearby_objects(
            agent.position, env.objects, env.world_size
        )
        for obj_key, base_name, dist in nearby:
            obj = env.objects.get(obj_key)
            if obj is not None and obj.properties[dim_idx] > threshold:
                return self.config.property_naming_reward
        return 0.0

    def refresh_vocabularies(self, agents, env, episode: int):
        """
        Re-ground all words using current encoder weights.

        The encoder evolves during training, so stored internal states
        become stale. This re-encodes known objects to keep vocabulary
        aligned with the agent's current representation space.
        """
        for agent in agents:
            aid = agent.config.agent_id
            if aid not in self.word_memories:
                continue

            refreshed_count = 0
            for word, memory in self.word_memories[aid].items():
                if not memory.grounded:
                    continue

                # Determine whether this is a property word ('prop_dangerous', etc.)
                # or an object word ('apple', 'rock', etc.).
                is_property_word = word.startswith('prop_')

                # For object words the env key base must match the word directly.
                # For property words we match against any object in the environment
                # (use the first object found — we only need an encoder state).
                if is_property_word:
                    # Use any object to get a representative encoder state.
                    any_obj = next(iter(env.objects.values()), None)
                    if any_obj is None:
                        continue
                    match_obj = any_obj
                else:
                    match_obj = None
                    for key, obj in env.objects.items():
                        base = key.split('_')[0] if '_' in key and key.split('_')[-1].isdigit() else key
                        if base == word:
                            match_obj = obj
                            break

                if match_obj is None:
                    continue

                # Get perception from object's location
                # (simulate agent being right next to it)
                flat_perc = env.get_flat_perception(
                    position=match_obj.position,
                    perception_radius=self.config.teaching_radius,
                )
                # Pad to agent's full perception_dim (adds utterance slots as zeros —
                # no agents are speaking during vocabulary refresh).
                expected_dim = agent.config.perception_dim
                if len(flat_perc) < expected_dim:
                    flat_perc = np.concatenate(
                        [flat_perc, np.zeros(expected_dim - len(flat_perc))]
                    )
                with torch.no_grad():
                    x = torch.FloatTensor(flat_perc).unsqueeze(0)
                    new_state = agent.encoder(x).squeeze().numpy()

                # Reset memory and re-accumulate
                memory.reset_for_refresh()
                memory.add_exposure(new_state)
                memory.exposures = self.config.min_exposures_to_ground  # Keep grounded
                memory.grounded = True
                memory.last_refresh_episode = episode

                if is_property_word:
                    prop_key = word[5:]  # strip 'prop_' prefix → bare property word
                    agent.property_vocabulary[prop_key] = new_state.copy()
                else:
                    agent.vocabulary[word] = new_state.copy()
                refreshed_count += 1

            if refreshed_count > 0:
                self.teaching_events.append({
                    'episode': episode,
                    'agent_id': aid,
                    'event': 'vocabulary_refreshed',
                    'n_words': refreshed_count,
                })

    def get_metrics(self) -> Dict[str, Any]:
        """Summary metrics for logging."""
        if not self.naming_tests:
            return {
                'total_teaching_events': len(self.teaching_events),
                'total_naming_tests': 0,
                'naming_accuracy': 0.0,
                'recent_accuracy': 0.0,
            }

        recent_n = 50
        recent_tests = self.naming_tests[-recent_n:]
        recent_correct = sum(1 for t in recent_tests if t['correct'])

        all_correct = sum(1 for t in self.naming_tests if t['correct'])

        return {
            'total_teaching_events': len(self.teaching_events),
            'total_naming_tests': len(self.naming_tests),
            'naming_accuracy': all_correct / len(self.naming_tests),
            'recent_accuracy': recent_correct / len(recent_tests) if recent_tests else 0.0,
        }

    def get_per_agent_metrics(self, agents) -> Dict[int, Dict]:
        """Per-agent vocabulary and naming metrics."""
        metrics = {}
        for agent in agents:
            aid = agent.config.agent_id
            agent_tests = [t for t in self.naming_tests if t['agent_id'] == aid]
            recent_tests = agent_tests[-20:]

            metrics[aid] = {
                'vocab_size': len(agent.vocabulary),
                'words_known': list(agent.vocabulary.keys()),
                'total_tests': len(agent_tests),
                'recent_accuracy': (
                    sum(1 for t in recent_tests if t['correct']) / len(recent_tests)
                    if recent_tests else 0.0
                ),
            }

            # Per-word exposure counts
            if aid in self.word_memories:
                metrics[aid]['word_exposures'] = {
                    word: mem.exposures
                    for word, mem in self.word_memories[aid].items()
                }

        return metrics
