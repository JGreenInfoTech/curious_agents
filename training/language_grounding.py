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

    # Naming reward: small approach-only bonus when agent names correctly
    naming_reward: float = 0.3

    # Naming test probability per step (passive testing)
    naming_test_prob: float = 0.05

    # Threshold for cosine similarity in naming
    naming_threshold: float = 0.5        # Lower than agent default (0.7) early on

    # Maximum teaching radius: how close agent must be to object
    teaching_radius: float = 25.0


# Which words to teach at each curriculum stage
TEACHING_CURRICULUM = {
    1: ['apple', 'rock', 'ball', 'flower'],
    2: ['apple', 'rock', 'ball', 'flower', 'banana', 'cat', 'dog', 'water', 'fire', 'book'],
    3: ['apple', 'rock', 'ball', 'flower', 'banana', 'cat', 'dog', 'water', 'fire', 'book'],
}


@dataclass
class WordMemory:
    """
    Multi-exposure word memory.

    Rather than storing a single snapshot, we accumulate multiple
    internal-state observations and average them. This creates a
    more robust, prototype-like representation — similar to how
    humans form category prototypes from multiple exemplars.
    """
    word: str
    base_name: str                              # Object library key (e.g., "apple")
    exposures: int = 0
    state_accumulator: Optional[np.ndarray] = None  # Running sum of internal states
    grounded: bool = False                      # Has enough exposures?
    last_refresh_episode: int = 0

    @property
    def prototype(self) -> Optional[np.ndarray]:
        """Average internal state across all exposures."""
        if self.exposures == 0 or self.state_accumulator is None:
            return None
        return self.state_accumulator / self.exposures

    def add_exposure(self, internal_state: np.ndarray):
        """Add a new observation of this word's referent."""
        if self.state_accumulator is None:
            self.state_accumulator = np.zeros_like(internal_state)
        self.state_accumulator += internal_state
        self.exposures += 1

    def reset_for_refresh(self):
        """Clear accumulated states for re-grounding with updated encoder."""
        self.state_accumulator = None
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
                )

            memory = self.word_memories[aid][base_name]
            memory.add_exposure(internal_state)

            # Check if word is now grounded (enough exposures)
            if (memory.exposures >= self.config.min_exposures_to_ground
                    and not memory.grounded):
                memory.grounded = True
                # Write the prototype into the agent's vocabulary
                agent.vocabulary[base_name] = memory.prototype.copy()
                self.teaching_events.append({
                    'episode': episode,
                    'agent_id': aid,
                    'word': base_name,
                    'exposures': memory.exposures,
                    'event': 'grounded',
                })
            elif memory.grounded:
                # Already grounded — update the prototype (continuous refinement)
                agent.vocabulary[base_name] = memory.prototype.copy()

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

                # Find an instance of this object in the environment
                for key, obj in env.objects.items():
                    base = key.split('_')[0] if '_' in key and key.split('_')[-1].isdigit() else key
                    if base == word:
                        # Get perception from object's location
                        # (simulate agent being right next to it)
                        flat_perc = env.get_flat_perception(
                            position=obj.position,
                            perception_radius=self.config.teaching_radius,
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

                        agent.vocabulary[word] = new_state.copy()
                        refreshed_count += 1
                        break

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
