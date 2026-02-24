# Spatial Memory + Communication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give agents persistent spatial memory (what they've seen, how novel it was) and the conditions for genuine inter-agent communication (spatial asymmetry + referral reward).

**Architecture:** SpatialMemory class stored per-agent and persisted in checkpoints. Perception dim expands 139D→149D (adding 10 decayed-salience memory dims). Trainer tracks utterance_log per episode to detect and reward successful information referrals. Asymmetric episode starts and directed discovery setup scaffold early communication.

**Tech Stack:** PyTorch, NumPy, existing CuriousAgent / Trainer / StructuredEnvironment

---

### Task 1: SpatialMemory class + AgentConfig additions

**Files:**
- Modify: `agents/curious_agent.py`

Add after the imports, before `AgentConfig`:

```python
@dataclass
class SpatialMemory:
    """Per-agent persistent memory of object encounters, decaying over episodes."""
    entries: Dict[int, Dict] = field(default_factory=dict)
    # entries[class_idx] = {
    #   'salience': float,     prediction_error at encounter
    #   'episode': int,        episode last seen
    #   'position': list,      [x, y]
    #   'times_visited': int,
    # }

    def update(self, class_idx: int, salience: float, episode: int, position: np.ndarray):
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
```

Add to AgentConfig (Learning section):
```python
memory_decay_horizon: int = 20        # Episodes for salience to decay to ~37%
memory_salience_threshold: float = 0.01  # Min prediction error to update memory
```

Change AgentConfig perception_radius: `30.0 → 15.0`

Add to CuriousAgent.__init__ (tracking vars section):
```python
self.spatial_memory = SpatialMemory()
self.current_episode: int = 0
```

Add new method to CuriousAgent:
```python
def update_memory(self, nearby_object_classes: List[Tuple[int, np.ndarray]],
                  prediction_error: float, episode: int):
    """Update spatial memory after a prediction error is computed."""
    if prediction_error < self.config.memory_salience_threshold:
        return
    for class_idx, position in nearby_object_classes:
        self.spatial_memory.update(class_idx, prediction_error, episode, position)
```

---

### Task 2: world.py + language_grounding.py radius reduction

**Files:**
- Modify: `environment/world.py` line ~382
- Modify: `training/language_grounding.py` TeachingConfig

world.py — update get_perception_dim:
```python
def get_perception_dim(self, max_objects: int = 8, n_utterance_classes: int = 0,
                        n_memory_classes: int = 0) -> int:
    return max_objects * (1 + PROPERTY_DIM) + 1 + n_utterance_classes + n_memory_classes
```

language_grounding.py — TeachingConfig:
```python
teaching_radius: float = 15.0   # was 25.0
```

---

### Task 3: Trainer changes — perception dim + helpers

**Files:**
- Modify: `training/trainer.py`

TrainerConfig additions:
```python
perception_radius: float = 15.0         # was 30.0
referral_reward: float = 0.4
joint_curiosity_bonus: float = 0.15
directed_discovery_prob: float = 0.20
```

Trainer.__init__ — perception_dim:
```python
perception_dim = self.env.get_perception_dim(
    n_utterance_classes=N_OBJECT_CLASSES,
    n_memory_classes=N_OBJECT_CLASSES,
)
```

Trainer.__init__ — new tracking vars:
```python
self.utterance_log: List[Dict] = []   # {agent_id, class_idx, step}
self.episode_step: int = 0
self._pre_step_salience: Dict[int, Dict[int, float]] = {}
```

Trainer.__init__ — update print:
```python
print(f"  {config.n_agents} agents, {perception_dim}D perception "
      f"(129 env + {N_OBJECT_CLASSES} utterance slots + {N_OBJECT_CLASSES} memory dims)")
```

New helper `_build_perception`:
```python
def _build_perception(self, agent: CuriousAgent, utterances: Dict) -> np.ndarray:
    base = self.env.get_flat_perception(
        position=tuple(agent.position),
        perception_radius=self.config.perception_radius,
    )
    utterance_slots = self._build_utterance_slots(agent, utterances)
    memory_vec = agent.spatial_memory.to_vec(
        current_episode=self.current_episode,
        n_classes=N_OBJECT_CLASSES,
        decay_horizon=agent.config.memory_decay_horizon,
    )
    return np.concatenate([base, utterance_slots, memory_vec])
```

New helper `_get_nearby_object_classes`:
```python
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
            result.append((class_idx, obj.position.copy()))
    return result
```

New helper `_reset_agent_positions`:
```python
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
        agent.position_history = [tuple(agent.position.copy())]
```

New helper `_setup_directed_discovery`:
```python
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
                knower.position = (obj.position + offset) % self.env.world_size
                knower.position_history = [tuple(knower.position.copy())]
                return
```

---

### Task 4: run_step additions

**Files:**
- Modify: `training/trainer.py` run_step

Replace Phase 1 + 4 perceive calls to use `_build_perception`.

After Phase 2 (actions executed), add Phase 2.5:
```python
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
        })
self.episode_step += 1
```

After Phase 4 (prediction errors computed), add Phase 4.5:
```python
# --- Phase 4.5: Update spatial memory ---
for agent in self.agents:
    nearby_classes = self._get_nearby_object_classes(agent)
    agent.update_memory(nearby_classes, agent.current_prediction_error,
                        self.current_episode)
```

After Phase 6.5 (communicative reward), add Phase 6.6 + 6.7:
```python
# --- Phase 6.6: Referral reward ---
# Agent A said word W recently; Agent B just found class W it didn't know about.
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
            if entry['class_idx'] != class_idx:
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
```

In Phase 7, add referral + joint rewards to the reward total:
```python
reward += naming_reward + comm_reward
reward += referral_rewards.get(agent.config.agent_id, 0.0)
reward += joint_rewards.get(agent.config.agent_id, 0.0)
```

---

### Task 5: run_episode + checkpoint updates

**Files:**
- Modify: `training/trainer.py`

run_episode — replace `reset_position` loop with:
```python
self._reset_agent_positions(episode)
self._setup_directed_discovery(episode)
self.prev_utterances = {}
self.utterance_log = []
self.episode_step = 0
for agent in self.agents:
    agent.current_episode = episode
    agent.reset_visit_counts()   # already called by reset_position internally? Check.
```

Note: check if reset_position does more than set position. If so keep calling it and just override position after.

save_checkpoint — add to each agent's dict:
```python
'spatial_memory': agent.spatial_memory.serialize(),
```

load_checkpoint — after restoring vocabulary, add:
```python
if 'spatial_memory' in data:
    agent.spatial_memory = SpatialMemory.deserialize(data['spatial_memory'])
agent.current_episode = self.current_episode
```

Also update load_checkpoint warning message:
```python
f"(perception_dim may have changed 139->149 for Phase 4). "
```

---

### Task 6: Smoke test + verify

```bash
python run.py --test
```

Expected: `[OK] Smoke test passed -- all components functional`
Verify print shows `149D perception (129 env + 10 utterance slots + 10 memory dims)`

---

### Task 7: Docs update

Update README.md, CLAUDE_SESSION_GUIDE.md, MEMORY.md with:
- perception_radius 30→15
- perception_dim 139→149 (+10 memory dims)
- SpatialMemory class
- Referral reward (+0.4)
- Joint curiosity bonus (+0.15)
- Asymmetric starts
- Directed discovery setup (20%)
