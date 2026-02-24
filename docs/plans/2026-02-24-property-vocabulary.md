# Property Vocabulary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give agents 5 property words (`dangerous`, `edible`, `animate`, `warm`, `bright`) grounded in the 15D property vector, expanding the policy to 20 actions and perception to 154D, so agents can communicate about object qualities not just object identity.

**Architecture:** Extend `OstensiveTeacher` to also teach property words when agents encounter high-property objects (same circular-buffer WordMemory mechanism). Add 5 property utterance actions to the policy (20 total: 5 move + 10 obj + 5 prop) and 5D property utterance perception slots (154D total: 129 base + 10 obj slots + 5 prop slots + 10 obj memory). Add a binary `property_discrimination_head` (64→16→5, Sigmoid) to CuriousAgent, trained with BCELoss. Add property communicative reward (+0.3) when agent A's property utterance matches nearby agent B's property_discrimination_head prediction.

**Tech Stack:** PyTorch, NumPy, existing CuriousAgent / OstensiveTeacher / Trainer / StructuredEnvironment

---

### Background: Property Schema and Thresholds

From `environment/world.py` PROPERTY_SCHEMA (0-indexed dimensions):
- dim 6: `animate`     — cat(1.0), dog(1.0), fire(0.5), water(0.3)
- dim 7: `edible`      — apple(1.0), banana(1.0), water(0.5)
- dim 8: `dangerous`   — fire(0.9), dog(0.2), cat(0.1)
- dim 9: `warm`        — fire(0.9), cat(0.5)
- dim 11: `bright`     — fire(0.9), flower(~0.8), apple(0.5)

Teaching threshold > 0.5 for animate/edible/dangerous; > 0.6 for warm/bright (fewer qualifying objects = more informative signal).

---

### Task 1: Property constants in language_grounding.py

**Files:**
- Modify: `training/language_grounding.py` (after `WORD_CLASS_MAP` at line 39)

Add after `WORD_CLASS_MAP`:
```python
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
```

Add to `TeachingConfig` (after `teaching_radius`):
```python
# Property vocabulary (Phase 3.5)
property_teach_prob: float = 0.10     # Per step, per agent
property_naming_reward: float = 0.2   # Approach-only, smaller than object reward
```

**Verify:** `python -c "from training.language_grounding import N_PROPERTY_CLASSES, ALL_PROPERTY_CLASSES; print(N_PROPERTY_CLASSES, ALL_PROPERTY_CLASSES)"` → `5 ['dangerous', 'edible', 'animate', 'warm', 'bright']`

---

### Task 2: Property teaching in OstensiveTeacher

**Files:**
- Modify: `training/language_grounding.py`

Add `teach_property_step()` method to `OstensiveTeacher` (after `teach_step`, before `test_naming`):

```python
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

        # For each nearby object, teach any property words it qualifies for
        for obj_key, base_name, dist in nearby:
            obj = env.objects.get(obj_key)
            if obj is None:
                continue
            for prop_word, dim_idx in PROPERTY_DIM_MAP.items():
                if obj.properties[dim_idx] <= PROPERTY_THRESHOLDS[prop_word]:
                    continue
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
```

Add `compute_property_naming_reward()` method to `OstensiveTeacher` (after `compute_naming_reward`):

```python
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
```

**Verify:** `python -c "from training.language_grounding import OstensiveTeacher, TeachingConfig; t=OstensiveTeacher(TeachingConfig()); print('teach_property_step' in dir(t))"` → `True`

---

### Task 3: CuriousAgent property architecture

**Files:**
- Modify: `agents/curious_agent.py`

**Step 1:** Add to `AgentConfig` (after `n_object_classes`):
```python
n_property_utterances: int = 5        # Property word emission actions (Phase 3.5)
n_property_classes: int = 5           # For property discrimination head
```

Update `AgentConfig.perception_dim` default:
```python
perception_dim: int = 129       # Base; trainer sets full dim (154 with all slots)
```
(No change needed if the default is already overridden by trainer. But note: `create_agent()` at bottom of file uses `perception_dim=149` — update that call-site default to `154`.)

**Step 2:** In `CuriousAgent.__init__`, update total actions computation (find `_total_actions = config.n_actions + config.n_utterance_classes`):
```python
_total_actions = config.n_actions + config.n_utterance_classes + config.n_property_utterances
```

**Step 3:** Add `property_discrimination_head` to `CuriousAgent.__init__` (after `discrimination_head`):
```python
self.property_discrimination_head = nn.Sequential(
    nn.Linear(config.internal_state_dim, config.hidden_dim // 4),
    nn.GELU(),
    nn.Linear(config.hidden_dim // 4, config.n_property_classes),
    nn.Sigmoid(),   # Binary per property — multiple can be true simultaneously
)
```

**Step 4:** Add `property_vocabulary` and `last_utterance_property` to `CuriousAgent.__init__` (near `self.vocabulary` and `self.last_utterance_class`):
```python
self.property_vocabulary: Dict[str, np.ndarray] = {}
self.last_utterance_property: Optional[int] = None  # None if not a property utterance
```

**Step 5:** Add `property_discrimination_head` to `language_optimizer`. Find the language_optimizer definition and add its parameters:
```python
self.language_optimizer = torch.optim.Adam(
    list(self.encoder.parameters())
    + list(self.discrimination_head.parameters())
    + list(self.property_discrimination_head.parameters()),
    lr=config.language_lr,
)
```

**Step 6:** Update `execute_action()`. Find the current utterance handling block:
```python
word_idx = action - self.config.n_actions
self.last_utterance_class = (
    word_idx if word_idx < len(ALL_OBJECT_CLASSES) else None
)
```
Replace with:
```python
action_offset = action - self.config.n_actions
if action_offset < self.config.n_utterance_classes:
    # Object utterance
    self.last_utterance_class = action_offset
    self.last_utterance_property = None
else:
    # Property utterance
    self.last_utterance_class = None
    self.last_utterance_property = action_offset - self.config.n_utterance_classes
```

**Step 7:** Add `train_property_losses()` method (after `train_language_losses`):
```python
def train_property_losses(self, property_word: str, property_idx: int) -> float:
    """
    Train property discrimination head on the current internal state.

    Uses BCELoss: the head should output 1.0 for property_idx (the property
    being taught) and anything for the rest. We only supervise the one property
    being taught per call — not the full vector.

    Returns: BCE loss value (float) for logging.
    """
    if self.internal_state is None:
        return 0.0

    # Build binary target: 1.0 for this property, 0.0 for others
    target = torch.zeros(1, self.config.n_property_classes)
    if 0 <= property_idx < self.config.n_property_classes:
        target[0, property_idx] = 1.0

    pred = self.property_discrimination_head(self.internal_state)  # (1, 5)
    loss = torch.nn.functional.binary_cross_entropy(pred, target) * 0.2

    self.language_optimizer.zero_grad()
    loss.backward(retain_graph=True)
    self.language_optimizer.step()
    return loss.item()
```

**Step 8:** In `apply_structured_initialization()`, find where utterance action biases are set and extend to cover property utterances:
```python
# Property utterance actions: same slight negative bias as object utterances
n_obj_utt = agent.config.n_utterance_classes
n_prop_utt = agent.config.n_property_utterances
total_utt = n_obj_utt + n_prop_utt
utterance_bias = torch.full((total_utt,), -0.2)
# (replace existing n_utterance_classes-only bias logic with total_utt)
```

**Step 9:** Update `create_agent()` at the bottom of the file:
```python
def create_agent(..., perception_dim: int = 154, ...):
```

**Verify:** `python -c "from agents.curious_agent import create_agent; a=create_agent(0); print(sum(p.numel() for p in a.parameters()), 'params')"` — Should be ~95K (up from 93K).

---

### Task 4: world.py perception dim update

**Files:**
- Modify: `environment/world.py` `get_perception_dim()`

Add `n_property_utterance_classes: int = 0` parameter:
```python
def get_perception_dim(self, max_objects: int = 8, n_utterance_classes: int = 0,
                        n_memory_classes: int = 0,
                        n_property_utterance_classes: int = 0) -> int:
    return (max_objects * (1 + PROPERTY_DIM) + 1
            + n_utterance_classes
            + n_memory_classes
            + n_property_utterance_classes)
```

---

### Task 5: Trainer wiring

**Files:**
- Modify: `training/trainer.py`

**Step 1:** Add imports at top:
```python
from training.language_grounding import (
    ...,  # existing
    N_PROPERTY_CLASSES, ALL_PROPERTY_CLASSES, PROPERTY_WORD_MAP,
)
```

**Step 2:** Add to `TrainerConfig`:
```python
property_comm_reward: float = 0.3    # When property utterance matches partner's prediction
```

**Step 3:** Update `perception_dim` in `Trainer.__init__`:
```python
perception_dim = self.env.get_perception_dim(
    n_utterance_classes=N_OBJECT_CLASSES,
    n_memory_classes=N_OBJECT_CLASSES,
    n_property_utterance_classes=N_PROPERTY_CLASSES,
)
```

**Step 4:** Update the print statement:
```python
print(f"  {config.n_agents} agents, {perception_dim}D perception "
      f"(129 env + {N_OBJECT_CLASSES} obj slots + {N_PROPERTY_CLASSES} prop slots "
      f"+ {N_OBJECT_CLASSES} memory dims)")
```

**Step 5:** Add `_build_property_utterance_slots()` helper (after `_build_utterance_slots`):
```python
def _build_property_utterance_slots(self, agent: CuriousAgent,
                                     utterances: Dict) -> np.ndarray:
    """5D slot: index i is set to 1.0 if any nearby agent uttered property i this step."""
    slot = np.zeros(N_PROPERTY_CLASSES, dtype=np.float32)
    for other in self.agents:
        if other.config.agent_id == agent.config.agent_id:
            continue
        prop_idx = utterances.get(other.config.agent_id, {}).get('property')
        if prop_idx is not None:
            dist = np.linalg.norm(agent.position - other.position)
            if dist <= self.config.helping_radius:
                slot[prop_idx] = 1.0
    return slot
```

**Step 6:** Update `_build_perception()` to concatenate property slots:
```python
def _build_perception(self, agent: CuriousAgent, utterances: Dict) -> np.ndarray:
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
    return np.concatenate([base, utterance_slots, property_slots, memory_vec])
```

**Step 7:** Update utterance tracking in `run_step`. Find Phase 2.5 where `step_utterances` and `self.utterance_log` are built. Currently only object utterances are tracked. Update:

In the step utterances dict (wherever `last_utterance_class` is read), also record `last_utterance_property`:
```python
step_utterances[agent.config.agent_id] = {
    'class': agent.last_utterance_class,
    'property': agent.last_utterance_property,
}
```
(Check current structure of `step_utterances` in run_step — adapt accordingly.)

Also log property utterances to `utterance_log`:
```python
if agent.last_utterance_property is not None:
    self.utterance_log.append({
        'agent_id': agent.config.agent_id,
        'class_idx': None,
        'property_idx': agent.last_utterance_property,
        'step': self.episode_step,
        'type': 'property',
    })
```

**Step 8:** Add property communicative reward to Phase 6.5. After the existing `step_comm_rewards` computation, add:
```python
# Property communicative reward: agent A says property P, nearby agent B's
# property_discrimination_head predicts P with > 0.5 confidence.
for agent_a in self.agents:
    if agent_a.last_utterance_property is None:
        continue
    prop_idx = agent_a.last_utterance_property
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
```

**Step 9:** Add property teaching call in `run_step`. Find where `self.teacher.teach_step(...)` is called and add:
```python
self.teacher.teach_property_step(self.agents, self.env,
                                  self.current_stage, self.current_episode)
```

**Step 10:** Add property naming reward in Phase 7. Find where `naming_reward` is computed:
```python
naming_reward = self.teacher.compute_naming_reward(agent, self.env)
prop_naming_reward = self.teacher.compute_property_naming_reward(agent, self.env)
reward += naming_reward + prop_naming_reward + comm_reward
```

**Step 11:** Update `save_checkpoint` per-agent dict:
```python
'property_vocabulary': {k: v.tolist() for k, v in agent.property_vocabulary.items()},
```

**Step 12:** Update `load_checkpoint` per-agent restore:
```python
if 'property_vocabulary' in data:
    agent.property_vocabulary = {k: np.array(v) for k, v in data['property_vocabulary'].items()}
```

Update warning message: `"(perception_dim may have changed 149->154 for Phase 3.5)"`

---

### Task 6: Episode metric tracking for property utterances

**Files:**
- Modify: `training/trainer.py` `collect_metrics()`
- Modify: `analyze_run.py`

In `collect_metrics()`, add to each agent's dict:
```python
'property_utterance_count': sum(
    1 for e in self.utterance_log
    if e['agent_id'] == agent.config.agent_id and e.get('type') == 'property'
),
'property_utterance_rate': sum(
    1 for e in self.utterance_log
    if e['agent_id'] == agent.config.agent_id and e.get('type') == 'property'
) / max(1, self.config.steps_per_episode),
'property_vocab_size': len(agent.property_vocabulary),
```

In `analyze_run.py`, extend `print_comm()` to add per-agent `pvoc` and `putt%` columns:
```python
pvoc    = d.get('property_vocab_size', 0)
putt    = d.get('property_utterance_rate', 0.0)
cols.append(f'{utt_rate:>7.3f} {putt:>5.3f} {ref_r:>5.2f} {jnt_r:>5.2f} {mem:>3d} {pvoc:>4d}')
```
Update header to match.

---

### Task 7: Smoke test + verify

```bash
python run.py --test
```

Expected output includes:
```
154D perception (129 env + 10 obj slots + 5 prop slots + 10 obj memory)
[OK] Smoke test passed -- all components functional
```

Also verify property fields appear in log JSON:
```bash
python -c "
import json
with open('logs/metrics_ep50.json') as f:
    d = json.load(f)
a = d[-1]['agents']['0']
print('property_vocab_size:', a.get('property_vocab_size', 'MISSING'))
print('property_utterance_rate:', a.get('property_utterance_rate', 'MISSING'))
"
```

---

### Task 8: Docs update

Update in this order:
1. `memory/MEMORY.md` — add Phase 3.5 section: property constants, property_discrimination_head, property_vocabulary dict, 154D perception
2. `README.md` — update perception dim (149→154), add property vocab to reward architecture, update Phases table (Phase 3.5 Active), update Design Decisions table
3. `CLAUDE_SESSION_GUIDE.md` — add Phase 3.5 journal entry + update constants table
