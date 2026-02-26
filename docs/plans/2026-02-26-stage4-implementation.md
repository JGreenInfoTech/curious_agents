# Stage 4 Reference Game — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement Stage 4 reference game: scout/runner role pairs use combined object+property words to disambiguate variant objects, with full per-episode observability logged to JSON.

**Architecture:** Six tasks in dependency order. Tasks 1–5 touch `trainer.py`, `world.py`, `agents/curious_agent.py`. Task 6 is purely `analyze_run.py`. Fresh training run required (perception_dim 154→164). Smoke test verifies each task before moving on.

**Tech Stack:** Python, PyTorch, NumPy. No new dependencies.

---

## Context

Read the design doc first: `docs/plans/2026-02-26-stage4-reference-game-design.md`

Key constants (from `training/language_grounding.py`):
- `N_OBJECT_CLASSES = 10`, `ALL_OBJECT_CLASSES` = list of 10 class names
- `N_PROPERTY_CLASSES = 5`, `ALL_PROPERTY_CLASSES` = `['dangerous','edible','animate','warm','bright']`
- `WORD_CLASS_MAP` = dict mapping class name → index
- `PROPERTY_DIM_MAP` = dict mapping property word → 15D property vector dim index

Current `perception_dim = 154`:
- 129 base env (8 objects × 16D + 1)
- +10 object utterance slots
- +5 property utterance slots
- +10 memory dims

After this plan: `perception_dim = 164` (+10D goal class token).

**Smoke test command:** `python run.py --test` (runs 50 episodes, 2 agents)

---

## Task 1: Goal token — perception dim +10D

**Files:**
- Modify: `environment/world.py` — `get_perception_dim()` lines 416–434
- Modify: `training/trainer.py` — `__init__()` lines 130–150, `_build_perception()` lines 288–301

### Step 1: Add `n_goal_classes` to `get_perception_dim()` in `world.py`

Find `get_perception_dim` (line 416). Replace:

```python
def get_perception_dim(self, max_objects: int = 8, n_utterance_classes: int = 0,
                       n_memory_classes: int = 0,
                       n_property_utterance_classes: int = 0) -> int:
    """Total dimension of flattened perception vector.

    Args:
        n_utterance_classes: number of word-emission slots appended by the trainer
                             for agent-to-agent communication (Phase 3). Default 0
                             keeps backward compatibility.
        n_memory_classes: number of spatial-memory dims appended by the trainer
                          (Phase 4). Default 0 keeps backward compatibility.
        n_property_utterance_classes: number of property utterance slots appended
                                      by the trainer (Phase 3.5). Default 0 keeps
                                      backward compatibility.
    """
    return (max_objects * (1 + PROPERTY_DIM) + 1
            + n_utterance_classes
            + n_memory_classes
            + n_property_utterance_classes)
```

With:

```python
def get_perception_dim(self, max_objects: int = 8, n_utterance_classes: int = 0,
                       n_memory_classes: int = 0,
                       n_property_utterance_classes: int = 0,
                       n_goal_classes: int = 0) -> int:
    """Total dimension of flattened perception vector.

    Args:
        n_utterance_classes: word-emission slots for agent-to-agent communication (Phase 3).
        n_memory_classes: spatial-memory dims (Phase 4).
        n_property_utterance_classes: property utterance slots (Phase 3.5).
        n_goal_classes: goal class token dims for reference game runner (Stage 4).
                        Always present (zeros for non-runner agents).
    """
    return (max_objects * (1 + PROPERTY_DIM) + 1
            + n_utterance_classes
            + n_memory_classes
            + n_property_utterance_classes
            + n_goal_classes)
```

### Step 2: Add `_goal_class_vecs` to `Trainer.__init__()`

In `trainer.py`, find the block after `self.hidden_states` init (around line 148). Add after the hidden_states dict:

```python
# Stage 4: Per-agent goal class vector for reference game runner.
# 10D one-hot indicating target class. Zeros for scout, free agent,
# and all agents during normal (non-reference-game) episodes.
self._goal_class_vecs: Dict[int, np.ndarray] = {
    i: np.zeros(N_OBJECT_CLASSES, dtype=np.float32)
    for i in range(config.n_agents)
}

# Stage 4: Reference game state for current episode (for metrics logging).
self._ref_game_state: Dict = {'active': False}
```

### Step 3: Update `get_perception_dim()` call in `Trainer.__init__()`

Find (lines 130–134):
```python
perception_dim = self.env.get_perception_dim(
    n_utterance_classes=N_OBJECT_CLASSES,
    n_memory_classes=N_OBJECT_CLASSES,
    n_property_utterance_classes=N_PROPERTY_CLASSES,
)
```

Replace with:
```python
perception_dim = self.env.get_perception_dim(
    n_utterance_classes=N_OBJECT_CLASSES,
    n_memory_classes=N_OBJECT_CLASSES,
    n_property_utterance_classes=N_PROPERTY_CLASSES,
    n_goal_classes=N_OBJECT_CLASSES,
)
```

### Step 4: Update `_build_perception()` to append goal vec

Find `_build_perception()` (lines 288–301). Replace the return line:

```python
return np.concatenate([base, utterance_slots, property_slots, memory_vec])
```

With:

```python
goal_vec = self._goal_class_vecs.get(
    agent.config.agent_id,
    np.zeros(N_OBJECT_CLASSES, dtype=np.float32)
)
return np.concatenate([base, utterance_slots, property_slots, memory_vec, goal_vec])
```

### Step 5: Update the init print statement

Find (line 196):
```python
print(f"  {config.n_agents} agents, {perception_dim}D perception "
      f"(129 env + {N_OBJECT_CLASSES} obj slots + {N_PROPERTY_CLASSES} prop slots + {N_OBJECT_CLASSES} memory dims)")
```

Replace with:
```python
print(f"  {config.n_agents} agents, {perception_dim}D perception "
      f"(129 env + {N_OBJECT_CLASSES} obj slots + {N_PROPERTY_CLASSES} prop slots "
      f"+ {N_OBJECT_CLASSES} memory dims + {N_OBJECT_CLASSES} goal token)")
```

### Step 6: Run smoke test

```bash
python run.py --test
```

Expected: Trainer initializes with `164D perception`, 50 episodes run without error, "[OK] Smoke test passed".

If it fails: check that `_goal_class_vecs` keys match agent IDs (0..n_agents-1).

### Step 7: Commit

```bash
git add environment/world.py training/trainer.py
git commit -m "feat: add 10D goal class token to perception (Stage 4 prep)"
```

---

## Task 2: Stage 4 curriculum + TrainerConfig fields

**Files:**
- Modify: `training/trainer.py` — `TrainerConfig` lines 34–88, `setup_curriculum()` lines 206–235, `load_checkpoint()` lines 964–972
- Modify: `environment/world.py` — add `setup_stage_4()` after `setup_stage_3()`

### Step 1: Add Stage 4 fields to `TrainerConfig`

Find the `# Phase 6: food events` block (around line 72). Add after it:

```python
# Stage 4: Reference game
ref_game_prob: float = 0.35          # Fraction of Stage 4 episodes that are ref games
ref_game_steps: int = 35             # Steps per ref game episode (vs 100 for normal)
ref_game_radius: float = 8.0         # Distance threshold: runner "reached" target
ref_game_success_reward: float = 2.0 # Terminal reward for correct variant
ref_game_wrong_penalty: float = -1.0 # Terminal reward for wrong variant
```

### Step 2: Add Stage 4 branch to `setup_curriculum()`

Find the end of `setup_curriculum()`. The last branch currently is:
```python
elif episode == self.config.stage_2_episodes and self.current_stage < 3:
    self.env.setup_stage_3()
    self.env.spawn_objects()
    self.current_stage = 3
    print("\n=== Stage 3: Dynamic environment (objects move) ===")
```

Add immediately after it (before the safety block):
```python
elif episode == self.config.stage_3_episodes and self.current_stage < 4:
    self.env.setup_stage_4()
    self.current_stage = 4
    print("\n=== Stage 4: Reference game (grammatical communication) ===")
```

### Step 3: Add `setup_stage_4()` to `world.py`

Find `setup_stage_3()` in world.py (around line 572). Add after it:

```python
def setup_stage_4(self):
    """
    Stage 4: No environment change — same 12-object world as Stage 3.
    Stage 4 changes episode structure (reference game), not world complexity.
    The environment is already populated with 12 objects from spawn_objects().
    """
    self._log_event('stage_setup', {'stage': 4, 'n_objects': len(self.objects)})
```

### Step 4: Update `load_checkpoint()` RuntimeError warning

Find the RuntimeError catch in `load_checkpoint()` (around line 905–911):
```python
except RuntimeError:
    from agents.curious_agent import apply_structured_initialization
    print(f"  Warning: checkpoint architecture incompatible for agent {aid} "
          f"(perception_dim may have changed 139->149 for Phase 4, "
          f"or 149->154 for Phase 3.5). "
          f"Re-applying structured initialization.")
    apply_structured_initialization(agent, seed=self.config.seed + aid * 1000)
```

Replace the print string with:
```python
    print(f"  Warning: checkpoint architecture incompatible for agent {aid} "
          f"(perception_dim may have changed 139->149 for Phase 4, "
          f"149->154 for Phase 3.5, or 154->164 for Stage 4). "
          f"Re-applying structured initialization.")
```

### Step 5: Update `load_checkpoint()` environment restore for Stage 4

Find the environment restore block (around lines 964–972):
```python
if self.current_stage >= 3:
    self.env.setup_stage_3()
elif self.current_stage >= 2:
    self.env.setup_stage_2()
elif self.current_stage >= 1:
    self.env.setup_stage_1()
self.env.spawn_objects()
```

Replace with:
```python
if self.current_stage >= 4:
    self.env.setup_stage_3()  # Stage 4 uses same 12-object world as Stage 3
    self.env.setup_stage_4()
elif self.current_stage >= 3:
    self.env.setup_stage_3()
elif self.current_stage >= 2:
    self.env.setup_stage_2()
elif self.current_stage >= 1:
    self.env.setup_stage_1()
self.env.spawn_objects()
```

### Step 6: Run smoke test

```bash
python run.py --test
```

Expected: passes, `164D perception`, no errors. Stage 4 won't trigger in the 50-episode smoke test (threshold is 5000) — that's expected.

### Step 7: Commit

```bash
git add training/trainer.py environment/world.py
git commit -m "feat: add Stage 4 curriculum trigger and TrainerConfig ref_game fields"
```

---

## Task 3: Agent placement geometry

**Files:**
- Modify: `training/trainer.py` — add `_place_ref_game_agents()` method after `_setup_directed_discovery()`

### Step 1: Add `_place_ref_game_agents()` method

Find `_setup_directed_discovery()` (around line 332). Add a new method after it:

```python
def _place_ref_game_agents(self, scout_id: int, runner_id: int,
                            target_obj) -> None:
    """
    Place agents for a reference game episode.

    Geometry (toroidal):
      TARGET <—10u—> SCOUT <—12u—> RUNNER

    - Scout: 10 units from target → within perception radius (15), sees target
    - Runner: 12 units from scout on far side → within hearing range of scout (12<15),
              but 22 units from target (22>15, blind to target)
    - Free agents: random position anywhere in world
    """
    rng = np.random.RandomState()
    target_pos = np.array(target_obj.position, dtype=float)

    # Scout: 10 units from target in a random direction
    angle = rng.uniform(0, 2 * np.pi)
    scout_offset = np.array([np.cos(angle), np.sin(angle)]) * 10.0
    scout_pos = (target_pos + scout_offset) % self.env.world_size

    # Runner: 12 units from scout in direction AWAY from target (opposite angle)
    away_angle = angle + np.pi
    runner_offset = np.array([np.cos(away_angle), np.sin(away_angle)]) * 12.0
    runner_pos = (scout_pos + runner_offset) % self.env.world_size

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
```

### Step 2: Verify geometry in smoke test with a quick assertion

Add a temporary check at end of `smoke_test()` in `run.py` to verify placement works:

```python
# Verify ref game placement geometry
from environment.world import StructuredEnvironment
from training.trainer import Trainer, TrainerConfig
_t = Trainer(TrainerConfig(n_agents=2, n_episodes=1, steps_per_episode=1,
                           stage_1_episodes=0, stage_2_episodes=0,
                           log_dir='logs_test', checkpoint_dir='checkpoints_test'))
_t.env.setup_stage_3()
_t.env.spawn_objects()
_target_obj = list(_t.env.objects.values())[0]
_t._place_ref_game_agents(0, 1, _target_obj)
_scout = _t.agents[0]
_runner = _t.agents[1]
_scout_dist = _t.env.toroidal_distance(tuple(_scout.position), _target_obj.position)
_runner_dist = _t.env.toroidal_distance(tuple(_runner.position), _target_obj.position)
_hearing_dist = _t.env.toroidal_distance(tuple(_runner.position), tuple(_scout.position))
assert _scout_dist <= 15.0, f"Scout not in perception range: {_scout_dist:.1f}"
assert _runner_dist > 15.0, f"Runner too close to target: {_runner_dist:.1f}"
assert _hearing_dist <= 15.0, f"Runner out of hearing range: {_hearing_dist:.1f}"
print(f"[OK] Placement geometry: scout={_scout_dist:.1f}u, runner={_runner_dist:.1f}u, hearing={_hearing_dist:.1f}u")
```

Run: `python run.py --test`

Expected: geometry assertions pass, distances print. Then remove the temporary assertion block.

### Step 3: Commit

```bash
git add training/trainer.py run.py
git commit -m "feat: add _place_ref_game_agents() with three-point geometry"
```

---

## Task 4: Reference game episode execution

**Files:**
- Modify: `training/trainer.py` — add `run_reference_game_episode()`, wire into `run_episode()`

### Step 1: Add `run_reference_game_episode()` method

Add this method after `_place_ref_game_agents()`. This is the core implementation:

```python
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
    parts = target_key.split('_')
    base_name = parts[0] if (len(parts) > 1 and parts[-1].isdigit()) else target_key
    target_class_idx = WORD_CLASS_MAP.get(base_name, -1)

    # Ambiguous: another object shares the same base name
    same_class_keys = [
        k for k in self.env.objects
        if k != target_key
        and (k.split('_')[0] if (len(k.split('_')) > 1
             and k.split('_')[-1].isdigit()) else k) == base_name
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
    self._place_ref_game_agents(scout_id, runner_id, target_obj)

    # --- Set goal class vec for runner (zeros for all others) ---
    for aid in aid_list:
        self._goal_class_vecs[aid] = np.zeros(N_OBJECT_CLASSES, dtype=np.float32)
    if 0 <= target_class_idx < N_OBJECT_CLASSES:
        self._goal_class_vecs[runner_id][target_class_idx] = 1.0

    # --- Initialize tracking state ---
    self._ref_game_state = {
        'active': True,
        'scout_id': scout_id,
        'runner_id': runner_id,
        'target_key': target_key,
        'target_class': base_name,
        'target_is_ambiguous': target_is_ambiguous,
        'outcome': 'timeout',
        'runner_min_distance': float('inf'),
        'scout_words_emitted': [],
        'scout_used_property': False,
        'scout_used_correct_property': False,
    }

    # --- Run steps ---
    runner_agent = next(a for a in self.agents if a.config.agent_id == runner_id)
    for _ in range(self.config.ref_game_steps):
        self.run_step()
        dist = self.env.toroidal_distance(
            tuple(runner_agent.position), target_obj.position
        )
        if dist < self._ref_game_state['runner_min_distance']:
            self._ref_game_state['runner_min_distance'] = dist

    # --- Collect scout utterances from utterance_log ---
    scout_obj_words = []
    scout_prop_words = []
    for entry in self.utterance_log:
        if entry['agent_id'] != scout_id:
            continue
        if entry.get('type') == 'object':
            idx = entry.get('class_idx')
            if idx is not None and 0 <= idx < N_OBJECT_CLASSES:
                scout_obj_words.append(ALL_OBJECT_CLASSES[idx])
        elif entry.get('type') == 'property':
            idx = entry.get('property_idx')
            if idx is not None and 0 <= idx < N_PROPERTY_CLASSES:
                scout_prop_words.append(ALL_PROPERTY_CLASSES[idx])

    self._ref_game_state['scout_words_emitted'] = scout_obj_words + scout_prop_words
    self._ref_game_state['scout_used_property'] = len(scout_prop_words) > 0
    self._ref_game_state['scout_used_correct_property'] = (
        correct_property_idx is not None
        and ALL_PROPERTY_CLASSES[correct_property_idx] in scout_prop_words
    )

    # --- Determine outcome ---
    dist_correct = self.env.toroidal_distance(
        tuple(runner_agent.position), target_obj.position
    )
    dist_wrong = float('inf')
    for wrong_key in same_class_keys:
        if wrong_key in self.env.objects:
            d = self.env.toroidal_distance(
                tuple(runner_agent.position),
                self.env.objects[wrong_key].position
            )
            dist_wrong = min(dist_wrong, d)

    if dist_correct <= self.config.ref_game_radius:
        outcome = 'correct'
        terminal_reward = self.config.ref_game_success_reward
    elif dist_wrong <= self.config.ref_game_radius:
        outcome = 'wrong_variant'
        terminal_reward = self.config.ref_game_wrong_penalty
    else:
        outcome = 'timeout'
        terminal_reward = 0.0

    self._ref_game_state['outcome'] = outcome

    # --- Apply terminal reward to scout and runner's last experience entry ---
    # This flows through REINFORCE: the last step's reward carries the game outcome.
    if terminal_reward != 0.0:
        scout_agent = next(a for a in self.agents if a.config.agent_id == scout_id)
        for target_agent in (runner_agent, scout_agent):
            if target_agent.experience_buffer:
                p, a, r = target_agent.experience_buffer[-1]
                target_agent.experience_buffer[-1] = (p, a, r + terminal_reward)
            target_agent.total_reward += terminal_reward

    # --- Policy update ---
    if episode % self.config.policy_update_freq == 0:
        for agent in self.agents:
            agent.update_policy(steps_per_episode=self.config.ref_game_steps)
```

### Step 2: Wire into `run_episode()`

Find `run_episode()` (line 649). After `self.setup_curriculum(episode)`, add the two new initializations, then insert the ref game check before the step loop.

Find:
```python
def run_episode(self, episode: int):
    """Run one full episode."""
    self.setup_curriculum(episode)

    # Space agents apart for information asymmetry, then possibly nudge one
    # toward an object its partner hasn't seen (directed discovery).
    self._reset_agent_positions(episode)
    self._setup_directed_discovery(episode)
```

Replace with:
```python
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
```

Then find the step loop and policy update block:
```python
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
```

Replace with:
```python
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
```

### Step 3: Test reference game execution via direct call in smoke_test

Add to `smoke_test()` in `run.py` after `trainer.train()`:

```python
# Test reference game episode directly (Stage 4 doesn't trigger in 50 ep, so call directly)
trainer.current_stage = 4
trainer.env.setup_stage_3()
trainer.env.spawn_objects()
trainer._goal_class_vecs = {
    a.config.agent_id: np.zeros(10, dtype='float32') for a in trainer.agents
}
trainer._ref_game_state = {'active': False}
trainer.prev_utterances = {}
trainer.utterance_log = []
trainer.episode_step = 0
trainer.episode_referral_totals = {a.config.agent_id: 0.0 for a in trainer.agents}
trainer.episode_joint_totals = {a.config.agent_id: 0.0 for a in trainer.agents}
trainer.episode_property_comm_totals = {a.config.agent_id: 0.0 for a in trainer.agents}
trainer.episode_property_approach_totals = {a.config.agent_id: 0.0 for a in trainer.agents}
trainer.episode_event_active = False
trainer.episode_event_agent_arrivals = set()
trainer.active_event_object = None
for agent in trainer.agents:
    trainer.hidden_states[agent.config.agent_id] = agent.reset_hidden()
    agent.current_episode = 50
trainer.run_reference_game_episode(episode=50)
rg = trainer._ref_game_state
assert rg['active'] == True, "ref_game_state.active should be True"
assert rg['outcome'] in ('correct', 'wrong_variant', 'timeout'), f"Bad outcome: {rg['outcome']}"
assert rg['runner_min_distance'] < float('inf'), "runner_min_distance not updated"
print(f"[OK] Reference game: outcome={rg['outcome']}, "
      f"target={rg['target_key']}, dist={rg['runner_min_distance']:.1f}, "
      f"scout_words={rg['scout_words_emitted'][:5]}")
```

Run: `python run.py --test`

Expected: `[OK] Reference game: outcome=..., target=..., dist=..., scout_words=[...]`

Then remove the test block from `run.py`.

### Step 4: Commit

```bash
git add training/trainer.py run.py
git commit -m "feat: implement run_reference_game_episode() and wire into run_episode()"
```

---

## Task 5: Metrics — ref_game JSON field + utterance rate fix

**Files:**
- Modify: `training/trainer.py` — `collect_metrics()` lines 720–779

### Step 1: Fix utterance_rate denominator for ref game episodes

In `collect_metrics()`, find:
```python
'utterance_count': sum(1 for e in self.utterance_log if e['agent_id'] == agent.config.agent_id),
'utterance_rate': sum(1 for e in self.utterance_log if e['agent_id'] == agent.config.agent_id) / max(1, self.config.steps_per_episode),
```

Replace with:
```python
_n_steps = (self.config.ref_game_steps
            if self._ref_game_state.get('active', False)
            else self.config.steps_per_episode)
'utterance_count': sum(1 for e in self.utterance_log if e['agent_id'] == agent.config.agent_id),
'utterance_rate': sum(1 for e in self.utterance_log if e['agent_id'] == agent.config.agent_id) / max(1, _n_steps),
```

Also fix `property_utterance_rate`:
```python
'property_utterance_rate': sum(
    1 for e in self.utterance_log
    if e['agent_id'] == agent.config.agent_id and e.get('type') == 'property'
) / max(1, _n_steps),
```

Note: `_n_steps` must be defined before the per-agent loop (or inside it). Define it once before the `for agent in self.agents:` loop in `collect_metrics()`.

### Step 2: Add `ref_game` key to metrics

Find in `collect_metrics()`:
```python
# Food event metrics (episode-level)
metrics['event_active'] = self.episode_event_active
metrics['event_arrivals'] = len(self.episode_event_agent_arrivals)
```

Add after it:
```python
# Stage 4: Reference game metrics (episode-level)
# runner_min_distance may be float('inf') for timeout — convert to None for JSON
ref_state = dict(self._ref_game_state)
if ref_state.get('runner_min_distance') == float('inf'):
    ref_state['runner_min_distance'] = None
metrics['ref_game'] = ref_state
```

### Step 3: Run smoke test

```bash
python run.py --test
```

Expected: passes. Open `logs_test/metrics_ep25.json` and verify each entry has a `"ref_game": {"active": false}` key.

```bash
python -c "import json; d=json.load(open('logs_test/metrics_ep25.json')); print(d[0].get('ref_game'))"
```

Expected: `{'active': False}`

### Step 4: Commit

```bash
git add training/trainer.py
git commit -m "feat: add ref_game key to metrics JSON and fix utterance_rate for short episodes"
```

---

## Task 6: analyze_run.py — `--section refgame`

**Files:**
- Modify: `analyze_run.py` — add `print_refgame()`, update `print_summary()`, update `main()`

### Step 1: Add `import numpy as np` if not present

Check line 1–30 of `analyze_run.py`. It uses `np.mean` in `print_refgame` so numpy must be imported. Add if missing:
```python
import numpy as np
```

### Step 2: Add `print_refgame()` function

Add after `print_comm()` (around line 144):

```python
def print_refgame(entries: List[Dict], aids: List[str]):
    """Reference game outcomes and grammar signals (Stage 4 only)."""
    ref_entries = [e for e in entries if e.get('ref_game', {}).get('active', False)]

    print(f'\n{"=== REFERENCE GAME METRICS (Stage 4)":<60}')
    if not ref_entries:
        print('  No reference game episodes in this window.')
        return

    print(f'{"EP":>6} {"scout":>5} {"run":>3} {"target":>10} {"amb":>3} '
          f'{"outcome":>12} {"prop":>4} {"cp":>3} {"dist":>6}')
    print('-' * 60)

    for e in ref_entries:
        rg = e['ref_game']
        ep      = e['episode']
        scout   = rg.get('scout_id', '?')
        runner  = rg.get('runner_id', '?')
        target  = rg.get('target_key', '?')[:10]
        amb     = 'Y' if rg.get('target_is_ambiguous', False) else 'N'
        outcome = rg.get('outcome', 'timeout')[:12]
        prop    = 'Y' if rg.get('scout_used_property', False) else 'N'
        cprop   = 'Y' if rg.get('scout_used_correct_property', False) else 'N'
        dist    = rg.get('runner_min_distance') or 999.9
        print(f'{ep:>6} {str(scout):>5} {str(runner):>3} {target:>10} {amb:>3} '
              f'{outcome:>12} {prop:>4} {cprop:>3} {dist:>6.1f}')

    # Summary statistics
    n = len(ref_entries)
    n_amb     = sum(1 for e in ref_entries if e['ref_game'].get('target_is_ambiguous'))
    n_correct = sum(1 for e in ref_entries if e['ref_game'].get('outcome') == 'correct')
    n_wrong   = sum(1 for e in ref_entries if e['ref_game'].get('outcome') == 'wrong_variant')
    n_timeout = sum(1 for e in ref_entries if e['ref_game'].get('outcome') == 'timeout')
    n_prop    = sum(1 for e in ref_entries if e['ref_game'].get('scout_used_property'))
    n_cprop   = sum(1 for e in ref_entries if e['ref_game'].get('scout_used_correct_property'))
    dists     = [e['ref_game'].get('runner_min_distance') or 999.9 for e in ref_entries]
    avg_dist  = float(np.mean(dists))

    print(f'\n  Summary ({n} ref game episodes in window):')
    print(f'    Ambiguous targets: {n_amb:>3}/{n} ({100*n_amb/n:>4.0f}%)')
    print(f'    Correct:           {n_correct:>3}/{n} ({100*n_correct/n:>4.0f}%)')
    print(f'    Wrong variant:     {n_wrong:>3}/{n} ({100*n_wrong/n:>4.0f}%)')
    print(f'    Timeout:           {n_timeout:>3}/{n} ({100*n_timeout/n:>4.0f}%)')
    print(f'    Scout used prop:   {n_prop:>3}/{n} ({100*n_prop/n:>4.0f}%)')
    print(f'    Correct prop word: {n_cprop:>3}/{n} ({100*n_cprop/n:>4.0f}%)')
    print(f'    Avg min distance:  {avg_dist:>6.1f}')
```

### Step 3: Add ref game summary to `print_summary()`

Find at the end of `print_summary()`:
```python
    # Event summary
    n_events = sum(1 for e in entries if e.get('event_active', False))
    ...
    print(f'    Avg arrivals per event: {avg_arrivals:.2f}')
```

Add after it:
```python
    # Reference game summary (Stage 4)
    rg_entries = [e for e in entries if e.get('ref_game', {}).get('active', False)]
    if rg_entries:
        n_rg = len(rg_entries)
        n_correct = sum(1 for e in rg_entries if e['ref_game'].get('outcome') == 'correct')
        n_cprop   = sum(1 for e in rg_entries if e['ref_game'].get('scout_used_correct_property'))
        print(f'\n  Reference Game (Stage 4):')
        print(f'    Games played:      {n_rg} / {len(entries)}')
        print(f'    Success rate:      {100*n_correct/n_rg:.0f}%')
        print(f'    Correct prop use:  {100*n_cprop/n_rg:.0f}%')
```

### Step 4: Update `main()` — add `refgame` section choice

Find:
```python
    parser.add_argument('--section', choices=['all', 'core', 'lang', 'comm'], default='all',
                        help='Which metrics to display (default: all)')
```

Replace with:
```python
    parser.add_argument('--section', choices=['all', 'core', 'lang', 'comm', 'refgame'],
                        default='all',
                        help='Which metrics to display (default: all)')
```

Find:
```python
    if args.section in ('all', 'comm'):
        print_comm(entries, aids)
```

Add after it:
```python
    if args.section in ('all', 'refgame'):
        print_refgame(entries, aids)
```

### Step 5: Verify analyze_run works on existing logs

```bash
python analyze_run.py --log-dir logs_phase6 --section refgame --ep-min 4000
```

Expected: `=== REFERENCE GAME METRICS (Stage 4)` header followed by `No reference game episodes in this window.` (Phase 6 logs predate Stage 4 — this is correct).

```bash
python analyze_run.py --log-dir logs_phase6 --section all --ep-min 4000
```

Expected: all existing sections still work, new refgame section at end shows no entries.

### Step 6: Commit

```bash
git add analyze_run.py
git commit -m "feat: add --section refgame to analyze_run.py with grammar signal metrics"
```

---

## Final Verification

### Run full smoke test

```bash
python run.py --test
```

Expected:
- `164D perception (129 env + 10 obj slots + 5 prop slots + 10 memory dims + 10 goal token)`
- 50 episodes run cleanly
- `[OK] Smoke test passed`

### Verify JSON structure

```bash
python -c "
import json
d = json.load(open('logs_test/metrics_ep25.json'))
e = d[0]
print('ref_game key present:', 'ref_game' in e)
print('ref_game:', e['ref_game'])
print('perception_dim check: agent 0 has naming_accuracy:', e['agents']['0'].get('naming_accuracy'))
"
```

Expected: `ref_game key present: True`, `ref_game: {'active': False}` (normal episode in smoke test).

### Verify analyze_run still works

```bash
python analyze_run.py --log-dir logs_test --section all
```

Expected: all 5 sections print without error, refgame shows no entries.

---

## Start a training run

```bash
python run.py --episodes 10000 --log-dir logs_stage4 --checkpoint-dir checkpoints_stage4
```

At ep5000 you should see:
```
=== Stage 4: Reference game (grammatical communication) ===
```

Monitor with:
```bash
python analyze_run.py --log-dir logs_stage4 --section refgame --ep-min 5000
```
