# Phase 5: Disambiguation Pressure & Property-Consequential World

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Create genuine compositional grammar pressure by (a) adding property-varying object instances so noun-alone is ambiguous, and (b) making object properties behaviorally consequential so agents have real motivation to communicate distinctions.

**Architecture:** Two new object variants (poisonous apple, feral cat) share class indices with their originals — forcing agents to combine noun + property to disambiguate. Per-step property-sensitive rewards (danger penalty, food bonus) give agents behavioral stakes in which variant they encounter, making property communication pragmatically useful rather than purely structural.

**Tech Stack:** Python, PyTorch, existing world/trainer/agent architecture. No perception dim changes, no new optimizer, no checkpoint migration required.

---

## Task 1: Add variant objects to world.py

**Files:**
- Modify: `environment/world.py`

**What to implement:**

Add two new entries to `OBJECT_LIBRARY`:

```python
'apple_2': lambda pos: WorldObject(
    name='apple',           # CRITICAL: same name as original for teaching
    category='fruit',
    properties=make_properties(
        color_r=0.9, color_g=0.1, color_b=0.1,
        size=0.3, shape_round=0.9, shape_long=0.1,
        animate=0.0, edible=0.0,    # was 1.0 — poisonous
        dangerous=0.6,              # was 0.0 — above 0.5 threshold
        warm=0.2, soft=0.4, bright=0.5, noisy=0.0,
        complexity=0.2
    ),
    position=pos,
    text_description="A round red fruit. Something seems wrong with it."
),
'cat_2': lambda pos: WorldObject(
    name='cat',             # CRITICAL: same name as original for teaching
    category='animal',
    properties=make_properties(
        color_r=0.5, color_g=0.4, color_b=0.3,
        size=0.4, shape_round=0.4, shape_long=0.6,
        animate=1.0, edible=0.0, dangerous=0.9,  # was 0.1 — feral
        warm=0.7, soft=0.8, bright=0.3, noisy=0.3,
        complexity=0.7
    ),
    position=pos,
    text_description="A small furry animal. It looks hostile."
),
```

Update `spawn_objects()` to include both variants in the base object list:

```python
base_objects = [
    'flower', 'rock', 'apple', 'ball', 'book',
    'dog', 'fire', 'cat', 'water', 'banana',
    'apple_2', 'cat_2',   # Phase 5 variants
]
```

**Verify:** `env.objects` has 12 keys after `spawn_objects()`. Both `apple_2` and `cat_2` have `name` equal to `"apple"` and `"cat"` respectively. The OstensiveTeacher's `teach_step` reads `obj.name`, so it will correctly teach "apple" for both apple instances with no code changes.

**Step: Commit**
```bash
git add environment/world.py
git commit -m "feat(phase5): add poisonous apple and feral cat variants to world"
```

---

## Task 2: Property-sensitive rewards in trainer.py

**Files:**
- Modify: `training/trainer.py`

**What to implement:**

Add to `TrainerConfig`:
```python
# Phase 5: property-consequential rewards
property_approach_penalty: float = -0.2   # per-step penalty near dangerous objects
property_food_bonus: float = 0.05         # per-step bonus near edible objects
danger_radius: float = 8.0
food_radius: float = 8.0
```

Add accumulator to `__init__`:
```python
self.episode_property_approach_totals: Dict[int, float] = {}
```

Reset in `run_episode` at episode start:
```python
self.episode_property_approach_totals = {a.config.agent_id: 0.0 for a in self.agents}
```

Add Phase 6.4 block in `run_step()`, after Phase 6.3 (joint curiosity) and before Phase 6.5 (communicative reward):

```python
# Phase 6.4 — property-sensitive approach rewards
dangerous_dim = PROPERTY_DIM_MAP.get('dangerous', 3)
edible_dim = PROPERTY_DIM_MAP.get('edible', 1)
for agent in self.agents:
    aid = agent.config.agent_id
    for obj_key, obj in env.objects.items():
        dist = np.linalg.norm(
            np.array(agent.position) - np.array(obj.position)
        )
        if dist <= self.config.danger_radius:
            if obj.properties[dangerous_dim] > 0.5:
                r = self.config.property_approach_penalty
                agent.total_reward += r
                self.episode_property_approach_totals[aid] = (
                    self.episode_property_approach_totals.get(aid, 0.0) + r
                )
        if dist <= self.config.food_radius:
            if obj.properties[edible_dim] > 0.5:
                r = self.config.property_food_bonus
                agent.total_reward += r
                self.episode_property_approach_totals[aid] = (
                    self.episode_property_approach_totals.get(aid, 0.0) + r
                )
```

Add to `collect_metrics()` per-agent dict:
```python
'property_approach_reward': self.episode_property_approach_totals.get(agent.config.agent_id, 0.0),
```

**Verify:** Smoke test shows `property_approach_reward` field in logs. Value is negative for agents that spent time near fire, positive for agents near apple/banana.

**Step: Commit**
```bash
git add training/trainer.py
git commit -m "feat(phase5): add property-sensitive approach rewards (danger penalty, food bonus)"
```

---

## Task 3: Metric tracking in analyze_run.py

**Files:**
- Modify: `analyze_run.py`

**What to implement:**

In `print_comm()`, add `prop_app` column to the agent header and data rows:

```python
# header — add prop_app after pvoc:
f'{"A"+a+":utt%":>7} {"putt":>5} {"ref_r":>5} {"prop_r":>6} {"prop_app":>8} {"jnt_r":>5} {"mem":>3} {"pvoc":>4}'

# data row — add prop_app:
prop_app = d.get('property_approach_reward', 0.0)
cols.append(f'{utt_rate:>7.3f} {putt:>5.3f} {ref_r:>5.2f} {prop_r:>6.2f} {prop_app:>8.2f} {jnt_r:>5.2f} {mem:>3d} {pvoc:>4d}')
```

In `print_summary()`, add property approach reward line after Prop vocab:
```python
app_r = d1.get('property_approach_reward', 0.0)
print(f'    Prop approach: {app_r:.2f}')
```

**Step: Commit**
```bash
git add analyze_run.py
git commit -m "feat(phase5): add prop_app column to analyze_run comm section"
```

---

## Task 4: Smoke test and README update

**Files:**
- Modify: `README.md`

**Smoke test:**
```bash
python run.py --test
```
Expected: 50 episodes complete, no errors. Log contains `property_approach_reward` field. `analyze_run.py --log-dir logs_test --section comm` shows `prop_app` column with non-zero values.

**README updates:**

1. **Status section** — Mark Phase 5 as Active
2. **Architecture section** — World now has 12 objects (10 base + 2 variants); note same-class-index design
3. **Phases table** — Add Phase 5 row: "Disambiguation pressure + property-consequential world"
4. **Reward architecture** — Add danger penalty and food bonus to the reward list
5. **What to Watch For** — Add: fire avoidance emerging early, poisonous apple approached less than normal apple, feral cat approached less than normal cat, noun+property co-utterance sequences near variants

**Step: Commit**
```bash
git add README.md
git commit -m "docs: update README for Phase 5 (disambiguation pressure)"
```
