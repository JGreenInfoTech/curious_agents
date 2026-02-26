# Stage 4 Reference Game — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans then superpowers:subagent-driven-development to implement this design.

**Goal:** Add a Stage 4 reference game that creates direct gradient pressure for grammatical communication — specifically, forcing agents to combine object words and property words to disambiguate variant objects (apple vs apple_2, cat vs cat_2).

**Architecture:** 35% of Stage 4 episodes are reference game episodes. Two agents are assigned scout/runner roles; a third continues normal exploration. Scout sees the target object, runner has a goal class token and must navigate to the correct variant. Short 35-step window prevents random search success. Shared asymmetric reward (correct / timeout / wrong-variant penalty) makes underdetermined single-word utterances actively harmful.

**Why grammar emerges:** Single-word utterance "apple" is ambiguous when both apple and apple_2 exist. Runner reaching wrong variant = -1.0 for both agents. Scout must emit "apple" + "dangerous" (or similar property word) for runner to distinguish. No other mechanism forces this combination — it must be learned.

**Perception change:** `perception_dim 154 → 164` (+10D goal class one-hot for runner, zeros otherwise). Requires fresh training run. Current ep4500 checkpoint is the Phase 6 baseline.

---

## Stage 4 Trigger

- `TrainerConfig.stage_3_episodes = 5000` already defined — use it
- `setup_curriculum()` adds fourth branch: `episode == config.stage_3_episodes and current_stage < 4`
- `env.setup_stage_4()` = no-op (same 12-object world as Stage 3)
- `load_checkpoint()` restore branch extended to handle `current_stage == 4`

## Reference Game Episode Selection

At the start of `run_episode()` in Stage 4:
- `is_ref_game = (current_stage == 4) and (random.random() < config.ref_game_prob)`
- `TrainerConfig.ref_game_prob: float = 0.35`
- If true: call `run_reference_game_episode()` instead of normal episode loop
- If false: normal episode (curiosity + language + comm rewards as before)

## Role Assignment

```python
scout_id, runner_id = random.sample([0, 1, 2], 2)
free_id = {0, 1, 2} - {scout_id, runner_id}
```

All six permutations sampled uniformly. Every agent practices both roles over time.

## Target Selection

- Pick random object from `env.objects.values()`
- `target_key`: the dict key (e.g., `"apple_2"`)
- `target_class`: `obj.name` (e.g., `"apple"`) → maps to class index via `ALL_OBJECT_CLASSES`
- `target_is_ambiguous`: True when another object shares the same class name (apple/apple_2, cat/cat_2)
- Naturally ~17% of games are ambiguous (2 of 12 objects are variants)

## Placement Geometry

Three-point layout (toroidal distances):

```
TARGET <—10u—> SCOUT <—12u—> RUNNER
```

- Scout: placed 10 units from target in random direction → within perception_radius (15), sees target
- Runner: placed 12 units from scout in direction *away* from target → 22 units from target (> 15, blind to target), 12 units from scout (< 15, can hear scout)
- Runner can navigate 22 units to target within 35 steps with directional movement
- Once runner closes to within 15 units of target, it perceives the object directly

Implementation: new `_place_ref_game_agents(scout_id, runner_id, target_obj)` in Trainer.

## Episode Structure

- 35 steps (vs 100 for normal episodes) — `TrainerConfig.ref_game_steps: int = 35`
- `run_step()` executes normally for all three agents each step
- Scout's utterances appear in runner's utterance slots from step 1 (already within range)
- Free agent runs standard curiosity loop for all 35 steps
- No turn-taking machinery needed — existing utterance slot mechanism handles communication

## Goal Token

- Runner's perception gains a 10D one-hot class vector: `goal_class_vec`
- Zeros for scout, free agent, and all agents during normal episodes
- `_build_perception()` concatenates `goal_class_vec` after existing dims
- `get_perception_dim()` updated: `154 + 10 = 164`
- `AgentConfig.perception_dim` updated accordingly

## Reward Structure

Evaluated at episode end based on runner's closest approach to any instance of the target class:

| Outcome | Runner reward | Scout reward | Condition |
|---------|--------------|-------------|-----------|
| Correct variant | +2.0 | +2.0 | Runner within `ref_game_radius` of correct variant |
| Wrong variant | -1.0 | -1.0 | Runner within `ref_game_radius` of wrong variant first |
| Timeout | 0.0 | 0.0 | Neither reached within 35 steps |

```
TrainerConfig.ref_game_success_reward: float = 2.0
TrainerConfig.ref_game_wrong_penalty: float = -1.0
TrainerConfig.ref_game_radius: float = 8.0
```

Reward added to agent's standard reward before `store_experience()`. Free agent reward unaffected.

For unambiguous targets (single instance of class): correct variant = only instance, wrong variant impossible → effectively correct or timeout only.

## Observability — JSON Logging

Each episode's metrics dict gains a `"ref_game"` key:

**Normal episode:**
```json
"ref_game": {"active": false}
```

**Reference game episode:**
```json
"ref_game": {
  "active": true,
  "scout_id": 0,
  "runner_id": 2,
  "target_key": "apple_2",
  "target_class": "apple",
  "target_is_ambiguous": true,
  "outcome": "correct",
  "runner_min_distance": 4.2,
  "scout_words_emitted": ["apple", "dangerous", "apple"],
  "runner_words_emitted": [],
  "scout_used_property": true,
  "scout_used_correct_property": true
}
```

- `scout_words_emitted`: list of word strings emitted by scout during episode (obj + property words)
- `scout_used_property`: any property word emitted by scout
- `scout_used_correct_property`: emitted the property word that correctly characterises the target variant's distinguishing property (e.g., "dangerous" for apple_2)

## Observability — analyze_run.py `--section refgame`

```
=== REFERENCE GAME METRICS ===
   EP   n_ref  amb%  suc%  wrng%  tout%  prop%  cprop%  avg_dist
 5000      18    17    42     8     50     31      22     18.4
 5500      21    17    58    12     30     47      39     12.1
 6000      19    17    71     5     24     63      57      8.3
```

Columns:
- `n_ref`: number of reference game episodes in window
- `amb%`: % of games with ambiguous target
- `suc%`: success rate (correct variant reached)
- `wrng%`: wrong-variant rate (grammar failure signal)
- `tout%`: timeout rate
- `prop%`: % of games where scout emitted any property word
- `cprop%`: % of games where scout emitted the *correct* property word
- `avg_dist`: average runner minimum distance to target

**Key diagnostic:** `cprop%` rising alongside `suc%` = grammar driving success. `suc%` rising with flat `cprop%` = runner succeeding by other means (investigate).

---

## Checkpoint Strategy

Current ep4500 run (task b0b0aa4) continues to ep10000 as the **Phase 6 baseline**. Stage 4 implementation starts a fresh training run with the updated architecture (`perception_dim 164`). No checkpoint migration needed — fresh start from ep0 with Stage 1→2→3→4 curriculum.

Phase 6 baseline provides:
- Confirmed coordination plateau at ~0.50/event
- Property approach reward going positive (danger avoidance)
- All agents at full 10-word + 5-property vocabulary
- disc_loss collapsed to 0.29 minimum (A0)

These are the pre-Stage-4 baselines against which Stage 4 improvements are measured.

---

## Files to Touch

| File | Change |
|------|--------|
| `training/trainer.py` | `setup_curriculum()` Stage 4 branch; `TrainerConfig` new fields; `run_reference_game_episode()`; `_place_ref_game_agents()`; `collect_metrics()` ref_game block; `load_checkpoint()` stage 4 restore |
| `agents/curious_agent.py` | `AgentConfig.perception_dim 154→164`; `get_perception_dim()` +10; `_build_perception()` goal_class_vec concat |
| `environment/world.py` | `setup_stage_4()` (no-op stub, documents intent) |
| `analyze_run.py` | `print_refgame()` function; `--section refgame` CLI option |
