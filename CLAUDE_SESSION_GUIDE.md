# Claude Session Guide — curious_agents Project
=== Journal Entry 2026-02-21T15:54:54.227614+00:00 ===
2026-02-21-15-54-53-phase2-teaching-debug.txt
[Debugging session investigating why OstensiveTeacher never fires during Phase 2 training (episodes 5000-8000). Traces through trainer.py curriculum management, load_checkpoint logic, and teaching system integration to identify why vocab_size=0 and total_teaching_events=0 despite agents being in Stage 3.]
=== Journal Entry 2026-02-21T15:58:56.335634+00:00 ===
2026-02-21-15-58-55-phase2-empty-env-bug.txt
[Root cause analysis of Phase 2 training failure where agents learned zero vocabulary (vocab_size=0, teaching_events=0) during episodes 5000-8000. Traces through checkpoint resume logic to identify that environment objects list was never restored, leaving agents in an empty world.]
=== Journal Entry 2026-02-22T15:38:14.501125+00:00 ===
2026-02-22-15-38-13-phase2-empty-env-fix.txt
[Bug fix implementation for Phase 2 training failure where environment objects were never restored from checkpoint, leaving agents in empty world for episodes 5000-8000. Contains root cause analysis, code fixes to trainer.py, and verification testing.]
=== Journal Entry 2026-02-22T16:27:01.324588+00:00 ===
2026-02-22-16-27-00-checkpoint-format-investigation.txt
[Investigation of checkpoint data structure revealing agents stored as dict with state_dicts (not integers), environment objects present in all checkpoints, and teacher word_memories keyed by agent IDs. Confirms checkpoint data is intact - the bug was in load_checkpoint() never restoring env.objects from saved env_state.]
=== Journal Entry 2026-02-22 (Analysis tooling improvements) ===
Three analysis gaps fixed before Phase 2 training run:
1. collect_metrics() in trainer.py now persists avg_naming_loss and avg_discrimination_loss from metacognitive_report() into JSON logs. Previously computed but silently discarded.
2. analyze_run.py fully rewritten: covers all episodes (no ep>=5000 filter), two sections (core: error/progress/confidence/reward; lang: vocab/naming_acc/naming_loss/disc_loss), CLI flags --ep-min, --every, --section, --log-dir.
3. visualizer.py expanded from 2x2 to 2x3 grid: added plot_naming_accuracy() (row 0, col 2) and plot_language_losses() (row 1, col 2). --visualize flag now useful for Phase 2 runs.
=== Journal Entry 2026-02-22 (Phase 2 encoder-shaping features) ===
Three features added to deepen language grounding:
1. NAMING LOSS: MSE(encoder_output, word_prototype) × 0.1 fires per teaching event for grounded words. Pulls encoder toward stable per-word prototype. Implemented as train_language_losses() in CuriousAgent, called from OstensiveTeacher.teach_step().
2. CIRCULAR-BUFFER VOCABULARY: WordMemory replaced cumulative sum with 8-slot ring buffer. write_pos cycles mod buffer_size; prototype = np.mean(buffer). Prevents early-training encodings from diluting later, more accurate ones. TeachingConfig.vocab_buffer_size=8.
3. DISCRIMINATION HEAD: Linear(64→32)→GELU→Linear(32→10) auxiliary classifier. Cross-entropy loss × 0.2. Forces class-separable encoder representations. WORD_CLASS_MAP and ALL_OBJECT_CLASSES added to language_grounding.py. Separate language_optimizer covers encoder + head.
Checkpoint format updated (buffer/write_pos replaces state_accumulator). Backward-compat load handles old format. state_dict loading now strict=False. Smoke test passes: 50ep, ~5ep/s, vocab grounds correctly, checkpoint round-trips cleanly.
=== Journal Entry 2026-02-23 (Entropy regularization) ===
Root cause analysis of utterance collapse revealed policy logit gap of ~1.3-1.6 between movement (+0.3 to +2.1) and utterances (~-0.9). Temperature floor fix alone insufficient — REINFORCE had already trained weights away from utterances over 2000 episodes.
Fix: entropy regularization in update_policy() (curious_agent.py):
  - Added entropy_coeff=0.02 to AgentConfig
  - Added entropy_losses list in update_policy loop: -entropy_coeff * dist.entropy() per sample
  - total_loss now includes entropy_losses.sum() alongside policy + value losses
  - Subtracting entropy from loss = maximizing entropy = penalizing logit concentration
  - At max entropy (log(15)≈2.7), contribution is ~0.054 per sample — small but persistent pressure against overconfidence
=== Journal Entry 2026-02-23 (Phase 3 utterance suppression fix) ===
Two hyperparameter fixes after observing utterance collapse in Phase 3 training:
At ep 1000, agents used utterances ~50% of steps. By ep 2000 (temp=0.300), utterances dropped to ~5% — policy went near-greedy and the -0.2 utterance init bias suppressed word emissions entirely.
Root cause: (1) temp floor 0.3 too low for stochastic utterances; (2) comm_reward 0.2 too small to compete with movement at low temp.
Fixes in TrainerConfig (trainer.py):
  - temperature_end: 0.3 → 0.6  (keeps utterance actions statistically viable)
  - communicative_reward: 0.2 → 0.5  (gives utterances enough expected value at exploitation regime)
  - load_checkpoint now uses max(checkpoint_temp, config.temperature_end) so a raised floor takes effect on resume without needing to edit the checkpoint.
=== Journal Entry 2026-02-23 (Phase 3 multi-agent communication) ===
Three features added for agent-to-agent word communication:
1. UTTERANCE ACTIONS: Policy expanded from 5 to 15 outputs (5 movement + 10 word emissions). Actions 5-14 = emit word at class index (action-5). execute_action() handles these — no movement, sets last_utterance_class. Forward model still only uses 5D movement one-hot (utterance → "stay" mapping). Policy bias init discourages utterances early (-0.2) so agents explore before talking.
2. WORD PERCEPTION SLOTS: perception_dim 129→139 (10 utterance slots appended). Trainer._build_utterance_slots() scans nearby agents for utterances within helping_radius and sets slot[class_idx]=1.0. Phase 1 perceive uses prev_utterances (last step, so agents know what was said when deciding next action). Phase 4 perceive uses step_utterances (current step, simultaneous communication). prev_utterances resets at episode boundary.
3. COMMUNICATIVE REWARD: Phase 6.5 in run_step. When agent A emits word with class_idx C, scan nearby agents' discrimination_head predictions. If B predicts class C, A gets +0.2 (communicative_reward in TrainerConfig). Approach-only — fires only when shared understanding confirmed by B's internal state. Near-zero early in training, rises as discrimination heads converge.
Checkpoint compat: loading old ep0-999 checkpoints will hit RuntimeError (encoder shape 129→139). load_checkpoint now wraps state_dict load in try/except, falls back to apply_structured_initialization for incompatible agents with a clear warning.
Smoke test: 50ep, 4.3ep/s, 139D perception confirmed, vocab grounds correctly, utterance actions in action_history.
=== Journal Entry 2026-02-24 (Phase 4: spatial memory + communication scaffolding) ===
Five features added to give agents persistent spatial memory and conditions for genuine inter-agent communication:
1. SPATIAL MEMORY: SpatialMemory dataclass in curious_agent.py. Per-agent entries[class_idx] = {salience, episode, position, times_visited}. Salience decays as exp(-delta/20). to_vec() returns 10D vector appended to perception. serialize/deserialize for checkpoint persistence.
2. PERCEPTION DIM 139→149: 10 memory dims appended after utterance slots. get_perception_dim() gains n_memory_classes param. Trainer._build_perception() replaces all inline perceive calls in run_step.
3. PERCEPTION RADIUS 30→15 (also teaching_radius 25→15): Tighter radius forces genuine information asymmetry — agents at radius 35 from center can't see each other's objects without communicating.
4. ASYMMETRIC STARTS + DIRECTED DISCOVERY: _reset_agent_positions() spaces agents evenly around world center (r≈35). _setup_directed_discovery() fires 20% of episodes — places a high-salience agent near an object its partner hasn't seen, scaffolding early referral communication.
5. REFERRAL REWARD (+0.4, Phase 6.6) + JOINT CURIOSITY BONUS (+0.15, Phase 6.7): Referral fires when agent A's recent utterance predicted agent B's novel discovery. Joint bonus fires when two curious agents (err>0.05) explore together within helping_radius. Both added to Phase 7 reward total.
Also: utterance_log and episode_step per episode; spatial_memory saved/loaded in checkpoints; load_checkpoint warning updated to 139->149 for Phase 4.
Smoke test: 50ep, ~5ep/s, 149D perception (129 env + 10 utterance slots + 10 memory dims) confirmed, all components functional.
**Last updated**: 2026-02-24
**Project root**: `C:\Users\Johnathan\ClaudeResearch\curious_agents\`
**Owner**: Johnathan (AI consciousness/emergence researcher)

---

## What Is This Project?

A multi-phase research program building curiosity-driven neural agents from scratch (~86K params each) to study emergent cognition, language grounding, and proto-metacognition. Runs on modest hardware (GTX 1080, 32GB RAM). Three agents explore a 2D structured environment, learning through intrinsic motivation (approach-only — no punishment).

**Design philosophy**: Prediction error is neutral information, not pain. Agents are drawn toward learnable complexity, not fleeing from confusion.

---

## Project Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1: Curiosity & Exploration | ✅ Complete (ep 0-5000) | Forward model, REINFORCE policy, confidence tracking |
| 2: Language Grounding | ✅ Complete | Ostensive teaching; circular-buffer vocabulary; naming loss + discrimination head |
| 3: Multi-Agent Communication | ✅ Complete | Utterance actions (15-action space); word perception slots (149D); communicative reward |
| 4: Spatial Memory + Communication Scaffolding | ✅ Active | SpatialMemory (149D perception); perception_radius 30→15; asymmetric starts; referral reward (+0.4); joint curiosity bonus (+0.15) |
| 5: Grammar / Compositionality | 🔮 Planned | 2-word utterances; referential games; disambiguation tasks |
| 6: Meta-Cognition | 🔮 Planned | Hierarchical self-modeling, awareness of awareness |
| 7: Conversation | 🔮 Planned | Human-agent dialogue grounded in shared perception |

---

## File Map (read priority order)

### 🔴 Must-read before any modification

| File | What it does |
|------|-------------|
| `agents/curious_agent.py` | **The agent**. Encoder, forward model, policy, value head, discrimination head, vocabulary, metacognitive report. All neural architecture. |
| `training/trainer.py` | **Training loop**. Multi-agent coordination, curriculum stages, reward computation, teaching integration, checkpointing, `collect_metrics`. |
| `training/language_grounding.py` | **Phase 2 teaching system**. OstensiveTeacher, WordMemory (circular buffer), WORD_CLASS_MAP, naming/discrimination loss dispatch. |
| `environment/world.py` | **The environment**. StructuredEnvironment, object library (10 objects × 15D properties), curriculum stages, perception system. |

### 🟡 Read if relevant to your task

| File | What it does |
|------|-------------|
| `analyze_run.py` | Tabular CLI analysis of JSON log files. Supports `--section core/lang`, `--every N`, `--ep-min`. |
| `analysis/visualizer.py` | 6-panel matplotlib dashboard (2×3): world map, prediction error, naming accuracy, learning progress, confidence, language losses. |
| `checkpoints/checkpoint_ep5000.pt` | Phase 1 trained weights (3 agents, 5000 episodes). Load to resume training. |
| `logs/` | Training metrics JSON files, written every 50 episodes. |
| `run.py` | Main entry point. |

### 🟢 Reference only

| File | What it does |
|------|-------------|
| `visually_grounded_metacognitive_agents.md` | Full research vision document. Architecture blueprints for all 5 phases. Pseudocode, not runnable. |

---

## Architecture Quick Reference

### Agent (`CuriousAgent`)
```
Perception (149D flat) → Encoder (3× NormedLinear+GELU → 64D Tanh)
  129D env + 10D words + 10D memory                            |
                        → Forward Model (state + 5D move-onehot → predicted next state)  [forward_model_optimizer]
                        → Policy (state → 15 action logits, softmax)                     [policy_optimizer]
                              5 movement + 10 utterance actions (word class 0-9)
                        → Value Head (state → scalar baseline)                            [policy_optimizer]
                        → Expression (state → 8D "body language")                         [policy_optimizer]
                        → Discrimination Head (state → 10 class logits)                  [language_optimizer]
                                                    ↑
                              naming loss (MSE) also flows through encoder via [language_optimizer]
```
- **~92K params** per agent (up from ~86K — policy output layer expanded 5→15)
- **15 actions**: north, south, east, west, stay, say_word_0 ... say_word_9
- **Utterance actions**: action >= n_actions (5) → no movement, sets `last_utterance_class = action - 5`
- **Forward model**: utterance actions → mapped to "stay" (action 4) before one-hot encoding
- **3 optimizers**: `policy_optimizer`, `forward_model_optimizer`, `language_optimizer` (encoder shared between policy and language, each manages its own zero_grad/step)
- **Vocabulary**: `dict[str, np.ndarray]` mapping words → circular-buffer prototype vectors
- **Key methods**: `perceive()`, `decide_action()`, `execute_action()`, `compute_prediction_error()`, `compute_intrinsic_reward()`, `learn_word()`, `try_to_name()`, `store_experience()`, `update_policy()`, `train_forward_model()`, `train_language_losses(word, class_idx)`, `get_predicted_class()`, `update_memory(nearby_object_classes, prediction_error, episode)`

### Environment (`StructuredEnvironment`)
- 100×100 toroidal world
- Objects have 15D property vectors (color RGB, size, shape, animate, edible, dangerous, warm, soft, bright, noisy, complexity, familiarity)
- 10 object types: apple, banana, cat, dog, rock, fire, water, flower, ball, book
- `get_flat_perception()` returns 129D vector (8 max objects × 16 features + 1 count)
- `get_perception_dim(n_utterance_classes=N, n_memory_classes=M)` returns 129+N+M (trainer passes N=M=10)
- 3 curriculum stages: simple (4 obj) → complex (8 obj) → dynamic (spawn/despawn/shift)
- **Trainer appends 10D utterance slots + 10D memory vec** to get full 149D perception — `get_flat_perception()` itself is unchanged

### Teaching System (`OstensiveTeacher` + `WordMemory`)
- Probabilistic teaching: each step, may "point and name" a nearby object
- Multi-exposure grounding: word needs 3+ exposures before entering vocabulary
- **`WordMemory` uses a circular ring buffer** (8 slots, `vocab_buffer_size`): `add_exposure()` appends until full, then overwrites oldest entry at `write_pos`. Prototype = `np.mean(buffer, axis=0)`. Old cumulative-sum approach is gone.
- On every teaching event for a grounded word: calls `agent.train_language_losses(word, class_idx)` which fires naming loss + discrimination loss in one backward pass
- `WORD_CLASS_MAP` and `ALL_OBJECT_CLASSES` in `language_grounding.py` define the 10-class index used by the discrimination head
- Naming tests: agent tries to match current perception to vocabulary via cosine similarity
- Vocabulary refresh: every 100 episodes, re-encodes all words with current encoder weights
- Teaching curriculum: Stage 1 teaches [apple, rock, ball, flower], Stage 2 adds [banana, cat, dog, water, fire, book]

### Reward Structure (approach-only)
```
total_reward = curiosity_reward     (max(0, prev_error - curr_error))
             + helping_reward       (others' error reduction when nearby)
             + exploration_bonus    (0.1 if moved — action < 4 only)
             + naming_reward        (0.3 if correctly named nearby object)
             + comm_reward          (0.5 × count of nearby agents whose discrimination
                                     prediction agrees with current utterance class)
             + referral_reward      (0.4 when agent's utterance led a nearby agent to
                                     a novel discovery within the last 20 steps)
             + joint_curiosity_bonus (0.15 when two curious agents explore together
                                     within helping_radius, both err > 0.05)
```
**Never negative.** Worst case = 0.

---

## How to Work Without Overflowing Context

### Rule 1: Don't read entire files upfront
Use targeted reads:
```
# Read just the class/method you need
Filesystem:read_text_file path=..., head=50   # first 50 lines
Filesystem:read_text_file path=..., tail=30   # last 30 lines

# Search for specific patterns
Desktop Commander:start_search  searchType=content  pattern="def compute_intrinsic_reward"
```

### Rule 2: Know the key line ranges (approximate)

**trainer.py** (~570 lines after Phase 3):
- Lines 1-65: Imports (now includes N_OBJECT_CLASSES, ALL_OBJECT_CLASSES), TrainerConfig (now includes communicative_reward)
- Lines 65-150: `Trainer.__init__()` (perception_dim uses n_utterance_classes, prev_utterances dict)
- Lines 150-175: `_build_utterance_slots()` helper
- Lines 175-195: `get_nearby_agents()`
- Lines 195-320: `run_step()` (phases 1-8 + 6.5 communicative reward; utterance tracking in phases 1+2+4)
- Lines 320-355: `run_episode()` (now resets prev_utterances at start)
- Lines 355-410: `collect_metrics()`, `log_metrics()`
- Lines 410-460: `save_checkpoint()`
- Lines 460-545: `load_checkpoint()` (try/except for shape mismatch)
- Lines 545-570: `train()` (main loop entry point)

**curious_agent.py** (~750 lines after Phase 2 additions):
- Lines 1-50: Docstring, imports, NormedLinear
- Lines 50-100: AgentConfig dataclass (includes n_object_classes, naming/discrimination weights)
- Lines 100-210: `CuriousAgent.__init__()` (encoder, forward model, policy, value, discrimination_head, expression, tracking vars)
- Lines 210-250: `_setup_optimizers()` (policy, forward_model, language optimizers)
- Lines 250-290: `perceive()`, `decide_action()`, `execute_action()`
- Lines 290-380: `compute_prediction_error()`, `compute_intrinsic_reward()`
- Lines 380-420: `train_forward_model()`
- Lines 420-510: `train_language_losses()`, `get_predicted_class()` ← NEW
- Lines 510-570: `store_experience()`, `update_policy()`
- Lines 570-640: `get_observable_state()`, `learn_word()`, `try_to_name()`
- Lines 640-700: `metacognitive_report()` (includes avg_naming_loss, avg_discrimination_loss)
- Lines 700-750: `apply_structured_initialization()` (includes discrimination_head init), `create_agent()`

**language_grounding.py** (~450 lines after Phase 2 additions):
- Lines 1-30: Imports
- Lines 30-50: ALL_OBJECT_CLASSES, N_OBJECT_CLASSES, WORD_CLASS_MAP ← NEW
- Lines 50-100: TeachingConfig (includes vocab_buffer_size), TEACHING_CURRICULUM
- Lines 100-170: WordMemory class (circular buffer: buffer, write_pos, add_exposure, reset_for_refresh) ← REFACTORED
- Lines 170-320: OstensiveTeacher (`teach_step` calls train_language_losses, `test_naming`, `compute_naming_reward`)
- Lines 320-450: `refresh_vocabularies()`, `get_metrics()`, `get_per_agent_metrics()`

**world.py** (~400 lines):
- Lines 1-80: Object property schema, object library
- Lines 80-150: WorldObject, StructuredEnvironment.__init__()
- Lines 150-250: `perceive_at()`, `get_flat_perception()`
- Lines 250-350: Curriculum stages, dynamic mode
- Lines 350-400: Utility methods

### Rule 3: Use search, not bulk reads
```
# Find where a specific method is called
Desktop Commander:start_search  path=C:\Users\Johnathan\ClaudeResearch\curious_agents
    pattern="refresh_vocabularies"  searchType=content

# Find a file by name
Desktop Commander:start_search  path=C:\Users\Johnathan\ClaudeResearch\curious_agents
    pattern="checkpoint"  searchType=files
```

### Rule 4: Check before creating
Before creating new files, check what already exists:
```
Filesystem:list_directory  path=C:\Users\Johnathan\ClaudeResearch\curious_agents
```
Existing scripts may already cover what you're about to write.

### Rule 5: Incremental edits over full rewrites
Use `Filesystem:edit_file` for targeted changes rather than rewriting entire files. The codebase is stable — surgical edits are safer than full replacements.

---

## Common Tasks

### Start / resume training
```bash
python run.py --episodes 1000                                    # Fresh run
python run.py --episodes 1000 --resume checkpoints/checkpoint_ep5000.pt  # Resume
python run.py --episodes 1000 --visualize --viz-freq 100         # With dashboard PNGs
```

### Analyze a completed run
```bash
python analyze_run.py                          # Full summary, all episodes
python analyze_run.py --section lang           # Phase 2 metrics only
python analyze_run.py --section core           # Curiosity/reward only
python analyze_run.py --every 100              # Thinned view
python analyze_run.py --ep-min 500             # Skip warmup
```

### Post-hoc visual from checkpoint
```python
from analysis.visualizer import TrainingVisualizer
viz = TrainingVisualizer(n_agents=3)
viz.plot_from_checkpoint('checkpoints/checkpoint_ep1000.pt', metrics_dir='logs')
```

### Run a quick smoke test
```bash
python run.py --test
```

### Check agent vocabulary state
```python
for agent in trainer.agents:
    print(f"Agent {agent.config.agent_id}: {list(agent.vocabulary.keys())}")
```

### Verify imports resolve
```bash
python -c "import sys; sys.path.insert(0,'.'); from training.trainer import Trainer, TrainerConfig; print('OK')"
```

---

## Key Design Decisions to Preserve

1. **Approach-only motivation**: Rewards are never negative. Don't add penalties.
2. **Multi-exposure grounding**: Words need 3+ exposures before entering vocabulary. Don't skip this.
3. **Circular-buffer vocabulary**: `WordMemory.buffer` is a fixed-size ring (8 slots). Do NOT revert to cumulative sum — that causes new sightings to have vanishing influence as exposure count grows.
4. **Vocabulary refresh**: Re-encode words as encoder evolves. The teacher handles this automatically.
5. **Three separate optimizers**: `policy_optimizer` (encoder+policy+value+expression), `forward_model_optimizer`, `language_optimizer` (encoder+discrimination_head). Each calls its own `zero_grad()`/`step()` — they can share encoder params safely because they never run concurrent backward passes.
6. **Naming loss fires only on grounded words**: `train_language_losses` checks `word in self.vocabulary` before computing naming MSE. No prototype = no naming loss, but discrimination loss still fires.
7. **Structured initialization**: Agents get biological-style weight init, not random. See `apply_structured_initialization()` — discrimination head + utterance bias init included.
8. **store_experience() signature**: `(perception, action, reward)` — 3 required params, no optionals. Phase 6: stores raw numpy perception so update_policy() can replay the full episode trajectory through encoder → GRU with fresh gradients.
9. **load_state_dict strict=False**: Allows loading pre-discrimination-head checkpoints cleanly. Keep this.
10. **n_object_classes must match N_OBJECT_CLASSES**: `AgentConfig.n_object_classes = 10` must stay in sync with `ALL_OBJECT_CLASSES` in `language_grounding.py`. If curriculum changes, update both.
11. **Utterance → "stay" mapping**: `execute_action(action >= n_actions)` sets `last_utterance_class` but does NOT move the agent. `compute_prediction_error` and `train_forward_model` both map `forward_action = min(action, n_actions-1)` so the forward model's one-hot dim stays at 5 (backward compatible).
12. **Utterance perception is one step delayed in Phase 1**: `prev_utterances` (last step) is used for the FIRST perceive call (action decision). `step_utterances` (this step) is used for the SECOND perceive call (observation). This is realistic — agents decide what to say before observing simultaneous utterances.
13. **Communicative reward requires trained discrimination heads**: `comm_reward` depends on `other.discrimination_head(other.internal_state)` returning a meaningful prediction. Early in training (random heads) it is near-zero. This is correct — the signal grows as Phase 2 grounding progresses.
14. **Entropy regularization**: `update_policy()` includes `-entropy_coeff * dist.entropy()` per sample in the loss. This prevents REINFORCE from concentrating logits so heavily on movement that utterance actions become unreachable. Without it, by ep 2000 the logit gap was ~1.4 (movement ~+1.5, utterances ~-0.9), and utterance rate fell from 50% to 5% despite temperature=0.6. entropy_coeff=0.02 closed the gap ~0.2 units per 2000 eps — raised to 0.05 for faster recovery.

---

## Gotchas

- **PowerShell**: Use `;` not `&&` to chain commands
- **sys.path**: The trainer already does `sys.path.insert(0, ...)` — don't duplicate
- **Checkpoint format (updated Feb 2026)**: `WordMemory` serializes as `{'buffer': [[...], ...], 'write_pos': int, 'exposures': int, 'grounded': bool, ...}`. The old `state_accumulator` key no longer exists in new checkpoints. `load_checkpoint` handles both formats for backward compat.
- **Checkpoint loading**: Uses `strict=False` so old checkpoints (without `discrimination_head` weights) load cleanly. A shape mismatch (e.g. Phase 3→4 migration where encoder changed 139→149) raises RuntimeError — `load_checkpoint` catches this and re-applies `apply_structured_initialization` for that agent with a printed warning. `spatial_memory` key is optional — missing = fresh SpatialMemory().
- **Phase 2→3 checkpoint break**: Old ep0-999 checkpoints have encoder input dim 129; Phase 3 uses 139. Triggers RuntimeError fallback. Phase 3→4 same: 139→149. Start a new run for clean Phase 4 training.
- **Perception dim**: 149 = 129 base + 10 utterance slots + 10 memory dims. Base: 8 max objects × (1 distance + 15 properties) + 1 count. Utterance slots and memory vec: appended by `_build_perception()` in Trainer, NOT by `get_flat_perception()`. If you change max_objects or property count, update both the base calc and the 149 default in `create_agent`.
- **Temperature**: Decays per-episode via `temp *= decay`. Starts at 2.0, floor now 0.6 (raised from 0.3 — 0.3 caused utterances to collapse to ~5% at late training as policy went near-greedy). `load_checkpoint` uses `max(checkpoint_temp, config.temperature_end)` so resumed runs respect the new floor.
- **language_optimizer backward timing**: `train_language_losses()` is called from `OstensiveTeacher.teach_step()` inside `run_step()`, after the second `perceive()` call. At that point `agent.internal_state` still has a live computation graph. The backward call there consumes the graph — `train_forward_model()` called later uses `.detach()` so there's no conflict.
- **Discrimination head class count**: If you add objects to the curriculum beyond the current 10, add them to `ALL_OBJECT_CLASSES` in `language_grounding.py` AND update `AgentConfig.n_object_classes`. Out-of-sync values will cause silent mismatch (discrimination head produces wrong number of logits).

---

## Transcripts

Previous conversation transcripts are stored at:
`/mnt/transcripts/` (Claude's filesystem, not Windows)

Check `journal.txt` in that directory for a catalog of past sessions.
