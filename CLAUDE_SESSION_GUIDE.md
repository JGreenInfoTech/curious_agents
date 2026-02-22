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
**Last updated**: 2026-02-22  
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
| 2: Language Grounding | ✅ Active | Ostensive teaching; circular-buffer vocabulary; naming loss + discrimination head |
| 3: Multi-Agent Communication | 🔮 Future | Agent-to-agent word use, referential games |
| 4: Meta-Cognition | 🔮 Future | Hierarchical self-modeling, awareness of awareness |
| 5: Conversation | 🔮 Future | Human-agent dialogue grounded in shared perception |

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
Perception (129D flat) → Encoder (3× NormedLinear+GELU → 64D Tanh)
                        → Forward Model (state+action → predicted next state)  [forward_model_optimizer]
                        → Policy (state → 5 action logits, softmax)            [policy_optimizer]
                        → Value Head (state → scalar baseline)                  [policy_optimizer]
                        → Expression (state → 8D "body language")               [policy_optimizer]
                        → Discrimination Head (state → 10 class logits)        [language_optimizer]
                                                    ↑
                              naming loss (MSE) also flows through encoder via [language_optimizer]
```
- **~90K params** per agent, 3 agents = ~270K total
- **5 actions**: north, south, east, west, stay
- **3 optimizers**: `policy_optimizer`, `forward_model_optimizer`, `language_optimizer` (encoder shared between policy and language, each manages its own zero_grad/step)
- **Vocabulary**: `dict[str, np.ndarray]` mapping words → circular-buffer prototype vectors
- **Key methods**: `perceive()`, `decide_action()`, `compute_prediction_error()`, `compute_intrinsic_reward()`, `learn_word()`, `try_to_name()`, `store_experience()`, `update_policy()`, `train_forward_model()`, `train_language_losses(word, class_idx)`, `get_predicted_class()`

### Environment (`StructuredEnvironment`)
- 100×100 toroidal world
- Objects have 15D property vectors (color RGB, size, shape, animate, edible, dangerous, warm, soft, bright, noisy, complexity, familiarity)
- 10 object types: apple, banana, cat, dog, rock, fire, water, flower, ball, book
- `get_flat_perception()` returns 129D vector (8 max objects × 16 features + 1 count)
- 3 curriculum stages: simple (4 obj) → complex (8 obj) → dynamic (spawn/despawn/shift)

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
             + exploration_bonus    (0.1 if moved)
             + naming_reward        (0.3 if correctly named nearby object)
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

**trainer.py** (~470 lines):
- Lines 1-60: Imports, TrainerConfig dataclass
- Lines 60-130: `Trainer.__init__()` (agents, env, teacher creation)
- Lines 130-240: `run_step()` (the per-step loop — phases 1-8)
- Lines 240-280: `run_episode()` (episode wrapper, vocab refresh)
- Lines 280-330: `collect_metrics()`, `log_metrics()`
- Lines 330-380: `save_checkpoint()`
- Lines 380-450: `load_checkpoint()`
- Lines 450-470: `train()` (main loop entry point)

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
7. **Structured initialization**: Agents get biological-style weight init, not random. See `apply_structured_initialization()` — discrimination head included.
8. **store_experience() signature**: `(log_prob, reward, state, action=None)` — 4 params, action is optional.
9. **load_state_dict strict=False**: Allows loading pre-discrimination-head checkpoints cleanly. Keep this.
10. **n_object_classes must match N_OBJECT_CLASSES**: `AgentConfig.n_object_classes = 10` must stay in sync with `ALL_OBJECT_CLASSES` in `language_grounding.py`. If curriculum changes, update both.

---

## Gotchas

- **PowerShell**: Use `;` not `&&` to chain commands
- **sys.path**: The trainer already does `sys.path.insert(0, ...)` — don't duplicate
- **Checkpoint format (updated Feb 2026)**: `WordMemory` serializes as `{'buffer': [[...], ...], 'write_pos': int, 'exposures': int, 'grounded': bool, ...}`. The old `state_accumulator` key no longer exists in new checkpoints. `load_checkpoint` handles both formats for backward compat.
- **Checkpoint loading**: Uses `strict=False` so old checkpoints (without `discrimination_head` weights) load cleanly — those head weights get structured init instead.
- **Perception dim**: 129 = 8 max objects × (1 distance + 15 properties) + 1 object count. If you change max_objects or property count, this cascades everywhere.
- **Temperature**: Decays per-episode via `temp *= decay`. Starts at 2.0, ends at 0.3. This controls exploration vs exploitation in action selection.
- **language_optimizer backward timing**: `train_language_losses()` is called from `OstensiveTeacher.teach_step()` inside `run_step()`, after the second `perceive()` call. At that point `agent.internal_state` still has a live computation graph. The backward call there consumes the graph — `train_forward_model()` called later uses `.detach()` so there's no conflict.
- **Discrimination head class count**: If you add objects to the curriculum beyond the current 10, add them to `ALL_OBJECT_CLASSES` in `language_grounding.py` AND update `AgentConfig.n_object_classes`. Out-of-sync values will cause silent mismatch (discrimination head produces wrong number of logits).

---

## Transcripts

Previous conversation transcripts are stored at:
`/mnt/transcripts/` (Claude's filesystem, not Windows)

Check `journal.txt` in that directory for a catalog of past sessions.
