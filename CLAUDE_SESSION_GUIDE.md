# Claude Session Guide — curious_agents Project

**Last updated**: 2026-02-21  
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
| 2: Language Grounding | ✅ Code complete, needs training | Ostensive teaching (point-and-name), vocabulary emergence |
| 3: Multi-Agent Communication | 🔮 Future | Agent-to-agent word use, referential games |
| 4: Meta-Cognition | 🔮 Future | Hierarchical self-modeling, awareness of awareness |
| 5: Conversation | 🔮 Future | Human-agent dialogue grounded in shared perception |

---

## File Map (read priority order)

### 🔴 Must-read before any modification

| File | Size | What it does |
|------|------|-------------|
| `agents/curious_agent.py` | ~16KB | **The agent**. Encoder, forward model, policy, value head, vocabulary, metacognitive report. All neural architecture lives here. |
| `training/trainer.py` | ~20KB | **Training loop**. Multi-agent coordination, curriculum stages, reward computation, teaching integration, checkpointing, metrics. |
| `training/language_grounding.py` | ~16KB | **Phase 2 teaching system**. OstensiveTeacher, WordMemory, teaching curriculum, naming tests, vocabulary refresh. |
| `environment/world.py` | ~15KB | **The environment**. StructuredEnvironment, object library (10 objects × 15D properties), curriculum stages, perception system. |

### 🟡 Read if relevant to your task

| File | What it does |
|------|-------------|
| `checkpoints/checkpoint_ep5000.pt` | Phase 1 trained weights (3 agents, 5000 episodes). Load to resume training. |
| `logs/` | Training metrics JSON files from previous runs |
| `run_training.py` | Entry point script (if it exists — check before creating a new one) |

### 🟢 Reference only (in project knowledge)

| File | What it does |
|------|-------------|
| `visually_grounded_metacognitive_agents.md` | Full research vision document. Architecture blueprints for all 5 phases. Pseudocode, not runnable. |
| `curious_agent.py` (project file) | Original agent design doc with extensive comments. The on-disk version may have diverged. |

---

## Architecture Quick Reference

### Agent (`CuriousAgent`)
```
Perception (129D flat) → Encoder (3× NormedLinear+GELU → 64D Tanh)
                        → Forward Model (state+action → predicted next state)
                        → Policy (state → 5 action logits, softmax)
                        → Value Head (state → scalar baseline)
                        → Expression (state → 8D "body language")
```
- **~86K params** per agent, 3 agents = ~258K total
- **5 actions**: north, south, east, west, stay
- **Vocabulary**: `dict[str, np.ndarray]` mapping words → internal state vectors
- **Key methods**: `perceive()`, `decide_action()`, `compute_prediction_error()`, `compute_intrinsic_reward()`, `learn_word()`, `try_to_name()`, `store_experience()`, `update_policy()`, `train_forward_model()`

### Environment (`StructuredEnvironment`)
- 100×100 toroidal world
- Objects have 15D property vectors (color RGB, size, shape, animate, edible, dangerous, warm, soft, bright, noisy, complexity, familiarity)
- 10 object types: apple, banana, cat, dog, rock, fire, water, flower, ball, book
- `get_flat_perception()` returns 129D vector (8 max objects × 16 features + 1 count)
- 3 curriculum stages: simple (4 obj) → complex (8 obj) → dynamic (spawn/despawn/shift)

### Teaching System (`OstensiveTeacher`)
- Probabilistic teaching: each step, may "point and name" a nearby object
- Multi-exposure grounding: word needs 3+ exposures before it enters vocabulary
- `WordMemory` accumulates internal state vectors across exposures, stores averaged prototype
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

**curious_agent.py** (~400 lines):
- Lines 1-50: Docstring, imports, NormedLinear
- Lines 50-80: AgentConfig dataclass
- Lines 80-180: `CuriousAgent.__init__()` (all networks, state vars)
- Lines 180-210: `perceive()`, `decide_action()`, `execute_action()`
- Lines 210-270: `compute_prediction_error()`, `compute_intrinsic_reward()`
- Lines 270-310: `train_forward_model()`, `store_experience()`, `update_policy()`
- Lines 310-350: `get_observable_state()`, `learn_word()`, `try_to_name()`
- Lines 350-400: `metacognitive_report()`, `create_agent()`, structured init

**language_grounding.py** (~400 lines):
- Lines 1-50: TeachingConfig, TEACHING_CURRICULUM
- Lines 50-100: WordMemory class
- Lines 100-250: OstensiveTeacher (`teach_step`, `test_naming`, `compute_naming_reward`)
- Lines 250-350: `refresh_vocabularies()`, `get_metrics()`, `get_per_agent_metrics()`

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

### Resume training from checkpoint
```python
trainer = Trainer(config)
trainer.load_checkpoint("checkpoints/checkpoint_ep5000.pt")
trainer.train()  # Continues from episode 5001
```

### Run a quick smoke test
```python
config = TrainerConfig(n_agents=2, n_episodes=5, steps_per_episode=20, log_freq=5, checkpoint_freq=999)
trainer = Trainer(config)
trainer.train()
```

### Check agent vocabulary state
```python
for agent in trainer.agents:
    print(f"Agent {agent.config.agent_id}: {list(agent.vocabulary.keys())}")
```

### Verify file parses without import errors
```bash
cd C:\Users\Johnathan\ClaudeResearch\curious_agents
python -c "import ast; ast.parse(open('training/trainer.py').read()); print('OK')"
```

### Check imports resolve
```bash
cd C:\Users\Johnathan\ClaudeResearch\curious_agents
python -c "import sys; sys.path.insert(0,'.'); from training.trainer import Trainer, TrainerConfig; print('OK')"
```

---

## Key Design Decisions to Preserve

1. **Approach-only motivation**: Rewards are never negative. Don't add penalties.
2. **Multi-exposure grounding**: Words need 3+ exposures before entering vocabulary. Don't skip this.
3. **Vocabulary refresh**: Re-encode words as encoder evolves. The teacher handles this automatically.
4. **Separate optimizers**: Policy optimizer (encoder+policy+value+expression) vs forward model optimizer. They have different learning rates.
5. **Structured initialization**: Agents get biological-style weight init, not random. See `apply_structured_initialization()`.
6. **store_experience() signature**: `(log_prob, reward, state, action=None)` — 4 params, action is optional.
7. **Agent position reset**: `agent.reset_position(seed)` is called each episode in trainer. If this method doesn't exist, the trainer sets `agent.position` directly.

---

## Gotchas

- **PowerShell**: Use `;` not `&&` to chain commands
- **sys.path**: The trainer already does `sys.path.insert(0, ...)` — don't duplicate
- **Checkpoint format**: Saves agent state_dicts + teacher state (word_memories, teaching_events, naming_tests). Teacher state uses custom serialization for WordMemory objects.
- **Perception dim**: 129 = 8 max objects × (1 distance + 15 properties) + 1 object count. If you change max_objects or property count, this cascades everywhere.
- **Temperature**: Decays per-episode via `temp *= decay`. Starts at 2.0, ends at 0.3. This controls exploration vs exploitation in action selection.

---

## Transcripts

Previous conversation transcripts are stored at:
`/mnt/transcripts/` (Claude's filesystem, not Windows)

Check `journal.txt` in that directory for a catalog of past sessions.
