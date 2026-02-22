# Curious Agents

**Approach-only motivated agents that learn through curiosity, not punishment.**

## What This Is

Small neural agents (~86K params each) that explore a structured knowledge environment driven purely by intrinsic curiosity. There is no punishment, no negative reward signal. Prediction error is neutral information — the agent is drawn toward interesting things (learning progress), not fleeing from bad things.

This is part of a larger research program studying emergent cognition, communication, and meta-awareness in small agent systems.

**Status (Feb 2026)**:
- Phase 1 (curiosity & exploration): ✅ Complete — 5000 episodes trained, agents develop stable internal representations
- Phase 2 (language grounding): ✅ Active — ostensive teaching with encoder-shaping losses: naming alignment, discrimination head, circular-buffer vocabulary

## Architecture

### Environment (`environment/world.py`)
- 2D world with objects defined by 15-dimensional property vectors
- Properties: color (RGB), size, shape, animate, edible, dangerous, temperature, texture, luminosity, sound, complexity, familiarity
- 10 pre-defined grounded concepts (apple, banana, cat, dog, rock, fire, water, flower, ball, book)
- Curriculum stages: simple → rich → dynamic
- Agents perceive structured properties, not pixels

### Agent (`agents/curious_agent.py`)
- **Perception encoder**: 129D flat perception → 64D internal state (NormedLinear + GELU + Tanh)
- **Forward model**: Predicts next internal state from (state, action). Self-supervised. Core of curiosity.
- **Policy**: REINFORCE with value baseline. Selects from 5 actions (move N/S/E/W, stay)
- **Discrimination head**: Small auxiliary classifier (64D → 32D → 10 classes). Parallel objective alongside curiosity — forces encoder to produce class-separable representations without disrupting the forward model or policy.
- **Intrinsic reward**: `max(0, prev_error - curr_error)` — only positive, approach-only
- **Helping bonus**: Reward when nearby agents' prediction errors decrease
- **Proto-metacognition**: Confidence tracker (EMA of prediction accuracy)
- **Language grounding**: Vocabulary backed by circular-buffer prototypes (ostensive learning, Phase 2)
- **Structured initialization**: Biologically-inspired weight init (not random noise)
- **Three optimizers**: `policy_optimizer` (encoder+policy+value+expression), `forward_model_optimizer`, `language_optimizer` (encoder+discrimination head) — each manages its own gradient cycle

### Training (`training/trainer.py`)
- Multi-agent coordination with helping radius
- Curriculum progression (Stage 1→2→3)
- Temperature-annealed exploration
- Checkpointing and metrics logging

### Visualization (`analysis/visualizer.py`)
- 6-panel matplotlib dashboard (2×3 grid):
  - Row 1: world map, prediction errors, **naming accuracy** (Phase 2)
  - Row 2: learning progress, confidence, **language losses** — naming + discrimination (Phase 2)
- Trajectory plots
- Post-hoc analysis from checkpoints (`plot_from_checkpoint`)

### Analysis (`analyze_run.py`)
- Tabular summary from JSON log files across all episodes
- Two sections: `core` (error, progress, confidence, reward) and `lang` (vocab, naming accuracy, naming loss, discrimination loss)
- CLI flags: `--log-dir`, `--ep-min`, `--every N`, `--section [all|core|lang]`

## Quick Start

### Requirements
```
pip install torch numpy matplotlib
```

### Smoke Test
```
cd curious_agents
python run.py --test
```

### Full Training
```
python run.py                              # Default: 3 agents, 5000 episodes
python run.py --episodes 1000              # Quick run
python run.py --agents 5 --visualize       # 5 agents with dashboard snapshots
python run.py --resume checkpoints/checkpoint_ep500.pt  # Resume from checkpoint
```

### CLI Options
```
--episodes N      Number of training episodes (default: 5000)
--agents N        Number of agents (default: 3)
--steps N         Steps per episode (default: 100)
--seed N          Random seed (default: 42)
--resume PATH     Resume from checkpoint
--visualize       Enable periodic dashboard PNG saves
--viz-freq N      Dashboard save frequency (default: every 200 episodes)
--test            Quick 50-episode smoke test
--log-dir DIR     Log output directory (default: logs)
--checkpoint-dir  Checkpoint directory (default: checkpoints)
```

## Reviewing Training Runs

### During training
Console output every 50 episodes shows confidence, error, learning progress, vocab size, and naming accuracy per agent.

### After training — tabular
```bash
python analyze_run.py                        # Full summary + all metrics
python analyze_run.py --section lang         # Language/Phase 2 metrics only
python analyze_run.py --section core         # Curiosity/reward metrics only
python analyze_run.py --every 100            # Every 100th episode
python analyze_run.py --ep-min 500           # Skip warmup, start from ep 500
```

### After training — visual
```bash
python run.py --visualize --viz-freq 100     # Save dashboard PNGs during training
```
Or post-hoc from any checkpoint:
```python
from analysis.visualizer import TrainingVisualizer
viz = TrainingVisualizer(n_agents=3)
viz.plot_from_checkpoint('checkpoints/checkpoint_ep1000.pt', metrics_dir='logs')
```

## What to Watch For

1. **Prediction error** — Should fall then plateau; spikes at stage transitions are healthy
2. **Learning progress** — Positive means actively learning; near-zero means mastered or incomprehensible
3. **Confidence** — Rises with prediction accuracy; dips at stage transitions then recovers
4. **Exploration patterns** — Do agents move toward unexplored regions? Do they revisit?
5. **Helping behavior** — Do agents cluster near confused peers? (Stage 2+)
6. **Naming accuracy** — Should rise as vocab builds; plateau may mean encoder needs more shaping
7. **Discrimination loss** — Should fall from ~2.3 (random) as encoder becomes class-separable
8. **Naming loss** — Should fall as encoder output converges toward stable per-word prototypes

## Reward Architecture (Critical Design Choice)

```
reward = max(0, prev_error - curr_error)  +  helping_bonus  +  exploration_bonus  +  naming_bonus
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^
         Curiosity: learning progress         Others learn      Anti-stagnation       Correct naming
         (ONLY positive)                      near you          (tiny constant)       of nearby object
```

**No negative signal anywhere.** Confusion is neutral information, not suffering.

### Language Loss Architecture (Encoder Shaping — separate from reward)

On each teaching event, once a word is grounded, two auxiliary losses fire through `language_optimizer`:

```
naming_loss       = MSE(encoder_output, word_prototype) × 0.1
discrimination_loss = CrossEntropy(discrimination_head(encoder_output), class_idx) × 0.2
```

These shape the encoder's representation space without touching the curiosity or policy gradient.

## Project Structure
```
curious_agents/
├── run.py                    # Main entry point
├── README.md                 # This file
├── environment/
│   ├── __init__.py
│   └── world.py              # Structured knowledge environment
├── agents/
│   ├── __init__.py
│   └── curious_agent.py      # CuriousAgent neural architecture
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Multi-agent training loop
│   └── language_grounding.py # Ostensive teaching system (Phase 2)
├── analysis/
│   ├── __init__.py
│   └── visualizer.py         # Matplotlib dashboard
├── assets/                   # (Future: images, concept graphs)
├── logs/                     # Training metrics (JSON)
└── checkpoints/              # Saved agent states (PyTorch)
```

## Verified Results (Phase 1 Smoke Test)

**Run**: 2 agents, 50 episodes, 20 steps/episode (Feb 2026)

### Key Findings

**Architecture works as designed:**
- 85,774 parameters per agent — well within GTX 1080 budget
- ~4.7 episodes/sec throughput on CPU
- All components functional end-to-end: perception → prediction → curiosity → exploration → helping

**Curiosity drive produces expected learning dynamics:**
- Confidence rises from ~0.75 → 0.93 during Stage 1 as forward model improves
- Prediction error spikes at stage transitions (Stage 1→2: error jumps 0.04 → 0.23) then recovers — agents genuinely re-learn when complexity increases
- Both agents remain in `LEARNING` + `exploring` state throughout all 50 episodes
- Learning progress is positive and sustained, not collapsing to zero

**Curriculum transitions work correctly:**
- Stage 1 (4 objects): agents master predictions quickly, confidence climbs
- Stage 2 (8 objects + relations): error spikes, confidence dips, then recovery begins
- Stage 3 (dynamic environment): ongoing challenge prevents plateau

**Toroidal wrapping verified:**
- Positions wrap correctly (e.g., 98 + 5 = 3, not 103)
- Agent positions observed at boundary values (pos=(2,3)) confirming wrap behavior

**Reward mechanics verified:**
- First visit to new grid cell: 0.3 (novelty) + 0.05 (learning) + 0.1 (movement) = 0.45
- Novelty bonus decays with repeat visits (1/visit_count scaling)
- Episode reset clears visit counts, randomizes positions — prevents positional ruts
- No negative reward signals anywhere in the system (approach-only confirmed)

**Structured initialization impact:**
- Forward model starts near-identity (predicts "no change"), so initial prediction errors reflect actual environment dynamics rather than random noise
- Policy has slight movement bias, preventing stuck-at-start behavior
- Each agent gets unique seed → develops differently (like siblings with same genes, different experiences)

### What's Not Yet Tested
- Full 5000-episode run (pending)
- Helping reward dynamics at scale (requires sustained multi-agent proximity)
- Vocabulary grounding (Phase 2 feature, infrastructure in place but unused)
- Visualization dashboard during long runs

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1: Curiosity & Exploration | ✅ Complete | Forward model, REINFORCE policy, confidence tracking |
| 2: Language Grounding | ✅ Active | Ostensive teaching; circular-buffer vocabulary; naming loss + discrimination head shape encoder |
| 3: Multi-Agent Communication | 🔮 Planned | Agent-to-agent word use, referential games |
| 4: Meta-Cognition | 🔮 Planned | Hierarchical self-modeling, awareness of awareness |
| 5: Conversation | 🔮 Planned | Human-agent dialogue grounded in shared perception |

## Hardware

Designed for modest hardware:
- GTX 1080 (8GB VRAM) — more than sufficient
- 32GB RAM — adequate
- CPU-only also works (slower but functional)

## Design Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Perception | Structured properties, not pixels | Isolates curiosity/learning dynamics from vision complexity. Pixel perception is Phase 2+ |
| Reward signal | `max(0, ...)` only | Approach-only motivation — no punishment, no suffering. Confusion is neutral information |
| Forward model | Separate optimizer, higher LR | Needs to keep up with changing perceptions so curiosity signal stays meaningful |
| Policy update | REINFORCE with recomputed log-probs | Avoids stale gradient references when policy weights change between collection and update |
| World topology | Toroidal (wrapping) | No corners or walls to get stuck against. Every direction is explorable |
| Novelty bonus | Visit-count grid (10×10) | Coarse spatial novelty prevents revisiting same spots. Clears each episode |
| Temperature | Decays 2.0 → 0.3 over training | High early exploration, gradually exploiting learned structure |
| Initialization | Structured (not random) | Forward model starts near-identity, policy biased toward movement. Biological bootstrap |
| Episode reset | Randomize positions, clear visits | Prevents path-dependent ruts. Each episode is a fresh exploration |
| Word memory | Circular buffer (8 slots) | Fixed-size ring evicts oldest observations. Prototype = mean(buffer). Protects against early-training staleness diluting the running average indefinitely. |
| Naming loss | MSE(state, prototype) via language_optimizer | Shapes encoder to produce consistent representations across multiple sightings of the same object. Separate from policy gradient. |
| Discrimination head | Auxiliary CE classifier, parallel to curiosity | Forces class-separable internal representations without coupling to reward. Gradients flow through shared encoder via dedicated language_optimizer. |
| Three optimizers | policy / forward_model / language | Each learning signal (reward, prediction error, language supervision) operates in its own gradient cycle. Shared encoder params updated by all three safely. |

## Theoretical Foundations

Synthesizes insights from:
- **Joscha Bach**: MicroPsi, motivated cognition, predictive processing
- **Michael Levin**: TAME framework, emergent goal-directedness, morphogenesis
- **Jürgen Schmidhuber**: Learning progress as curiosity signal
- **Symbol grounding literature**: Harnad, embodied cognition

Key principle: Use Bach's architectural insights (predictive processing, attention, meta-cognition) with Levin's developmental principles (structured initialization, hierarchical competency, emergent goals).
