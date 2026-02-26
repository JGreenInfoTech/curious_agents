# Curious Agents

**Approach-only motivated agents that learn through curiosity, not punishment.**

## What This Is

Small neural agents (~86K params each) that explore a structured knowledge environment driven purely by intrinsic curiosity. There is no punishment, no negative reward signal. Prediction error is neutral information — the agent is drawn toward interesting things (learning progress), not fleeing from bad things.

This is part of a larger research program studying emergent cognition, communication, and meta-awareness in small agent systems.

**Status (Feb 2026)**:
- Phase 1 (curiosity & exploration): ✅ Complete — 5000 episodes trained, agents develop stable internal representations
- Phase 2 (language grounding): ✅ Complete — ostensive teaching with encoder-shaping losses: naming alignment, discrimination head, circular-buffer vocabulary
- Phase 3 (multi-agent communication): ✅ Complete — utterance actions (word emission), word perception slots, communicative reward
- Phase 3.5 (property vocabulary): ✅ Complete — 5 property words (dangerous/edible/animate/warm/bright), 20-action policy, 154D perception, property discrimination head
- Phase 4 (spatial memory + communication scaffolding): ✅ Complete — SpatialMemory (154D perception with decay), asymmetric starts, directed discovery, referral reward, joint curiosity bonus
- Phase 5 (disambiguation pressure): ✅ Complete — property-varying object instances + property-consequential rewards (danger penalty, food bonus)
- Phase 6 (GRU recurrent architecture + food events): 🔄 Active — GRU hidden state (64D h_t) replaces single-step encoder state; food event bonus (+0.5/step for 20-step windows); event coordination tracking

## Architecture

### Environment (`environment/world.py`)
- 2D world with objects defined by 15-dimensional property vectors
- Properties: color (RGB), size, shape, animate, edible, dangerous, temperature, texture, luminosity, sound, complexity, familiarity
- 12 objects total (10 unique classes + poisonous apple variant + feral cat variant; variants share class index with originals to force noun+property disambiguation)
- Curriculum stages: simple → rich → dynamic
- Agents perceive structured properties, not pixels

### Agent (`agents/curious_agent.py`)
- **Perception encoder**: 154D flat perception → 64D internal state (NormedLinear + GELU + Tanh)
  - 129D environment perception + 10D object utterance slots + 5D property utterance slots + 10D spatial memory
- **GRU hidden state**: 64D h_t persists across steps within an episode; reset to zeros at episode start. Encoder output + h_{t-1} → h_t via GRUCell. Hidden states are detached during collection; BPTT runs through full trajectory during policy update (Phase 6).
- **Forward model**: Predicts next internal state from (state, movement-action). Self-supervised. Core of curiosity. Utterance actions map to "stay" for forward model purposes.
- **Policy**: REINFORCE with value baseline. Selects from **20 actions**: move N/S/E/W, stay, + 10 object-word emissions + 5 property-word emissions
- **Discrimination head**: Small auxiliary classifier (64D → 32D → 10 classes). Forces encoder to produce class-separable representations.
- **Property discrimination head**: Binary sigmoid classifier (64D → 32D → 5 properties). Learns which properties apply to current observation. BCELoss, trained on detached encoder output (Phase 3.5).
- **Intrinsic reward**: `max(0, prev_error - curr_error)` — only positive, approach-only
- **Helping bonus**: Reward when nearby agents' prediction errors decrease
- **Communicative reward**: Reward when agent's word emission matches a nearby agent's discrimination-head prediction (shared understanding signal, Phase 3)
- **Referral reward**: Reward (+0.4) when agent's recent utterance led a nearby agent to discover a novel object (Phase 4)
- **Joint curiosity bonus**: Reward (+0.15) when two agents both explore novel regions together (Phase 4)
- **Spatial memory**: SpatialMemory tracks per-class encounter salience (decaying over episodes); 10D memory vector appended to perception (Phase 4)
- **Proto-metacognition**: Confidence tracker (EMA of prediction accuracy)
- **Language grounding**: Object vocabulary + property vocabulary, both backed by circular-buffer prototypes (ostensive learning)
- **Structured initialization**: Biologically-inspired weight init (not random noise)
- **Three optimizers**: `policy_optimizer` (encoder+policy+value+expression), `forward_model_optimizer`, `language_optimizer` (encoder+discrimination head+property discrimination head) — each manages its own gradient cycle

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
- Three sections: `core` (error, progress, confidence, reward), `lang` (vocab, naming accuracy, losses), `comm` (utterance rates, referral/joint rewards, spatial memory, property vocab)
- CLI flags: `--log-dir`, `--ep-min`, `--every N`, `--section [all|core|lang|comm]`

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
python run.py                              # Default: 3 agents, 5000 episodes (logs_phase6/, checkpoints_phase6/)
python run.py --episodes 1000              # Quick run
python run.py --agents 5 --visualize       # 5 agents with dashboard snapshots
python run.py --resume checkpoints_phase6/checkpoint_ep500.pt  # Resume from Phase 6 checkpoint
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
--log-dir DIR     Log output directory (default: logs_phase6)
--checkpoint-dir  Checkpoint directory (default: checkpoints_phase6)
```

## Reviewing Training Runs

### During training
Console output every 50 episodes shows confidence, error, learning progress, vocab size, and naming accuracy per agent.

### After training — tabular
```bash
python analyze_run.py                        # Full summary + all metrics
python analyze_run.py --section lang         # Language/Phase 2 metrics only
python analyze_run.py --section core         # Curiosity/reward metrics only
python analyze_run.py --section comm         # Communication/memory/property metrics (Phase 3+)
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
9. **Utterance actions** — Do agents begin emitting words? Early training: mostly movement. Later: word emissions should increase as vocabulary grounds and communicative reward starts firing
10. **Communicative reward** — Fires when agent A's utterance class matches nearby agent B's discrimination prediction. Near-zero early (heads untrained), should rise as shared representations converge
11. **Property utterances** — Do agents emit property words? `--section comm` shows `putt%` column
12. **Property vocab size** — Should grow toward 5 as agents encounter qualifying objects; shown in `pvoc` column of `--section comm`
13. **Fire avoidance** — Fire had `dangerous=0.9` before Phase 5; expect avoidance to emerge earliest among dangerous objects as the danger penalty fires immediately on proximity
14. **Poisonous apple vs. normal apple** — Same class index, different `dangerous` property. Agents that learn to discriminate will approach normal apple and avoid poisonous variant; visible as diverging approach patterns for the same noun
15. **Feral cat vs. normal cat** — Same dynamic as apple pair; feral cat has high `dangerous`, normal cat does not
16. **Noun+property co-utterance sequences** — Near variant objects, watch for agents emitting both an object word and a property word within a few steps; early sign of compositional pressure
17. **`prop_app` trending negative then toward 0** — Column in `--section comm`. Starts near 0 (random proximity). Goes negative as danger penalty trains avoidance. Trends back toward 0 (or slightly positive) as agents successfully avoid dangerous objects and preferentially approach edible ones
18. **Event coordination rate** — `arr` column in `--section comm`. Shows how many distinct agents reached the event object during its active window. Starts near 1 (only the agent that stumbled on it). Should approach 3 as utterances reliably guide peers to the event location before it closes.
19. **Property utterances preceding peer arrivals** — In episodes with a food event, watch for agents emitting a food-related property word (edible) shortly before a second agent arrives at the event object. Early sign that property communication is providing directional value beyond object words alone.
20. **`disc_loss` stabilizing after GRU convergence** — The GRU changes the internal representation space; discrimination loss may spike briefly as the encoder adjusts to temporal context. Should re-stabilize once GRU hidden states develop consistent structure.

## Reward Architecture (Critical Design Choice)

```
reward = max(0, prev_error - curr_error)  +  helping_bonus  +  exploration_bonus  +  naming_bonus  +  prop_naming  +  comm_bonus  +  prop_comm  +  referral_bonus  +  joint_bonus  +  food_bonus  +  danger_penalty
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^     ^^^^^^^^^^     ^^^^^^^^^^     ^^^^^^^^^     ^^^^^^^^^^^^^     ^^^^^^^^^^     ^^^^^^^^^^     ^^^^^^^^^^^^^^
         Curiosity: learning progress         Others learn      Anti-stagnation       Correct obj      Correct prop   Obj utterance  Prop utterance Agent's word      Two agents     +0.05/step     -0.2/step
         (ONLY positive)                      near you          (tiny constant)       name nearby      word nearby    matches peer   matches peer   led partner to    exploring       within r=8     within r=8
                                                                                                                       disc. head     prop. head     novel discovery   together        of edible      of dangerous
                                                                                                                       (Phase 3)      (Phase 3.5)    (Phase 4)         (Phase 4)       objects        objects
                                                                                                                                                                                        (Phase 5)      (Phase 5)
```

**Danger penalty** (Phase 5): `−0.2` per step while within radius 8 of a dangerous object (fire, poisonous apple, feral cat). The first negative signal in the reward architecture — introduced to create genuine approach/avoid disambiguation pressure on properties.

**Food bonus** (Phase 5): `+0.05` per step while within radius 8 of an edible object (apple, banana, water). Mild positive signal for proximity to food sources.

**Food event bonus** (Phase 6): `+0.5` per step (replaces the baseline `+0.05`) while within radius 8 of the event object during the active 20-step event window. Fires in 30% of episodes. Creates temporal coordination pressure: agents that communicate the event location to peers unlock higher reward density for both.

**No other negative signals.** Confusion remains neutral information, not suffering.

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
├── logs_phase6/              # Training metrics (JSON)
└── checkpoints_phase6/       # Saved agent states (PyTorch)
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
| 2: Language Grounding | ✅ Complete | Ostensive teaching; circular-buffer vocabulary; naming loss + discrimination head shape encoder |
| 3: Multi-Agent Communication | ✅ Complete | Utterance actions (15-action space); word perception slots; communicative reward |
| 3.5: Property Vocabulary | ✅ Active | 5 property words; 20-action policy; 154D perception; property discrimination head; property comm reward |
| 4: Spatial Memory + Communication Scaffolding | ✅ Complete | SpatialMemory; perception_radius 30→15; asymmetric starts; referral reward (+0.4); joint curiosity bonus (+0.15) |
| 5: Disambiguation Pressure | ✅ Complete | Property-varying instances (poisonous apple, feral cat); danger penalty (−0.2/step near dangerous objects); food bonus (+0.05/step near edible objects) |
| 6: GRU Recurrent Architecture + Food Events | 🔄 Active | GRU hidden state (64D h_t) for temporal context across steps; food event bonus (+0.5/step during 20-step event window, 30% of episodes); event coordination tracking (arrivals metric) |
| 7: Grammar / Compositionality | 🔮 Planned | 2-word utterances; referential games; disambiguation tasks |
| 7: Meta-Cognition | 🔮 Planned | Hierarchical self-modeling, awareness of awareness |
| 8: Conversation | 🔮 Planned | Human-agent dialogue grounded in shared perception |

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
| Temperature | Decays 2.0 → 0.6 over training (floor raised from 0.3) | High early exploration, gradually exploiting learned structure. Floor of 0.6 keeps utterance actions viable at late training — at 0.3 the greedy policy suppressed word emissions entirely. |
| Initialization | Structured (not random) | Forward model starts near-identity, policy biased toward movement. Biological bootstrap |
| Episode reset | Randomize positions, clear visits | Prevents path-dependent ruts. Each episode is a fresh exploration |
| Word memory | Circular buffer (8 slots) | Fixed-size ring evicts oldest observations. Prototype = mean(buffer). Protects against early-training staleness diluting the running average indefinitely. |
| Naming loss | MSE(state, prototype) via language_optimizer | Shapes encoder to produce consistent representations across multiple sightings of the same object. Separate from policy gradient. |
| Discrimination head | Auxiliary CE classifier, parallel to curiosity | Forces class-separable internal representations without coupling to reward. Gradients flow through shared encoder via dedicated language_optimizer. |
| Three optimizers | policy / forward_model / language | Each learning signal (reward, prediction error, language supervision) operates in its own gradient cycle. Shared encoder params updated by all three safely. |
| Utterance actions | Policy outputs 15 logits (5 movement + 10 words) | Expands action space without touching forward model architecture. Utterances map to "stay" for the physical forward model — only movement matters for world prediction. |
| Word perception slots | 10D appended (part of 149D flat perception) | Nearby utterances appear as extra perception channels. Phase 1 perceive includes last step's utterances (deciding what to say next). Phase 4 perceive includes this step's (observing simultaneous communication). |
| Spatial memory | 10D memory vec appended (149D total) | Per-agent decaying salience per object class. Agents carry memory of what they've seen and how novel it was. Decays over ~20 episodes so stale memories don't dominate. |
| Perception radius | 15.0 (reduced from 30.0) | Tighter radius forces genuine information asymmetry — agents start at radius 35 from center and cannot directly observe each other's objects, making communication necessary. |
| Communicative reward | `+0.5` when utterance class matches nearby agent's discrimination prediction (raised from 0.2) | Approach-only shared-understanding signal. Raised from 0.2: at low temperature the policy becomes near-greedy and 0.2 was insufficient to compete with movement rewards. 0.5 gives utterances enough expected value to remain in the policy's action distribution. |
| Entropy regularization | `entropy_coeff=0.05` subtracted from policy loss | Penalizes overconfident action distributions. Without this, REINFORCE concentrates logits on movement (~+1.5) and drives utterance logits to ~−1.0 — a gap of ~1.4 that survives even at temperature 0.6. Entropy bonus directly counteracts logit concentration, keeping utterance actions statistically reachable. Raised from 0.02 after observing gap narrowing too slowly (~0.2 units per 2000 eps). |
| Utterance bias init | −0.2 for all utterance actions | Discourages random word emissions during early exploration. Agents move first, talk later as reward signal for utterances becomes meaningful. |

## Theoretical Foundations

Synthesizes insights from:
- **Joscha Bach**: MicroPsi, motivated cognition, predictive processing
- **Michael Levin**: TAME framework, emergent goal-directedness, morphogenesis
- **Jürgen Schmidhuber**: Learning progress as curiosity signal
- **Symbol grounding literature**: Harnad, embodied cognition

Key principle: Use Bach's architectural insights (predictive processing, attention, meta-cognition) with Levin's developmental principles (structured initialization, hierarchical competency, emergent goals).
