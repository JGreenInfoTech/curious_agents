# Phase 6: Recurrent Architecture & Food Events

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add a GRU recurrent hidden state to each agent and introduce time-limited food events, creating conditions for emergent compositional utterance sequences (noun + property) without explicitly rewarding the composition.

**Architecture:** GRU cell inserted between encoder and all downstream heads. Hidden state `h_t` (64D) replaces raw encoder output as the input to policy, value, naming, and discrimination heads. Forward model now predicts next hidden state. Food events fire in 30% of episodes — a random edible object pays a large proximity bonus for 20 steps, creating urgency for peers to broadcast and arrive in time.

**Starting fresh:** No checkpoint migration. GRU and feedforward architectures are incompatible. New checkpoint dir (`checkpoints_phase6/`) and log dir (`logs_phase6/`). All Phase 5 world objects and reward signals are retained.

**Tech Stack:** Python, PyTorch, existing world/trainer/agent architecture. GRU via `torch.nn.GRUCell`. No perception dim changes. No new optimizer. Full-episode BPTT (100 steps, tractable).

---

## Architecture Detail

### GRU placement

```
Current:  perception(149D) → encoder → internal_state(64D) → heads
Phase 6:  perception(149D) → encoder → encoded(64D) → GRUCell → h_t(64D) → heads
```

- `GRUCell(input_size=64, hidden_size=64)`
- `h_0 = torch.zeros(64)` at episode start, reset each episode
- All heads (policy 20D, value 1D, naming, discrimination) receive `h_t` instead of raw encoder output
- Net param increase: ~24K (3 gates × 64×64 + bias). Total stays under 200K.

### Forward model change

Previously predicted next encoder output. Now predicts next hidden state:

```python
predicted_h_next = forward_model(h_t, action_one_hot_5d)
prediction_error = ||predicted_h_next - actual_h_next||²
```

The target is richer than before — `h_t` carries temporal context, so the forward model is learning to predict a temporally-integrated representation.

### Parameter sharing

GRU weights are shared between `policy_optimizer` and `language_optimizer` — same pattern as the encoder. Each optimizer calls its own `zero_grad/step`. No conflict.

### Update mechanics (recurrent REINFORCE)

At episode end, re-run the stored trajectory `[(perception_t, action_t, reward_t)]` through encoder+GRU to produce fresh `h_t` tensors with gradients intact. Compute REINFORCE loss on fresh policy head outputs, call `backward()` once. Gradient flows through all 100 GRU steps. No truncated BPTT needed at this episode length.

`store_experience()` stores `(perception_t, action_t, reward_t)` only — log_prob is recomputed during the update re-run, not stored at collection time.

---

## Food Events

### Mechanics

Each episode, with probability `p_event=0.3`:
- A random edible object (apple, banana, or water) becomes "active"
- Event starts at a random step between steps 10–60
- Lasts `event_duration=20` steps
- During the window: proximity to that object (radius 8) pays `event_food_bonus=+0.5/step`
- After window closes: reverts to baseline `+0.05/step`

### No explicit event perception

Agents are not told an event is active. They infer it entirely from the reward signal — a suddenly large food bonus when near the right object. The GRU hidden state is the mechanism for carrying "I just got a big bonus here" across steps and translating it into a useful utterance for peers.

### Referral chain

The existing referral reward (+0.4) applies when agent B arrives at the event object during the active window, having heard agent A's recent utterance. Full incentive chain:

```
A notices event → A broadcasts noun+property → B arrives during window
→ A earns referral reward, B earns event_food_bonus
```

No new reward signals required.

### New metrics per episode

Added to `collect_metrics()`:
- `event_active` (bool): whether a food event fired this episode
- `event_arrivals` (int): count of agents (excluding discoverer) who reached the event object during the window

---

## What We're Measuring

### Baseline recovery (ep0→5000)

First sign the GRU is working: naming accuracy climbs back to 90%+, prediction error falls, vocabulary grounds. The GRU takes longer than a feedforward net to stabilize. If naming accuracy hasn't recovered by ep5000, investigate hidden state initialization or learning rate.

### Event coordination (ep5000→8000)

`event_arrivals` per episode. Early: 0–0.2 (peers rarely arrive in time). If the GRU is learning to broadcast useful sequences: climbs toward 1.0+ (multiple peers arriving per event). This is the first coordination signal.

### Composition proxy signals

We cannot directly observe utterance pair timing without utterance logging. Two proxy signals in existing metrics:
- `ref_r` rising specifically in `event_active=True` episodes
- `putt%` and `prop_r` rising together faster than in Phase 5

### The smoking gun

`event_arrivals` correlates with `putt%` rather than `utt%` alone. Peers arrive because an agent said property-relevant things near the event object — not because it said anything at all. Property utterances predict coordination outcomes; undifferentiated utterances don't. That is functional composition without explicit reward.

### Failure modes

- GRU collapses to feedforward behavior (hidden state stays near zero)
- Agents broadcast only during self-discovered events (no peer-mediated transfer)
- `event_arrivals` plateaus at 0 despite high `putt%`

---

## Run Plan

- **Episode target:** 10,000 episodes minimum (GRU needs more time than feedforward to develop hidden state dynamics)
- **Checkpoint dir:** `checkpoints_phase6/`
- **Log dir:** `logs_phase6/`
- **Analysis interval:** every 2,000 episodes
- **Curriculum:** full 4-stage schedule from stage 1 — do not skip ahead
- **World:** 12 objects (10 base + apple_2 + cat_2) from Phase 5, retained unchanged
- **All Phase 5 rewards retained:** danger penalty, food bonus, referral reward, joint curiosity bonus, property comm reward
