# Pokémon TCG AI — Improvement Plan

> Generated: 2026-08-05  
> Based on: `reinforcement-learning-and-mcts-sample-code.ipynb`

---

## Table of Contents

1. [Model Architecture & State Representation](#1-model-architecture--state-representation)
2. [Action Space & Reward Function](#2-action-space--reward-function)
3. [Monte Carlo Tree Search](#3-monte-carlo-tree-search)
4. [Deck Construction & Card Evaluation](#4-deck-construction--card-evaluation)
5. [Training Loop & Optimization](#5-training-loop--optimization)
6. [Code Quality & Bug Fixes](#6-code-quality--bug-fixes)
7. [Priority Summary](#7-priority-summary)

---

## 1. Model Architecture & State Representation

### 1a. Critical missing features from the API

The API exposes many fields that are completely absent from the current encoder. These are high-value, zero-cost additions.

#### `Pokemon` fields not encoded

| Field | Why it matters |
|---|---|
| `poke.hp / poke.maxHp` | Remaining HP *ratio* — a Pokémon at 30/30 looks very different from 30/300 |
| `poke.appearThisTurn` | If `True`, the Pokémon can't retreat and often can't attack — critical for threat assessment |
| `poke.energies` (list of `EnergyType`) | Actual energy *types*, not just card IDs; special energies providing multiple types are silently mislabeled |
| `poke.preEvolution` | Knowing whether something can still evolve affects tempo planning |

#### `State` fields not encoded

| Field | Why it matters |
|---|---|
| `state.supporterPlayed` | Can you still play Iono/Research this turn? |
| `state.energyAttached` | Is the manual attachment spent? |
| `state.retreated` | Is retreat still available this turn? |
| `state.turnActionCount` | Proxy for how much has happened this turn |

#### `CardData` fields available via `card_table` but not used

| Field | Why it matters |
|---|---|
| `retreatCost` | Retreat pressure is a core competitive concept — a Charizard ex stuck active with no energy is dead |
| `weakness` / `resistance` | Type matchup is essential for threat evaluation |
| `ex` / `megaEx` | 2-prize and 3-prize targets — worth encoding separately |
| `tera` | Benched Tera Pokémon take no bench damage — affects placement decisions |
| `basic` / `stage1` / `stage2` | Stage matters for evolution readiness |

---

### 1b. HP encoding fix (Bug)

```python
# Current (wrong — 400 is not max HP for all Pokémon):
sv.add_single(poke.hp / 400)

# Fixed:
sv.add_single(poke.hp / poke.maxHp)
```

- [ ] **TODO:** Apply this fix in `add_pokemon()`

---

### 1c. Add turn state word to encoder

Add a dedicated encoder word encoding:
- `supporterPlayed`
- `energyAttached`
- `retreated`
- `turnActionCount`

These are currently absent. `num_words_encoder` will need to increase from 24.

- [ ] **TODO:** Add turn state word in `get_encoder_input()`

---

### 1d. Prize mapping

Currently only `len(ps.prize)` is encoded. The API exposes `prize: list[Card | None]` where face-up prizes are visible.

- Add the **prize differential** as an explicit scalar: `(opponent_prizes_remaining - your_prizes_remaining) / 6`
- Encode the types/card IDs of *known* face-up prize cards separately

> As a competitive player: seeing a VS Seeker or Boss's Orders in prizes completely changes your win condition plan.

- [ ] **TODO:** Add prize differential scalar to the misc word

---

### 1e. Encoder architecture depth

Currently only **1 `TransformerEncoderLayer`**. Increase to **2–3 layers** with the same `d_model=128`. The model is tiny; cost is negligible and depth matters for relational reasoning across board slots.

```python
# Current:
model = MyModel(128, 2, 256, 1, 1)

# Suggested:
model = MyModel(128, 2, 256, 3, 2)
```

- [ ] **TODO:** Increase encoder/decoder depth

---

### 1f. Decoder: encode remaining energy cost

`SelectData.remainEnergyCost` is available but not encoded. When the model is selecting energies to discard for a retreat, it doesn't know how many more are needed.

- [ ] **TODO:** Add `remainEnergyCost` to decoder word in `get_decoder_input()`

---

## 2. Action Space & Reward Function

### 2a. Where the action space is defined

The combinatorial enumeration is in `create_node()`:

```python
indices = list(range(obs.select.maxCount))
for _ in range(64):
    actions.append(indices.copy())
    # next-combination logic...
```

This generates up to 64 combinations in **lexicographic order** — the first 64 may not include the best actions (e.g., attack options may be far into the list).

**Fix:** Sort/filter options before enumerating with domain priority:
1. `ATTACK` options first (highest base damage first via `card_table`)
2. `PLAY` options for Supporters
3. `EVOLVE` options
4. `ATTACH` options
5. `END` last

- [ ] **TODO:** Implement priority-ordered action enumeration

---

### 2b. Where the reward function is computed

**Terminal rewards** — in `create_node()`:
```python
if state.result == 2: node.value = 0
elif state.result == your_index: node.value = 1
else: node.value = -1
```

**TD target** — in the self-play loop:
```python
label = (value + sample.value) * 0.5
value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
```

There is **no intermediate reward shaping at all.**

---

### 2c. Reward shaping suggestions

These map directly to heuristics a strong human player uses:

| Heuristic | Suggested shape | Competitive rationale |
|---|---|---|
| Prize differential | `+0.15 × (opp_prizes − your_prizes)` per turn | Taking prizes is the win condition |
| KO bonus | `+0.2` when you take a prize | Immediate feedback on correct aggression |
| 2-prize KO | `+0.1` extra for `ex`/`megaEx` KOs | Accelerated prize race is the backbone of modern TCG |
| Energy denial | `+0.05` when opponent's energy is discarded via your effect | Knocking out a charged vs uncharged Charizard is very different |
| Setup penalty | `−0.03` per turn where active can't attack | Tempo loss is concrete |
| Bench pressure | `+0.02` for full bench (5) vs opponent | Resource advantage |
| Paralysis/Sleep landing | `+0.03` | Temporal denial — often decisive |

- [ ] **TODO:** Implement reward shaping using `obs.logs` to detect KOs, energy discards

---

### 2d. Using Search API for reward shaping

`search_step` returns full `SearchState` including `logs` (`LogType.HP_CHANGE`, `LogType.RESULT`). You can detect after each search step whether a KO occurred and use it as an **exact shaped reward** rather than a heuristic.

- [ ] **TODO:** Parse `search_state.observation.logs` after `search_step` calls to detect KOs

---

## 3. Monte Carlo Tree Search

### 3a. Search count is far too low

`SEARCH_COUNT = 10` — the tree has at most 10 nodes. This is essentially random play with one look-ahead step.

| Environment | Suggested minimum |
|---|---|
| Local CPU (development) | 50 |
| Local CPU (full run) | 100–200 |
| Kaggle T4 GPU | 200–400 |

- [ ] **TODO:** Increase `SEARCH_COUNT` to at least 50 for development, 200 for Kaggle

---

### 3b. UCB exploration constant

Current: `c = 0.4 * math.sqrt(current.visit)`

At `0.4` the model barely explores alternatives. Standard AlphaZero uses `c_puct = 1.0–2.0`.

```python
# Suggested:
c = 1.25 * math.sqrt(current.visit)
```

- [ ] **TODO:** Tune `c_puct` constant

---

### 3c. `next` and `sum` shadow Python builtins

```python
# Bug — shadows Python built-in next():
next = child

# Bug — shadows Python built-in sum():
sum = 0.0
```

These will cause confusing bugs if Python's built-ins are ever called in the same scope.

- [ ] **TODO:** Rename to `best_child` and `prob_sum`

---

### 3d. Root Dirichlet noise for exploration

Without exploration noise, self-play converges to a single strategy very quickly.

```python
import numpy as np
noise = np.random.dirichlet([0.3] * len(root.children))
for i, child in enumerate(root.children):
    child.prob = 0.75 * child.prob + 0.25 * noise[i]
```

Add this after the root node is created in `mcts_agent()`.

- [ ] **TODO:** Add Dirichlet noise at MCTS root

---

### 3e. Domain-biased priors

Before normalizing action probabilities in `create_node()`, apply a multiplicative bonus:

| Condition | Multiplier | Rationale |
|---|---|---|
| Attack can KO opponent's active (HP check via `card_table`) | ×2.0 | Finishing a KO is always the priority play |
| Supporter play | ×1.5 | Draw power is universally strong |
| Energy attachment to active | ×1.3 | Powering up your attacker is correct tempo |

These are zero-inference-cost heuristics that dramatically improve MCTS signal quality at low search counts.

- [ ] **TODO:** Implement prior boosting in `create_node()`

---

## 4. Deck Construction & Card Evaluation

### 4a. Deck location

```python
# Cell 4, near the bottom:
sample_deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,
               1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,
               3,3,3,...(×33)]
```

The deck has **33 basic energy (card ID 3)**. No competitive deck runs more than 10–12. This means:
- The agent draws dead energy constantly instead of Trainers
- The model will never learn to sequence draw Supporters because there are none

---

### 4b. Suggested deck structure for training stability

Start with a **simple, consistent deck** to give the RL loop useful signal:

| Slot | Count | Purpose |
|---|---|---|
| Basic attacker (1 line) | 4 | Consistent attacker |
| Draw Supporter (e.g., Research equivalent) | 4 | Teach supporter sequencing |
| Search Item (e.g., Nest Ball equivalent) | 4 | Teach searching for Pokémon |
| Basic Energy | 8–10 | Enough to attack, not a dead draw |
| Recovery / Switching cards | Remainder | Board management |

Then increase deck complexity as training stabilizes.

- [ ] **TODO:** Redesign `sample_deck` with competitive ratios

---

### 4c. Card evaluation features for the model

Using `card_table` (already loaded), add card-type-based features to the encoder:

```python
card_data = card_table.get(card.id)
if card_data:
    sv.add_single(card_data.cardType == CardType.SUPPORTER)
    sv.add_single(card_data.retreatCost / 4)
    sv.add_single(card_data.ex or card_data.megaEx)
```

- [ ] **TODO:** Add `card_table` lookups to encoder feature functions

---

## 5. Training Loop & Optimization

### 5a. Broken print statement (Bug)

At the end of the training loop, the epoch summary print is malformed — f-strings appear without a `print(` call:

```python
# Current (broken):
    remaining_secs = elapsed_secs / (counter + 1) * (TOTAL_EPOCHS - counter - 1)
          f"Games: {total_games} | "
          f"Elapsed: {elapsed_secs/60:.1f}m | "
          f"ETA: {remaining_secs/60:.1f}m", flush=True)

# Fixed:
    print(
        f"Epoch: {counter+1}/{TOTAL_EPOCHS} | "
        f"Games: {total_games} | "
        f"Elapsed: {elapsed_secs/60:.1f}m | "
        f"ETA: {remaining_secs/60:.1f}m",
        flush=True
    )
```

- [ ] **TODO:** Fix broken print statement at end of training loop

---

### 5b. Missing gradient clipping

With sparse inputs and Huber loss, gradient spikes can destabilize training:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # add this line
optimizer.step()
```

- [ ] **TODO:** Add gradient clipping

---

### 5c. Learning rate schedule

Flat `3e-4` throughout. Add cosine annealing for better convergence:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

# After each epoch (end of outer for loop):
scheduler.step()
```

- [ ] **TODO:** Add LR scheduler

---

### 5d. Self-play volume is very low

100 games × 5 epochs = 500 total games. AlphaZero-style training requires orders of magnitude more. Suggested minimums:

| Setting | Games/epoch | Epochs |
|---|---|---|
| Quick test | 100 | 5 |
| Meaningful training | 500 | 20 |
| Kaggle submission | 1000+ | 50+ |

- [ ] **TODO:** Increase `self_play_games` and `TOTAL_EPOCHS`

---

### 5e. Evaluation opponent

The current eval uses `random_agent` (uniform random legal moves). This gives a noisy win-rate signal.

Upgrade to a **heuristic agent** using this priority:
1. Attack if possible (highest damage attack)
2. Attach energy to active
3. Play a Supporter
4. End turn

- [ ] **TODO:** Implement `heuristic_agent()` for evaluation

---

## 6. Code Quality & Bug Fixes

| # | Issue | Location | Fix |
|---|---|---|---|
| 🐛 | `hp / 400` wrong normalization | `add_pokemon()` | Use `poke.hp / poke.maxHp` |
| 🐛 | Broken print statement | End of training loop | Add `print(f"Epoch: ..." ...)` |
| 🐛 | `next` shadows built-in | `mcts_agent()` MCTS loop | Rename to `best_child` |
| 🐛 | `sum` shadows built-in | `create_node()` | Rename to `prob_sum` |
| ⚠️ | `os.path.dirname(os.path.abspath("__file__"))` | Cell 2 | `"__file__"` is a string literal — use `os.getcwd()` or hardcode path |
| ⚠️ | `import glob` unused | Cell 4 | Remove |
| ⚠️ | No random seed | Top of notebook | Add `random.seed(42); torch.manual_seed(42)` |
| ⚠️ | No logging | Training loop | Track loss/win-rate per epoch, plot after training |
| ℹ️ | `import time` only in Cell 4 | Cell 2 | Add `import time` to Cell 2 if cells run independently |

---

## 7. Priority Summary

Work through these in order — earlier items have the highest return-on-effort ratio:

| # | Task | Impact | Effort |
|---|---|---|---|
| 1 | Fix HP encoding bug (`hp/400` → `hp/maxHp`) | High | Trivial |
| 2 | Fix broken print + `battle_finish()` | Correctness | Trivial |
| 3 | Add missing turn state features (`supporterPlayed`, `energyAttached`, `retreated`) | High | Low |
| 4 | Rename `next`/`sum` to avoid shadowing builtins | Correctness | Trivial |
| 5 | Increase `SEARCH_COUNT` to 50+ | High | Trivial |
| 6 | Fix the deck (reduce to 8–10 energy, add Supporters) | High | Low |
| 7 | Add prize differential reward shaping | High | Low |
| 8 | Add Dirichlet root noise to MCTS | Medium | Low |
| 9 | Encode `ex`/`weakness`/`retreatCost` from `card_table` | Medium | Medium |
| 10 | Increase self-play games and epochs | High | Config change |
| 11 | Domain-biased MCTS priors | Medium | Medium |
| 12 | Deeper model (2–3 encoder layers) | Medium | Low |
| 13 | Gradient clipping + LR schedule | Medium | Low |
| 14 | Upgrade eval opponent to heuristic agent | Medium | Medium |
| 15 | Priority-ordered action enumeration | Medium | Medium |

---

*This document tracks the improvement plan discussed on 2026-08-05. Check off TODOs as you implement them.*
