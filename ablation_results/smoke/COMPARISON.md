# Reward ablation comparison

## Overall, across the whole panel

| arm | games | W-L-D | win_rate | wilson95 | deckout_rate | attack_rate | turns_to_win | turns_mean | win_by_cause |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 8 | 0-8-0 | 0.0% | [0.0,32.4] | 25.0% | 0.0% | n/a | 22.4 | deckout:W=0,L=2,D=0|prize:W=0,L=6,D=0 |
| deckout_penalty | 8 | 2-6-0 | 25.0% | [7.1,59.1] | 25.0% | n/a | 55.5 | 22.1 | deckout:W=2,L=0,D=0|other:W=0,L=6,D=0 |

## Per panel opponent

### vs first

| arm | games | W-L-D | win_rate | wilson95 | deckout_rate | attack_rate | turns_to_win | turns_mean | win_by_cause |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 4 | 0-4-0 | 0.0% | [0.0,49.0] | 0.0% | 0.0% | n/a | 14.5 | prize:W=0,L=4,D=0 |
| deckout_penalty | 4 | 2-2-0 | 50.0% | [15.0,85.0] | 50.0% | n/a | 55.5 | 30.5 | deckout:W=2,L=0,D=0|other:W=0,L=2,D=0 |

### vs random

| arm | games | W-L-D | win_rate | wilson95 | deckout_rate | attack_rate | turns_to_win | turns_mean | win_by_cause |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 4 | 0-4-0 | 0.0% | [0.0,49.0] | 50.0% | 0.0% | n/a | 30.2 | deckout:W=0,L=2,D=0|prize:W=0,L=2,D=0 |
| deckout_penalty | 4 | 0-4-0 | 0.0% | [0.0,49.0] | 0.0% | n/a | n/a | 13.8 | other:W=0,L=4,D=0 |

## Reward specs

- **baseline** — The repo's reward as-is: full shaping into the MCTS leaf value, terminal target +1/0/-1, DECK_DIFF_COEF from the environment. The control arm — must reproduce pre-harness behaviour exactly. (`name=baseline`)
- **deckout_penalty** — Baseline shaping, but a win by deck-out is worth 0.25 instead of 1.0 at the training target. Directly attacks the FINDINGS.md pathology: milling still wins the game, it just stops being worth as much as taking prizes. (`name=deckout_penalty deckout_win=0.25`)

Win rates are shown with Wilson 95% intervals. At the game counts a local run can afford, overlapping intervals are the normal outcome: the deck-out rate and attack rate move first and are what these arms are actually steering.
