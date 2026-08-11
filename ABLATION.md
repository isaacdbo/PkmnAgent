# Reward-ablation harness

FINDINGS.md records the shape of the problem: deck-out share sits at 80–90% of
self-play games and climbs monotonically with search depth (100% at 800 sims),
while `PRIZE_REWARD_REACHED` stays flat under 1%. Sweeping `DECK_DIFF_COEF`
across a 4× range moved it not at all. That pattern says the reward function
itself makes running the opponent out of cards the better-valued outcome — so
the next thing to vary is the reward, one term at a time, with everything else
held fixed.

This harness is the apparatus for doing that:

| piece | file | what it is |
|---|---|---|
| reward registry | `ablation/rewards.py` | named reward specs, chosen with `REWARD_SPEC=<name>` |
| diagnostic metrics | `ablation/metrics.py` | win rate + Wilson CI, deck-out rate, attack rate, turns-to-win, WIN_BY_CAUSE |
| pinned eval panel | `eval_panel.py`, `panel_bots/` | random, first, starter_rule, iono_rule, old_m2 |
| panel runner | `ablation/eval_runner.py` | plays the panel, writes per-game JSONL + summaries |
| autoresearch driver | `ablation/driver.py` | one arm per reward spec: train → evaluate → compare |
| raw-log ingester | `ablation/ingest.py` | the repo's existing eval output → the same metrics |
| local runner | `ablation/run_local.sh` | the container wrapper that makes all of it reproducible |

## The reproducible local run

The engine is a Linux x86-64 ELF (`cg-lib/cg/libcg.so`) and `cg-lib/cg/sim.py`
looks for `libcg.dylib` on Darwin, so nothing that touches the simulator runs
natively on an Apple-silicon Mac. Everything simulation-related runs in a
`linux/amd64` container; on Apple silicon that is Rosetta emulation, the same
setup that produced the local training receipts already in this repo.

```bash
# One command. Two reward arms, each trained from scratch and scored on the
# pinned panel, with a comparison table at the end.
./ablation/run_local.sh smoke
```

That writes `results/ablation-smoke/`:

```
results/ablation-smoke/
  COMPARISON.md              cross-arm table: win rate, deck-out rate, attack rate, turns
  comparison.json            the same numbers, machine-readable
  comparison.csv
  <arm>/train.log            the actual RLTRM2.py run for that arm
  <arm>/train_receipt.json   command, env, exit code, wall time, checkpoint produced
  <arm>/checkpoints/m2/*.pth arm-private checkpoints (never shared between arms)
  <arm>/eval/games.jsonl     one row per game
  <arm>/eval/summary.json    per-opponent metrics
```

Other modes, all from the same script:

```bash
./ablation/run_local.sh dry       # print the plan, run nothing, no container
./ablation/run_local.sh tests     # host-side unit tests (no engine needed)
./ablation/run_local.sh ingest    # organise the repo's raw eval output
./ablation/run_local.sh ablation  # the full arm list at the default size
./ablation/run_local.sh eval checkpoints/m2/model_2026-08-11_00-16.pth
```

Sizing is via environment variables: `PKMN_ARMS`, `PKMN_GAMES`, `PKMN_PANEL`,
`PKMN_IMAGE`. First run downloads torch into `.dockerhome/` (gitignored) and is
slow; later runs reuse it.

Prerequisites: a Docker daemon that can run `linux/amd64` images, and
`pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv` present
(it is gitignored competition material; `eval_panel` also accepts it inside
`pkmnagent_remote.tar.gz`). Only the `random`, `first` and `starter_rule` panel
members need it — `iono_rule` ships its own `deck.csv`.

## Reward specs

Selected with `REWARD_SPEC=<name>`; the default is `baseline`, which is exactly
the reward this repo had before the harness existed. `tests/test_rewards.py`
pins that equivalence, because if the control arm drifts, no comparison against
it means anything.

| spec | what it changes |
|---|---|
| `baseline` | nothing — full shaping, terminal target ±1/0, ambient `DECK_DIFF_COEF` |
| `terminal_only` | every shaping term zeroed and `board_reward` off; only the game result remains |
| `deckout_penalty` | a win by deck-out is worth 0.25 instead of 1.0 |
| `deckout_penalty_hard` | a win by deck-out is worth −0.25 — deliberately over-corrected, to bracket the effect |
| `turns_to_win_mild` | terminal target discounted by game length (a turn-40 win is worth 0.75) |
| `turns_to_win_strong` | the same at 2× strength (a turn-40 win is worth 0.5) |
| `deckout_penalty_turns` | both levers at once |

Two hooks in `RLTRM2.py`, and only two:

- `shaped_reward_terms` returns `REWARD_SPEC.apply_shaping(terms)`, which scales
  each term. This feeds the MCTS **leaf value** only — it steers search, never
  the training target.
- `_backup_and_store` computes the value target through
  `ablation_rewards.terminal_value(...)`, which can depend on how the game ended
  (deck-out vs prizes) and on the turn count. This is what the value head
  actually regresses on.

Terminal targets are clamped to [−1, 1]: the value head is a `tanh`, so
anything outside that range would just saturate it.

Adding a shaping term to `shaped_reward_terms` without registering it in
`ablation/rewards.py` raises at import (`validate_terms`) rather than silently
surviving an arm that meant to zero all shaping.

## The pinned panel

Every arm is scored against the same opponents, so results are comparable
across arms and across days:

- **random** — the official `sample_submission` logic: uniform-random legal
  action, its own deck.
- **first** — always takes the first legal action. A degeneracy check: an agent
  that beats `random` but not `first` is exploiting randomness, not playing.
- **iono_rule** — a real Kaggle rule-based agent (Apache 2.0, leaderboard score
  525.8), vendored verbatim with provenance in
  `panel_bots/iono_rule/PROVENANCE.md`.
- **starter_rule**, **old_m2** — also available; not in the default panel.

Seats alternate every game and results are attributed by identity, never by raw
seat index. FINDINGS.md records reading `result=` as "player 1 won" as a
falsified hypothesis; both the runner and the ingester are written against that
mistake, and `tests/test_ingest.py` tests for it.

## Metrics

`ablation/metrics.py` is the single definition consumed by the runner, the
driver and the ingester:

- **win_rate** with a Wilson 95% interval. At 20–40 games per opponent, most
  arm-to-arm differences will not clear it — which is the point of showing it.
- **deckout_rate**, split into wins-by and losses-by deck-out.
- **attack_rate** — of the candidate's decisions where an attack was legal, the
  share that took it. Measured from the trajectory, so it is defined for
  scripted bots too, unlike the MCTS-internal `ROOT_OPTION_STATS` counters.
- **turns_to_win** / **turns_to_loss**, mean and median.
- **win_by_cause** — the WIN_BY_CAUSE tally, kept per-cause.

Metrics that cannot be computed report `n/a`, never `0`. A log with no reason
codes must not read as "0% deck-out".

## Ingesting the existing eval output

The repo root holds ~40 files of eval output in six formats written by
different scripts on different days. `ablation/ingest.py` reads all of them into
one schema:

```bash
python3 -m ablation.ingest . --out-dir results/ingested
```

Output: `games.jsonl` (per-game rows), `diag_dumps.jsonl` (per DIAG_DUMP block,
with derived deck-out and attack rates), `summary.json`, `summary.csv`, and
`INDEX.md` — an inventory saying what each file is and its headline numbers.
A committed snapshot lives in `ablation_results/ingested/`.

The parsers are tolerant by design: the `EVAL_GAME_DONE` format gained
`winner=`, `turn=` and `cand_prizes=` partway through the project's history, so
older logs produce records with those fields unset rather than failing to parse.

## Adding an arm

1. Add a `RewardSpec` to `REGISTRY` in `ablation/rewards.py`.
2. Add a test to `tests/test_rewards.py` asserting what it changes — and, if it
   should not change something, that too.
3. `PKMN_ARMS=baseline,<new_arm> ./ablation/run_local.sh ablation`

No change to `RLTRM2.py` is needed: both hooks already route through the spec.
