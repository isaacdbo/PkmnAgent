# Smoke-run receipt — 2026-08-11

Output of one `./ablation/run_local.sh smoke` on an Apple-silicon host, in a
`linux/amd64` container under Rosetta, sharing the machine with two other
training containers.

```
PKMN_ARMS=baseline,deckout_penalty  PKMN_PANEL=random,first  PKMN_GAMES=4
train: FAST_TEST=0 M2_ONLY=1 SKIP_CHECKPOINT_LOAD=1 SIMULATIONS_PER_MOVE=5
       WARMUP_EPOCHS=1 WARMUP_SELF_PLAY_GAMES=2 MAIN_EPOCHS=1
       MAIN_SELF_PLAY_M2_GAMES=2 STOP_AFTER_MAIN_EPOCH=1 SELF_PLAY_WORKERS=1
       SELF_PLAY_BASE_SEED=20260810
eval:  SIMULATIONS_PER_MOVE=5, base seed 20260809, 4 games per opponent
```

Both arms trained to completion (`returncode: 0`, 491s and 703s) and each
wrote its checkpoint into its own `CHECKPOINT_ROOT`. Per-arm `train.log` and
`train_receipt.json` here record the exact command and environment. The
checkpoints themselves are not committed (`*.pth` is gitignored).

## What this receipt is evidence of

The harness runs end to end: reward spec → training → arm-private checkpoint →
pinned panel → per-game records → per-arm metrics → cross-arm comparison. Both
`[CONFIG] reward: ...` lines appear in the spawned self-play workers, which is
the check that `REWARD_SPEC` survives the multiprocessing `spawn` boundary.

The metrics behave as specified on real data: `attack_rate` reads `0.0%` when
attacks were legal and never taken, and `n/a` when no attack was ever legal —
never conflating the two. Wilson intervals on 4-game matchups run
`[0.0, 49.0]`, which is the honest width.

## What it is NOT evidence of

Nothing about which reward arm is better. Each arm saw **4 self-play games**
and ~5 gradient batches — the networks are barely off their initialisation, and
the 8-game panel gives intervals wide enough to contain almost any hypothesis.
`deckout_penalty` shows 2 wins to `baseline`'s 0; at n=8 with
`[7.1, 59.1]` vs `[0.0, 32.4]` that is noise, and it would be a
misreading to report it as an effect.

Sizing the real experiment is the next step. The driver takes it unchanged:

```bash
PKMN_ARMS=baseline,terminal_only,deckout_penalty,turns_to_win_mild \
PKMN_GAMES=40 PKMN_PANEL=random,first,iono_rule \
./ablation/run_local.sh ablation
```

## Files

- `COMPARISON.md`, `comparison.json`, `comparison.csv` — cross-arm tables
- `<arm>/train.log` — the actual `RLTRM2.py` run
- `<arm>/train_receipt.json` — command, env, exit code, wall time, checkpoint
- `<arm>/eval/games.jsonl` — one row per panel game
- `<arm>/eval/summary.{json,csv,txt}` — per-opponent metrics
