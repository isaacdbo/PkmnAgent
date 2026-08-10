# FINDINGS

Append-only log. Each entry: date, what was run, raw protocol-format numbers,
one-line conclusion. No interpretation beyond that line.

Protocol notes (apply to every entry below):
- `WIN_BY_CAUSE` W and L tag the same games (every non-draw game increments
  both its cause's W and L by 1). Report the game count as W alone (or L
  alone — they're equal), never W+L.
- `TRUE_RESULT_TALLY` is pooled P0_WINS/P1_WINS across the run and is NOT
  candidate-specific when first-player alternates by game. Candidate/opponent
  win rate must come from role-mapped harness bookkeeping (the harness's own
  per-game win/loss counters), not from `TRUE_RESULT_TALLY`.
- `ROOT_OPTION_STATS.<kind>.available` counts root decisions where that kind
  was legal (post-truncation), not total legal sub-options and not total
  decisions.
- `PRIZE_REWARD_REACHED` (decision-level, simulated prize reward reached
  during search) and games where `WIN_BY_CAUSE=PRIZE` was the actual ending
  cause are different metrics — never conflated.
- `eval_panel.py`'s per-game `EVAL_GAME_DONE=... result=N` line: `result` is
  the engine's raw winning-seat index into that game's `p0`/`p1` (which flips
  every game since first-player alternates), not a fixed candidate/opponent
  code — `result=1` means "whoever was p1 in this line won," which is the
  opponent in some games and the candidate in others. Read the `winner=`
  field (added 2026-08-11) instead, which already resolves seat to identity;
  do not infer identity from `result` alone. Same class of gotcha as the
  `TRUE_RESULT_TALLY` note above.
- `winner=`/`p0=`/`p1=` print **display names**, not the internal role keys
  ("candidate"/"opponent") used for stats aggregation (updated 2026-08-11,
  `Side.display`): the candidate always prints as `candidate`; panel
  opponents print with a parenthetical describing their policy —
  `sample_bot(random)` (uniform-random legal-action selection, the
  competition's actual sample_submission) and `old_m2(checkpoint)`
  (`model_2026-08-08_09-48.pth`, a trained MCTS checkpoint, not a scripted
  bot). Do not confuse `sample_bot` with the "stalling" behavior described
  elsewhere in this file — that behavior belongs to the OLD m2 checkpoint (a
  trained agent that passively ran games to deck-out), not to sample_bot,
  which has no learned policy to stall with.

---

## 2026-08-08 — DECK_DIFF_COEF sweep

**Run**: OLD checkpoint (`model_2026-08-08_09-48`), sims=20, opponent=self
(mirror match, same checkpoint both sides, `sweep_deck_diff.py`), games=20
per arm. Source: `sweep_out/coef_{-0.01,0.0,0.0025,0.01}.txt`.

| DECK_DIFF_COEF | WIN_BY_CAUSE (games) | deck-out share | PRIZE_REWARD_REACHED |
|---|---|---|---|
| -0.01 | DECK_OUT=18, NO_ACTIVE=2 | 18/20 = 90.0% | 0.35% (12/3436) |
| 0.0 | DECK_OUT=16, NO_ACTIVE=4 | 16/20 = 80.0% | 0.45% (14/3118) |
| 0.0025 | DECK_OUT=17, NO_ACTIVE=3 | 17/20 = 85.0% | 0.30% (9/3011) |
| 0.01 | DECK_OUT=17, NO_ACTIVE=3 | 17/20 = 85.0% | 0.39% (13/3335) |

**Conclusion**: deck-out share stays within 80-90% across a 4x coefficient range with no monotonic trend — DECK_DIFF_COEF is not a lever on deck-out share at this n.

---

## 2026-08-08 — sims sweep on OLD checkpoint (self-play mirror)

**Run**: OLD checkpoint, opponent=self (mirror match), games=20 per arm (sims=800 arm: 5 games). Source: `diagnostic/diag_2026-08-08_16-47-32.log` (sims=5), `..._16-48-28.log` (20), `..._16-51-31.log` (25), `..._16-54-52.log` (100), `..._17-07-42.log` (400), `..._18-09-53.log` (800). Checkpoint identity per prior session record; not independently re-verifiable from these raw diag logs (no ARM_CHECKPOINT marker in this file format).

| sims | games | WIN_BY_CAUSE (games) | deck-out share | PRIZE_REWARD_REACHED | SEARCH_MEAN_DEPTH |
|---|---|---|---|---|---|
| 5 | 20 | DECK_OUT=15, NO_ACTIVE=5 | 15/20 = 75.0% | 0.54% (15/2802) | 1.6418 |
| 20 | 20 | DECK_OUT=16, NO_ACTIVE=4 | 16/20 = 80.0% | 0.33% (10/3073) | 2.4668 |
| 25 | 20 | DECK_OUT=16, NO_ACTIVE=4 | 16/20 = 80.0% | 0.42% (14/3303) | 2.4883 |
| 100 | 20 | DECK_OUT=18, NO_ACTIVE=2 | 18/20 = 90.0% | 0.25% (9/3537) | 3.8826 |
| 400 | 20 | DECK_OUT=19, NO_ACTIVE=1 | 19/20 = 95.0% | 0.41% (15/3661) | 5.7398 |
| 800 | 5 | DECK_OUT=5 | 5/5 = 100.0% | 0.72% (7/978) | 6.6182 |

**Conclusion**: PRIZE_REWARD_REACHED stays flat (0.25-0.72%) across a 160x sims range while deck-out share rises monotonically to 100%.

---

## 2026-08-09 — attach prior trajectory, warm-up 1 -> main 8 (Task C retrain)

**Run**: Task C retrain, FAST_TEST=0, MAIN_EPOCHS=25 (stopped at 8), sims=20, ATTACH_PRIOR_FLOOR=0, fresh random init (SKIP_CHECKPOINT_LOAD=1). Source: `task_c_retrain.log`, 75 diag dump windows aggregated by phase.

- Plain-group ATTACH_PRIOR_DIST mean: 0.0955 (Warm-up 1) -> 0.0243 (Main 8), 12 phases, near-monotonic (one non-monotonic step at Warm4->Main1: 0.0461->0.0505).
- Special/domain-bonus-group mean: 0.1529 -> 0.1010, noisy, no monotonic trend (range 0.08-0.15 throughout).
- Code check (`RLTRM2.py`): no `temperature`, no `dirichlet`/`noise` anywhere in the file (grep, zero hits). Policy target = raw visit-count proportions (`RLTRM2.py:1004-1012`), unannealed. Move selection = deterministic argmax over visit count (`RLTRM2.py:936-943`), not sampled. `c_puct=1.0` hardcoded (`RLTRM2.py:882`).

**Conclusion**: plain-group attach prior collapses ~4x over 12 epochs while the domain-bonus group does not, under a search/training setup with no temperature and no Dirichlet noise anywhere.

---

## 2026-08-09 — OLD checkpoint vs sample_bot

**Run**: OLD checkpoint (`model_2026-08-08_09-48`), sims=20, opponent=sample_bot (competition sample_submission, uniform random), games=40. Source: `eval_panel_validation.log`.

- Candidate win rate (harness role-mapped): 12.5% (5/40). 5+35=40.
- `ROOT_OPTION_STATS.ATTACK`: available=40, chosen=0.
- `WIN_BY_CAUSE` (games): PRIZE=9, DECK_OUT=6, NO_ACTIVE=25. 9+6+25=40.
- deck-out share: 6/40 = 15.0%.
- mean sec/game: 2.11s.

**Conclusion**: OLD loses to uniform-random play and never selects ATTACK when available (0/40).

---

## 2026-08-09 — NEW checkpoint (epoch 8) vs sample_bot, sims=20

**Run**: NEW checkpoint (`model_2026-08-09_03-38`), sims=20, opponent=sample_bot, games=40. Source: `eval_panel_new_vs_bot.log`.

- Candidate win rate: 22.5% (9/40). 9+31=40.
- `ROOT_OPTION_STATS.ATTACK`: available=77, chosen=28.
- `ROOT_OPTION_STATS.ENERGY_ATTACH`: available=1045, chosen=353.
- `WIN_BY_CAUSE` (games): DECK_OUT=17, NO_ACTIVE=10, PRIZE=13. 17+10+13=40.
- deck-out share: 17/40 = 42.5%.
- `PRIZE_REWARD_REACHED`: 1.67% (58/3465 decisions).
- SEARCH_MEAN_DEPTH: 3.1931 (SEARCH_MAX_DEPTH=19).
- mean sec/game: 4.58s.

**Conclusion**: NEW beats OLD's win rate against sample_bot (22.5% vs 12.5%) and selects ATTACK at a non-zero rate (28/77), but still loses the majority of games to uniform-random play.

---

## 2026-08-09 — NEW checkpoint (epoch 8) vs sample_bot, sims=200

**Run**: NEW checkpoint, sims=200, opponent=sample_bot, games=40. Source: `eval_panel_arm_C.log`.

- Candidate win rate: 42.5% (17/40). 17+23=40.
- `ROOT_OPTION_STATS.ATTACK`: available=158, chosen=58.
- `ROOT_OPTION_STATS.ENERGY_ATTACH`: available=912, chosen=303.
- `WIN_BY_CAUSE` (games): DECK_OUT=9, NO_ACTIVE=17, PRIZE=14. 9+17+14=40.
- deck-out share: 9/40 = 22.5%.
- `PRIZE_REWARD_REACHED`: 8.53% (254/2979 decisions).
- SEARCH_MEAN_DEPTH: 6.3012 (SEARCH_MAX_DEPTH=31).
- mean sec/game: 34.85s.

**Conclusion**: 10x sims (20->200) roughly doubles win rate (22.5%->42.5%) and roughly doubles SEARCH_MEAN_DEPTH (3.19->6.30), not 10x either.

---

## 2026-08-09 — RANDOM-INIT frozen net vs sample_bot, sims=20

**Run**: freshly random-initialised, frozen `MyModel` (no checkpoint loaded), sims=20, opponent=sample_bot, games=40. Source: `eval_panel_arm_B.log`.

- Candidate win rate: 0.0% (0/40). 0+40=40.
- `ROOT_OPTION_STATS.ATTACK`: available=3, chosen=0.
- `WIN_BY_CAUSE` (games): NO_ACTIVE=38, PRIZE=2. 38+2=40 (no DECK_OUT games).
- `PRIZE_REWARD_REACHED`: 0.00% (0/654 decisions).
- mean sec/game: 0.88s.

**Conclusion**: candidate ruled invalid as a search-without-value-function baseline — ATTACK was legal at the root in only 3 of 40 games and 38/40 games ended via NO_ACTIVE in under a second each; the agent never establishes a board.

---

## 2026-08-09 — Sweep D: NEW checkpoint vs sample_bot, sims=50 and sims=100

**Run**: NEW checkpoint (`model_2026-08-09_03-38`), opponent=sample_bot, games=40 per arm. Sources: `eval_panel_arm_D50.log` (sims=50), `eval_panel_arm_D100.log` (sims=100). Combined with the already-recorded sims=20 and sims=200 arms above for the full curve.

| sims | win rate | ATTACK avail/chosen | ENERGY_ATTACH avail/chosen | WIN_BY_CAUSE games (DECK_OUT/NO_ACTIVE/PRIZE) | deck-out share | PRIZE_REWARD_REACHED | SEARCH_MEAN_DEPTH | mean sec/game |
|---|---|---|---|---|---|---|---|---|
| 20 | 22.5% (9/40) | 77/28 | 1045/353 | 17/10/13 | 42.5% | 1.67% (58/3465) | 3.1931 | 4.58s |
| 50 | 22.5% (9/40) | 79/27 | 936/337 | 17/9/14 | 42.5% | 2.41% (81/3359) | 4.6187 | 10.72s |
| 100 | 30.0% (12/40) | 83/33 | 869/320 | 12/11/17 | 30.0% | 4.06% (128/3153) | 5.4707 | 18.84s |
| 200 | 42.5% (17/40) | 158/58 | 912/303 | 9/17/14 | 22.5% | 8.53% (254/2979) | 6.3012 | 34.85s |

All W+L game checks pass (each row's win/loss and WIN_BY_CAUSE columns sum to 40). Win rate is the harness role-mapped candidate figure, not TRUE_RESULT_TALLY.

**Conclusion**: win rate is flat from sims=20 to sims=50 (22.5%->22.5%) then rises at sims=100 and sims=200 (30.0%, 42.5%) — the curve does not flatten in the 20-200 range tested, and the flat segment is at the low end (20-50), not the high end. Mean sec/game scales sublinearly with sims (0.229, 0.214, 0.188, 0.174 sec/sim/game at 20/50/100/200) and stays far under the 600s/game budget through sims=200.

---

## 2026-08-09 — FLAG (not yet actioned): update_belief call-site coverage gap

Not a run — a code-reading finding to preserve before opponent belief is wired into the encoder as a live feature.

- `update_belief` is called from exactly one place: `run_cross_play` (`RLTRM2.py:1271`). Confirmed via grep for `update_belief(` across the whole repo — a single call site.
- Confirmed absent from: `run_self_play`, `evaluate()` (RLTRM2.py's own `__main__` block), `eval_panel.py`, `checkpoint_h2h.py`.

**Conclusion**: if opponent belief is later wired into the encoder, self-play and eval paths currently see nothing (would default to `initial_belief()`'s uniform 1/6) while cross-play alone sees genuine sharp distributions — a train/test mismatch. Belief tracking must run identically in every path that plays a game, including eval harnesses, before that wiring goes live. Not actioned now, per instruction.

---

## 2026-08-09 — decks/ text decklists vs RLTRM2.py hardcoded numeric lists: agreement check

Read-only, no changes made anywhere. Compared the 5 human-readable decklists in `decks/{dragapult,grimm,lucario,megalopunny,slop}` (name+set+number format) against the hardcoded card-ID Python lists in `RLTRM2.py` (`dragapult`, `grimmsnarl`, `lucario`, `mega_lopunny`, `slop_box`). Crosswalk: `Card_List.xlsx`'s id column verified to match the engine's `card_id` exactly (id=119/121 -> Dreepy/Dragapult ex, TWM 128/130, matching `decks/dragapult`'s text lines). Entries `Card_List.xlsx` didn't resolve (diacritic/apostrophe encoding differences, and basic-energy cards named differently per source — e.g. decks/* says "Fire Energy", the engine's `all_card_data()` calls the same card "Basic {R} Energy") were resolved via name-only fallback against `card_table`.

- **lucario, mega_lopunny, slop_box (3 of 5): full agreement.** Every card and quantity in the text file has an exact-count match in the hardcoded list. Several cards share a name across multiple prints (Riolu: ids 333/677/974; Buneary: 758/848; Dunsparce: 65/305; Snorunt: 103/860; Abra: 109/741; Stunfisk: 588/869; Chien-Pao: 209/1063) — in every case the hardcoded list's chosen id is a valid print of the named card at the exact stated quantity, so these are agreements, not discrepancies; name-only matching just can't disambiguate the print automatically.
- **dragapult, grimmsnarl (2 of 5): one deliberate substitution each, same substitution both times.** Both `decks/dragapult` and `decks/grimm` list `1 Special Red Card CRI 82`. No card named "Special Red Card" (or "Red Card") exists anywhere in the ~1268-card database (`all_card_data()`, full-text search, zero matches). In both hardcoded Python lists, the corresponding leftover slot is `id=1213` = **Judge** (1 copy), not Special Red Card. Every other card in both decks matches exactly. **Correction (2026-08-09, user): this is not a transcription error — Special Red Card is not tournament-legal in this engine's card pool, and Judge was manually chosen as a functionally equivalent disruption card.**

**Conclusion**: the hardcoded numeric lists in `RLTRM2.py` are the correct, authoritative representation. `decks/` is the stale representation — it records the original decklist including a card that was later swapped for a legal substitute when the numeric list was built, and that swap was never carried back into the text files. Not a bug; no action needed.

---

## 2026-08-09 — archetype-identifying card verification

Not a game run — verified user-supplied card identifiers against the hardcoded numeric card-ID lists in `RLTRM2.py` (`ARCHETYPES`/`_ARCHETYPE_SETS`) via `card_table` lookups. Read-only.

| claim | id(s) checked | owners found in `_ARCHETYPE_SETS` |
|---|---|---|
| Dreepy / Drakloak / Dragapult ex -> dragapult | 119 / 120 / 121 | dragapult only (each) |
| Budew -> dragapult, "not 100%" | 235 | dragapult AND grimmsnarl (shared) |
| Buneary -> mega_lopunny | 848 | mega_lopunny only (a second Buneary print, id 758, exists in the card database but is unused by any of the 5 archetype decks) |
| Riolu -> lucario | 677 | lucario only (two other Riolu prints, 333/974, unused here) |
| Marnie's Impidimp + Spikemuth Gym -> grimmsnarl | 646 / 1259 | grimmsnarl only (each) |
| Ogerpon + Lillie's Clefairy -> slop | 108 / 272 | slop_box only (each independently, not just in combination); actual card is "Wellspring Mask Ogerpon ex", not "Teal Mask" as originally stated |

**Conclusion**: all 6 identifier claims confirmed against the hardcoded numeric data, with one naming correction (Wellspring vs Teal Mask Ogerpon). User confirmed 2026-08-09: "the archetype card sets are correct. No changes needed."

---

## 2026-08-09 — FLAG (not yet actioned): update_belief's uniform in-deck likelihood doesn't distinguish staples from signature cards

Not a run — a design observation, independent of the Special Red Card / Judge substitution above (that substitution is deliberate and correct; this issue stands on its own).

- `update_belief` (`RLTRM2.py:1120-1130`) multiplies every archetype's weight by a flat `1.0` if the revealed card is anywhere in that archetype's set, or `0.01` if not — regardless of how common that card is across the other archetypes' sets.
- Concretely: a generic disruption/staple card that appears across several of the 5 archetype decklists (e.g. Judge, Boss's Orders, Ultra Ball) gets the same `1.0` evidentiary weight as a genuinely archetype-defining card (e.g. Dreepy, Marnie's Impidimp, unique to one archetype). Revealing a staple should barely move the belief distribution; revealing a signature card should move it sharply. The current rule can't tell the difference between the two.

**Conclusion**: calibration gap in the likelihood model itself, independent of any specific decklist content. Flagging for whenever belief goes live; not actioned now.

---

## 2026-08-09 — cg engine has its own internal randomness, not controllable via Python's random.seed()

Discovered while verifying the parallel self-play workers (byte-identical-dump check, see the parallel-workers entry below). Not a bug in anything built today — a pre-existing property of the compiled engine (`cg-lib/cg/libcg.so`).

- Same process, same code, `random.seed(12345+i)` + `torch.manual_seed(12345+i)` set identically before each of 5 self-play games, run twice back to back: `TRUNC_TOTAL_NODES` differed (16593 vs 17847), full dumps differed throughout. Model, deck, sims all identical.
- Isolated further with **zero torch involvement**: `random_agent` (pure `random.sample` over legal options, no model) with `battle_start`/`battle_select` only, `random.seed(777)` before each of two runs — different games entirely (179 vs 195 steps, different final result, move-by-move trace not equal at any point). This rules out torch/model nondeterminism specifically — the divergence happens inside the engine itself, upstream of anything Python's `random` module controls.

**Conclusion**: the compiled cg engine has its own internal randomness source that `random.seed()`/`torch.manual_seed()` do not reach. **Retroactive methodology note**: every prior `FINDINGS.md` entry that used per-game seeding (`random.seed(base_seed+g)`) across separate runs/arms did not produce matched game sequences between those runs — each "seeded" run is an independent sample from the same distribution, not a paired/matched comparison. This doesn't invalidate the aggregate win-rate/statistic comparisons already recorded (those were always treated as independent-sample comparisons, e.g. the n-needed-for-10-points power calculation), but any implicit assumption that "same seed = same games" in earlier entries' framing was wrong. Small effects need correspondingly larger n to detect, exactly as the Task 1 sample-size calculation already accounted for.

---

## 2026-08-10

Now working on Vast.AI remote compute for faster iterations:

Arm B (sims=200, smoothing=0) is the first non-collapsing configuration.
plain_attach_mean holds ~0.05 across 4 main epochs (all prior runs: 0.009-0.023).
ABILITY zero_visit ~0%. ENERGY_DEV ends 50.24% vs Arm A's 21.16%.

---

## 2026-08-10/11 — Arm B extended to 12 main epochs, fresh init, then eval vs sample_bot

**Run 1 (training)**: `SIMULATIONS_PER_MOVE=200 POLICY_LABEL_SMOOTHING=0 M2_ONLY=1
WARMUP_EPOCHS=2 MAIN_EPOCHS=12 SKIP_CHECKPOINT_LOAD=1 FAST_TEST=0
SELF_PLAY_WORKERS=8`, fresh random init. Source: `run_e12_2026-08-10.log`
(14 DIAG_DUMP phases: Warm-up 1-2, Main 1-12). Final checkpoint:
`checkpoints/m2/model_2026-08-10_22-01.pth`.

`plain` group `attach_prior` mean by phase (extends the 4-epoch arm-B result
above to the full 12):

| phase | plain_attach_mean | phase | plain_attach_mean |
|---|---|---|---|
| Warm-up 1 | 0.0820 | Main 6 | 0.0283 |
| Warm-up 2 | 0.0463 | Main 7 | 0.0360 |
| Main 1 | 0.0310 | Main 8 | 0.0332 |
| Main 2 | 0.0345 | Main 9 | 0.0404 |
| Main 3 | 0.0371 | Main 10 | 0.0356 |
| Main 4 | 0.0355 | Main 11 | 0.0393 |
| Main 5 | 0.0297 | Main 12 | 0.0355 |

Range across all 12 main epochs: 0.0283-0.0404, no downward drift (Main 12 ==
Main 2 to 3 decimal places; the low point Main 6 is followed by a rise, not
continued decay). `special` group mean by phase ranges 0.1382 (Warm-up 1) to
0.2430 (Main 7), noisy, trending mildly upward, never collapsing toward the
`plain` group.

**Run 2 (eval)**: candidate=`model_2026-08-10_22-01.pth` (M2Deck.xlsx),
opponent=sample_bot, sims=200, games=40. Source:
`eval_m2only_sims200_smooth00_main12.log`.

- Candidate win rate (harness role-mapped): 55.0% (22/40). 22+18=40.
- `WIN_BY_CAUSE` (games, from the run's own diag dump — not eval_panel's
  coarser `cause=` label, see protocol notes): DECK_OUT=6, NO_ACTIVE=16,
  PRIZE=18. 6+16+18=40. **This tally is role-blind (identical on the
  candidate and opponent rows) and reports how games ended, not who won
  them — do not read it as a candidate win breakdown.**
- deck-out share: 6/40 = 15.0%.
- `PRIZE_REWARD_REACHED`: 10.23% (320/3128 decisions).
- SEARCH_MEAN_DEPTH: 6.1228 (SEARCH_MAX_DEPTH=25).
- mean sec/game: 19.46s (OPP_MEAN_SEC_PER_GAME).

**Candidate win rate by ending cause** (role-resolved by parsing all 40
per-game `EVAL_GAME_DONE` lines and mapping `winner = by_player[result]`,
verified by an independent hand-count that reproduces the harness's 22-18
exactly):

| cause | candidate record | win rate | games |
|---|---|---|---|
| NO_ACTIVE (`other`) | 16-0 | 100.0% | 16 |
| PRIZE | 5-13 | 27.8% | 18 |
| DECK_OUT | 1-5 | 16.7% | 6 |
| **total** | **22-18** | **55.0%** | **40** |

Every one of the candidate's 22 wins but six came from `NO_ACTIVE`-ending
games, where it won all 16. It is a net loser in both prize-race games
(5-13) and deck-out games (1-5).

Comparison to every prior sims=200-vs-sample_bot candidate in this file:

| checkpoint | win rate | deck-out share | PRIZE_REWARD_REACHED |
|---|---|---|---|
| OLD (sims=20) | 12.5% (5/40) | 15.0% (6/40, but via NO_ACTIVE-dominant loss) | n/a |
| NEW epoch-8, sims=200 | 42.5% (17/40) | 22.5% (9/40) | 8.53% |
| **this checkpoint (12 main epochs), sims=200** | **55.0% (22/40)** | **15.0% (6/40)** | **10.23%** |

**Conclusion**: extending arm-B's non-collapsing config from 4 to 12 main
epochs does not reintroduce the plain-group attach-prior collapse (stays in
the 0.028-0.040 band throughout, never drops toward the 0.009-0.023 collapsed
range). The resulting checkpoint is the first to cross 50% win rate against
sample_bot at sims=200 (55.0%, vs the prior best of 42.5%). **Correction to
an earlier draft of this entry**: the lower deck-out share does not mean the
model is winning by contesting the board — the win-by-cause breakdown shows
the opposite. The candidate loses the majority of its prize-race games
(5-13) and deck-out games (1-5); its win total is carried almost entirely by
`NO_ACTIVE`-ending games (16-0, 100%). Whether that reflects genuinely
strong play in the specific board states that lead to `NO_ACTIVE`, or
sample_bot's uniform-random policy stumbling into losing its last Pokémon
disproportionately often, is not yet established — flagging as open, not
concluding either way.

---

## 2026-08-11 — win-by-cause decomposition of the epoch-8 checkpoint (arm C), compared against tonight's 12-epoch checkpoint

Re-analysis of an already-recorded run (`eval_panel_arm_C.log`, the "NEW
epoch-8, sims=200" row above) using the same role-resolution method as the
correction above, to see whether the 12-epoch checkpoint's improvement is
broad-based or concentrated in one cause bucket. Verified independently by
re-parsing all 40 `EVAL_GAME_DONE` lines in `eval_panel_arm_C.log` with
`winner = by_player[result]`; reconciles exactly against that log's own
`WIN_BY_CAUSE=DECK_OUT:9,NO_ACTIVE:17,PRIZE:14` and 17-23 harness total.

**Arm C (NEW epoch-8, sims=200) candidate record by cause**:

| cause | candidate record | win rate | games |
|---|---|---|---|
| NO_ACTIVE (`other`) | 16-1 | 94.1% | 17 |
| DECK_OUT | 1-8 | 11.1% | 9 |
| PRIZE | 0-14 | 0.0% | 14 |
| **total** | **17-23** | **42.5%** | **40** |

**Side by side with tonight's 12-epoch checkpoint** (previous entry):

| cause | arm C (epoch 8) | tonight (epoch 12) |
|---|---|---|
| NO_ACTIVE | 16-1 | 16-0 |
| DECK_OUT | 1-8 | 1-5 |
| PRIZE | 0-14 | 5-13 |
| **total** | **17-23 (42.5%)** | **22-18 (55.0%)** |

NO_ACTIVE wins are flat (16 -> 16) — every win the newer checkpoint gained
over the older one came from prize-race games: 0/14 -> 5/18. Arm C never won
a single prize game.

**Consequence for how to read the headline win rate**: sample_bot strands
its own active Pokémon in ~16-17 of 40 games regardless of which checkpoint
it faces, and both checkpoints win nearly all of those — that bucket carries
no discriminating signal between checkpoints. Collapsing the other two
causes (deck-out + prize) into one non-NO_ACTIVE group: arm C is 1/23 (4.3%),
tonight is 6/24 (25.0%) — a much larger relative move than 42.5% -> 55.0%
suggests on its face.

Fisher's exact test (hypergeometric, computed directly — no scipy on this
box — via `P(X<=k_obs) = C(K,k)*C(N-K,n-k) / C(N,n)` summed from 0 to
`k_obs`), one-tailed, on the two subgroup comparisons:

- Prize subgroup (arm C 0/14 vs tonight 5/18): p ~= 0.0425.
- Non-NO_ACTIVE overall (arm C 1/23 vs tonight 6/24): p ~= 0.0547.

Both borderline, and this is a post-hoc subgroup comparison (the split into
NO_ACTIVE vs non-NO_ACTIVE was chosen after seeing the data) — not to be
read as an established, pre-registered result. It is the first
training-side signal in this file located in a specific, behaviorally
meaningful bucket (prize-race play) rather than an aggregate win rate.

Corroborating metrics from the same two diag dumps, all pointing the same
direction: `MISSED_ATTACH` 201 -> 166, `PRIZE_REWARD_REACHED` 8.53% ->
10.23%, `ATTACK` chosen 58/158 (36.7%) -> 96/195 (49.2%), deck-out games 9
-> 6. Attribution to any single training change is unavailable — MAIN_EPOCHS
(8 vs 12), M2_ONLY (0 vs 1), training-time `SIMULATIONS_PER_MOVE` (20 vs
200), and `POLICY_LABEL_SMOOTHING` all differ between arm C's training run
and tonight's.

Two smaller notes: arm C's `ROOT_OPTION_STATS` has no `ABILITY` entry at
all (that metric postdates arm C's run — not comparable across the two).
And `OPP_MEAN_SEC_PER_GAME` roughly halved (34.85 -> 19.46) between the two
eval runs at identical `SIMULATIONS_PER_MOVE=200` — **caveat, not yet in the
verified claim**: `eval_panel.py` hardcodes `device=torch.device("cpu")`
([eval_panel.py:172](eval_panel.py#L172)) regardless of machine, and arm
C's log predates the "now working on Vast.AI remote compute" note in this
file by about a day, so it most likely ran on a different physical
machine (local, pre-rental) than tonight's run (the rented box). The
sec/game halving is more likely a raw-CPU-speed difference between two
different computers than evidence that search got cheaper — do not use it
to budget a sims=800 inference-time estimate without re-measuring both
ends on the same machine.

**Conclusion**: the 12-epoch checkpoint's win-rate gain over the epoch-8
checkpoint is not broad-based — it is concentrated entirely in prize-race
games (0/14 -> 5/18), while the NO_ACTIVE-win bucket (which supplies the
majority of both checkpoints' total wins) stayed exactly flat. Recommend
tracking non-NO_ACTIVE win rate as a secondary headline metric alongside raw
win rate going forward, since raw win rate spends ~40% of its resolution on
a bucket that has so far shown zero discrimination between checkpoints.
