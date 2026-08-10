"""A dependency-free reference agent, used as the executable spec for the contract.

This is deliberately NOT the trained agent. `submission_main.py` at the repo root
is the real entry point; this file exists so the contract in `tests/` has a
subject that runs anywhere, with no torch, no pandas, and no model weights. It
also doubles as a fallback bundle if a submission slot is ever needed before the
model is ready.

The bundle is a .tar.gz with THIS FILE AT THE TOP LEVEL as `main.py`, plus a
`deck.csv`, unpacked at evaluation time to /kaggle_simulations/agent/. Nothing
here may import from the rest of the repo -- the bundle ships only what sits next
to this file.

The contract, from the competition Overview and the cabt API docs
(https://matsuoinstitute.github.io/cabt/):

  agent(obs_dict: dict) -> list[int]

  - obs["select"] is None at the very first call: this is initial deck selection,
    and the return value is the 60 card IDs of the deck -- NOT option indices.
  - Otherwise return indices into obs["select"]["option"], subject to
    minCount <= len(result) <= maxCount, each 0 <= i < len(option), no duplicates.
    The engine only ever offers legal options, so moves are never generated here.
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time

try:
    AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Kaggle exec()s the file, so __file__ can be undefined.
    AGENT_DIR = "/kaggle_simulations/agent"
KAGGLE_DIR = "/kaggle_simulations/agent"

# Wall-clock ceiling per decision. The round-1 per-move limit is not stated in the
# competition rules, so this is a self-imposed bound: an anytime search that always
# holds a current best move is correct under any limit, including one discovered
# the hard way.
MOVE_BUDGET_SECONDS = float(os.environ.get("PTCG_MOVE_BUDGET", "2.0"))
SAFETY_MARGIN = 0.8

_rng = random.Random(20260807)


def _resolve(filename: str) -> str:
    """Find a bundled file whether running locally or under /kaggle_simulations."""
    local = os.path.join(AGENT_DIR, filename)
    return local if os.path.exists(local) else os.path.join(KAGGLE_DIR, filename)


def _load_deck() -> list[int]:
    """Read the 60 card IDs from deck.csv.

    The docs specify a deck.csv in the bundle, and the agent must also return the
    decklist at initial selection. Reading the same file for both keeps them from
    diverging -- a mismatch between the declared deck and the played deck is a
    silent and expensive bug.
    """
    deck: list[int] = []
    with open(_resolve("deck.csv"), newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            head = row[0].strip()
            if not head or head.startswith("#"):
                continue
            try:
                card_id = int(head)
            except ValueError:
                continue  # header line
            count = int(row[1]) if len(row) > 1 and row[1].strip().isdigit() else 1
            deck.extend([card_id] * count)
    return deck


try:
    _DECK = _load_deck()
except Exception as exc:  # noqa: BLE001
    print(f"could not read deck.csv: {exc!r}", file=sys.stderr)
    _DECK = []


def _select(obs: dict) -> dict | None:
    return obs.get("select") if isinstance(obs, dict) else None


def _choose(obs: dict, select: dict, deadline: float) -> list[int]:
    """Return chosen option indices.

    Reference policy: a uniform random legal selection. A real search replaces the
    body and keeps the count contract and the deadline. `deadline` is a
    time.monotonic() timestamp, already safety-margined.
    """
    options = select.get("option") or []
    if not options:
        return []

    low = select.get("minCount") or 0
    high = select.get("maxCount") or 1
    high = min(high, len(options))
    low = min(low, high)
    count = high if high == low else _rng.randint(low, high)

    # A real search loop goes here, shaped like:
    #     while time.monotonic() < deadline and not converged:
    #         run one more simulation
    # always holding a current best selection so the deadline can never catch it
    # empty-handed.
    _ = deadline

    return _rng.sample(range(len(options)), count)


def agent(obs_dict: dict, config=None) -> list[int]:
    """Entry point the environment calls once per decision."""
    deadline = time.monotonic() + MOVE_BUDGET_SECONDS * SAFETY_MARGIN
    try:
        select = _select(obs_dict)
        if select is None:
            # Initial deck selection: return card IDs, not option indices.
            return _DECK
        return _choose(obs_dict, select, deadline)
    except Exception as exc:  # noqa: BLE001
        # Never raise out of the agent: an exception forfeits the episode, while a
        # legal-but-bad move only loses ground. Log to stderr for the replay.
        print(f"agent error, falling back: {exc!r}", file=sys.stderr)
        select = _select(obs_dict) or {}
        options = select.get("option") or []
        minimum = min(select.get("minCount") or 0, len(options))
        return list(range(max(minimum, 1 if options else 0)))
