# iono_rule panel bot — provenance

- Source: Kaggle notebook "A Sample Rule-Based Agent Iono's Deck" by Kiyota
  (collaborators include The Pokémon Company 01/02/03), Version 9,
  https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck
- License: Apache 2.0 (as declared on the notebook page).
- Retrieved 2026-08-11 from the notebook's rendered `__results__.html`
  (public signed kaggleusercontent URL): the `%%writefile main.py` cell was
  extracted verbatim into `main.py` (no modifications).
- `deck.csv` reconstructed from the notebook's own deck image
  (per-card IDs and counts shown on the card montage), cross-checked against
  the decklist constants hardcoded in `main.py` (`Iono_Voltorb = 265  # ×3`,
  etc.). 15 distinct cards, 60 total. Card IDs verified against the engine's
  `all_card_data()` names at panel load time.
- Kaggle Best Score of this notebook's submission: 525.8 (V7), i.e. a real
  leaderboard-participating rule-based baseline, not a local stub.
