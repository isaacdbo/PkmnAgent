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

## Independent re-verification (2026-08-11, second session)

The first retrieval was re-checked from scratch rather than trusted, because a
hand-written stub masquerading as a real bot would invalidate every panel
number that cites it.

1. `kaggle` CLI 2.2.4 was installed and
   `kaggle kernels pull kiyotah/a-sample-rule-based-agent-iono-s-deck` was
   attempted. It fails closed with
   `Authentication required to call the Kaggle API.` — the CLI's own remedy is
   the interactive `kaggle auth login` OAuth browser flow, and no
   `~/.kaggle/access_token`, `~/.kaggle/kaggle.json`, or `KAGGLE_API_TOKEN`
   exists on this host. **Anyone with Kaggle credentials should re-run that
   pull as the primary check; the steps below are the credential-free
   substitute, not a replacement for it.**
2. The notebook page was loaded in a browser and the live
   `#rendered-kernel-content` iframe URL (a fresh signed
   `kaggleusercontent.com/kf/327782691/.../__results__.html` link, re-issued on
   this load) was fetched over plain HTTPS with no session cookie — so the
   fetch is reproducible by any anonymous reader.
3. `extract_notebook_cell.py` pulls the `%%writefile main.py` cell body out of
   that HTML. The result is **byte-identical** to the committed `main.py`:

   ```text
   MD5 eb1aac45bd00f968430f01319edfcaa0  (extracted from live notebook)
   MD5 eb1aac45bd00f968430f01319edfcaa0  panel_bots/iono_rule/main.py
   415 lines, diff exit status 0
   ```

4. `deck.csv` was re-checked against the decklist constants in `main.py`: 60
   cards, 15 distinct ids, and every one of the 15 `# ×N` declared counts
   matches its multiplicity in the CSV exactly (22× Basic Lightning Energy, 4×
   Lillie's Determination, 4× Canari, 3× each of the five Iono Pokémon lines,
   and so on). No undeclared card ids appear in the deck.
5. Page metadata confirmed live: "Version 9 of 9", "released under the Apache
   2.0 open source license".

Reproduce (1)–(3) with:

```sh
python3 panel_bots/iono_rule/extract_notebook_cell.py <saved __results__.html> \
  | diff -u panel_bots/iono_rule/main.py -
```
