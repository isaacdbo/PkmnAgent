# Creating a Submission

## Prerequisites
- Trained model saved to `out/` (e.g. `out/model4.pth`) — produced by running the training loop in `RL_TRM2.ipynb`
- [text](decks)`M2Deck.xlsx` in the project root
- `cg-lib/` in the project root

## Steps

```bash
python create_submission.py out/model4.pth
```

This produces `submission.zip` containing:

| File | Source |
|---|---|
| `main.py` | `submission_main.py` (the agent entry point) |
| `agent_core.py` | Auto-extracted from `RL_TRM2.ipynb` (the `class MyModel` cell) |
| `model.pth` | Your trained weights |
| `deck.xlsx` | Your deck |
| `cg-lib/` | Game library |

Upload `submission.zip` to the competition.

## Options

```bash
python create_submission.py out/model4.pth --deck MyDeck.xlsx --output my_submission.zip
```

## Keeping things in sync

- **Model architecture / encoding / MCTS** — edit in `RL_TRM2.ipynb`. Re-run `create_submission.py` and `agent_core.py` is regenerated automatically.
- **Model size** (`128, 2, 256, 3, 1`) — if you change `MyModel(...)` in the notebook, update the matching line in `submission_main.py`.
- **Deck** — update `M2Deck.xlsx` or pass `--deck` flag.
