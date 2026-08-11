# Local Training Reproduction

Current local verification was performed on macOS with the existing
`pkmn-amd64` Colima profile because `cg-lib/cg/libcg.so` is a Linux x86-64 ELF
binary and `cg-lib/cg/sim.py` expects `libcg.dylib` on Darwin.

Native macOS command and result:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pandas openpyxl torch
.venv/bin/python RLTRM2.py
```

Result: Python dependencies install, but engine load fails with missing
`cg-lib/cg/libcg.dylib`.

Linux amd64 container smoke:

```bash
colima start pkmn-amd64
docker run --rm --dns 8.8.8.8 -u 501:20 -e HOME=/tmp \
  -v /Users/vbonnet/worktrees/PkmnAgent/reward-harness-core:/work \
  -w /work python:3.11-slim sh -lc \
  'python -m pip install --user --no-cache-dir pandas openpyxl >/tmp/pip.log &&
   python -m pip install --user --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch >>/tmp/pip.log &&
   python RLTRM2.py'
```

Verified in the Linux amd64 container:

- `RLTRM2.py` imports the vendored `cg-lib` engine successfully.
- FAST_TEST self-play ran for all six agents.
- FAST_TEST cross-play reached and completed all listed M2 matchups.
- Checkpoints were written for all six agents under `checkpoints/<agent>/`.

Observed checkpoint set from the smoke:

```text
checkpoints/m2/model_2026-08-11_00-16.pth
checkpoints/dragapult/model_2026-08-11_00-16.pth
checkpoints/grimmsnarl/model_2026-08-11_00-16.pth
checkpoints/lucario/model_2026-08-11_00-16.pth
checkpoints/mega_lopunny/model_2026-08-11_00-16.pth
checkpoints/slop_box/model_2026-08-11_00-16.pth
```

Not yet verified: full script exit through final evaluation. The bounded smoke
was stopped after checkpoint creation and cross-play because CPU training under
amd64 emulation was still running silently after 34 minutes.
