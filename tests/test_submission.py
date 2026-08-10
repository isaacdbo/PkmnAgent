"""Guards on the submission contract and the bundle layout.

These catch failure modes that cost a submission slot: an agent that returns the
wrong shape at initial selection, violates the count/duplicate rules, raises
instead of moving, or a bundle whose main.py ends up nested inside a directory.

The subject is submission/main.py -- the dependency-free reference agent -- so
these run in CI with no torch, no pandas, and no model weights. The rules they
encode apply equally to submission_main.py; that one needs the trained model to
import, so it is exercised by local self-play rather than here.

Contract source: competition Overview and the cabt API docs
(https://matsuoinstitute.github.io/cabt/).
"""

import ast
import importlib.util
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSION = REPO / "submission"


def _load_main():
    spec = importlib.util.spec_from_file_location("reference_main", SUBMISSION / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs(options, min_count=1, max_count=1):
    return {
        "select": {"option": list(options), "minCount": min_count, "maxCount": max_count},
        "logs": [],
        "current": {"yourIndex": 0},
    }


def test_bundle_has_the_two_required_files():
    """The docs specify a .tar.gz containing main.py at the top level and a deck.csv."""
    assert (SUBMISSION / "main.py").is_file()
    assert (SUBMISSION / "deck.csv").is_file()


def test_reference_main_only_imports_the_standard_library():
    """Round 1 gives 2 vCPU and no network; the reference bundle stays dependency-free."""
    tree = ast.parse((SUBMISSION / "main.py").read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
    assert roots <= {"csv", "os", "random", "sys", "time", "__future__"}


def test_agent_returns_legal_option_indices():
    main = _load_main()
    for _ in range(50):
        chosen = main.agent(_obs(["a", "b", "c"]))
        assert isinstance(chosen, list)
        assert len(chosen) == 1
        assert 0 <= chosen[0] < 3


def test_agent_respects_min_and_max_count():
    main = _load_main()
    for _ in range(50):
        chosen = main.agent(_obs(range(6), min_count=2, max_count=4))
        assert 2 <= len(chosen) <= 4
        assert len(set(chosen)) == len(chosen), "duplicate indices are rejected by the engine"
        assert all(0 <= i < 6 for i in chosen)


def test_agent_never_exceeds_the_option_count():
    """maxCount is documented as never exceeding len(option), but clamp anyway."""
    main = _load_main()
    chosen = main.agent(_obs(["only"], min_count=1, max_count=5))
    assert chosen == [0]


def test_agent_returns_the_deck_at_initial_selection():
    """obs['select'] is None once, at setup, and wants card IDs -- not indices."""
    main = _load_main()
    returned = main.agent({"select": None, "logs": [], "current": None})
    assert returned == main._DECK


def test_agent_handles_an_empty_option_set():
    main = _load_main()
    assert main.agent(_obs([])) == []


def test_agent_never_raises_on_a_malformed_observation():
    """An exception forfeits the episode; a bad move only loses ground."""
    main = _load_main()
    for junk in [object(), {"select": {"option": None}}, 42, {"select": {}}]:
        assert isinstance(main.agent(junk), list)


def test_build_script_puts_main_py_at_the_archive_root(tmp_path):
    """main.py nested inside a directory is a documented upload failure mode."""
    out = tmp_path / "submission.tar.gz"
    subprocess.run(
        ["bash", str(REPO / "scripts" / "build_submission.sh"), str(SUBMISSION), str(out)],
        check=True,
        capture_output=True,
    )
    listing = subprocess.run(
        ["tar", "-tzf", str(out)], capture_output=True, text=True, check=True
    ).stdout
    entries = {line.removeprefix("./").strip() for line in listing.splitlines()}
    assert "main.py" in entries
    assert "deck.csv" in entries


def test_build_script_excludes_bytecode_caches(tmp_path):
    """A stale .pyc shipped next to its source is a needless upload risk."""
    out = tmp_path / "submission.tar.gz"
    subprocess.run(
        ["bash", str(REPO / "scripts" / "build_submission.sh"), str(SUBMISSION), str(out)],
        check=True,
        capture_output=True,
    )
    listing = subprocess.run(
        ["tar", "-tzf", str(out)], capture_output=True, text=True, check=True
    ).stdout
    assert "__pycache__" not in listing
    assert ".pyc" not in listing


def test_build_script_rejects_a_bundle_without_a_deck_csv(tmp_path):
    """A bundle missing deck.csv should fail loudly at build time, not on upload."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "main.py").write_text("def agent(obs, config=None):\n    return []\n")
    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "build_submission.sh"), str(bundle),
         str(tmp_path / "out.tar.gz")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "deck.csv" in result.stderr
