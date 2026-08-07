"""
Usage:
    python create_submission.py out/model4.pth
    python create_submission.py out/model4.pth --source RLTRM2.py --deck M2Deck.xlsx --output submission.zip

agent_core.py is auto-generated from either:
  - A .py file (used directly as agent_core)
  - A .ipynb notebook (extracts the first cell containing 'class MyModel')
No manual sync needed — just retrain and repackage.
"""
import argparse
import json
import os
import zipfile


def extract_agent_core(source_path: str) -> str:
    """Return agent core source from a .py file or a .ipynb notebook cell."""
    if source_path.endswith(".py"):
        with open(source_path, encoding="utf-8") as f:
            return f.read()
    # .ipynb: extract first code cell containing 'class MyModel'
    with open(source_path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            source = "".join(cell["source"])
            if "class MyModel" in source:
                return source
    raise ValueError(f"No cell containing 'class MyModel' found in {source_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the .pth model file (e.g. out/model4.pth)")
    parser.add_argument("--source", default="RL_TRM2.ipynb", help="Agent source: a .py file or .ipynb notebook")
    parser.add_argument("--deck",   default="M2Deck.xlsx",   help="Deck Excel file")
    parser.add_argument("--output", default="submission.zip", help="Output zip path")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def full(p): return p if os.path.isabs(p) else os.path.join(script_dir, p)

    for path, label in [(args.model, "model"), (args.deck, "deck"),
                        ("submission_main.py", "submission_main.py"),
                        (args.source, "source"), ("cg-lib", "cg-lib")]:
        if not os.path.exists(full(path)):
            raise FileNotFoundError(f"Required {label} not found: {full(path)}")

    print(f"Extracting agent_core from {args.source}...")
    agent_core_code = extract_agent_core(full(args.source))

    out_path = full(args.output)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(full("submission_main.py"), "main.py")
        zf.writestr("agent_core.py", f"# Auto-generated from {args.source} — do not edit by hand\n\n{agent_core_code}")
        zf.write(full(args.model), "model.pth")
        zf.write(full(args.deck),  "deck.xlsx")
        for root, _, files in os.walk(full("cg-lib")):
            for file in files:
                fp = os.path.join(root, file)
                zf.write(fp, os.path.relpath(fp, script_dir))

    total_mb = os.path.getsize(out_path) / 1_048_576
    names = zipfile.ZipFile(out_path).namelist()
    print(f"Created {out_path}  ({total_mb:.1f} MB,  {len(names)} files)")
    print("  " + "\n  ".join(sorted(names)[:20]))
    if len(names) > 20:
        print(f"  ... and {len(names) - 20} more")


if __name__ == "__main__":
    main()
