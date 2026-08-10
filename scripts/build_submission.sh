#!/usr/bin/env bash
# Package a bundle directory into a Kaggle submission archive.
#
# The competition docs specify a .tar.gz containing main.py at the TOP LEVEL of
# the archive -- not nested inside a directory -- plus a deck.csv. Tarring a
# directory by name (`tar -czf out.tar.gz submission/`) nests everything under
# `submission/`, so this script tars from *inside* the bundle directory instead.
#
# Usage:
#   scripts/build_submission.sh [bundle_dir] [output.tar.gz]
#
# Defaults to the dependency-free reference bundle in submission/. Point it at
# any staging directory -- e.g. one assembled by create_submission.py -- to
# package that instead.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="${1:-$repo_root/submission}"
out="${2:-$repo_root/submission.tar.gz}"

if [[ ! -d "$src_dir" ]]; then
  echo "error: bundle directory $src_dir does not exist" >&2
  exit 1
fi

if [[ ! -f "$src_dir/main.py" ]]; then
  echo "error: $src_dir/main.py is missing -- the docs require main.py at the archive root" >&2
  exit 1
fi

if [[ ! -f "$src_dir/deck.csv" ]]; then
  if [[ -f "$src_dir/deck.xlsx" ]]; then
    echo "error: $src_dir has deck.xlsx but no deck.csv -- the docs specify deck.csv" >&2
  else
    echo "error: $src_dir/deck.csv is missing" >&2
  fi
  exit 1
fi

if grep -q '^# PLACEHOLDER' "$src_dir/deck.csv" 2>/dev/null; then
  echo "warning: deck.csv is still the placeholder and is not playable" >&2
fi

rm -f "$out"
# -C ... . tars the *contents*, so entries are ./main.py rather than submission/main.py.
# Byte-compiled caches are excluded: they bloat the bundle and can ship a stale .pyc
# alongside the source it was compiled from.
tar -czf "$out" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='.ipynb_checkpoints' \
  -C "$src_dir" .

# 197.7 MiB limit, per the competition rules.
limit=$((197 * 1024 * 1024))
size=$(wc -c < "$out" | tr -d ' ')
if (( size > limit )); then
  echo "error: bundle is $size bytes, over the 197.7 MiB limit" >&2
  exit 1
fi

# Verify the layout we just claimed to produce, rather than trusting the tar flags.
entries="$(tar -tzf "$out" | sed 's|^\./||')"
for required in main.py deck.csv; do
  if ! grep -qx "$required" <<<"$entries"; then
    echo "error: $required is not at the top level of $out" >&2
    exit 1
  fi
done

echo "built $out ($size bytes)"
echo "contents:"
tar -tzf "$out"
