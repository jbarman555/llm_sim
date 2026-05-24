#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
PYTHONPATH=src "$PYTHON_BIN" -m llm_sim_replication.cli \
  --input data/toy_kg.tsv \
  --noise 0.2 \
  --backend heuristic \
  --output runs/toy-20 \
  --save-prompts

echo "Toy run complete: runs/toy-20"
