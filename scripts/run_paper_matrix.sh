#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
SEED="${SEED:-13}"

cd "$REPO_ROOT"

for DATASET in wn18rr fb15k-237; do
  for NOISE in 0.1 0.2 0.3; do
    OUT="runs/${DATASET}-noise-${NOISE}-llm-sim"
    PYTHONPATH=src "$PYTHON_BIN" -m llm_sim_replication.cli \
      --dataset "$DATASET" \
      --split train \
      --noise "$NOISE" \
      --seed "$SEED" \
      --backend hf \
      --llm-model "$MODEL" \
      --require-sentence-transformer \
      --use-context \
      --context-k 1 \
      --output "$OUT"
  done
done

echo "Paper matrix complete under runs/"
