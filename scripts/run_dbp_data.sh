#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/DBP15K_or_DBP100K_data_dir [noise]" >&2
  echo "Example: $0 /data/DBP15K/zh_en 0.1" >&2
  exit 2
fi

DATA_DIR="$1"
NOISE="${2:-0.1}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
BACKEND="${BACKEND:-hf}"
SEED="${SEED:-42}"

if [[ -f "$DATA_DIR/ent_ids_1" && -f "$DATA_DIR/ent_ids_2" ]]; then
  ENTITY_PREFIX="${ENTITY_PREFIX:-ent_ids_}"
elif [[ -f "$DATA_DIR/ent_1" && -f "$DATA_DIR/ent_2" ]]; then
  ENTITY_PREFIX="${ENTITY_PREFIX:-ent_}"
elif [[ -f "$DATA_DIR/Ent_1" && -f "$DATA_DIR/Ent_2" ]]; then
  ENTITY_PREFIX="${ENTITY_PREFIX:-Ent_}"
else
  ENTITY_PREFIX="${ENTITY_PREFIX:-ent_ids_}"
fi

cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/inject_kg_noise.py \
  --data-dir "$DATA_DIR" \
  --noise "$NOISE" \
  --seed "$SEED" \
  --mode tail \
  --entity-prefix "$ENTITY_PREFIX" \
  --avoid-existing \
  --save-maps \
  --summary-file "$DATA_DIR/noise_summary.json"

ARGS=(
  -m llm_sim_replication.dbp_data
  --data-dir "$DATA_DIR"
  --noise "$NOISE"
  --seed "$SEED"
  --backend "$BACKEND"
  --llm-model "$MODEL"
  --use-context
  --save-prompts
)

if [[ "$BACKEND" == "hf" ]]; then
  ARGS+=(--require-sentence-transformer)
fi

PYTHONPATH=src "$PYTHON_BIN" "${ARGS[@]}"
