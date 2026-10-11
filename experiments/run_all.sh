#!/usr/bin/env bash
# Reproduce every result in results/. Runs are sequential (one local model).
# Usage: bash experiments/run_all.sh [configs...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-.venv/bin/python}
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(
    configs/ck26/training-free.yaml
    configs/ck26/uniqgen.yaml
    configs/ck26/uniqgen-gold-oracle.yaml
    configs/db26/training-free.yaml
    configs/ck26/algebra-agent.yaml
    configs/db26/algebra-agent.yaml
  )
fi
for config in "${CONFIGS[@]}"; do
  "$PY" -m hermes run "$config"
done
"$PY" -m hermes report --write
