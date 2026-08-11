#!/bin/zsh
# Portable hardware-focused regression entry point.
set -e
ROOT="${GPU_TSU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export MK="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

zsh "$ROOT/ci/lint.sh"

for model in fifo mul uart xoshiro pbit prng_check gibbs_grid tensor quant4 qsite_golden sampling_isa; do
  "$PY" -m "golden.$model"
done

zsh "$ROOT/ci/run_units.sh"
echo "GPU-TSU CHIP GATES: PASS"
