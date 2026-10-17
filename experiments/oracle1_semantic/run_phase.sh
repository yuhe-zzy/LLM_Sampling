#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?Missing frozen launch root}
PHASE=${2:?Missing phase}
UPDATES=${3:?Missing update limit}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec /work/users/y/u/yuhe32/h100env312/bin/python "$ROOT/source/experiments/oracle1_semantic/worker.py" \
  --root "$ROOT" --phase "$PHASE" --max-updates "$UPDATES"
