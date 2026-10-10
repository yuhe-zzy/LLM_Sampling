#!/usr/bin/env bash
#SBATCH --job-name=hist_trend_1003
#SBATCH --partition=h100_all
#SBATCH --account=rc_fanyao_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-43%6
#SBATCH --no-requeue
#SBATCH --output=/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_trend_1003-%A_%a.out
#SBATCH --error=/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_trend_1003-%A_%a.err
set -euo pipefail
ROOT=/work/users/y/u/yuhe32/ipo/diagnostics/history_trend_sweep_20261003
PYTHON=/work/users/y/u/yuhe32/h100env312/bin/python
INDEX=${SLURM_ARRAY_TASK_ID:?Missing task index}
[[ "$INDEX" =~ ^[0-9]+$ ]] && ((INDEX < 44)) || exit 2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PYTHON" "$ROOT/queue_sweep.py" --verify
RUN_ID=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][int(sys.argv[2])]["run_id"])' "$ROOT/experiment_plan.json" "$INDEX")
exec "$PYTHON" "$ROOT/source/experiments/cyclic_history/run_cyclic_history.py" \
    --plan "$ROOT/experiment_plan.json" --run-id "$RUN_ID" \
    --execute --approve-calibrated-micropilot
