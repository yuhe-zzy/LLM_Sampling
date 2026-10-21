#!/usr/bin/env bash
#SBATCH --job-name=hist_ref90_0929
#SBATCH --partition=h100_all
#SBATCH --account=rc_fanyao_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_ref90_0929-%A_%a.out
#SBATCH --error=/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_ref90_0929-%A_%a.err
set -euo pipefail
ROOT=/work/users/y/u/yuhe32/ipo/diagnostics/history_reference90_20260929
PYTHON=/work/users/y/u/yuhe32/h100env312/bin/python
RUNS=(ipo_reference90_a1_b100_s0 dpo_reference90_a1_b100_s0)
INDEX=${SLURM_ARRAY_TASK_ID:?Missing task index}
[[ "$INDEX" =~ ^[0-1]$ ]] || exit 2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PYTHON" "$ROOT/queue_reference90.py" --verify
exec "$PYTHON" "$ROOT/source/experiments/cyclic_history/run_cyclic_history.py" \
    --plan "$ROOT/experiment_plan.json" --run-id "${RUNS[$INDEX]}" \
    --execute --approve-calibrated-micropilot
