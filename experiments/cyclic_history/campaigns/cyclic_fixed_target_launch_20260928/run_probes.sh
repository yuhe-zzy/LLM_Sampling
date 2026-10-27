#!/usr/bin/env bash
#SBATCH --job-name=hist_fixed_target_0928
#SBATCH --partition=h100_all
#SBATCH --account=rc_fanyao_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_fixed_target_0928-%A_%a.out
#SBATCH --error=/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_fixed_target_0928-%A_%a.err
set -euo pipefail
ROOT=/work/users/y/u/yuhe32/ipo/diagnostics/history_fixed_target_20260928
PYTHON=/work/users/y/u/yuhe32/h100env312/bin/python
SOURCE_RUNS=/work/users/y/u/yuhe32/ipo_runs/cyclic_history_calibrated_20260928/cyclic_history_calibrated
OUTPUT=/work/users/y/u/yuhe32/ipo_runs/cyclic_fixed_target_20260928
RUNS=(ipo_ordinary_calibrated_s0 ipo_stable_calibrated_s0 dpo_ordinary_calibrated_s0 dpo_stable_calibrated_s0)
INDEX=${SLURM_ARRAY_TASK_ID:?Missing task index}
[[ "$INDEX" =~ ^[0-3]$ ]] || exit 2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PYTHON" "$ROOT/queue_probes.py" --verify
exec "$PYTHON" "$ROOT/source/run_fixed_target_probe.py" \
    --source-run "$SOURCE_RUNS/${RUNS[$INDEX]}" \
    --output "$OUTPUT/${RUNS[$INDEX]}_fixed_t20" --execute
