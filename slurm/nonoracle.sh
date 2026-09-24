#!/usr/bin/env bash
#SBATCH --job-name=nonoracle-preference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-0%2
#SBATCH --output=nonoracle-%A_%a.out
#SBATCH --error=nonoracle-%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch, not on a login node}"
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
exec "${PYTHON:-python}" scripts/experiment.py \
  --config "${1:-configs/nonoracle_transitive.json}" --index "${SLURM_ARRAY_TASK_ID:-0}" --execute
