#!/usr/bin/env bash
#SBATCH --job-name=oracle-sequencesum
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-0%1
#SBATCH --output=oracle-%A_%a.out
#SBATCH --error=oracle-%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch, not on a login node}"
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
exec "${PYTHON:-python}" scripts/experiment.py \
  --config "${1:-configs/oracle.json}" --index "${SLURM_ARRAY_TASK_ID:-0}" --execute
