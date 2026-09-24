#!/usr/bin/env bash
#SBATCH --job-name=cyclic-history
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-5%2
#SBATCH --output=history-%A_%a.out
#SBATCH --error=history-%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch only after the pilot is approved}"
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
runs=(ipo_baseline_s0 ipo_reference_s0 ipo_sampling_s0 dpo_baseline_s0 dpo_reference_s0 dpo_sampling_s0)
index="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$index" =~ ^[0-5]$ ]] || { echo "History index must be 0..5" >&2; exit 2; }
exec "${PYTHON:-python}" experiments/cyclic_history/run_cyclic_history.py \
  --run-id "${runs[$index]}" \
  --model-path "${MODEL_PATH:-model/Qwen2.5-1.5B}" \
  --eval-path "${DATA_ROOT:-data/processed}/eval_prompt_responses_cyclic_1000.jsonl" \
  --output-root "${OUTPUT_ROOT:-outputs}/cyclic_history" --execute
