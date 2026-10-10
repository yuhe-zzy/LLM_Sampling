#!/usr/bin/env bash
#SBATCH --job-name=history-calibrated
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-3%2
#SBATCH --output=history-calibrated-%A_%a.out
#SBATCH --error=history-calibrated-%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch only after approval and account preflight}"
stage="${1:?Specify approved stage A, B, or C}"
[[ "${APPROVED_CALIBRATED_STAGE:-}" == "$stage" ]] || {
  echo "Set APPROVED_CALIBRATED_STAGE to the separately approved stage" >&2; exit 2;
}
case "$stage" in
  A) runs=(ipo_ordinary_calibrated_s0 ipo_stable_calibrated_s0 dpo_ordinary_calibrated_s0 dpo_stable_calibrated_s0) ;;
  B) runs=(ipo_reference_calibrated_s0 ipo_feedback_calibrated_s0 dpo_reference_calibrated_s0 dpo_feedback_calibrated_s0) ;;
  C) runs=(ipo_mixed_calibrated_s0 dpo_mixed_calibrated_s0) ;;
  *) echo "Stage must be A, B, or C" >&2; exit 2 ;;
esac
index="${SLURM_ARRAY_TASK_ID:?An explicit array is required}"
[[ "$index" =~ ^[0-3]$ ]] && (( index < ${#runs[@]} )) || {
  echo "Invalid array index for this stage; Stage C requires --array=0-1%2" >&2; exit 2;
}
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
exec "${PYTHON:-python}" experiments/cyclic_history/run_cyclic_history.py \
  --plan experiments/cyclic_history/calibration/2026-09-28/experiment_plan.json \
  --run-id "${runs[$index]}" \
  --model-path "${MODEL_PATH:-model/Qwen2.5-1.5B}" \
  --eval-path "${DATA_ROOT:-data/processed}/eval_prompt_responses_cyclic_1000.jsonl" \
  --output-root "${OUTPUT_ROOT:-outputs}/cyclic_history_calibrated" \
  --execute --approve-calibrated-micropilot
