#!/usr/bin/env bash
set -euo pipefail
ROOT=${ORACLE2_LAUNCH_ROOT:?Submit through the checked queue entry point}
PYTHON=/work/users/y/u/yuhe32/h100env312/bin/python
CODE="$ROOT/source/experiments/oracle2"
PLAN="$ROOT/plan.json"
OUT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "$PLAN")
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PYTHON" "$CODE/campaigns/oracle2_real_20261009/queue.py" --root "$ROOT" --verify
"$PYTHON" "$CODE/campaigns/oracle2_real_20261009/queue.py" --root "$ROOT" --check-budget
case "${1:?Missing phase}" in
  audit)
    "$PYTHON" "$CODE/score_records.py" --plan "$PLAN" \
      --records "$ROOT/private_data/candidate_records.jsonl" --output "$OUT/candidate_scores"
    "$PYTHON" "$CODE/audit_candidates.py" --plan "$PLAN" --scores "$OUT/candidate_scores"
    ;;
  baseline)
    "$PYTHON" "$CODE/generate_evaluation.py" --plan "$PLAN"
    ;;
  train|generate)
    INDEX=${SLURM_ARRAY_TASK_ID:?Missing array index}
    [[ "$INDEX" =~ ^[0-5]$ ]] || exit 2
    RUN_ID=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][int(sys.argv[2])]["run_id"])' "$PLAN" "$INDEX")
    if [[ "$1" == train ]]; then
      "$PYTHON" "$CODE/campaigns/oracle2_real_20261009/queue.py" --root "$ROOT" --check-baseline
      COMMIT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$ROOT/deployment.json")
      "$PYTHON" "$CODE/train_oracle2.py" --plan "$PLAN" --run-id "$RUN_ID" \
        --review "$ROOT/audit_review.json" --source-commit "$COMMIT" --execute
    else
      "$PYTHON" "$CODE/generate_evaluation.py" --plan "$PLAN" --run-id "$RUN_ID"
    fi
    ;;
  wrscore)
    "$PYTHON" "$CODE/evaluate_wr.py" collect --plan "$PLAN" --records "$OUT/evaluation_records.jsonl"
    "$PYTHON" "$CODE/score_records.py" --plan "$PLAN" \
      --records "$OUT/evaluation_records.jsonl" --output "$OUT/evaluation_scores"
    "$PYTHON" "$CODE/evaluate_wr.py" summarize --plan "$PLAN" \
      --records "$OUT/evaluation_records.jsonl" --scores "$OUT/evaluation_scores" --output "$OUT/wr_results"
    ;;
  *) exit 2 ;;
esac
