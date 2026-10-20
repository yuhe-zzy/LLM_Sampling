#!/usr/bin/env bash
# Five Qwen2.5-3B-Instruct dynamics runs (one GPU each), at most 2 at a time (Fan's 2-GPU allowance, 2026-10-19).
#SBATCH --job-name=psipo3b
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/work/users/f/a/fanyao/claude/psipo3b/logs/%x-%A_%a.out
set -euo pipefail
export PANEL_CH=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 128 rows per micro-batch ran out of memory at 3B
cd /work/users/f/a/fanyao/claude/psipo3b
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/runs.txt); NAME=${LINE%% *}; ARGS=${LINE#* }
mkdir -p runs/$NAME
echo "{\"job\": \"$SLURM_ARRAY_JOB_ID\", \"task\": $SLURM_ARRAY_TASK_ID, \"name\": \"$NAME\", \"node\": \"$(hostname)\", \"start\": \"$(date -Is)\", \"script_sha256\": \"$(sha256sum src/panel_po_steer.py | cut -c1-16)\"}" > runs/$NAME/receipt.json
/work/users/f/a/fanyao/claude/env/bin/python src/panel_po_steer.py --panels data/panels_q3b.json --policy /work/users/f/a/fanyao/claude/models/Qwen2.5-3B-Instruct --inner 40 --lr 2e-5 --ids $(cat data/ids_q3b.txt) $ARGS --out runs/$NAME
