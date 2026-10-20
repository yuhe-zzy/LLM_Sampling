#!/usr/bin/env bash
# One GPU: score the 64 training panels (3B's own panels) with the 3B model: s_0 for the frozen predictions (no training: --rounds 0).
# (First used on 2026-10-19 with the 0.5B panels and --rounds 2 as the timing test: 59 s per round, peak 66.2 GiB.)
#SBATCH --job-name=psipo3b-score
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/users/f/a/fanyao/claude/psipo3b/logs/timing-%j.out
set -euo pipefail
export PANEL_CH=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 128 rows per micro-batch ran out of memory at 3B
cd /work/users/f/a/fanyao/claude/psipo3b
IDS=$(cat data/ids_q3b.txt)
/work/users/f/a/fanyao/claude/env/bin/python src/panel_po_steer.py --panels data/panels_q3b.json --policy /work/users/f/a/fanyao/claude/models/Qwen2.5-3B-Instruct --inner 40 --lr 2e-5 --ids $IDS --objective ipo --oracle npz:data/oracle_planted_q3b.npz --lam 1.0 --alpha 0.2 --beta 9.6 --rounds 0 --out runs/score_q3b
nvidia-smi --query-gpu=name,memory.used --format=csv
