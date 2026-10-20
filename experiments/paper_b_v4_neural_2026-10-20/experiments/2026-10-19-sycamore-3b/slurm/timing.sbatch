#!/usr/bin/env bash
# One GPU: score the 64 panels with the 3B model (s_0 for the predictions) and time two rounds of the most expensive
# configuration (planted cycle, alpha 0.2, beta 9.6). Prints peak GPU memory.
#SBATCH --job-name=psipo3b-timing
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
/work/users/f/a/fanyao/claude/env/bin/python src/panel_po_steer.py --panels data/panels_q3b.json --policy /work/users/f/a/fanyao/claude/models/Qwen2.5-3B-Instruct --inner 40 --lr 2e-5 --ids $IDS --objective ipo --oracle npz:data/oracle_planted.npz --lam 1.0 --alpha 0.2 --beta 9.6 --rounds 2 --out runs/timing
nvidia-smi --query-gpu=name,memory.used --format=csv
