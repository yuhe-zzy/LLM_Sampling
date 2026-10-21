#!/usr/bin/env bash
#SBATCH --job-name=hist_cal_A_0928
#SBATCH --partition=h100_all
#SBATCH --account=rc_fanyao_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-3%4
#SBATCH --dependency=afterok:4593149
#SBATCH --no-requeue
#SBATCH --output=/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_cal_A_0928-%A_%a.out
#SBATCH --error=/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_cal_A_0928-%A_%a.err
set -euo pipefail
ROOT=/work/users/y/u/yuhe32/ipo/diagnostics/history_calibrated_20260928_stage_a
export PROJECT_ROOT="$ROOT/source"
export PYTHON=/work/users/y/u/yuhe32/h100env312/bin/python
export MODEL_PATH=/work/users/y/u/yuhe32/ipo/model/Qwen2.5-1.5B
export DATA_ROOT=/work/users/y/u/yuhe32/ipo/data/processed
export OUTPUT_ROOT=/work/users/y/u/yuhe32/ipo_runs/cyclic_history_calibrated_20260928
export APPROVED_CALIBRATED_STAGE=A
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PYTHON" "$ROOT/queue_stage_a.py" --verify
exec bash "$PROJECT_ROOT/slurm/cyclic_history_calibrated.sh" A
