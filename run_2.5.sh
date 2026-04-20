#!/bin/bash
#SBATCH -p h100_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=40:00:00
#SBATCH -J ipo_grid
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -e
set -u

if [ "$#" -ne 2 ]; then
  echo "Usage: sbatch run_2.5.sh <alpha> <lambda>"
  exit 1
fi

ALPHA="$1"
LAMBDA="$2"
OUT_TAG="a${ALPHA}_l${LAMBDA}"

source ~/h100env/bin/activate
cd /nas/longleaf/home/fanyao/ipo

mkdir -p logs
mkdir -p /nas/longleaf/home/fanyao/ipo/logs_qwen25_3b_test
mkdir -p /nas/longleaf/home/fanyao/ipo/checkpoints_qwen25_3b_${OUT_TAG}

echo "===== JOB START ====="
date
hostname
echo "alpha=${ALPHA}, lambda_on=${LAMBDA}"
nvidia-smi

python scripts/run_iterative_ipo_fast.py \
  --model_path /nas/longleaf/home/fanyao/ipo/model/Qwen2.5-3B \
  --pairs_path /nas/longleaf/home/fanyao/ipo/data/processed/pairs_train.jsonl \
  --eval_prompts_path /nas/longleaf/home/fanyao/ipo/data/processed/eval_prompt_responses.jsonl \
  --out_dir /nas/longleaf/home/fanyao/ipo/checkpoints_qwen25_3b_${OUT_TAG} \
  --log_dir /nas/longleaf/home/fanyao/ipo/logs_qwen25_3b_test \
  --seed 0 \
  --auto_stop 1 \
  --max_iters 150 \
  --stop_min_iters 15 \
  --stop_patience 5 \
  --stop_tv_abs 0.005 \
  --osc_detect 1 \
  --osc_window 8 \
  --osc_min_switches 4 \
  --osc_tv_floor 0.01 \
  --alpha "${ALPHA}" \
  --lambda_on "${LAMBDA}" \
  --tau 10 \
  --beta 10 \
  --train_sample_size 1000 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 5e-5 \
  --warmup_ratio 0.03 \
  --score_batch_size 1 \
  --epochs_per_iter 1 \
  --max_length 1537 \
  --dump_each_iter 1

echo "===== JOB END ====="
date