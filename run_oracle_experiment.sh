#!/bin/bash
#SBATCH -p h100_all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=72:00:00
#SBATCH -J pref_oracle
#SBATCH -o /work/users/f/a/fanyao/ipo/logs/%x-%j.out
#SBATCH -e /work/users/f/a/fanyao/ipo/logs/%x-%j.err

set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: sbatch run_oracle_experiment.sh <ipo|dpo> <transitive|cyclic> <alpha> <lambda> <beta>"
  echo "Example: sbatch run_oracle_experiment.sh ipo transitive 0.8 0.8 10"
  exit 1
fi

LOSS="$1"
CASE="$2"
ALPHA="$3"
LAMBDA="$4"
BETA="$5"

if [ "${LOSS}" = "ipo" ]; then
  RUN_SCRIPT="scripts/run_ipo_oracle.py"
elif [ "${LOSS}" = "dpo" ]; then
  RUN_SCRIPT="scripts/run_dpo_oracle.py"
else
  echo "LOSS must be ipo or dpo, got: ${LOSS}"
  exit 1
fi

source ~/h100env/bin/activate
cd /nas/longleaf/home/fanyao/ipo

WORKROOT=/work/users/f/a/fanyao/ipo
MODEL_PATH="${MODEL_PATH:-/nas/longleaf/home/fanyao/ipo/model/Qwen2.5-3B}"
OUT_TAG="${LOSS}_${CASE}_a${ALPHA}_l${LAMBDA}_b${BETA}"

if [ "${CASE}" = "transitive" ]; then
  PAIRS_PATH=/nas/longleaf/home/fanyao/ipo/data/processed/pairs_train.jsonl
  EVAL_PATH=/nas/longleaf/home/fanyao/ipo/data/processed/eval_prompt_responses_1000.jsonl
elif [ "${CASE}" = "cyclic" ]; then
  PAIRS_PATH=/nas/longleaf/home/fanyao/ipo/data/processed/pairs_train_cyclic.jsonl
  EVAL_PATH=/nas/longleaf/home/fanyao/ipo/data/processed/eval_prompt_responses_cyclic_1000.jsonl
else
  echo "CASE must be transitive or cyclic, got: ${CASE}"
  exit 1
fi

mkdir -p "${WORKROOT}/logs"
mkdir -p "${WORKROOT}/checkpoints_oracle/${OUT_TAG}"
mkdir -p "${WORKROOT}/logs_oracle/${OUT_TAG}"
mkdir -p "${WORKROOT}/oracle_baselines"

BASE_MODEL_NAME="$(basename "${MODEL_PATH}")"
ORACLE_BASELINE_CACHE="${WORKROOT}/oracle_baselines/pi0_${BASE_MODEL_NAME}_${CASE}_prompts500_M4_seed777.jsonl"

echo "===== JOB START ====="
date
hostname
echo "loss=${LOSS}, case=${CASE}, alpha=${ALPHA}, lambda_on=${LAMBDA}, beta=${BETA}"
echo "model=${MODEL_PATH}"
echo "pairs=${PAIRS_PATH}"
echo "eval=${EVAL_PATH}"
echo "oracle_cache=${ORACLE_BASELINE_CACHE}"
nvidia-smi

python "${RUN_SCRIPT}" \
  --model_path "${MODEL_PATH}" \
  --pairs_path "${PAIRS_PATH}" \
  --eval_prompts_path "${EVAL_PATH}" \
  --preference_case "${CASE}" \
  --out_dir "${WORKROOT}/checkpoints_oracle/${OUT_TAG}" \
  --log_dir "${WORKROOT}/logs_oracle/${OUT_TAG}" \
  --seed 0 \
  --auto_stop 0 \
  --iters 100 \
  --alpha "${ALPHA}" \
  --lambda_on "${LAMBDA}" \
  --tau 1 \
  --beta "${BETA}" \
  --mix_eps 0.05 \
  --w_clip_min 0.1 \
  --w_clip_max 10.0 \
  --train_sample_size 1000 \
  --pairs_per_prompt 2 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-5 \
  --warmup_ratio 0.03 \
  --score_batch_size 4 \
  --epochs_per_iter 1 \
  --max_length 1537 \
  --dump_each_iter 1 \
  --save_iter_adapters 0 \
  --save_initial_adapter 0 \
  --save_final_adapter 0 \
  --generated_eval_num_prompts 500 \
  --generated_eval_num_candidates 10 \
  --generated_eval_keep_top_k 5 \
  --generated_eval_max_new_tokens 256 \
  --generated_eval_do_sample 1 \
  --generated_eval_temperature 0.8 \
  --generated_eval_top_p 0.95 \
  --generated_eval_seed 123 \
  --cycle_burn_in 10 \
  --enable_oracle 1 \
  --oracle_model_path nvidia/Llama-3.1-Nemotron-70B-Reward-HF \
  --oracle_torch_dtype bfloat16 \
  --oracle_device_map auto \
  --oracle_max_length 4096 \
  --oracle_batch_size 4 \
  --oracle_eval_every 1 \
  --oracle_num_prompts 500 \
  --oracle_num_responses 4 \
  --oracle_generation_batch_size 8 \
  --oracle_max_new_tokens 256 \
  --oracle_do_sample 1 \
  --oracle_temperature 0.8 \
  --oracle_top_p 0.95 \
  --oracle_seed 777 \
  --oracle_baseline_cache_path "${ORACLE_BASELINE_CACHE}" \
  --oracle_reuse_baseline_cache 1

echo "===== JOB END ====="
date

