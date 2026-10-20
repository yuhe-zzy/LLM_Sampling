#!/usr/bin/env bash
# One GPU: build the 3B model's own short-answer panels with the v4 builder (same prompts, K = 4, M = 48, min_prob 0.05, seed 0)
# and score them with Skywork-Reward-Llama-3.1-8B-v0.2 at revision d4117fbfd81b72f41b96341238baa1e3e90a4ce1 (the v4 reward model).
#SBATCH --job-name=psipo3b-panels
#SBATCH --partition=h100_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/users/f/a/fanyao/claude/psipo3b/logs/panels-%j.out
set -euo pipefail
cd /work/users/f/a/fanyao/claude/psipo3b
ENV=/work/users/f/a/fanyao/claude/env; M=/work/users/f/a/fanyao/claude/models
$ENV/bin/pip install -q pandas pyarrow
RMD=$M/Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b
[ -f $RMD/model.safetensors.index.json ] || $ENV/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Skywork/Skywork-Reward-Llama-3.1-8B-v0.2', revision='d4117fbfd81b72f41b96341238baa1e3e90a4ce1', local_dir='$RMD')"
$ENV/bin/python src/build_panels.py --policy $M/Qwen2.5-3B-Instruct --data /work/users/f/a/fanyao/claude/psipo3b/data/novelty-bench --out data/panels_q3b.json --rms $RMD
