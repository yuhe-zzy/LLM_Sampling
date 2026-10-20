#!/usr/bin/env bash
# CPU job: build Fan's own Python environment (versions match the Azure VM runs) and download Qwen2.5-3B-Instruct.
#SBATCH --job-name=psipo3b-setup
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/users/f/a/fanyao/claude/psipo3b/logs/setup-%j.out
set -euo pipefail
ENV=/work/users/f/a/fanyao/claude/env
module load anaconda/2025.12-2
[ -x $ENV/bin/python ] || conda create -y -q -p $ENV python=3.12
$ENV/bin/pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
$ENV/bin/pip install -q transformers==5.13.1 numpy==2.5.1 scipy
$ENV/bin/python -c "import torch, transformers, numpy; print('torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__)"
$ENV/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='/work/users/f/a/fanyao/claude/models/Qwen2.5-3B-Instruct'))"
du -sh /work/users/f/a/fanyao/claude/models/Qwen2.5-3B-Instruct $ENV
