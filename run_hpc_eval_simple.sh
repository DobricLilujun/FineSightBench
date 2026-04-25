#!/bin/bash -l
#SBATCH --job-name=finesightbench
#SBATCH --partition gpu
#SBATCH --account=p201223
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH -q default
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:2
#SBATCH --chdir=/home/users/u101059/FineSightBench

# 1) 基础目录
WORKDIR="/home/users/u101059/FineSightBench"
echo "Working directory: $WORKDIR"
REPO_DIR="$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 2) 安装 uv（如果没有）
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 3) 更新仓库
if [ -d "$REPO_DIR/.git" ]; then
  git pull --ff-only
else
  echo "Repo not found at $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"
mkdir -p logs


# 5) 创建虚拟环境
if [ ! -d ".venv" ]; then
  uv venv --python 3.11 .venv
fi

nvidia-smi 
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # 6) 安装依赖（优先锁文件）
# if [ -f "uv.lock" ] && [ -f "pyproject.toml" ]; then
#   uv sync
# elif [ -f "requirements.txt" ]; then
#   uv pip install -r requirements.txt
# elif [ -f "pyproject.toml" ]; then
#   uv pip install -e .
# else
#   echo "No dependency file found."
#   exit 1
# fi

python scripts/eval_all_requested_vlms_mass.py --models "google/gemma-4-31B-it"
