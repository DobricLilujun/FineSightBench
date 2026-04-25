#!/bin/bash
#SBATCH --job-name=finesightbench
#SBATCH --partition=your_partition
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

echo "[$(date)] start on $(hostname)"

# 1) 基础目录
WORKDIR="$HOME/work"
REPO_DIR="$WORKDIR/FineSightBench"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 2) 安装 uv（如果没有）
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 3) 克隆或更新仓库
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/DobricLilujun/FineSightBench.git
else
  cd "$REPO_DIR"
  git pull --ff-only
  cd "$WORKDIR"
fi

cd "$REPO_DIR"
mkdir -p logs

# 4) 可选模块加载（你们集群如果不用 module，就删掉）
module purge || true
module load python/3.11 || true
module load cuda/12.1 || true


cd FineSightBench
# 5) 创建虚拟环境
if [ ! -d ".venv" ]; then
  uv venv --python 3.11 .venv
fi
source .venv/bin/activate

# 6) 安装依赖（优先锁文件）
if [ -f "uv.lock" ] && [ -f "pyproject.toml" ]; then
  uv sync
elif [ -f "requirements.txt" ]; then
  uv pip install -r requirements.txt
elif [ -f "pyproject.toml" ]; then
  uv pip install -e .
else
  echo "No dependency file found."
  exit 1
fi

python scripts/eval_all_requested_vlms_mass.py --models "google/gemma-4-31B-it"
