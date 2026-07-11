#!/bin/bash
# Build the conda env once, under the project directory. Run in an
# interactive session with internet access:
#   srun -p short --cpus-per-task=8 --mem=32G --time=02:00:00 --pty /bin/bash
#   bash environment_setup.sh
set -euo pipefail

export PROJECT_BASE="${PROJECT_BASE:-$(cd "$(dirname "$0")" && pwd)}"
export HF_HOME="$PROJECT_BASE/hf_cache"
mkdir -p "$PROJECT_BASE" "$HF_HOME"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

if [ ! -d "$PROJECT_BASE/envs/evi" ]; then
  conda create --prefix="$PROJECT_BASE/envs/evi" -c conda-forge python=3.11 -y
fi

PY="$PROJECT_BASE/envs/evi/bin/python"
PIP="$PROJECT_BASE/envs/evi/bin/pip"

"$PIP" install --upgrade pip
# torch matching the loaded CUDA 12.1 build.
"$PIP" install torch==2.5.1 --index-url \
  https://download.pytorch.org/whl/cu121
"$PIP" install -r requirements.txt

"$PY" -c "import torch; print('torch', torch.__version__, \
'cuda', torch.cuda.is_available())"
echo "Env ready at $PROJECT_BASE/envs/evi"
