#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --partition=gpu_h100
#SBATCH --time=40:00:00
#SBATCH --mem=84G
#SBATCH --exclusive
#SBATCH --job-name=qwen1
#SBATCH -o ./log/qwen_%j.out

# =============== 生成时间戳 ===============
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_DIR="./log"
OUTPUT_DIR="./check_qwen3/checkpoints_${TIMESTAMP}"



# =============== 加载 Snellius 2023 工具链 + CUDA ===============
module load 2023
module load CUDA/12.4.0

nvidia-smi

if [ -z "$CUDA_HOME" ]; then
    echo "❌ CUDA_HOME not set after module load!"
    exit 1
fi
echo "✅ CUDA_HOME = $CUDA_HOME"

# =============== 激活 Conda 环境 ===============
source /home/khe/miniconda3/bin/activate qwen

# =============== 跳过 DeepSpeed CUDA 算子编译（关键！）===============
export DS_SKIP_CUDA_BUILD=1

# =============== 环境检查（直接内联，避免 /tmp 问题）===============
echo "🔍 Checking PyTorch and CUDA..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
if not torch.cuda.is_available():
    exit(1)
print('✅ CUDA check passed.')
"

if [ $? -ne 0 ]; then
    echo "❌ CUDA check failed!"
    exit 1
fi

# =============== 启动训练 ===============
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=4




torchrun --nproc_per_node=$NPROC_PER_NODE train.py \
  --dataset DGM4 \
  --epochs 8 \
  --batch-size 4 \
  --lr 5e-5 \
  --lr-scheduler cosine \
  --warmup-ratio 0.1 \
  --lr-scale 1.5 \
  --regular-weight 0.05 \
  --train-json /home/aorus/He/qwen-vl-finetune/SAMM_data/SAMM-with-CAP/train1.json \
  --val-json /home/aorus/He/qwen-vl-finetune/SAMM_data/SAMM-with-CAP/val.json
