#!/usr/bin/bash
#SBATCH -n 4
#SBATCH -p cs-all-gcondo --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -o Job.out
#SBATCH -e Job.err
#SBATCH -t 24:00:00
#SBATCH --mail-user=tassallah_abdullahi@brown.edu
#SBATCH --mail-type=END

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/users/tabdull1/scratch/huggingface
# export HF_HOME="/gpfs/data/superlab/cache/huggingface"

#export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH

cd /gpfs/data/ceickhof/aabdul/ms-swift
module load python/3.11.0s-ixrhc3q 
source ASR_omni/bin/activate
module load gcc/10.1.0-mojgbnp
module load cuda

which python

MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset AI-ModelScope/LaTeX_OCR:human_handwrite#20000 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
    
    
#SBATCH -p cs-all-gcondo --gres=gpu:1
#SBATCH -p rsingh47-gcondo --gres=gpu:4