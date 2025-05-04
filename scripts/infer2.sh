#!/usr/bin/bash
#SBATCH -n 4
#SBATCH -p rsingh47-gcondo --gres=gpu:4
#SBATCH --mem=128G
#SBATCH -o Job_infer2.out
#SBATCH -e Job_infer2.err
#SBATCH -t 48:00:00
#SBATCH --mail-user=tassallah_abdullahi@brown.edu
#SBATCH --mail-type=ALL

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


# python /gpfs/data/ceickhof/aabdul/ms-swift/data/create_infer_data.py
python /gpfs/data/ceickhof/aabdul/ms-swift/afrivox_infer_transcribe.py



# swift infer --adapters /gpfs/data/ceickhof/aabdul/ms-swift/output_transcribe/v0-20250424-195749/checkpoint-5960 \
#              --val_dataset /gpfs/data/ceickhof/aabdul/ms-swift/data/afrivox_transcribe2.jsonl \
#              --max_batch_size 4 \
#              --val_dataset_sample 10 \
#              --max_new_tokens 512 \
#             #  --infer_backend vllm \
            
            
            
#SBATCH -p 3090-gcondo --gres=gpu:4
#SBATCH -p rsingh47-gcondo --gres=gpu:4