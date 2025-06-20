#!/bin/bash
#SBATCH --partition=uri-gpu,gpu     # Request the GPU partition
#SBATCH --cpus-per-task=8           # Request CPUs (adjust based on data loading/needs)
#SBATCH --mem=48G                   # Request memory (e.g., 24GB); adjust as needed
#SBATCH --time=1-00:00:00           # Max wall time (e.g., 1 day); adjust as needed
#SBATCH --gres=gpu:h100:1           # 1x H100 GPU

# Output and Error Log Files (%j will be replaced by the job ID)
#SBATCH --output=slurm_logs/fine_tune_%j.log
#SBATCH --error=slurm_logs/fine_tune_%j.err

# Email notifications
#SBATCH --mail-type=ALL                 # Send email on ALL job events (BEGIN, END, FAIL)
#SBATCH --mail-user=rhbuckley@uri.edu   # Replace with your email

#----------------------------------------------------------
# Environment Setup
#----------------------------------------------------------

# Create log directory if it doesn't exist
mkdir -p slurm_logs

module load --ignore_cache cuda/12.1
module load --ignore_cache shpc/0.1.26
module load --ignore_cache python/3.12
module load --ignore_cache conda/latest
module load --ignore_cache ffmpeg/7.0.2


# --- Activate Conda Environment ---
conda activate text2midi

# --- For Mistral-Finetune ---
# cd src_finetune
# export PYTHONPATH="$(pwd)"
# torchrun --nproc-per-node 1 --master_port=$((RANDOM + 10000)) -m train config/7B.yaml

# --- For Unsloth-Finetune ---

python -m src.mistral --finetune --checkpoint