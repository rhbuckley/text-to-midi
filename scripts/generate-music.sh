#!/bin/bash
#SBATCH --partition=gpu-preempt          # Request the GPU partition
#SBATCH --cpus-per-task=8                # Request CPUs (adjust based on data loading/needs)
#SBATCH --mem=16G                        # Request memory (e.g., 24GB); adjust as needed
#SBATCH --time=0-01:00:00                # Max wall time (e.g., 1 day); adjust as needed
#SBATCH --gres=gpu:2080ti:1              # 4x A100s GPU

#----------------------------------------------------------
# Environment Setup
#----------------------------------------------------------

module load conda/latest
conda activate text2midi

#----------------------------------------------------------
# Run the script
#----------------------------------------------------------

python -m src.deploy.handler \
    --test_input '{ "input": { "model": "mistral", "prompt": "a piano solo with a club beat" }}'