#!/bin/bash
#SBATCH --partition=uri-gpu         # Request the GPU partition
#SBATCH --cpus-per-task=8           # Request CPUs (adjust based on data loading/needs)
#SBATCH --mem=32G                   # Request memory (e.g., 24GB); adjust as needed
#SBATCH --time=1-00:00:00           # Max wall time (e.g., 1 day); adjust as needed
#SBATCH --gres=gpu:l40s:1           # 1x V100 GPU

# Output and Error Log Files (%j will be replaced by the job ID)
#SBATCH --output=slurm_logs/midi_train_%j.log
#SBATCH --error=slurm_logs/midi_train_%j.err

# Email notifications
#SBATCH --mail-type=ALL                 # Send email on ALL job events (BEGIN, END, FAIL)
#SBATCH --mail-user=rhbuckley@uri.edu   # Replace with your email

#----------------------------------------------------------
# Environment Setup
#----------------------------------------------------------

# Create log directory if it doesn't exist
mkdir -p slurm_logs

module load cuda/12.1
module load python/3.12
module load conda/latest
module load ffmpeg/7.0.2

# --- Activate Conda Environment ---
conda activate text2midi

# --- Run the Training Script ---
python -m src.train