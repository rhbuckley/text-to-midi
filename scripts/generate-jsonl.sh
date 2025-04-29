#!/bin/bash
#SBATCH --partition=cpu             # Request a CPU partition (adjust if needed)
#SBATCH --cpus-per-task=8           # Request CPUs (adjust based on data loading/needs)
#SBATCH --mem=8G                    # Request memory (e.g., 24GB); adjust as needed
#SBATCH --time=1-00:00:00           # Max wall time (e.g., 1 day); adjust as needed
#SBATCH --array=0-10                # Job array: 0-10 tasks

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

# --- Run the Generation Script ---
# Pass the task ID and total tasks for distribution
echo "Starting Midistral generation task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"

# The MidiCaps dataset is 168385 entries ... we have 10 tasks,
# so each task will generate 16838 entries

# We need to skip the number of entries equal to the task ID
# so task 0 will skip 0 entries, task 1 will skip 16838 entries,
# task 2 will skip 33676 entries, etc.

JSONL_SIZE=$((168385 / 10))
SKIP_ENTRIES=$((SLURM_ARRAY_TASK_ID * 16838))

python -m src.mistral --jsonl --jsonl-dir ./output_jsonl --jsonl-skip $SKIP_ENTRIES --jsonl-size $JSONL_SIZE
echo "Finished Midistral generation task $SLURM_ARRAY_TASK_ID"