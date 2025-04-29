from typing import Literal, TypedDict
from src.model import ModelConfig

# ================================================
# WandB configuration
# ================================================

import os
import wandb
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

# ================================================
# Configuration
# ================================================


class Config(TypedDict):
    wandb_project: str
    wandb_job_type: str
    wandb_mode: Literal["disabled", "online", "offline"]

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_split: float
    num_workers: int
    lr_scheduler_type: str
    warmup_steps: int
    gradient_accumulation_steps: int
    model_save_path: str

    model_config: ModelConfig
    sample_prompts: list[str]


# fmt: off

CONFIG: Config = {
    "wandb_project": "text2midi",       # Project name for W&B
    "wandb_job_type": "train",          # Job type for W&B
    "wandb_mode": "online",             # Mode for W&B

    "epochs": 20,				        # Number of training epochs
    "batch_size": 8,				    # Size of the training batch
    "learning_rate": 1e-4,				# Learning rate for the optimizer
    "weight_decay": 0.01,				# Weight decay for regularization
    "validation_split": 0.005,			# Fraction of data to use for validation
    "num_workers": 4,				    # Number of worker processes for data loading
    "lr_scheduler_type": "linear",		# Type of learning rate scheduler
    "warmup_steps": 500,				# Number of warmup steps for the learning rate scheduler
    "gradient_accumulation_steps": 8,	# Number of steps to accumulate gradients before updating
    "model_save_path": "checkpoints8",   # Path to save the model checkpoints

    "model_config": {
        "max_length": 1024,				# Maximum sequence length for the model
        "temperature": 0.7,				# Sampling temperature for text generation
        "top_k": 50,				    # Top-k sampling parameter
        "top_p": 0.9,				    # Top-p (nucleus) sampling parameter
        "n_embd": 768,				    # Dimensionality of the embeddings and hidden states
        "n_layer": 12,				    # Number of transformer layers
        "n_head": 12,				    # Number of attention heads
        "output": "generated_midi.mid"	# Path to save the generated MIDI file
    },

    "sample_prompts": [
        "A melancholic piano piece in C minor.",
        "Upbeat electronic dance track with a strong bassline.",
        "Simple acoustic guitar folk song.",
        "Jazz improvisation on a saxophone.",
        "Epic orchestral score for a battle scene.",
    ]
}

# fmt: on
