import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb
import random
import math
import warnings
from dotenv import load_dotenv
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Config
from transformers.optimization import get_scheduler # For learning rate scheduling (optional but good)

# Assuming these exist in your src directory
from src.midi_utils import midi_to_mp3
from src.tokenizer import MidiTokenizer
from src.download import MidiCaps # Assuming MidiCaps is a torch.utils.data.Dataset

# ================================================
# WandB configuration
# ================================================
WANDB_PROJECT = "text-to-midi-refactored" # Consider a new project name
WANDB_JOB_TYPE = "train"
WANDB_MODE = "disabled" # Set to "disabled" to turn off wandb

# Load API Key only if W&B is enabled
if WANDB_MODE != "disabled":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("Warning: WANDB_API_KEY not found in .env file. Wandb login skipped.")
        # Optionally switch mode to disabled if login fails or key is missing
        # WANDB_MODE = "disabled"

# ================================================
# Configuration
# ================================================
CONFIG = {
    "epochs": 20,
    "batch_size": 64, # Adjust based on GPU memory
    "learning_rate": 1e-4, # Common starting point for fine-tuning transformers
    "weight_decay": 0.01,
    "max_length": 1024, # Max sequence length for model and tokenizer
    "temperature": 0.7, # For generation
    "top_k": 50,        # For generation
    "top_p": 0.9,       # For generation
    "validation_split": 0.005, # Percentage of data to use for validation
    "num_workers": 4,       # Number of workers for DataLoader (adjust based on CPU cores)
    "lr_scheduler_type": "linear", # Type of learning rate scheduler
    "warmup_steps": 500,         # Steps for learning rate warmup
    "gradient_accumulation_steps": 1, # Accumulate gradients over N steps (effective batch size = batch_size * grad_accum)
    "save_best_model": True, # Save checkpoint only when validation loss improves
}

# Sample prompts for generation during training
SAMPLE_PROMPTS = [
    "A melancholic piano piece in C minor.",
    "Upbeat electronic dance track with a strong bassline.",
    "Simple acoustic guitar folk song.",
    "Jazz improvisation on a saxophone.",
    "Epic orchestral score for a battle scene.",
]

# ================================================
# Device configuration
# ================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

class CollateFn:
    def __init__(self, tokenizer, max_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, batch):
        # Assuming batch is a list of (text, midi_file_path) tuples
        texts = [item[0] for item in batch]
        midi_paths = [item[1] for item in batch]

        batch_input_ids = []
        batch_attention_masks = []
        valid_indices = [] # Keep track of successfully tokenized items

        for i, (text, path) in enumerate(zip(texts, midi_paths)):
            try:
                # Tokenize directly here. Assumes tokenize_from_file handles text+MIDI
                # and returns {'input_ids': tensor, 'attention_mask': tensor}
                tokenized_output = self.tokenizer.tokenize_from_file(text, path, max_length=self.max_length)

                if tokenized_output is not None and 'input_ids' in tokenized_output and 'attention_mask' in tokenized_output:
                    # Ensure tensors before padding
                    input_ids = torch.tensor(tokenized_output['input_ids'], dtype=torch.long)
                    attention_mask = torch.tensor(tokenized_output['attention_mask'], dtype=torch.long)

                    # Truncate if necessary
                    if len(input_ids) > self.max_length:
                        input_ids = input_ids[:self.max_length]
                        attention_mask = attention_mask[:self.max_length]

                    batch_input_ids.append(input_ids)
                    batch_attention_masks.append(attention_mask)
                    valid_indices.append(i)
                else:
                    print(f"Warning: Skipping invalid tokenized output for item {i} (Text: '{text}', Path: '{path}')")
            except Exception as e:
                print(f"Warning: Error tokenizing item {i} (Text: '{text}', Path: '{path}'): {e}. Skipping.")
                continue

        if not batch_input_ids: # If no items in batch were valid
            return None

        # Pad sequences within the batch
        current_max_len = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(batch_input_ids, batch_attention_masks):
            padding_length = current_max_len - len(ids)
            if padding_length > 0:
                # Pad using the tokenizer's pad_token_id
                pad_tensor = torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                padded_ids = torch.cat([ids, pad_tensor])

                # Pad attention mask with 0
                mask_pad_tensor = torch.zeros(padding_length, dtype=mask.dtype)
                padded_mask = torch.cat([mask, mask_pad_tensor])
            else:
                padded_ids = ids
                padded_mask = mask

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        # Stack tensors for the batch
        input_ids_tensor = torch.stack(padded_input_ids).to(self.device)
        attention_mask_tensor = torch.stack(padded_attention_masks).to(self.device)

        # Labels are the same as input_ids for LM training
        labels = input_ids_tensor.clone()
        # Ignore padding tokens in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels
        }

class TextToMIDIModel:
    def __init__(self, config=CONFIG, model_path=None, from_pretrained=None):
        self.config = config
        self.tokenizer = MidiTokenizer(self.config["max_length"]) # Assuming MidiTokenizer takes max_length

        # Ensure tokenizer has necessary attributes
        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
             warnings.warn("Tokenizer does not have pad_token_id set. Attempting to use eos_token_id.")
             if not hasattr(self.tokenizer, 'eos_token_id') or self.tokenizer.eos_token_id is None:
                   raise AttributeError("Tokenizer must have 'pad_token_id' or 'eos_token_id'.")
             # If you use eos_token_id as pad_token_id, make sure it's handled correctly everywhere
             # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if not hasattr(self.tokenizer, 'eos_token_id') or self.tokenizer.eos_token_id is None:
             raise AttributeError("Tokenizer must have 'eos_token_id'.")


        vocab_size = len(self.tokenizer) # Get vocab size dynamically
        print(f"Tokenizer vocabulary size: {vocab_size}")

        if from_pretrained:
            print(f"Loading model from pretrained: {from_pretrained}")
            self.model = GPT2LMHeadModel.from_pretrained(from_pretrained)
            # Resize embeddings if the tokenizer vocab size differs from the pre-trained model's original vocab size
            self.model.resize_token_embeddings(vocab_size)
            print(f"Resized model token embeddings to: {vocab_size}")
        else:
            print("Initializing new GPT2 model from scratch.")
            model_config = GPT2Config(
                # Use dynamic vocab size
                vocab_size=vocab_size,
                # Match tokenizer max length
                n_positions=self.config["max_length"],
                n_ctx=self.config["max_length"],
                n_embd=768,        # Dimension of embeddings (GPT-2 small)
                n_layer=6,         # Number of layers (Consider 12 for GPT-2 small standard)
                n_head=12,         # Number of attention heads (GPT-2 small)
                # Add pad_token_id to config if model uses it internally
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,  # type: ignore
            )
            self.model = GPT2LMHeadModel(model_config)
            # No need to resize if initializing from scratch with correct vocab size

        self.model.to(device)  # type: ignore
        print(f"Model loaded on device: {self.model.device}")

        if model_path:
            print(f"Loading model weights from local path: {model_path}")
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                print(f"Error loading state dict from {model_path}: {e}")
                raise

        # Prevent loading from both Hugging Face and local path
        assert not (model_path and from_pretrained), \
            "Cannot provide both model_path and from_pretrained."

    def generate_midi(self, prompt, output_dir="generated_midi"):
        """Generates MIDI from a text prompt."""
        os.makedirs(output_dir, exist_ok=True)
        output_basename = prompt.lower().replace(" ", "_")[:30] # Create a simple filename
        output_mid_path = os.path.join(output_dir, f"{output_basename}.mid")
        output_mp3_path = os.path.join(output_dir, f"{output_basename}.mp3")

        self.model.eval() # Set model to evaluation mode

        try:
            # Tokenize the prompt text ONLY. Assumes tokenizer handles text prompts.
            # We need input_ids and attention_mask for the prompt.
            # Using __call__ is standard for HuggingFace tokenizers.
            tokenized_prompt = self.tokenizer.tokenize(
                 prompt,
                 max_length=self.config["max_length"] # Use configured max length
            )
            input_ids_tensor = torch.tensor(tokenized_prompt['input_ids'], dtype=torch.long).to(device)  # type: ignore
            attention_mask_tensor = torch.tensor(tokenized_prompt['attention_mask'], dtype=torch.long).to(device)  # type: ignore

        except Exception as e:
            print(f"Error tokenizing prompt: {e}")
            return None, None

        print(f"Generating MIDI for prompt: '{prompt}'")
        with torch.no_grad(): # Disable gradient calculations for inference
            output_sequences = self.model.generate(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                max_length=self.config["max_length"], # Max length of entire sequence (prompt + generation)
                temperature=self.config["temperature"],
                top_k=self.config["top_k"],
                top_p=self.config["top_p"],
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id, # Use pad token id
                eos_token_id=self.tokenizer.eos_token_id # Use eos token id for stopping
            )

        if output_sequences is None or len(output_sequences) == 0:
             print("Model generation failed.")
             return None, None

        # Decode the generated sequence
        generated_sequence = output_sequences[0]

        try:
            # Detokenize the generated sequence (including prompt part)
            midi_string = self.tokenizer.detokenize(generated_sequence, return_strings=True) # Get string representation
            # Save the generated tokens to a MIDI file
            self.tokenizer.detokenize_to_file(generated_sequence, output_mid_path)
            print(f"MIDI file saved to: {output_mid_path}")

            # Convert MIDI to MP3
            midi_to_mp3(output_mid_path, output_mp3_path)
            print(f"MP3 file saved to: {output_mp3_path}")

            return output_mp3_path, midi_string # Return paths and string representation
        except Exception as e:
            print(f"Error during detokenization or MP3 conversion: {e}")
            return None, None

    def _evaluate(self, dataloader):
        """Performs evaluation on the validation set."""
        self.model.eval() # Set model to evaluation mode
        total_eval_loss = 0
        num_eval_batches = 0

        with torch.no_grad(): # Disable gradients during evaluation
             for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                  if batch is None: # Skip potentially empty batches from collate_fn
                      continue

                  # Data is already on device from collate_fn if implemented that way,
                  # otherwise move batch items to device here.
                  # input_ids = batch['input_ids'].to(device)
                  # attention_mask = batch['attention_mask'].to(device)
                  # labels = batch['labels'].to(device)
                  # Assuming collate_fn puts tensors on the correct device:
                  input_ids = batch['input_ids']
                  attention_mask = batch['attention_mask']
                  labels = batch['labels']

                  try:
                       outputs = self.model(
                           input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels
                       )
                       loss = outputs.loss
                       if loss is not None:
                            total_eval_loss += loss.item()
                            num_eval_batches += 1
                  except Exception as e:
                       print(f"Error during evaluation forward pass: {e}. Skipping batch.")
                       continue

        if num_eval_batches == 0:
            return float('inf') # Return infinity if no batches were evaluated

        avg_eval_loss = total_eval_loss / num_eval_batches
        self.model.train() # Set model back to training mode
        return avg_eval_loss

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Trains the model using DataLoaders and includes validation."""
        # Initialize wandb run
        run = wandb.init(
             project=WANDB_PROJECT,
             job_type=WANDB_JOB_TYPE,
             config=self.config, # Log hyperparameters
             mode=WANDB_MODE
        )
        if run is None and WANDB_MODE != "disabled":
             print("Warning: wandb.init() returned None despite mode='online'. Check W&B setup/connection.")

        self.model.train() # Ensure model is in training mode
        print(f"Starting training on device: {self.model.device}")

        # Create collate function instance
        collate_fn = CollateFn(self.tokenizer, self.config["max_length"], device)

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config["num_workers"],
            pin_memory=True if device.type == 'cuda' else False
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config["num_workers"],
            pin_memory=True if device.type == 'cuda' else False
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )

        # Calculate total training steps for scheduler
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.config["gradient_accumulation_steps"])
        max_train_steps = self.config["epochs"] * num_update_steps_per_epoch

        # Setup learning rate scheduler
        lr_scheduler = get_scheduler(
            name=self.config["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=self.config["warmup_steps"] * self.config["gradient_accumulation_steps"],
            num_training_steps=max_train_steps,
        )

        print(f"Total training steps: {max_train_steps}")
        print(f"Number of epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Gradient Accumulation Steps: {self.config['gradient_accumulation_steps']}")
        print(f"Effective batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")


        best_val_loss = float('inf')
        global_step = 0

        # --- Training Loop ---
        for epoch in range(self.config["epochs"]):
            self.model.train() # Ensure model is in train mode at start of epoch
            total_train_loss = 0
            num_train_batches = 0

            progress_bar = tqdm(
                 train_dataloader,
                 desc=f"Epoch {epoch+1}/{self.config['epochs']} Training",
                 unit="batch"
            )

            for step, batch in enumerate(progress_bar):
                if batch is None: # Skip potentially empty batches from collate_fn
                    continue

                # Data is already on device from collate_fn if implemented that way,
                # otherwise move batch items to device here.
                # input_ids = batch['input_ids'].to(device)
                # attention_mask = batch['attention_mask'].to(device)
                # labels = batch['labels'].to(device)
                # Assuming collate_fn puts tensors on the correct device:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                try:
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                    if loss is None:
                        print(f"Warning: Loss is None for step {step}. Skipping batch.")
                        continue

                    # Normalize loss for gradient accumulation
                    loss = loss / self.config["gradient_accumulation_steps"]

                    # Backward pass
                    loss.backward()

                except Exception as model_e:
                    print(f"Error during model forward/backward pass: {model_e}. Skipping step.")
                    # Optionally clear gradients if an error occurred mid-accumulation
                    if (step + 1) % self.config["gradient_accumulation_steps"] != 0:
                         optimizer.zero_grad() # Clear potentially corrupted gradients
                    continue

                total_train_loss += loss.item() * self.config["gradient_accumulation_steps"] # Scale back up for logging avg loss
                num_train_batches += 1 # Count actual batches processed before grad accum scaling

                # Optimizer step (perform after accumulation steps)
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log metrics per optimization step
                    if run:
                         current_lr = lr_scheduler.get_last_lr()[0]
                         run.log({
                             "train/step_loss": loss.item() * self.config["gradient_accumulation_steps"], # Log un-normalized loss for the step
                             "train/learning_rate": current_lr,
                             "global_step": global_step
                         })

                # Update progress bar description
                progress_bar.set_postfix(loss=f"{loss.item() * self.config['gradient_accumulation_steps']:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")


            # --- End of Epoch ---
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('nan')
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} completed.")
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            # --- Validation ---
            print("Running validation...")
            avg_val_loss = self._evaluate(val_dataloader)
            print(f"Average Validation Loss: {avg_val_loss:.4f}")

            # Log epoch metrics to wandb
            if run:
                run.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_train_loss,
                    "val/epoch_loss": avg_val_loss
                })

            # --- Saving Checkpoint ---
            output_dir = f"model_checkpoints_epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model_save_path = os.path.join(output_dir, "pytorch_model.bin") # Common name for HF models
            tokenizer_save_path = os.path.join(output_dir) # Directory to save tokenizer

            save_checkpoint = False
            if self.config["save_best_model"]:
                 if avg_val_loss < best_val_loss:
                      print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                      best_val_loss = avg_val_loss
                      save_checkpoint = True
                      # Save best model marker or specific name
                      best_model_path = "best_model_checkpoint/pytorch_model.bin"
                      os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                      torch.save(self.model.state_dict(), best_model_path)
                      # Save tokenizer with best model
                      self.tokenizer.save_pretrained(os.path.dirname(best_model_path)) # Assuming HF compatible tokenizer save
                      print(f"Best model checkpoint saved to {os.path.dirname(best_model_path)}")
                      # Log best metric to wandb summary
                      if run:
                           run.summary["best_validation_loss"] = best_val_loss
                           run.summary["best_epoch"] = epoch + 1
                 else:
                      print(f"Validation loss did not improve from {best_val_loss:.4f}.")
            else:
                 # Save every epoch if save_best_model is False
                 print(f"Saving model checkpoint for epoch {epoch+1}...")
                 save_checkpoint = True

            if save_checkpoint:
                 # Save current epoch model state
                 torch.save(self.model.state_dict(), model_save_path)
                 # Save tokenizer config with the checkpoint
                 self.tokenizer.save_pretrained(tokenizer_save_path) # Assumes tokenizer has save_pretrained
                 print(f"Model checkpoint saved to {output_dir}")

                 # Log model artifact to wandb
                 if run:
                      try:
                           artifact_name = f"model_epoch_{epoch+1}" if not self.config["save_best_model"] else f"model_best_val_loss_{avg_val_loss:.4f}"
                           artifact = wandb.Artifact(
                                artifact_name,
                                type="model",
                                metadata={"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
                           )
                           artifact.add_dir(output_dir) # Add the whole directory (model + tokenizer files)
                           run.log_artifact(artifact, aliases=["latest", f"epoch_{epoch+1}"])
                           if self.config["save_best_model"] and avg_val_loss == best_val_loss:
                                # Add 'best' alias to the artifact if it's the best one so far
                                run.link_artifact(artifact, f"{WANDB_PROJECT}/model_best", aliases=["best"])
                           print(f"Logged artifact '{artifact.name}' to W&B.")
                      except Exception as artifact_e:
                           print(f"Error logging model artifact to W&B: {artifact_e}")


            # --- Generate Sample Audio ---
            print("\nGenerating sample audio...")
            random_prompt = random.choice(SAMPLE_PROMPTS)
            generated_mp3_path, generated_midi_string = self.generate_midi(
                 random_prompt, output_dir="generated_samples"
            )

            # Logging generated audio to wandb
            if generated_mp3_path and os.path.exists(generated_mp3_path) and run:
                 try:
                      print(f"Logging generated audio: {generated_mp3_path}")
                      audio = wandb.Audio(generated_mp3_path, caption=f"Epoch {epoch+1}: {random_prompt}")
                      run.log({f"sample_audio_epoch_{epoch+1}": audio}) # Unique key per epoch
                      # Log the generated MIDI string representation as text
                      run.log({f"sample_midi_text_epoch_{epoch+1}": wandb.Html(f"<pre>{generated_midi_string}</pre>")})
                 except Exception as audio_e:
                      print(f"Error logging generated audio/text to W&B: {audio_e}")
            elif generated_mp3_path:
                 print(f"Generated sample audio file (not logged to W&B): {generated_mp3_path}")
            else:
                 print("Sample audio generation failed for this epoch.")

            print("-" * 50) # Separator between epochs

        # --- End of Training ---
        print("Training finished.")
        # Save final model checkpoint
        final_output_dir = "final_model_checkpoint"
        os.makedirs(final_output_dir, exist_ok=True)
        final_model_path = os.path.join(final_output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), final_model_path)
        self.tokenizer.tokenizer.save_pretrained(final_output_dir)
        print(f"Final model checkpoint saved to {final_output_dir}")

        if run:
             # Log final model as artifact
             try:
                  final_artifact = wandb.Artifact("final_model", type="model", metadata={"epochs": self.config["epochs"], "final_val_loss": avg_val_loss})
                  final_artifact.add_dir(final_output_dir)
                  run.log_artifact(final_artifact, aliases=["final"])
                  print("Logged final model artifact to W&B.")
             except Exception as e:
                  print(f"Error logging final model artifact: {e}")

             wandb.finish()


if __name__ == "__main__":
    print("Setting up model and dataset...")

    # --- Initialize Model ---
    # Choose one: fine-tune "gpt2" or train from scratch
    model = TextToMIDIModel(from_pretrained="gpt2", config=CONFIG)

    # --- Load and Prepare Dataset ---
    # Assuming MidiCaps is your torch.utils.data.Dataset
    # It should return (text, midi_file_path) pairs
    try:
        full_dataset = MidiCaps(tokenizer=model.tokenizer) # Pass tokenizer if needed by dataset
        print(f"Loaded dataset with {len(full_dataset)} samples.") # type: ignore
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # --- Split Dataset ---
    if CONFIG["validation_split"] > 0:
        total_size = len(full_dataset) # type: ignore
        val_size = int(total_size * CONFIG["validation_split"])
        train_size = total_size - val_size

        if train_size == 0 or val_size == 0:
            raise ValueError(f"Dataset split resulted in zero samples for train ({train_size}) or validation ({val_size}). Check validation_split ({CONFIG['validation_split']}) and dataset size ({total_size}).")

        print(f"Splitting dataset: Train ({train_size}), Validation ({val_size})")
        # Ensure reproducibility of split if needed
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42) # for reproducible splits
        )
    else:
        print("Using full dataset for training (no validation split).")
        train_dataset = full_dataset
        # Create a dummy validation dataset if none is provided, to avoid errors
        # Or adjust the train loop to handle val_dataset being None
        # For simplicity here, we'll raise an error if validation is expected but split is 0
        if CONFIG["save_best_model"]:
             raise ValueError("save_best_model is True, but validation_split is 0. Cannot determine best model without validation.")
        # If not saving best model, we could technically proceed without validation, but it's not recommended.
        # Let's create a small dummy validation set from train for the code structure to work.
        print("Warning: No validation split. Using a small subset of training data for validation metrics (not recommended for reliable evaluation).")
        if len(train_dataset) < 2: raise ValueError("Dataset too small for even a dummy validation split.")
        train_dataset, val_dataset = random_split(train_dataset, [len(train_dataset)-1, 1])


    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # --- Start Training ---
    print("Starting training process...")
    model.train(train_dataset, val_dataset)

    print("Script finished.")