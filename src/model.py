import os
import torch
from tqdm import tqdm
import wandb
import random
from dotenv import load_dotenv
from src.midi_utils import midi_to_mp3
from src.tokenizer import MidiTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Config

# ================================================
# WandB configuration (outside of class for convenience)
# ================================================
WANDB_PROJECT = "text-to-midi"
WANDB_JOB_TYPE = "train"
WANDB_MODE = "online"

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Configuration for the model, and its training
# ================================================
# Configuration for the model, and its training
# process.
# ================================================
CONFIG = {
    "epochs": 20, # number of epochs to train the model
    "batch_size": 16, # number of samples per batch
    "learning_rate": 0.0001, # learning rate for the optimizer
    "weight_decay": 0.01, # weight decay for the optimizer
    "max_length": 1024, # maximum length of the input sequence
    "temperature": 0.7, # temperature for the softmax function
    "top_k": 50, # number of top k tokens to sample from
    "top_p": 0.9 # top p for the softmax function
}


# ================================================
# Device configuration.
# ================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class TextToMIDIModel:
    def __init__(self, model_path=None, from_pretrained=None):
        self.tokenizer = MidiTokenizer(CONFIG["max_length"])

        if from_pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(from_pretrained)
        else:
            config = GPT2Config(
                vocab_size=50277,  # Size of the vocabulary (number of unique tokens). Increased from default 50257 to accommodate additional music-specific tokens
                n_positions=1024,  # Maximum sequence length the model can handle. This determines how many tokens can be processed in a single sequence
                n_ctx=1024,        # Context window size for attention mechanism. Should match n_positions for GPT-2
                n_embd=768,        # Dimension of the embedding vectors and hidden states throughout the model
                n_layer=6,         # Number of transformer layers in the model. More layers = more capacity but slower training
                n_head=12          # Number of attention heads. Each head can focus on different aspects of the input sequence
            )
            self.model = GPT2LMHeadModel(config)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)  # type: ignore

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))

        assert not (model_path and from_pretrained), "Cannot provide both model_path and from_pretrained"

    def generate_midi(self, prompt, max_length=1024, temperature=0.7):
        # Ensure your tokenize method returns attention_mask
        tokenized_output = self.tokenizer.tokenize(prompt)

        # Check if tokenized_output contains the expected keys 'input_ids', 'attention_mask'
        # Directly attempt to access the keys. This is more robust
        # than checking the type first, as HF tokenizer outputs
        # might be custom objects that behave like dicts.
        try:
            input_ids_tensor = tokenized_output['input_ids'].to(device)  # type: ignore
            attention_mask_tensor = tokenized_output['attention_mask'].to(device)  # type: ignore
        except (TypeError, KeyError) as e:
            print(f"Error accessing input_ids or attention_mask from tokenizer output: {e}")
            return None
        
        input_ids_tensor = tokenized_output['input_ids'].to(device)  # type: ignore
        attention_mask_tensor = tokenized_output['attention_mask'].to(device) # type: ignore

        # generate output
        output = self.model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=CONFIG["top_k"],
            top_p=CONFIG["top_p"],
            pad_token_id=self.tokenizer.eos_token_id
        )

        try:
            # Ensure detokenize_to_file handles tensor input correctly
            self.tokenizer.detokenize_to_file(output[0], "output.mid")
            midi_to_mp3("output.mid", "output.mp3")
            return "output.mp3", self.tokenizer.detokenize(output[0], return_strings=True)
        except Exception as e:
            print(f"Failed to generate MIDI from text: {e}")
            return None, None # Return None on failure

    def train(self, text_to_midi_pairs):
        # Initialize wandb run
        run = wandb.init(project=WANDB_PROJECT, job_type=WANDB_JOB_TYPE, config=CONFIG, mode=WANDB_MODE)
        if run is None:
             print("Warning: wandb.init() returned None. Check WANDB_MODE.")

        self.model.train() # Set model to training mode
        print(f"Training on device: {self.model.device}")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )

        # Convert text_to_midi_pairs to list if it's not already
        pairs_list = list(text_to_midi_pairs)
        total_pairs = len(pairs_list)
        print(f"Total available pairs: {total_pairs}")

        # --- Outer loop for epochs with tqdm ---
        for epoch in tqdm(range(CONFIG["epochs"]), desc="Training Epochs"):
            total_loss = 0
            num_processed = 0

            # Randomly sample 10000 pairs for this epoch
            epoch_pairs = random.sample(pairs_list, min(10000, total_pairs))

            # --- Inner loop for data pairs with tqdm ---
            progress_bar = tqdm(epoch_pairs, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", unit="pair", leave=False)
            for text, midi_file_path in progress_bar: # Assuming it yields path now
                # 1. Tokenize (on CPU)
                # This takes the string prompt and the string path to the MIDI file
                tokenized_output = self.tokenizer.tokenize_from_file(text, midi_file_path)

                # Check if tokenization was successful
                if tokenized_output is None:
                    # Optionally log a warning, but avoid printing too much inside the loop
                    # print(f"Skipping pair due to tokenization issue: {text}, {midi_file}")
                    continue # Skip this iteration

                # 2. Move tensors to the correct device
                # Ensure tokenized_output contains the expected keys 'input_ids', 'attention_mask'
                try:
                    input_ids = tokenized_output['input_ids'].to(device)  # type: ignore
                    attention_mask = tokenized_output['attention_mask'].to(device)  # type: ignore
                except KeyError as e:
                    print(f"Error accessing keys in tokenized_output: {e}. Skipping item.")
                    continue
                except AttributeError: # Handle cases where output might not be dict with tensors
                     print(f"Error: tokenized_output is not in expected format. Skipping item.")
                     continue


                # 3. Perform forward pass, calculate loss
                try:
                    # Use keyword arguments for the model call
                    # Labels are typically input_ids for Causal LM
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    loss = outputs.loss

                    if loss is None: # Should not happen if labels are provided, but check
                        print("Warning: Loss is None. Skipping backward pass for this item.")
                        continue

                except Exception as model_e:
                     print(f"Error during model forward pass: {model_e}. Skipping item.")
                     continue

                # 4. Backward pass and optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_processed += 1
                # Optional: Update tqdm postfix with running loss
                progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/num_processed)


            # --- End of Epoch ---
            if num_processed > 0:
                 avg_loss = total_loss / num_processed
                 print(f"Epoch {epoch+1}/{CONFIG['epochs']} completed. Average Loss: {avg_loss:.4f}")
                 # Log average loss to wandb
                 if run: run.log({"epoch": epoch + 1, "average_loss": avg_loss})
            else:
                 print(f"Epoch {epoch+1}/{CONFIG['epochs']} completed. No items processed.")
                 if run: run.log({"epoch": epoch + 1, "average_loss": None})


            # --- Saving and Validation ---
            # Save model checkpoint locally (consider saving based on validation loss later)
            model_save_path = f"model_epoch_{epoch+1}.pt"
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Model checkpoint saved to {model_save_path}")

            # Log model artifact to wandb
            if run:
                try:
                    artifact = wandb.Artifact(f"model_epoch_{epoch+1}", type="model", metadata={"epoch": epoch+1, "loss": avg_loss if num_processed > 0 else None})
                    artifact.add_file(model_save_path)
                    run.log_artifact(artifact)
                    print(f"Logged artifact {artifact.name} to W&B.")
                except Exception as artifact_e:
                    print(f"Error logging model artifact to W&B: {artifact_e}")


            # Generate a sample MIDI file from the model and log to wandb
            print("Generating sample audio...")
            # Set model to eval mode for generation
            self.model.eval()
            with torch.no_grad(): # Disable gradient calculation for generation
                 generated_mp3_path = None
                 generated_midi_strings = None
                 ret = self.generate_midi("A happy song about a frog") # Using a fixed prompt for sampling
                 if ret is not None:
                    generated_mp3_path, generated_midi_strings = ret

            # Set model back to train mode
            self.model.train()

            if generated_mp3_path and os.path.exists(generated_mp3_path) and run:
                try:
                    print(f"Logging generated audio: {generated_mp3_path}")
                    audio = wandb.Audio(generated_mp3_path, caption=f"Sample Audio Epoch {epoch+1}")
                    run.log({"sample_audio": audio}) # Use a distinct key like "sample_audio"
                    run.log({"sample_midi": generated_midi_strings})
                except Exception as audio_e:
                    print(f"Error logging generated audio to W&B: {audio_e}")
            elif generated_mp3_path:
                 print(f"Generated audio file (not logged to W&B): {generated_mp3_path}")
            else:
                 print("Sample audio generation failed for this epoch.")

        # --- End of Training ---
        print("Training finished.")
        if run:
             wandb.finish()


if __name__ == "__main__":
    from src.download import MidiCaps

    model = TextToMIDIModel(from_pretrained="gpt2")
    dataset = MidiCaps(tokenizer=model.tokenizer)
    model.train(dataset)