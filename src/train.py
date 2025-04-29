import math
import os
import random
from typing import Any
import torch
import wandb
from tqdm.auto import tqdm
from src.config import CONFIG
from src.download import MidiCaps
from src.tokenizer import MidiTokenizer
from src.model import TextToMIDIModel, device
from transformers.optimization import get_scheduler
from torch.utils.data import random_split, DataLoader


class CollateFn:
    tokenizer: MidiTokenizer

    def __init__(self, tokenizer, max_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        assert isinstance(
            self.tokenizer.pad_token_id, int
        ), "Tokenizer pad_token_id must be an integer"
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    def __call__(self, batch):
        batch_input_ids = []
        batch_attention_masks = []
        valid_indices = []  # Keep track of successfully tokenized items

        for i, (text, path) in enumerate(batch):
            try:
                # Tokenize directly here. Assumes tokenize_from_file handles text+MIDI
                # and returns {'input_ids': tensor, 'attention_mask': tensor}
                tokenized_output = self.tokenizer.tokenize_from_file(
                    text, path, max_length=self.max_length
                )

                if (
                    tokenized_output is None
                    or "input_ids" not in tokenized_output
                    or "attention_mask" not in tokenized_output
                ):
                    print(
                        f"Warning: Skipping invalid tokenized output for item {i} (Text: '{text}', Path: '{path}')"
                    )
                    continue

                # Ensure tensors before padding
                input_ids = tokenized_output["input_ids"].detach().clone().long()  # type: ignore
                attention_mask = tokenized_output["attention_mask"].detach().clone().long()  # type: ignore

                # Apply squeeze(0) only if tensor is 2D and first dim is 1
                if input_ids.ndim == 2 and input_ids.shape[0] == 1:
                    input_ids = input_ids.squeeze(0)

                if attention_mask.ndim == 2 and attention_mask.shape[0] == 1:
                    attention_mask = attention_mask.squeeze(0)

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                valid_indices.append(i)

            except Exception as e:
                print(
                    f"Warning: Error tokenizing item {i} (Text: '{text}', Path: '{path}'): {e}. Skipping."
                )
                continue

        if not batch_input_ids:
            return None

        # Pad sequences within the batch
        current_max_len = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(batch_input_ids, batch_attention_masks):
            padding_length = current_max_len - len(ids)
            if padding_length > 0:
                # Pad using the tokenizer's pad_token_id
                pad_tensor = torch.full(
                    (padding_length,), self.pad_token_id, dtype=torch.long
                )
                padded_ids = torch.cat([ids, pad_tensor])

                # Pad attention mask with 0
                mask_pad_tensor = torch.zeros(padding_length, dtype=torch.long)
                padded_mask = torch.cat([mask, mask_pad_tensor])
            else:
                padded_ids = ids
                padded_mask = mask

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        # Stack tensors for the batch (keep on CPU for multiprocessing)
        input_ids_tensor = torch.stack(padded_input_ids)
        attention_mask_tensor = torch.stack(padded_attention_masks)

        # Labels are the same as input_ids for LM training
        labels = input_ids_tensor.clone()

        # Ignore padding tokens in loss calculation
        labels[labels == self.pad_token_id] = -100

        # Move to device in training loop
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels,
        }


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Train the Text-to-MIDI model")
    parser.add_argument(
        "--from_checkpoint",
        type=str,
        help="Path to a checkpoint to load from",
        default=None,
    )
    args = parser.parse_args()

    # ================================================
    # Model / Initial Dataset
    # ================================================

    # --- Initialize Model ---
    model = TextToMIDIModel(from_pretrained="gpt2", config=CONFIG["model_config"])
    tokenizer = model.tokenizer

    # Load from checkpoint if specified
    if args.from_checkpoint:
        # Initialize tokenizer with the same max length as the model config
        tokenizer = MidiTokenizer(CONFIG["model_config"]["max_length"])

        # Load the tokenizer from the checkpoint directory
        tokenizer.load_pretrained(args.from_checkpoint)

        # Initialize and load the model
        model = TextToMIDIModel(
            config=CONFIG["model_config"],
            from_model_path=f"./{args.from_checkpoint}/pytorch_model.bin",
        )

        # Ensure the model is using the loaded tokenizer
        model.tokenizer = tokenizer

    # --- Load and Prepare Dataset ---
    dataset = MidiCaps(tokenizer=model.tokenizer)

    # --- Split Dataset ---
    total_size = len(dataset)
    val_size = int(total_size * CONFIG["validation_split"])
    train_size = total_size - val_size

    # Ensure split is valid
    assert (
        train_size > 0 and val_size > 0
    ), f"Dataset split resulted in zero samples for train ({train_size}) or validation ({val_size}). Check validation_split ({CONFIG['validation_split']}) and dataset size ({total_size})."
    print(f"Splitting dataset: Train ({train_size}), Validation ({val_size})")

    # Ensure reproducibility of split if needed
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

    # ================================================
    # Dataloaders / Collate Function
    # ================================================

    # --- Create Collate Function ---
    collate_fn = CollateFn(
        tokenizer=tokenizer,
        max_length=CONFIG["model_config"]["max_length"],
        device=device,
    )

    # --- Create DataLoaders ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )

    # ================================================
    # Optimizers / Schedulers
    # ================================================

    # --- Initialize Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # --- Initialize Scheduler ---
    scheduler = get_scheduler(
        name=CONFIG["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=CONFIG["epochs"] * len(train_dataloader),
    )

    # ================================================
    # Training Loop
    # ================================================

    # Initialize metrics
    run = wandb.init(
        project=CONFIG["wandb_project"],
        job_type=CONFIG["wandb_job_type"],
        mode=CONFIG["wandb_mode"],
        config=dict(CONFIG),
    )

    n_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / CONFIG["gradient_accumulation_steps"]
    )
    n_total_steps = CONFIG["epochs"] * n_update_steps_per_epoch

    print(f"Total training steps: {n_total_steps}")
    print(f"Number of epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Gradient Accumulation Steps: {CONFIG['gradient_accumulation_steps']}")
    print(
        f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}"
    )

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{CONFIG['epochs']} Training",
            unit="batch",
        )

        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                # Forward pass
                outputs = model.forward(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss: Any = outputs.loss  # type: ignore
                assert loss is not None, "Loss is None"

                if loss is None:
                    print(f"Warning: Loss is None for step {step}. Skipping batch.")
                    continue

                # Normalize loss for gradient accumulation
                loss = loss / CONFIG["gradient_accumulation_steps"]

                # Backward pass
                loss.backward()

            except Exception as model_e:
                print(
                    f"Error during model forward/backward pass: {model_e}. Skipping step."
                )

                # clear gradients if an error occurred mid-accumulation
                if (step + 1) % CONFIG["gradient_accumulation_steps"] != 0:
                    optimizer.zero_grad()  # Clear potentially corrupted gradients
                    continue

            total_train_loss += loss.item() * CONFIG["gradient_accumulation_steps"]
            num_train_batches += 1

            # Optimizer step (perform after accumulation steps)
            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0 or step == len(
                train_dataloader
            ) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Log metrics per optimization step
            if run:
                current_lr = scheduler.get_last_lr()[0]
                run.log(
                    {
                        "train/step_loss": loss.item()
                        * CONFIG["gradient_accumulation_steps"],
                        "train/learning_rate": current_lr,
                        "global_step": global_step,
                    }
                )

            # Update progress bar description
            progress_bar.set_postfix(
                loss=f"{loss.item() * CONFIG['gradient_accumulation_steps']:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        # ================================================
        # End of Epoch
        # ================================================

        avg_train_loss = (
            total_train_loss / num_train_batches
            if num_train_batches > 0
            else float("nan")
        )
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} completed.")
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # ================================================
        # Validation
        # ================================================

        print("Running validation...")

        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        for batch in val_dataloader:
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model.forward(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss: Any = outputs.loss  # type: ignore
                assert loss is not None, "Loss is None"

                if loss is None:
                    print(f"Warning: Loss is None for step {step}. Skipping batch.")
                    continue

                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = (
            total_val_loss / num_val_batches if num_val_batches > 0 else float("nan")
        )
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Log epoch metrics to wandb
        if run:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_train_loss,
                    "val/epoch_loss": avg_val_loss,
                }
            )

        # --- Saving Checkpoint ---
        output_dir = f"{CONFIG['model_save_path']}/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)

        model_save_path = os.path.join(output_dir, "pytorch_model.bin")
        tokenizer_save_path = os.path.join(output_dir)

        # Save current epoch model state
        torch.save(model.state_dict(), model_save_path)
        model.tokenizer.save_pretrained(tokenizer_save_path)
        print(f"Model checkpoint saved to {output_dir}")

        # Log model artifact to wandb
        if run:
            try:
                artifact_name = f"model_epoch_{epoch+1}"
                artifact = wandb.Artifact(
                    artifact_name,
                    type="model",
                    metadata={
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    },
                )

                artifact.add_dir(
                    output_dir
                )  # Add the whole directory (model + tokenizer files)
                run.log_artifact(artifact, aliases=["latest", f"epoch_{epoch+1}"])
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    run.link_artifact(
                        artifact,
                        f"{CONFIG['wandb_project']}/model_best",
                        aliases=["best"],
                    )
                print(f"Logged artifact '{artifact.name}' to W&B.")

            except Exception as artifact_e:
                print(f"Error logging model artifact to W&B: {artifact_e}")

        # --- Generate Sample Audio ---
        try:
            print("\nGenerating sample audio...")
            random_prompt = random.choice(CONFIG["sample_prompts"])
            generated_outputs = model.generate(random_prompt, cleanup=True)

            if generated_outputs is None:
                print("Sample audio generation failed for this epoch.")
                continue

            # Unpack the generated outputs
            generated_mp3_path = generated_outputs[0]
            generated_midi_string = generated_outputs[1]

            # Logging generated audio to wandb
            if generated_mp3_path and os.path.exists(generated_mp3_path) and run:
                try:
                    print(f"Logging generated audio: {generated_mp3_path}")
                    audio = wandb.Audio(
                        generated_mp3_path, caption=f"Epoch {epoch+1}: {random_prompt}"
                    )

                    # log the audio and the midi string
                    run.log({f"sample_audio_epoch_{epoch+1}": audio})
                    run.log(
                        {
                            f"sample_midi_text_epoch_{epoch+1}": wandb.Html(
                                f"<pre>{generated_midi_string}</pre>"
                            )
                        }
                    )

                    # delete the generated mp3 file
                    os.remove(generated_mp3_path)

                except Exception as audio_e:
                    print(f"Error logging generated audio/text to W&B: {audio_e}")

            elif generated_mp3_path:
                print(
                    f"Generated sample audio file (not logged to W&B): {generated_mp3_path}"
                )

            else:
                print("Sample audio generation failed for this epoch.")
        except Exception as e:
            print(f"Error generating sample audio: {e}")

        print("-" * 50)

    # ================================================
    # End of Training
    # ================================================

    print("Training finished.")

    # Log final model as artifact
    if run:
        try:
            final_artifact = wandb.Artifact(
                "final_model",
                type="model",
                metadata={"epochs": CONFIG["epochs"], "final_val_loss": avg_val_loss},
            )
            final_artifact.add_dir(
                CONFIG["model_save_path"] + "/epoch_" + str(CONFIG["epochs"] + 1)
            )
            run.log_artifact(final_artifact, aliases=["final"])
            print("Logged final model artifact to W&B.")
        except Exception as e:
            print(f"Error logging final model artifact: {e}")
        finally:
            run.finish()
            wandb.finish()
