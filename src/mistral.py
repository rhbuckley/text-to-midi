import unsloth  # important: this must be imported first
import os
import glob
import json
import pretty_midi
import soundfile as sf
from datasets import load_dataset
from src.midi.parse import midi_to_json
from src.midi.synthesize import midi_to_wav, pretty_midi_to_base64_wav

import torch
from trl import SFTTrainer
from dotenv import load_dotenv, find_dotenv
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments, pipeline
from unsloth import FastLanguageModel, is_bfloat16_supported
from src.midi.synthesize import pretty_midi_to_wav
from src.midi.tokenize import encode_midi_to_tokens, decode_tokens_to_midi

# ================= UNSLOTH CONFIG =================
BASE_MODEL_NAME = "unsloth/mistral-7b-v0.3"
MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = False
CHAT_TEMPLATE = "chatml"

# ================= WANDB CONFIG =================

load_dotenv(find_dotenv())


def create_jsonl_file(output_dir: str, job_id: int = 0, total_jobs: int = 1):
    """
    Create a JSONL file from the given data.

    This creates multiple JSONL files, one for every 100k lines.
    They will be saved to the output directory with the same name as
    the output file, but with a number added to the end.
    """

    from src.download import MidiCaps
    from src.tokenizer import MidiTokenizer

    tokenizer = MidiTokenizer()
    ds = MidiCaps(tokenizer)

    os.makedirs(output_dir, exist_ok=True)

    file_idx = job_id
    start = len(ds) // total_jobs * job_id
    limit = len(ds) // total_jobs * (job_id + 1)

    with open(f"{output_dir}/part_{file_idx:05d}.jsonl", "w") as f:
        for i in range(start, limit):
            caption, midi_path = ds[i]

            tokenized_midi = encode_midi_to_tokens(midi_path)
            tokenized_midi = " ".join(tokenized_midi)

            json_obj = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who converts text prompts into MIDI-like tokens.",
                    },
                    {
                        "role": "user",
                        "content": caption,
                    },
                    {
                        "role": "assistant",
                        "content": tokenized_midi,
                    },
                ],
            }

            f.write(json.dumps(json_obj) + "\n")


def finetune(dataset_path: str, resume_from_checkpoint: bool = False):
    os.environ["WANDB_PROJECT"] = "text2midi-llm"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # get the peft model
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    files = glob.glob(f"{dataset_path}/*.jsonl")
    dataset = load_dataset(
        "json", data_files=files, split="train", cache_dir="./.cache"
    )
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # type: ignore
        train_dataset=dataset,
        dataset_num_proc=2,  # type: ignore
        packing=False,  # type: ignore # unsloth: Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            # max_steps=60,  # Set num_train_epochs = 1 for full training runs
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="steps",
            save_steps=50,
            report_to="wandb",  # Use this for WandB etc
            resume_from_checkpoint="outputs",
            save_total_limit=5,
        ),
    )

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    model.save_pretrained("lora_model")  # Local saving
    tokenizer.save_pretrained("lora_model")


def generate(
    prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 512,
    save_to_disk: bool = False,
    model_checkpoint_path: str = "lora_model",
):
    """
    Generate a MIDI from a text prompt using a trained model.
    """
    # get the pretrained model / tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto",
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    # load the model checkpoint
    try:
        print(f"Loading model checkpoint from {model_checkpoint_path}")
        model.load_adapter(model_checkpoint_path)
        print(f"Model checkpoint loaded from {model_checkpoint_path}")
    except Exception as e:
        print(f"Error loading model checkpoint from {model_checkpoint_path}: {e}")
        raise e

    # prepare for inference
    model.eval()

    # create the inference pipeline
    inference_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # note: do not specify device, let the `accelerate` library handle it
    )

    # --- Prepare inference input ---
    # This is the history leading up to where you want the model to generate.
    inference_conversation = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # --- Apply the chat template for INFERENCE ---
    formatted_input_prompt = tokenizer.apply_chat_template(
        inference_conversation,
        tokenize=False,  # pipeline does this
        add_generation_prompt=True,  # <--- IMPORTANT FOR INFERENCE
    )

    outputs = inference_pipe(
        formatted_input_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        # eos_token_id=tokenizer.eos_token_id # Pipeline often handles this, but good to be aware
        # pad_token_id=tokenizer.eos_token_id # Often set pad = eos for open-ended generation
    )

    # get the generated text
    encoded_midi_string = outputs[0]["generated_text"]  # type: ignore

    # decode the MIDI string
    midi = decode_tokens_to_midi(encoded_midi_string)

    # save the MIDI to a file
    wav_data = pretty_midi_to_base64_wav(midi)
    midi_json = midi_to_json(midi)

    if save_to_disk:
        midi.write("output.mid")
        midi_to_wav("output.mid")

    # return the WAV path
    return wav_data, midi_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", action="store_true")

    # The directory to save the JSONL files.
    parser.add_argument("--jsonl-dir", type=str, default="jsonl_data", required=False)

    # This is used as an offset (you can skip lines so we can batch)
    parser.add_argument("--jsonl-job-id", type=int, default=0)
    parser.add_argument("--jsonl-total-jobs", type=int, default=1)

    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--checkpoint", action="store_true", default=False)
    args = parser.parse_args()

    if args.jsonl:
        create_jsonl_file(
            args.jsonl_dir,
            job_id=args.jsonl_job_id,
            total_jobs=args.jsonl_total_jobs,
        )

    if args.finetune:
        finetune(args.jsonl_dir, resume_from_checkpoint=args.checkpoint)
