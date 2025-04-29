import os
import torch
from typing import Optional, TypedDict
from src.tokenizer import MidiTokenizer
from src.midi_utils import midi_to_wav
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Config


# ================================================
# Model Config
# ================================================


class ModelConfig(TypedDict):
    temperature: float
    max_length: int
    top_k: int
    top_p: float
    n_embd: int
    n_layer: int
    n_head: int
    output: str


# ================================================
# Device
# ================================================


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# ================================================
# Model
# ================================================


class TextToMIDIModel:
    def __init__(
        self,
        config: ModelConfig,
        from_model_path: Optional[str] = None,
        from_pretrained: Optional[str] = None,
    ):
        """
        Initialize the model with a given configuration.

        Args:
            config (ModelConfig): The configuration for the model.
            from_model_path (str, optional): The path to a model checkpoint to load.
            from_pretrained (str, optional): The name of a pre-trained model to load.
        """
        self.config = config
        self.tokenizer = MidiTokenizer(config["max_length"])

        # ================================================
        # Initialize the model
        # ================================================

        vocab_size = len(self.tokenizer)
        if from_pretrained:
            print(f"Loading model from pretrained: {from_pretrained}")
            self.model = GPT2LMHeadModel.from_pretrained(from_pretrained)
            self.model.resize_token_embeddings(vocab_size)
            print(f"Resized model token embeddings to: {vocab_size}")
        else:
            print("Initializing new GPT2 model from default config.")
            model_config = GPT2Config(
                # Use dynamic vocab size
                vocab_size=vocab_size,
                # Match tokenizer max length
                n_positions=self.config["max_length"],
                n_ctx=self.config["max_length"],
                n_embd=self.config["n_embd"],
                n_layer=self.config["n_layer"],
                n_head=self.config["n_head"],
                # Add the id of the end of sequence token
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # No need to resize if initializing from scratch with correct vocab size
            self.model = GPT2LMHeadModel(model_config)

        # ================================================
        # Load the pretrained weights if provided
        # ================================================

        if from_model_path:
            print(f"Loading model weights from local path: {from_model_path}")
            try:
                self.model.load_state_dict(
                    torch.load(from_model_path, map_location=device)
                )
            except Exception as e:
                print(f"Error loading model weights: {e}")
                raise e

        # ================================================
        # Load the model onto the device
        # ================================================

        self.model = self.model.to(device)  # type: ignore
        print(f"Model loaded on device: {self.model.device}")

        # ================================================
        # Prevent loading from both Hugging Face and local path
        # ================================================

        assert not (
            from_model_path and from_pretrained
        ), "Cannot provide both model_path and from_pretrained."

        # ================================================
        # Override the __call__ method to return a CausalLMOutputWithPast
        # ================================================

    def __call__(self, input_ids, attention_mask=None, labels=None):
        return self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cleanup: bool = False,
    ):
        """
        Generate a MIDI sequence from a text prompt.

        Args:
            prompt (str): The text prompt to generate a MIDI sequence from.
            max_length (int, optional): The maximum length of the generated sequence.
            temperature (float, optional): The temperature for the softmax function.
            top_k (int, optional): The number of top tokens to consider.
            top_p (float, optional): The cumulative probability threshold for top-k.
            cleanup (bool, optional): Whether to delete the generated MIDI file.

        Returns:
            torch.Tensor: The generated MIDI sequence.
        """
        # Set default values if not provided
        max_length = max_length or self.config["max_length"]
        temperature = temperature or self.config["temperature"]
        top_k = top_k or self.config["top_k"]
        top_p = top_p or self.config["top_p"]

        # Tokenize the prompt
        tokenized = self.tokenizer.tokenize(prompt)
        input_ids_tensor = tokenized["input_ids"].detach().cpu().long().to(device)  # type: ignore
        attention_mask_tensor = tokenized["attention_mask"].detach().cpu().long().to(device)  # type: ignore

        # Generate the MIDI sequence
        print(f"Generating MIDI for prompt: '{prompt}'")
        with torch.no_grad():  # Disable gradient calculations for inference
            output_sequences = self.model.generate(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,  # Use pad token id
                eos_token_id=self.tokenizer.eos_token_id,  # Use eos token id for stopping
            )

        if output_sequences is None or len(output_sequences) == 0:
            print("Model generation failed.")
            return None, None

        # Decode the generated sequence
        generated_sequence = output_sequences[0]

        try:
            # Detokenize the generated sequence (including prompt part)
            midi_string = self.tokenizer.detokenize(
                generated_sequence, return_strings=True
            )

            # Save the generated tokens to a MIDI file
            self.tokenizer.detokenize_to_file(generated_sequence, self.config["output"])
            print(f"MIDI file saved to: {self.config['output']}")

            # Convert MIDI to wav
            output_wav_path = midi_to_wav(self.config["output"])
            print(f"MP3 file saved to: {output_wav_path}")

            if cleanup:
                os.remove(self.config["output"])

            return output_wav_path, midi_string
        except Exception as e:
            print(f"Error during detokenization or MP3 conversion: {e}")
            return None, None

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
