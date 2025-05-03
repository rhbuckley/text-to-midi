import unsloth
import base64
import numpy as np
from src.config import CONFIG
from src.mistral import generate
from src.deploy.utils import get_model
from src.model import TextToMIDIModel, MidiTokenizer

# ================= WANDB CONFIG =================
WANDB_GPT2_PROJECT_NAME = "text2midi"
WANDB_GPT2_MODEL_NAME = "model_best"
WANDB_GPT2_VERSION = "latest"

WANDB_MISTRAL_PROJECT_NAME = "text2midi-llm"
WANDB_MISTRAL_MODEL_NAME = "model-a0czmwoi"
WANDB_MISTRAL_VERSION = "latest"


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
    

def download_mistral_model():
    """
    Load the model and tokenizer from a checkpoint.
    """
    checkpoint_path = get_model(
        WANDB_MISTRAL_PROJECT_NAME, WANDB_MISTRAL_MODEL_NAME, WANDB_MISTRAL_VERSION
    )
    
    return checkpoint_path

def load_gpt2_model():
    """
    Load the model and tokenizer from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint directory

    Returns:
        TextToMIDIModel: The loaded model with its tokenizer
    """
    checkpoint_path = get_model(
        WANDB_GPT2_PROJECT_NAME, WANDB_GPT2_MODEL_NAME, WANDB_GPT2_VERSION
    )

    # Initialize tokenizer with the same max length as the model config
    tokenizer = MidiTokenizer(CONFIG["model_config"]["max_length"])

    # Load the tokenizer from the checkpoint directory
    tokenizer.load_pretrained(checkpoint_path)

    # Initialize and load the model
    model = TextToMIDIModel(
        config=CONFIG["model_config"],
        from_model_path=f"./{checkpoint_path}/pytorch_model.bin",
    )

    # Ensure the model is using the loaded tokenizer
    model.tokenizer = tokenizer
    return model


def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata
        ```json
        {
            "input": {
                "prompt": <your prompt here>,
                "model": <"gpt2" or "mistral">,
                "temperature": <temperature>,
                "top_p": <top_p>,
                "top_k": <top_k>,
                "max_tokens": <max_new_tokens>
            }
        }
        ```

    Returns:
        Any: The result to be returned to the client
    """

    # extract the input data from the event
    input_data = event.get("input", {})

    # what model to use?
    model_name = input_data.get("model", "gpt2")
    prompt = input_data.get("prompt", "")
    temperature = input_data.get("temperature", 0.8)
    top_p = input_data.get("top_p", 0.9)
    top_k = input_data.get("top_k", 50)
    max_new_tokens = input_data.get("max_tokens", 512)

    if not prompt:
        return {"error": "Prompt is required"}

    if model_name == "gpt2":
        model = load_gpt2_model()
        outputs = model.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=max_new_tokens,
        )

    elif model_name == "mistral":
        outputs = generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            model_checkpoint_path=download_mistral_model(),
        )

    else:
        return {"error": "Invalid model name"}

    # check the outputs
    if not outputs or len(outputs) != 2 or outputs[0] is None or outputs[1] is None:
        return {"error": "Invalid outputs"}

    # unpack the outputs
    wav_data, midi_json = outputs

    return {"wav_file": wav_data, "midi_data": convert_numpy_types(midi_json)}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
