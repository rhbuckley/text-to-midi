import base64
from src.config import CONFIG
from rp_deploy.get_model import get_model
from src.model import TextToMIDIModel, MidiTokenizer

WANDB_PROJECT_NAME = "text2midi"
WANDB_MODEL_NAME = "model_best"


def load_model_from_checkpoint(checkpoint_path: str):
    """
    Load the model and tokenizer from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint directory

    Returns:
        TextToMIDIModel: The loaded model with its tokenizer
    """
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

    Returns:
        Any: The result to be returned to the client
    """

    # extract the input data from the event
    input_data = event.get("input", {})

    # get the model from wandb
    model_path = get_model(WANDB_PROJECT_NAME, WANDB_MODEL_NAME)

    # load the model
    model = load_model_from_checkpoint(model_path)

    # generate the MIDI
    wav_path, midi_string = model.generate(input_data["prompt"])

    # base64 encode the wav file
    wav_base64 = ""
    if wav_path:
        with open(wav_path, "rb") as wav_file:
            wav_data = wav_file.read()
            wav_base64 = base64.b64encode(wav_data).decode("utf-8")

    return {"wav_file": wav_base64, "midi_data": midi_string}


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler })