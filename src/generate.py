from src.model import TextToMIDIModel
from src.config import CONFIG
from src.tokenizer import MidiTokenizer

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
        from_model_path=f"./{checkpoint_path}/pytorch_model.bin"
    )
    
    # Ensure the model is using the loaded tokenizer
    model.tokenizer = tokenizer
    
    return model

def generate_midi(prompt: str, model: TextToMIDIModel):
    """
    Generate MIDI from a text prompt.
    
    Args:
        prompt (str): The text prompt to generate MIDI from
        model (TextToMIDIModel): The loaded model
        
    Returns:
        tuple: (mp3_path, midi_string) or (None, None) if generation fails
    """
    return model.generate(prompt)

if __name__ == "__main__":
    # Load model from checkpoint
    model = load_model_from_checkpoint("./checkpoint3")
    
    # Example generation
    prompt = "A melancholic piano piece in C minor."
    wav_path, midi_string = generate_midi(prompt, model)
    
    if wav_path and midi_string:
        print(f"Generated WAV saved to: {wav_path}")
        print(f"Generated MIDI string: {midi_string}")
    else:
        print("Generation failed.") 