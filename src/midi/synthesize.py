import base64
import io
import os
import zipfile
import requests
import subprocess
import pretty_midi
import soundfile as sf


def get_soundfont():
    """
    Downloads and extracts the FluidR3_GM soundfont if it doesn't exist locally.

    This function checks if the FluidR3_GM soundfont (.sf2) file exists in the local
    directory. If not, it downloads a zip file containing the soundfont from an S3 bucket
    and extracts it to the appropriate location.

    Returns:
        str: The filepath to the extracted soundfont file (.sf2)
    """
    # Define the target path for the soundfont file
    soundfont_filepath = "./soundfont/FluidR3_GM/FluidR3_GM.sf2"

    # Create the directory structure if it doesn't exist and download if needed
    if not os.path.exists(soundfont_filepath):
        # Create all necessary parent directories
        os.makedirs(os.path.dirname(soundfont_filepath), exist_ok=True)

        # URL for the zipped soundfont file
        url = "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip"

        # Download the zip file and extract its contents
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(os.path.dirname(soundfont_filepath))
        print("Soundfont downloaded and extracted to", soundfont_filepath)

    return soundfont_filepath


def pretty_midi_to_wav(midi: pretty_midi.PrettyMIDI, output_path: str, fs: int = 44100):
    """
    Convert a pretty_midi.PrettyMIDI object to a WAV file.
    """
    audio = midi.fluidsynth(fs)
    sf.write(output_path, audio, fs)


def pretty_midi_to_base64_wav(midi: pretty_midi.PrettyMIDI, fs: int = 44100) -> str:
    """
    Convert a pretty_midi.PrettyMIDI object to a base64 encoded WAV string.

    Args:
        midi: The pretty_midi.PrettyMIDI object to convert.
        fs: The desired sample rate for the output audio.

    Returns:
        A base64 encoded string representing the WAV audio data.
    """
    # Synthesize the MIDI data into audio samples (NumPy array)
    audio_data = midi.fluidsynth(fs=fs)

    # Create an in-memory binary buffer
    memory_file = io.BytesIO()

    # Write the audio data to the buffer as a WAV file
    # Explicitly specify the format since there's no filename extension
    sf.write(memory_file, audio_data, fs, format="WAV")

    # Reset the buffer's position to the beginning
    # This is crucial before reading the content back
    memory_file.seek(0)

    # Read the binary WAV data from the buffer
    wav_bytes = memory_file.read()

    # Close the buffer (optional, but good practice)
    memory_file.close()

    # Encode the binary WAV data to base64 bytes
    base64_bytes = base64.b64encode(wav_bytes)

    # Decode the base64 bytes into a string (UTF-8)
    base64_string = base64_bytes.decode("utf-8")

    return base64_string


def midi_to_wav(midi_filepath, wav_filepath=None, overwrite=True):
    """
    Converts a MIDI file to a WAV file using fluidsynth.

    Args:
        midi_filepath (str): Path to the MIDI file.
        wav_filepath (str, optional): Path to save the WAV file. If not provided,
            the function will create a default path by replacing the '.mid' extension
            with '.wav' in the MIDI file's path.

    Returns:
        str: The filepath to the generated WAV file.
    """

    # just rename the midi file to wav
    if wav_filepath is None:
        wav_filepath = midi_filepath.replace(".mid", ".wav")

    # Check if the .wav file already exists
    if os.path.isfile(wav_filepath) and not overwrite:
        print(f"{wav_filepath} already exists, skipping")
        return wav_filepath
    else:
        print(f"Creating {wav_filepath} from {midi_filepath}")

        # Run the fluidsynth command to convert MIDI to WAV
        command = f"fluidsynth -r 48000 {get_soundfont()} -g 1.0 --quiet --no-shell {midi_filepath} -T wav -F {wav_filepath}"
        print(f"Running command: {command}")
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        _, stderr = process.communicate()

        if process.returncode != 0:
            print(
                f"Error converting {midi_filepath} to {wav_filepath}: {stderr.decode('utf-8')}"
            )
        else:
            print(f"Successfully created {wav_filepath}")

        return wav_filepath
