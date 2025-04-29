import io
import os
import zipfile
import requests
import warnings
import subprocess
import pretty_midi


def parse_midi_pretty(midi_file_path):
    """
    Parses a MIDI file using pretty_midi to extract note events.

    Args:
        midi_file_path (str): Path to the MIDI file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a note
              and contains 'pitch', 'velocity', 'time' (start time in seconds),
              and 'duration' (duration in seconds). Returns empty list on failure.
              Notes are sorted by start time.
    """
    notes = []
    try:
        # Load the MIDI file
        with warnings.catch_warnings():
            # Suppress common warnings like "Tempo event found at time..." if desired
            warnings.simplefilter("ignore", category=RuntimeWarning)
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)

        # Iterate over all instruments in the MIDI file
        for instrument in midi_data.instruments:
            # Skip drums if desired (optional)
            # if instrument.is_drum:
            #     continue

            # Iterate over all notes played by this instrument
            for note in instrument.notes:
                start_time = note.start  # Start time in seconds
                end_time = note.end  # End time in seconds
                duration = end_time - start_time

                # Ensure duration is positive (handle potential rounding errors or zero-length notes)
                if duration > 1e-5:
                    notes.append(
                        {
                            "pitch": note.pitch,
                            "velocity": note.velocity,
                            "time": start_time,
                            "duration": duration,
                            # Optional: Add instrument info if needed later
                            # 'instrument_program': instrument.program,
                            # 'instrument_name': instrument.name
                        }
                    )

        # Sort notes by start time (important for sequence representation)
        notes.sort(key=lambda x: x["time"])

    except Exception as e:
        print(f"Error parsing MIDI file {midi_file_path} with pretty_midi: {e}")
        return []

    return notes


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


def midi_to_wav(midi_filepath, wav_filepath=None):
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
    if os.path.isfile(wav_filepath):
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
