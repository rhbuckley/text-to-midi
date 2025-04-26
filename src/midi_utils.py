import os
import warnings
import mido
import pretty_midi
from pathlib import Path
from midi2audio import FluidSynth
from typing import Union, List, Dict, Any


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
                start_time = note.start # Start time in seconds
                end_time = note.end     # End time in seconds
                duration = end_time - start_time

                # Ensure duration is positive (handle potential rounding errors or zero-length notes)
                if duration > 1e-5: # Use a small threshold
                    notes.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'time': start_time,
                        'duration': duration
                        # Optional: Add instrument info if needed later
                        # 'instrument_program': instrument.program,
                        # 'instrument_name': instrument.name
                    })

        # Sort notes by start time (important for sequence representation)
        notes.sort(key=lambda x: x['time'])

    except Exception as e:
        print(f"Error parsing MIDI file {midi_file_path} with pretty_midi: {e}")
        return [] # Return empty list on error

    return notes


def midi_to_mp3(midi_file: Union[str, Path], output_file: Union[str, Path]) -> str:
    """
    Convert a MIDI file to an MP3 file.
    
    Args:
        midi_file (Union[str, Path]): Path to the input MIDI file
        
    Returns:
        str: Path to the output MP3 file
    """
    # Create output filename by replacing extension with .wav
    output_wav = os.path.splitext(midi_file)[0] + '.wav'
    output_mp3 = os.path.splitext(output_file)[0] + '.mp3'
    
    # Convert MIDI to WAV using FluidSynth
    fs = FluidSynth()
    fs.midi_to_audio(midi_file, output_wav)
    
    # Convert WAV to MP3 using ffmpeg
    os.system(f'ffmpeg -i {output_wav} -y -f mp3 -ab 128k -ac 2 -ar 44100 -vn {output_mp3}')
    
    # Clean up the temporary WAV file
    os.remove(output_wav)
    
    return output_mp3
