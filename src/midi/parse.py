from pathlib import Path
import warnings
import pretty_midi


def parse_midi_pretty(midi: str | Path | pretty_midi.PrettyMIDI):
    """
    Parses a MIDI file using pretty_midi to extract note events.

    Args:
        midi (str | Path | pretty_midi.PrettyMIDI): Path to the MIDI file or a pretty_midi.PrettyMIDI object.

    Returns:
        list: A list of dictionaries, where each dictionary represents a note
              and contains 'pitch', 'velocity', 'time' (start time in seconds),
              and 'duration' (duration in seconds). Returns empty list on failure.
              Notes are sorted by start time.
    """
    if isinstance(midi, str | Path):
        midi_data = pretty_midi.PrettyMIDI(midi)
    elif isinstance(midi, pretty_midi.PrettyMIDI):
        midi_data = midi
    else:
        raise ValueError(f"Invalid MIDI input: {type(midi)}")

    notes = []
    try:
        # Load the MIDI file
        with warnings.catch_warnings():
            # Suppress common warnings like "Tempo event found at time..." if desired
            warnings.simplefilter("ignore", category=RuntimeWarning)

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
                            "instrument": instrument.name,
                            "program": instrument.program,
                        }
                    )

        # Sort notes by start time (important for sequence representation)
        notes.sort(key=lambda x: x["time"])

    except Exception as e:
        print(f"Error parsing MIDI file {midi} with pretty_midi: {e}")
        return []

    return notes


def midi_to_json(midi: str | Path | pretty_midi.PrettyMIDI):
    """
    Converts a MIDI file to a JSON representation using pretty_midi.

    Args:
        midi (str | Path | pretty_midi.PrettyMIDI): Path to the MIDI file or a pretty_midi.PrettyMIDI object.

    Returns:
        str: The JSON representation of the MIDI file.
    """
    # parse the MIDI file
    notes = parse_midi_pretty(midi)

    # return the JSON representation
    return {"data": notes}
