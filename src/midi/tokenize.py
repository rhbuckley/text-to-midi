import pretty_midi


# Constants for token representation
TIME_RESOLUTION = 100  # Steps per second
MAX_TIME_SHIFT = 100  # Maximum time shift in steps (1 second)
VELOCITY_BINS = 32  # Number of bins to quantize velocity


PIECE_START = "<>"
PIECE_END = "</>"
INSTRUMENT = "I"
TIME_SHIFT = "T"
VELOCITY = "V"
NOTE_ON = "N"
NOTE_OFF = "O"


def quantize_velocity(velocity, bins=VELOCITY_BINS):
    """Quantize velocity into discrete bins."""
    if velocity < 0:
        velocity = 0
    if velocity > 127:
        velocity = 127
    return min(int(velocity / (128 / bins)), bins - 1)


def dequantize_velocity(bin_index, bins=VELOCITY_BINS):
    """Dequantize velocity from bin index."""
    # Return the middle value of the bin
    return int((bin_index + 0.5) * (128 / bins))


def encode_midi_to_tokens(midi_path, time_resolution=TIME_RESOLUTION):
    """
    Encodes a MIDI file into a sequence of tokens suitable for language model processing.

    The token format uses special markers:
    - `PIECE_START`: Marks the beginning of the MIDI sequence.
    - `INSTRUMENT=<program>`: Selects the active instrument (MIDI program number).
    - `TIME_SHIFT=<steps>`: Advances time by the specified number of steps. Time is quantized
                             based on `time_resolution`. Maximum shift is `MAX_TIME_SHIFT`.
    - `NOTE_ON=<pitch>`: Starts a note with the given MIDI pitch.
    - `NOTE_OFF=<pitch>`: Stops the note with the given MIDI pitch.
    - `VELOCITY=<bin>`: Sets the velocity for the *next* NOTE_ON event. Velocity is
                        quantized into `VELOCITY_BINS` bins.
    - `PIECE_END`: Marks the end of the MIDI sequence.

    Args:
        midi_path (str): Path to the MIDI file.
        time_resolution (int): Steps per second for time quantization.

    Returns:
        list[str]: A list of string tokens representing the MIDI events.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return []  # Return empty list on error

    events = []
    for instrument in midi.instruments:
        # Sort notes by start time, then pitch (for determinism)
        sorted_notes = sorted(instrument.notes, key=lambda x: (x.start, x.pitch))
        for note in sorted_notes:
            start_step = round(note.start * time_resolution)
            end_step = round(note.end * time_resolution)
            velocity_bin = quantize_velocity(note.velocity)

            # Add events: (time_step, type, value, instrument_program)
            events.append(
                (start_step, INSTRUMENT, instrument.program, instrument.program)
            )
            events.append((start_step, VELOCITY, velocity_bin, instrument.program))
            events.append((start_step, NOTE_ON, note.pitch, instrument.program))
            events.append((end_step, NOTE_OFF, note.pitch, instrument.program))

    if not events:
        return [PIECE_START, PIECE_END]

    # Sort all events by time, then by type priority (INSTRUMENT, VELOCITY, NOTE_ON, NOTE_OFF)
    type_priority = {INSTRUMENT: 0, VELOCITY: 1, NOTE_ON: 2, NOTE_OFF: 3}
    events.sort(key=lambda x: (x[0], type_priority.get(x[1], 99)))

    tokens = [PIECE_START]
    current_time_step = 0

    for time_step, event_type, value, instr_prog in events:
        # --- Add Time Shift ---
        time_diff = time_step - current_time_step
        if time_diff > 0:
            # Decompose large time shifts into smaller chunks
            while time_diff > 0:
                shift = min(time_diff, MAX_TIME_SHIFT)
                tokens.append(f"{TIME_SHIFT}={shift}")
                time_diff -= shift
                current_time_step += shift  # Update current time *after* adding token

        # --- Add Instrument Change (if needed) ---
        # Only add INSTRUMENT token if it changes *for this specific event's instrument*
        # This logic might need refinement depending on how multi-instrument handling is desired downstream.
        # A simpler approach might be to always emit INSTRUMENT before the first event of that instrument.
        # For now, let's assume the model learns the context.
        # if instr_prog != active_instrument: # Re-enable if explicit instrument switching is needed per event
        #    tokens.append(f"INSTRUMENT={instr_prog}")
        #    active_instrument = instr_prog

        # --- Add Event Token ---
        tokens.append(f"{event_type}={value}")

        # Update current_time_step *after* processing events at this step
        # current_time_step = time_step # This was updated during time shift handling

    tokens.append(PIECE_END)
    return tokens


def decode_tokens_to_midi(
    tokens, time_resolution=TIME_RESOLUTION
) -> pretty_midi.PrettyMIDI:
    """
    Decodes a sequence of tokens (generated by `encode_midi_to_tokens`) back into a MIDI object.
    Handles overlapping notes correctly.

    Args:
        tokens (list[str]): The list of string tokens.
        time_resolution (int): Steps per second used during encoding.

    Returns:
        pretty_midi.PrettyMIDI: The decoded MIDI object.
    """
    if isinstance(tokens, str):
        tokens = tokens.split(" ")

    midi = pretty_midi.PrettyMIDI()
    instruments = {}  # Dictionary to hold instruments {program: Instrument}
    current_time = 0.0
    current_instrument_program = 0  # Default instrument program
    # Store active notes as a list associated with each (instrument, pitch) pair
    # active_notes = {(instrument_program, pitch): [(start_time1, velocity1), (start_time2, velocity2), ...]}
    active_notes = {}
    pending_velocity = 64  # Default velocity if not specified

    for token in tokens:
        if token == PIECE_START or token == PIECE_END:
            continue

        try:
            event_type, value = token.split("=", 1)
        except ValueError:
            print(f"Warning: Skipping malformed token: {token}")
            continue

        if event_type == INSTRUMENT:
            try:
                current_instrument_program = int(value)
                if current_instrument_program not in instruments:
                    instruments[current_instrument_program] = pretty_midi.Instrument(
                        program=current_instrument_program
                    )
            except ValueError:
                print(f"Warning: Invalid instrument program value: {value}. Skipping.")
        elif event_type == TIME_SHIFT:
            try:
                steps = int(value)
                if steps < 0:
                    print(
                        f"Warning: Negative time shift {steps} encountered. Skipping."
                    )
                    continue
                current_time += steps / time_resolution
            except ValueError:
                print(f"Warning: Invalid time shift value: {value}. Skipping.")
        elif event_type == VELOCITY:
            try:
                velocity_bin = int(value)
                if 0 <= velocity_bin < VELOCITY_BINS:
                    pending_velocity = dequantize_velocity(velocity_bin)
                else:
                    print(
                        f"Warning: Velocity bin {velocity_bin} out of range [0, {VELOCITY_BINS-1}]. Using default."
                    )
                    pending_velocity = 64
            except ValueError:
                print(f"Warning: Invalid velocity value: {value}. Skipping.")
                pending_velocity = 64  # Use default on error

        elif event_type == NOTE_ON:
            try:
                pitch = int(value)
                if not (0 <= pitch <= 127):
                    print(f"Warning: Invalid pitch value {pitch}. Skipping NOTE_ON.")
                    continue

                if current_instrument_program not in instruments:
                    print(
                        f"Warning: NOTE_ON for pitch {pitch} encountered before INSTRUMENT definition for program {current_instrument_program}. Creating default instrument."
                    )
                    instruments[current_instrument_program] = pretty_midi.Instrument(
                        program=current_instrument_program
                    )

                note_key = (current_instrument_program, pitch)
                # Initialize list if this is the first note for this key
                if note_key not in active_notes:
                    active_notes[note_key] = []

                # Add the new note start information to the list
                active_notes[note_key].append((current_time, pending_velocity))

                # Sort the list by start time to ensure NOTE_OFF pairs with the earliest NOTE_ON
                active_notes[note_key].sort(key=lambda x: x[0])

                # Reset pending velocity after use? Or keep it for subsequent notes? Let's keep it.
                # pending_velocity = 64 # Uncomment to reset velocity after each NOTE_ON
            except ValueError:
                print(f"Warning: Invalid pitch value: {value}. Skipping NOTE_ON.")
        elif event_type == NOTE_OFF:
            try:
                pitch = int(value)
                if not (0 <= pitch <= 127):
                    print(f"Warning: Invalid pitch value {pitch}. Skipping NOTE_OFF.")
                    continue

                note_key = (current_instrument_program, pitch)
                # Check if there are active notes for this key
                if note_key in active_notes and active_notes[note_key]:
                    # Retrieve the earliest started note (FIFO)
                    start_time, velocity = active_notes[note_key].pop(
                        0
                    )  # Remove the first element

                    # Ensure end time is not before start time
                    end_time = max(start_time, current_time)
                    if end_time > start_time:  # Avoid zero-duration notes
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=end_time,
                        )
                        if current_instrument_program in instruments:
                            instruments[current_instrument_program].notes.append(note)
                        else:
                            # Should not happen if NOTE_ON created the instrument, but handle defensively
                            print(
                                f"Error: Instrument {current_instrument_program} not found for NOTE_OFF. Discarding note."
                            )
                    # else: Note has zero or negative duration, discard.

                    # Clean up dictionary entry if list becomes empty
                    if not active_notes[note_key]:
                        del active_notes[note_key]
                else:
                    # Note OFF event without a corresponding active Note ON. Ignore or log.
                    # This can happen if the MIDI data is unusual or encoding/decoding has issues.
                    print(
                        f"Warning: NOTE_OFF for pitch {pitch} without active NOTE_ON for instrument {current_instrument_program} at time {current_time}. Skipping."
                    )
            except ValueError:
                print(f"Warning: Invalid pitch value: {value}. Skipping NOTE_OFF.")
        else:
            print(f"Warning: Unknown event type: {event_type}. Skipping token: {token}")

    # After processing all tokens, handle any remaining NOTE_ON events
    # (notes that started but never received a NOTE_OFF).
    # End them at the current_time.
    for (instr_prog, pitch), remaining_notes in active_notes.items():
        # print(f"Warning: Found {len(remaining_notes)} dangling NOTE_ON event(s) for instrument {instr_prog}, pitch {pitch}. Ending at time {current_time}.")
        for start_time, velocity in remaining_notes:
            # Ensure end time is not before start time
            end_time = max(start_time, current_time)
            if end_time > start_time:  # Avoid zero-duration notes
                note = pretty_midi.Note(
                    velocity=velocity, pitch=pitch, start=start_time, end=end_time
                )
                if instr_prog in instruments:
                    instruments[instr_prog].notes.append(note)
                else:
                    # This case should ideally not be reached if NOTE_ON created the instrument
                    print(
                        f"Error: Instrument {instr_prog} not found for dangling NOTE_ON cleanup. Discarding note."
                    )
            # else: Note has zero or negative duration, discard.

    # Add all created instruments to the MIDI object
    for instrument in instruments.values():
        if instrument.notes:  # Only add instruments that have notes
            # Sort notes within each instrument before adding (good practice)
            instrument.notes.sort(key=lambda note: note.start)
            midi.instruments.append(instrument)

    # If no instruments were added, return empty MIDI
    # if not midi.instruments:
    #    midi.instruments.append(pretty_midi.Instrument(program=0))

    return midi
