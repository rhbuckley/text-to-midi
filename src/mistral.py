import json
import os
import torch
import pretty_midi
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
import math

# Constants for token representation
TIME_RESOLUTION = 100  # Steps per second
MAX_TIME_SHIFT = 100  # Maximum time shift in steps (1 second)
VELOCITY_BINS = 32  # Number of bins to quantize velocity


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
                (start_step, "INSTRUMENT", instrument.program, instrument.program)
            )
            events.append((start_step, "VELOCITY", velocity_bin, instrument.program))
            events.append((start_step, "NOTE_ON", note.pitch, instrument.program))
            events.append((end_step, "NOTE_OFF", note.pitch, instrument.program))

    if not events:
        return ["PIECE_START", "PIECE_END"]

    # Sort all events by time, then by type priority (INSTRUMENT, VELOCITY, NOTE_ON, NOTE_OFF)
    type_priority = {"INSTRUMENT": 0, "VELOCITY": 1, "NOTE_ON": 2, "NOTE_OFF": 3}
    events.sort(key=lambda x: (x[0], type_priority.get(x[1], 99)))

    tokens = ["PIECE_START"]
    current_time_step = 0
    active_instrument = -1  # Track active instrument

    for time_step, event_type, value, instr_prog in events:
        # --- Add Time Shift ---
        time_diff = time_step - current_time_step
        if time_diff > 0:
            # Decompose large time shifts into smaller chunks
            while time_diff > 0:
                shift = min(time_diff, MAX_TIME_SHIFT)
                tokens.append(f"TIME_SHIFT={shift}")
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

    tokens.append("PIECE_END")
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
    midi = pretty_midi.PrettyMIDI()
    instruments = {}  # Dictionary to hold instruments {program: Instrument}
    current_time = 0.0
    current_instrument_program = 0  # Default instrument program
    # Store active notes as a list associated with each (instrument, pitch) pair
    # active_notes = {(instrument_program, pitch): [(start_time1, velocity1), (start_time2, velocity2), ...]}
    active_notes = {}
    pending_velocity = 64  # Default velocity if not specified

    for token in tokens:
        if token == "PIECE_START" or token == "PIECE_END":
            continue

        try:
            event_type, value = token.split("=", 1)
        except ValueError:
            print(f"Warning: Skipping malformed token: {token}")
            continue

        if event_type == "INSTRUMENT":
            try:
                current_instrument_program = int(value)
                if current_instrument_program not in instruments:
                    instruments[current_instrument_program] = pretty_midi.Instrument(
                        program=current_instrument_program
                    )
            except ValueError:
                print(f"Warning: Invalid instrument program value: {value}. Skipping.")
        elif event_type == "TIME_SHIFT":
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
        elif event_type == "VELOCITY":
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

        elif event_type == "NOTE_ON":
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
        elif event_type == "NOTE_OFF":
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


def pretty_midi_to_wav(midi: pretty_midi.PrettyMIDI, output_path: str, fs: int = 44100):
    """
    Convert a pretty_midi.PrettyMIDI object to a WAV file.
    """
    audio = midi.fluidsynth(fs)
    sf.write(output_path, audio, fs)


def create_jsonl_file(output_dir: str, job_id: int = 0, total_jobs: int = 1):
    """
    Create a JSONL file from the given data.

    This creates multiple JSONL files, one for every 100k lines.
    They will be saved to the output directory with the same name as
    the output file, but with a number added to the end.
    """

    INSTRUCTION = "Generate MIDI from the given text prompt.\n"

    def create_text_row(input, output):
        text_row = f"""<s>[INST] {INSTRUCTION} {input} [/INST] \\n {output} </s>"""
        return text_row

    from src.download import MidiCaps
    from src.tokenizer import MidiTokenizer

    tokenizer = MidiTokenizer()
    ds = MidiCaps(tokenizer)

    os.makedirs(output_dir, exist_ok=True)

    file_idx = job_id
    jsonl_size = len(ds) // total_jobs
    skip = len(ds) // total_jobs * job_id

    lines = 0
    f = open(f"{output_dir}/part_{file_idx:05d}.jsonl", "w")

    for i in range(skip, len(ds)):
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

        lines += 1
        if lines > jsonl_size:
            f.close()
            file_idx += 1
            f = open(f"{output_dir}_part_{file_idx:05d}.jsonl", "w")
            lines = 0

    f.close()


def fine_tune_model(jsonl_dir: str):
    from src.mistral_config import (
        model_name,
        bnb_4bit_compute_dtype,
        bnb_4bit_quant_type,
        use_4bit,
        use_nested_quant,
    )

    train_dataset = load_dataset(
        "json", data_files=f"{jsonl_dir}/*.jsonl", split="train"
    )

    # Load the base model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map={"": 0}
    )

    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Load MitsralAi tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", action="store_true")

    # The directory to save the JSONL files.
    parser.add_argument("--jsonl-dir", type=str, default="jsonl_data", required=False)

    # This is used as an offset (you can skip lines so we can batch)
    parser.add_argument("--jsonl-job-id", type=int, default=0)
    parser.add_argument("--jsonl-total-jobs", type=int, default=1)

    args = parser.parse_args()

    if args.jsonl:
        create_jsonl_file(
            args.jsonl_dir,
            job_id=args.jsonl_job_id,
            total_jobs=args.jsonl_total_jobs,
        )
