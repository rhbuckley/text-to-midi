import torch
from midiutil import MIDIFile
from numpy import random
from transformers.models.gpt2 import GPT2Tokenizer
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.midi_utils import parse_midi_pretty

# ================================================
# Device configuration.
# ================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


# ================================================
# There are many tokens that we need to use within our
# sequence to sequence tokenizer. Let's define them here.
# ================================================
# SOS_TOKEN       = "<SOS>"       # we don't need a sos token for the gpt2 model, 
                                  # because it is a decoder only model
NOTE_TOKEN      = "<NOTE>"      # note of the sequence
NOTE_SEP_TOKEN  = "<NOTE_SEP>"  # note separator
PITCH_TOKEN     = "<PITCH>"     # pitch of the note
VELOCITY_TOKEN  = "<VELOCITY>"  # velocity of the note
TIME_TOKEN      = "<TIME>"      # time of the note
DURATION_TOKEN  = "<DURATION>"  # duration of the note
REST_TOKEN      = "<REST>"      # rest of the note
INSTRUMENT_TOKEN = "<INSTRUMENT>" # instrument of the note


class MidiTokenizer:
    def __init__(self, max_length=100):
        """
        Initialize the MidiTokenizer with a pre-trained GPT-2 tokenizer.
        This tokenizer will be used to convert between MIDI sequences and token sequences.
        """
        self.tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.tokenizer.add_special_tokens(
            special_tokens_dict={
                "additional_special_tokens": [
                    NOTE_TOKEN,
                    NOTE_SEP_TOKEN,
                    PITCH_TOKEN,
                    VELOCITY_TOKEN,
                    TIME_TOKEN,
                    DURATION_TOKEN,
                    REST_TOKEN,
                    INSTRUMENT_TOKEN
                ] # type: ignore
            }
        )

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, text, midi_seq = None, max_length=None):
        """
        Convert a MIDI sequence into a sequence of tokens.
        
        Args:
            midi_seq: A sequence of MIDI events containing pitch, velocity, time, and duration
            
        Returns:
            A list of tokens representing the MIDI sequence
        """
        if midi_seq is None:
            input_text = text + " " + NOTE_TOKEN
        else:
            midi = self._midi_to_tokens(midi_seq)
            input_text = text + " " + NOTE_TOKEN + " " + midi + " " + self.tokenizer.eos_token
        
        # If max_length is provided, use it, otherwise use the default max_length
        max_len = max_length if max_length is not None else self.max_length

        return self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=max_len, 
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )

    def detokenize(self, tokens, debug=False, return_strings=False):
        """
        Convert a sequence of tokens back into a MIDI sequence.
        
        Args:
            tokens: A sequence of tokens representing a MIDI sequence
            
        Returns:
            A list of MIDI events containing pitch, velocity, time, and duration
        """
        token_strings = self.tokenizer.convert_ids_to_tokens(tokens)
        
        if debug:
            print(token_strings)
            
        if return_strings:
            return token_strings

        midi_sequence = self._tokens_to_midi(token_strings)
        return midi_sequence
    
    def tokenize_from_file(self, text, midi_file):
        """
        Read a MIDI file, parse it into a note sequence with durations,
        and convert it into tokens suitable for the model.

        Args:
            text (str): Optional text to prepend to the token sequence.
            midi_file (str): Path to the MIDI file to be tokenized.

        Returns:
            (depends on your tokenize method's return type, likely PyTorch tensors
             if the previous correction was applied): Tokenized representation.
             Returns None or raises error if parsing/tokenization fails.
        """
        # Parse the MIDI file to get the structured sequence with durations
        midi_seq = parse_midi_pretty(midi_file)

        if not midi_seq:
            print(f"Warning: Could not parse MIDI data from {midi_file} or it contained no notes.")
            # Decide how to handle: return None, empty tokens, or raise error
            # Example: Return tokenized text only
            # input_text = text # Or maybe text + " <NOTE> <EOS>" to indicate empty MIDI?
            # return self.tokenizer(input_text, return_tensors="pt")
            return None # Or handle as appropriate for your application

        # Pass the parsed sequence to the main tokenize method
        # Ensure the main tokenize method handles the midi_seq correctly
        return self.tokenize(text, midi_seq=midi_seq)
    
    def detokenize_to_file(self, tokens, midi_file):
        """
        Convert a sequence of tokens into a MIDI file.
        
        Args:
            tokens: A sequence of tokens representing a MIDI sequence
            midi_file: Path where the MIDI file should be saved
            
        Returns:
            The path to the created MIDI file
        """
        if hasattr(tokens, 'tolist'): tokens = tokens.tolist() 
        token_strings = self.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=False)
        sequence = self._tokens_to_midi(token_strings)

        midi = MIDIFile(numTracks=1)

        track = 0
        time = 0

        midi.addTrackName(track, time, "Generated Track")
        midi.addTempo(track, time, 120)

        for event in sequence:
            tempo = 120 # The tempo set earlier
            beats_per_second = tempo / 60.0
            time_in_beats = event['time'] * beats_per_second
            duration_in_beats = event['duration'] * beats_per_second
            midi.addNote(
                track=track,
                channel=0,
                pitch=event['pitch'],
                time=time_in_beats,
                duration=duration_in_beats,
                volume=event['velocity']
            )

        with open(midi_file, "wb") as f:
            midi.writeFile(f)
            
        return midi_file
    
    def _extract_between(self, tokens, start, end):
        """
        Extract tokens between two specified tokens.
        
        Args:
            tokens: List of tokens to search through
            start: The starting token to search for
            end: The ending token to search for
            
        Returns:
            A list of tokens found between the start and end tokens
        """
        i = tokens.index(start)
        j = tokens.index(end, i)
        return tokens[i+1:j]
    
    def _extract_from_token(self, tokens, token):
        """
        Extract the token that comes immediately after a specified token.
        
        Args:
            tokens: List of tokens to search through
            token: The token to search for
            
        Returns:
            The token that follows the specified token
        """
        i = tokens.index(token)
        return tokens[i+1]
    
    def _midi_to_tokens(self, midi_seq):
        """
        Convert a MIDI sequence into a string of tokens.
        
        Args:
            midi_seq: A sequence of MIDI events containing pitch, velocity, time, and duration
            
        Returns:
            A string of tokens representing the MIDI sequence, with special tokens marking
            the start and end of the sequence and separating individual notes
        """
        tokens = []
        for note in midi_seq:
            tokens.append(PITCH_TOKEN)
            tokens.append(note['pitch'])
            
            tokens.append(VELOCITY_TOKEN)
            tokens.append(note['velocity'])

            tokens.append(TIME_TOKEN)
            tokens.append(note['time'])

            tokens.append(DURATION_TOKEN)
            tokens.append(note['duration'])

            # mark the end of the note
            tokens.append(NOTE_SEP_TOKEN)

        return " ".join(map(str, tokens))
    
    def _tokens_to_midi(self, token_strings):
        events = []

        # Find the start of the MIDI section if needed, e.g., find the first <NOTE>
        try:
            # Find the first actual MIDI marker to start processing
            # Adjust based on the expected overall structure from the model output
            first_midi_token_index = -1
            midi_markers = [NOTE_TOKEN, PITCH_TOKEN, VELOCITY_TOKEN, TIME_TOKEN, DURATION_TOKEN, REST_TOKEN, INSTRUMENT_TOKEN]
            for i, token in enumerate(token_strings):
                if token in midi_markers:
                    first_midi_token_index = i
                    break
            if first_midi_token_index == -1:
                return [] # No MIDI found

            token_strings = token_strings[first_midi_token_index:]

        except ValueError:
            print("Warning: Could not find start of MIDI sequence.")
            return []

        note_data = {}
        expecting_value_for = None # Stores the type (e.g., PITCH_TOKEN) expecting a value

        for token in token_strings:
            if token in [PITCH_TOKEN, VELOCITY_TOKEN, TIME_TOKEN, DURATION_TOKEN]:
                expecting_value_for = token
            elif expecting_value_for:
                # Found the value for the preceding marker
                try:
                    if expecting_value_for == PITCH_TOKEN:
                        note_data['pitch'] = int(token)
                    elif expecting_value_for == VELOCITY_TOKEN:
                        note_data['velocity'] = int(token)
                    elif expecting_value_for == TIME_TOKEN:
                        note_data['time'] = float(token)
                    elif expecting_value_for == DURATION_TOKEN:
                        note_data['duration'] = float(token)
                except ValueError:
                    print(f"Warning: Could not convert token '{token}' for {expecting_value_for}. Skipping value.")
                    # Decide how to handle errors - skip note? Use default?
                expecting_value_for = None # Reset after consuming value
            elif token == NOTE_SEP_TOKEN:
                # End of a note definition, add if valid
                if all(k in note_data for k in ['pitch', 'velocity', 'time', 'duration']):
                    events.append(note_data)
                else:
                    print(f"Warning: Incomplete note data skipped: {note_data}")
                note_data = {} # Reset for next note
                expecting_value_for = None
            elif token in [self.tokenizer.eos_token]:
                # Stop processing if EOS/END encountered
                break

        # Add the last note if sequence ends without a final NOTE_SEP
        if all(k in note_data for k in ['pitch', 'velocity', 'time', 'duration']):
            events.append(note_data)

        return events


    def _generate_random_midi(self, n_notes=20):
        """
        Generate a random MIDI sequence for testing or fallback purposes.
        
        Args:
            n_notes: Number of notes to generate (default: 20)
            
        Returns:
            A list of randomly generated MIDI events, where each event contains
            pitch, velocity, time, and duration information. The pitches are based
            on a predefined set of base notes, with some variation.
        """
        events = []
        base_notes = [72, 74, 76, 79, 81, 84]

        for i in range(n_notes):
            events.append({
                "pitch": random.choice(base_notes) if i % 2 == 0 else random.choice(base_notes) - 12,
                "velocity": random.randint(70, 110),
                "time": i * 0.25 + random.uniform(-0.05, 0.05) if i > 0 else 0,
                "duration": random.uniform(0.1, 0.3)
            })
        
        return events
    
    def save_pretrained(self, output_dir):
        self.tokenizer.save_pretrained(output_dir)

    def load_pretrained(self, input_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return len(self.tokenizer)
