#!/usr/bin/env python
# coding: utf-8

# In[1]:

# ensure that you login to huggingface using `huggingface-cli login` command
# or in a notebook do:
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(os.environ["HF_TOKEN"])  # or: # notebook_login()

import os
import torch
import tarfile
from huggingface_hub import hf_hub_download
from datasets import load_dataset, DatasetDict

# some cool links:
# https://huggingface.co/dx2102/llama-midi

class MidiCaps(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        ds = load_dataset("amaai-lab/MidiCaps")
        assert type(ds) == DatasetDict

        self.ds = ds
        self.tokenizer = tokenizer
        self.midi_root_dir = "lmd_full"

        # now we need to download the corresponding midi files
        # 1, download the tar file from huggingface
        if not os.path.exists("midicaps.tar.gz"):
            hf_hub_download(
                repo_id="amaai-lab/MidiCaps",
                repo_type="dataset",
                filename="midicaps.tar.gz",
                local_dir=".",
            )

        # 2, extract the tar file
        if not os.path.exists(self.midi_root_dir):
            with tarfile.open("midicaps.tar.gz", "r:gz") as tar:
                tar.extractall()
        
    def __len__(self):
        # note that the dataset is automatically placed into train,
        # however, there are no splits for test / validation sets.
        # as such, we will declare the whole dataset as the training set.
        return len(self.ds["train"])
    
    def __getitem__(self, idx):
        item = self.ds["train"][idx]

        # item has the following properties:
        # - "location": the file path indicating where the MIDI file is stored.
        # - "caption": a descriptive text summarizing the musical characteristics and mood of the piece.
        # - "genre": a list of musical genres that the piece is classified under.
        # - "genre_prob": a list of probabilities corresponding to each genre in the "genre" list, indicating the confidence of the classification.
        # - "mood": a list of moods or emotional qualities evoked by the music.
        # - "mood_prob": a list of probabilities associated with each mood in the "mood" list, reflecting the strength of that emotional characteristic.
        # - "key": the musical key in which the composition is primarily written.
        # - "time_signature": the rhythmic framework of the music, indicating the number of beats per measure and the type of note that receives one beat.
        # - "tempo": the speed of the music, measured in beats per minute (BPM).
        # - "tempo_word": a descriptive word indicating the tempo range (e.g., Allegro).
        # - "duration": the length of the musical piece in seconds.
        # - "duration_word": a qualitative description of the song's length (e.g., Short song).
        # - "chord_summary": a list of the most frequently occurring chords in the piece.
        # - "chord_summary_occurence": the number of times the chords in "chord_summary" appear in the "all_chords" list.
        # - "instrument_summary": a list of the prominent musical instruments featured in the composition.
        # - "instrument_numbers_sorted": a list of MIDI program numbers corresponding to the instruments in the piece, sorted numerically. The inclusion of 128 likely refers to a drum kit (percussion).
        # - "all_chords": a complete list of all the chords detected in the musical piece, in the order they appear.
        # - "all_chords_timestamps": a list of timestamps (in seconds) indicating when each chord in the "all_chords" list occurs.
        # - "test_set": a boolean value indicating whether this piece was part of a test dataset.
        
        caption = item["caption"]
        relative_midi_path = item["location"] # e.g., "0/0a2f96915e8f47633a30f6aec3a46d3f.mid"

        # Construct the full path to the MIDI file
        # Assumes 'lmd_full' directory exists in self.base_dir
        full_midi_path = relative_midi_path

        # Optional: Check if the specific MIDI file exists, handle if not
        # if not os.path.exists(full_midi_path):
        #     print(f"Warning: Specific MIDI file not found for index {idx} at {full_midi_path}")
        #     # Return None or skip, requires handling in training loop or dataloader collate_fn
        #     return None, None # Example: return None if file missing

        # Return the caption string and the full PATH string to the MIDI file
        return caption, full_midi_path


if __name__ == "__main__":
    print("Downloading dataset...")
    midi_caps = MidiCaps(None)
    print("Dataset downloaded successfully.")