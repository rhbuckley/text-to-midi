# https://github.com/Addy771/MIDI-to-MP3/blob/master/midi.py
"""
File: midi.py
Author: Addy771
Description: 
A script which converts MIDI files to WAV and optionally to MP3 using ffmpeg. 
Works by playing each file and using the stereo mix device to record at the same time
"""


import pyaudio  # audio recording
import wave     # file saving
import pygame   # midi playback
import fnmatch  # name matching
import os       # file listing


def midi_to_wav(midi_file):
    """
    Convert a MIDI file to a WAV file.
    """
    convert_midi_files(single_file=midi_file)


def convert_midi_files(
    do_ffmpeg_convert=True,    # Uses FFmpeg to convert WAV files to MP3
    do_wav_cleanup=True,       # Deletes WAV files after conversion to MP3
    sample_rate=44100,         # Sample rate used for WAV/MP3
    channels=2,                # Audio channels (1 = mono, 2 = stereo)
    buffer=1024,               # Audio buffer size
    mp3_bitrate=128,           # Bitrate to save MP3 with in kbps (CBR)
    input_device=1,            # Which recording device to use
    search_dir="./",           # Directory to search for MIDI files
    single_file=None           # Optional single MIDI file to convert
):
    """
    Convert MIDI files to WAV and optionally to MP3 using ffmpeg.
    
    Args:
        do_ffmpeg_convert (bool): Whether to convert WAV to MP3
        do_wav_cleanup (bool): Whether to delete WAV files after MP3 conversion
        sample_rate (int): Sample rate for WAV/MP3
        channels (int): Number of audio channels
        buffer (int): Audio buffer size
        mp3_bitrate (int): MP3 bitrate in kbps
        input_device (int): Recording device index
        search_dir (str): Directory to search for MIDI files
        single_file (str): Optional path to a single MIDI file to convert
    """
    # Begins playback of a MIDI file
    def play_music(music_file):
        try:
            pygame.mixer.music.load(music_file)
        except pygame.error:
            print("Couldn't play %s! (%s)" % (music_file, pygame.get_error()))
            return
        pygame.mixer.music.play()

    # Init pygame playback
    bitsize = -16   # unsigned 16 bit
    pygame.mixer.init(sample_rate, bitsize, channels, buffer)
    # Set volume to 0 to make playback silent
    pygame.mixer.music.set_volume(0.0)

    # Init pyAudio
    format = pyaudio.paInt16
    audio = pyaudio.PyAudio()

    try:
        # Get list of files to process
        matches = []
        if single_file:
            if os.path.exists(single_file) and single_file.lower().endswith('.mid'):
                matches.append(single_file)
            else:
                print(f"Error: File {single_file} not found or not a MIDI file")
                return
        else:
            for root, dirnames, filenames in os.walk(search_dir):
                for filename in fnmatch.filter(filenames, '*.mid'):
                    matches.append(os.path.join(root, filename))
                
        # Play each song in the list
        for song in matches:
            # Create a filename with a .wav extension
            file_name = os.path.splitext(os.path.basename(song))[0]
            new_file = file_name + '.wav'

            # Open the stream and start recording
            stream = audio.open(format=format, channels=channels, rate=sample_rate, 
                              input=True, input_device_index=input_device, 
                              frames_per_buffer=buffer)
            
            # Playback the song
            print("Playing " + file_name + ".mid\n")
            play_music(song)
            
            frames = []
            
            # Record frames while the song is playing
            while pygame.mixer.music.get_busy():
                frames.append(stream.read(buffer))
                
            # Stop recording
            stream.stop_stream()
            stream.close()

            # Configure wave file settings
            wave_file = wave.open(new_file, 'wb')
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(audio.get_sample_size(format))
            wave_file.setframerate(sample_rate)
            
            print("Saving " + new_file)   
            
            # Write the frames to the wave file
            wave_file.writeframes(b''.join(frames))
            wave_file.close()
            
            # Call FFmpeg to handle the MP3 conversion if desired
            if do_ffmpeg_convert:
                os.system('ffmpeg -i ' + new_file + ' -y -f mp3 -ab ' + str(mp3_bitrate) + 
                         'k -ac ' + str(channels) + ' -ar ' + str(sample_rate) + 
                         ' -vn ' + file_name + '.mp3')
                
                # Delete the WAV file if desired
                if do_wav_cleanup:        
                    os.remove(new_file)
        
        # End PyAudio    
        audio.terminate()    
 
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

if __name__ == "__main__":
    # Example usage with default parameters
    convert_midi_files()
    
    # Example usage with a single file
    # convert_midi_files(single_file="path/to/your/file.mid")