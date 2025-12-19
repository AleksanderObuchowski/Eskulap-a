import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import subprocess
import tempfile
from datasets import load_from_disk

def play_audio(audio_data, sampling_rate):
    """
    Saves audio array to a temp wav file and plays it using aplay.
    """
    try:
        # Create a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_filename = tf.name
            
        # Write wav
        # Ensure data is in the right format for wavfile.write
        # arrays in datasets are often float32 or int16
        data = audio_data
        if data.dtype == np.float32:
             # Convert to int16 for compatibility if needed, or just write float
             # wavfile.write handles float32
             pass
        
        wav.write(temp_filename, sampling_rate, data)
        
        # Play
        # Use aplay (ALSA) which is common on Linux
        print(f"Playing audio ({len(data)/sampling_rate:.2f}s)...")
        subprocess.run(["aplay", temp_filename], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    except FileNotFoundError:
        print("Error: 'aplay' not found. Please ensure you have ALSA utilities installed.")
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def find_text_column(features):
    candidates = ["sentence", "text", "transcription", "transcript"]
    for c in candidates:
        if c in features:
            return c
    # Fallback: look for string features
    for k, v in features.items():
        if v.dtype == 'string':
            return k
    return None

def main():
    print("Loading dataset 'prepared_data'...")
    try:
        dataset_dict = load_from_disk("prepared_data")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Prefer 'train' split if available, else first available
    split_name = "train"
    if split_name not in dataset_dict:
        split_name = list(dataset_dict.keys())[0]
    
    ds = dataset_dict[split_name]
    print(f"Loaded split '{split_name}' with {len(ds)} examples.")
    
    text_col = find_text_column(ds.features)
    if not text_col:
        print("Warning: Could not identify a text/transcript column.")
    else:
        print(f"Using '{text_col}' as transcript column.")

    current_idx = 0
    total = len(ds)
    
    # For search
    active_indices = list(range(total))
    search_query = None

    while True:
        if not active_indices:
            print("No matching records found.")
            search_query = None
            active_indices = list(range(total))
            current_idx = 0
            continue

        # Ensure index is within bounds of active_indices
        if current_idx >= len(active_indices):
            current_idx = len(active_indices) - 1
        
        real_idx = active_indices[current_idx]
        item = ds[real_idx]
        
        # Clear screen (optional, maybe just print separator)
        print("\n" + "="*60)
        print(f"Index: {current_idx + 1}/{len(active_indices)} (Original ID: {real_idx})")
        if search_query:
            print(f"Search Filter: '{search_query}'")
        
        # Display Text
        transcript = item.get(text_col, "<No Text>") if text_col else "<No Text>"
        print(f"Transcript: {transcript}")
        
        # Audio Info
        audio_info = item.get("path") # 'path' is the audio column name in prepare_data.py
        
        audio_array = None
        sampling_rate = None

        # Handle torchcodec AudioDecoder
        if hasattr(audio_info, "get_all_samples"):
             try:
                 samples = audio_info.get_all_samples()
                 audio_array = samples.data.numpy() if hasattr(samples.data, "numpy") else samples.data
                 # If stereo (C, T), convert to (T,) or (T, C) ?
                 # wavfile.write expects (T, C) or (T,).
                 # torchcodec usually returns (C, T).
                 if len(audio_array.shape) > 1 and audio_array.shape[0] < audio_array.shape[1]:
                     audio_array = audio_array.T
                 sampling_rate = samples.sample_rate
             except Exception as e:
                 print(f"Error decoding audio: {e}")
        # Handle standard dict
        elif isinstance(audio_info, dict) and "array" in audio_info:
            audio_array = audio_info["array"]
            sampling_rate = audio_info["sampling_rate"]
        
        if audio_array is not None and sampling_rate:
            duration = len(audio_array) / sampling_rate
            print(f"Duration: {duration:.2f}s | Sample Rate: {sampling_rate}Hz")
        else:
            print("Audio: <Missing or Invalid>")

        print("-" * 60)
        print("[N]ext | [P]rev | [J]ump | [S]earch | [C]lear Search | [Pl]ay | [E]xport | [Q]uit")
        
        cmd = input("Command: ").strip().lower()
        
        if cmd == 'n':
            if current_idx < len(active_indices) - 1:
                current_idx += 1
            else:
                print("End of list.")
        elif cmd == 'p':
            if current_idx > 0:
                current_idx -= 1
            else:
                print("Start of list.")
        elif cmd == 'j':
            try:
                val = int(input(f"Jump to index (1-{len(active_indices)}): "))
                if 1 <= val <= len(active_indices):
                    current_idx = val - 1
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid input.")
        elif cmd == 's':
            query = input("Enter search text: ").strip()
            if query:
                print("Searching...")
                # Simple linear search
                new_indices = []
                for i in range(total):
                    # We access raw dataset to avoid loading audio for search if possible
                    # But huggingface datasets are lazy, accessing the row loads it.
                    # To be faster, we might want to only decode the text column?
                    # ds[i] loads everything. 
                    # Optimization: ds.select_columns(text_col)[i] might be faster?
                    # Let's just do simple loop for now, might be slow for huge datasets.
                    txt = ds[i].get(text_col, "")
                    if query.lower() in txt.lower():
                        new_indices.append(i)
                
                if new_indices:
                    active_indices = new_indices
                    current_idx = 0
                    search_query = query
                    print(f"Found {len(active_indices)} matches.")
                else:
                    print("No matches found.")
        elif cmd == 'c':
            active_indices = list(range(total))
            search_query = None
            current_idx = 0
            print("Search cleared.")
        elif cmd == 'pl':
            if audio_array is not None:
                play_audio(audio_array, sampling_rate)
            else:
                print("No audio data to play.")
        elif cmd == 'e':
            if audio_array is not None:
                filename = f"audio_{real_idx}.wav"
                try:
                    wav.write(filename, sampling_rate, audio_array)
                    print(f"Successfully saved audio to: {os.path.abspath(filename)}")
                    print("You can now download this file to play it locally.")
                except Exception as e:
                    print(f"Error saving file: {e}")
            else:
                print("No audio data to save.")
        elif cmd == 'q':
            break
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()
