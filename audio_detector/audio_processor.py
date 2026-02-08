import librosa
import soundfile as sf
import numpy as np
import os
import shutil
from config import SAMPLE_RATE, TEMP_DIR

class AudioProcessor:
    def __init__(self):
        # Create temp directory if it doesn't exist
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

    def clean_temp(self):
        """Removes the temporary chunk directory."""
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)

    def process_file(self, file_path, chunk_duration=4.0, overlap=1.0):
        """
        Loads audio, resamples to 16kHz, normalizes, and slices into chunks.
        Returns a list of paths to the saved chunk files.
        """
        print(f"[*] Processing audio: {file_path}")
        
        try:
            # 1. Load and Resample
            # librosa.load automatically resamples if sr is provided
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            
            # 2. Peak Normalization (-3dB target usually, here just 0-1 float norm)
            # This ensures volume levels don't bias the model
            norm_y = librosa.util.normalize(y)
            
            # 3. Chunking Logic
            chunk_samples = int(chunk_duration * sr)
            hop_length = int((chunk_duration - overlap) * sr)
            
            chunk_paths = []
            
            # If audio is shorter than one chunk, pad it
            if len(norm_y) < chunk_samples:
                norm_y = np.pad(norm_y, (0, chunk_samples - len(norm_y)))
            
            # Sliding window
            idx = 0
            for start in range(0, len(norm_y), hop_length):
                end = start + chunk_samples
                
                # Handle the last chunk
                if end > len(norm_y):
                    # Option A: Drop last short chunk
                    # break 
                    # Option B: Pad last chunk (Selected)
                    segment = norm_y[start:]
                    segment = np.pad(segment, (0, chunk_samples - len(segment)))
                else:
                    segment = norm_y[start:end]
                
                # Save chunk
                chunk_filename = f"{TEMP_DIR}/chunk_{idx}.wav"
                sf.write(chunk_filename, segment, sr)
                chunk_paths.append(chunk_filename)
                idx += 1
                
            print(f"[*] Generated {len(chunk_paths)} chunks.")
            return chunk_paths

        except Exception as e:
            print(f"[!] Error processing audio: {str(e)}")
            return []