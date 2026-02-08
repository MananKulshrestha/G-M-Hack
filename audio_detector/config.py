import os
from pathlib import Path
from dotenv import load_dotenv

# Define the path to the .env file in the parent directory
# Path(__file__) is this file (config.py)
# .parent is the directory containing config.py (audio_detector)
# .parent.parent is the directory above that (project root)
env_path = Path(__file__).resolve().parent.parent / '.env'

# Load the environment variables
load_dotenv(dotenv_path=env_path)

HF_TOKEN = os.getenv("HF_TOKEN")

# Global Settings
CHUNK_DURATION = 4.0  # Seconds
CHUNK_OVERLAP = 1.0   # Seconds
SAMPLE_RATE = 16000   # 16kHz mandated by most speech models
TEMP_DIR = "temp_chunks"

# The "Bucket of Models" Registry
# Weights can be adjusted based on empirical trust in the model
MODEL_REGISTRY = {
    "xlsr_waveform": {
        "name": "XLS-R (Raw Waveform)",
        "id": "nii-yamagishilab/xls-r-2b-anti-deepfake", 
        "type": "hf_inference",
        "weight": 1.2, # High weight for the 2B parameter model
        "description": "Multilingual raw waveform analysis (Phase/Time domain)"
    },
    "wavlm_denoising": {
        "name": "WavLM (Noise Robust)",
        "id": "abhishtagatya/wavlm-base-960h-itw-deepfake",
        "type": "hf_inference",
        "weight": 1.0,
        "description": "Denoising specialist for 'In-The-Wild' audio"
    },
    "ast_spectral": {
        "name": "AST (Spectral Vision)",
        "id": "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection",
        "type": "hf_inference",
        "weight": 1.0,
        "description": "Vision Transformer looking for spectral vocoder artifacts"
    },
    "unispeech_semantic": {
        "name": "UniSpeech (Semantic/Speaker)",
        "id": "microsoft/unispeech-sat-base-100h-libri-ft",
        "type": "hf_inference",
        "weight": 0.9,
        "description": "Speaker-aware training to detect prosodic inconsistencies"
    },
    # Note: Gradio Spaces often sleep. Ensure this space is active or replace with a live alternative.
    "aasist_graph": {
        "name": "AASIST (Graph Network)",
        "id": "arnabdas8901/aasist-trained-asvspoof2024", 
        "type": "gradio",
        "weight": 1.1,
        "description": "Graph Attention Network checking spectro-temporal structural integrity"
    }
}