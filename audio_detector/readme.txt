Unified Forensic Architecture for Synthetic Audio Detection

This project implements a lightweight, serverless deepfake audio detection system using a multi-model ensemble ("Bucket of Models"). It processes audio locally, chunks it, and queries Hugging Face Inference APIs for forensic analysis.

Features

5-Model Ensemble: XLS-R (Waveform), WavLM (Denoising), AST (Spectral), UniSpeech (Semantic), AASIST (Graph).

Consensus Strength Index (CSI): Advanced statistical judging to quantify uncertainty.

Serverless: No local GPU required.

Robust: Handles various audio formats (.wav, .mp3, .flac).

Prerequisites

Python 3.9+

A Hugging Face Account & Access Token (Read permissions)

Installation

Install Dependencies:

pip install -r requirements.txt


Configure Environment:
Create a .env file in the project root and add your Hugging Face Token:

HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx


Note: Without a token, the API will likely rate-limit or reject requests.

Usage

Run the main script with the path to your audio file:

python main.py path/to/audio_sample.wav


How it Works

Preprocessing: Audio is resampled to 16kHz, normalized, and sliced into 4-second overlapping chunks.

Dispatch: Chunks are sent asynchronously to 5 different neural networks hosted on Hugging Face.

Aggregation: The system takes the maximum "Fake" probability per model across all chunks (Max Pooling).

Consensus: The ConsensusEngine calculates the mean, standard deviation, and CSI to generate a final verdict.

Troubleshooting

503 Model Loading: If models are "cold", the script acts as a wake-up call. It may take 20-40 seconds for the first run to complete while HF loads the models into memory.

Inconclusive Results: If fewer than 3 models respond successfully, the system will return an inconclusive verdict to prevent false confidence.