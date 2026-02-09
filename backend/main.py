from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import shutil
import os
from dotenv import load_dotenv, find_dotenv
from blockchain import DigitalNotary

# Ensure env vars are loaded at entry point as well
load_dotenv(find_dotenv())

# ---------------------------------------------------------
# IMPORT YOUR AI MODELS HERE
# ---------------------------------------------------------
# Example:
# from detectors.audio_detector import predict_audio
# from detectors.video_detector import predict_video
# from detectors.image_detector import predict_image

app = FastAPI(title="Media Verification & Blockchain Notary")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Blockchain Client
try:
    notary = DigitalNotary()
    print("✅ Blockchain Client Connected")
except Exception as e:
    print(f"⚠️ Blockchain Warning: {e}")
    notary = None

def hash_file(file_content: bytes) -> str:
    """Generate SHA-256 hash of the file content"""
    return hashlib.sha256(file_content).hexdigest()

@app.post("/verify")
async def verify_media(file: UploadFile = File(...)):
    """
    1. Receives file
    2. Hashes file
    3. Runs AI Detection (Mocked here)
    4. Writes result to Blockchain
    """
    try:
        # Read file
        content = await file.read()
        file_hash = hash_file(content)
        filename = file.filename

        # -----------------------------------------------------
        # CALL YOUR AI MODEL HERE
        # -----------------------------------------------------
        # logic:
        # if filename.endswith('.mp3'):
        #     prediction = predict_audio(content)
        # elif filename.endswith('.mp4'):
        #     prediction = predict_video(content)
        
        # MOCK RESULT FOR DEMO:
        ai_result = "REAL"  # Replace this with actual model output
        confidence = 0.98
        
        blockchain_tx = None
        
        # Write to Blockchain if Notary is active
        if notary:
            print(f"🔗 Writing to blockchain... Hash: {file_hash}, Status: {ai_result}")
            blockchain_tx = notary.register_verification(file_hash, ai_result)

        return {
            "filename": filename,
            "verification_result": ai_result,
            "confidence": confidence,
            "digital_signature": file_hash,
            "blockchain_tx": blockchain_tx,  # The "Receipt"
            "explorer_url": f"https://amoy.polygonscan.com/tx/{blockchain_tx}" if blockchain_tx else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Media Verification API is running"}