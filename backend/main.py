import sys
import os
import shutil
import uuid
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Path Injection for Sibling Modules ---
# This "flattens" the folder structure so Python can find the detection scripts.
BASE_DIR = Path(__file__).resolve().parent.parent

# Add sibling directories to sys.path
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "Video-detector"))
sys.path.append(str(BASE_DIR / "image_detection"))

# --- 2. Import Detection Modules ---
try:
    # Importing from 'Video-detector/main1.py'
    # Since we added the folder to path, we import 'main1' directly
    from main1 import VideoDeepfakeDetector, LocalEnsembleDetector
    
    # Importing from 'image_detection/image_detector.py'
    from image_detector import LocalImageDetector
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("Ensure 'Video-detector' and 'image_detection' folders exist in the project root.")
    # We allow the app to load so you can see the error, but endpoints will fail
    VideoDeepfakeDetector = None
    LocalEnsembleDetector = None
    LocalImageDetector = None

from blockchain import DigitalNotary

# --- 3. Lifespan State Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    logging.info("Initializing AI Models... This may take several minutes.")
    
    try:
        if LocalEnsembleDetector and VideoDeepfakeDetector:
            # Initialize Video/Image Ensemble (Heavy Load)
            # We reuse the ensemble for video to save memory if possible, 
            # or load specific classes as per your original file structure.
            local_ensemble = LocalEnsembleDetector() 
            app.state.video_model = VideoDeepfakeDetector(local_ensemble)
        
        if LocalImageDetector:
            # Initialize specific Image Detector
            app.state.image_model = LocalImageDetector()
            
        logging.info("AI Models Loaded Successfully.")
    except Exception as e:
        logging.error(f"Failed to load AI models: {e}")
        app.state.video_model = None
        app.state.image_model = None

    try:
        app.state.notary = DigitalNotary()
        logging.info("Blockchain Notary Connected.")
    except Exception as e:
        logging.error(f"Blockchain Connection Failed: {e}")
        app.state.notary = None

    yield
    
    # --- Shutdown Logic ---
    logging.info("Shutting down. Clearing resources.")
    if hasattr(app.state, 'video_model'):
        del app.state.video_model
    if hasattr(app.state, 'image_model'):
        del app.state.image_model
    
    # Optional: Clear GPU cache
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="G-M-Hack Unified Forensic API", lifespan=lifespan)

# --- 4. CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for hackathon/testing
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. Helper Functions ---
def hash_file(file_path: str) -> str:
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to avoid memory overload
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# --- 6. Endpoints ---

@app.post("/verify/video")
async def verify_video_endpoint(file: UploadFile = File(...)):
    if not app.state.video_model:
        raise HTTPException(status_code=503, detail="AI Service Unavailable (Models not loaded)")

    # Create temp directory
    os.makedirs("temp_uploads", exist_ok=True)
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join("temp_uploads", temp_filename)

    try:
        # 1. Save Upload to Disk
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Run Inference
        # analyze_video in main1.py is async, so we await it directly
        result = await app.state.video_model.analyze_video(temp_path)

        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])

        # 3. Blockchain Registration
        tx_hash = None
        explorer_url = None
        if app.state.notary:
            file_hash = hash_file(temp_path)
            verdict = result.get("verdict", "UNKNOWN")
            tx_hash = app.state.notary.register_verification(file_hash, verdict)
            if tx_hash:
                explorer_url = f"https://amoy.polygonscan.com/tx/{tx_hash}"

        return {
            "filename": file.filename,
            "verdict": result.get("verdict"),
            "confidence": result.get("average_fake_probability"),
            "details": result,
            "blockchain_tx": tx_hash,
            "explorer_url": explorer_url
        }

    except Exception as e:
        logging.error(f"Video Verification Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/verify/image")
async def verify_image_endpoint(file: UploadFile = File(...)):
    if not app.state.image_model:
        raise HTTPException(status_code=503, detail="AI Service Unavailable (Models not loaded)")

    os.makedirs("temp_uploads", exist_ok=True)
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join("temp_uploads", temp_filename)

    try:
        # 1. Save Upload
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Run Inference
        # analyze_image in image_detector.py is sync (def analyze_image), 
        # so we run it in a thread to prevent blocking the server.
        result = await asyncio.to_thread(app.state.image_model.analyze_image, temp_path)

        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])

        # 3. Blockchain Registration
        tx_hash = None
        explorer_url = None
        if app.state.notary:
            file_hash = hash_file(temp_path)
            verdict = result.get("verdict", "UNKNOWN")
            tx_hash = app.state.notary.register_verification(file_hash, verdict)
            if tx_hash:
                explorer_url = f"https://amoy.polygonscan.com/tx/{tx_hash}"

        return {
            "filename": file.filename,
            "verdict": result.get("verdict"),
            "confidence": result.get("ensemble_probability"),
            "details": result,
            "blockchain_tx": tx_hash,
            "explorer_url": explorer_url
        }

    except Exception as e:
        logging.error(f"Image Verification Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def home():
    return {
        "message": "Unified Forensic Architecture API is Running",
        "models_loaded": {
            "video": app.state.video_model is not None,
            "image": app.state.image_model is not None
        },
        "blockchain_connected": app.state.notary is not None
    }