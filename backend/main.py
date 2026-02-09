import sys
import os
import shutil
import uuid
import logging
import asyncio
import json
import qrcode
import hashlib
from pathlib import Path
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Image & Video Processing
from PIL import Image
import cv2
import numpy as np

# --- 1. Path Injection ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "Video-detector"))
sys.path.append(str(BASE_DIR / "image_detection"))

# --- 2. Dynamic Imports ---
try:
    from main1 import VideoDeepfakeDetector, LocalEnsembleDetector
    from image_detector import LocalImageDetector
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    VideoDeepfakeDetector = None
    LocalEnsembleDetector = None
    LocalImageDetector = None

from blockchain import DigitalNotary

# --- 3. Configuration & State ---
logging.basicConfig(level=logging.INFO)
DB_FILE = "verification_db.json"

class SimpleDB:
    def __init__(self):
        self.db_path = DB_FILE
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({}, f)

    def save_record(self, record_id: str, data: Dict):
        with open(self.db_path, "r") as f:
            try:
                db = json.load(f)
            except json.JSONDecodeError:
                db = {}
        db[record_id] = data
        with open(self.db_path, "w") as f:
            json.dump(db, f, indent=4)

    def get_record(self, record_id: str):
        if not os.path.exists(self.db_path):
            return None
        with open(self.db_path, "r") as f:
            try:
                db = json.load(f)
            except json.JSONDecodeError:
                return None
        return db.get(record_id)

db = SimpleDB()

# --- 4. Helper Services ---
def hash_file(path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in chunks to avoid memory overload
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class WatermarkEngine:
    def generate_qr(self, data: str) -> Image.Image:
        qr = qrcode.QRCode(box_size=10, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        return qr.make_image(fill_color="black", back_color="white").convert('RGB')

    def overlay_image(self, original_path: str, output_path: str, url: str):
        """Overlays QR code on the top-right of an image."""
        try:
            img = Image.open(original_path).convert("RGB")
            qr_img = self.generate_qr(url)
            
            # Resize QR to 15% of image width
            target_width = int(img.width * 0.15)
            # Ensure QR is not too small
            target_width = max(target_width, 100)
            
            ratio = target_width / qr_img.width
            target_height = int(qr_img.height * ratio)
            qr_img = qr_img.resize((target_width, target_height))

            # Position: Top Right with padding
            padding = 20
            pos_x = img.width - target_width - padding
            pos_y = padding

            # Ensure coordinates are valid
            if pos_x < 0: pos_x = 0
            if pos_y < 0: pos_y = 0

            img.paste(qr_img, (pos_x, pos_y))
            img.save(output_path)
            return output_path
        except Exception as e:
            logging.error(f"Watermark Error (Image): {e}")
            return original_path

    def overlay_video(self, original_path: str, output_path: str, url: str):
        """Overlays QR code on video frames using OpenCV."""
        try:
            cap = cv2.VideoCapture(original_path)
            
            # Video Properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Avoid divide by zero
            if width == 0 or height == 0:
                logging.error("Invalid video dimensions")
                return original_path

            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v is widely supported
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Generate QR once
            qr_pil = self.generate_qr(url)
            
            # Resize QR
            target_width = int(width * 0.15)
            target_width = max(target_width, 100) # Min size
            
            ratio = target_width / qr_pil.width
            target_height = int(qr_pil.height * ratio)
            qr_pil = qr_pil.resize((target_width, target_height))
            
            # Convert QR to format suitable for OpenCV overlay
            qr_cv = np.array(qr_pil)
            qr_cv = cv2.cvtColor(qr_cv, cv2.COLOR_RGB2BGR)
            
            # Padding
            padding = 20
            x_offset = width - target_width - padding
            y_offset = padding
            
            if x_offset < 0: x_offset = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Overlay QR
                try:
                    frame[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = qr_cv
                except ValueError:
                    pass # Skip frame if sizing mismatch
                
                out.write(frame)

            cap.release()
            out.release()
            return output_path
        except Exception as e:
            logging.error(f"Watermark Error (Video): {e}")
            return original_path

watermarker = WatermarkEngine()

# --- 5. App & Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Initializing AI Models...")
    try:
        if LocalEnsembleDetector and VideoDeepfakeDetector:
            local_ensemble = LocalEnsembleDetector() 
            app.state.video_model = VideoDeepfakeDetector(local_ensemble)
        if LocalImageDetector:
            app.state.image_model = LocalImageDetector()
        logging.info("AI Models Ready.")
    except Exception as e:
        logging.error(f"AI Load Error: {e}")
        app.state.video_model = None
        app.state.image_model = None

    try:
        app.state.notary = DigitalNotary()
        logging.info("Blockchain Notary Ready.")
    except Exception as e:
        logging.error(f"Blockchain Error: {e}")
        app.state.notary = None

    yield
    
    # Shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="TruthStamp Unified Forensic API", lifespan=lifespan)

# --- CRITICAL FIX: Ensure directories exist BEFORE mounting StaticFiles ---
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/watermarked", exist_ok=True)

# Mount Static Files (to serve images/videos)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Strict for Frontend Integration
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 6. Endpoints ---

# Data Contract for Documentation (Matches Frontend types.ts)
class VerificationResponse(BaseModel):
    hash: str
    confidence: float
    status: str
    timestamp: str
    network: str
    txHash: Optional[str]
    blockNumber: int
    watermarked_file: Optional[str]

@app.post("/verify/image", response_model=VerificationResponse)
async def verify_image_endpoint(request: Request, file: UploadFile = File(...)):
    if not app.state.image_model:
        raise HTTPException(503, "AI Models not loaded")

    # 1. Generate IDs and Paths
    request_id = str(uuid.uuid4())
    filename = f"{request_id}_{file.filename}"
    original_path = f"static/uploads/{filename}"
    watermarked_filename = f"qr_{filename}"
    watermarked_path = f"static/watermarked/{watermarked_filename}"

    # 2. Save Original
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. AI Inference
    result = await asyncio.to_thread(app.state.image_model.analyze_image, original_path)
    if "error" in result: raise HTTPException(500, result["error"])

    # 4. Data Normalization for Frontend
    file_hash = hash_file(original_path)
    
    # Map verdict strings to strict Frontend Enums
    verdict_raw = result.get("verdict", "UNKNOWN").upper()
    status = "FLAGGED_FAKE" if "FAKE" in verdict_raw or "ARTIFICIAL" in verdict_raw else "VERIFIED_REAL"
    
    # Scale confidence to 0-100%
    raw_conf = result.get("ensemble_probability", 0.5)
    confidence = round(raw_conf * 100, 2)
    if status == "VERIFIED_REAL" and raw_conf < 0.5:
        confidence = round((1.0 - raw_conf) * 100, 2)

    # 5. Blockchain
    tx_hash = "0x"
    if app.state.notary:
        tx_hash = app.state.notary.register_verification(file_hash, status) or "0x_failed"

    # 6. Generate Verification URL
    base_url = str(request.base_url)
    verify_url = f"{base_url}verdict/{request_id}"

    # 7. Watermark
    await asyncio.to_thread(watermarker.overlay_image, original_path, watermarked_path, verify_url)

    # 8. Save to DB
    db_record = {
        "id": request_id,
        "filename": filename,
        "verdict": status, # Store normalized status
        "confidence": raw_conf, # Store raw for DB consistency
        "tx_hash": tx_hash,
        "hash": file_hash,
        "original_url": f"/static/uploads/{filename}",
        "watermarked_url": f"/static/watermarked/{watermarked_filename}",
        "timestamp": str(asyncio.get_event_loop().time()),
        "type": "image"
    }
    db.save_record(request_id, db_record)

    # 9. Return Frontend-Compatible Response
    return {
        "hash": file_hash,
        "confidence": confidence,
        "status": status,
        "timestamp": db_record["timestamp"],
        "network": "Polygon Amoy",
        "txHash": tx_hash,
        "blockNumber": 0, # Placeholder until tx confirms
        "watermarked_file": f"{base_url}static/watermarked/{watermarked_filename}",
        # Extra fields preserved but not strictly in interface
        "verification_page": verify_url 
    }

@app.post("/verify/video", response_model=VerificationResponse)
async def verify_video_endpoint(request: Request, file: UploadFile = File(...)):
    if not app.state.video_model:
        raise HTTPException(503, "AI Models not loaded")

    request_id = str(uuid.uuid4())
    filename = f"{request_id}_{file.filename}"
    original_path = f"static/uploads/{filename}"
    watermarked_filename = f"qr_{filename}"
    watermarked_path = f"static/watermarked/{watermarked_filename}"

    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await app.state.video_model.analyze_video(original_path)
    if "error" in result: raise HTTPException(500, result["error"])

    # Normalization
    file_hash = hash_file(original_path)
    verdict_raw = result.get("verdict", "UNKNOWN").upper()
    status = "FLAGGED_FAKE" if "FAKE" in verdict_raw else "VERIFIED_REAL"
    
    raw_conf = result.get("average_fake_probability", 0.5)
    confidence = round(raw_conf * 100, 2)
    if status == "VERIFIED_REAL" and raw_conf < 0.5:
        confidence = round((1.0 - raw_conf) * 100, 2)

    tx_hash = "0x"
    if app.state.notary:
        tx_hash = app.state.notary.register_verification(file_hash, status) or "0x_failed"

    base_url = str(request.base_url)
    verify_url = f"{base_url}verdict/{request_id}"

    # Watermark Video (Heavy Process)
    await asyncio.to_thread(watermarker.overlay_video, original_path, watermarked_path, verify_url)

    db_record = {
        "id": request_id,
        "filename": filename,
        "verdict": status,
        "confidence": raw_conf,
        "tx_hash": tx_hash,
        "hash": file_hash,
        "original_url": f"/static/uploads/{filename}",
        "watermarked_url": f"/static/watermarked/{watermarked_filename}",
        "timestamp": str(asyncio.get_event_loop().time()),
        "type": "video"
    }
    db.save_record(request_id, db_record)

    return {
        "hash": file_hash,
        "confidence": confidence,
        "status": status,
        "timestamp": db_record["timestamp"],
        "network": "Polygon Amoy",
        "txHash": tx_hash,
        "blockNumber": 0,
        "watermarked_file": f"{base_url}static/watermarked/{watermarked_filename}"
    }

# --- 7. Credentials Endpoint (Added for Integration) ---
@app.post("/credentials/claim")
async def submit_claim(
    file: UploadFile = File(...),
    name: str = Form(...),
    org: str = Form(...),
    event: str = Form(...),
    date: str = Form(...),
    category: str = Form(...)
):
    """Handles the async certificate claim flow."""
    claim_id = str(uuid.uuid4())
    # In a full impl, we'd save this to DB. Returning valid placeholder for frontend.
    return {"id": claim_id, "status": "PENDING", "hash": "sha256_placeholder"}

# --- 8. The Public Verification Page ---
@app.get("/verdict/{record_id}", response_class=HTMLResponse)
async def verification_page(request: Request, record_id: str):
    record = db.get_record(record_id)
    if not record:
        return HTMLResponse("<h1>Record not found</h1>", status_code=404)

    # Determine color
    color = "red" if "FAKE" in record['verdict'] else "green"
    # Mapping old 'AMBIGUOUS' just in case
    if "AMBIGUOUS" in record['verdict']: color = "orange"

    # Construct Media Player
    display_url = record['original_url']
    download_link = record.get('watermarked_url', '#')

    if record.get("type") == "video":
        media_html = f'''
        <video controls class="w-full rounded-lg shadow-lg border border-gray-700">
            <source src="{display_url}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        '''
    else:
        media_html = f'<img src="{display_url}" class="w-full rounded-lg shadow-lg border border-gray-700" alt="Analyzed Media">'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TruthStamp Forensic Report</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/@phosphor-icons/web"></script>
    </head>
    <body class="bg-gray-900 text-white min-h-screen flex items-center justify-center p-4">
        <div class="max-w-2xl w-full bg-gray-800 rounded-xl shadow-2xl overflow-hidden border border-gray-700">
            <div class="bg-gray-700 p-6 border-b border-gray-600">
                <h1 class="text-2xl font-bold text-center tracking-wide flex items-center justify-center gap-2">
                    <i class="ph-fill ph-shield-check text-green-400"></i> TruthStamp Report
                </h1>
                <p class="text-gray-400 text-center text-sm mt-1 font-mono">ID: {record_id}</p>
            </div>
            <div class="p-6 space-y-6">
                <div class="flex flex-col items-center">
                    <span class="text-gray-400 uppercase text-xs font-semibold tracking-wider">AI Verdict</span>
                    <h2 class="text-4xl font-extrabold text-{color}-500 mt-2 tracking-tight">{record['verdict']}</h2>
                    <div class="mt-2 px-4 py-1 bg-gray-700/50 rounded-full text-sm border border-gray-600">
                        Confidence: <span class="font-bold text-white">{float(record['confidence'])*100:.1f}%</span>
                    </div>
                </div>
                <div class="space-y-2">
                    <div class="flex justify-between items-end">
                         <p class="text-gray-400 text-xs uppercase font-semibold">Analyzed Content</p>
                         <a href="{download_link}" download class="text-xs flex items-center gap-1 text-blue-400 hover:text-blue-300 transition-colors">
                            <i class="ph-bold ph-download-simple"></i> Download Proof
                         </a>
                    </div>
                    {media_html}
                </div>
                <div class="bg-gray-700/30 p-4 rounded-lg border border-gray-600 hover:bg-gray-700/50 transition-colors">
                    <div class="flex items-center justify-between mb-3">
                        <span class="text-sm font-semibold text-gray-300 flex items-center gap-2">
                            <i class="ph-fill ph-link"></i> Blockchain Notarization
                        </span>
                        <img src="https://cryptologos.cc/logos/polygon-matic-logo.png" class="h-5 w-5 opacity-80" alt="Polygon">
                    </div>
                    <div class="relative group">
                         <div class="break-all text-[10px] font-mono text-blue-300 bg-gray-900/50 p-3 rounded border border-gray-700/50">
                            {record.get('tx_hash', 'Not Notarized')}
                        </div>
                    </div>
                    {f'''
                    <a href="https://amoy.polygonscan.com/tx/{record["tx_hash"]}" target="_blank" 
                       class="mt-3 w-full block text-center py-2 text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors">
                       View Transaction on PolygonScan
                    </a>
                    ''' if record.get('tx_hash') else ''}
                </div>
            </div>
            <div class="bg-gray-900 p-4 text-center">
                <p class="text-[10px] text-gray-500 uppercase tracking-widest">Secured by TruthStamp Unified Forensic Architecture</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)