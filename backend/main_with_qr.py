import sys
import os
import shutil
import uuid
import logging
import asyncio
import json
import qrcode
from pathlib import Path
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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

# --- 3. Simple JSON Database ---
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

# --- 4. Watermarking Engine ---
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 6. Endpoints ---

@app.post("/verify/image")
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

    # 4. Blockchain
    tx_hash = None
    if app.state.notary:
        # Simple hash of file path string for uniqueness in this demo context
        # In prod, hash the file content
        file_hash = request_id 
        tx_hash = app.state.notary.register_verification(file_hash, result.get("verdict", "UNKNOWN"))

    # 5. Generate Verification URL
    base_url = str(request.base_url) # e.g., http://localhost:8000/
    verify_url = f"{base_url}verdict/{request_id}"

    # 6. Watermark
    await asyncio.to_thread(watermarker.overlay_image, original_path, watermarked_path, verify_url)

    # 7. Save to DB
    db_record = {
        "id": request_id,
        "filename": filename,
        "verdict": result.get("verdict"),
        "confidence": result.get("ensemble_probability"),
        "tx_hash": tx_hash,
        "original_url": f"/static/uploads/{filename}",
        "watermarked_url": f"/static/watermarked/{watermarked_filename}",
        "timestamp": str(asyncio.get_event_loop().time()),
        "type": "image"
    }
    db.save_record(request_id, db_record)

    return {
        "verdict": result.get("verdict"),
        "confidence": result.get("ensemble_probability"),
        "blockchain_tx": tx_hash,
        "watermarked_file": db_record["watermarked_url"],
        "verification_page": verify_url
    }

@app.post("/verify/video")
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

    tx_hash = None
    if app.state.notary:
        tx_hash = app.state.notary.register_verification(request_id, result.get("verdict", "UNKNOWN"))

    base_url = str(request.base_url)
    verify_url = f"{base_url}verdict/{request_id}"

    # Watermark Video (Might be slow!)
    await asyncio.to_thread(watermarker.overlay_video, original_path, watermarked_path, verify_url)

    db_record = {
        "id": request_id,
        "filename": filename,
        "verdict": result.get("verdict"),
        "confidence": result.get("average_fake_probability"),
        "tx_hash": tx_hash,
        "original_url": f"/static/uploads/{filename}",
        "watermarked_url": f"/static/watermarked/{watermarked_filename}",
        "type": "video"
    }
    db.save_record(request_id, db_record)

    return {
        "verdict": result.get("verdict"),
        "confidence": result.get("average_fake_probability"),
        "watermarked_file": db_record["watermarked_url"],
        "verification_page": verify_url
    }

# --- 7. The Public Verification Page ---
# This section defines the HTML website that users see when scanning the QR code.
# The HTML code is stored in the 'html_content' string.
@app.get("/verdict/{record_id}", response_class=HTMLResponse)
async def verification_page(request: Request, record_id: str):
    record = db.get_record(record_id)
    if not record:
        return HTMLResponse("<h1>Record not found</h1>", status_code=404)

    # Determine color
    color = "red" if "FAKE" in record['verdict'] else "green"
    if "AMBIGUOUS" in record['verdict']: color = "orange"

    media_html = ""
    if record.get("type") == "video":
        media_html = f'''
        <video controls class="w-full rounded-lg shadow-lg border border-gray-700">
            <source src="{record['original_url']}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        '''
    else:
        media_html = f'<img src="{record["original_url"]}" class="w-full rounded-lg shadow-lg border border-gray-700" alt="Analyzed Media">'

    # START OF HTML TEMPLATE (This is Python code storing HTML in a string)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TruthStamp Forensic Report</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white min-h-screen flex items-center justify-center p-4">
        <div class="max-w-2xl w-full bg-gray-800 rounded-xl shadow-2xl overflow-hidden">
            <!-- Header -->
            <div class="bg-gray-700 p-6 border-b border-gray-600">
                <h1 class="text-2xl font-bold text-center tracking-wide">🛡️ TruthStamp Forensic Report</h1>
                <p class="text-gray-400 text-center text-sm mt-1">ID: {record_id}</p>
            </div>

            <!-- Content -->
            <div class="p-6 space-y-6">
                
                <!-- Verdict Section -->
                <div class="flex flex-col items-center">
                    <span class="text-gray-400 uppercase text-xs font-semibold tracking-wider">AI Verdict</span>
                    <h2 class="text-4xl font-extrabold text-{color}-500 mt-2">{record['verdict']}</h2>
                    <div class="mt-2 px-4 py-1 bg-gray-700 rounded-full text-sm">
                        Confidence: <span class="font-bold text-white">{float(record['confidence'])*100:.1f}%</span>
                    </div>
                </div>

                <!-- Media Section -->
                <div>
                    <p class="text-gray-400 text-sm mb-2">Original Content Analyzed:</p>
                    {media_html}
                </div>

                <!-- Blockchain Section -->
                <div class="bg-gray-700/50 p-4 rounded-lg border border-gray-600">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-sm font-semibold text-gray-300">Blockchain Proof</span>
                        <img src="https://cryptologos.cc/logos/polygon-matic-logo.png" class="h-5 w-5" alt="Polygon">
                    </div>
                    <div class="break-all text-xs font-mono text-blue-400 bg-gray-800 p-3 rounded">
                        {record.get('tx_hash', 'Not Notarized')}
                    </div>
                    {f'<a href="https://amoy.polygonscan.com/tx/{record["tx_hash"]}" target="_blank" class="block mt-2 text-center text-xs text-gray-400 hover:text-white underline">View on Explorer</a>' if record.get('tx_hash') else ''}
                </div>
            </div>

            <!-- Footer -->
            <div class="bg-gray-900 p-4 text-center text-xs text-gray-500">
                Generated by TruthStamp Unified Forensic Architecture
            </div>
        </div>
    </body>
    </html>
    """
    # END OF HTML TEMPLATE

    return HTMLResponse(content=html_content)