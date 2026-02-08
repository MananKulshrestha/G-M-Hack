import asyncio
import os
import math
import logging
import argparse
import random
import cv2
import shutil
import tempfile
import torch
import warnings
from typing import Dict, List, Any
from transformers import pipeline
from termcolor import colored
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
ENSEMBLE_CONFIG = [
    {
        "id": "dima806/deepfake_vs_real_image_detection",
        "name": "Semantic Baseline (ViT)",
        "weight": 0.10,
        "target_label": "fake",
        "label_mapping": {"label_0": "fake", "label_1": "real"}
    },
    {
        "id": "Organika/sdxl-detector",
        "name": "Diffusion Specialist (Swin)",
        "weight": 0.50,
        "target_label": "artificial",
        # Inverted mapping logic handles inside the analyzer
        "label_mapping": {
            "label_1": "real", 
            "label_0": "fake", 
            "artificial": "real", 
            "human": "fake"
        }
    },
    {
        "id": "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "name": "Generalist Forensics", 
        "weight": 0.15,
        "target_label": "Deepfake",
        "label_mapping": {"deepfake": "fake", "realism": "real"}
    },
    {
        "id": "dima806/ai_vs_real_image_detection",
        "name": "Broad AI/Real Classifier",
        "weight": 0.15,
        "target_label": "AI",
        "label_mapping": {"ai": "fake", "real": "real"}
    },
    {
        "id": "umm-maybe/AI-image-detector",
        "name": "Artistic/Style Analyst",
        "weight": 0.10,
        "target_label": "artificial",
        "label_mapping": {"artificial": "fake", "human": "real"}
    }
]

class LocalEnsembleDetector:
    def __init__(self):
        print(colored("Loading models locally... (This may take a while first time)", "cyan"))
        self.pipelines = {}
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Inference Device: {'GPU' if self.device == 0 else 'CPU'}")

        for config in ENSEMBLE_CONFIG:
            try:
                print(f"  - Loading {config['name']}...")
                # Initialize the pipeline
                pipe = pipeline("image-classification", model=config['id'], device=self.device)
                self.pipelines[config['id']] = pipe
            except Exception as e:
                print(colored(f"  ❌ Failed to load {config['name']}: {e}", "red"))

        if not self.pipelines:
            raise RuntimeError("No models could be loaded. Check your internet connection.")

    def _normalize_prediction(self, predictions, config) -> float:
        """Converts model output to a 'fake_probability' float."""
        fake_prob = 0.5
        
        try:
            # Predictions is a list like [{'label': 'real', 'score': 0.99}, ...]
            found_fake = False
            
            for item in predictions:
                label_lower = str(item['label']).lower()
                score = float(item['score'])
                
                # Check Mapping
                if config['label_mapping']:
                    mapped = config['label_mapping'].get(label_lower)
                    if mapped == 'fake':
                        fake_prob = score
                        found_fake = True
                        break
                    elif mapped == 'real':
                        fake_prob = 1.0 - score
                        found_fake = True
                        break
            
            # Fallback if mapping failed
            if not found_fake:
                target = config['target_label'].lower()
                for item in predictions:
                    if str(item['label']).lower() == target:
                        fake_prob = float(item['score'])
                        break

        except Exception as e:
            logger.warning(f"Norm Error: {e}")
            
        return fake_prob

    def calculate_entropy(self, probability: float) -> float:
        if probability <= 0 or probability >= 1:
            return 0.0
        return - (probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))

    def _analyze_image_sync(self, image_path: str) -> Dict:
        """Synchronous version of analysis to be run in a thread."""
        try:
            pil_image = Image.open(image_path)
        except Exception as e:
            return {"error": f"Image load error: {e}"}

        results = []
        total_weight = 0.0
        weighted_sum = 0.0
        probs = []
        
        for config in ENSEMBLE_CONFIG:
            model_id = config['id']
            if model_id not in self.pipelines:
                continue
                
            try:
                pipe = self.pipelines[model_id]
                # Run Inference
                output = pipe(pil_image)
                
                # Normalize
                fake_prob = self._normalize_prediction(output, config)
                
                # Apply Inversion for Swin model
                if "Swin" in config["name"]:
                    fake_prob = 1.0 - fake_prob

                # Add to ensemble
                weight = config['weight']
                weighted_sum += weight * fake_prob
                total_weight += weight
                probs.append(fake_prob)
                
                results.append({
                    "model_name": config["name"],
                    "fake_prob": fake_prob,
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "model_name": config["name"],
                    "error": str(e),
                    "status": "failed"
                })

        if total_weight == 0:
            return {"ensemble_probability": 0.5, "entropy": 0.0, "variance": 0.0, "model_breakdown": results}

        ensemble_prob = weighted_sum / total_weight
        variance = sum([((p - ensemble_prob) ** 2) for p in probs]) / len(probs) if probs else 0
        entropy = self.calculate_entropy(ensemble_prob)

        return {
            "ensemble_probability": ensemble_prob,
            "entropy": entropy,
            "variance": variance,
            "model_breakdown": results
        }

    async def analyze_image(self, image_path: str) -> Dict:
        """Async wrapper for the synchronous local analysis."""
        # We run the blocking inference in a separate thread to keep asyncio event loop responsive
        return await asyncio.to_thread(self._analyze_image_sync, image_path)

class VideoDeepfakeDetector:
    def __init__(self, detector: LocalEnsembleDetector):
        self.image_detector = detector

    def extract_random_frames(self, video_path: str, num_frames: int = 20, temp_dir: str = "temp_frames") -> List[str]:
        """Extracts N random frames from a video and saves them to a temp directory."""
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.error("Could not determine frame count.")
            return []

        # Generate random indices (sorted to seek forward efficiently)
        num_to_extract = min(num_frames, total_frames)
        frame_indices = sorted(random.sample(range(total_frames), num_to_extract))
        
        frame_paths = []
        
        print(colored(f"Extracting {num_to_extract} frames from {total_frames} total frames...", "cyan"))

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(temp_dir, f"frame_{idx}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            else:
                logger.warning(f"Failed to read frame at index {idx}")

        cap.release()
        return frame_paths

    async def analyze_video(self, video_path: str, num_frames: int = 20):
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Extract Frames
            frame_paths = self.extract_random_frames(video_path, num_frames, temp_dir)
            
            if not frame_paths:
                return {"error": "No frames extracted"}

            print(colored(f"Analyzing {len(frame_paths)} frames...", "blue"))
            
            # 2. Analyze Frames (Sequentially for local inference to avoid OOM)
            # We use a Semaphore of 1 to ensure one frame is processed at a time on GPU/CPU
            semaphore = asyncio.Semaphore(1) 
            
            async def analyze_with_limit(path):
                async with semaphore:
                    return await self.image_detector.analyze_image(path)

            tasks = [analyze_with_limit(fp) for fp in frame_paths]
            
            # Show progress bar effect
            results = []
            for f in asyncio.as_completed(tasks):
                res = await f
                results.append(res)
                print(".", end="", flush=True)
            print("\n")

            # 3. Aggregation & Detailed Reporting
            print(colored("--- DETAILED FRAME ANALYSIS ---", "yellow"))
            
            total_fake_prob = 0.0
            total_entropy = 0.0
            valid_frames = 0
            frames_fake_count = 0
            
            frame_details = []

            for i, res in enumerate(results):
                # Print Frame Header
                print(colored(f"\n[Frame Processed #{i+1}]", "blue"))
                
                if "error" in res:
                    print(colored(f"  Error: {res['error']}", "red"))
                    continue

                prob = res["ensemble_probability"]
                total_fake_prob += prob
                total_entropy += res["entropy"]
                valid_frames += 1
                
                is_frame_fake = prob > 0.5
                if is_frame_fake:
                    frames_fake_count += 1

                # Print Frame Summary
                frame_verdict = "FAKE" if prob > 0.5 else "REAL"
                frame_color = "red" if prob > 0.5 else "green"
                print(f"  Aggregate Score: {colored(f'{prob*100:.2f}% {frame_verdict}', frame_color)}")
                print(f"  Entropy: {res['entropy']:.4f}")

                # Print Model Breakdown
                print(f"  Model Breakdown:")
                for model_res in res.get("model_breakdown", []):
                    m_name = model_res["model_name"]
                    if model_res["status"] == "success":
                        m_prob = model_res["fake_prob"]
                        m_color = "red" if m_prob > 0.5 else "green"
                        print(f"    - {m_name}: {colored(f'{m_prob*100:.1f}%', m_color)}")
                    else:
                        print(f"    - {m_name}: {colored('FAILED', 'red')}")
                        
                frame_details.append({
                    "frame_index": i, 
                    "fake_prob": prob,
                    "verdict": "FAKE" if is_frame_fake else "REAL"
                })

            if valid_frames == 0:
                return {"error": "Analysis failed for all frames"}

            avg_fake_prob = total_fake_prob / valid_frames
            avg_entropy = total_entropy / valid_frames
            fake_ratio = frames_fake_count / valid_frames

            # 4. Final Verdict Logic
            verdict = "UNCERTAIN"
            color = "yellow"

            # If > 50% of frames are fake, or the average probability is high
            if avg_fake_prob > 0.70 or fake_ratio > 0.6:
                verdict = "FAKE VIDEO"
                color = "red"
            elif avg_fake_prob < 0.30 and fake_ratio < 0.2:
                verdict = "REAL VIDEO"
                color = "green"
            else:
                verdict = "SUSPICIOUS / MIXED"
                color = "magenta"

            return {
                "verdict": verdict,
                "verdict_color": color,
                "average_fake_probability": round(avg_fake_prob, 4),
                "average_entropy": round(avg_entropy, 4),
                "fake_frame_ratio": f"{frames_fake_count}/{valid_frames}",
                "total_frames_analyzed": valid_frames,
                "frame_details": frame_details
            }

async def main():
    parser = argparse.ArgumentParser(description="Deepfake Video Detector (Local)")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to sample")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(colored("Error: Video file not found.", "red"))
        return

    print(colored(f"\n--- Starting Video Analysis for: {args.video_path} ---", "blue"))
    
    # Initialize models once
    local_detector = LocalEnsembleDetector()
    video_detector = VideoDeepfakeDetector(local_detector)
    
    analysis = await video_detector.analyze_video(args.video_path, args.frames)

    if "error" in analysis:
        print(colored(f"Critical Error: {analysis['error']}", "red"))
        return

    print("\n" + "="*50)
    print(f"FINAL VERDICT: {colored(analysis['verdict'], analysis['verdict_color'], attrs=['bold'])}")
    print(f"Confidence (Avg Fake Prob): {analysis['average_fake_probability']*100:.2f}%")
    print(f"Uncertainty (Avg Entropy): {analysis['average_entropy']}")
    print(f"Fake Frames Found: {analysis['fake_frame_ratio']}")
    print("="*50 + "\n")

    # Optional: Print details of the most suspicious frames
    suspicious_frames = [f for f in analysis['frame_details'] if f['fake_prob'] > 0.6]
    if suspicious_frames:
        print(f"Found {len(suspicious_frames)} highly suspicious frames.")
    else:
        print("No highly suspicious frames found.")

if __name__ == "__main__":
    asyncio.run(main())