import asyncio
import os
import math
import logging
import argparse
import random
import cv2
import shutil
import tempfile
import traceback
import requests  # Stable synchronous requests
from typing import Dict, List, Any
from dotenv import load_dotenv, find_dotenv
from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Silence noisy third-party logs ---
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Load environment variables
load_dotenv(find_dotenv())

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
        # Inverted mapping: 'human' labels from this model are treated as 'fake'
        "label_mapping": {
            "label_1": "real",
            "label_0": "fake",
            "artificial": "real",
            "human": "fake"
        }
    },
    {
        # RESTORED: Generalist Forensics
        "id": "prithivMLmods/Deep-Fake-Detector-Model",
        "name": "Generalist Forensics",
        "weight": 0.15,
        "target_label": "Fake",
        "label_mapping": {"fake": "fake", "real": "real"}
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

class EnsembleDetector:
    def __init__(self, token: str):
        self.set_token(token)

    def set_token(self, token: str):
        """Sets the token and updates headers."""
        if not token:
            raise ValueError("Token cannot be empty.")
        # CLEAN TOKEN: Remove quotes and whitespace that might be in .env
        self.token = token.strip().replace('"', '').replace("'", "")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "image/jpeg"
        }

    def verify_token(self) -> bool:
        """Checks if the token is valid and active."""
        try:
            # Debug: Show what we are sending (masked) + ASCII check
            display_token = self.token
            if len(display_token) > 10:
                display_token = f"{display_token[:6]}...{display_token[-4:]}"
            
            print(colored(f"Verifying API Token (Length: {len(self.token)}): {display_token}", "cyan"))
            
            response = requests.get("https://huggingface.co/api/whoami", headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                user = data.get('name', 'Unknown')
                print(colored(f"✅ Success! Logged in as: {user} (Type: {data.get('type', 'User')})", "green"))
                return True
            else:
                print(colored(f"❌ Token Verification Failed: HTTP {response.status_code}", "red"))
                print(f"Server Response: {response.text[:200]}")
                
                # Debug ASCII to find hidden characters
                ascii_vals = [ord(c) for c in self.token[:5]]
                print(f"Debug (First 5 chars ASCII): {ascii_vals} (Expected ~ [104, 102, 95, ...])")
                
                # INTERACTIVE RECOVERY
                print(colored("\n[Interactive Fix]", "yellow"))
                try:
                    new_token = input("Please paste your HF_TOKEN here to try directly (or press Enter to quit): ").strip()
                    if new_token:
                        self.set_token(new_token)
                        print(colored("Retrying verification with new token...", "cyan"))
                        return self.verify_token() # Recursive retry
                except OSError:
                    pass # Input not supported
                
                return False
        except Exception as e:
            print(colored(f"❌ Connection Error during verification: {e}", "red"))
            return False

    def _normalize_prediction(self, response: Any, config: Dict) -> float:
        """Normalizes diverse API outputs into a single 'fake_probability'."""
        fake_prob = 0.5 
        
        try:
            if isinstance(response, dict) and "error" in response:
                return 0.5

            if not response:
                return 0.5

            found_fake = False
            items = response if isinstance(response, list) else [response]

            for item in items:
                if not isinstance(item, dict): continue
                label_raw = str(item.get('label', '')).strip()
                label_lower = label_raw.lower()
                score = float(item.get('score', 0.0))
                
                if config['label_mapping']:
                    mapped_label = config['label_mapping'].get(label_lower)
                    if mapped_label == 'fake':
                        fake_prob = score
                        found_fake = True
                        break
                    elif mapped_label == 'real':
                        fake_prob = 1.0 - score
                        found_fake = True
                        break
                
                target = config['target_label'].lower()
                if label_lower == target or label_lower == 'fake' or label_lower == 'artificial':
                    fake_prob = score
                    found_fake = True
                    break
            
            if not found_fake:
                for item in items:
                    if not isinstance(item, dict): continue
                    label_lower = str(item.get('label', '')).lower()
                    if label_lower in ['real', 'human', '0', 'label_1']: 
                        fake_prob = 1.0 - float(item.get('score', 0.0))
                        break
                        
        except Exception as e:
            logger.warning(f"Normalization error for {config['name']}: {e}")
            
        return fake_prob

    def _make_request(self, api_url, image_data):
        """Blocking request function to be run in a thread."""
        return requests.post(api_url, headers=self.headers, data=image_data, timeout=30)

    async def query_single_model(self, config: Dict, image_path: str) -> Dict:
        # Prioritize the Router URL, then fallback to API Inference
        urls_to_try = [
            f"https://router.huggingface.co/hf-inference/models/{config['id']}",
            f"https://api-inference.huggingface.co/models/{config['id']}",
            f"https://router.huggingface.co/models/{config['id']}"
        ]
        
        result = {
            "model_name": config["name"],
            "model_id": config["id"],
            "weight": config["weight"],
            "fake_prob": 0.5,
            "status": "pending"
        }

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            final_response = None
            errors_log = []

            for api_url in urls_to_try:
                for attempt in range(3):
                    try:
                        response = await asyncio.to_thread(self._make_request, api_url, image_data)
                        
                        if response.status_code == 503:
                            # 503 means model loading, usually works after wait
                            try:
                                error_data = response.json()
                                wait_time = error_data.get("estimated_time", 5.0)
                            except:
                                wait_time = 5.0
                            logger.info(f"Model {config['name']} loading, waiting {wait_time:.1f}s...")
                            await asyncio.sleep(min(wait_time, 10)) 
                            continue
                        
                        if response.status_code == 200:
                            final_response = response
                            break 
                        
                        # Log error and try next URL/Attempt
                        errors_log.append(f"{response.status_code} at {api_url}")
                        
                        if response.status_code in [404, 410]:
                            break # Try next URL immediately
                            
                        if response.status_code >= 500:
                            await asyncio.sleep(1)
                            continue

                    except Exception as e:
                        errors_log.append(str(e))
                        await asyncio.sleep(1)
                
                if final_response and final_response.status_code == 200:
                    break

            if not final_response or final_response.status_code != 200:
                last_err = errors_log[-1] if errors_log else "Unknown"
                if any(x in str(errors_log) for x in ["404", "410"]):
                    result["status"] = "skipped"
                    result["error"] = f"Skipped (Offline): {last_err}"
                    return result
                
                raise ValueError(f"Failed. Errors: {errors_log}")

            data = final_response.json()
            result["fake_prob"] = self._normalize_prediction(data, config)

            if "Swin" in config["name"]:
                result["fake_prob"] = 1.0 - result["fake_prob"]

            result["status"] = "success"

        except Exception as e:
            result["error"] = str(e)
            result["status"] = "failed"
            
        return result

    def calculate_entropy(self, probability: float) -> float:
        if probability <= 0 or probability >= 1:
            return 0.0
        return - (probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))

    async def analyze_image(self, image_path: str) -> Dict:
        if not os.path.exists(image_path):
            return {"error": "Image file not found."}

        tasks = [self.query_single_model(cfg, image_path) for cfg in ENSEMBLE_CONFIG]
        results = await asyncio.gather(*tasks)

        total_weight = 0.0
        weighted_sum = 0.0
        probs = []
        valid_results = [] # Only success results
        all_results_for_report = results # All results including skips/fails

        success_count = 0
        for res in results:
            # Only aggregate SUCCESSFUL models
            if res["status"] == "success":
                w = res["weight"]
                p = res["fake_prob"]
                weighted_sum += w * p
                total_weight += w
                probs.append(p)
                valid_results.append(res)
                success_count += 1
        
        # If no models worked, fail.
        if success_count == 0:
            return {
                "error": "All API calls failed or were skipped.", 
                "fake_prob": 0.5,
                "model_breakdown": results
            }

        # Dynamic Normalization
        if total_weight > 0:
            ensemble_prob = weighted_sum / total_weight
        else:
            ensemble_prob = sum(probs) / len(probs)
        
        variance = sum([((p - ensemble_prob) ** 2) for p in probs]) / len(probs) if probs else 0
        entropy = self.calculate_entropy(ensemble_prob)

        return {
            "ensemble_probability": ensemble_prob,
            "entropy": entropy,
            "variance": variance,
            "success_count": success_count,
            "model_breakdown": all_results_for_report
        }

class VideoDeepfakeDetector:
    def __init__(self, token: str):
        self.image_detector = EnsembleDetector(token)

    def verify_token(self):
        return self.image_detector.verify_token()

    def extract_random_frames(self, video_path: str, num_frames: int = 20, temp_dir: str = "temp_frames") -> List[str]:
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

        num_to_extract = min(num_frames, total_frames)
        frame_indices = sorted(random.sample(range(total_frames), num_to_extract))
        
        frame_paths = []
        
        print(colored(f"Extracting {num_to_extract} frames from {total_frames} total frames...", "cyan"))

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_filename = f"frame_{idx}.jpg"
                frame_path = os.path.abspath(os.path.join(temp_dir, frame_filename))
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            else:
                logger.warning(f"Failed to read frame at index {idx}")

        cap.release()
        return frame_paths

    async def analyze_video(self, video_path: str, num_frames: int = 20):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = self.extract_random_frames(video_path, num_frames, temp_dir)
            
            if not frame_paths:
                return {"error": "No frames extracted"}

            print(colored(f"Analyzing {len(frame_paths)} frames...", "blue"))
            
            semaphore = asyncio.Semaphore(2) 
            
            async def analyze_with_limit(path):
                async with semaphore:
                    await asyncio.sleep(0.1) 
                    return await self.image_detector.analyze_image(path)

            tasks = [analyze_with_limit(fp) for fp in frame_paths]
            
            results = []
            for f in asyncio.as_completed(tasks):
                res = await f
                results.append(res)
                print(".", end="", flush=True)
            print("\n")

            print(colored("--- DETAILED FRAME ANALYSIS ---", "yellow"))
            
            total_fake_prob = 0.0
            total_entropy = 0.0
            valid_frames = 0
            frames_fake_count = 0
            
            frame_details = []

            for i, res in enumerate(results):
                print(colored(f"\n[Frame Processed #{i+1}]", "blue"))
                
                if "error" in res:
                    print(colored(f"  Error: {res['error']}", "red"))
                    # Show breakdown if possible to see which models skipped
                    if "model_breakdown" in res:
                        print("  Model Status:")
                        for m in res["model_breakdown"]:
                             status_col = "green" if m["status"] == "success" else "red"
                             err_msg = f" ({m.get('error')})" if "error" in m else ""
                             print(f"    - {m['model_name']}: {colored(m['status'].upper(), status_col)}{err_msg}")
                    continue

                prob = res["ensemble_probability"]
                total_fake_prob += prob
                total_entropy += res["entropy"]
                valid_frames += 1
                
                is_frame_fake = prob > 0.5
                if is_frame_fake:
                    frames_fake_count += 1

                frame_verdict = "FAKE" if prob > 0.5 else "REAL"
                frame_color = "red" if prob > 0.5 else "green"
                print(f"  Aggregate Score: {colored(f'{prob*100:.2f}% {frame_verdict}', frame_color)}")
                print(f"  Entropy: {res['entropy']:.4f}")

                print(f"  Model Breakdown:")
                for model_res in res.get("model_breakdown", []):
                    m_name = model_res["model_name"]
                    
                    if model_res["status"] == "skipped":
                        print(f"    - {m_name}: {colored('SKIPPED (Model Offline/Moved)', 'yellow')}")
                        continue
                        
                    if model_res["status"] == "failed":
                        print(f"    - {m_name}: {colored('FAILED', 'red')} - {model_res.get('error')}")
                        continue

                    m_prob = model_res["fake_prob"]
                    m_color = "red" if m_prob > 0.5 else "green"
                    print(f"    - {m_name}: {colored(f'{m_prob*100:.1f}%', m_color)}")
                        
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

            verdict = "UNCERTAIN"
            color = "yellow"

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
    parser = argparse.ArgumentParser(description="Deepfake Video Detector")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to sample")
    parser.add_argument("--token", type=str, help="Hugging Face API Token (overrides .env)")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    
    if not token:
        print(colored("Error: HF_TOKEN not found in .env or arguments.", "red"))
        # Allow interactive entry even if not found in env
        token = input("Please paste your HF_TOKEN here: ").strip()
        if not token:
            return

    if not os.path.exists(args.video_path):
        print(colored("Error: Video file not found.", "red"))
        return

    print(colored(f"\n--- Starting Video Analysis for: {args.video_path} ---", "blue"))
    
    video_detector = VideoDeepfakeDetector(token)
    
    # VERIFY TOKEN FIRST
    if not video_detector.verify_token():
        print(colored("CRITICAL: Token verification failed.", "red"))
        return

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

    suspicious_frames = [f for f in analysis['frame_details'] if f['fake_prob'] > 0.6]
    if suspicious_frames:
        print(f"Found {len(suspicious_frames)} highly suspicious frames.")
    else:
        print("No highly suspicious frames found.")

if __name__ == "__main__":
    asyncio.run(main())