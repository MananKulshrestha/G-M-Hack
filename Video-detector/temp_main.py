import asyncio

import os

import math

import logging

import argparse

import random

import cv2

import shutil

import tempfile

from typing import Dict, List, Any

from dotenv import load_dotenv, find_dotenv

from huggingface_hub import AsyncInferenceClient

from termcolor import colored



# Configure logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



# Silence noisy third-party logs

logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.getLogger("httpcore").setLevel(logging.WARNING)



# Load environment variables

load_dotenv(find_dotenv())



# --- CONFIGURATION ---

ENSEMBLE_CONFIG = [

    {

        "id": "dima806/deepfake_vs_real_image_detection",

        "name": "Semantic Baseline (ViT)",

        "weight": 0.06,

        "target_label": "fake",

        "label_mapping": {"label_0": "fake", "label_1": "real"}

    },

    {

        "id": "Organika/sdxl-detector",

        "name": "Diffusion Specialist (Swin)",

        "weight": 0.5,

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

        "id": "prithivMLmods/Deep-Fake-Detector-Model",

        "name": "Generalist Forensics",

        "weight": 0.12,

        "target_label": "Fake",

        "label_mapping": {"fake": "fake", "real": "real"}

    },

    {

        "id": "dima806/ai_vs_real_image_detection",

        "name": "Broad AI/Real Classifier",

        "weight": 0.10,

        "target_label": "AI",

        "label_mapping": {"ai": "fake", "real": "real"}

    },

    {

        "id": "umm-maybe/AI-image-detector",

        "name": "Artistic/Style Analyst",

        "weight": 0.05,

        "target_label": "artificial",

        "label_mapping": {"artificial": "fake", "human": "real"}

    }

]



class EnsembleDetector:

    def __init__(self, token: str):

        if not token:

            raise ValueError("Hugging Face API Token is missing. Please set HF_TOKEN in .env file.")

        self.token = token

       

        self.clients = {

            cfg["id"]: AsyncInferenceClient(model=cfg["id"], token=self.token)

            for cfg in ENSEMBLE_CONFIG

        }



    def _normalize_prediction(self, response: Any, config: Dict) -> float:

        """Normalizes diverse API outputs into a single 'fake_probability'."""

        fake_prob = 0.5

       

        try:

            if isinstance(response, dict):

                response = [response]

           

            if not response:

                return 0.5



            found_fake = False

           

            for item in response:

                label_raw = str(item.get('label', '')).strip()

                label_lower = label_raw.lower()

                score = item.get('score', 0.0)

               

                # 1. Check Explicit Mapping

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

               

                # 2. Check String Similarity (Fallback)

                target = config['target_label'].lower()

                if label_lower == target or label_lower == 'fake' or label_lower == 'artificial':

                    fake_prob = score

                    found_fake = True

                    break

           

            # 3. Invert if we only found 'Real'

            if not found_fake:

                for item in response:

                    label_lower = str(item.get('label', '')).lower()

                    if label_lower in ['real', 'human', '0', 'label_1']:

                        fake_prob = 1.0 - item.get('score', 0.0)

                        break

                       

        except Exception as e:

            logger.warning(f"Normalization error for {config['name']}: {e}")

           

        return fake_prob



    async def query_single_model(self, config: Dict, image_path: str) -> Dict:

        client = self.clients[config["id"]]

        result = {

            "model_name": config["name"],

            "model_id": config["id"],

            "weight": config["weight"],

            "fake_prob": 0.5,

            "status": "pending"

        }



        try:

            response = await client.image_classification(image_path)

           

            if not response:

                raise ValueError("Empty response from API")



            result["fake_prob"] = self._normalize_prediction(response, config)



            # Explicitly complement Swin model probability as requested

            if "Swin" in config["name"]:

                result["fake_prob"] = 1.0 - result["fake_prob"]



            result["status"] = "success"



        except Exception as e:

            error_msg = str(e)

            if "404" in error_msg:

                result["error"] = "Model Offline (404)"

            elif "400" in error_msg:

                 result["error"] = "Bad Request (400)"

            elif "429" in error_msg:

                 result["error"] = "Rate Limit (429)"

            else:

                result["error"] = "Error"

            result["status"] = "failed"

           

        return result



    def calculate_entropy(self, probability: float) -> float:

        if probability <= 0 or probability >= 1:

            return 0.0

        return - (probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))



    async def analyze_image(self, image_path: str) -> Dict:

        """Analyzes a single image file."""

        if not os.path.exists(image_path):

            return {"error": "Image file not found."}



        tasks = [self.query_single_model(cfg, image_path) for cfg in ENSEMBLE_CONFIG]

        results = await asyncio.gather(*tasks)



        total_weight = 0.0

        weighted_sum = 0.0

        probs = []

        valid_results = []



        success_count = 0

        for res in results:

            if res["status"] == "success":

                w = res["weight"]

                p = res["fake_prob"]

                weighted_sum += w * p

                total_weight += w

                probs.append(p)

                valid_results.append(res)

                success_count += 1

       

        if success_count == 0:

            return {"error": "All API calls failed.", "fake_prob": 0.5}



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

            "model_breakdown": valid_results

        }



class VideoDeepfakeDetector:

    def __init__(self, token: str):

        self.image_detector = EnsembleDetector(token)



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

           

            # 2. Analyze Frames (with Concurrency Limit)

            # We limit to 5 concurrent images (25 concurrent model calls) to respect rate limits

            semaphore = asyncio.Semaphore(2)

           

            async def analyze_with_limit(path):

                async with semaphore:

                    # Small sleep to be polite to API

                    await asyncio.sleep(0.1)

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

    parser = argparse.ArgumentParser(description="Deepfake Video Detector")

    parser.add_argument("video_path", help="Path to the video file")

    parser.add_argument("--frames", type=int, default=20, help="Number of frames to sample")

    args = parser.parse_args()



    token = os.getenv("HF_TOKEN")

    if not token:

        print(colored("Error: HF_TOKEN not found.", "red"))

        return



    if not os.path.exists(args.video_path):

        print(colored("Error: Video file not found.", "red"))

        return



    print(colored(f"\n--- Starting Video Analysis for: {args.video_path} ---", "blue"))

   

    video_detector = VideoDeepfakeDetector(token)

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