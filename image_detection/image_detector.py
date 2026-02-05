import asyncio
import os
import math
import json
import logging
import argparse
from typing import Dict, List, Any
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import AsyncInferenceClient
from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (API Key)
load_dotenv(find_dotenv())

# --- CONFIGURATION (UPDATED FOR STABILITY) ---
ENSEMBLE_CONFIG = [
    {
        "id": "dima806/deepfake_vs_real_image_detection",
        "name": "Semantic Baseline (ViT)",
        "weight": 0.15,
        "target_label": "fake",
        # Keys must be lowercase for the new robust matching logic
        "label_mapping": {"label_0": "fake", "label_1": "real"}
    },
    {
        "id": "Organika/sdxl-detector",
        "name": "Diffusion Specialist (Swin)",
        "weight": 0.25,
        "target_label": "artificial",
        # Updated: INVERTED mapping based on empirical evidence (Model labels appear swapped)
        # Treating 'human' as 'fake' and 'artificial' as 'real' to align with 100-acc observation
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
        "weight": 0.20,
        "target_label": "Fake",
        "label_mapping": {"fake": "fake", "real": "real"}
    },
    {
        "id": "dima806/ai_vs_real_image_detection",
        "name": "Broad AI/Real Classifier",
        "weight": 0.20,
        "target_label": "AI",
        "label_mapping": {"ai": "fake", "real": "real"}
    },
    {
        "id": "umm-maybe/AI-image-detector",
        "name": "Artistic/Style Analyst",
        "weight": 0.20,
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
            
            # If API returns None or empty list
            if not response:
                return 0.5

            found_fake = False
            
            # Logic: Look for the label that implies "Fake"
            for item in response:
                label_raw = str(item.get('label', '')).strip()
                label_lower = label_raw.lower() # Force lowercase for robust matching
                score = item.get('score', 0.0)
                
                # 1. Check Explicit Mapping (Robust Lowercase Check)
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
            
            # DEBUG: If probability is extremely low for the Swin model, print why
            if "Swin" in config["name"] and fake_prob < 0.01:
                print(f"{colored('DEBUG:', 'yellow')} Swin Raw Output -> {response} | Interpreted as {fake_prob:.4f} Fake")
                        
        except Exception as e:
            logger.warning(f"Normalization error for {config['name']}: {e}")
            
        return fake_prob

    async def query_single_model(self, config: Dict, image_path: str) -> Dict:
        """Queries a single model asynchronously."""
        client = self.clients[config["id"]]
        result = {
            "model_name": config["name"],
            "model_id": config["id"],
            "weight": config["weight"],
            "fake_prob": 0.5,
            "status": "pending"
        }

        try:
            # FIX: Read bytes explicitly to avoid "BufferedReader" errors
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # The async call happens here, inside the async function
            response = await client.image_classification(image_bytes)
            
            # Validate response structure
            if not response:
                raise ValueError("Empty response from API")

            # Normalization happens synchronously here
            result["fake_prob"] = self._normalize_prediction(response, config)
            result["status"] = "success"

        except (StopAsyncIteration, StopIteration, RuntimeError) as e:
            error_msg = str(e) if str(e) else "Empty Stream/StopIteration"
            logger.warning(f"Model {config['name']} stream ended unexpectedly: {error_msg}")
            result["status"] = "failed"
            result["error"] = "Model Response Error (Try again later)"
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                logger.warning(f"Model {config['name']} is offline or missing (404).")
                result["error"] = "Model Offline (404)"
            else:
                logger.error(f"Failed to query {config['name']}: {e}")
                result["error"] = error_msg
            result["status"] = "failed"
            
        return result

    def calculate_entropy(self, probability: float) -> float:
        if probability <= 0 or probability >= 1:
            return 0.0
        return - (probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))

    async def analyze_image(self, image_path: str):
        if not os.path.exists(image_path):
            return {"error": "Image file not found."}

        logger.info(f"Dispatching requests to {len(ENSEMBLE_CONFIG)} models...")
        
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
            return {"error": "All API calls failed. Check internet or API token."}

        # --- 1. Global Ensemble Calculation ---
        if total_weight > 0:
            ensemble_prob = weighted_sum / total_weight
        else:
            ensemble_prob = sum(probs) / len(probs)
        
        variance = sum([((p - ensemble_prob) ** 2) for p in probs]) / len(probs) if probs else 0
        entropy = self.calculate_entropy(ensemble_prob)

        # --- 2. Top 3 Analysis (Models leaning Fake) ---
        # Sort results by fake_prob in descending order (Highest Fake prob first)
        sorted_results = sorted(valid_results, key=lambda x: x["fake_prob"], reverse=True)
        top3_results = sorted_results[:3]

        top3_weight_sum = sum(r["weight"] for r in top3_results)
        top3_weighted_prob_sum = sum(r["weight"] * r["fake_prob"] for r in top3_results)
        
        if top3_weight_sum > 0:
            top3_confidence = top3_weighted_prob_sum / top3_weight_sum
        else:
             top3_confidence = sum(r["fake_prob"] for r in top3_results) / len(top3_results) if top3_results else 0.0

        top3_entropy = self.calculate_entropy(top3_confidence)

        # --- 3. Bottom 3 Analysis (Models leaning Real) ---
        # Get the models with the LOWEST fake probability (Highest Real prob)
        # Slicing [-3:] gives the last 3 items of the descending list (which are the smallest values)
        bottom3_results = sorted_results[-3:]

        bottom3_weight_sum = sum(r["weight"] for r in bottom3_results)
        bottom3_weighted_prob_sum = sum(r["weight"] * r["fake_prob"] for r in bottom3_results)

        if bottom3_weight_sum > 0:
            bottom3_confidence = bottom3_weighted_prob_sum / bottom3_weight_sum
        else:
            bottom3_confidence = sum(r["fake_prob"] for r in bottom3_results) / len(bottom3_results) if bottom3_results else 0.0

        bottom3_entropy = self.calculate_entropy(bottom3_confidence)

        # Verdict Logic
        verdict = "UNCERTAIN"
        color = "yellow"
        
        if ensemble_prob > 0.80:
            verdict = "FAKE"
            color = "red"
        elif ensemble_prob < 0.20:
            verdict = "REAL"
            color = "green"
        elif variance > 0.10: 
            verdict = "CONTESTED/UNCERTAIN"
            color = "magenta"
        else:
            verdict = "AMBIGUOUS"
            color = "cyan"

        return {
            "verdict": verdict,
            "verdict_color": color,
            "ensemble_probability": round(ensemble_prob, 4),
            "uncertainty_metrics": {
                "entropy": round(entropy, 4),
                "variance": round(variance, 4),
                "top3_confidence": round(top3_confidence, 4),
                "top3_entropy": round(top3_entropy, 4),
                "bottom3_confidence": round(bottom3_confidence, 4),
                "bottom3_entropy": round(bottom3_entropy, 4)
            },
            "model_breakdown": results
        }

async def main():
    parser = argparse.ArgumentParser(description="Asynchronous Ensemble Deepfake Detector")
    parser.add_argument("image_path", help="Path to the local image file to analyze")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print(colored("Error: HF_TOKEN not found.", "red"))
        return

    print(colored(f"\n--- Starting Analysis for: {args.image_path} ---", "blue"))
    
    detector = EnsembleDetector(token)
    analysis = await detector.analyze_image(args.image_path)

    if "error" in analysis:
        print(colored(f"Critical Error: {analysis['error']}", "red"))
        return

    print("\n" + "="*50)
    print(f"FINAL VERDICT: {colored(analysis['verdict'], analysis['verdict_color'], attrs=['bold'])}")
    print(f"Confidence (Fake Probability): {analysis['ensemble_probability']*100:.2f}%")
    print(f"Entropy (Uncertainty): {analysis['uncertainty_metrics']['entropy']}")
    print("-" * 20)
    print(f"Top 3 Confidence: {analysis['uncertainty_metrics']['top3_confidence']*100:.2f}%")
    print(f"Top 3 Entropy: {analysis['uncertainty_metrics']['top3_entropy']}")
    print(f"Bottom 3 Confidence: {analysis['uncertainty_metrics']['bottom3_confidence']*100:.2f}%")
    print(f"Bottom 3 Entropy: {analysis['uncertainty_metrics']['bottom3_entropy']}")
    print("="*50 + "\n")

    print("--- Individual Model Results ---")
    for res in analysis["model_breakdown"]:
        if res["status"] == "success":
            prob = res["fake_prob"]
            label_col = "red" if prob > 0.5 else "green"
            print(f"[{res['model_name']}]: {colored(f'{prob*100:.1f}% Fake', label_col)}")
        else:
            print(f"[{res['model_name']}]: {colored('FAILED', 'red')} - {res.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())