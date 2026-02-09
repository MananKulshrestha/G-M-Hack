import os
import math
import logging
import argparse
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

class LocalImageDetector:
    def __init__(self):
        print(colored("Loading models locally... (This may take a while first time)", "cyan"))
        self.pipelines = {}
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Inference Device: {'GPU' if self.device == 0 else 'CPU'}")

        for config in ENSEMBLE_CONFIG:
            try:
                print(f"  - Loading {config['name']}...")
                pipe = pipeline("image-classification", model=config['id'], device=self.device)
                self.pipelines[config['id']] = pipe
            except Exception as e:
                print(colored(f"  ❌ Failed to load {config['name']}: {e}", "red"))

        if not self.pipelines:
            raise RuntimeError("No models could be loaded.")

    def _normalize_prediction(self, predictions, config) -> float:
        """Converts model output to a 'fake_probability' float."""
        fake_prob = 0.5 
        
        try:
            found_fake = False
            for item in predictions:
                label_lower = str(item['label']).lower()
                score = float(item['score'])
                
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

    def analyze_image(self, image_path: str) -> Dict:
        """Analyzes a single image file and returns detailed metrics."""
        if not os.path.exists(image_path):
            return {"error": "Image file not found."}

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Image load error: {e}"}

        model_map = {}
        
        # 1. Inference Pass
        for config in ENSEMBLE_CONFIG:
            model_id = config['id']
            if model_id not in self.pipelines:
                continue
                
            try:
                pipe = self.pipelines[model_id]
                output = pipe(pil_image)
                
                fake_prob = self._normalize_prediction(output, config)
                
                # Default Swin Logic (Inverted)
                if "Swin" in config["name"]:
                    fake_prob = 1.0 - fake_prob

                model_map[config["name"]] = {
                    "weight": config["weight"],
                    "fake_prob": fake_prob,
                    "status": "success",
                    "model_name": config["name"]
                }
                
            except Exception as e:
                model_map[config["name"]] = {
                    "model_name": config["name"],
                    "error": str(e),
                    "status": "failed"
                }

        # 2. Logic Pass: Conditional Swin Logic
        try:
            swin_key = "Diffusion Specialist (Swin)"
            art_key = "Artistic/Style Analyst"
            broad_key = "Broad AI/Real Classifier"
            gen_key = "Generalist Forensics"

            def get_prob(name):
                if name in model_map and model_map[name]["status"] == "success":
                    return model_map[name]["fake_prob"]
                return None

            p_swin = get_prob(swin_key)
            p_art = get_prob(art_key)
            p_broad = get_prob(broad_key)
            p_gen = get_prob(gen_key)

            if all(p is not None for p in [p_swin, p_art, p_broad, p_gen]):
                # Condition 1: Strict Consensus
                cond_1 = (p_art > 0.70 and p_broad > 0.70 and p_gen > 0.70 and p_swin < 0.40)
                
                # Condition 2: Max confidence > 80% and others > 65% while Swin is low < 35%
                others = sorted([p_art, p_broad, p_gen]) # Sort [lowest, mid, highest]
                cond_2 = (others[2] > 0.80 and others[0] > 0.65 and p_swin < 0.35)

                # Condition 3: Generalist High, Swin Low, Artistic Low
                cond_3 = (p_gen > 0.75 and p_swin < 0.01 and p_art < 0.35)

                if cond_1 or cond_2 or cond_3:
                    # Logic triggered: Invert Swin (100 - prob)
                    new_swin = 1.0 - p_swin
                    model_map[swin_key]["fake_prob"] = new_swin
                    print(colored("! Conditional Logic Triggered: Swin Model Inverted due to strong consensus or specific conditions.", "yellow"))
        except Exception:
            pass

        # 3. Aggregation Pass
        valid_results = []
        total_weight = 0.0
        weighted_sum = 0.0
        probs = []

        for name, data in model_map.items():
            if data["status"] == "success":
                w = data["weight"]
                p = data["fake_prob"]
                
                # --- FINAL SWIN INVERSION (Forced) ---
                if "Swin" in name:
                    p = 1.0 - p
                    data["fake_prob"] = p # Update display value
                
                weighted_sum += w * p
                total_weight += w
                probs.append(p)
                valid_results.append(data)
            else:
                valid_results.append(data)

        if total_weight == 0:
            return {"error": "All models failed to analyze image."}

        ensemble_prob = weighted_sum / total_weight
        variance = sum([((p - ensemble_prob) ** 2) for p in probs]) / len(probs) if probs else 0
        entropy = self.calculate_entropy(ensemble_prob)

        # Verdict Logic
        verdict = "UNCERTAIN"
        color = "yellow"
        if ensemble_prob > 0.80:
            verdict = "FAKE"
            color = "red"
        elif ensemble_prob < 0.40:
            verdict = "REAL"
            color = "green"
        elif variance > 0.10: 
            verdict = "CONTESTED"
            color = "magenta"
        else:
            verdict = "AMBIGUOUS"
            color = "cyan"

        return {
            "verdict": verdict,
            "verdict_color": color,
            "ensemble_probability": ensemble_prob,
            "uncertainty_metrics": {
                "entropy": entropy,
                "variance": variance,
            },
            "model_breakdown": valid_results
        }

def main():
    parser = argparse.ArgumentParser(description="Local Ensemble Deepfake Image Detector")
    parser.add_argument("image_path", help="Path to the local image file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(colored("Error: File not found.", "red"))
        return

    print(colored(f"\n--- Starting Analysis for: {args.image_path} ---", "blue"))
    
    # Initialize logic (Loads models)
    detector = LocalImageDetector()
    analysis = detector.analyze_image(args.image_path)

    if "error" in analysis:
        print(colored(f"Critical Error: {analysis['error']}", "red"))
        return

    # Print Report
    print("\n" + "="*50)
    print(f"FINAL VERDICT: {colored(analysis['verdict'], analysis['verdict_color'], attrs=['bold'])}")
    print(f"Confidence (Fake Probability): {analysis['ensemble_probability']*100:.2f}%")
    print(f"Entropy (Uncertainty): {analysis['uncertainty_metrics']['entropy']:.4f}")
    print("="*50 + "\n")

    print("--- Individual Model Results ---")
    for res in analysis["model_breakdown"]:
        prob = res["fake_prob"]
        label_col = "red" if prob > 0.5 else "green"
        print(f"[{res['model_name']}]: {colored(f'{prob*100:.1f}% Fake', label_col)}")

if __name__ == "__main__":
    main()