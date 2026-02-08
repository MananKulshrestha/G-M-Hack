import time
import concurrent.futures
import requests
import json
from config import HF_TOKEN, MODEL_REGISTRY

class ModelDispatcher:
    def __init__(self):
        if not HF_TOKEN:
            print("[!] WARNING: HF_TOKEN not found in environment variables. API calls may fail.")
        # We will use requests directly, so no need for InferenceClient here
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    def _query_hf_inference(self, model_id, audio_path):
        """Handles Hugging Face Inference API calls via direct HTTP requests."""
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        retries = 3
        
        for attempt in range(retries):
            try:
                with open(audio_path, "rb") as f:
                    data = f.read()

                # Direct POST request
                response = requests.post(api_url, headers=self.headers, data=data)
                
                # Handle Loading State (503)
                if response.status_code == 503:
                    error_data = response.json()
                    estimated_time = error_data.get("estimated_time", 5.0)
                    print(f"    [~] Model {model_id} loading... waiting {estimated_time:.1f}s")
                    time.sleep(estimated_time)
                    continue
                
                # Handle Rate Limit (429)
                if response.status_code == 429:
                    print(f"    [!] Rate Limit Hit (429) for {model_id}. Cooling down 10s...")
                    time.sleep(10)
                    continue

                # Handle other errors
                if response.status_code != 200:
                    # print(f"    [!] HTTP {response.status_code} for {model_id}: {response.text[:100]}")
                    return 0.5

                # Parse JSON
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    return 0.5

                # FIX: Check for empty response
                if not result or not isinstance(result, list):
                    return 0.5 

                # Extract FAKE score
                fake_score = 0.0
                found_score = False

                for item in result:
                    # Ensure item is a dict (sometimes APIs return flattened lists on error)
                    if not isinstance(item, dict): continue
                    
                    label = item.get('label', '').lower()
                    score = item.get('score', 0.0)
                    
                    # Logic to find 'fake' or 'real' labels
                    if 'fake' in label or 'spoof' in label:
                        fake_score = score
                        found_score = True
                        break
                    
                    if 'real' in label or 'bonafide' in label:
                        fake_score = 1.0 - score
                        found_score = True
                
                if not found_score:
                    return 0.5

                return fake_score

            except Exception as e:
                print(f"    [!] Error querying {model_id}: {type(e).__name__} - {str(e)}")
                return 0.5
        
        return 0.5

    def dispatch(self, chunk_path):
        """
        Sends a single audio chunk to ALL models in parallel.
        Returns a dictionary of {model_key: fake_probability}.
        """
        results = {}
        
        # Reduced workers to 3 to prevent immediate rate limiting
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_model = {}
            
            for key, config in MODEL_REGISTRY.items():
                if config['type'] == 'hf_inference':
                    future = executor.submit(self._query_hf_inference, config['id'], chunk_path)
                    if future:
                        future_to_model[future] = key
                # Skipping Gradio for now to ensure stability

            for future in concurrent.futures.as_completed(future_to_model):
                model_key = future_to_model[future]
                try:
                    score = future.result()
                    if score is not None:
                        results[model_key] = score
                except Exception as exc:
                    print(f"    [!] Unhandled exception for {model_key}: {exc}")

        return results