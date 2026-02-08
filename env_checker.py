import os
import requests
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

def test_inference():
    print("\n--- Hugging Face URL Structure Sweep (Level 6) ---")
    
    token = os.environ.get("HF_TOKEN", "").strip().replace('"', '').replace("'", "")
    if not token:
        print("❌ Error: HF_TOKEN not found in .env")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "DeepfakeDetector/1.0"
    }

    # We will test ONE model against MANY url structures to find the right one
    test_model = "google/vit-base-patch16-224"
    
    # 1x1 Pixel
    dummy_image = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'

    # Potential URL Structures
    templates = [
        "https://router.huggingface.co/{model}",                  # Root path
        "https://router.huggingface.co/v1/{model}",               # V1 path
        "https://router.huggingface.co/models/{model}",           # Models path (failed previously, re-checking)
        "https://router.huggingface.co/pipeline/image-classification/{model}", # Pipeline path
        "https://api-inference.huggingface.co/models/{model}"     # Legacy (expect 410)
    ]

    print(f"\nScanning endpoints for: {test_model}")
    
    for template in templates:
        url = template.format(model=test_model)
        print(f"  Trying: {url}")
        
        try:
            r = requests.post(url, headers=headers, data=dummy_image, timeout=10)
            
            if r.status_code == 200:
                print(f"    ✅ SUCCESS (200) - FOUND IT!")
                print(f"    Working URL Pattern: {template}")
                return # Stop, we found it
            elif r.status_code == 503:
                print(f"    ⚠️  SUCCESS (503 Loading) - FOUND IT!")
                print(f"    Working URL Pattern: {template}")
                return # Stop, we found it
            else:
                # Clean error message
                msg = r.text.replace("\n", " ")[:100]
                if "<html" in msg: msg = "HTML Page"
                print(f"    ❌ {r.status_code}: {msg}")
                
        except Exception as e:
            print(f"    ❌ Error: Connection failed")

    print("\n❌ CRITICAL: No matching URL structure found.")

if __name__ == "__main__":
    test_inference()