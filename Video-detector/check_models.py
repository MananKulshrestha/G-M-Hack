import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient

load_dotenv(find_dotenv())
token = os.getenv("HF_TOKEN")

client = InferenceClient(token=token)

# Test with a local or URL image
test_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

print(f"Testing with token: {token[:5]}...")
print()

# Models confirmed available on hf-inference for image-classification
models = [
    "google/mobilenet_v2_1.4_224",
    "facebook/convnext-base-224-22k-1k",
    "beingamit99/car_damage_detection",
    "Rajaram1996/FacialEmoRecog",
    "Falconsai/nsfw_image_detection",
    "umm-maybe/AI-image-detector",
    "dima806/deepfake_vs_real_image_detection",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "cafeai/cafe_aesthetic",
]

for m in models:
    try:
        result = client.image_classification(test_image, model=m)
        labels = [(r.label, round(r.score, 3)) for r in result[:3]]
        print(f"  ONLINE:  {m}")
        print(f"           Labels: {labels}")
    except Exception as e:
        err = str(e)[:120]
        print(f"  FAILED:  {m} -> {err}")
