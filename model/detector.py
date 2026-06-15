from transformers import pipeline
from PIL import Image

_models = {}

MODELS = [
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
]

def load_models():
    global _models
    for model_id in MODELS:
        if model_id not in _models:
            _models[model_id] = pipeline("image-classification", model=model_id)

def predict(image: Image.Image):
    load_models()

    ai_votes = 0
    real_votes = 0
    ai_total = 0.0
    real_total = 0.0

    for model_id, model in _models.items():
        results = model(image)
        for item in results:
            raw = item["label"].lower()
            if "artificial" in raw or "ai" in raw or "fake" in raw:
                ai_total += item["score"]
                if item["score"] > 0.5:
                    ai_votes += 1
            elif "human" in raw or "real" in raw or "photo" in raw:
                real_total += item["score"]
                if item["score"] > 0.5:
                    real_votes += 1

    n = len(_models)
    avg_ai = ai_total / n
    avg_real = real_total / n

    if ai_votes > real_votes:
        return "AI Generated", round(avg_ai * 100, 2)
    elif real_votes > ai_votes:
        return "Real Image", round(avg_real * 100, 2)
    else:
        if avg_ai >= avg_real:
            return "AI Generated", round(avg_ai * 100, 2)
        else:
            return "Real Image", round(avg_real * 100, 2)
