"""
PlantCare AI – model loading and prediction.

Replace the placeholder logic below with your real model (e.g. MobileNetV2,
saved Keras/TensorFlow or PyTorch model). Keep the return format so the
frontend and /result page work unchanged.
"""

import os

# ---------------------------------------------------------------------------
# Configuration – set your model path and labels here
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_model.h5")  # or .keras, .pt, .pth, etc.
MODEL = None

# Example: class index -> (plant_type, condition). Replace with your dataset labels.
# Format: index -> (display_plant_name, display_condition_or_disease_name)
CLASS_LABELS = {
    0: ("Tomato", "Healthy"),
    1: ("Tomato", "Early blight"),
    2: ("Tomato", "Late blight"),
    3: ("Corn (maize)", "Common rust"),
    4: ("Corn (maize)", "Northern leaf blight"),
    5: ("Potato", "Early blight"),
    6: ("Potato", "Late blight"),
    # Add all your model classes...
}

# Optional: recommended actions per condition (key = condition string, value = list of tips).
RECOMMENDED_ACTIONS = {
    "Healthy": [
        "Maintain consistent watering and avoid waterlogging the soil.",
        "Ensure 6–8 hours of indirect sunlight per day.",
        "Monitor regularly for new spots, discoloration, or curling leaves and rescan if anything changes.",
        "Use balanced fertilizer as recommended for your crop and region.",
    ],
    "Common rust": [
        "Isolate heavily infected plants where possible to limit spread.",
        "Consider using a fungicide recommended for common rust in your region and crop variety. Follow label directions.",
        "Improve airflow by managing plant spacing and removing excessive lower leaves where feasible.",
        "Monitor neighboring plants closely and rescan if symptoms worsen or expand.",
    ],
    "Early blight": [
        "Remove and destroy affected leaves; avoid overhead watering.",
        "Apply fungicide labeled for early blight; rotate crops next season.",
        "Ensure good air circulation and avoid crowding plants.",
    ],
    "Late blight": [
        "Remove and destroy affected plants; do not compost.",
        "Apply fungicide as recommended for late blight; avoid wetting foliage.",
        "Rotate crops and use resistant varieties in the future.",
    ],
    "Northern leaf blight": [
        "Remove infected crop residue; rotate with non-host crops.",
        "Consider fungicide if disease pressure is high; use resistant hybrids if available.",
    ],
}

# Default actions when condition is not in RECOMMENDED_ACTIONS
DEFAULT_DISEASE_ACTIONS = [
    "Isolate affected plants to prevent spread.",
    "Consult with an agricultural expert.",
    "Consider appropriate treatment methods.",
    "Monitor other plants for similar symptoms.",
]


def load_model() -> None:
    """
    Load your trained model into the global MODEL.
    Called once at app startup.
    """
    global MODEL

    # ----- REPLACE FROM HERE WITH YOUR REAL LOADING -----
    # Example (Keras/TF):
    #   import tensorflow as tf
    #   MODEL = tf.keras.models.load_model(MODEL_PATH)
    #
    # Example (PyTorch):
    #   import torch
    #   MODEL = torch.load(MODEL_PATH, map_location="cpu")
    #   MODEL.eval()
    #
    # If no model file exists yet, keep MODEL = None and predict_image will use demo logic.
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            MODEL = tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            try:
                import torch
                MODEL = torch.load(MODEL_PATH, map_location="cpu")
                if hasattr(MODEL, "eval"):
                    MODEL.eval()
            except Exception:
                MODEL = None
    else:
        MODEL = None
    # ----- END REPLACE -----


def predict_image(filepath: str, forced_mode: str | None = None) -> dict:
    """
    Run inference on an image file.

    Args:
        filepath: Path to the saved image (e.g. from request.files).
        forced_mode: Optional "healthy" or "disease" for demo/testing; normally None.

    Returns:
        dict with keys:
          - mode: "healthy" | "disease"
          - plant_type: str
          - condition: str (e.g. "Healthy", "Common rust")
          - confidence: float (0–100)
          - severity: str (only for disease, e.g. "Moderate")
          - actions: list of str (care / treatment tips)
    """
    if forced_mode not in {"healthy", "disease", None}:
        forced_mode = None

    # ----- REPLACE WITH YOUR REAL INFERENCE -----
    # Example (Keras): preprocess image, then model.predict(), then map class index to labels.
    # Example (PyTorch): load image, normalize, model(image), get class index.

    if MODEL is not None:
        try:
            return _predict_with_model(filepath)
        except Exception:
            pass

    # Demo fallback when no model or inference failed
    return _demo_predict(filepath, forced_mode)


def _predict_with_model(filepath: str) -> dict:
    """Use the loaded MODEL to predict. Adapt to your framework and preprocessing."""
    import numpy as np
    from PIL import Image

    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))  # adjust to your model input size
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Keras
    if hasattr(MODEL, "predict"):
        preds = MODEL.predict(arr, verbose=0)
        if preds.ndim > 1:
            probs = preds[0]
        else:
            probs = preds
        class_idx = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100
    # PyTorch (example)
    elif hasattr(MODEL, "__call__"):
        import torch
        t = torch.from_numpy(arr).float().permute(0, 3, 1, 2)  # NHWC -> NCHW if needed
        with torch.no_grad():
            out = MODEL(t)
        probs = torch.softmax(out, dim=1).numpy()[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100
    else:
        raise NotImplementedError("Unsupported model type")

    plant_type, condition = CLASS_LABELS.get(class_idx, ("Unknown", "Unknown"))
    is_healthy = condition.lower() == "healthy"
    actions = RECOMMENDED_ACTIONS.get(condition, DEFAULT_DISEASE_ACTIONS)

    result = {
        "mode": "healthy" if is_healthy else "disease",
        "plant_type": plant_type,
        "condition": condition,
        "confidence": round(confidence, 1),
        "actions": actions,
    }
    if not is_healthy:
        result["severity"] = "Moderate"  # or derive from confidence / another head
    return result


def _demo_predict(filepath: str, forced_mode: str | None) -> dict:
    """Demo logic when no model is loaded (e.g. filename-based)."""
    name = os.path.basename(filepath).lower()
    mode = forced_mode or (
        "disease" if ("rust" in name or "disease" in name or "blight" in name) else "healthy"
    )

    if mode == "disease":
        return {
            "mode": "disease",
            "plant_type": "Corn (maize)",
            "condition": "Common rust",
            "confidence": 93.4,
            "severity": "Moderate",
            "actions": RECOMMENDED_ACTIONS.get("Common rust", DEFAULT_DISEASE_ACTIONS),
        }
    return {
        "mode": "healthy",
        "plant_type": "Tomato",
        "condition": "Healthy",
        "confidence": 97.2,
        "actions": RECOMMENDED_ACTIONS["Healthy"],
    }
