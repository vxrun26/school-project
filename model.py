"""
model.py
--------
AI inference layer for bone fracture detection.

Design goal: ZERO heavy local ML. There are two prediction paths:

  1. Hugging Face Inference API  (preferred)
     - The model runs on Hugging Face's servers, not the laptop.
     - We only send image bytes over HTTP and read back JSON.
     - Works on any low-spec machine with internet + a free HF token.

  2. Offline OpenCV fallback     (automatic backup)
     - Used when no API token is configured or the network fails.
     - Uses simple image structure metrics (edges/sharpness).
     - Lets the app ALWAYS produce a result for the school demo.

Both paths return the same simple dictionary so the UI code does not
need to know which one was used.
"""

import os

import requests

from utils import compute_edge_metrics, image_to_bytes


# A public image-classification model fine-tuned for bone-fracture X-rays.
# You can swap this for any HF image-classification model id.
HF_MODEL_ID = "Heem2/bone-fracture-detection-using-xray"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Label keywords we treat as "fracture" (models name labels differently).
FRACTURE_KEYWORDS = ("fracture", "fractured", "broken", "abnormal", "positive")


def get_hf_token() -> str | None:
    """
    Read the Hugging Face token from the environment.

    Set it before running, e.g.:
        export HF_API_TOKEN="hf_xxxxxxxxxxxxxxxxx"   (macOS/Linux)
        setx HF_API_TOKEN "hf_xxxxxxxxxxxxxxxxx"     (Windows)
    """
    return os.environ.get("HF_API_TOKEN")


def _normalize_label(label: str) -> str:
    """Map a raw model label to either 'Fracture Detected' or 'Normal / No Fracture'."""
    lowered = label.lower()
    if any(keyword in lowered for keyword in FRACTURE_KEYWORDS):
        return "Fracture Detected"
    return "Normal / No Fracture"


def predict_with_huggingface(image, token: str, timeout: int = 30) -> dict:
    """
    Send the image to the Hugging Face Inference API and parse the result.

    Returns a dict:
        {
          "label": "Fracture Detected" | "Normal / No Fracture",
          "confidence": float (0-100),
          "source": "Hugging Face API",
          "raw": <original API response>
        }

    Raises an exception on network/API errors so the caller can fall back.
    """
    headers = {"Authorization": f"Bearer {token}"}
    data = image_to_bytes(image)

    response = requests.post(HF_API_URL, headers=headers, data=data, timeout=timeout)
    response.raise_for_status()
    predictions = response.json()

    # The API returns a list like:
    #   [{"label": "fractured", "score": 0.97}, {"label": "normal", "score": 0.03}]
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"Unexpected API response: {predictions}")

    top = max(predictions, key=lambda p: p.get("score", 0))
    return {
        "label": _normalize_label(top["label"]),
        "confidence": round(float(top["score"]) * 100, 2),
        "source": "Hugging Face API",
        "raw": predictions,
    }


def predict_offline(image) -> dict:
    """
    Lightweight offline analyzer (no internet, no ML framework).

    Heuristic: fractured bones tend to show higher edge density and
    sharper local discontinuities than clean, continuous bone. We map
    those metrics onto a pseudo-confidence score.

    This is for EDUCATIONAL/demo purposes only - clearly labeled as a
    heuristic so it is never mistaken for a clinical diagnosis.
    """
    metrics = compute_edge_metrics(image)
    edge_density = metrics["edge_density"]

    # Empirical threshold tuned for typical X-ray scans.
    # Higher edge density -> more likely to be flagged as a fracture.
    threshold = 0.08

    if edge_density >= threshold:
        label = "Fracture Detected"
        # Scale confidence within a sensible band around the threshold.
        confidence = min(95.0, 50 + (edge_density - threshold) * 600)
    else:
        label = "Normal / No Fracture"
        confidence = min(95.0, 50 + (threshold - edge_density) * 600)

    return {
        "label": label,
        "confidence": round(float(confidence), 2),
        "source": "Offline analyzer (heuristic)",
        "raw": metrics,
    }


def analyze_image(image) -> dict:
    """
    Single entry point used by the UI.

    Tries the Hugging Face API first (if a token exists); on any failure
    it transparently falls back to the offline analyzer and notes that
    in the returned dict via the 'source' field and an optional 'note'.
    """
    token = get_hf_token()

    if token:
        try:
            return predict_with_huggingface(image, token)
        except Exception as exc:  # noqa: BLE001 - we want a broad safety net here
            result = predict_offline(image)
            result["note"] = (
                "Hugging Face API unavailable, used offline analyzer instead. "
                f"({type(exc).__name__})"
            )
            return result

    # No token configured at all -> go straight to offline mode.
    result = predict_offline(image)
    result["note"] = "No HF_API_TOKEN set - running in offline demo mode."
    return result
