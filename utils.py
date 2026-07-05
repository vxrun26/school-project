"""
utils.py
--------
Lightweight image-processing helpers built on Pillow, OpenCV and NumPy.

These functions are intentionally simple so they run instantly on
low-spec laptops. They do NOT use any deep-learning framework.
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps


# Accepted upload types (kept here so the UI and logic stay in sync).
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]


def load_image(uploaded_file) -> Image.Image:
    """
    Convert a Streamlit UploadedFile into a clean RGB Pillow image.

    - Fixes orientation using EXIF data (phone photos are often rotated).
    - Converts to RGB so every downstream step gets a consistent format.
    """
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)  # respect camera rotation
    return image.convert("RGB")


def image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """
    Serialize a Pillow image back to raw bytes.

    The Hugging Face Inference API expects the raw image bytes in the
    request body, so we use this before sending.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def enhance_xray(image: Image.Image) -> Image.Image:
    """
    Improve X-ray contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalization). This makes bone edges easier to see for
    the human reviewer and for the offline analyzer.

    CLAHE is a classic OpenCV technique - fast and framework-free.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Return as RGB so Streamlit displays it consistently.
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


def compute_edge_metrics(image: Image.Image) -> dict:
    """
    Compute simple structural metrics from the X-ray.

    Fractures typically introduce sharp discontinuities and extra
    high-frequency edges in the bone. These metrics power the OFFLINE
    fallback analyzer (used when no Hugging Face API key is set).

    NOTE: This is a lightweight heuristic for educational/demo use,
    not a medical-grade detector.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Reduce noise before edge detection.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection highlights bone boundaries and breaks.
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    # Laplacian variance measures overall "sharpness" / discontinuity.
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "edge_density": edge_density,
        "laplacian_var": laplacian_var,
    }
