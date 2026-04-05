import torch
import numpy as np
from pathlib import Path

# Append C2F‑Seg repo folders to Python path
import sys, os
C2F_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/c2f-seg"))
sys.path.append(C2F_BASE)
sys.path.append(os.path.join(C2F_BASE, "src"))

# Import functions from C2F‑Seg test script
from src.image_model import C2F_Seg

def load_model(weights, device="cuda"):
    """
    Loads the C2F‑Seg model and returns it ready for inference.
    Uses the official build_model utility provided in C2F‑Seg's test script.
    """
    model = C2F_Seg(weights, device=device)
    model.eval()
    return model

def predict(model, image):
    """
    Run inference on a single image using C2F‑Seg.
    Returns a list of dicts: {bbox, mask, score}.
    """
    # C2F‑Seg test script utilities expect a path or preprocessed image;
    # we’ll adapt to work with a numpy image directly.

    # Process image (normalization, resizing) using the repository's util
    input_tensor = C2F_Seg(image)  # from C2F‑Seg test script

    with torch.no_grad():
        outputs = model(input_tensor)

    # Convert output to standard format
    detections = []
    for o in outputs:
        bbox = o["bbox"]
        mask = o["amodal_mask"]  # C2F‑Seg predicts amodal masks
        score = float(o["score"])
        detections.append({
            "bbox": bbox,
            "mask": (mask > 0.5).astype(np.uint8),
            "score": score
        })

    return detections