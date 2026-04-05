import numpy as np
import torch
import cv2
from depth import depth_model

def run_depth(config, seg_outputs):
    model = depth_model(config["depth"])

    updated_results = []

    for frame_data in seg_outputs:
        image = frame_data["image"]

        # Preprocess
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

        if torch.cuda.is_available():
            img = img.cuda()

        with torch.no_grad():
            depth_map = model(img)[0, 0].cpu().numpy()

        detections = frame_data["detections"]

        for det in detections:
            mask = det["mask"].astype(bool)

            depth_vals = depth_map[mask]
            det["depth"] = float(depth_vals.mean()) if len(depth_vals) > 0 else 999

        # Sort: smaller depth = closer
        detections.sort(key=lambda x: x["depth"])

        for i, det in enumerate(detections):
            det["depth_rank"] = i

        updated_results.append({
            "frame": frame_data["frame"],
            "image": image,
            "detections": detections
        })

    return updated_results