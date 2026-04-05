import os
import numpy as np
import torch

# If you have a real C2F-Seg model, you can import it here
# from src.image_model import C2F_Seg

def run_segmentation(config):
    """
    Runs segmentation on all frames in input_frames.
    If no checkpoint is found, returns dummy masks and boxes.
    """
    input_dir = config["data"]["input_frames"]
    output_dir = os.path.join(config["data"]["output_dir"], "segmentation")
    os.makedirs(output_dir, exist_ok=True)

    # Check for checkpoint
    g_path = config["segmentation"].get("checkpoint", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if g_path is None or not os.path.exists(g_path):
        print("WARNING: No checkpoint found - running dummy segmentation")

        # Dummy model that returns one empty detection per image
        class DummyModel:
            def eval(self): pass
            def to(self, device): return self
            def __call__(self, x):
                batch_size = len(x)
                dummy_output = []
                for _ in range(batch_size):
                    dummy_output.append([
                        {
                            "bbox": [0, 0, 0, 0],
                            "amodal_mask": np.zeros((x[0].shape[0], x[0].shape[1]), dtype=np.uint8),
                            "score": 0.0
                        }
                    ])
                return dummy_output

        model = DummyModel()
    else:
        from src.image_model import C2F_Seg
        model = C2F_Seg(g_path, mode="test")
        model.eval()
        model = model.to(device)

    # Load images
    image_paths = sorted(os.listdir(input_dir))
    results = []

    for img_name in image_paths:
        img_path = os.path.join(input_dir, img_name)
        image = np.zeros((480, 640, 3), dtype=np.uint8)  # default dummy if image read fails
        try:
            import cv2
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Warning: could not read {img_name}, using black image")
                image = np.zeros((480, 640, 3), dtype=np.uint8)
        except Exception:
            pass

        # Wrap image in list for batch compatibility
        preds_batch = model([image])
        preds = preds_batch[0]  # single image

        detections = []
        for p in preds:
            mask = p["amodal_mask"]
            bbox = p["bbox"]
            score = float(p["score"])
            detections.append({
                "bbox": bbox,
                "mask": mask.astype(np.uint8),
                "score": score
            })

        results.append({
            "frame": img_name,
            "image": image,
            "detections": detections
        })

    return results