import os
import numpy as np
import cv2

# =========================
# Dummy Segmentation
# =========================
def run_segmentation(config):
    input_dir = config["data"]["input_frames"]
    results = []

    # Make output folder
    output_dir = os.path.join(config["data"]["output_dir"], "segmentation")
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(os.listdir(input_dir))
    for img_name in image_paths:
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            # fallback black image
            image = np.zeros((480, 640, 3), dtype=np.uint8)

        # dummy segmentation: single box covering the center
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = image.shape[:2]
        mask[h//4:h*3//4, w//4:w*3//4] = 1
        bbox = [w//4, h//4, w*3//4, h*3//4]

        results.append({
            "frame": img_name,
            "image": image,
            "detections": [{
                "bbox": bbox,
                "mask": mask,
                "score": 1.0
            }]
        })

    return results

# =========================
# Dummy Depth
# =========================
def run_depth(seg_outputs, config):
    print("Running dummy depth estimation...")
    for frame in seg_outputs:
        h, w = frame["image"].shape[:2]
        frame["depth"] = np.zeros((h, w), dtype=np.float32)  # all zeros
    return seg_outputs

# =========================
# Dummy Tracking
# =========================
def run_tracking(depth_outputs, config):
    print("Running dummy tracking...")
    track_id = 1
    for frame in depth_outputs:
        # Assign all detections a track id
        tracks = []
        for det in frame["detections"]:
            tracks.append(det["bbox"] + [track_id])
            track_id += 1
        frame["tracks"] = tracks
    return depth_outputs

# =========================
# Main pipeline
# =========================
def main():
    # Dummy config dictionary
    config = {
        "data": {
            "input_frames": r"C:\Users\19109\Desktop\images",  # change to your folder
            "output_dir": r"Z:\DS_capstone\project2\output"
        },
        "segmentation": {"checkpoint": None},
        "depth": {"model": None},
        "tracking": {"model": "oc_sort"}
    }

    print("=== STARTING DUMMY PIPELINE ===")

    # 1. Segmentation
    seg_outputs = run_segmentation(config)
    print(f"Segmentation complete: {len(seg_outputs)} frames")

    # 2. Depth
    depth_outputs = run_depth(seg_outputs, config)
    print("Depth estimation complete")

    # 3. Tracking
    track_outputs = run_tracking(depth_outputs, config)
    print("Tracking complete")

    # 4. Show sample output
    print("\n=== SAMPLE OUTPUT ===")
    for i in range(min(3, len(track_outputs))):
        print(f"Frame {i}: {track_outputs[i]['tracks']}")

    print("\n Pipeline ran successfully!")

if __name__ == "__main__":
    main()