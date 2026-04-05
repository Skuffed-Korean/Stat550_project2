import torch
import sys

sys.path.append("Z:\DS_capstone\project2\external\depth_anything")

from depth_anything_v2.dpt import DepthAnythingV2

def load_model(config):
    model = DepthAnythingV2.from_pretrained(config["weights"])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model