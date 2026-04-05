# tracking_inference.py

import sys
import os
import numpy as np

# Ensure current folder is on Python path
sys.path.append(os.path.dirname(__file__))

# Import Tracker class from tracking_tracker.py
from tracking_tracker import Tracker

# Example function to initialize tracker
def initialize_tracker(config):
    """
    Initialize a Tracker object with a configuration dict.
    Example config: {"model": "oc_sort"} or {"model": "bytetrack"}
    """
    tracker = Tracker(config)
    return tracker

# Example function to run tracking on a set of boxes/scores
def run_tracking(tracker, boxes, scores):
    """
    Updates the tracker with bounding boxes and confidence scores.
    - boxes: list of [x1, y1, x2, y2]
    - scores: list of floats
    Returns a list of tracked boxes with track IDs: [x1, y1, x2, y2, track_id]
    """
    results = tracker.update(boxes, scores)
    return results