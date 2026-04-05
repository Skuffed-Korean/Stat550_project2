import numpy as np
import sys

class Tracker:
    def __init__(self, config):
        self.type = config["model"]

        if self.type == "oc_sort":
            sys.path.append("Z:\DS_capstone\project2\external\ocsort")
            from trackers.ocsort_tracker.ocsort import OCSort
            self.tracker = OCSort()

        elif self.type == "bytetrack":
            sys.path.append("Z:\DS_capstone\project2\external\bytetrack")
            from yolox.tracker.byte_tracker import BYTETracker

            class Args:
                track_thresh = 0.5
                match_thresh = 0.8
                track_buffer = 30

            self.tracker = BYTETracker(Args())

    def update(self, boxes, scores):
        if len(boxes) == 0:
            return []

        dets = []
        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = b
            dets.append([x1, y1, x2, y2, s])

        dets = np.array(dets)

        tracks = self.tracker.update(dets)

        results = []
        for t in tracks:
            x1, y1, x2, y2, track_id = t[:5]
            results.append([int(x1), int(y1), int(x2), int(y2), int(track_id)])

        return results