# anomaly_model.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import time

# --------------------------
# Simple utilities
# --------------------------
def iou(boxA, boxB) -> float:
    # boxes: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def center_of(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )

def euclid(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

# --------------------------
# Track entities
# --------------------------
@dataclass
class Track:
    id: int
    cls: str
    box: Tuple[float,float,float,float]
    conf: float
    first_time: float
    last_time: float
    first_center: Tuple[float,float] = field(default_factory=lambda:(0,0))
    alerted: bool = False  # prevent duplicate alerts

    def update(self, box, conf, t):
        self.box = box
        self.conf = conf
        self.last_time = t

# --------------------------
# Main anomaly detector
# --------------------------
class AnomalyDetector:
    """
    Very light-weight tracker + rules:
      - Loitering: person stays within small radius for > LOITER_TIME_S
      - Abandoned object: handbag has no nearby person for > ABANDON_TIME_S
    Notes:
      * Works on YOLOv8 results (boxes.xyxy, cls names).
      * Designed for single camera per process (keeps in-memory tracks).
    """
    # Tunables (you can tweak)
    MATCH_IOU_THR = 0.3
    LOITER_TIME_S = 30.0
    LOITER_RADIUS_PX = 40.0
    ABANDON_TIME_S = 20.0
    NEAR_PERSON_DIST_PX = 80.0
    MAX_TRACK_AGE_S = 5.0  # drop tracks if not seen recently

    PERSON_LABELS = {"person"}
    OBJECT_LABELS = {"handbag", "backpack", "suitcase"}  # extend as needed

    def __init__(self):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    # --------------- public API ---------------
    def step(self, detections: List[Tuple[str, float, Tuple[float,float,float,float]]],
             now_s: Optional[float] = None) -> List[dict]:
        """
        detections: list of (cls_name, conf, (x1,y1,x2,y2))
        returns: list of alerts dicts
        """
        t = time.time() if now_s is None else now_s
        self._associate(detections, t)
        self._purge_stale(t)
        alerts = []
        alerts += self._check_loitering(t)
        alerts += self._check_abandoned(t)
        return alerts

    # --------------- internals ---------------
    def _associate(self, dets, t):
        # Greedy IoU matching to existing tracks
        unmatched = set(range(len(dets)))
        # try matching per class for stability
        for tid, tr in list(self.tracks.items()):
            best_j, best_score = -1, 0.0
            for j in unmatched:
                cls, conf, box = dets[j]
                if cls != tr.cls:
                    continue
                score = iou(tr.box, box)
                if score > best_score:
                    best_score, best_j = score, j
            if best_j != -1 and best_score >= self.MATCH_IOU_THR:
                # update
                cls, conf, box = dets[best_j]
                tr.update(box, conf, t)
                if tr.first_center == (0,0):
                    tr.first_center = center_of(box)
                unmatched.remove(best_j)

        # create tracks for the rest
        for j in unmatched:
            cls, conf, box = dets[j]
            tid = self.next_id
            self.next_id += 1
            c = center_of(box)
            self.tracks[tid] = Track(
                id=tid, cls=cls, box=box, conf=conf,
                first_time=t, last_time=t, first_center=c
            )

    def _purge_stale(self, t):
        stale = [tid for tid, tr in self.tracks.items()
                 if (t - tr.last_time) > self.MAX_TRACK_AGE_S]
        for tid in stale:
            del self.tracks[tid]

    def _check_loitering(self, t) -> List[dict]:
        alerts = []
        for tr in self.tracks.values():
            if tr.alerted:
                continue
            if tr.cls not in self.PERSON_LABELS:
                continue
            dur = t - tr.first_time
            if dur < self.LOITER_TIME_S:
                continue
            # has the person mostly stayed nearby?
            cur_center = center_of(tr.box)
            disp = euclid(cur_center, tr.first_center)
            if disp <= self.LOITER_RADIUS_PX:
                tr.alerted = True
                alerts.append({
                    "type": "loitering",
                    "track_id": tr.id,
                    "since_s": round(dur, 1),
                    "box": tr.box,
                    "conf": tr.conf
                })
        return alerts

    def _check_abandoned(self, t) -> List[dict]:
        alerts = []
        # Build a quick list of current person centers
        people = [center_of(tr.box) for tr in self.tracks.values()
                  if tr.cls in self.PERSON_LABELS and (t - tr.last_time) < 1.0]
        for tr in self.tracks.values():
            if tr.alerted:
                continue
            if tr.cls not in self.OBJECT_LABELS:
                continue
            # Is any person near the object?
            c = center_of(tr.box)
            near = any(euclid(c, pc) <= self.NEAR_PERSON_DIST_PX for pc in people)
            if near:
                # reset first_time so timer restarts while a person is close
                tr.first_time = t if tr.first_center == (0,0) else tr.first_time
                continue
            # nobody near -> consider timer
            dur = t - tr.first_time
            if dur >= self.ABANDON_TIME_S:
                tr.alerted = True
                alerts.append({
                    "type": "abandoned_object",
                    "track_id": tr.id,
                    "since_s": round(dur, 1),
                    "box": tr.box,
                    "conf": tr.conf,
                    "label": tr.cls
                })
        return alerts
