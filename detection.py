# detection.py
import os
import csv
from datetime import datetime
from typing import List, Tuple

import cv2
from ultralytics import YOLO

from anomaly_model import AnomalyDetector

# -------- setup --------
model = YOLO("yolov8n.pt")  # ensure weights exist
detector = AnomalyDetector()

ALERT_DIR = "outputs"
ALERT_CSV = os.path.join(ALERT_DIR, "alerts.csv")
os.makedirs(ALERT_DIR, exist_ok=True)

# CSV header if not exists
if not os.path.exists(ALERT_CSV):
    with open(ALERT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "type", "label", "track_id", "since_s", "image"])

COCO_ALLOWED = {"person", "handbag", "backpack", "suitcase"}  # speed-up

def to_xyxy(bxyxy) -> Tuple[float,float,float,float]:
    return float(bxyxy[0]), float(bxyxy[1]), float(bxyxy[2]), float(bxyxy[3])

def detect_objects(video_path: str, display: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, verbose=False)
        r = results[0]

        # Build detection list (cls_name, conf, (x1,y1,x2,y2))
        dets = []
        for b in r.boxes:
            cls_idx = int(b.cls)
            cls_name = r.names[cls_idx]
            if cls_name not in COCO_ALLOWED:
                continue
            conf = float(b.conf)
            box = to_xyxy(b.xyxy[0].tolist())
            dets.append((cls_name, conf, box))

        # Anomaly step
        alerts = detector.step(dets)

        # Annotate frame
        annotated = r.plot()
        for a in alerts:
            x1,y1,x2,y2 = map(int, a["box"])
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(annotated, f'{a["type"]}', (x1, max(20, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Save & log if any alert
        if alerts:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_path = os.path.join(ALERT_DIR, f"alert_{ts}.jpg")
            cv2.imwrite(img_path, annotated)

            with open(ALERT_CSV, "a", newline="") as f:
                w = csv.writer(f)
                for a in alerts:
                    w.writerow([
                        ts,
                        a["type"],
                        a.get("label", "person"),
                        a["track_id"],
                        a["since_s"],
                        os.path.basename(img_path),
                    ])
            print(f"🚨 {len(alerts)} alert(s) logged -> {img_path}")

        if display:
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects("data/testing_videos/01.avi", display=True)
