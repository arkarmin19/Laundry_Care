"""YOLO symbol detector (Ultralytics). Imported lazily so the rest of the
project (rules, tests, demo mode) works without ultralytics installed."""
from __future__ import annotations

import cv2
import numpy as np

from .rules import Detection


class SymbolDetector:
    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names  # {id: name}, comes from training data.yaml

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(img_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for box in res.boxes:
            cls = int(box.cls[0])
            out.append(Detection(
                name=self.names[cls],
                confidence=float(box.conf[0]),
                box=tuple(float(v) for v in box.xyxy[0].tolist()),
            ))
        return out


def draw_detections(img_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = img_bgr.copy()
    for d in detections:
        if d.box is None:
            continue
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 160, 60), 2)
        label = f"{d.name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (40, 160, 60), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out
