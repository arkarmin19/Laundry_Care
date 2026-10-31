"""End-to-end pipeline: photo -> preprocess -> YOLO -> OCR -> rules -> care guide.

Two stages, so an app can show results progressively:
  1. detect_stage: preprocess + YOLO   (fast: ~0.1 s on a laptop CPU)
  2. ocr_stage:    EasyOCR             (slow on CPU: several seconds for Thai + English)
build() combines them with the rules. run() does everything in one call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .fabric import parse_care_notes, parse_fabric
from .preprocess import preprocess, resize_max
from .rules import CareGuide, Detection, build_care_guide

BLUR_THRESHOLD = 60.0


@dataclass
class ScanResult:
    image: np.ndarray            # preprocessed image the model saw
    detections: list[Detection]  # raw detector output
    ocr_text: str
    fabrics: dict
    guide: CareGuide
    info: dict = field(default_factory=dict)  # timings, blur, cropped


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class CareScanPipeline:
    def __init__(self, model_path: str, conf: float = 0.25, use_ocr: bool = True,
                 crop: bool = True, ocr_languages: tuple[str, ...] = ("th", "en")):
        from .detector import SymbolDetector

        self.detector = SymbolDetector(model_path, conf=conf)
        self.reader = None
        if use_ocr:
            from .ocr import LabelReader
            self.reader = LabelReader(ocr_languages)
        self.crop = crop
        self.conf = conf

    # ---- stage 1: symbols ----
    def detect_stage(self, img_bgr: np.ndarray) -> tuple[np.ndarray, list[Detection], dict]:
        t0 = time.perf_counter()
        img, info = preprocess(img_bgr, crop=self.crop)
        t_pre = _ms(t0)
        t1 = time.perf_counter()
        dets = self.detector.detect(img)
        info["timings_ms"] = {"preprocess": t_pre, "detect": _ms(t1)}
        return img, dets, info

    # ---- stage 2: text ----
    def ocr_stage(self, img: np.ndarray, ocr_max_side: int = 1280) -> tuple[str, int]:
        """OCR reads a copy shrunk to ocr_max_side. Smaller is faster, but small
        print (especially Thai) can be lost. Returns (text, milliseconds)."""
        if not self.reader:
            return "", 0
        t0 = time.perf_counter()
        text = self.reader.read(resize_max(img, ocr_max_side))
        return text, _ms(t0)

    # ---- rules ----
    def build(self, dets: list[Detection], text: str, info: dict,
              colour: str | None = None, is_new: bool = False) -> tuple[CareGuide, dict]:
        fabrics = parse_fabric(text)
        guide = build_care_guide(dets, fabrics, colour, is_new,
                                 parse_care_notes(text), min_conf=self.conf)
        if info.get("blur", 1e9) < BLUR_THRESHOLD:
            guide.tips.insert(0, "Photo looks blurry — retake it if results seem wrong.")
        return guide, fabrics

    def run(self, img_bgr: np.ndarray, colour: str | None = None,
            is_new: bool = False, ocr_max_side: int = 1280) -> ScanResult:
        t0 = time.perf_counter()
        img, dets, info = self.detect_stage(img_bgr)
        text, ocr_ms = self.ocr_stage(img, ocr_max_side)
        t_rules = time.perf_counter()
        guide, fabrics = self.build(dets, text, info, colour, is_new)
        info["ocr_max_side"] = ocr_max_side
        info["timings_ms"].update({"ocr": ocr_ms, "rules": _ms(t_rules), "total": _ms(t0)})
        return ScanResult(img, dets, text, fabrics, guide, info)


def guide_from_symbols(symbols: list[str], ocr_text: str = "", colour: str | None = None,
                       is_new: bool = False) -> CareGuide:
    """Demo / testing mode: build a care guide from symbol names typed by hand,
    without a trained model."""
    dets = [Detection(name=s) for s in symbols]
    return build_care_guide(dets, parse_fabric(ocr_text), colour, is_new,
                            parse_care_notes(ocr_text))
