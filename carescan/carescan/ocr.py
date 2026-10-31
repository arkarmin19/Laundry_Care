"""EasyOCR wrapper. Lazy import: EasyOCR downloads its models on first use.

Default languages are Thai + English (EasyOCR's Thai model also reads English).
To add another language, check EasyOCR's list of language codes and which
languages can be combined: e.g. ("ch_sim", "en") for Simplified Chinese.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LANGUAGES = ("th", "en")


class LabelReader:
    def __init__(self, languages: tuple[str, ...] = DEFAULT_LANGUAGES, gpu: bool = False):
        import easyocr  # lazy import

        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    def read(self, img_bgr: np.ndarray, min_conf: float = 0.3) -> str:
        results = self.reader.readtext(img_bgr[:, :, ::-1])  # EasyOCR expects RGB
        return " ".join(text for _, text, conf in results if conf >= min_conf)
