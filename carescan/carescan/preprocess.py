"""OpenCV preprocessing: resize, find/crop the label, boost contrast (CLAHE)."""
from __future__ import annotations

import cv2
import numpy as np


def resize_max(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast boost on the lightness channel only, so colours don't shift."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
                    dtype=np.float32)


def find_label(img: np.ndarray, min_area_frac: float = 0.15) -> np.ndarray | None:
    """Find the biggest 4-cornered shape (the label) and return its corners, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * img.shape[0] * img.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32))
    return None


def warp_label(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-correct the label so it is flat and upright."""
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (w, h))


def blur_score(img: np.ndarray) -> float:
    """Variance of the Laplacian: low values (< ~60) mean a blurry photo."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def preprocess(img: np.ndarray, crop: bool = True) -> tuple[np.ndarray, dict]:
    """Full preprocessing. Returns (image, info)."""
    img = resize_max(img)
    info = {"cropped": False, "blur": blur_score(img)}
    if crop:
        corners = find_label(img)
        if corners is not None:
            img = warp_label(img, corners)
            info["cropped"] = True
    return clahe(img), info
