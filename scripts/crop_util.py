"""Detection cropping utility.

Shared by rerank_detections.py and crop_mosaic.py.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from flakefinder.segmentation import draw_scale_bar


def crop_detection(
    frame_path: Path,
    frame_json_path: Path,
    det_id: int,
    um_per_px: float,
    pad: int = 250,
    overlay: str = "contour",
) -> np.ndarray | None:
    """Crop a detection from a scan frame with overlay and scale bar.

    Loads the frame image and per-frame segmentation JSON, draws the requested
    overlay (contour, bbox, or none), crops around the detection bbox with
    padding, and adds a scale bar.

    Returns the cropped BGR image array, or None if files are missing.
    """
    if not frame_path.exists():
        print(f"  Warning: {frame_path} not found, skipping crop")
        return None
    if not frame_json_path.exists():
        print(f"  Warning: {frame_json_path} not found, skipping crop")
        return None

    with open(frame_json_path) as f:
        full_det = json.load(f)["detections"][det_id]

    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"  Warning: failed to read {frame_path}, skipping crop")
        return None
    h, w = img.shape[:2]

    bx, by, bw, bh = full_det["bbox"]

    if overlay == "contour":
        contour = full_det.get("contour")
        if contour and len(contour) >= 3:
            pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    elif overlay == "bbox":
        bp = 4
        cv2.rectangle(img, (bx - bp, by - bp), (bx + bw + bp, by + bh + bp), (0, 255, 0), 1)

    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(w, bx + bw + pad)
    y1 = min(h, by + bh + pad)
    crop = img[y0:y1, x0:x1]
    draw_scale_bar(crop, um_per_px)
    return crop
