"""Material-specific detector configuration and flake segmentation.

Merges detector configuration, frame segmentation, and parallel worker
logic into a single package module.  Provides the core detection pipeline
used by scripts/segment_flakes.py, scripts/segment_chip_scan.py, and
find_flakes.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple, TypedDict

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import kurtosis as scipy_kurtosis

from flakefinder.scan_utils import apply_flatfield
from flakefinder.types import ContrastRGB, PixelPolygon, Point2F, XYWHRect

# ============================================================================
# Detection types
# ============================================================================


class _DetectionBase(TypedDict):
    bbox: XYWHRect
    center: Point2F
    size_px: int
    mean_contrast: float
    contrast_rgb: ContrastRGB
    cal_dist: float
    solidity: float
    circularity: float
    perim_ratio: float
    aspect_ratio: float
    r_std: float
    r_kurt: float
    g_std: float
    g_kurt: float
    b_std: float
    b_kurt: float
    grad_energy: float
    r_entropy: float
    g_entropy: float
    b_entropy: float
    entropy: float
    size_um2: float
    hull: PixelPolygon
    contour: PixelPolygon


class Detection(_DetectionBase, total=False):
    """Flake detection with optional classification fields.

    Base fields from _analyze_component. Optional fields added
    in-place by classify_detections/score_detections.
    """

    classification: str | None
    tier: int
    score: float
    frame: str
    stage_x: float
    stage_y: float
    det_idx: int


# ============================================================================
# Detector configuration (from scripts/detector_config.py)
# ============================================================================


class ContrastMode(Enum):
    """Direction of flake-to-substrate contrast."""

    ABOVE = "above"  # flakes brighter than substrate (hBN)
    BELOW = "below"  # flakes darker than substrate (graphene)


@dataclass
class DetectorConfig:
    """All material-specific parameters for flake detection."""

    # -- Segmentation --
    contrast_mode: ContrastMode
    contrast_offset: float
    min_size_um2: float
    edge_margin_px: int
    morph_kernel_size: int

    # -- Calibration curve: R = poly(G) --
    cal_poly: tuple[float, ...]
    cal_g_range: tuple[float, float]

    # -- Classification thresholds --
    cal_dist_match: float  # max distance for thin/medium/thick
    cal_dist_possible: float  # max distance for "possible"
    g_thin_max: float  # G < this -> thin
    g_medium_max: float  # G <= this -> medium, else thick
    non_match_label: str  # label for detections far from cal curve

    # -- Scoring tier gates --
    tier1_perim_ratio: float
    tier1_cal_dist: float
    tier1_g_min: float
    tier1_g_max: float
    tier1_r_max: float
    tier1_entropy_max: float
    tier2_perim_ratio: float
    tier2_cal_dist: float
    tier2_entropy_max: float

    # Precomputed calibration curve
    _cal_g_curve: np.ndarray = field(init=False, repr=False, compare=False)
    _cal_r_curve: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        g = np.linspace(self.cal_g_range[0], self.cal_g_range[1], 500)
        self._cal_g_curve = g
        self._cal_r_curve = np.polyval(self.cal_poly, g)

    def cal_distance(self, r: float, g: float) -> float:
        """Minimum distance from (g, r) to the calibration curve."""
        return float(np.sqrt((self._cal_g_curve - g) ** 2 + (self._cal_r_curve - r) ** 2).min())

    def classify(self, r: float, g: float) -> str:
        """Classify by R-G calibration distance and G contrast."""
        d = self.cal_distance(r, g)
        if d < self.cal_dist_match:
            if g < self.g_thin_max:
                return "thin"
            elif g <= self.g_medium_max:
                return "medium"
            else:
                return "thick"
        elif d < self.cal_dist_possible:
            return "possible"
        return self.non_match_label

    def score_detection(self, det: Detection) -> tuple[int, float]:
        """Compute (tier, score) for a detection dict."""
        pr = det["perim_ratio"]
        cd = det["cal_dist"]
        g = det["contrast_rgb"][1]
        ent = det.get("entropy", det.get("g_entropy", 99.0))
        r = det["contrast_rgb"][0]
        ar = det.get("aspect_ratio", 1.0)
        if (
            pr < self.tier1_perim_ratio
            and cd < self.tier1_cal_dist
            and g >= self.tier1_g_min
            and g < self.tier1_g_max
            and r < self.tier1_r_max
            and ent < self.tier1_entropy_max
            and ar < 6.0
        ):
            tier = 1
        elif pr < self.tier2_perim_ratio and cd < self.tier2_cal_dist and ent < self.tier2_entropy_max:
            tier = 2
        else:
            tier = 3
        ge = det.get("grad_energy", 0)
        size_um2 = det.get("size_um2", det["size_px"] * 0.52)
        ar_penalty = float(np.exp(-(max(ar - 3, 0) ** 2) / 8))
        g_penalty = 1.0 / (1.0 + 2.0 * max(g - 0.5, 0))
        score = round(
            np.log2(max(size_um2, 1.0)) * np.exp(-cd * 8) * (1.0 / (1.0 + ge)) * ar_penalty * g_penalty,
            4,
        )
        return tier, score

    @classmethod
    def hbn(cls) -> DetectorConfig:
        """hBN detection preset (alias for hbn_thin)."""
        return cls.hbn_thin()

    @classmethod
    def hbn_thin(cls) -> DetectorConfig:
        """hBN thin flake detection preset."""
        return cls(
            contrast_mode=ContrastMode.ABOVE,
            contrast_offset=15.0,
            min_size_um2=400.0,
            edge_margin_px=50,
            morph_kernel_size=5,
            cal_poly=(0.193, -0.217, -0.604),
            cal_g_range=(-0.5, 6.0),
            cal_dist_match=0.5,
            cal_dist_possible=1.0,
            g_thin_max=1.0,
            g_medium_max=2.5,
            non_match_label="non-hBN",
            tier1_perim_ratio=1.50,
            tier1_cal_dist=0.3,
            tier1_g_min=0.0,
            tier1_g_max=1.2,
            tier1_r_max=-0.5,
            tier1_entropy_max=99.0,
            tier2_perim_ratio=1.35,
            tier2_cal_dist=0.3,
            tier2_entropy_max=4.5,
        )

    @classmethod
    def hbn_thick(cls) -> DetectorConfig:
        """hBN thick flake detection preset."""
        return cls(
            contrast_mode=ContrastMode.ABOVE,
            contrast_offset=15.0,
            min_size_um2=400.0,
            edge_margin_px=50,
            morph_kernel_size=5,
            cal_poly=(0.193, -0.217, -0.604),
            cal_g_range=(-0.5, 6.0),
            cal_dist_match=0.5,
            cal_dist_possible=1.0,
            g_thin_max=1.0,
            g_medium_max=2.5,
            non_match_label="non-hBN",
            tier1_perim_ratio=1.20,
            tier1_cal_dist=0.3,
            tier1_g_min=-99.0,
            tier1_g_max=99.0,
            tier1_r_max=99.0,
            tier1_entropy_max=4.5,
            tier2_perim_ratio=1.35,
            tier2_cal_dist=0.3,
            tier2_entropy_max=5.5,
        )

    @classmethod
    def graphene(cls) -> DetectorConfig:
        """Graphene detection preset (stub -- no calibration curve yet)."""
        return cls(
            contrast_mode=ContrastMode.BELOW,
            contrast_offset=10.0,
            min_size_um2=130.0,
            edge_margin_px=50,
            morph_kernel_size=5,
            cal_poly=(0.0, 0.0, 0.0),
            cal_g_range=(-6.0, 0.5),
            cal_dist_match=0.5,
            cal_dist_possible=1.0,
            g_thin_max=-1.0,
            g_medium_max=-2.5,
            non_match_label="non-graphene",
            tier1_perim_ratio=1.20,
            tier1_cal_dist=0.3,
            tier1_g_min=-99.0,
            tier1_g_max=4.0,
            tier1_r_max=99.0,
            tier1_entropy_max=99.0,
            tier2_perim_ratio=1.35,
            tier2_cal_dist=0.3,
            tier2_entropy_max=99.0,
        )

    @classmethod
    def from_material(cls, name: str) -> DetectorConfig:
        """Create config from material name."""
        presets = {"hbn": cls.hbn, "hbn_thin": cls.hbn_thin, "hbn_thick": cls.hbn_thick, "graphene": cls.graphene}
        if name not in presets:
            raise ValueError(f"Unknown material: {name!r}. Choose from: {', '.join(presets)}")
        return presets[name]()


# ============================================================================
# Segmentation functions (from scripts/segment_flakes.py)
# ============================================================================


def histogram_mode(image: np.ndarray, channel: int | None = None) -> float:
    """Find the histogram peak (mode) of an image or single channel."""
    if channel is not None:
        data = image[:, :, channel]
    else:
        data = image
    hist, _ = np.histogram(data.ravel(), bins=256, range=(0, 256))
    hist[:20] = 0
    hist[230:] = 0
    return float(np.argmax(hist))


def compute_dark_frac(image: np.ndarray, threshold: float = 30.0) -> float:
    """Fraction of pixels with mean brightness below threshold (off-chip indicator)."""
    return float((image.mean(axis=2) < threshold).mean())


def _analyze_component(
    image: np.ndarray,
    component: np.ndarray,
    bg_modes: np.ndarray,
    norm_contrast: np.ndarray,
    grad_mag: np.ndarray,
    config: DetectorConfig,
    um_per_px: float = 1.0,
) -> Detection:
    """Analyze a binary component mask and return detection metrics."""
    rows = np.any(component, axis=1)
    cols = np.any(component, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = int(y_indices[0]), int(y_indices[-1])
    x_min, x_max = int(x_indices[0]), int(x_indices[-1])

    region_pixels = image[component].astype(np.float32)
    mean_contrast = float(np.mean(region_pixels - bg_modes))

    ch_means = region_pixels.mean(axis=0)  # BGR
    norm_contrast_bgr = (ch_means - bg_modes) / np.maximum(bg_modes, 1.0)

    # Shape metrics via contour analysis
    comp_u8 = component.astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea) if contours else None
    if cnt is not None:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perim = cv2.arcLength(hull, True)
        solidity = area / max(hull_area, 1)
        circularity = (4 * np.pi * area) / max(perim**2, 1)
        perim_ratio = perim / max(hull_perim, 1.0)
        hull_pts = hull.reshape(-1, 2).tolist()
        contour_pts = cnt.reshape(-1, 2).tolist()
        _, (bw, bh), _ = cv2.minAreaRect(cnt)
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1e-6)
    else:
        solidity = 0.0
        circularity = 0.0
        perim_ratio = 0.0
        aspect_ratio = 1.0
        hull_pts = []
        contour_pts = []

    # Color uniformity: std of per-pixel normalized contrast within blob (BGR channel order)
    b_vals = norm_contrast[:, :, 0][component]
    b_std = float(b_vals.std())
    b_kurt = float(scipy_kurtosis(b_vals, fisher=True))
    g_vals = norm_contrast[:, :, 1][component]
    g_std = float(g_vals.std())
    g_kurt = float(scipy_kurtosis(g_vals, fisher=True))
    r_vals = norm_contrast[:, :, 2][component]
    r_std = float(r_vals.std())
    r_kurt = float(scipy_kurtosis(r_vals, fisher=True))

    # Internal gradient energy (Sobel on G channel, masked to blob)
    grad_energy = float(grad_mag[component].mean())

    # Histogram entropy of per-channel normalized contrast within blob
    # Fixed range (-1, 1) so homogeneous blobs → low entropy, heterogeneous → high
    def _hist_entropy(vals: np.ndarray) -> float:
        hist, _ = np.histogram(vals, bins=50, range=(-1.0, 1.0))
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs)))

    r_entropy = _hist_entropy(r_vals)
    g_entropy = _hist_entropy(g_vals)
    b_entropy = _hist_entropy(b_vals)

    r_contrast = round(float(norm_contrast_bgr[2]), 4)
    g_contrast = round(float(norm_contrast_bgr[1]), 4)
    b_contrast = round(float(norm_contrast_bgr[0]), 4)

    return Detection(
        bbox=XYWHRect(x_min, y_min, x_max - x_min, y_max - y_min),
        center=Point2F(round((x_min + x_max) / 2, 1), round((y_min + y_max) / 2, 1)),
        size_px=int(component.sum()),
        size_um2=round(int(component.sum()) * um_per_px**2, 1),
        mean_contrast=round(mean_contrast, 1),
        contrast_rgb=ContrastRGB(r_contrast, g_contrast, b_contrast),
        cal_dist=round(config.cal_distance(r_contrast, g_contrast), 4),
        solidity=round(solidity, 4),
        circularity=round(circularity, 4),
        perim_ratio=round(perim_ratio, 4),
        aspect_ratio=round(aspect_ratio, 4),
        r_std=round(r_std, 4),
        r_kurt=round(r_kurt, 4),
        g_std=round(g_std, 4),
        g_kurt=round(g_kurt, 4),
        b_std=round(b_std, 4),
        b_kurt=round(b_kurt, 4),
        grad_energy=round(grad_energy, 2),
        r_entropy=round(r_entropy, 4),
        g_entropy=round(g_entropy, 4),
        b_entropy=round(b_entropy, 4),
        entropy=round(max(r_entropy, g_entropy, b_entropy), 4),
        hull=hull_pts,
        contour=contour_pts,
    )


def _otsu_split(
    contrast_channels: np.ndarray,
    component: np.ndarray,
    min_size_px: int,
) -> list[np.ndarray] | None:
    """Try one Otsu split on the highest-variance channel. Returns sub-components or None."""
    # Crop to component bounding box -- all ops run on the small ROI
    rows = np.any(component, axis=1)
    cols = np.any(component, axis=0)
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    sl = (slice(y_idx[0], y_idx[-1] + 1), slice(x_idx[0], x_idx[-1] + 1))
    roi_comp = component[sl]
    roi_cc = contrast_channels[sl]

    # Pick the channel with highest within-blob variance
    stds = [roi_cc[:, :, c][roi_comp].std() for c in range(3)]
    best_ch = int(np.argmax(stds))
    roi_ch = roi_cc[:, :, best_ch]
    blob_vals = roi_ch[roi_comp]

    if blob_vals.std() < 0.8:
        return None

    v_min, v_max = blob_vals.min(), blob_vals.max()
    if v_max - v_min < 0.5:
        return None

    scaled = ((blob_vals - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = v_min + (thresh_val / 255) * (v_max - v_min)

    sub_components = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    comp_size = int(roi_comp.sum())
    for is_high in [False, True]:
        roi_sub = roi_comp & ((roi_ch >= thresh) if is_high else (roi_ch < thresh))
        roi_clean = cv2.morphologyEx(
            roi_sub.astype(np.uint8), cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        roi_labels, n_sub = ndimage.label(roi_clean)

        for j in range(1, n_sub + 1):
            roi_piece = roi_labels == j
            if int(roi_piece.sum()) >= min_size_px // 2:
                # Expand back to full frame
                full = np.zeros(component.shape, dtype=bool)
                full[sl] = roi_piece
                sub_components.append(full)

    if len(sub_components) >= 2:
        return sub_components
    if len(sub_components) == 1 and int(sub_components[0].sum()) < comp_size * 0.95:
        return sub_components
    return None


def _subsegment_by_contrast(
    image: np.ndarray,
    component: np.ndarray,
    bg_modes: np.ndarray,
    min_size_px: int,
    norm_contrast: np.ndarray,
    max_depth: int = 3,
) -> list[np.ndarray]:
    """Iteratively split a blob using Otsu on the highest-variance color channel.

    Each split produces sub-components that are checked again for further
    splitting, up to *max_depth* levels. This handles cases where extreme
    outlier pixels pull the first Otsu threshold away from subtler gradients.
    """
    # Iterative: keep a work queue of components to try splitting
    final = []
    queue = [(component, 0)]

    while queue:
        comp, depth = queue.pop()
        if depth >= max_depth:
            final.append(comp)
            continue

        pieces = _otsu_split(norm_contrast, comp, min_size_px)
        if pieces is None:
            final.append(comp)
        else:
            for piece in pieces:
                queue.append((piece, depth + 1))

    return final if final else [component]


def segment_frame(
    image: np.ndarray,
    config: DetectorConfig,
    um_per_px: float,
    perim_ratio_thresh: float = 0.0,
) -> list[Detection]:
    """Segment flakes by thresholding relative to background mode.

    For 'above' contrast mode (hBN), finds pixels brighter than background.
    For 'below' contrast mode (graphene), finds pixels darker than background.
    Returns list of detected regions with bbox, size, center, mean_contrast.
    """
    min_size_px = int(config.min_size_um2 / (um_per_px**2))

    bg_modes = np.array([histogram_mode(image, c) for c in range(3)])

    above = image.astype(np.float32) - bg_modes[np.newaxis, np.newaxis, :]
    if config.contrast_mode == ContrastMode.ABOVE:
        mask = np.any(above > config.contrast_offset, axis=2)
    else:
        mask = np.any(above < -config.contrast_offset, axis=2)
    # Reuse above buffer for normalized contrast (used by subsegment + analyze)
    norm_contrast = above
    norm_contrast /= np.maximum(bg_modes[np.newaxis, np.newaxis, :], 1.0)

    # Precompute gradient magnitude on G channel (used by analyze)
    gray_g = image[:, :, 1].astype(np.float32)
    sx = cv2.Sobel(gray_g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx * sx + sy * sy)

    ks = config.morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    labels, n_labels = ndimage.label(mask_clean)
    component_sizes = ndimage.sum(mask_clean, labels, range(1, n_labels + 1))
    component_slices = ndimage.find_objects(labels)

    h, w = image.shape[:2]
    detections = []

    for i in range(n_labels):
        if component_slices[i] is None or component_sizes[i] < min_size_px:
            continue

        sl = component_slices[i]
        cx = (sl[1].start + sl[1].stop) / 2
        cy = (sl[0].start + sl[0].stop) / 2

        if cx < config.edge_margin_px or cx > w - config.edge_margin_px:
            continue
        if cy < config.edge_margin_px or cy > h - config.edge_margin_px:
            continue

        component = labels == (i + 1)

        # Try contrast-based sub-segmentation for large blobs
        sub_components = _subsegment_by_contrast(image, component, bg_modes, min_size_px, norm_contrast)

        for sub_comp in sub_components:
            det = _analyze_component(image, sub_comp, bg_modes, norm_contrast, grad_mag, config, um_per_px)
            # Re-check edge margin for sub-components
            scx, scy = det["center"]
            if scx < config.edge_margin_px or scx > w - config.edge_margin_px:
                continue
            if scy < config.edge_margin_px or scy > h - config.edge_margin_px:
                continue
            detections.append(det)

    detections.sort(key=lambda d: d["size_px"], reverse=True)
    classify_detections(detections, config, perim_ratio_thresh=perim_ratio_thresh)
    return detections


def score_detections(detections: list[Detection], config: DetectorConfig) -> None:
    """Compute tier and score for detections in place.

    Reads existing keys (perim_ratio, cal_dist, contrast_rgb, size_px).
    Adds keys: tier, score.
    """
    for det in detections:
        det["tier"], det["score"] = config.score_detection(det)


def classify_detections(
    detections: list[Detection],
    config: DetectorConfig,
    perim_ratio_thresh: float = 0.0,
) -> None:
    """Classify detections and compute ranking scores in place.

    Adds keys: classification, tier, score.
    """
    if not detections:
        return

    # Tape classification
    for det in detections:
        if perim_ratio_thresh > 0 and det.get("perim_ratio", 0) >= perim_ratio_thresh:
            det["classification"] = "tape"
        else:
            det["classification"] = None

    score_detections(detections, config)


# ============================================================================
# Visualization (from scripts/segment_flakes.py)
# ============================================================================


def draw_scale_bar(image: np.ndarray, um_per_px: float) -> None:
    """Draw a scale bar in the bottom-right corner of *image* (mutates in place)."""
    h, w = image.shape[:2]
    for candidate_um in [500, 200, 100, 50, 20, 10]:
        candidate_px = int(candidate_um / um_per_px)
        if candidate_px <= w * 0.20:
            break
    bar_px = int(candidate_um / um_per_px)
    bar_h = max(4, h // 200)
    margin = max(15, h // 60)
    bx = w - margin - bar_px
    by = h - margin - bar_h
    cv2.rectangle(image, (bx - 1, by - 1), (bx + bar_px + 1, by + bar_h + 1), (0, 0, 0), -1)
    cv2.rectangle(image, (bx, by), (bx + bar_px, by + bar_h), (255, 255, 255), -1)
    label = f"{candidate_um} um"
    font_scale = max(0.4, h / 2000)
    thick = max(1, int(h / 800))
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    lx = bx + (bar_px - lw) // 2
    ly = by - max(4, int(h / 200))
    cv2.putText(image, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 2)
    cv2.putText(image, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    um_per_px: float = 0.0,
    draw_bbox: bool = True,
    draw_hull: bool = False,
    draw_contour: bool = True,
) -> np.ndarray:
    """Draw detection overlays on image. Returns a copy."""
    vis = image.copy()
    for i, d in enumerate(detections):
        s = d["size_px"]
        c = d.get("mean_contrast", 0)
        is_tape = d.get("classification") == "tape"
        if is_tape:
            color = (128, 128, 128)
            thickness = 1
        else:
            color = (0, 255, 0) if s > 1000 else (0, 255, 255) if s > 500 else (0, 0, 255)
            thickness = 2 if s > 1000 else 1

        bx, by, bw, bh = d["bbox"]
        if draw_bbox:
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, thickness)

        hull = d.get("hull")
        if draw_hull and hull and len(hull) >= 3:
            pts = np.array(hull, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        contour = d.get("contour")
        if draw_contour and contour and len(contour) >= 3:
            pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        label = f"#{i} {s}px c={c:.0f}"
        cv2.putText(vis, label, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, label, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if um_per_px > 0:
        draw_scale_bar(vis, um_per_px)
    return vis


def save_plot(
    image: np.ndarray,
    detections: list[Detection],
    title: str,
    output_path: Path,
    config: DetectorConfig,
):
    """Save matplotlib figure with image + threshold mask side by side."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for d in detections:
        bx, by, bw, bh = d["bbox"]
        s = d["size_px"]
        color = "lime" if s > 1000 else "yellow" if s > 500 else "red"
        lw = 2 if s > 1000 else 1
        ax1.add_patch(Rectangle((bx, by), bw, bh, linewidth=lw, edgecolor=color, facecolor="none"))
        ax1.text(bx, by - 4, f"{s}px", color=color, fontsize=7, fontweight="bold")
    ax1.set_title(f"{title}\n{len(detections)} detections")

    bg_modes = np.array([histogram_mode(image, c) for c in range(3)])
    above = image.astype(np.float32) - bg_modes[np.newaxis, np.newaxis, :]
    if config.contrast_mode == ContrastMode.ABOVE:
        mask = np.any(above > config.contrast_offset, axis=2)
        mode_label = f"bg_mode + {config.contrast_offset}"
    else:
        mask = np.any(above < -config.contrast_offset, axis=2)
        mode_label = f"bg_mode - {config.contrast_offset}"
    ax2.imshow(mask, cmap="gray")
    ax2.set_title(f"Threshold mask ({mode_label})\nbg_modes BGR: {bg_modes.astype(int)}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ============================================================================
# Parallel worker helpers (from scripts/segment_chip_scan.py)
# ============================================================================

# Per-process flatfield cache for ProcessPoolExecutor workers
_cached_ff: np.ndarray | None = None
_cached_ff_path: str | None = None


class FrameResult(NamedTuple):
    frame_name: str
    detections: list[Detection]
    dark_frac: float
    skipped: bool


def process_frame(
    frame_path: str,
    flatfield_path: str | None,
    config: DetectorConfig,
    um_per_px: float,
    dark_frac_cutoff: float,
) -> FrameResult:
    """Worker: load image, apply flatfield, segment."""
    frame_name = Path(frame_path).stem
    raw = cv2.imread(frame_path)
    if raw is None:
        return FrameResult(frame_name, [], 0.0, True)

    if flatfield_path:
        global _cached_ff, _cached_ff_path
        if _cached_ff_path != flatfield_path:
            _cached_ff = np.load(flatfield_path).astype(np.float32)[:, :, ::-1]
            _cached_ff_path = flatfield_path
        assert _cached_ff is not None
        corrected = apply_flatfield(raw, _cached_ff)
    else:
        corrected = raw

    dark_frac = compute_dark_frac(corrected)
    if dark_frac > dark_frac_cutoff:
        return FrameResult(frame_name, [], dark_frac, True)

    detections = segment_frame(corrected, config, um_per_px)
    return FrameResult(frame_name, detections, dark_frac, False)


def natural_sort_key(path: Path) -> int:
    """Extract frame number for natural sorting."""
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else 0


def strip_geometry(det: Detection) -> Detection:
    """Return detection dict without hull/contour (large point lists)."""
    return {k: v for k, v in det.items() if k not in ("hull", "contour")}  # type: ignore[return-value]


def dedup_detections(
    detections: list[Detection],
    radius_um: float = 50.0,
) -> list[Detection]:
    """Greedy NMS spatial dedup on stage coordinates.

    Expects detections pre-sorted by priority (tier asc, score desc).
    Detections without stage_x/stage_y are kept unconditionally.
    """
    from scipy.spatial import KDTree

    with_coords = []
    without_coords = []
    for d in detections:
        if d.get("stage_x") is not None:
            with_coords.append(d)
        else:
            without_coords.append(d)

    if not with_coords:
        return list(detections)

    import numpy as np

    coords = np.array([(d["stage_x"], d["stage_y"]) for d in with_coords])
    tree = KDTree(coords)
    suppressed: set[int] = set()
    kept = list(without_coords)
    for idx, d in enumerate(with_coords):
        if idx in suppressed:
            continue
        kept.append(d)
        for n_idx in tree.query_ball_point(coords[idx], radius_um):
            if n_idx > idx:
                suppressed.add(n_idx)

    return kept
