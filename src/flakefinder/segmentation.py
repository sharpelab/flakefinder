"""Material-specific detector configuration and flake segmentation.

Merges detector configuration, frame segmentation, and parallel worker
logic into a single package module.  Provides the core detection pipeline
used by scripts/segment_flakes.py, scripts/segment_chip_scan.py, and
find_flakes.py.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple, TypedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage
from scipy.spatial import KDTree
from scipy.stats import kurtosis as scipy_kurtosis

from flakefinder.scan_utils import apply_flatfield
from flakefinder.types import ContrastRGB, GainRGB, PixelPolygon, Point2F, XYWHRect

# ============================================================================
# Detection types
# ============================================================================


class CalProjection(NamedTuple):
    """Result of projecting an (R, G) point onto the calibration curve."""

    dist: float
    thickness_nm: float | None


# AFM-verified hBN calibration data on 90nm SiO₂ (50x, Leica DM6M).
# Source: docs/bn_thickness_calibration.md
HBN_CAL_POINTS: tuple[tuple[float, float, float], ...] = (
    (-0.600, 0.286, 4.6),
    (-0.671, 0.079, 5.6),
    (-0.597, 0.219, 6.3),
    (-0.645, 0.313, 7.0),
    (-0.649, 0.376, 8.1),
    (-0.670, 0.409, 8.6),
    (-0.673, 0.526, 10.2),
    (-0.692, 0.948, 14.1),
    (-0.481, 1.613, 18.0),
    (0.421, 2.907, 26.0),
    (2.520, 4.636, 46.0),
)

# Starter calibration for hBN on 285nm SiO₂ (10x/20x measurements, no AFM).
# Source: docs/hbn_285nm_calibration.md
HBN_285NM_CAL_POINTS: tuple[tuple[float, float, float], ...] = (
    (-0.29, 0.15, 1.0),  # very thin — clipboard crop, left/right contrast
    (-0.46, 0.19, 1.5),  # very thin — clipboard crop, top/bottom
    (-0.70, 0.30, 2.0),  # thin — clipboard crop, top/bottom
    (-0.63, 0.39, 2.5),  # thin — clipboard crop, 20x top-left/bottom-right
    (-0.90, 0.42, 3.0),
    (-0.90, 0.55, 5.0),
    (-0.82, 0.65, 7.0),
    (-0.88, 0.79, 8.0),  # clipboard crop, bottom flake
    (-0.93, 0.83, 8.5),  # clipboard crop, top-left flake
    (-0.98, 0.83, 9.0),  # clipboard crop, thick per user
    (-0.88, 1.00, 10.0),  # clipboard crop, bottom flake
    (-0.81, 1.00, 10.0),  # clipboard crop, top flake
    (-0.74, 1.13, 12.0),
    (-0.56, 1.39, 16.0),
    (-0.43, 1.48, 17.0),  # clipboard crop, bottom-right flake
    (-0.53, 1.62, 18.0),
)


class _DetectionBase(TypedDict):
    bbox: XYWHRect
    center: Point2F
    size_px: int
    mean_contrast: float
    contrast_rgb: ContrastRGB
    cal_dist: float
    thickness_nm: float | None
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
    uniform_region_ent_um2: float
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
    det_id: int
    chip_idx: int


# ============================================================================
# Scoring functions (module-level for ProcessPoolExecutor pickle safety)
# ============================================================================


def _score_hbn_thin(det: Detection) -> tuple[int, float]:
    """hBN thin: size + cal_dist dominant, light thickness penalty."""
    pr = det["perim_ratio"]
    cd = det["cal_dist"]
    g = det["contrast_rgb"][1]
    r = det["contrast_rgb"][0]
    ent = det.get("entropy", det.get("g_entropy", 99.0))
    ar = det.get("aspect_ratio", 1.0)
    size_um2 = det["size_um2"]

    if pr < 1.50 and cd < 0.3 and g >= 0.0 and g < 1.2 and r < -0.5 and ent < 99.0 and ar < 6.0 and size_um2 >= 0.0:
        tier = 1
    elif pr < 1.35 and cd < 0.3 and ent < 4.5:
        tier = 2
    else:
        tier = 3

    ar_penalty = float(np.exp(-(max(ar - 3, 0) ** 2) / 8))
    g_penalty = 1.0 / (1.0 + 0.2 * max(g - 1.0, 0))
    log2_size = float(np.log2(max(size_um2, 1.0)))
    score = round(
        log2_size * log2_size * np.exp(-cd * 8) * ar_penalty * g_penalty,
        4,
    )
    return tier, score


def _score_hbn_medium(det: Detection) -> tuple[int, float]:
    """hBN medium: size + cal_dist dominant, light thickness penalty."""
    pr = det["perim_ratio"]
    cd = det["cal_dist"]
    g = det["contrast_rgb"][1]
    r = det["contrast_rgb"][0]
    ent = det.get("entropy", det.get("g_entropy", 99.0))
    ar = det.get("aspect_ratio", 1.0)
    size_um2 = det["size_um2"]

    b = det["contrast_rgb"][2]
    bg_ratio = b / g if g > 0.01 else 99.0

    if pr < 1.50 and cd < 0.15 and g >= 0.0 and g < 3.0 and r < 0.0 and ent < 99.0 and ar < 6.0 and size_um2 >= 500.0:
        tier = 1
        # Demote purple medium/thick flakes and messy interiors to T2
        if (g > 0.8 and bg_ratio > 1.2) or ent > 4.65:
            tier = 2
    elif pr < 1.50 and cd < 0.15 and ent < 4.65 and size_um2 >= 500.0:
        tier = 2
    else:
        tier = 3

    ar_penalty = float(np.exp(-(max(ar - 3, 0) ** 2) / 8))
    g_penalty = 1.0 / (1.0 + 0.2 * max(g - 1.0, 0))
    log2_size = float(np.log2(max(size_um2, 1.0)))
    score = round(
        log2_size * log2_size * np.exp(-cd * 8) * ar_penalty * g_penalty,
        4,
    )
    return tier, score


def _score_hbn_medium_285nm(det: Detection) -> tuple[int, float]:
    """hBN medium on 285nm SiO₂: tighter R gate to reject tape residue."""
    pr = det["perim_ratio"]
    cd = det["cal_dist"]
    g = det["contrast_rgb"][1]
    r = det["contrast_rgb"][0]
    ent = det.get("entropy", det.get("g_entropy", 99.0))
    ar = det.get("aspect_ratio", 1.0)
    size_um2 = det["size_um2"]

    b = det["contrast_rgb"][2]
    # On 285nm, real hBN has R < -0.7 (thin) to -0.5 (medium).
    # Tape residue sits at R ~ -0.63. Gate at R < -0.7 rejects most tape.
    # B < -0.05 separates real thin hBN (B ~ -0.12) from tape (B ~ -0.02).
    t1 = (
        pr < 1.50
        and cd < 0.20
        and g >= 0.0
        and g < 3.0
        and r < -0.70
        and b < -0.08
        and ent < 4.5
        and ar < 6.0
        and size_um2 >= 400.0
    )
    if t1:
        tier = 1
    elif pr < 1.50 and cd < 0.25 and r < -0.45 and ent < 4.65 and size_um2 >= 400.0:
        tier = 2
    else:
        tier = 3

    ar_penalty = float(np.exp(-(max(ar - 3, 0) ** 2) / 8))
    g_penalty = 1.0 / (1.0 + 0.2 * max(g - 1.0, 0))
    log2_size = float(np.log2(max(size_um2, 1.0)))
    score = round(
        log2_size * log2_size * np.exp(-cd * 8) * ar_penalty * g_penalty,
        4,
    )
    return tier, score


def _score_graphene(det: Detection) -> tuple[int, float]:
    """Graphene: size + cal_dist dominant, light thickness penalty (stub)."""
    pr = det["perim_ratio"]
    cd = det["cal_dist"]
    g = det["contrast_rgb"][1]
    r = det["contrast_rgb"][0]
    ent = det.get("entropy", det.get("g_entropy", 99.0))
    ar = det.get("aspect_ratio", 1.0)
    size_um2 = det["size_um2"]

    if pr < 1.20 and cd < 0.3 and g >= -99.0 and g < 4.0 and r < 99.0 and ent < 99.0 and ar < 6.0 and size_um2 >= 0.0:
        tier = 1
    elif pr < 1.35 and cd < 0.3 and ent < 99.0:
        tier = 2
    else:
        tier = 3

    ar_penalty = float(np.exp(-(max(ar - 3, 0) ** 2) / 8))
    g_penalty = 1.0 / (1.0 + 0.2 * max(g - 1.0, 0))
    log2_size = float(np.log2(max(size_um2, 1.0)))
    score = round(
        log2_size * log2_size * np.exp(-cd * 8) * ar_penalty * g_penalty,
        4,
    )
    return tier, score


def _score_wse2(det: Detection) -> tuple[int, float]:
    """WSe2: size-based scoring with blue contrast gate."""
    b = det["contrast_rgb"][2]
    ar = det.get("aspect_ratio", 1.0)
    size_um2 = det["size_um2"]

    # Blue contrast gate: reject non-blue smudges
    if b >= -0.1:
        tier = 3
    else:
        tier = 1

    ar_penalty = float(np.exp(-(max(ar - 3, 0) ** 2) / 8))
    log2_size = float(np.log2(max(size_um2, 1.0)))
    score = round(log2_size * log2_size * ar_penalty, 4)
    return tier, score


ScoreFn = Callable[[Detection], tuple[int, float]]


# ============================================================================
# Detector configuration
# ============================================================================


class ContrastMode(Enum):
    """Direction of flake-to-substrate contrast."""

    ABOVE = "above"  # flakes brighter than substrate (hBN)
    BELOW = "below"  # flakes darker than substrate (graphene)


@dataclass
class DetectorConfig:
    """All material-specific parameters for flake detection."""

    name: str  # human-readable display name, e.g. "hBN (thin) · 90nm SiO₂"

    # -- Segmentation --
    contrast_mode: ContrastMode
    contrast_offset: float
    min_size_um2: float
    edge_margin_px: int
    morph_kernel_size: int
    entropy_threshold: float  # percentile of local-std within component for uniform region metric
    subseg_min_std: float  # min within-blob std to attempt Otsu subsegmentation

    # -- Calibration curve: (R, G, thickness_nm) anchor points --
    cal_points: tuple[tuple[float, float, float], ...] | None
    cal_g_range: tuple[float, float]

    # -- Classification thresholds --
    cal_dist_match: float  # max distance for thin/medium/thick
    cal_dist_possible: float  # max distance for "possible"
    thin_max_nm: float  # thickness < this -> thin
    medium_max_nm: float  # thickness <= this -> medium, else thick
    non_match_label: str  # label for detections far from cal curve

    # -- Capture --
    white_balance: GainRGB

    # -- Scoring --
    score_fn: ScoreFn

    # -- Viewer filter defaults (not used by scoring) --
    tier1_perim_ratio: float
    tier1_cal_dist: float
    tier1_g_min: float
    tier1_g_max: float
    tier1_r_max: float
    tier1_entropy_max: float
    tier1_min_size_um2: float
    tier2_perim_ratio: float
    tier2_cal_dist: float
    tier2_entropy_max: float

    # Precomputed calibration curve (derived from cal_points or cal_g_range)
    _cal_g_curve: np.ndarray = field(init=False, repr=False, compare=False)
    _cal_r_curve: np.ndarray = field(init=False, repr=False, compare=False)
    _cal_thickness_curve: np.ndarray | None = field(init=False, repr=False, compare=False)
    cal_poly: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        g_grid = np.linspace(self.cal_g_range[0], self.cal_g_range[1], 500)
        self._cal_g_curve = g_grid

        if self.cal_points is not None:
            rs = np.array([p[0] for p in self.cal_points])
            gs = np.array([p[1] for p in self.cal_points])
            nms = np.array([p[2] for p in self.cal_points])

            # Fit R = poly(G) from anchor points
            self.cal_poly = tuple(float(c) for c in np.polyfit(gs, rs, 2))
            self._cal_r_curve = np.polyval(self.cal_poly, g_grid)

            # Interpolate thickness onto the same G grid
            order = np.argsort(gs)
            self._cal_thickness_curve = np.interp(g_grid, gs[order], nms[order])
        else:
            self.cal_poly = (0.0, 0.0, 0.0)
            self._cal_r_curve = np.polyval(self.cal_poly, g_grid)
            self._cal_thickness_curve = None

    def cal_curve(self, r: float, g: float) -> CalProjection:
        """Project (R, G) onto the calibration curve.

        Returns distance to curve and estimated thickness in nm.
        """
        dists = np.sqrt((self._cal_g_curve - g) ** 2 + (self._cal_r_curve - r) ** 2)
        idx = int(dists.argmin())
        dist = float(dists[idx])
        thickness_nm: float | None = None
        if self._cal_thickness_curve is not None:
            thickness_nm = round(float(self._cal_thickness_curve[idx]), 1)
        return CalProjection(round(dist, 4), thickness_nm)

    def classify(self, r: float, g: float) -> str:
        """Classify by R-G calibration distance and projected thickness."""
        proj = self.cal_curve(r, g)
        if proj.dist < self.cal_dist_match and proj.thickness_nm is not None:
            if proj.thickness_nm < self.thin_max_nm:
                return "thin"
            elif proj.thickness_nm <= self.medium_max_nm:
                return "medium"
            else:
                return "thick"
        elif proj.dist < self.cal_dist_possible:
            return "possible"
        return self.non_match_label

    def score_detection(self, det: Detection) -> tuple[int, float]:
        """Compute (tier, score) via preset-specific scoring function."""
        return self.score_fn(det)

    @classmethod
    def hbn_thin(cls) -> DetectorConfig:
        """hBN thin flake detection preset."""
        return cls(
            name="hBN (thin) · 90nm SiO₂",
            contrast_mode=ContrastMode.ABOVE,
            contrast_offset=15.0,
            min_size_um2=400.0,
            edge_margin_px=50,
            morph_kernel_size=5,
            entropy_threshold=0.4,
            subseg_min_std=0.8,
            cal_points=HBN_CAL_POINTS,
            cal_g_range=(-0.5, 6.0),
            cal_dist_match=0.5,
            cal_dist_possible=1.0,
            thin_max_nm=15.0,
            medium_max_nm=24.0,
            non_match_label="non-hBN",
            white_balance=GainRGB(red=1.41, green=1.02, blue=2.51),
            score_fn=_score_hbn_thin,
            tier1_perim_ratio=1.50,
            tier1_cal_dist=0.3,
            tier1_g_min=0.0,
            tier1_g_max=1.2,
            tier1_r_max=-0.5,
            tier1_entropy_max=99.0,
            tier1_min_size_um2=0.0,
            tier2_perim_ratio=1.35,
            tier2_cal_dist=0.3,
            tier2_entropy_max=4.5,
        )

    @classmethod
    def hbn_medium(cls) -> DetectorConfig:
        """hBN medium flake detection preset."""
        return cls(
            name="hBN (medium) · 90nm SiO₂",
            contrast_mode=ContrastMode.ABOVE,
            contrast_offset=15.0,
            min_size_um2=400.0,
            edge_margin_px=50,
            morph_kernel_size=5,
            entropy_threshold=0.4,
            subseg_min_std=0.8,
            cal_points=HBN_CAL_POINTS,
            cal_g_range=(-0.5, 6.0),
            cal_dist_match=0.5,
            cal_dist_possible=1.0,
            thin_max_nm=15.0,
            medium_max_nm=24.0,
            non_match_label="non-hBN",
            white_balance=GainRGB(red=1.41, green=1.02, blue=2.51),
            score_fn=_score_hbn_medium,
            tier1_perim_ratio=1.50,
            tier1_cal_dist=0.15,
            tier1_g_min=0.0,
            tier1_g_max=3.0,
            tier1_r_max=0.6,
            tier1_entropy_max=99.0,
            tier1_min_size_um2=500.0,
            tier2_perim_ratio=1.50,
            tier2_cal_dist=0.15,
            tier2_entropy_max=4.65,
        )

    @classmethod
    def hbn_medium_285nm(cls) -> DetectorConfig:
        """hBN medium flake detection on 285nm SiO₂ substrates."""
        return cls(
            name="hBN (medium) · 285nm SiO₂",
            contrast_mode=ContrastMode.ABOVE,
            contrast_offset=7.0,
            min_size_um2=400.0,
            edge_margin_px=50,
            morph_kernel_size=5,
            entropy_threshold=0.4,
            subseg_min_std=0.4,
            cal_points=HBN_285NM_CAL_POINTS,
            cal_g_range=(-0.5, 3.0),
            cal_dist_match=0.5,
            cal_dist_possible=1.0,
            thin_max_nm=15.0,
            medium_max_nm=24.0,
            non_match_label="non-hBN",
            white_balance=GainRGB(red=1.41, green=1.02, blue=2.51),
            score_fn=_score_hbn_medium_285nm,
            tier1_perim_ratio=1.50,
            tier1_cal_dist=0.20,
            tier1_g_min=0.0,
            tier1_g_max=3.0,
            tier1_r_max=-0.70,
            tier1_entropy_max=4.5,
            tier1_min_size_um2=400.0,
            tier2_perim_ratio=1.50,
            tier2_cal_dist=0.25,
            tier2_entropy_max=4.65,
        )

    @classmethod
    def graphene(cls) -> DetectorConfig:
        """Graphene detection preset (stub -- no calibration curve yet)."""
        return cls(
            name="Graphene · 90nm SiO₂",
            contrast_mode=ContrastMode.BELOW,
            contrast_offset=10.0,
            min_size_um2=130.0,
            edge_margin_px=50,
            morph_kernel_size=5,
            entropy_threshold=0.4,
            subseg_min_std=0.8,
            cal_points=None,
            cal_g_range=(-6.0, 0.5),
            cal_dist_match=0.5,
            cal_dist_possible=1.0,
            thin_max_nm=15.0,
            medium_max_nm=24.0,
            non_match_label="non-graphene",
            white_balance=GainRGB(red=1.41, green=1.02, blue=2.51),
            score_fn=_score_graphene,
            tier1_perim_ratio=1.20,
            tier1_cal_dist=0.3,
            tier1_g_min=-99.0,
            tier1_g_max=4.0,
            tier1_r_max=99.0,
            tier1_entropy_max=99.0,
            tier1_min_size_um2=0.0,
            tier2_perim_ratio=1.35,
            tier2_cal_dist=0.3,
            tier2_entropy_max=99.0,
        )

    @classmethod
    def wse2(cls) -> DetectorConfig:
        """WSe2 detection preset (stub -- no calibration curve yet)."""
        return cls(
            name="WSe₂ · 90nm SiO₂",
            contrast_mode=ContrastMode.BELOW,
            contrast_offset=35.0,
            min_size_um2=100.0,
            edge_margin_px=50,
            morph_kernel_size=5,
            entropy_threshold=0.4,
            subseg_min_std=0.12,
            cal_points=None,
            cal_g_range=(-6.0, 0.5),
            cal_dist_match=0.5,
            cal_dist_possible=1.0,
            thin_max_nm=15.0,
            medium_max_nm=24.0,
            non_match_label="non-WSe2",
            white_balance=GainRGB(red=1.41, green=1.02, blue=1.70),
            score_fn=_score_wse2,
            tier1_perim_ratio=1.20,
            tier1_cal_dist=0.3,
            tier1_g_min=-99.0,
            tier1_g_max=4.0,
            tier1_r_max=99.0,
            tier1_entropy_max=99.0,
            tier1_min_size_um2=0.0,
            tier2_perim_ratio=1.35,
            tier2_cal_dist=0.3,
            tier2_entropy_max=99.0,
        )

    @classmethod
    def _presets(cls) -> dict[str, Callable[[], DetectorConfig]]:
        return {
            "hbn_thin": cls.hbn_thin,
            "hbn_medium": cls.hbn_medium,
            "hbn_medium_285nm": cls.hbn_medium_285nm,
            "graphene": cls.graphene,
            "wse2": cls.wse2,
        }

    @classmethod
    def material_names(cls) -> list[str]:
        """Available material preset names."""
        return list(cls._presets())

    @classmethod
    def display_names(cls) -> dict[str, str]:
        """Map of preset key → human-readable display name."""
        return {key: factory().name for key, factory in cls._presets().items()}

    @classmethod
    def from_material(cls, name: str) -> DetectorConfig:
        """Create config from material name."""
        presets = cls._presets()
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


def _compute_local_std(channel: np.ndarray, radius: int = 5) -> np.ndarray:
    """Local standard deviation via box filter: sqrt(E[X²] - E[X]²)."""
    ch = channel.astype(np.float32)
    ksize = 2 * radius + 1
    mean = cv2.blur(ch, (ksize, ksize))
    mean_sq = cv2.blur(ch * ch, (ksize, ksize))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0))


def _uniform_region_area_um2(
    local_std: np.ndarray,
    component: np.ndarray,
    um_per_px: float,
    bbox: tuple[int, int, int, int],
    entropy_threshold: float,
) -> float:
    """Area (µm²) of the largest contiguous low-local-std region within a component.

    bbox is (y_min, y_max, x_min, x_max) used to crop to the component's
    bounding box for efficiency (avoids full-frame ops on 17k+ detections).
    entropy_threshold is the fraction (0–1) used as a percentile cutoff on
    within-component local-std values.
    """
    y0, y1, x0, x1 = bbox
    roi_std = local_std[y0 : y1 + 1, x0 : x1 + 1]
    roi_comp = component[y0 : y1 + 1, x0 : x1 + 1]

    flake_stds = roi_std[roi_comp]
    if flake_stds.size < 10:
        return 0.0

    threshold = float(np.percentile(flake_stds, entropy_threshold * 100))
    uniform = roi_comp & (roi_std <= threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    uniform_clean = cv2.morphologyEx(uniform.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    labeled, n_labels = ndimage.label(uniform_clean)
    if n_labels == 0:
        return 0.0

    sizes = ndimage.sum(uniform_clean, labeled, range(1, n_labels + 1))
    best_label = int(np.argmax(sizes)) + 1
    area_px = int((labeled == best_label).sum())
    return round(area_px * um_per_px**2, 1)


def _analyze_component(
    image: np.ndarray,
    component: np.ndarray,
    bg_modes: np.ndarray,
    norm_contrast: np.ndarray,
    grad_mag: np.ndarray,
    local_std: np.ndarray,
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

    # Largest contiguous uniform region (low local std)
    uniform_region = _uniform_region_area_um2(
        local_std, component, um_per_px, (y_min, y_max, x_min, x_max), config.entropy_threshold
    )

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

    proj = config.cal_curve(r_contrast, g_contrast)

    return Detection(
        bbox=XYWHRect(x_min, y_min, x_max - x_min, y_max - y_min),
        center=Point2F(round((x_min + x_max) / 2, 1), round((y_min + y_max) / 2, 1)),
        size_px=int(component.sum()),
        size_um2=round(int(component.sum()) * um_per_px**2, 1),
        mean_contrast=round(mean_contrast, 1),
        contrast_rgb=ContrastRGB(r_contrast, g_contrast, b_contrast),
        cal_dist=proj.dist,
        thickness_nm=proj.thickness_nm,
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
        uniform_region_ent_um2=uniform_region,
        hull=hull_pts,
        contour=contour_pts,
    )


def _otsu_split(
    contrast_channels: np.ndarray,
    component: np.ndarray,
    min_size_px: int,
    min_std: float = 0.8,
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

    if blob_vals.std() < min_std:
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
            if int(roi_piece.sum()) >= min_size_px:
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
    min_std: float = 0.8,
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

        pieces = _otsu_split(norm_contrast, comp, min_size_px, min_std=min_std)
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

    # Precompute local std on G channel (used by uniform region metric)
    local_std = _compute_local_std(image[:, :, 1])

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
        sub_components = _subsegment_by_contrast(
            image,
            component,
            bg_modes,
            min_size_px,
            norm_contrast,
            min_std=config.subseg_min_std,
        )

        for sub_comp in sub_components:
            det = _analyze_component(image, sub_comp, bg_modes, norm_contrast, grad_mag, local_std, config, um_per_px)
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
        if perim_ratio_thresh > 0 and det["perim_ratio"] >= perim_ratio_thresh:
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
        c = d["mean_contrast"]
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

        hull = d["hull"]
        if draw_hull and hull and len(hull) >= 3:
            pts = np.array(hull, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        contour = d["contour"]
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
    with_coords = []
    without_coords = []
    for d in detections:
        if d.get("stage_x") is not None:
            with_coords.append(d)
        else:
            without_coords.append(d)

    if not with_coords:
        return list(detections)

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
