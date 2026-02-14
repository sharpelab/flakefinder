"""Material-specific detector configuration for flake segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


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
    min_size_px: int
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
    tier1_g_max: float
    tier2_perim_ratio: float
    tier2_cal_dist: float

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

    def score_detection(self, det: dict) -> tuple[int, float]:
        """Compute (tier, score) for a detection dict."""
        pr = det["perim_ratio"]
        cd = det["cal_dist"]
        g = det["contrast_rgb"][1]
        if pr < self.tier1_perim_ratio and cd < self.tier1_cal_dist and g < self.tier1_g_max:
            tier = 1
        elif pr < self.tier2_perim_ratio or cd < self.tier2_cal_dist:
            tier = 2
        else:
            tier = 3
        ge = det.get("grad_energy", 0)
        score = round(
            np.sqrt(det["size_px"]) * (1.0 / (1.0 + cd)) * (1.0 / (1.0 + ge)) ** 2,
            4,
        )
        return tier, score

    @classmethod
    def hbn(cls) -> DetectorConfig:
        """hBN detection preset (current production defaults)."""
        return cls(
            contrast_mode=ContrastMode.ABOVE,
            contrast_offset=15.0,
            min_size_px=1000,
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
            tier1_g_max=4.0,
            tier2_perim_ratio=1.35,
            tier2_cal_dist=0.3,
        )

    @classmethod
    def graphene(cls) -> DetectorConfig:
        """Graphene detection preset (stub -- no calibration curve yet)."""
        return cls(
            contrast_mode=ContrastMode.BELOW,
            contrast_offset=8.0,
            min_size_px=1000,
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
            tier1_g_max=4.0,
            tier2_perim_ratio=1.35,
            tier2_cal_dist=0.3,
        )

    @classmethod
    def from_material(cls, name: str) -> DetectorConfig:
        """Create config from material name."""
        presets = {"hbn": cls.hbn, "graphene": cls.graphene}
        if name not in presets:
            raise ValueError(f"Unknown material: {name!r}. Choose from: {', '.join(presets)}")
        return presets[name]()
