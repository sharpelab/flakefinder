"""Shared surface calibration data and Z profile functions for Z curve tracking tests.

Surface data is from autofocus measurements at different Y positions on the chip.
Points are (x_um, residual_um) where residual is deviation from the plane fit.
"""

from collections.abc import Callable
from typing import TypedDict

import numpy as np
from scipy.interpolate import CubicSpline


class SurfaceEntry(TypedDict, total=False):
    y_um: int  # required
    points: list[tuple[int, float]]  # required
    max_gradient_um_per_mm: float
    max_curvature_um_per_mm2: float


class PlaneFit(TypedDict):
    a_um_per_mm: float  # X slope
    b_um_per_mm: float  # Y slope
    c_um: float  # Intercept


SURFACE_DATA: dict[str, SurfaceEntry] = {
    # Top row (Y=8175 µm) - sharpest features
    "top": {
        "y_um": 8175,
        "points": [
            (12112, -4.6),
            (16612, +6.0),
            (21112, +2.0),
            (25612, -4.6),
            (30112, +4.3),
            (34612, -2.4),
        ],
        "max_gradient_um_per_mm": 2.35,
        "max_curvature_um_per_mm2": 0.77,
    },
    # Middle row (Y=12675 µm) - gentler
    "middle": {
        "y_um": 12675,
        "points": [
            (12112, -3.5),
            (16612, -0.7),
            (21112, +4.7),
            (25612, -2.0),
            (30112, +1.7),
            (34612, -4.9),
        ],
        "max_gradient_um_per_mm": 1.47,
        "max_curvature_um_per_mm2": 0.59,
    },
}

# Plane fit from autofocus data
# Z = a*X_mm + b*Y_mm + c
PLANE_FIT: PlaneFit = {
    "a_um_per_mm": 1.4722,
    "b_um_per_mm": -0.6527,
    "c_um": 24666.85,
}


def build_z_profile(
    surface_name: str,
    x_start_um: float,
    x_end_um: float,
) -> tuple[Callable[[float], float], Callable[[float], float], dict]:
    """Build Z profile function from surface data.

    Args:
        surface_name: "top" or "middle"
        x_start_um: Start X position
        x_end_um: End X position

    Returns:
        (z_func, dzdx_func, info_dict) where:
        - z_func(x_um) returns desired Z in µm
        - dzdx_func(x_um) returns dZ/dX in µm/µm
        - info_dict contains metadata about the profile
    """
    surface = SURFACE_DATA[surface_name]
    y_um = surface["y_um"]
    points = surface["points"]

    x_points = np.array([p[0] for p in points])
    residuals = np.array([p[1] for p in points])

    spline = CubicSpline(x_points, residuals, extrapolate=True)
    spline_deriv = spline.derivative()

    a = PLANE_FIT["a_um_per_mm"]
    b = PLANE_FIT["b_um_per_mm"]
    c = PLANE_FIT["c_um"]
    y_mm = y_um / 1000

    def z_total(x_um: float) -> float:
        """Total Z = plane + residual."""
        x_mm = x_um / 1000
        z_plane = a * x_mm + b * y_mm + c
        z_residual = float(spline(x_um))
        return z_plane + z_residual

    def dzdx_total(x_um: float) -> float:
        """Total dZ/dX = plane_slope + residual_slope (in µm/µm)."""
        plane_slope = a / 1000  # Convert µm/mm to µm/µm
        return plane_slope + float(spline_deriv(x_um))

    x_test = np.linspace(x_start_um, x_end_um, 1000)
    z_test = np.array([z_total(x) for x in x_test])
    dzdx_test = np.array([dzdx_total(x) for x in x_test])

    info = {
        "surface_name": surface_name,
        "y_um": y_um,
        "x_range_um": (x_start_um, x_end_um),
        "z_range_um": (float(z_test.min()), float(z_test.max())),
        "z_delta_um": float(z_test[-1] - z_test[0]),
        "max_gradient_um_per_um": float(np.max(np.abs(dzdx_test))),
        "max_gradient_um_per_mm": float(np.max(np.abs(dzdx_test)) * 1000),
        "plane_slope_um_per_mm": a,
        "num_data_points": len(points),
    }

    return z_total, dzdx_total, info


def compute_ideal_z(surface_name: str, x_um: np.ndarray, z_offset: float) -> np.ndarray:
    """Compute ideal Z profile for given X positions.

    Args:
        surface_name: "top" or "middle"
        x_um: Array of X positions in µm
        z_offset: Z offset applied in test

    Returns:
        Array of ideal Z positions in µm
    """
    surface = SURFACE_DATA[surface_name]
    y_um = surface["y_um"]
    points = surface["points"]

    x_points = np.array([p[0] for p in points])
    residuals = np.array([p[1] for p in points])

    spline = CubicSpline(x_points, residuals, extrapolate=True)

    a = PLANE_FIT["a_um_per_mm"]
    b = PLANE_FIT["b_um_per_mm"]
    c = PLANE_FIT["c_um"]
    y_mm = y_um / 1000

    z_ideal = []
    for x in x_um:
        x_mm = x / 1000
        z_plane = a * x_mm + b * y_mm + c
        z_residual = float(spline(x))
        z_ideal.append(z_plane + z_residual + z_offset)

    return np.array(z_ideal)
