"""Predict camera-channel contrast for thin films on SiO₂/Si via transfer matrix method.

Supports hBN and graphene. Uses the propagation matrix approach from Aaron's
graphene_optics notebook, convolving with the Sony IMX183 (Leica K5C) RGB Bayer
filter response to predict per-channel contrast.

References:
  - Transfer matrix: Heavens, "Optical Properties of Thin Solid Films"
  - Si/SiO₂ index data: refractiveindex.info (Malitson for SiO₂, Aspnes for Si)
  - hBN index: Lee et al. 2019 (Sellmeier), Zotev et al. 2023 (extraordinary)
  - Graphene: n = 2.4 - 1j, t = 0.335 nm/layer (Aaron's graphene_optics)
  - IMX183 spectral response: Basler acA5472-5gc documentation
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Material data (Si, SiO₂) — loaded from Aaron's CSV files
# ---------------------------------------------------------------------------
_DATA_DIR = Path.home() / "sharpelab" / "graphene_optics"


def _load_index_csv(name: str) -> np.ndarray:
    """Load refractive index CSV, return array with wavelength in nm."""
    data = np.loadtxt(_DATA_DIR / name, delimiter=",", skiprows=1)
    data[:, 0] *= 1000  # µm → nm
    return data


_SI_DATA = _load_index_csv("Si_index.csv")
_SIO2_DATA = _load_index_csv("SiO2_index.csv")


def _interp_index(lamb_nm: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Interpolate complex refractive index n - jk at wavelengths (nm)."""
    n = np.interp(lamb_nm, data[:, 0], data[:, 1])
    k = np.interp(lamb_nm, data[:, 0], data[:, 2])
    return n - 1j * k


# ---------------------------------------------------------------------------
# Material layer thicknesses
# ---------------------------------------------------------------------------
_HBN_LAYER_THICKNESS_NM = 0.333  # monolayer, nm
_GRAPHENE_LAYER_THICKNESS_NM = 0.335  # monolayer, nm
_WSE2_LAYER_THICKNESS_NM = 0.649  # monolayer, nm


# ---------------------------------------------------------------------------
# WSe₂ refractive index (Zotev et al. 2023, bulk, ordinary/in-plane)
# ---------------------------------------------------------------------------
_WSE2_DATA_NM = np.array(
    [
        [400, 3.741, 2.126],
        [410, 3.792, 2.090],
        [420, 3.850, 2.066],
        [430, 3.916, 2.052],
        [440, 3.990, 2.041],
        [450, 4.076, 2.025],
        [460, 4.180, 1.995],
        [470, 4.287, 1.944],
        [480, 4.380, 1.880],
        [490, 4.460, 1.800],
        [500, 4.528, 1.704],
        [510, 4.573, 1.605],
        [520, 4.594, 1.520],
        [530, 4.601, 1.448],
        [540, 4.598, 1.394],
        [550, 4.597, 1.371],
        [560, 4.624, 1.380],
        [570, 4.710, 1.396],
        [580, 4.873, 1.343],
        [590, 5.004, 1.166],
        [600, 5.024, 0.969],
        [610, 4.976, 0.810],
        [620, 4.900, 0.690],
        [630, 4.822, 0.606],
        [640, 4.754, 0.550],
        [650, 4.691, 0.507],
        [660, 4.627, 0.473],
        [670, 4.564, 0.447],
        [680, 4.507, 0.430],
        [690, 4.449, 0.422],
        [700, 4.383, 0.425],
    ]
)


def n_wse2(lamb_nm: np.ndarray) -> np.ndarray:
    """WSe₂ complex refractive index interpolated from Zotev et al. 2023."""
    n = np.interp(lamb_nm, _WSE2_DATA_NM[:, 0], _WSE2_DATA_NM[:, 1])
    k = np.interp(lamb_nm, _WSE2_DATA_NM[:, 0], _WSE2_DATA_NM[:, 2])
    return n - 1j * k


# ---------------------------------------------------------------------------
# hBN refractive index models
# ---------------------------------------------------------------------------


def n_hbn_lee(lamb_nm: np.ndarray) -> np.ndarray:
    """Lee et al. 2019 Sellmeier (ordinary / in-plane). ~2.1–2.2 in visible."""
    lam_um = lamb_nm / 1000
    n2 = 1 + 3.263 * lam_um**2 / (lam_um**2 - 0.1644**2)
    return np.sqrt(n2) + 0j


_ZOTEV_E_DATA_NM = np.array(
    [
        [380, 1.6708],
        [400, 1.6639],
        [450, 1.6462],
        [500, 1.6289],
        [550, 1.5879],
        [600, 1.5800],
        [650, 1.5656],
        [700, 1.5568],
        [750, 1.5545],
        [800, 1.5450],
    ]
)


def n_hbn_zotev_e(lamb_nm: np.ndarray) -> np.ndarray:
    """Zotev et al. 2023 extraordinary (out-of-plane). ~1.55–1.67 in visible."""
    n = np.interp(lamb_nm, _ZOTEV_E_DATA_NM[:, 0], _ZOTEV_E_DATA_NM[:, 1])
    return n + 0j


def n_hbn_constant(value: float):
    """Return a callable that gives constant (real) n at any wavelength."""

    def _n(lamb_nm: np.ndarray) -> np.ndarray:
        return np.full_like(lamb_nm, value, dtype=complex)

    return _n


_N_MODELS: dict[str, object] = {
    "lee": n_hbn_lee,
    "zotev-e": n_hbn_zotev_e,
    "sqrt3": n_hbn_constant(math.sqrt(3)),
}


# ---------------------------------------------------------------------------
# Graphene refractive index
# ---------------------------------------------------------------------------
def n_graphene_constant(n_re: float = 2.4, n_im: float = 1.0):
    """Return a callable that gives constant complex n at any wavelength."""

    def _n(lamb_nm: np.ndarray) -> np.ndarray:
        return np.full_like(lamb_nm, n_re - 1j * n_im, dtype=complex)

    return _n


# ---------------------------------------------------------------------------
# NA-cone Gauss-Legendre quadrature (shared between reflectance() and widget)
# ---------------------------------------------------------------------------
_DEFAULT_NA_QUAD_NODES = 21


def _na_gauss_legendre(na: float, n_angles: int = _DEFAULT_NA_QUAD_NODES):
    """Nodes and weights for ∫ sin(θ)cos(θ) f(θ) dθ over the NA cone.

    Returns (thetas, weights) where the weights already absorb the
    sin(θ)cos(θ) factor for uniform-illumination averaging.  Divide any
    weighted sum by weights.sum() to normalise.
    """
    theta_max = np.arcsin(na)
    nodes, ws = np.polynomial.legendre.leggauss(n_angles)
    thetas = 0.5 * theta_max * (nodes + 1)
    weights = 0.5 * theta_max * ws * np.sin(thetas) * np.cos(thetas)
    return thetas, weights


# ---------------------------------------------------------------------------
# Transfer matrix reflectance (vectorised over wavelength)
# ---------------------------------------------------------------------------
def _reflectance_at_angle(
    lamb_nm: np.ndarray,
    n_film_fn,
    n_layers: int,
    t_oxide_nm: float,
    theta0: float,
    pol: str,
    layer_thickness_nm: float = _HBN_LAYER_THICKNESS_NM,
) -> np.ndarray:
    """R(λ) for air/film/SiO₂/Si at incidence angle theta0, single polarisation.

    pol='s': TE (E perpendicular to plane of incidence)
    pol='p': TM (E in plane of incidence)
    """
    n0 = 1.0
    sin_t0 = np.sin(theta0)
    cos_t0 = np.cos(theta0)

    if n_layers == 0:
        n1 = np.ones_like(lamb_nm, dtype=complex)
        cos_t1 = np.full_like(lamb_nm, cos_t0, dtype=complex)
        beta1 = np.zeros_like(lamb_nm, dtype=complex)
    else:
        n1 = n_film_fn(lamb_nm)
        cos_t1 = np.sqrt(1 - (n0 * sin_t0 / n1) ** 2)
        t_film = layer_thickness_nm * n_layers
        beta1 = 2 * np.pi * n1 * cos_t1 * t_film / lamb_nm

    n2 = _interp_index(lamb_nm, _SIO2_DATA)
    cos_t2 = np.sqrt(1 - (n0 * sin_t0 / n2) ** 2)
    beta2 = 2 * np.pi * n2 * cos_t2 * t_oxide_nm / lamb_nm

    n3 = _interp_index(lamb_nm, _SI_DATA)
    cos_t3 = np.sqrt(1 - (n0 * sin_t0 / n3) ** 2)

    if pol == "s":
        r01 = (n0 * cos_t0 - n1 * cos_t1) / (n0 * cos_t0 + n1 * cos_t1)
        r12 = (n1 * cos_t1 - n2 * cos_t2) / (n1 * cos_t1 + n2 * cos_t2)
        r23 = (n2 * cos_t2 - n3 * cos_t3) / (n2 * cos_t2 + n3 * cos_t3)
    else:
        r01 = (n1 * cos_t0 - n0 * cos_t1) / (n1 * cos_t0 + n0 * cos_t1)
        r12 = (n2 * cos_t1 - n1 * cos_t2) / (n2 * cos_t1 + n1 * cos_t2)
        r23 = (n3 * cos_t2 - n2 * cos_t3) / (n3 * cos_t2 + n2 * cos_t3)

    # Use explicit exp(-1j*…) for the negative-phase terms rather than
    # np.conj(exp(1j*…)).  For real β the two are equivalent, but with complex
    # β (absorbing film, Im(β)<0 in the n-i*k convention) the conj trick
    # silently drops the round-trip absorption decay —
    # |conj(exp(1j*β))| = exp(+|Im β|) but |exp(-1j*β)| = exp(-|Im β|).
    eb1pb2 = np.exp(1j * (beta1 + beta2))
    eb1mb2 = np.exp(1j * (beta1 - beta2))
    eb1pb2_neg = np.exp(-1j * (beta1 + beta2))
    eb1mb2_neg = np.exp(-1j * (beta1 - beta2))

    num = r01 * eb1pb2 + r12 * eb1mb2_neg + r23 * eb1pb2_neg + r01 * r12 * r23 * eb1mb2
    den = eb1pb2 + r01 * r12 * eb1mb2_neg + r01 * r23 * eb1pb2_neg + r12 * r23 * eb1mb2

    r = num / den
    return (r * r.conjugate()).real


def reflectance(
    lamb_nm: np.ndarray,
    n_film_fn,
    n_layers: int,
    t_oxide_nm: float,
    na: float = 0.0,
    n_angles: int = _DEFAULT_NA_QUAD_NODES,
    layer_thickness_nm: float = _HBN_LAYER_THICKNESS_NM,
) -> np.ndarray:
    """Reflectance R(λ) for air / film / SiO₂ / Si stack.

    na=0: normal incidence.  na>0: integrate over the objective cone,
    averaging s- and p-polarisations, weighted by sin(θ)cos(θ).
    """
    if na <= 0:
        return _reflectance_at_angle(
            lamb_nm,
            n_film_fn,
            n_layers,
            t_oxide_nm,
            0.0,
            "s",
            layer_thickness_nm=layer_thickness_nm,
        )

    thetas, weights = _na_gauss_legendre(na, n_angles)

    R_avg = np.zeros_like(lamb_nm)
    norm = 0.0
    for theta, w in zip(thetas, weights, strict=True):
        R_s = _reflectance_at_angle(
            lamb_nm,
            n_film_fn,
            n_layers,
            t_oxide_nm,
            theta,
            "s",
            layer_thickness_nm=layer_thickness_nm,
        )
        R_p = _reflectance_at_angle(
            lamb_nm,
            n_film_fn,
            n_layers,
            t_oxide_nm,
            theta,
            "p",
            layer_thickness_nm=layer_thickness_nm,
        )
        R_avg += 0.5 * (R_s + R_p) * w
        norm += w
    return R_avg / norm


def spectral_contrast(
    lamb_nm: np.ndarray,
    n_film_fn,
    n_layers: int,
    t_oxide_nm: float,
    na: float = 0.0,
    layer_thickness_nm: float = _HBN_LAYER_THICKNESS_NM,
) -> np.ndarray:
    """(R₀ - R) / R₀ at each wavelength."""
    R = reflectance(lamb_nm, n_film_fn, n_layers, t_oxide_nm, na=na, layer_thickness_nm=layer_thickness_nm)
    R0 = reflectance(lamb_nm, n_film_fn, 0, t_oxide_nm, na=na, layer_thickness_nm=layer_thickness_nm)
    return (R0 - R) / R0


# ---------------------------------------------------------------------------
# Sony IMX183 (Leica K5C) Bayer filter spectral response
# ---------------------------------------------------------------------------
# From Basler acA5472-5gc documentation, 10 nm intervals, relative 0–1.
_IMX183_WAVELENGTHS = np.arange(400, 710, 10, dtype=float)
_IMX183_BLUE = np.array(
    [
        0.444,
        0.541,
        0.615,
        0.683,
        0.749,
        0.787,
        0.792,
        0.764,
        0.700,
        0.600,
        0.478,
        0.341,
        0.238,
        0.169,
        0.124,
        0.091,
        0.063,
        0.047,
        0.040,
        0.036,
        0.031,
        0.026,
        0.026,
        0.030,
        0.037,
        0.047,
        0.057,
        0.066,
        0.072,
        0.076,
        0.081,
    ]
)
_IMX183_GREEN = np.array(
    [
        0.072,
        0.057,
        0.045,
        0.039,
        0.046,
        0.060,
        0.115,
        0.300,
        0.562,
        0.790,
        0.909,
        0.967,
        0.994,
        0.997,
        0.990,
        0.957,
        0.905,
        0.839,
        0.760,
        0.640,
        0.493,
        0.355,
        0.257,
        0.201,
        0.170,
        0.151,
        0.147,
        0.161,
        0.194,
        0.234,
        0.275,
    ]
)
_IMX183_RED = np.array(
    [
        0.097,
        0.070,
        0.050,
        0.037,
        0.028,
        0.024,
        0.024,
        0.028,
        0.034,
        0.037,
        0.043,
        0.060,
        0.081,
        0.082,
        0.068,
        0.057,
        0.062,
        0.204,
        0.524,
        0.820,
        0.926,
        0.911,
        0.880,
        0.847,
        0.823,
        0.789,
        0.748,
        0.700,
        0.657,
        0.634,
        0.645,
    ]
)


def _blackbody(lamb_nm: np.ndarray, T: float = 3200.0) -> np.ndarray:
    """Planck spectral radiance (arbitrary units), wavelength in nm."""
    lamb_m = lamb_nm * 1e-9
    h = 6.626e-34
    c = 3e8
    k = 1.381e-23
    return 2 * h * c**2 / lamb_m**5 / (np.exp(h * c / (lamb_m * k * T)) - 1)


def camera_channel_contrast(
    n_film_fn,
    n_layers: int,
    t_oxide_nm: float,
    illumination: np.ndarray | None = None,
    na: float = 0.0,
    layer_thickness_nm: float = _HBN_LAYER_THICKNESS_NM,
) -> tuple[float, float, float]:
    """Camera-channel contrast: (flake - substrate) / substrate per channel.

    Integrates reflectance × sensor response × illumination per channel,
    then takes the ratio.  Matches empirical: (pixel_flake - pixel_sub) / pixel_sub.

    Returns (red_contrast, green_contrast, blue_contrast).
    """
    lamb = _IMX183_WAVELENGTHS
    R = reflectance(lamb, n_film_fn, n_layers, t_oxide_nm, na=na, layer_thickness_nm=layer_thickness_nm)
    R0 = reflectance(lamb, n_film_fn, 0, t_oxide_nm, na=na, layer_thickness_nm=layer_thickness_nm)

    if illumination is None:
        illum = np.ones_like(lamb)
    else:
        illum = illumination

    contrasts = []
    for S in (_IMX183_RED, _IMX183_GREEN, _IMX183_BLUE):
        w = S * illum
        V_flake = np.trapezoid(R * w, lamb)
        V_sub = np.trapezoid(R0 * w, lamb)
        contrasts.append(float((V_flake - V_sub) / V_sub))
    return contrasts[0], contrasts[1], contrasts[2]


# ---------------------------------------------------------------------------
# Empirical calibration data (90nm SiO₂, 50x, AFM-verified)
# From docs/bn_thickness_calibration.md
# ---------------------------------------------------------------------------
_CAL_DATA = np.array(
    [
        # thickness_nm, R_contrast, G_contrast
        [4.6, -0.600, 0.286],
        # [5.6,  -0.671, 0.079],  # excluded — anomalously low G
        [6.3, -0.597, 0.219],
        [7.0, -0.645, 0.313],
        [8.1, -0.649, 0.376],
        [8.6, -0.670, 0.409],
        [10.2, -0.673, 0.526],
        [14.1, -0.692, 0.948],
        [18.0, -0.481, 1.613],
        [26.0, 0.421, 2.907],
        [46.0, 2.520, 4.636],
    ]
)

# Empirical 285nm data (thickness = sequential index, R/G from real flakes)
_CAL_DATA_285 = np.array(
    [
        # thickness_nm, R_contrast, G_contrast
        [1, 1.02, 1.70],
        [2, -0.98, 1.86],
        [3, -0.97, 1.22],
        [4, -0.97, 1.86],
        [5, -0.36, 0.18],
        [6, -0.99, 1.35],
        [7, -0.95, 1.72],
        [8, -0.95, 1.38],
        [9, -0.99, 1.44],
        [10, -0.41, 2.54],
        [11, -0.31, 2.66],
        [12, -0.97, 0.64],
        [13, -0.98, 1.36],
        [14, -0.91, 1.71],
        [15, -0.72, 0.26],
        [16, 0.03, 2.88],
        [17, -1.0, 2.0],
        [18, -0.97, 1.75],
        [19, -0.99, 1.73],
        [20, 0.03, 2.32],
        [21, 0.91, 2.5],
        [22, -0.99, 1.74],
        [23, -0.42, 2.07],
        [24, -0.98, 0.63],
        [25, -0.97, 1.29],
        [26, -0.35, 2.64],
        [27, -0.29, 2.45],
        [28, -0.30, 2.10],
        [29, -0.48, 2.43],
        [30, -0.99, 0.74],
        [31, -0.47, 2.29],
        [32, -0.48, 2.40],
        [33, -0.99, 1.60],
        [34, -0.26, 2.50],
        [35, -0.61, 2.13],
        [36, -0.99, 1.52],
        [37, -0.98, 1.41],
        [38, -0.99, 1.43],
        [39, 0.57, 2.95],
        [40, 0.70, 3.10],
        [41, -0.87, 1.93],
        [42, 0.79, 3.23],
        [43, -0.97, 1.86],
        [44, 0.61, 2.61],
        [45, -1.00, 0.58],
        [46, -1.00, 0.73],
        [47, -0.97, 1.24],
        [48, -0.59, 2.31],
        [49, -0.96, 1.87],
        [50, -0.84, 1.86],
        [51, 4.14, 0.30],
        [52, -0.98, 1.12],
        [53, -0.59, 1.78],
        [54, 1.91, 1.11],
        [55, -0.98, 1.19],
        [56, 0.30, 2.80],
        [57, -0.33, 2.44],
        [58, 0.39, 2.68],
        [59, -0.97, 1.25],
        [60, -0.99, 1.87],
    ]
)

# Graphene empirical calibration data (90nm SiO₂, 50x)
# Layer-count assignments from optical contrast; not AFM-verified.
# Format: [thickness_nm, R_contrast, G_contrast]
_CAL_DATA_GRAPHENE = np.array(
    [
        [0.335, -0.14, -0.15],  # 1 layer, N=41
        [0.670, -0.26, -0.28],  # 2 layers, N=42
        [1.005, -0.38, -0.39],  # 3 layers, N=34
        [1.340, -0.48, -0.49],  # 4 layers, N=18
        [1.675, -0.57, -0.58],  # 5 layers, N=17
    ]
)

# WSe₂ empirical calibration data (300nm SiO₂)
# Pixel-pick contrast from collaborator images; layer counts optical.
# Format: [thickness_nm, R_contrast, G_contrast]
_CAL_DATA_WSE2 = np.array(
    [
        [0.649, -0.201, -0.199],  # 1 layer, N=3
        [1.298, -0.323, -0.385],  # 2 layers, N=1
    ]
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material",
        choices=["hbn", "graphene"],
        default="hbn",
        help="Material type (default: hbn)",
    )
    parser.add_argument(
        "--n-hbn",
        default="sqrt3",
        help='Refractive index model: "lee", "zotev-e", "sqrt3", or a float (default: sqrt3)',
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=200,
        help="Max hBN layers to compute (default: 200, ~67 nm)",
    )
    parser.add_argument(
        "--oxide",
        default="90,285",
        help="Comma-separated oxide thicknesses in nm (default: 90,285)",
    )
    parser.add_argument(
        "--na",
        type=float,
        default=0.75,
        help="Objective NA for angle averaging (0 = normal incidence, default: 0.75 for 50x)",
    )
    parser.add_argument(
        "--lamp",
        type=float,
        default=None,
        metavar="TEMP_K",
        help="Include blackbody illumination at TEMP_K (e.g. 3200 for halogen)",
    )
    parser.add_argument(
        "--wb",
        type=str,
        default=None,
        metavar="R,G,B",
        help="Derive illumination from WB gains (lamp ∝ 1/gain per channel). E.g. '1.41,1.02,2.51' for hBN WB",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit ε_r (and optionally t_oxide) to empirical 90nm data",
    )
    parser.add_argument(
        "--fit-oxide",
        action="store_true",
        help="Also fit oxide thickness (default: fix at 90nm)",
    )
    parser.add_argument(
        "--fit-lamp",
        action="store_true",
        help="Also fit lamp color temperature as blackbody",
    )
    parser.add_argument(
        "--r-offset",
        type=float,
        default=0.0,
        help="Additive offset applied to empirical R data (for scope calibration)",
    )
    parser.add_argument(
        "--g-offset",
        type=float,
        default=0.0,
        help="Additive offset applied to empirical G data (for scope calibration)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save plot to file instead of showing",
    )
    args = parser.parse_args()

    # Resolve material and n model
    is_graphene = args.material == "graphene"
    layer_t = _GRAPHENE_LAYER_THICKNESS_NM if is_graphene else _HBN_LAYER_THICKNESS_NM
    material_name = "Graphene" if is_graphene else "hBN"

    if is_graphene:
        n_fn = n_graphene_constant(2.4, 1.0)
        model_label = "2.4 − 1.0j"
    elif args.n_hbn in _N_MODELS:
        n_fn = _N_MODELS[args.n_hbn]
        model_label = args.n_hbn
        if args.n_hbn == "sqrt3":
            model_label = f"√3 ≈ {math.sqrt(3):.4f}"
    else:
        try:
            val = float(args.n_hbn)
        except ValueError:
            parser.error(f"Unknown n model: {args.n_hbn!r}")
        n_fn = n_hbn_constant(val)
        model_label = f"{val:.4f}"

    oxides = [float(x) for x in args.oxide.split(",")]

    # Illumination spectrum (must be set before fit)
    illum = None
    lamp_label = ""
    if args.wb:
        wb_vals = [float(x) for x in args.wb.split(",")]
        if len(wb_vals) != 3:
            parser.error("--wb requires R,G,B (3 values)")
        wb_r, wb_g, wb_b = wb_vals
        illum = _IMX183_RED / wb_r + _IMX183_GREEN / wb_g + _IMX183_BLUE / wb_b
        illum /= illum.max()
        lamp_label = f", WB {args.wb}"
    elif args.lamp:
        illum = _blackbody(_IMX183_WAVELENGTHS, args.lamp)
        illum /= illum.max()
        lamp_label = f", lamp {args.lamp:.0f}K"

    # --- Fit ε_r to empirical data (hBN only) ---
    if args.fit and is_graphene:
        parser.error("--fit is not supported for graphene (no empirical calibration data)")
    if args.fit:
        from scipy.optimize import minimize

        cal = _CAL_DATA.copy()
        max_t = cal[:, 0].max()
        fit_layers = np.arange(0, int(max_t / _HBN_LAYER_THICKNESS_NM) + 2)

        from scipy.interpolate import interp1d

        def _rg_residuals(params):
            i = 0
            eps_r = params[i]
            i += 1
            t_ox = params[i] if args.fit_oxide else 90.0
            if args.fit_oxide:
                i += 1
            T_lamp = params[i] if args.fit_lamp else None
            if args.fit_lamp:
                i += 1

            n_val = np.sqrt(eps_r)
            n_fn_fit = n_hbn_constant(n_val)

            fit_illum = None
            if T_lamp is not None:
                fit_illum = _blackbody(_IMX183_WAVELENGTHS, T_lamp)
                fit_illum /= fit_illum.max()
            elif illum is not None:
                fit_illum = illum

            r_arr, g_arr = [], []
            for nl in fit_layers:
                r, g, _b = camera_channel_contrast(n_fn_fit, int(nl), t_ox, fit_illum, na=args.na)
                r_arr.append(r)
                g_arr.append(g)
            r_thy = np.array(r_arr)
            g_thy = np.array(g_arr)
            t_thy = fit_layers * _HBN_LAYER_THICKNESS_NM

            r_interp = interp1d(t_thy, r_thy, kind="linear", fill_value="extrapolate")
            g_interp = interp1d(t_thy, g_thy, kind="linear", fill_value="extrapolate")

            total = 0.0
            for row in cal:
                t_emp, r_emp, g_emp = row
                r_t = float(r_interp(t_emp))
                g_t = float(g_interp(t_emp))
                total += (r_emp - r_t) ** 2 + (g_emp - g_t) ** 2
            return total

        x0, bounds = [3.0], [(1.5, 20.0)]
        if args.fit_oxide:
            x0.append(90.0)
            bounds.append((70.0, 110.0))
        if args.fit_lamp:
            x0.append(3200.0)
            bounds.append((2000.0, 8000.0))

        result = minimize(_rg_residuals, x0, method="L-BFGS-B", bounds=bounds)

        i = 0
        eps_fit = result.x[i]
        i += 1
        n_fit = np.sqrt(eps_fit)
        t_ox_fit = result.x[i] if args.fit_oxide else 90.0
        if args.fit_oxide:
            i += 1
        T_lamp_fit = result.x[i] if args.fit_lamp else None
        if args.fit_lamp:
            i += 1

        print(f"\n{'=' * 50}")
        print("FIT RESULT:")
        print(f"  ε_r    = {eps_fit:.4f}")
        print(f"  n      = √ε_r = {n_fit:.4f}")
        print(f"  t_ox   = {t_ox_fit:.1f} nm")
        if T_lamp_fit is not None:
            print(f"  T_lamp = {T_lamp_fit:.0f} K")
        print(f"  RSS    = {result.fun:.6f}")
        print(f"{'=' * 50}\n")

        # Apply fit results
        n_fn = n_hbn_constant(n_fit)
        model_label = f"fit: n={n_fit:.3f} (ε_r={eps_fit:.3f})"
        if args.fit_oxide:
            model_label += f", t_ox={t_ox_fit:.1f}nm"
            oxides = [t_ox_fit] + [o for o in oxides if o != 90]
        if T_lamp_fit is not None:
            model_label += f", {T_lamp_fit:.0f}K"
            illum = _blackbody(_IMX183_WAVELENGTHS, T_lamp_fit)
            illum /= illum.max()
            lamp_label = f", lamp {T_lamp_fit:.0f}K"

    layers = np.arange(0, args.max_layers + 1)
    thickness_nm = layers * layer_t

    # Compute per-channel contrast for each oxide thickness
    results = {}
    for t_ox in oxides:
        r_arr, g_arr, b_arr = [], [], []
        for nl in layers:
            r, g, b = camera_channel_contrast(
                n_fn,
                int(nl),
                t_ox,
                illum,
                na=args.na,
                layer_thickness_nm=layer_t,
            )
            r_arr.append(r)
            g_arr.append(g)
            b_arr.append(b)
        results[t_ox] = {
            "R": np.array(r_arr),
            "G": np.array(g_arr),
            "B": np.array(b_arr),
        }

    # Plot: R/G contrast space, one subplot per oxide
    n_ox = len(oxides)
    fig, axes = plt.subplots(1, n_ox, figsize=(7 * n_ox, 6), squeeze=False)
    na_label = f", NA={args.na}" if args.na > 0 else ", normal inc."
    fig.suptitle(
        f"{material_name} on SiO₂/Si — R/G contrast space (n = {model_label}{lamp_label}{na_label})",
        fontsize=14,
    )

    # Thickness annotations: label every N layers
    label_layers = [5, 10, 15, 20, 30, 45, 60, 90, 120, 150]
    label_layers = [nl for nl in label_layers if nl <= args.max_layers]

    for col, t_ox in enumerate(oxides):
        data = results[t_ox]
        ax = axes[0, col]

        # Theory curve
        ax.plot(data["G"], data["R"], "k-", linewidth=1.5, label="Theory", zorder=2)

        # Annotate thickness at select points
        for nl in label_layers:
            t_nm = nl * layer_t
            g_val = data["G"][nl]
            r_val = data["R"][nl]
            ax.plot(g_val, r_val, "ko", markersize=4, zorder=3)
            ax.annotate(
                f"{t_nm:.0f}nm",
                (g_val, r_val),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="0.3",
            )

        # Overlay empirical data (hBN only)
        if col == 0 and not is_graphene:
            cal_r = _CAL_DATA[:, 1] + args.r_offset
            cal_g = _CAL_DATA[:, 2] + args.g_offset
            ax.scatter(
                cal_g,
                cal_r,
                c="tab:orange",
                marker="o",
                s=50,
                zorder=5,
                edgecolors="k",
                linewidths=0.7,
                label="AFM (90nm, shifted)",
            )
            for i, row in enumerate(_CAL_DATA):
                t_nm = row[0]
                ax.annotate(
                    f"{t_nm:.0f}",
                    (cal_g[i], cal_r[i]),
                    textcoords="offset points",
                    xytext=(6, -6),
                    fontsize=7,
                    color="tab:orange",
                    fontweight="bold",
                )

        # Overlay 285nm empirical data on the 285nm panel (hBN only)
        show_285 = not is_graphene and abs(t_ox - 285) < 10 and len(_CAL_DATA_285) > 0
        if show_285:
            cal285_r = _CAL_DATA_285[:, 1] + args.r_offset
            cal285_g = _CAL_DATA_285[:, 2] + args.g_offset
            ax.scatter(
                cal285_g,
                cal285_r,
                c="tab:red",
                marker="s",
                s=50,
                zorder=5,
                edgecolors="k",
                linewidths=0.7,
                label="285nm empirical",
            )
            for i, row in enumerate(_CAL_DATA_285):
                t_nm = row[0]
                ax.annotate(
                    f"~{t_nm:.0f}",
                    (cal285_g[i], cal285_r[i]),
                    textcoords="offset points",
                    xytext=(6, -6),
                    fontsize=7,
                    color="tab:red",
                    fontweight="bold",
                )

        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xlabel("Green contrast")
        ax.set_ylabel("Red contrast")
        ax.set_title(f"{t_ox:.0f} nm SiO₂")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()

    # Print fit diagnostics for the first oxide thickness (hBN only)
    diag_ox = oxides[0]
    if not is_graphene and diag_ox in results:
        data_diag = results[diag_ox]
        from scipy.interpolate import interp1d

        r_interp = interp1d(thickness_nm, data_diag["R"], kind="linear")
        g_interp = interp1d(thickness_nm, data_diag["G"], kind="linear")
        mask = _CAL_DATA[:, 0] <= thickness_nm[-1]
        cal = _CAL_DATA[mask]
        r_residuals = []
        g_residuals = []
        print(f"\n--- {diag_ox:.0f}nm fit diagnostics ---")
        print(f"{'t(nm)':>6} {'R_emp':>8} {'R_thy':>8} {'R_off':>8} {'G_emp':>8} {'G_thy':>8} {'G_off':>8}")
        for row in cal:
            t, r_e, g_e = row
            r_e += args.r_offset
            g_e += args.g_offset
            r_t = float(r_interp(t))
            g_t = float(g_interp(t))
            r_residuals.append(r_e - r_t)
            g_residuals.append(g_e - g_t)
            print(f"{t:6.1f} {r_e:8.3f} {r_t:8.3f} {r_e - r_t:8.3f} {g_e:8.3f} {g_t:8.3f} {g_e - g_t:8.3f}")
        r_res = np.array(r_residuals)
        g_res = np.array(g_residuals)
        print(f"\nR residual: rms={np.sqrt(np.mean(r_res**2)):.3f} mae={np.mean(np.abs(r_res)):.3f}")
        print(f"G residual: rms={np.sqrt(np.mean(g_res**2)):.3f} mae={np.mean(np.abs(g_res)):.3f}")

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
