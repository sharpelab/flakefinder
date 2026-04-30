"""Interactive widget for exploring hBN, graphene, and WSe₂ R/G contrast.

Tabbed layout with per-tab slider visibility. Heavy compute runs on a
background thread with a single-slot mailbox (last slider value wins), so
slider drags stay responsive.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

sys.path.insert(0, str(Path(__file__).parent))
from hbn_contrast import (
    _CAL_DATA,
    _CAL_DATA_285,
    _CAL_DATA_GRAPHENE,
    _CAL_DATA_WSE2,
    _DEFAULT_NA_QUAD_NODES,
    _GRAPHENE_LAYER_THICKNESS_NM,
    _HBN_LAYER_THICKNESS_NM,
    _IMX183_GREEN,
    _IMX183_RED,
    _IMX183_WAVELENGTHS,
    _SI_DATA,
    _SIO2_DATA,
    _WSE2_LAYER_THICKNESS_NM,
    _blackbody,
    _interp_index,
    _na_gauss_legendre,
    n_wse2,
)

lamb = _IMX183_WAVELENGTHS


def _load_elijah_led(target_lamb: np.ndarray) -> np.ndarray:
    """Load Elijah's LED spectrum and resample onto target_lamb (nm).

    Source: calibration/elijah_led_spectrum.csv — 3648-pt spectrometer
    trace, 200–1011 nm.  Two columns (wavelength_nm, intensity, no header)."""
    csv_path = Path(__file__).resolve().parents[1] / "calibration" / "elijah_led_spectrum.csv"
    raw = np.loadtxt(csv_path, delimiter=",")
    return np.interp(target_lamb, raw[:, 0], raw[:, 1])


def _load_objective_transmission_10x(target_lamb: np.ndarray) -> np.ndarray:
    """Load 10x objective transmission and resample onto target_lamb (nm).

    Source: calibration/objective_10x_transmission.csv — DataThief'd from
    a chart whose X-axis annotation didn't survive digitization, so column-0
    values are pixel coordinates spanning [101.34, 297.35].  We remap
    linearly to [400, 700] nm under the assumption that the chart spanned
    the visible band; the right edge looks slightly sharp under that
    assumption (cuts hard near 700 nm) so the chart may have actually been
    400–680 or 400–690.  Treat this axis as speculative until verified.
    """
    csv_path = Path(__file__).resolve().parents[1] / "calibration" / "objective_10x_transmission.csv"
    raw = np.loadtxt(csv_path, delimiter=",")
    raw = raw[raw[:, 0].argsort()]
    x_min, x_max = float(raw[0, 0]), float(raw[-1, 0])
    lamb_remapped = 400.0 + (raw[:, 0] - x_min) / (x_max - x_min) * 300.0
    return np.interp(target_lamb, lamb_remapped, raw[:, 1])


class _Illuminant(NamedTuple):
    """One illuminant option for the widget toggle.

    red_lit / green_lit are pre-multiplied (bandpass × illuminant) on the
    _IMX183_WAVELENGTHS grid; that's the only product compute_rg needs.
    """

    label: str
    red_lit: np.ndarray
    green_lit: np.ndarray


# Available illuminants.  Halogen 3200 K is the historical default — matches
# the 50x AFM cal points; the Leica's actual lamp is a white LED, so the
# elijah_led entry uses a measured trace from a sister microscope (Elijah's
# machine) that should be representative.
_ILLUM_3200 = _blackbody(_IMX183_WAVELENGTHS, 3200.0)
_ILLUM_ELIJAH = _load_elijah_led(_IMX183_WAVELENGTHS)

_ILLUMINANTS: dict[str, _Illuminant] = {
    "halogen_3200K": _Illuminant(
        label="Halogen 3200K",
        red_lit=_IMX183_RED * _ILLUM_3200,
        green_lit=_IMX183_GREEN * _ILLUM_3200,
    ),
    "elijah_led": _Illuminant(
        label="Elijah LED",
        red_lit=_IMX183_RED * _ILLUM_ELIJAH,
        green_lit=_IMX183_GREEN * _ILLUM_ELIJAH,
    ),
}

# Module-level aliases retained for the regression test, which imports
# _IMX183_RED_LIT/_IMX183_GREEN_LIT directly.  Always halogen 3200 K.
_IMX183_RED_LIT = _ILLUMINANTS["halogen_3200K"].red_lit
_IMX183_GREEN_LIT = _ILLUMINANTS["halogen_3200K"].green_lit


class _Transmission(NamedTuple):
    """One objective-transmission option for the widget toggle.

    `factor` is the per-λ multiplier on (bandpass × illuminant); the
    light passes through the objective twice (incident + reflected) so
    real-objective T(λ) is squared before storing here."""

    label: str
    factor: np.ndarray


_TRANSMISSIONS: dict[str, _Transmission] = {
    "off": _Transmission(label="off", factor=np.ones_like(_IMX183_WAVELENGTHS)),
    "10x": _Transmission(
        label="10x (T²)",
        factor=_load_objective_transmission_10x(_IMX183_WAVELENGTHS) ** 2,
    ),
}

n_sio2 = _interp_index(lamb, _SIO2_DATA)
n_si = _interp_index(lamb, _SI_DATA)

LABEL_LAYERS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200, 250, 300]

# 10x line-fit measurements on graphite gate flakes (four AFM-cut regions, 2026-04-29).
# Background line-fit RGB = (64, 69, 91); contrast = (I_flake - I_bg) / I_bg per channel.
# Thicknesses 6-14 nm = thick multilayer graphene used as gates (~19-41 layers).
# cut 3 and cut 4 have R=0 in the line fit, so C_R clips at -1.0 — flagged as
# unreliable below so the widget renders them with hollow markers + low alpha.
# Format: [thickness_nm, R_contrast, G_contrast]
_CAL_DATA_GRAPHITE_10X = np.array(
    [
        [13.7, -0.391, 0.362],  # cut 1
        [11.8, -0.797, -0.087],  # cut 2
        # cut 3 — flake #001a59, local bg #40445b (G=68, not the global G=69).
        [9.3, -1.000, -0.618],  # R clipped
        [6.4, -1.000, -0.768],  # cut 4 — R clipped
    ]
)
_CAL_DATA_GRAPHITE_10X_R_RELIABLE = np.array([True, True, False, False])

# Per-material max compute layers (plot + cache extent)
HBN_MAX_LAYERS = 300
GRAPHENE_MAX_LAYERS = 60
WSE2_MAX_LAYERS = 15
WSE2_PLOT_LAYERS = 10  # label + line extent on the WSe₂ panel


def compute_rg(
    n_film: complex | np.ndarray,
    t_oxide: float,
    na: float,
    layer_thickness_nm: float = _HBN_LAYER_THICKNESS_NM,
    max_layers: int = HBN_MAX_LAYERS,
    *,
    red_lit: np.ndarray | None = None,
    green_lit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch-compute R/G contrast for layers 1..max_layers.

    n_film can be a scalar complex (constant n) or an ndarray of complex values
    (one per wavelength in _IMX183_WAVELENGTHS, for dispersive materials).

    red_lit/green_lit are pre-multiplied (camera bandpass × illuminant) on the
    _IMX183_WAVELENGTHS grid; default to halogen 3200K.

    Returns (r_contrast, g_contrast, thickness_nm).
    """
    if red_lit is None:
        red_lit = _IMX183_RED_LIT
    if green_lit is None:
        green_lit = _IMX183_GREEN_LIT
    if isinstance(n_film, np.ndarray):
        n1 = np.array(n_film, dtype=np.complex128)
    else:
        n1 = np.full(len(lamb), n_film, dtype=np.complex128)

    if na <= 0:
        thetas = np.array([0.0])
        raw_weights = np.array([1.0])
    else:
        thetas, raw_weights = _na_gauss_legendre(na, _DEFAULT_NA_QUAD_NODES)

    total_w = raw_weights.sum()

    nl_arr = np.arange(1, max_layers + 1)
    t_films = nl_arr * layer_thickness_nm

    R_flake = np.zeros((max_layers, len(lamb)))
    R_sub = np.zeros(len(lamb))

    for theta, w in zip(thetas, raw_weights, strict=True):
        sin_t0 = np.sin(theta)
        cos_t0 = np.cos(theta)

        cos_t1 = np.sqrt(1 - (sin_t0 / n1) ** 2)
        cos_t2 = np.sqrt(1 - (sin_t0 / n_sio2) ** 2)
        cos_t3 = np.sqrt(1 - (sin_t0 / n_si) ** 2)

        beta2 = 2 * np.pi * n_sio2 * cos_t2 * t_oxide / lamb
        beta1_all = 2 * np.pi * n1[None, :] * cos_t1[None, :] * t_films[:, None] / lamb[None, :]

        eb2 = np.exp(1j * beta2)
        eb2c = np.conj(eb2)

        for pol in ("s", "p"):
            if pol == "s":
                r01 = (cos_t0 - n1 * cos_t1) / (cos_t0 + n1 * cos_t1)
                r12 = (n1 * cos_t1 - n_sio2 * cos_t2) / (n1 * cos_t1 + n_sio2 * cos_t2)
                r23 = (n_sio2 * cos_t2 - n_si * cos_t3) / (n_sio2 * cos_t2 + n_si * cos_t3)
                r_as = (cos_t0 - n_sio2 * cos_t2) / (cos_t0 + n_sio2 * cos_t2)
            else:
                r01 = (n1 * cos_t0 - cos_t1) / (n1 * cos_t0 + cos_t1)
                r12 = (n_sio2 * cos_t1 - n1 * cos_t2) / (n_sio2 * cos_t1 + n1 * cos_t2)
                r23 = (n_si * cos_t2 - n_sio2 * cos_t3) / (n_si * cos_t2 + n_sio2 * cos_t3)
                r_as = (n_sio2 * cos_t0 - cos_t2) / (n_sio2 * cos_t0 + cos_t2)

            # Use explicit exp(-1j*…) for the negative-phase terms rather than
            # np.conj(exp(1j*…)).  For real β the two are equivalent, but with
            # complex β (absorbing film, Im(β)<0 in the n-i*k convention) the
            # conj trick silently drops the round-trip absorption decay —
            # |conj(exp(1j*β))| = exp(+|Im β|) but |exp(-1j*β)| = exp(-|Im β|).
            eb1pb2 = np.exp(1j * (beta1_all + beta2[None, :]))
            eb1mb2 = np.exp(1j * (beta1_all - beta2[None, :]))
            eb1pb2_neg = np.exp(-1j * (beta1_all + beta2[None, :]))
            eb1mb2_neg = np.exp(-1j * (beta1_all - beta2[None, :]))
            num = r01 * eb1pb2 + r12 * eb1mb2_neg + r23 * eb1pb2_neg + r01 * r12 * r23 * eb1mb2
            den = eb1pb2 + r01 * r12 * eb1mb2_neg + r01 * r23 * eb1pb2_neg + r12 * r23 * eb1mb2
            r = num / den
            R_flake += 0.5 * w * (r * r.conjugate()).real

            num_s = r_as * eb2 + r23 * eb2c
            den_s = eb2 + r_as * r23 * eb2c
            r_s = num_s / den_s
            R_sub += 0.5 * w * (r_s * r_s.conjugate()).real

    R_flake /= total_w
    R_sub /= total_w

    V_sub_r = np.trapezoid(R_sub * red_lit, lamb)
    V_sub_g = np.trapezoid(R_sub * green_lit, lamb)
    V_flake_r = np.trapezoid(R_flake * red_lit[None, :], lamb, axis=1)
    V_flake_g = np.trapezoid(R_flake * green_lit[None, :], lamb, axis=1)

    return (
        (V_flake_r - V_sub_r) / V_sub_r,
        (V_flake_g - V_sub_g) / V_sub_g,
        t_films,
    )


def _setup_panel(ax, title):
    """Common axis setup."""
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("Green contrast")
    ax.set_ylabel("Red contrast")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 5.5)
    ax.set_ylim(-1.5, 4)


def _make_annotations(ax, layer_thickness_nm, max_layers, label_layers=None, label_fmt="thickness"):
    """Create thickness annotations and dot artist for a panel.

    label_fmt: "thickness" for "Xnm" labels, "layers" for "XL" labels.
    """
    if label_layers is None:
        label_layers = [nl for nl in LABEL_LAYERS if nl <= max_layers]
    ann = []
    dots = ax.plot([], [], "ko", markersize=4, zorder=3)[0]
    for nl in label_layers:
        if nl <= max_layers:
            if label_fmt == "layers":
                text = f"{nl}L"
            else:
                text = f"{nl * layer_thickness_nm:.0f}nm"
            a = ax.annotate(
                text,
                (0, 0),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="0.3",
            )
            ann.append((nl, a))
    return ann, dots


def _update_annotations(ann_list, dots_artist, r_t, g_t):
    dg, dr = [], []
    for nl, ann in ann_list:
        idx = nl - 1
        if idx < len(r_t):
            gv, rv = g_t[idx], r_t[idx]
            dg.append(gv)
            dr.append(rv)
            ann.xy = (gv, rv)
            ann.set_visible(True)
        else:
            ann.set_visible(False)
    dots_artist.set_data(dg, dr)


# Axis positions (left, bottom, width, height)
_PLOT_BOTTOM = 0.38
_PLOT_HEIGHT = 0.52
_HBN_LEFT_POS = (0.06, _PLOT_BOTTOM, 0.42, _PLOT_HEIGHT)
_HBN_RIGHT_POS = (0.55, _PLOT_BOTTOM, 0.42, _PLOT_HEIGHT)
_SINGLE_POS = (0.08, _PLOT_BOTTOM, 0.86, _PLOT_HEIGHT)
# Park hidden panels off-canvas so they can't intercept mouse grabs (and
# silently eat clicks destined for sliders/buttons beneath them).
_OFFSCREEN_POS = (-2.0, -2.0, 0.01, 0.01)


# --- Background worker plumbing --------------------------------------------
@dataclass(frozen=True)
class _Vals:
    """Immutable snapshot of slider state + active tab."""

    tab: str
    illum: str
    trans: str
    n: float
    n_gr: float
    k: float
    oxide: float
    oxide_285: float
    oxide_wse2: float
    na: float
    r_off: float
    g_off: float
    r_off_gr: float
    g_off_gr: float


_EMPTY_ARR = np.zeros(0, dtype=float)


class _TabResult(NamedTuple):
    """Compute output for a single tab.

    Only the fields relevant to the tab that produced the result are
    populated; the rest stay as empty arrays.  Keeping them all as
    ``np.ndarray`` (rather than Optional) lets the drawing code pass them
    straight to matplotlib without narrowing dances.

    NamedTuple (not dataclass) because numpy-array defaults trip Python
    3.11+'s "unhashable default = mutable" check on dataclasses; NamedTuple
    has no such check and the shared default is safe since we never mutate.
    """

    # hBN tab has two panels' worth of data
    r90: np.ndarray = _EMPTY_ARR
    g90: np.ndarray = _EMPTY_ARR
    t_hbn: np.ndarray = _EMPTY_ARR
    r285: np.ndarray = _EMPTY_ARR
    g285: np.ndarray = _EMPTY_ARR
    # Graphene
    r_gr: np.ndarray = _EMPTY_ARR
    g_gr: np.ndarray = _EMPTY_ARR
    t_gr: np.ndarray = _EMPTY_ARR
    # WSe₂
    r_wse2: np.ndarray = _EMPTY_ARR
    g_wse2: np.ndarray = _EMPTY_ARR
    t_wse2: np.ndarray = _EMPTY_ARR


def _compute_for_tab(vals: _Vals, n_wse2_arr: np.ndarray) -> _TabResult:
    illum = _ILLUMINANTS[vals.illum]
    trans_factor = _TRANSMISSIONS[vals.trans].factor
    red_lit = illum.red_lit * trans_factor
    green_lit = illum.green_lit * trans_factor
    if vals.tab == "hbn":
        hbn_n = complex(vals.n)
        r90, g90, t_hbn = compute_rg(
            hbn_n, vals.oxide, vals.na, max_layers=HBN_MAX_LAYERS, red_lit=red_lit, green_lit=green_lit
        )
        r285, g285, _ = compute_rg(
            hbn_n, vals.oxide_285, vals.na, max_layers=HBN_MAX_LAYERS, red_lit=red_lit, green_lit=green_lit
        )
        return _TabResult(r90=r90, g90=g90, t_hbn=t_hbn, r285=r285, g285=g285)
    if vals.tab == "graphene":
        gr_n = vals.n_gr - 1j * vals.k
        rg, gg, tg = compute_rg(
            gr_n,
            vals.oxide,
            vals.na,
            layer_thickness_nm=_GRAPHENE_LAYER_THICKNESS_NM,
            max_layers=GRAPHENE_MAX_LAYERS,
            red_lit=red_lit,
            green_lit=green_lit,
        )
        return _TabResult(r_gr=rg, g_gr=gg, t_gr=tg)
    if vals.tab == "wse2":
        rw, gw, tw = compute_rg(
            n_wse2_arr,
            vals.oxide_wse2,
            vals.na,
            layer_thickness_nm=_WSE2_LAYER_THICKNESS_NM,
            max_layers=WSE2_MAX_LAYERS,
            red_lit=red_lit,
            green_lit=green_lit,
        )
        return _TabResult(r_wse2=rw, g_wse2=gw, t_wse2=tw)
    return _TabResult()


def main():
    fig = plt.figure(figsize=(15, 8))

    init = {
        "n": 2.152,
        # Graphite/graphene constant-n approximation (Bruna & Borini APL 2009,
        # Blake et al. APL 2007).  Real graphite is dispersive — n drops ~0.4
        # from 700→400 nm — so single (n,k) is a fit compromise across visible.
        "n_gr": 2.65,
        "k": 1.30,
        "oxide": 90.0,
        "oxide_285": 285.0,
        "oxide_wse2": 300.0,
        "na": 0.25,
        "r_off": 0.54,
        "g_off": -0.2,
        "r_off_gr": 0.0,
        "g_off_gr": 0.0,
    }

    n_wse2_arr = n_wse2(lamb)

    # --- Create all axes upfront ---
    ax_90 = fig.add_axes(_HBN_LEFT_POS)
    ax_285 = fig.add_axes(_HBN_RIGHT_POS)
    ax_gr = fig.add_axes(_SINGLE_POS)
    ax_wse2 = fig.add_axes(_SINGLE_POS)

    active_illum = {"name": "halogen_3200K"}
    active_trans = {"name": "off"}

    # --- Compute initial curves (synchronous so first frame is drawn) ---
    def _init_vals(tab: str) -> _Vals:
        return _Vals(
            tab=tab,
            illum=active_illum["name"],
            trans=active_trans["name"],
            n=init["n"],
            n_gr=init["n_gr"],
            k=init["k"],
            oxide=init["oxide"],
            oxide_285=init["oxide_285"],
            oxide_wse2=init["oxide_wse2"],
            na=init["na"],
            r_off=init["r_off"],
            g_off=init["g_off"],
            r_off_gr=init["r_off_gr"],
            g_off_gr=init["g_off_gr"],
        )

    init_vals_hbn = _init_vals("hbn")
    init_vals_gr = _init_vals("graphene")
    init_vals_wse2 = _init_vals("wse2")
    init_hbn = _compute_for_tab(init_vals_hbn, n_wse2_arr)
    init_gr = _compute_for_tab(init_vals_gr, n_wse2_arr)
    init_wse2 = _compute_for_tab(init_vals_wse2, n_wse2_arr)

    r90, g90 = init_hbn.r90, init_hbn.g90
    r285, g285 = init_hbn.r285, init_hbn.g285
    r_gr, g_gr = init_gr.r_gr, init_gr.g_gr
    r_wse2, g_wse2 = init_wse2.r_wse2, init_wse2.g_wse2

    # --- hBN 90nm panel ---
    _setup_panel(ax_90, f"hBN · {init['oxide']:.0f} nm SiO₂")
    (line_90,) = ax_90.plot(g90, r90, "k-", lw=1.5, label="Theory")
    ann_90, dots_90 = _make_annotations(ax_90, _HBN_LAYER_THICKNESS_NM, HBN_MAX_LAYERS)

    cal_r = _CAL_DATA[:, 1] + init["r_off"]
    cal_g = _CAL_DATA[:, 2] + init["g_off"]
    scat_90 = ax_90.scatter(
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
    cal_ann_90 = []
    for i, row in enumerate(_CAL_DATA):
        a = ax_90.annotate(
            f"{row[0]:.0f}",
            (cal_g[i], cal_r[i]),
            textcoords="offset points",
            xytext=(6, -6),
            fontsize=7,
            color="tab:orange",
            fontweight="bold",
        )
        cal_ann_90.append(a)
    rms_text_90 = ax_90.text(
        0.02,
        0.98,
        "",
        transform=ax_90.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )
    ax_90.legend(fontsize=9, loc="lower right")

    # --- hBN 285nm panel ---
    _setup_panel(ax_285, "hBN · 285 nm SiO₂")
    (line_285,) = ax_285.plot(g285, r285, "k-", lw=1.5, label="Theory")
    ann_285, dots_285 = _make_annotations(ax_285, _HBN_LAYER_THICKNESS_NM, HBN_MAX_LAYERS)

    if len(_CAL_DATA_285) > 0:
        cal285_r = _CAL_DATA_285[:, 1] + init["r_off"]
        cal285_g = _CAL_DATA_285[:, 2] + init["g_off"]
        scat_285 = ax_285.scatter(
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
        cal_ann_285 = []
        for i, row in enumerate(_CAL_DATA_285):
            a = ax_285.annotate(
                f"~{row[0]:.0f}",
                (cal285_g[i], cal285_r[i]),
                textcoords="offset points",
                xytext=(6, -6),
                fontsize=7,
                color="tab:red",
                fontweight="bold",
            )
            cal_ann_285.append(a)
    else:
        scat_285 = None
        cal_ann_285 = []
    rms_text_285 = ax_285.text(
        0.02,
        0.98,
        "",
        transform=ax_285.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )
    ax_285.legend(fontsize=9, loc="lower right")

    # --- Graphene 90nm panel ---
    _setup_panel(ax_gr, "Graphene · 90 nm SiO₂")
    # Tighter range than the default — graphene/graphite contrast tops out near unity.
    ax_gr.set_xlim(-1, 3)
    ax_gr.set_ylim(-1.5, 2)
    (line_gr,) = ax_gr.plot(g_gr, r_gr, "k-", lw=1.5, label="Theory")
    ann_gr, dots_gr = _make_annotations(ax_gr, _GRAPHENE_LAYER_THICKNESS_NM, GRAPHENE_MAX_LAYERS)

    if len(_CAL_DATA_GRAPHENE) > 0:
        cal_gr_r = _CAL_DATA_GRAPHENE[:, 1] + init["r_off_gr"]
        cal_gr_g = _CAL_DATA_GRAPHENE[:, 2] + init["g_off_gr"]
        scat_gr = ax_gr.scatter(
            cal_gr_g,
            cal_gr_r,
            c="tab:green",
            marker="D",
            s=50,
            zorder=5,
            edgecolors="k",
            linewidths=0.7,
            label="Graphene empirical",
        )
        cal_ann_gr = []
        for i, row in enumerate(_CAL_DATA_GRAPHENE):
            a = ax_gr.annotate(
                f"{row[0]:.0f}",
                (cal_gr_g[i], cal_gr_r[i]),
                textcoords="offset points",
                xytext=(6, -6),
                fontsize=7,
                color="tab:green",
                fontweight="bold",
            )
            cal_ann_gr.append(a)
    else:
        scat_gr = None
        cal_ann_gr = []

    # 10x graphite gate-flake overlay (split into two scatters: reliable vs R-clipped).
    cal10_r = _CAL_DATA_GRAPHITE_10X[:, 1] + init["r_off_gr"]
    cal10_g = _CAL_DATA_GRAPHITE_10X[:, 2] + init["g_off_gr"]
    rel = _CAL_DATA_GRAPHITE_10X_R_RELIABLE
    scat_gr_10x_ok = ax_gr.scatter(
        cal10_g[rel],
        cal10_r[rel],
        c="tab:cyan",
        marker="^",
        s=55,
        zorder=5,
        edgecolors="k",
        linewidths=0.7,
        label="10x graphite (AFM)",
    )
    scat_gr_10x_clip = ax_gr.scatter(
        cal10_g[~rel],
        cal10_r[~rel],
        facecolors="none",
        edgecolors="tab:cyan",
        marker="^",
        s=55,
        zorder=4,
        linewidths=1.2,
        alpha=0.55,
        label="10x · R clipped",
    )
    cal_ann_gr_10x = []
    for i, row in enumerate(_CAL_DATA_GRAPHITE_10X):
        a = ax_gr.annotate(
            f"{row[0]:.1f}",
            (cal10_g[i], cal10_r[i]),
            textcoords="offset points",
            xytext=(6, -6),
            fontsize=7,
            color="tab:cyan",
            fontweight="bold",
            alpha=1.0 if rel[i] else 0.55,
        )
        cal_ann_gr_10x.append(a)
    ax_gr.legend(fontsize=9, loc="lower right")

    # --- WSe₂ 300nm panel ---
    _setup_panel(ax_wse2, f"WSe₂ · {init['oxide_wse2']:.0f} nm SiO₂")
    (line_wse2,) = ax_wse2.plot(
        g_wse2[:WSE2_PLOT_LAYERS],
        r_wse2[:WSE2_PLOT_LAYERS],
        "k-",
        lw=1.5,
        label="Theory",
    )
    ann_wse2, dots_wse2 = _make_annotations(
        ax_wse2,
        _WSE2_LAYER_THICKNESS_NM,
        WSE2_MAX_LAYERS,
        label_layers=list(range(1, WSE2_PLOT_LAYERS + 1)),
        label_fmt="layers",
    )

    if len(_CAL_DATA_WSE2) > 0:
        cal_wse2_r = _CAL_DATA_WSE2[:, 1] + init["r_off"]
        cal_wse2_g = _CAL_DATA_WSE2[:, 2] + init["g_off"]
        scat_wse2 = ax_wse2.scatter(
            cal_wse2_g,
            cal_wse2_r,
            c="tab:purple",
            marker="^",
            s=50,
            zorder=5,
            edgecolors="k",
            linewidths=0.7,
            label="WSe₂ empirical",
        )
        cal_ann_wse2 = []
        for i, row in enumerate(_CAL_DATA_WSE2):
            a = ax_wse2.annotate(
                f"{round(row[0] / _WSE2_LAYER_THICKNESS_NM):.0f}L",
                (cal_wse2_g[i], cal_wse2_r[i]),
                textcoords="offset points",
                xytext=(6, -6),
                fontsize=7,
                color="tab:purple",
                fontweight="bold",
            )
            cal_ann_wse2.append(a)
    else:
        scat_wse2 = None
        cal_ann_wse2 = []
    ax_wse2.legend(fontsize=9, loc="lower right")

    # --- Tab buttons ---
    active_tab = {"name": "hbn"}

    ax_tab_hbn = fig.add_axes((0.06, 0.93, 0.08, 0.04))
    ax_tab_gr = fig.add_axes((0.15, 0.93, 0.10, 0.04))
    ax_tab_wse2 = fig.add_axes((0.26, 0.93, 0.08, 0.04))
    btn_hbn = Button(ax_tab_hbn, "hBN")
    btn_gr = Button(ax_tab_gr, "Graphene")
    btn_wse2 = Button(ax_tab_wse2, "WSe₂")

    # --- All sliders (created upfront, visibility toggled per tab) ---
    slider_defs = {
        "n": ("n (real)", 1.4, 3.0, init["n"]),
        "n_gr": ("n (real, gr)", 1.4, 3.0, init["n_gr"]),
        "k": ("k (imag)", 0.0, 4.0, init["k"]),
        "oxide": ("t_oxide (nm)", 70, 120, init["oxide"]),
        "oxide_285": ("t_285 (nm)", 250, 310, init["oxide_285"]),
        "oxide_wse2": ("t_wse2 (nm)", 270, 330, init["oxide_wse2"]),
        "na": ("NA", 0.0, 0.9, init["na"]),
        "r_off": ("R offset", -1.0, 1.5, init["r_off"]),
        "g_off": ("G offset", -1.0, 1.0, init["g_off"]),
        "r_off_gr": ("R offset (gr)", -1.0, 1.5, init["r_off_gr"]),
        "g_off_gr": ("G offset (gr)", -1.0, 1.0, init["g_off_gr"]),
    }

    # Which sliders each tab uses
    tab_sliders = {
        "hbn": ["n", "oxide", "oxide_285", "na", "r_off", "g_off"],
        "graphene": ["n_gr", "k", "oxide", "na", "r_off_gr", "g_off_gr"],
        "wse2": ["oxide_wse2", "na", "r_off", "g_off"],
    }

    # Create all slider widgets (initially invisible, positioned later)
    all_sliders: dict[str, tuple[plt.Axes, Slider]] = {}
    for key, (label, vmin, vmax, vinit) in slider_defs.items():
        ax_s = fig.add_axes((0.10, 0.01, 0.80, 0.022))  # placeholder position
        ax_s.set_visible(False)
        vstep = 0.01 if "offset" in label else None
        s = Slider(ax_s, label, vmin, vmax, valinit=vinit, valstep=vstep)
        all_sliders[key] = (ax_s, s)

    def _slider(key: str) -> Slider:
        return all_sliders[key][1]

    def _read_vals(tab: str) -> _Vals:
        return _Vals(
            tab=tab,
            illum=active_illum["name"],
            trans=active_trans["name"],
            n=_slider("n").val,
            n_gr=_slider("n_gr").val,
            k=_slider("k").val,
            oxide=_slider("oxide").val,
            oxide_285=_slider("oxide_285").val,
            oxide_wse2=_slider("oxide_wse2").val,
            na=_slider("na").val,
            r_off=_slider("r_off").val,
            g_off=_slider("g_off").val,
            r_off_gr=_slider("r_off_gr").val,
            g_off_gr=_slider("g_off_gr").val,
        )

    def _active_offsets(vals: _Vals) -> tuple[float, float]:
        """Return (r_off, g_off) for the tab's overlay scatters."""
        if vals.tab == "graphene":
            return vals.r_off_gr, vals.g_off_gr
        return vals.r_off, vals.g_off

    # --- Compute cache: last valid result per tab (keyed by inputs that affect it) ---
    cache: dict[str, tuple[tuple, _TabResult]] = {
        "hbn": (
            (init["n"], init["oxide"], init["oxide_285"], init["na"], active_illum["name"], active_trans["name"]),
            init_hbn,
        ),
        "graphene": (
            (init["n_gr"], init["k"], init["oxide"], init["na"], active_illum["name"], active_trans["name"]),
            init_gr,
        ),
        "wse2": (
            (init["oxide_wse2"], init["na"], active_illum["name"], active_trans["name"]),
            init_wse2,
        ),
    }

    def _compute_key(tab: str, vals: _Vals) -> tuple:
        if tab == "hbn":
            return (vals.n, vals.oxide, vals.oxide_285, vals.na, vals.illum, vals.trans)
        if tab == "graphene":
            return (vals.n_gr, vals.k, vals.oxide, vals.na, vals.illum, vals.trans)
        return (vals.oxide_wse2, vals.na, vals.illum, vals.trans)

    # --- Background worker plumbing ---
    # _pending holds the latest desired _Vals (mailbox size 1, last wins).
    # _result holds the most recent completed (_Vals, _TabResult) for the main
    # thread to pick up.
    lock = threading.Lock()
    cond = threading.Condition(lock)
    _pending: list[_Vals | None] = [None]
    _result: list[tuple[_Vals, _TabResult] | None] = [None]
    _shutdown = [False]

    def worker():
        while True:
            with cond:
                while _pending[0] is None and not _shutdown[0]:
                    cond.wait()
                if _shutdown[0]:
                    return
                vals = _pending[0]
                _pending[0] = None
            assert vals is not None  # guaranteed by the wait() loop above
            # Compute outside the lock.
            try:
                r = _compute_for_tab(vals, n_wse2_arr)
            except Exception as e:
                print(f"widget worker error: {e}")
                continue
            with cond:
                _result[0] = (vals, r)

    worker_thread = threading.Thread(target=worker, daemon=True, name="contrast-worker")
    worker_thread.start()

    def _request_compute(vals: _Vals):
        # Skip if this tab's cache is already up to date.
        key = _compute_key(vals.tab, vals)
        cached_key, _ = cache[vals.tab]
        if key == cached_key:
            return
        with cond:
            _pending[0] = vals
            cond.notify()

    # --- Artist updates (main thread only) ---
    def _draw_hbn(vals: _Vals, res: _TabResult):
        r9, g9, t = res.r90, res.g90, res.t_hbn
        r2, g2 = res.r285, res.g285

        line_90.set_data(g9, r9)
        _update_annotations(ann_90, dots_90, r9, g9)
        ax_90.set_title(f"hBN · {vals.oxide:.0f} nm SiO₂")

        cal_r_s = _CAL_DATA[:, 1] + vals.r_off
        cal_g_s = _CAL_DATA[:, 2] + vals.g_off
        scat_90.set_offsets(np.column_stack([cal_g_s, cal_r_s]))
        for i, ann in enumerate(cal_ann_90):
            ann.xy = (cal_g_s[i], cal_r_s[i])
        r_res = cal_r_s - np.interp(_CAL_DATA[:, 0], t, r9)
        g_res = cal_g_s - np.interp(_CAL_DATA[:, 0], t, g9)
        rms_text_90.set_text(f"R rms={np.sqrt(np.mean(r_res**2)):.3f}  G rms={np.sqrt(np.mean(g_res**2)):.3f}")

        line_285.set_data(g2, r2)
        ax_285.set_title(f"hBN · {vals.oxide_285:.0f} nm SiO₂")
        _update_annotations(ann_285, dots_285, r2, g2)
        if scat_285 is not None:
            c285_r = _CAL_DATA_285[:, 1] + vals.r_off
            c285_g = _CAL_DATA_285[:, 2] + vals.g_off
            scat_285.set_offsets(np.column_stack([c285_g, c285_r]))
            for i, ann in enumerate(cal_ann_285):
                ann.xy = (c285_g[i], c285_r[i])
            r_res_285 = c285_r - np.interp(_CAL_DATA_285[:, 0], t, r2)
            g_res_285 = c285_g - np.interp(_CAL_DATA_285[:, 0], t, g2)
            rms_text_285.set_text(
                f"R rms={np.sqrt(np.mean(r_res_285**2)):.3f}  G rms={np.sqrt(np.mean(g_res_285**2)):.3f}"
            )

    def _draw_graphene(vals: _Vals, res: _TabResult):
        line_gr.set_data(res.g_gr, res.r_gr)
        _update_annotations(ann_gr, dots_gr, res.r_gr, res.g_gr)
        if scat_gr is not None:
            cgr_r = _CAL_DATA_GRAPHENE[:, 1] + vals.r_off_gr
            cgr_g = _CAL_DATA_GRAPHENE[:, 2] + vals.g_off_gr
            scat_gr.set_offsets(np.column_stack([cgr_g, cgr_r]))
            for i, ann in enumerate(cal_ann_gr):
                ann.xy = (cgr_g[i], cgr_r[i])
        cal10_r_s = _CAL_DATA_GRAPHITE_10X[:, 1] + vals.r_off_gr
        cal10_g_s = _CAL_DATA_GRAPHITE_10X[:, 2] + vals.g_off_gr
        scat_gr_10x_ok.set_offsets(np.column_stack([cal10_g_s[rel], cal10_r_s[rel]]))
        scat_gr_10x_clip.set_offsets(np.column_stack([cal10_g_s[~rel], cal10_r_s[~rel]]))
        for i, ann in enumerate(cal_ann_gr_10x):
            ann.xy = (cal10_g_s[i], cal10_r_s[i])

    def _draw_wse2(vals: _Vals, res: _TabResult):
        rw, gw = res.r_wse2, res.g_wse2
        line_wse2.set_data(gw[:WSE2_PLOT_LAYERS], rw[:WSE2_PLOT_LAYERS])
        ax_wse2.set_title(f"WSe₂ · {vals.oxide_wse2:.0f} nm SiO₂")
        _update_annotations(ann_wse2, dots_wse2, rw, gw)
        if scat_wse2 is not None:
            cwse2_r = _CAL_DATA_WSE2[:, 1] + vals.r_off
            cwse2_g = _CAL_DATA_WSE2[:, 2] + vals.g_off
            scat_wse2.set_offsets(np.column_stack([cwse2_g, cwse2_r]))
            for i, ann in enumerate(cal_ann_wse2):
                ann.xy = (cwse2_g[i], cwse2_r[i])

    def _draw_active(vals: _Vals, res: _TabResult):
        if vals.tab == "hbn":
            _draw_hbn(vals, res)
        elif vals.tab == "graphene":
            _draw_graphene(vals, res)
        elif vals.tab == "wse2":
            _draw_wse2(vals, res)

    def _draw_offsets_only(vals: _Vals):
        """Update only the offset-dependent artists on the active tab."""
        if vals.tab == "hbn":
            _, res = cache["hbn"]
            cal_r_s = _CAL_DATA[:, 1] + vals.r_off
            cal_g_s = _CAL_DATA[:, 2] + vals.g_off
            scat_90.set_offsets(np.column_stack([cal_g_s, cal_r_s]))
            for i, ann in enumerate(cal_ann_90):
                ann.xy = (cal_g_s[i], cal_r_s[i])
            r_res = cal_r_s - np.interp(_CAL_DATA[:, 0], res.t_hbn, res.r90)
            g_res = cal_g_s - np.interp(_CAL_DATA[:, 0], res.t_hbn, res.g90)
            rms_text_90.set_text(f"R rms={np.sqrt(np.mean(r_res**2)):.3f}  G rms={np.sqrt(np.mean(g_res**2)):.3f}")
            if scat_285 is not None:
                c285_r = _CAL_DATA_285[:, 1] + vals.r_off
                c285_g = _CAL_DATA_285[:, 2] + vals.g_off
                scat_285.set_offsets(np.column_stack([c285_g, c285_r]))
                for i, ann in enumerate(cal_ann_285):
                    ann.xy = (c285_g[i], c285_r[i])
                r_res_285 = c285_r - np.interp(_CAL_DATA_285[:, 0], res.t_hbn, res.r285)
                g_res_285 = c285_g - np.interp(_CAL_DATA_285[:, 0], res.t_hbn, res.g285)
                rms_text_285.set_text(
                    f"R rms={np.sqrt(np.mean(r_res_285**2)):.3f}  G rms={np.sqrt(np.mean(g_res_285**2)):.3f}"
                )
        elif vals.tab == "graphene":
            if scat_gr is not None:
                cgr_r = _CAL_DATA_GRAPHENE[:, 1] + vals.r_off_gr
                cgr_g = _CAL_DATA_GRAPHENE[:, 2] + vals.g_off_gr
                scat_gr.set_offsets(np.column_stack([cgr_g, cgr_r]))
                for i, ann in enumerate(cal_ann_gr):
                    ann.xy = (cgr_g[i], cgr_r[i])
            cal10_r_s = _CAL_DATA_GRAPHITE_10X[:, 1] + vals.r_off_gr
            cal10_g_s = _CAL_DATA_GRAPHITE_10X[:, 2] + vals.g_off_gr
            scat_gr_10x_ok.set_offsets(np.column_stack([cal10_g_s[rel], cal10_r_s[rel]]))
            scat_gr_10x_clip.set_offsets(np.column_stack([cal10_g_s[~rel], cal10_r_s[~rel]]))
            for i, ann in enumerate(cal_ann_gr_10x):
                ann.xy = (cal10_g_s[i], cal10_r_s[i])
        elif vals.tab == "wse2" and scat_wse2 is not None:
            cwse2_r = _CAL_DATA_WSE2[:, 1] + vals.r_off
            cwse2_g = _CAL_DATA_WSE2[:, 2] + vals.g_off
            scat_wse2.set_offsets(np.column_stack([cwse2_g, cwse2_r]))
            for i, ann in enumerate(cal_ann_wse2):
                ann.xy = (cwse2_g[i], cwse2_r[i])

    # --- Main-thread timer: polls worker result, updates artists ---
    last_drawn_offsets = {"r": init["r_off"], "g": init["g_off"], "tab": "hbn"}

    def _poll_result():
        with cond:
            r = _result[0]
            _result[0] = None
        vals_now = _read_vals(active_tab["name"])
        dirty = False

        if r is not None:
            vals, res = r
            cache[vals.tab] = (_compute_key(vals.tab, vals), res)
            # Only draw if the result is still for the active tab.
            if vals.tab == active_tab["name"]:
                _draw_active(vals, res)
                r_off, g_off = _active_offsets(vals)
                last_drawn_offsets["r"] = r_off
                last_drawn_offsets["g"] = g_off
                last_drawn_offsets["tab"] = vals.tab
                dirty = True

        # Offset-only redraw: if offsets changed since last draw and current tab's
        # compute is up-to-date, update the scatter/rms without recomputing.
        key = _compute_key(vals_now.tab, vals_now)
        cached_key, _ = cache[vals_now.tab]
        r_off_now, g_off_now = _active_offsets(vals_now)
        if key == cached_key and (
            r_off_now != last_drawn_offsets["r"]
            or g_off_now != last_drawn_offsets["g"]
            or last_drawn_offsets["tab"] != vals_now.tab
        ):
            _draw_offsets_only(vals_now)
            last_drawn_offsets["r"] = r_off_now
            last_drawn_offsets["g"] = g_off_now
            last_drawn_offsets["tab"] = vals_now.tab
            dirty = True

        if dirty:
            fig.canvas.draw_idle()

    poll_timer = fig.canvas.new_timer(interval=16)
    poll_timer.add_callback(_poll_result)
    poll_timer.start()

    # --- Slider and tab-switch callbacks ---
    def on_slider(_val=None):
        vals = _read_vals(active_tab["name"])
        _request_compute(vals)

    def _layout_sliders(tab_name: str):
        keys = tab_sliders[tab_name]
        for key, (ax_s, _) in all_sliders.items():
            ax_s.set_visible(key in keys)
        for i, key in enumerate(keys):
            ax_s, _ = all_sliders[key]
            ax_s.set_position((0.10, 0.26 - i * 0.033, 0.80, 0.022))

    def show_tab(name):
        active_tab["name"] = name
        is_hbn = name == "hbn"
        is_gr = name == "graphene"
        is_wse2 = name == "wse2"
        ax_90.set_visible(is_hbn)
        ax_285.set_visible(is_hbn)
        ax_gr.set_visible(is_gr)
        ax_wse2.set_visible(is_wse2)
        ax_90.set_position(_HBN_LEFT_POS if is_hbn else _OFFSCREEN_POS)
        ax_285.set_position(_HBN_RIGHT_POS if is_hbn else _OFFSCREEN_POS)
        ax_gr.set_position(_SINGLE_POS if is_gr else _OFFSCREEN_POS)
        ax_wse2.set_position(_SINGLE_POS if is_wse2 else _OFFSCREEN_POS)
        ax_tab_hbn.set_facecolor("0.85" if is_hbn else "0.95")
        ax_tab_gr.set_facecolor("0.85" if is_gr else "0.95")
        ax_tab_wse2.set_facecolor("0.85" if is_wse2 else "0.95")
        _layout_sliders(name)

        # Redraw from cache for the new tab, then request a fresh compute if stale.
        vals = _read_vals(name)
        _, cached_res = cache[name]
        _draw_active(vals, cached_res)
        r_off, g_off = _active_offsets(vals)
        last_drawn_offsets["r"] = r_off
        last_drawn_offsets["g"] = g_off
        last_drawn_offsets["tab"] = name
        _request_compute(vals)
        fig.canvas.draw_idle()

    for _, s in all_sliders.values():
        s.on_changed(on_slider)

    btn_hbn.on_clicked(lambda _: show_tab("hbn"))
    btn_gr.on_clicked(lambda _: show_tab("graphene"))
    btn_wse2.on_clicked(lambda _: show_tab("wse2"))

    # Illuminant toggle — cycles through _ILLUMINANTS and re-renders the active tab.
    ax_illum = fig.add_axes((0.40, 0.93, 0.18, 0.04))
    btn_illum = Button(ax_illum, f"Illum: {_ILLUMINANTS[active_illum['name']].label}")

    def toggle_illum(_event=None):
        names = list(_ILLUMINANTS.keys())
        active_illum["name"] = names[(names.index(active_illum["name"]) + 1) % len(names)]
        btn_illum.label.set_text(f"Illum: {_ILLUMINANTS[active_illum['name']].label}")
        # Request a fresh compute for the active tab; the new illum name is in
        # the cache key so the request will miss and dispatch.  Other tabs will
        # recompute lazily on their next show.
        _request_compute(_read_vals(active_tab["name"]))
        fig.canvas.draw_idle()

    btn_illum.on_clicked(toggle_illum)

    # Objective transmission toggle — multiplies bandpass×illuminant by T(λ)².
    ax_trans = fig.add_axes((0.59, 0.93, 0.16, 0.04))
    btn_trans = Button(ax_trans, f"Trans: {_TRANSMISSIONS[active_trans['name']].label}")

    def toggle_trans(_event=None):
        names = list(_TRANSMISSIONS.keys())
        active_trans["name"] = names[(names.index(active_trans["name"]) + 1) % len(names)]
        btn_trans.label.set_text(f"Trans: {_TRANSMISSIONS[active_trans['name']].label}")
        _request_compute(_read_vals(active_tab["name"]))
        fig.canvas.draw_idle()

    btn_trans.on_clicked(toggle_trans)

    # Export button — lives in the tab-button row so it's always reachable
    ax_btn = fig.add_axes((0.88, 0.93, 0.10, 0.04))
    btn_export = Button(ax_btn, "Export all")
    export_path = Path("/tmp/hbn_contrast_params.json")

    def export(_event=None):
        r_off = _slider("r_off").val
        g_off = _slider("g_off").val
        r_off_gr = _slider("r_off_gr").val
        g_off_gr = _slider("g_off_gr").val

        # With lazy per-tab compute, non-active tabs can carry stale cache
        # entries.  Synchronously refresh any tab whose cached inputs don't
        # match the current sliders before we dump the file.
        for tab_name in ("hbn", "graphene", "wse2"):
            tab_vals = _read_vals(tab_name)
            key = _compute_key(tab_name, tab_vals)
            cached_key, _ = cache[tab_name]
            if key != cached_key:
                cache[tab_name] = (key, _compute_for_tab(tab_vals, n_wse2_arr))

        def _sample_cal(r_arr, g_arr, t_arr, r_off, g_off):
            cal = []
            for t_nm_val in np.arange(2.0, 102.0, 2.0):
                if t_nm_val > t_arr[-1]:
                    break
                r_val = float(np.interp(t_nm_val, t_arr, r_arr))
                g_val = float(np.interp(t_nm_val, t_arr, g_arr))
                cal.append([round(r_val + r_off, 4), round(g_val + g_off, 4), t_nm_val])
            return cal

        # Use cached results where available; recompute tabs whose cache is stale.
        hbn_res = cache["hbn"][1]
        gr_res = cache["graphene"][1]
        wse2_res = cache["wse2"][1]

        hbn_90_cal = _sample_cal(hbn_res.r90, hbn_res.g90, hbn_res.t_hbn, r_off, g_off)
        graphene_cal = _sample_cal(gr_res.r_gr, gr_res.g_gr, gr_res.t_gr, r_off_gr, g_off_gr)
        wse2_cal = _sample_cal(wse2_res.r_wse2, wse2_res.g_wse2, wse2_res.t_wse2, r_off, g_off)

        params = {
            "n_hbn": _slider("n").val,
            "t_oxide_90": _slider("oxide").val,
            "t_oxide_285": _slider("oxide_285").val,
            "t_oxide_wse2": _slider("oxide_wse2").val,
            "na": _slider("na").val,
            "r_offset": r_off,
            "g_offset": g_off,
            "graphene_n_re": _slider("n_gr").val,
            "graphene_k": _slider("k").val,
            "graphene_r_offset": r_off_gr,
            "graphene_g_offset": g_off_gr,
            "hbn_90_cal_points": hbn_90_cal,
            "graphene_cal_points": graphene_cal,
            "wse2_cal_points": wse2_cal,
        }
        export_path.write_text(json.dumps(params, indent=2) + "\n")
        print(f"Exported to {export_path}:", flush=True)
        print(json.dumps(params, indent=2), flush=True)

    btn_export.on_clicked(export)

    show_tab("hbn")

    try:
        plt.show()
    finally:
        with cond:
            _shutdown[0] = True
            cond.notify_all()


if __name__ == "__main__":
    main()
