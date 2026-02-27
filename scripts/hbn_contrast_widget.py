"""Interactive widget for exploring hBN R/G contrast parameters.

Two panels: ~90nm oxide (with AFM data) and 285nm oxide (with empirical data).
Sliders for n, oxide thickness, NA, and R/G offsets.
Uses vectorized transfer matrix for fast updates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

sys.path.insert(0, str(Path(__file__).parent))
from hbn_contrast import (
    _CAL_DATA,
    _CAL_DATA_285,
    _HBN_LAYER_THICKNESS_NM,
    _IMX183_GREEN,
    _IMX183_RED,
    _IMX183_WAVELENGTHS,
    _SI_DATA,
    _SIO2_DATA,
    _interp_index,
)

lamb = _IMX183_WAVELENGTHS
n_sio2 = _interp_index(lamb, _SIO2_DATA)
n_si = _interp_index(lamb, _SI_DATA)

MAX_LAYERS = 300
LABEL_LAYERS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200, 250, 300]
T_OXIDE_285 = 285.0


def compute_rg(
    n_hbn: float,
    t_oxide: float,
    na: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch-compute R/G contrast for layers 1..MAX_LAYERS.

    Vectorised over layer count for fast interactive updates.
    Returns (r_contrast, g_contrast, thickness_nm).
    """
    n1 = np.full(len(lamb), n_hbn, dtype=complex)

    if na <= 0:
        thetas = np.array([0.0])
        raw_weights = np.array([1.0])
    else:
        theta_max = np.arcsin(na)
        nodes, ws = np.polynomial.legendre.leggauss(11)
        thetas = 0.5 * theta_max * (nodes + 1)
        raw_weights = 0.5 * theta_max * ws * np.sin(thetas) * np.cos(thetas)

    total_w = raw_weights.sum()

    nl_arr = np.arange(1, MAX_LAYERS + 1)
    t_films = nl_arr * _HBN_LAYER_THICKNESS_NM

    R_flake = np.zeros((MAX_LAYERS, len(lamb)))
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

            eb1pb2 = np.exp(1j * (beta1_all + beta2[None, :]))
            eb1mb2 = np.exp(1j * (beta1_all - beta2[None, :]))
            num = r01 * eb1pb2 + r12 * np.conj(eb1mb2) + r23 * np.conj(eb1pb2) + r01 * r12 * r23 * eb1mb2
            den = eb1pb2 + r01 * r12 * np.conj(eb1mb2) + r01 * r23 * np.conj(eb1pb2) + r12 * r23 * eb1mb2
            r = num / den
            R_flake += 0.5 * w * (r * r.conjugate()).real

            num_s = r_as * eb2 + r23 * eb2c
            den_s = eb2 + r_as * r23 * eb2c
            r_s = num_s / den_s
            R_sub += 0.5 * w * (r_s * r_s.conjugate()).real

    R_flake /= total_w
    R_sub /= total_w

    V_sub_r = np.trapezoid(R_sub * _IMX183_RED, lamb)
    V_sub_g = np.trapezoid(R_sub * _IMX183_GREEN, lamb)
    V_flake_r = np.trapezoid(R_flake * _IMX183_RED[None, :], lamb, axis=1)
    V_flake_g = np.trapezoid(R_flake * _IMX183_GREEN[None, :], lamb, axis=1)

    return (
        (V_flake_r - V_sub_r) / V_sub_r,
        (V_flake_g - V_sub_g) / V_sub_g,
        t_films,
    )


def _setup_panel(ax, title):
    """Common axis setup for both panels."""
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("Green contrast")
    ax.set_ylabel("Red contrast")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 5.5)
    ax.set_ylim(-1.5, 4)


def main():
    fig, (ax_90, ax_285) = plt.subplots(1, 2, figsize=(15, 7))
    plt.subplots_adjust(bottom=0.32, wspace=0.25)

    init = {"n": 2.152, "oxide": 89.46, "oxide_285": 285.0, "na": 0.25, "r_off": 0.54, "g_off": -0.2}

    # --- Compute initial curves ---
    r90, g90, t_nm = compute_rg(init["n"], init["oxide"], init["na"])
    r285, g285, _ = compute_rg(init["n"], init["oxide_285"], init["na"])

    # --- Left panel: ~90nm ---
    _setup_panel(ax_90, f"{init['oxide']:.0f} nm SiO₂")
    (line_90,) = ax_90.plot(g90, r90, "k-", lw=1.5, label="Theory")

    ann_90 = []
    dots_90 = ax_90.plot([], [], "ko", markersize=4, zorder=3)[0]
    for nl in LABEL_LAYERS:
        if nl <= MAX_LAYERS:
            a = ax_90.annotate(
                f"{nl * _HBN_LAYER_THICKNESS_NM:.0f}nm",
                (0, 0),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="0.3",
            )
            ann_90.append((nl, a))

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

    # --- Right panel: 285nm ---
    _setup_panel(ax_285, "285 nm SiO₂")
    (line_285,) = ax_285.plot(g285, r285, "k-", lw=1.5, label="Theory")

    ann_285 = []
    dots_285 = ax_285.plot([], [], "ko", markersize=4, zorder=3)[0]
    for nl in LABEL_LAYERS:
        if nl <= MAX_LAYERS:
            a = ax_285.annotate(
                f"{nl * _HBN_LAYER_THICKNESS_NM:.0f}nm",
                (0, 0),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="0.3",
            )
            ann_285.append((nl, a))

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

    # --- Sliders ---
    slider_specs = [
        ("n_hBN", 1.4, 2.8, init["n"]),
        ("t_oxide (nm)", 70, 120, init["oxide"]),
        ("t_285 (nm)", 250, 310, init["oxide_285"]),
        ("NA", 0.0, 0.9, init["na"]),
        ("R offset", -1.0, 1.5, init["r_off"]),
        ("G offset", -1.0, 1.0, init["g_off"]),
    ]
    sliders = []
    for i, (label, vmin, vmax, vinit) in enumerate(slider_specs):
        ax_s = plt.axes((0.10, 0.20 - i * 0.033, 0.80, 0.022))
        s = Slider(ax_s, label, vmin, vmax, valinit=vinit, valstep=0.01 if "offset" in label else None)
        sliders.append(s)

    cache = {
        "n": init["n"],
        "oxide": init["oxide"],
        "oxide_285": init["oxide_285"],
        "na": init["na"],
        "r90": r90,
        "g90": g90,
        "r285": r285,
        "g285": g285,
        "t_nm": t_nm,
    }

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

    def update(_val=None):
        n_val = sliders[0].val
        oxide_val = sliders[1].val
        oxide_285_val = sliders[2].val
        na_val = sliders[3].val
        r_off = sliders[4].val
        g_off = sliders[5].val

        # Check what changed before updating cache
        left_changed = n_val != cache["n"] or oxide_val != cache["oxide"] or na_val != cache["na"]
        right_changed = n_val != cache["n"] or oxide_285_val != cache["oxide_285"] or na_val != cache["na"]

        if left_changed:
            r9, g9, t = compute_rg(n_val, oxide_val, na_val)
            cache.update(n=n_val, oxide=oxide_val, na=na_val, r90=r9, g90=g9, t_nm=t)

        if right_changed:
            r2, g2, _ = compute_rg(n_val, oxide_285_val, na_val)
            cache.update(oxide_285=oxide_285_val, r285=r2, g285=g2)

        r9 = cache["r90"]
        g9 = cache["g90"]
        r2 = cache["r285"]
        g2 = cache["g285"]
        t = cache["t_nm"]

        # Left panel: ~90nm
        line_90.set_data(g9, r9)
        _update_annotations(ann_90, dots_90, r9, g9)
        ax_90.set_title(f"{oxide_val:.0f} nm SiO₂")

        cal_r_s = _CAL_DATA[:, 1] + r_off
        cal_g_s = _CAL_DATA[:, 2] + g_off
        scat_90.set_offsets(np.column_stack([cal_g_s, cal_r_s]))
        for i, ann in enumerate(cal_ann_90):
            ann.xy = (cal_g_s[i], cal_r_s[i])

        r_res = cal_r_s - np.interp(_CAL_DATA[:, 0], t, r9)
        g_res = cal_g_s - np.interp(_CAL_DATA[:, 0], t, g9)
        rms_text_90.set_text(f"R rms={np.sqrt(np.mean(r_res**2)):.3f}  G rms={np.sqrt(np.mean(g_res**2)):.3f}")

        # Right panel: 285nm
        line_285.set_data(g2, r2)
        ax_285.set_title(f"{oxide_285_val:.0f} nm SiO₂")
        _update_annotations(ann_285, dots_285, r2, g2)

        if scat_285 is not None:
            c285_r = _CAL_DATA_285[:, 1] + r_off
            c285_g = _CAL_DATA_285[:, 2] + g_off
            scat_285.set_offsets(np.column_stack([c285_g, c285_r]))
            for i, ann in enumerate(cal_ann_285):
                ann.xy = (c285_g[i], c285_r[i])

            r_res_285 = c285_r - np.interp(_CAL_DATA_285[:, 0], t, r2)
            g_res_285 = c285_g - np.interp(_CAL_DATA_285[:, 0], t, g2)
            rms_text_285.set_text(
                f"R rms={np.sqrt(np.mean(r_res_285**2)):.3f}  G rms={np.sqrt(np.mean(g_res_285**2)):.3f}"
            )

        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    # Export button
    ax_btn = plt.axes((0.82, 0.005, 0.12, 0.03))
    btn = Button(ax_btn, "Export")
    export_path = Path("/tmp/hbn_contrast_params.json")

    def export(_event=None):
        params = {
            "n_hbn": sliders[0].val,
            "t_oxide_90": sliders[1].val,
            "t_oxide_285": sliders[2].val,
            "na": sliders[3].val,
            "r_offset": sliders[4].val,
            "g_offset": sliders[5].val,
        }
        export_path.write_text(json.dumps(params, indent=2) + "\n")
        print(f"Exported to {export_path}:")
        print(json.dumps(params, indent=2))

    btn.on_clicked(export)

    update()  # draw initial RMS values
    plt.show()


if __name__ == "__main__":
    main()
