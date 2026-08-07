"""Graphene/graphite presets under the measured K5C pipeline model (2026-08-06).

Read-only analysis — no preset is modified.  Quantifies how the measured
pipeline (Elijah lamp x per-objective T^2, 5800K CCM after WB, veiling-glare
floors; scripts/colorchecker_cal.py) moves predicted (R, G) contrast for
thin graphene (1-4L) and the gate-thick band (5-10 nm) on 90 nm SiO2,
compares against the empirical anchors each preset rests on, and projects
the shifts through the deployed preset gates.

Key structural point: with a CCM the contrast normalisation no longer
cancels WB — output contrast becomes a weighted mix of the raw channel
contrasts with weights M_cj * g_j * V_j.  The two graphene presets scan at
different WBs (thin: hBN 1.41/1.02/2.51; thick: 1.4/1.0/1.7), so they live
in different output contrast spaces on the same scope.

Run directly for the comparison tables; --plot / --output for the R-G figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from colorchecker_cal import OBJECTIVE_NA, WB_REQUESTS_HBN_SCAN, derive_pipeline
from hbn_contrast import (
    _CAL_DATA_GRAPHENE,
    _IMX183_BLUE,
    _IMX183_GREEN,
    _IMX183_RED,
    _IMX183_WAVELENGTHS,
    _blackbody,
    reflectance,
)
from hbn_contrast_widget import (
    _CAL_DATA_GRAPHENE_10X,
    _CAL_DATA_GRAPHITE_10X,
    _CAL_DATA_GRAPHITE_10X_R_RELIABLE,
)

from flakefinder.segmentation import GRAPHENE_THICK_90NM_CAL_POINTS, DetectorConfig

LAMB = _IMX183_WAVELENGTHS
BP = np.vstack([_IMX183_RED, _IMX183_GREEN, _IMX183_BLUE])

PIPE = derive_pipeline(LAMB)
HALOGEN = _blackbody(LAMB, 3200.0)

N_LIT = 2.6 - 1.3j  # Weber et al. 2010 graphene @ 550 nm (2DMatGMM ref [36])
N_PRESET_THICK = 2.817 - 1.317j  # hand-fit behind GRAPHENE_THICK_90NM_CAL_POINTS
T_OXIDE = 90.0

_thin_cfg = DetectorConfig.graphene_thin_90nm()
_thick_cfg = DetectorConfig.graphene_thick_90nm()
WB_THIN = np.array(_thin_cfg.white_balance)  # (1.41, 1.02, 2.51) — hBN scan WB
WB_THICK = np.array(_thick_cfg.white_balance)  # (1.4, 1.0, 1.7)

THIN_LAYER_NM = _thin_cfg.layer_spacing_nm or 0.335


def _const_n(nc: complex):
    def _n(lamb: np.ndarray) -> np.ndarray:
        return np.full_like(lamb, nc, dtype=complex)

    return _n


def channel_signals(
    nc: complex, thicknesses_nm: np.ndarray, t_oxide: float, na: float, chain: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Raw sensor channel signals for film thicknesses + bare substrate.

    Returns (V_flake (3, N), V_sub (3,)) — integrals of reflectance x
    bandpass x illumination chain.  Arbitrary thickness via n_layers=1
    with layer_thickness_nm=t.
    """
    lits = BP * chain
    r_sub = reflectance(LAMB, _const_n(nc), 0, t_oxide, na=na)
    v_sub = np.trapezoid(r_sub[None, :] * lits, LAMB, axis=1)
    v_flake = np.empty((3, len(thicknesses_nm)))
    for j, t in enumerate(thicknesses_nm):
        r = reflectance(LAMB, _const_n(nc), 1, t_oxide, na=na, layer_thickness_nm=float(t))
        v_flake[:, j] = np.trapezoid(r[None, :] * lits, LAMB, axis=1)
    return v_flake, v_sub


def pipeline_contrast(
    v_flake: np.ndarray,
    v_sub: np.ndarray,
    glare_f: np.ndarray | None = None,
    mix: np.ndarray | None = None,
) -> np.ndarray:
    """(3, N) per-channel contrast with optional glare (signal space) + mix."""
    if glare_f is not None:
        v_flake = v_flake + glare_f[:, None] * v_sub[:, None]
        v_sub = v_sub * (1.0 + glare_f)
    if mix is not None:
        v_flake = mix @ v_flake
        v_sub = mix @ v_sub
    return (v_flake - v_sub[:, None]) / v_sub[:, None]


def predict(
    thicknesses_nm: np.ndarray,
    nc: complex,
    objective: str = "10x",
    wb: np.ndarray | None = None,
    glare: bool = False,
    t_oxide: float = T_OXIDE,
    ccm: np.ndarray | None = None,
) -> np.ndarray:
    """Full-model (3, N) contrast for one objective.

    wb=None -> raw channel space (no CCM); wb given -> output space via
    CCM @ diag(wb).  glare uses the objective's raw-space floors.
    """
    op = PIPE.objectives[objective]
    v_flake, v_sub = channel_signals(nc, thicknesses_nm, t_oxide, OBJECTIVE_NA[objective], op.chain_spd)
    mix = None if wb is None else (PIPE.ccm if ccm is None else ccm) @ np.diag(wb)
    glare_f = np.array(op.glare_f_raw) if glare else None
    return pipeline_contrast(v_flake, v_sub, glare_f, mix)


def predict_legacy_halogen(thicknesses_nm: np.ndarray, nc: complex, na: float, t_oxide: float) -> np.ndarray:
    """The historical widget space: halogen 3200K, raw channels, no T^2/CCM/glare."""
    v_flake, v_sub = channel_signals(nc, thicknesses_nm, t_oxide, na, HALOGEN)
    return pipeline_contrast(v_flake, v_sub)


def fit_nk(
    thicknesses_nm: np.ndarray,
    r_meas: np.ndarray,
    g_meas: np.ndarray,
    r_ok: np.ndarray,
    model_fn,
) -> tuple[complex, float]:
    """Fit constant n - jk (offsets pinned at zero); returns (n, rms).

    model_fn(ts, nc) -> (3, N) contrast.  R residuals masked by r_ok
    (clipped rows contribute G only); rms over the used residuals.
    """

    def resid(p: np.ndarray) -> np.ndarray:
        c = model_fn(thicknesses_nm, p[0] - 1j * p[1])
        return np.concatenate([(c[0] - r_meas)[r_ok], c[1] - g_meas])

    sol = least_squares(resid, x0=[2.6, 1.3], bounds=([1.5, 0.2], [4.0, 2.5]))
    rms = float(np.sqrt(np.mean(sol.fun**2)))
    return complex(sol.x[0], -sol.x[1]), rms


def rms_rg(c: np.ndarray, r_meas: np.ndarray, g_meas: np.ndarray, r_ok: np.ndarray) -> float:
    res = np.concatenate([(c[0] - r_meas)[r_ok], c[1] - g_meas])
    return float(np.sqrt(np.mean(res**2)))


def _fmt_rgb(c: np.ndarray, j: int) -> str:
    return f"({c[0, j]:+.3f}, {c[1, j]:+.3f}, {c[2, j]:+.3f})"


def _fmt_rg(c: np.ndarray, j: int) -> str:
    return f"({c[0, j]:+.3f}, {c[1, j]:+.3f})"


def section_validation() -> None:
    """Reproduce the deployed thick cal table from its recorded widget params."""
    print("=" * 78)
    print("S0  Validation: reproduce GRAPHENE_THICK_90NM_CAL_POINTS from recorded params")
    print("    (halogen 3200K raw channels, n=2.817-1.317j, oxide 90.06, NA 0.25,")
    print("     camera space = model - offset, r_off +0.08 g_off +0.01)")
    ts = np.array([p[2] for p in GRAPHENE_THICK_90NM_CAL_POINTS])
    c = predict_legacy_halogen(ts, N_PRESET_THICK, na=0.25, t_oxide=90.06)
    print(f"    {'t_nm':>5} {'deployed (R, G)':>20} {'reproduced':>20} {'delta':>16}")
    for j, (r_dep, g_dep, t) in enumerate(GRAPHENE_THICK_90NM_CAL_POINTS):
        r_m, g_m = c[0, j] - 0.08, c[1, j] - 0.01
        print(
            f"    {t:5.1f} ({r_dep:+.3f}, {g_dep:+.3f})    ({r_m:+.3f}, {g_m:+.3f})"
            f"    ({r_m - r_dep:+.4f}, {g_m - g_dep:+.4f})"
        )


def section_staged_shifts() -> None:
    print("=" * 78)
    print("S1  Staged pipeline shifts, thin graphene 1-4L (10x, 90 nm oxide, n=2.6-1.3j)")
    ts = np.arange(1, 5) * THIN_LAYER_NM
    raw = predict(ts, N_LIT)
    mix_thin = predict(ts, N_LIT, wb=WB_THIN)
    mix_glare = predict(ts, N_LIT, wb=WB_THIN, glare=True)
    mix_thick = predict(ts, N_LIT, wb=WB_THICK)
    mix_unity = predict(ts, N_LIT, wb=np.ones(3))
    print(f"    {'L':>2} {'raw (R,G,B)':>26} {'+CCM.hBNwb (R,G)':>18} {'dRG':>16} {'+glare':>18} {'dRG':>16}")
    for j in range(len(ts)):
        d1 = f"({mix_thin[0, j] - raw[0, j]:+.3f}, {mix_thin[1, j] - raw[1, j]:+.3f})"
        d2 = f"({mix_glare[0, j] - mix_thin[0, j]:+.3f}, {mix_glare[1, j] - mix_thin[1, j]:+.3f})"
        print(
            f"    {j + 1:2d} {_fmt_rgb(raw, j):>26} {_fmt_rg(mix_thin, j):>18} {d1:>16}"
            f" {_fmt_rg(mix_glare, j):>18} {d2:>16}"
        )
    print("    WB dependence through the CCM (same physics, different WB request):")
    print(f"    {'L':>2} {'CCM.unity':>18} {'CCM.hBN(1.41/1.02/2.51)':>24} {'CCM.thick(1.4/1.0/1.7)':>24}")
    for j in range(len(ts)):
        print(f"    {j + 1:2d} {_fmt_rg(mix_unity, j):>18} {_fmt_rg(mix_thin, j):>24} {_fmt_rg(mix_thick, j):>24}")

    print()
    print("S1b Staged shifts, gate-thick band (10x, 90 nm oxide)")
    ts = np.array([5.0, 6.0, 8.0, 10.0])
    for nc, tag in ((N_PRESET_THICK, "n=2.817-1.317j (preset)"), (N_LIT, "n=2.6-1.3j (lit)")):
        raw = predict(ts, nc)
        mix_t = predict(ts, nc, wb=WB_THICK)
        mix_tg = predict(ts, nc, wb=WB_THICK, glare=True)
        mix_h = predict(ts, nc, wb=WB_THIN)
        print(f"    -- {tag}")
        print(f"    {'t_nm':>5} {'raw (R,G,B)':>26} {'+CCM.thickwb':>18} {'dRG':>16} {'+glare':>18} {'CCM.hBNwb':>18}")
        for j in range(len(ts)):
            d1 = f"({mix_t[0, j] - raw[0, j]:+.3f}, {mix_t[1, j] - raw[1, j]:+.3f})"
            print(
                f"    {ts[j]:5.1f} {_fmt_rgb(raw, j):>26} {_fmt_rg(mix_t, j):>18} {d1:>16}"
                f" {_fmt_rg(mix_tg, j):>18} {_fmt_rg(mix_h, j):>18}"
            )


def section_rclip() -> None:
    """Where does the CCM drive output R below the sensor floor (contrast -1)?

    Raw-channel contrast can never go below -1 (signals are nonnegative);
    output-space contrast can, because the CCM's negative off-diagonals can
    drive the R pixel value negative — the camera then floors at 0 counts and
    the measured contrast clips at exactly -1.0 (as seen on AFM cuts 3/4).
    """
    print()
    print("S1c Predicted R-clip band (output R <= -1.0 -> camera floors, measured R = -1.0)")
    ts = np.linspace(3.0, 14.0, 111)
    for nc, tag in ((N_PRESET_THICK, "n_preset"), (N_LIT, "n_lit")):
        for wb, wtag in ((WB_THICK, "thick WB"), (WB_THIN, "hBN WB")):
            c = predict(ts, nc, wb=wb, glare=True)
            clipped = ts[c[0] <= -1.0]
            band = f"{clipped.min():.1f}-{clipped.max():.1f} nm" if len(clipped) else "none"
            print(f"    {tag:<9} {wtag:<9} R<=-1 for t in {band}")


def section_anchor_fits() -> None:
    print("=" * 78)
    print("S2  Anchors vs model ladder (rms over R+G residuals; offsets = 0 everywhere)")

    thin_t = _CAL_DATA_GRAPHENE_10X[:, 0]
    thin_r = _CAL_DATA_GRAPHENE_10X[:, 1]
    thin_g = _CAL_DATA_GRAPHENE_10X[:, 2]
    thin_ok = np.ones(len(thin_t), dtype=bool)

    thick_t = _CAL_DATA_GRAPHITE_10X[:, 0]
    thick_r = _CAL_DATA_GRAPHITE_10X[:, 1]
    thick_g = _CAL_DATA_GRAPHITE_10X[:, 2]
    thick_ok = _CAL_DATA_GRAPHITE_10X_R_RELIABLE

    ladders: list[tuple[str, dict]] = [
        ("legacy halogen raw", {"legacy": True}),
        ("pipe raw (lamp x T^2)", {}),
        ("pipe + CCM.hBNwb", {"wb": WB_THIN}),
        ("pipe + CCM.hBNwb + glare", {"wb": WB_THIN, "glare": True}),
        ("pipe + CCM.thickwb", {"wb": WB_THICK}),
        ("pipe + CCM.thickwb + glare", {"wb": WB_THICK, "glare": True}),
    ]

    def model_fn(kw):
        if kw.get("legacy"):
            return lambda ts, nc: predict_legacy_halogen(ts, nc, na=0.25, t_oxide=T_OXIDE)
        return lambda ts, nc: predict(ts, nc, wb=kw.get("wb"), glare=kw.get("glare", False))

    for label, t, r, g, ok in (
        ("thin 1-4L 10x anchors (2026-04, scanned at hBN WB)", thin_t, thin_r, thin_g, thin_ok),
        ("graphite gate cuts 10x (AFM; R clipped rows G-only)", thick_t, thick_r, thick_g, thick_ok),
    ):
        print(f"    -- {label}")
        print(f"    {'model':<28} {'rms(n_lit)':>10} {'fit n':>16} {'rms(fit)':>9}")
        for name, kw in ladders:
            fn = model_fn(kw)
            rms_lit = rms_rg(fn(t, N_LIT), r, g, ok)
            n_fit, rms_fit = fit_nk(t, r, g, ok, fn)
            print(f"    {name:<28} {rms_lit:>10.3f} {n_fit.real:>7.3f}{n_fit.imag:+.3f}j {rms_fit:>9.3f}")

    # 50x previous-scope anchors: unknown WB, so just bracket.
    t50 = _CAL_DATA_GRAPHENE[:, 0]
    r50 = _CAL_DATA_GRAPHENE[:, 1]
    g50 = _CAL_DATA_GRAPHENE[:, 2]
    ok50 = np.ones(len(t50), dtype=bool)
    print("    -- thin 1-5L 50x anchors (scan 154, old system; WB unknown)")
    for name, fn in (
        ("pipe 50x raw", lambda ts, nc: predict(ts, nc, objective="50x")),
        ("pipe 50x + CCM.hBNwb + glare", lambda ts, nc: predict(ts, nc, objective="50x", wb=WB_THIN, glare=True)),
    ):
        rms_lit = rms_rg(fn(t50, N_LIT), r50, g50, ok50)
        n_fit, rms_fit = fit_nk(t50, r50, g50, ok50, fn)
        print(f"    {name:<28} {rms_lit:>10.3f} {n_fit.real:>7.3f}{n_fit.imag:+.3f}j {rms_fit:>9.3f}")


def section_preset_impact() -> None:
    print("=" * 78)
    print("S3  Deployed preset gates vs pipeline-model predictions")

    # Thin preset: nearest-reference-point classification of model + anchors.
    print("    -- graphene_thin_90nm (RG point cal; tier1 cal_dist 0.06, match 0.08, possible 0.15)")
    ts = np.arange(1, 5) * THIN_LAYER_NM
    model = predict(ts, N_LIT, wb=WB_THIN, glare=True)
    print(f"    {'src':<10} {'L':>2} {'(R, G)':>18} {'cal_dist':>9} {'-> class':>9}")
    for j in range(len(ts)):
        proj = _thin_cfg.cal_projection(model[0, j], model[1, j], 0.0)
        print(f"    {'model':<10} {j + 1:>2} {_fmt_rg(model, j):>18} {proj.dist:>9.3f} {proj.layers!s:>9}")
    for row in _CAL_DATA_GRAPHENE_10X:
        n_l = round(row[0] / THIN_LAYER_NM)
        proj = _thin_cfg.cal_projection(row[1], row[2], 0.0)
        print(f"    {'anchor':<10} {n_l:>2} ({row[1]:+.3f}, {row[2]:+.3f}) {proj.dist:>9.3f} {proj.layers!s:>9}")
    ref = np.array([(p.r, p.g) for p in _thin_cfg.cal_reference_points])
    spacing = float(np.mean(np.hypot(*np.diff(ref, axis=0).T)))
    print(f"    inter-layer reference spacing ~{spacing:.3f} in (R,G)")

    # Thick preset: curve distance + thickness estimation error.
    print("    -- graphene_thick_90nm (curve cal; tier1 cal_dist 0.06, match 0.15, possible 0.25)")
    ts = np.array([5.0, 6.0, 8.0, 10.0])
    print(f"    {'src':<22} {'t_nm':>5} {'(R, G)':>18} {'cal_dist':>9} {'t_est':>6} {'dt':>6}")
    for nc, tag in ((N_PRESET_THICK, "model n_preset"), (N_LIT, "model n_lit")):
        c = predict(ts, nc, wb=WB_THICK, glare=True)
        for j in range(len(ts)):
            proj = _thick_cfg.cal_curve(c[0, j], c[1, j])
            dt = (proj.thickness_nm or 0.0) - ts[j]
            print(
                f"    {tag:<22} {ts[j]:>5.1f} {_fmt_rg(c, j):>18} {proj.dist:>9.3f} {proj.thickness_nm!s:>6} {dt:+6.1f}"
            )
    for row, ok in zip(_CAL_DATA_GRAPHITE_10X, _CAL_DATA_GRAPHITE_10X_R_RELIABLE, strict=True):
        proj = _thick_cfg.cal_curve(row[1], row[2])
        dt = (proj.thickness_nm or 0.0) - row[0]
        flag = "" if ok else " [R clipped]"
        print(
            f"    {'anchor (AFM)':<22} {row[0]:>5.1f} ({row[1]:+.3f}, {row[2]:+.3f})"
            f" {proj.dist:>9.3f} {proj.thickness_nm!s:>6} {dt:+6.1f}{flag}"
        )


def section_cross_objective() -> None:
    print("=" * 78)
    print("S4  Cross-objective transfer (full pipe: chain + NA + glare, CCM at each preset's WB)")
    thin_ts = np.array([1, 2, 4]) * THIN_LAYER_NM
    thick_ts = np.array([6.0, 10.0])
    for label, ts, wb in (("thin (hBN WB)", thin_ts, WB_THIN), ("thick (1.4/1.0/1.7 WB)", thick_ts, WB_THICK)):
        print(f"    -- {label}")
        header = "    " + f"{'t_nm':>6}" + "".join(f" {o:>20}" for o in ("10x", "20x", "50x"))
        print(header)
        preds = {o: predict(ts, N_LIT, objective=o, wb=wb, glare=True) for o in ("10x", "20x", "50x")}
        for j in range(len(ts)):
            row = f"    {ts[j]:>6.2f}" + "".join(f" {_fmt_rg(preds[o], j):>20}" for o in ("10x", "20x", "50x"))
            print(row)


def section_bcol_sensitivity() -> None:
    print("=" * 78)
    print("S5  CCM B-column sensitivity (column scaled x0.8 / x1.2; Phase-2 will lock this)")
    for label, ts, wb in (
        ("thin 1L", np.array([THIN_LAYER_NM]), WB_THIN),
        ("thin 4L", np.array([4 * THIN_LAYER_NM]), WB_THIN),
        ("thick 6 nm", np.array([6.0]), WB_THICK),
        ("thick 10 nm", np.array([10.0]), WB_THICK),
    ):
        base = predict(ts, N_LIT, wb=wb, glare=True)
        row = f"    {label:<12} base {_fmt_rg(base, 0)}"
        for s in (0.8, 1.2):
            ccm = PIPE.ccm.copy()
            ccm[:, 2] *= s
            c = predict(ts, N_LIT, wb=wb, glare=True, ccm=ccm)
            row += f"  x{s:.1f} ({c[0, 0] - base[0, 0]:+.3f}, {c[1, 0] - base[1, 0]:+.3f})"
        print(row + "   (deltas)")


def make_plot(output: str | None) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_thin, ax_thick) = plt.subplots(1, 2, figsize=(14, 7))

    ts_thin = np.arange(1, 6) * THIN_LAYER_NM
    stages: list[tuple[str, np.ndarray | None, bool, str]] = [
        ("raw channels", None, False, "tab:gray"),
        ("+CCM (hBN WB)", WB_THIN, False, "tab:orange"),
        ("+CCM+glare", WB_THIN, True, "tab:red"),
    ]
    for label, wb, glare, color in stages:
        c = predict(ts_thin, N_LIT, wb=wb, glare=glare)
        ax_thin.plot(c[1], c[0], "o-", color=color, label=label, ms=4)
    ax_thin.plot(_CAL_DATA_GRAPHENE_10X[:, 2], _CAL_DATA_GRAPHENE_10X[:, 1], "k^", label="10x anchors 1-4L", ms=8)
    ref = np.array([(p.g, p.r) for p in _thin_cfg.cal_reference_points])
    ax_thin.plot(ref[:, 0], ref[:, 1], "bs", label="deployed thin cal (scan 154)", ms=7, mfc="none")
    ax_thin.set_title(f"Thin graphene 1-5L (10x, n={N_LIT})")

    ts_band = np.linspace(3.0, 12.0, 28)
    stages_thick: list[tuple[str, np.ndarray | None, bool, str]] = [
        ("raw channels", None, False, "tab:gray"),
        ("+CCM (thick WB)", WB_THICK, False, "tab:orange"),
        ("+CCM+glare", WB_THICK, True, "tab:red"),
        ("+CCM (hBN WB)+glare", WB_THIN, True, "tab:purple"),
    ]
    for label, wb, glare, color in stages_thick:
        c = predict(ts_band, N_PRESET_THICK, wb=wb, glare=glare)
        ax_thick.plot(c[1], c[0], "-", color=color, label=label)
    dep = np.array(GRAPHENE_THICK_90NM_CAL_POINTS)
    ax_thick.plot(dep[:, 1], dep[:, 0], "bs", label="deployed thick cal 5-10nm", ms=7, mfc="none")
    rel = _CAL_DATA_GRAPHITE_10X_R_RELIABLE
    ax_thick.plot(_CAL_DATA_GRAPHITE_10X[rel, 2], _CAL_DATA_GRAPHITE_10X[rel, 1], "k^", label="AFM cuts (R ok)", ms=8)
    ax_thick.plot(
        _CAL_DATA_GRAPHITE_10X[~rel, 2],
        _CAL_DATA_GRAPHITE_10X[~rel, 1],
        "^",
        color="k",
        mfc="none",
        label="AFM cuts (R clipped)",
        ms=8,
    )
    ax_thick.axhline(-1.0, color="k", lw=1.0, ls=":", alpha=0.8)
    ax_thick.text(
        0.02, 0.02, "R = -1 sensor floor (output R below this clips)", transform=ax_thick.transAxes, fontsize=8
    )
    ax_thick.set_title(f"Gate-thick band 3-12 nm (10x, n={N_PRESET_THICK})")

    for ax in (ax_thin, ax_thick):
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel("Green contrast")
        ax.set_ylabel("Red contrast")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Graphene presets through the measured K5C pipeline (2026-08-06 model)")
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=120)
        print(f"saved {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphene presets under the measured K5C pipeline model")
    parser.add_argument("--plot", action="store_true", help="show the R-G plane figure")
    parser.add_argument("--output", help="save the figure to a file instead of showing")
    args = parser.parse_args()

    print(f"WB thin preset {WB_THIN}, thick preset {WB_THICK}, hBN scan {WB_REQUESTS_HBN_SCAN}")
    section_validation()
    section_staged_shifts()
    section_rclip()
    section_anchor_fits()
    section_preset_impact()
    section_cross_objective()
    section_bcol_sensitivity()
    if args.plot or args.output:
        make_plot(args.output)


if __name__ == "__main__":
    main()
