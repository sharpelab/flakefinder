"""Fit thick-hBN calibration models against the combined AFM anchor set.

Anchors: round 1 (scan 287 / run_20260804_1354, 45.5-57.5 nm, 98982
excluded) + round 2 (scan 288 / run_20260804_1303, 55-85 nm, 99280
excluded).  Two model shapes, n as the only free parameter:

  legacy:   halogen 3200K illuminant, no glare, NA 0.25, offsets pinned to
            the v3 values (+0.54 R / -0.20 G) — the configuration that
            produced the v3 cal table (n=2.269 on round-1 anchors only).
  pipeline: measured K5C chain (Elijah lamp x T^2_10x, CCM @ hBN-scan WB),
            10x raw-space glare floors, NA 0.25, zero hand offsets
            (scripts/colorchecker_cal.py).

For each model, reports per-anchor camera-space R/G residuals and the
nearest-point thickness re-prediction (the segmentation.py projection),
broken out by round, plus alias readings for the past-fold points and the
excluded 99280.  --table emits the winning model's cal table in
segmentation.py HBN_THICK_50_100_90NM_CAL_POINTS format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent))
from colorchecker_cal import OBJECTIVE_NA, WB_REQUESTS_HBN_SCAN, derive_pipeline
from hbn_contrast import (
    _CAL_DATA_HBN_10X_THICK,
    _CAL_DATA_HBN_10X_THICK_FIT_OK,
    _CAL_DATA_HBN_10X_THICK_R2,
    _CAL_DATA_HBN_10X_THICK_R2_EXCLUDED,
    _CAL_DATA_HBN_10X_THICK_R2_FIT_OK,
    _CAL_DATA_HBN_10X_THICK_R2_MARGINAL,
    _CAL_DATA_HBN_10X_THICK_R2_PASTFOLD,
    _HBN_LAYER_THICKNESS_NM,
    _IMX183_BLUE,
    _IMX183_GREEN,
    _IMX183_RED,
    _IMX183_WAVELENGTHS,
    _blackbody,
)
from hbn_contrast_widget import _GLARE_F, _ILLUMINANTS, _TRANSMISSIONS, compute_rg

# v3 legacy-model hand offsets (camera space = model - offset); see
# segmentation.py HBN_THICK_50_100_90NM_CAL_POINTS provenance comment.
_LEGACY_R_OFF = 0.54
_LEGACY_G_OFF = -0.20

_T_OXIDE_NM = 90.0
_MAX_LAYERS = 360  # ~120 nm — past the table's 110 nm endpoint

_PIPELINE = derive_pipeline(_IMX183_WAVELENGTHS)


class ModelCurve(NamedTuple):
    """Camera-space contrast curve sampled per layer."""

    r: np.ndarray
    g: np.ndarray
    t_nm: np.ndarray


def model_curve(model: str, n: float) -> ModelCurve:
    """Camera-space (R, G, t) curve for one model shape at constant n."""
    if model == "legacy":
        illum = _blackbody(_IMX183_WAVELENGTHS, 3200.0)
        r, g, t = compute_rg(
            complex(n),
            _T_OXIDE_NM,
            OBJECTIVE_NA["10x"],
            max_layers=_MAX_LAYERS,
            red_lit=_IMX183_RED * illum,
            green_lit=_IMX183_GREEN * illum,
        )
        return ModelCurve(r - _LEGACY_R_OFF, g - _LEGACY_G_OFF, t)
    if model == "pipeline":
        op = _PIPELINE.objectives["10x"]
        r, g, t = compute_rg(
            complex(n),
            _T_OXIDE_NM,
            OBJECTIVE_NA["10x"],
            max_layers=_MAX_LAYERS,
            red_lit=_IMX183_RED * op.chain_spd,
            green_lit=_IMX183_GREEN * op.chain_spd,
            blue_lit=_IMX183_BLUE * op.chain_spd,
            mix=_PIPELINE.ccm @ np.diag(WB_REQUESTS_HBN_SCAN),
            glare_f=np.array(op.glare_f_raw),
        )
        return ModelCurve(r, g, t)  # zero offsets: output space is camera space
    raise ValueError(f"unknown model {model!r}")


def export_curve(
    path: Path, n: float | None = None, r_off: float | None = None, g_off: float | None = None
) -> ModelCurve:
    """Camera-space curve from a widget 'Export all' params file.

    n / r_off / g_off default to the exported values; passing them evaluates
    the same illuminant/transmission/glare/NA config at different values
    (used by --fit-export)."""
    p = json.loads(path.read_text())
    illum = _ILLUMINANTS[p["illuminant"]]
    trans = _TRANSMISSIONS[p["transmission"]].factor
    glare_f = None
    if p.get("glare", "off") != "off":
        glare_f = _GLARE_F[p["glare"]]["raw" if illum.mix is not None else "out"]
    r, g, t = compute_rg(
        complex(p["n_hbn"] if n is None else n),
        p["t_oxide_90"],
        p["na"],
        max_layers=_MAX_LAYERS,
        red_lit=illum.red_lit * trans,
        green_lit=illum.green_lit * trans,
        blue_lit=illum.blue_lit * trans if illum.blue_lit is not None else None,
        mix=illum.mix,
        glare_f=glare_f,
    )
    r_off = p["r_offset"] if r_off is None else r_off
    g_off = p["g_offset"] if g_off is None else g_off
    return ModelCurve(r - r_off, g - g_off, t)


def fit_export_config(path: Path, anchors: np.ndarray) -> tuple[float, float, float]:
    """Least-squares (n, r_off, g_off) under the export's optical config."""
    from scipy.optimize import minimize

    p = json.loads(path.read_text())

    def loss(x: np.ndarray) -> float:
        return _rss(export_curve(path, n=x[0], r_off=x[1], g_off=x[2]), anchors)

    res = minimize(loss, [p["n_hbn"], p["r_offset"], p["g_offset"]], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def combined_anchors(
    stability_filter: bool = False, drop_marginal: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(anchors, round_tag, dropped_r2) — round-1 + round-2 fit anchors.

    Default keeps all round-2 anchors (Zack's 2026-08-13 ruling: the
    erosion-stability filter discards the entire new-regime band, ignore
    it).  stability_filter=True applies the fold_degeneracy interior-
    stability mask (and drop_marginal additionally drops 99242) for
    sensitivity checks; dropped_r2 rows are reported as non-anchor
    readouts."""
    r1 = _CAL_DATA_HBN_10X_THICK[_CAL_DATA_HBN_10X_THICK_FIT_OK]
    ok = np.ones(len(_CAL_DATA_HBN_10X_THICK_R2), dtype=bool)
    if stability_filter:
        ok &= _CAL_DATA_HBN_10X_THICK_R2_FIT_OK
        if drop_marginal:
            ok &= ~_CAL_DATA_HBN_10X_THICK_R2_MARGINAL
    r2 = _CAL_DATA_HBN_10X_THICK_R2[ok]
    dropped = _CAL_DATA_HBN_10X_THICK_R2[~ok]
    rounds = np.concatenate([np.ones(len(r1), dtype=int), np.full(len(r2), 2)])
    return np.vstack([r1, r2]), rounds, dropped


def _rss(curve: ModelCurve, anchors: np.ndarray) -> float:
    r_res = anchors[:, 1] - np.interp(anchors[:, 0], curve.t_nm, curve.r)
    g_res = anchors[:, 2] - np.interp(anchors[:, 0], curve.t_nm, curve.g)
    return float(np.sum(r_res**2) + np.sum(g_res**2))


def fit_n(model: str, anchors: np.ndarray) -> float:
    res = minimize_scalar(
        lambda n: _rss(model_curve(model, n), anchors),
        bounds=(1.7, 2.7),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(res.x)


def project_thickness(curve: ModelCurve, r: float, g: float) -> tuple[float, float]:
    """Nearest-point projection in (R, G) — the segmentation.py readout.

    Returns (thickness_nm, rg_distance); a large distance means the point
    is not actually near the curve and the reading is not meaningful."""
    d2 = (curve.r - r) ** 2 + (curve.g - g) ** 2
    idx = int(np.argmin(d2))
    return float(curve.t_nm[idx]), float(np.sqrt(d2[idx]))


def report(title: str, curve: ModelCurve, anchors: np.ndarray, rounds: np.ndarray, dropped: np.ndarray) -> None:
    print(f"\n=== {title} ===")
    header = f"{'rnd':>3} {'t_afm':>6} {'R_emp':>7} {'R_mod':>7} {'dR':>7} {'G_emp':>7} {'G_mod':>7} {'dG':>7}"
    print(header + f" {'t_pred':>7} {'dt':>6}")
    r_res = anchors[:, 1] - np.interp(anchors[:, 0], curve.t_nm, curve.r)
    g_res = anchors[:, 2] - np.interp(anchors[:, 0], curve.t_nm, curve.g)
    dts = []
    for i, (t_afm, r_emp, g_emp) in enumerate(anchors):
        r_mod = float(np.interp(t_afm, curve.t_nm, curve.r))
        g_mod = float(np.interp(t_afm, curve.t_nm, curve.g))
        t_pred, _ = project_thickness(curve, r_emp, g_emp)
        dts.append(t_pred - t_afm)
        print(
            f"{rounds[i]:>3} {t_afm:6.1f} {r_emp:7.3f} {r_mod:7.3f} {r_emp - r_mod:+7.3f}"
            f" {g_emp:7.3f} {g_mod:7.3f} {g_emp - g_mod:+7.3f} {t_pred:7.1f} {dts[-1]:+6.1f}"
        )
    dts = np.array(dts)
    for label, m in (("round 1", rounds == 1), ("round 2", rounds == 2), ("combined", np.ones_like(rounds, bool))):
        print(
            f"  {label:>8}: R rms={np.sqrt(np.mean(r_res[m] ** 2)):.3f}"
            f"  G rms={np.sqrt(np.mean(g_res[m] ** 2)):.3f}"
            f"  dt={np.mean(dts[m]):+.2f}±{np.std(dts[m]):.2f} nm"
        )

    print("  non-anchor readouts (nearest-point projection; dist = RG distance to curve):")
    for t_afm, r_emp, g_emp in dropped:
        t_pred, dist = project_thickness(curve, r_emp, g_emp)
        print(f"    dropped unstable AFM {t_afm:.1f} nm: reads {t_pred:.1f} nm (dist {dist:.2f})")
    for t_afm, r_emp, g_emp in _CAL_DATA_HBN_10X_THICK_R2_EXCLUDED:
        t_pred, dist = project_thickness(curve, r_emp, g_emp)
        print(f"    99280 excluded (AFM {t_afm:.0f}? region mismatch): reads {t_pred:.1f} nm (dist {dist:.2f})")
    for t_afm, r_emp, g_emp in _CAL_DATA_HBN_10X_THICK_R2_PASTFOLD:
        t_pred, dist = project_thickness(curve, r_emp, g_emp)
        print(f"    past-fold AFM {t_afm:.0f} nm: nearest in-band point {t_pred:.1f} nm (dist {dist:.2f})")


def emit_table(title: str, curve: ModelCurve) -> None:
    print(f"\n# {title}, camera space, 40-110 nm @ 2 nm")
    for t_nm in np.arange(40.0, 112.0, 2.0):
        r = float(np.interp(t_nm, curve.t_nm, curve.r))
        g = float(np.interp(t_nm, curve.t_nm, curve.g))
        layers = round(t_nm / _HBN_LAYER_THICKNESS_NM)
        print(f"    CalPointRG(layers={layers}, r={r:.4f}, g={g:.4f}),  # {t_nm:.0f} nm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["legacy", "pipeline", "both"], default="both")
    parser.add_argument("--n", type=float, default=None, help="evaluate at fixed n instead of fitting")
    parser.add_argument("--table", action="store_true", help="emit segmentation.py cal table(s)")
    parser.add_argument(
        "--max-t",
        type=float,
        default=None,
        help="fit only anchors with AFM thickness <= this (nm); sensitivity checks, e.g. 75 drops the fold region",
    )
    parser.add_argument(
        "--stability-filter",
        action="store_true",
        help="apply the fold_degeneracy interior-stability mask (drops 99234/99285/99236)",
    )
    parser.add_argument(
        "--drop-marginal",
        action="store_true",
        help="with --stability-filter, also drop the marginal anchor (99242)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="also evaluate a widget 'Export all' params JSON against the anchors (no fitting)",
    )
    parser.add_argument(
        "--fit-export",
        action="store_true",
        help="with --export: least-squares refit (n, r_off, g_off) under the export's optical config",
    )
    args = parser.parse_args()

    anchors, rounds, dropped = combined_anchors(
        stability_filter=args.stability_filter, drop_marginal=args.drop_marginal
    )
    if args.max_t is not None:
        keep = anchors[:, 0] <= args.max_t
        anchors, rounds = anchors[keep], rounds[keep]
    models = ["legacy", "pipeline"] if args.model == "both" else [args.model]
    t_lo, t_hi = anchors[:, 0].min(), anchors[:, 0].max()
    n1, n2 = np.sum(rounds == 1), np.sum(rounds == 2)
    print(f"{len(anchors)} anchors: {n1} round-1 + {n2} round-2, {t_lo:.1f}-{t_hi:.1f} nm")
    for model in models:
        n = args.n if args.n is not None else fit_n(model, anchors)
        report(f"{model}: n = {n:.4f}", model_curve(model, n), anchors, rounds, dropped)
        if args.table:
            emit_table(f"{model} model, n={n:.4f}", model_curve(model, n))
    if args.export is not None:
        report(f"export: {args.export}", export_curve(args.export), anchors, rounds, dropped)
        if args.table:
            emit_table(f"export {args.export}", export_curve(args.export))
        if args.fit_export:
            n, r_off, g_off = fit_export_config(args.export, anchors)
            report(
                f"export config refit: n={n:.4f} r_off={r_off:+.3f} g_off={g_off:+.3f}",
                export_curve(args.export, n=n, r_off=r_off, g_off=g_off),
                anchors,
                rounds,
                dropped,
            )


if __name__ == "__main__":
    main()
