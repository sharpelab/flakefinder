"""Effective-film model of substrate background colour and its effect on graphene contrast.

A thin dielectric on the SiO₂ (tape residue, adsorbates) shifts the interference
colour of the substrate the same way a thicker oxide does. This module predicts,
per objective and in scan space (WB + measured 5800K CCM), the background R/G and
B/G for an effective oxide thickness, and the per-channel contrast of N-layer
graphene on it. Illumination chain: lamp × T²_objective × ε, with ε the
common-path tilt anchored on the golden blank refs (see docs/blank_refs_20260813.md).

As a CLI it sweeps effective oxide thickness, fits one Δd to each measured
background shift, and prints the predicted graphene contrast shift per layer.

Usage:
    uv run python scripts/film_model.py                       # sweep + built-in shift fits
    uv run python scripts/film_model.py --shift 0.960,2.159 0.944,2.193 --wb thin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from colorchecker_cal import (  # noqa: E402
    _CAL_DIR,
    OBJECTIVE_NA,
    _channel_integrals,
    derive_blank_ref_tilt,
    derive_pipeline,
)
from hbn_contrast import _IMX183_WAVELENGTHS as LAMB  # noqa: E402
from hbn_contrast import n_graphene_constant, reflectance  # noqa: E402

_MODEL = derive_pipeline(LAMB)
_EPS = derive_blank_ref_tilt(_MODEL, LAMB)
_CCM_MODELS = json.loads((_CAL_DIR / "colorchecker_phase2_20260806" / "ccm_models.json").read_text())
_CCM = np.array(_CCM_MODELS["idx2_real_units"])
_PEDESTAL = float(_CCM_MODELS["pedestal_counts"])
_RAW_GOLDEN = json.loads((_CAL_DIR / "blank_refs_raw_20260813.json").read_text())
WB = {"thin": np.array([1.41, 1.02, 2.51]), "thick": np.array([1.40, 1.00, 1.70])}
GRAPHENE_LAYER_NM = 0.335
OXIDE_NM = 90.0


def chain(objective: str) -> np.ndarray:
    """Illumination × objective chain with the common-path correction applied."""
    return _MODEL.objectives[objective].chain_spd * _EPS


def to_scan(raw_rgb: np.ndarray, wb: str) -> np.ndarray:
    """Raw sensor R,G,B → scan space through WB gains and the 5800K CCM."""
    return (raw_rgb * WB[wb] + _PEDESTAL) @ _CCM.T - _PEDESTAL


def substrate_raw(objective: str, oxide_nm: float) -> np.ndarray:
    """Raw R,G,B channel integrals for bare SiO₂/Si at the objective's NA."""
    r = reflectance(LAMB, None, 0, oxide_nm, na=OBJECTIVE_NA[objective])
    return _channel_integrals(chain(objective) * r, LAMB)


def bg_ratios(objective: str, oxide_nm: float, wb: str) -> tuple[float, float]:
    """Scan-space background (R/G, B/G), scaled from the golden raw refs at 90 nm."""
    raw0 = np.array(_RAW_GOLDEN[objective]["rgb"])
    s = to_scan(raw0 * substrate_raw(objective, oxide_nm) / substrate_raw(objective, OXIDE_NM), wb)
    return float(s[0] / s[1]), float(s[2] / s[1])


def graphene_contrast(objective: str, n_layers: int, oxide_nm: float, wb: str) -> np.ndarray:
    """Scan-space (R, G, B) contrast of N-layer graphene on the given effective oxide."""
    na = OBJECTIVE_NA[objective]
    raw0 = np.array(_RAW_GOLDEN[objective]["rgb"])
    r_flake = reflectance(LAMB, n_graphene_constant(), n_layers, oxide_nm, na=na, layer_thickness_nm=GRAPHENE_LAYER_NM)
    r_sub = reflectance(LAMB, None, 0, oxide_nm, na=na)
    v_flake = _channel_integrals(chain(objective) * r_flake, LAMB)
    v_sub = _channel_integrals(chain(objective) * r_sub, LAMB)
    s0 = to_scan(raw0, wb)
    s1 = to_scan(raw0 * v_flake / v_sub, wb)
    return (s1 - s0) / s0


def fit_oxide_delta(
    dln_rg: float, dln_bg: float, wb: str, objective: str = "10x", span: tuple[float, float] = (-6.0, 8.0)
) -> float:
    """Effective oxide change (nm) whose modelled Δln(R/G), Δln(B/G) best match the measured pair."""
    rg0, bg0 = bg_ratios(objective, OXIDE_NM, wb)
    best = (np.inf, 0.0)
    for dd in np.arange(span[0], span[1] + 1e-9, 0.125):
        rg, bg = bg_ratios(objective, OXIDE_NM + dd, wb)
        err = (np.log(rg / rg0) - dln_rg) ** 2 + (np.log(bg / bg0) - dln_bg) ** 2
        if err < best[0]:
            best = (err, float(dd))
    return best[1]


_BUILTIN_SHIFTS = {
    # (wb, R/G before, B/G before, R/G after, B/G after) — Gr_-_082726 control carrier, 2026-08-27
    "golden -> fresh chip_0 (thin)": ("thin", 1.0098, 2.0440, 0.960, 2.159),
    "fresh -> tape chip_1 (thin)": ("thin", 0.960, 2.159, 0.944, 2.193),
    "fresh -> tape chip_1 (thick)": ("thick", 1.042, 1.216, 1.028, 1.240),
    "fresh -> chips 4/6 (thin)": ("thin", 0.960, 2.159, 0.995, 2.055),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wb", choices=list(WB), default="thin", help="WB preset for --shift pairs (default thin)")
    ap.add_argument("--objective", default="10x", choices=list(OBJECTIVE_NA), help="Objective for the sweep and fits")
    ap.add_argument(
        "--shift",
        nargs=2,
        metavar="RG,BG",
        action="append",
        default=[],
        help="A measured background shift: 'R/G,B/G' before and after. Repeatable.",
    )
    args = ap.parse_args()
    obj, wb = args.objective, args.wb

    rg0, bg0 = bg_ratios(obj, OXIDE_NM, wb)
    print(f"Model at {obj}, NA {OBJECTIVE_NA[obj]:.2f}, {wb} WB: 90 nm -> R/G {rg0:.4f}  B/G {bg0:.4f}")
    print(f"\n{'d_ox':>6} {'R/G':>8} {'B/G':>8} {'dln R/G':>9} {'dln B/G':>9}")
    for d in (84, 86, 88, 89, 90, 91, 92, 93, 94, 96, 98, 100):
        rg, bg = bg_ratios(obj, float(d), wb)
        print(f"{d:6d} {rg:8.4f} {bg:8.4f} {np.log(rg / rg0):+9.4f} {np.log(bg / bg0):+9.4f}")

    shifts = {k: v for k, v in _BUILTIN_SHIFTS.items() if not args.shift}
    for i, (a, b) in enumerate(args.shift):
        ra, ba = (float(x) for x in a.split(","))
        rb, bb = (float(x) for x in b.split(","))
        shifts[f"shift {i + 1}"] = (wb, ra, ba, rb, bb)
    print("\nOne-parameter fit of effective oxide change to each shift (both ratios):")
    for name, (w, ra, ba, rb, bb) in shifts.items():
        drg, dbg = np.log(rb / ra), np.log(bb / ba)
        dd = fit_oxide_delta(drg, dbg, w, obj)
        r0 = bg_ratios(obj, OXIDE_NM, w)
        r1 = bg_ratios(obj, OXIDE_NM + dd, w)
        print(
            f"  {name:32} measured {drg:+.4f} {dbg:+.4f} | dd {dd:+6.2f} nm -> model "
            f"{np.log(r1[0] / r0[0]):+.4f} {np.log(r1[1] / r0[1]):+.4f}"
        )

    print(f"\nGraphene contrast shift d(R-G) for +dd nm effective oxide, {obj}, {wb} WB (film under the flake):")
    for dd in (1, 2, 3, 5):
        parts = []
        for n in range(1, 6):
            c0 = graphene_contrast(obj, n, OXIDE_NM, wb)
            c1 = graphene_contrast(obj, n, OXIDE_NM + dd, wb)
            parts.append(f"N{n} {(c1[0] - c1[1]) - (c0[0] - c0[1]):+.3f}")
        print(f"  +{dd} nm: " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
