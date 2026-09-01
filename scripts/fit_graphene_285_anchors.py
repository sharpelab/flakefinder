"""Fit raw-space graphene n against the 2026-08-31 identity-CCM 285 nm anchors.

Empirical anchor ladder measured on Jordan's 285 nm chip (Scan Notebook
2026-08-31; identity CCM, 10x, flatfielded float averages / local-ratio
eyedrops, condition-independence verified to ±0.004): (R, G) contrast per
layer, 1-5L. B is recorded for reference but excluded from the fit — the
chain model's B is biased in identity space (~+0.10 at 90 nm, 08-07 deccm
eval) and could not reproduce the measured 285 blank bg ratios.

Fits constant n = n_re - j·k (optionally with additive camera-space R/G
offsets, camera = model - offset) through the anchors using the measured
K5C pipeline in raw channel space (graphene_pipeline_effects.predict,
wb=None, glare on, 10x), then emits the segmentation.py cal table:
**empirical anchors verbatim for 1-5L**, fitted-model extrapolation for
6-10L. Oxide is fixed at the wafer spec (2850 Å ± 0.5%, Jordan 2026-08-31)
— the blank-shot oxide fit is blocked on the chain model's identity-space
channel ratios, and the fit here is over layer-contrast differences where
d(R,G)/dt_oxide is far weaker than in the absolute bg color.

Usage:
    uv run python scripts/fit_graphene_285_anchors.py            # fit report
    uv run python scripts/fit_graphene_285_anchors.py --table    # emit cal table
    uv run python scripts/fit_graphene_285_anchors.py --no-offsets --oxide 283
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from graphene_pipeline_effects import predict  # noqa: E402

LAYER_NM = 0.335
OXIDE_SPEC_NM = 285.0  # 2850 Å ± 0.5% wafer spec (Jordan, Slack 2026-08-31)

# (layers, R, G, B) — B for reference only, not fitted.
# 1L/2L: Zack's local-ratio eyedrops on the 10x identity average at stage
# (64220, 41438)/(64218, 41430); 2L cross-checked against the flatfielded
# float-window sample (exact match). 3-5L: flatfielded 3 px window samples
# on the anchor_f1 terraces (68928, 41983), layer-assigned by step ladder +
# model absolutes (terraces A/B/C = 3/4/5L; ladder closes with no gaps).
ANCHORS_285_IDENTITY_10X: tuple[tuple[int, float, float, float], ...] = (
    (1, -0.059, -0.092, -0.007),
    (2, -0.138, -0.199, -0.005),
    (3, -0.171, -0.253, -0.022),
    (4, -0.221, -0.322, -0.031),
    (5, -0.262, -0.377, -0.050),
)

EXTRAP_LAYERS = range(6, 11)


def model_rg(nc: complex, layers: np.ndarray, t_oxide: float) -> np.ndarray:
    """(2, N) raw-space (R, G) contrast at 10x for integer layer counts."""
    c = predict(layers * LAYER_NM, nc, "10x", wb=None, glare=True, t_oxide=t_oxide)
    return c[:2]


def fit(t_oxide: float, with_offsets: bool) -> tuple[complex, float, float, float]:
    """Least-squares (n, [r_off, g_off]) over the anchor R/G residuals.

    Returns (n, r_off, g_off, rms). Offsets are camera-space additive:
    camera = model - offset.
    """
    layers = np.array([a[0] for a in ANCHORS_285_IDENTITY_10X], dtype=float)
    meas = np.array([[a[1] for a in ANCHORS_285_IDENTITY_10X], [a[2] for a in ANCHORS_285_IDENTITY_10X]])

    def resid(p: np.ndarray) -> np.ndarray:
        nc = p[0] - 1j * p[1]
        off = p[2:4, None] if with_offsets else 0.0
        return (model_rg(nc, layers, t_oxide) - off - meas).ravel()

    x0 = [2.6, 1.3] + ([0.0, 0.0] if with_offsets else [])
    lo = [1.5, 0.2] + ([-0.3, -0.3] if with_offsets else [])
    hi = [4.5, 2.5] + ([0.3, 0.3] if with_offsets else [])
    sol = least_squares(resid, x0=np.array(x0), bounds=(np.array(lo), np.array(hi)))
    rms = float(np.sqrt(np.mean(sol.fun**2)))
    r_off, g_off = (float(sol.x[2]), float(sol.x[3])) if with_offsets else (0.0, 0.0)
    return complex(sol.x[0], -sol.x[1]), r_off, g_off, rms


def table_points(nc: complex, r_off: float, g_off: float, t_oxide: float) -> list[tuple[int, float, float]]:
    """Cal table rows: empirical 1-5L verbatim + model extrapolation 6-10L."""
    rows = [(int(a[0]), a[1], a[2]) for a in ANCHORS_285_IDENTITY_10X]
    layers = np.array(list(EXTRAP_LAYERS), dtype=float)
    c = model_rg(nc, layers, t_oxide)
    rows += [(int(n), float(c[0, j] - r_off), float(c[1, j] - g_off)) for j, n in enumerate(layers)]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--oxide", type=float, default=OXIDE_SPEC_NM, help="Oxide thickness nm (default: spec 285.0)")
    parser.add_argument("--no-offsets", action="store_true", help="Fit n only (offsets pinned at 0)")
    parser.add_argument("--table", action="store_true", help="Emit segmentation.py cal table")
    args = parser.parse_args()

    nc, r_off, g_off, rms = fit(args.oxide, with_offsets=not args.no_offsets)
    layers = np.array([a[0] for a in ANCHORS_285_IDENTITY_10X], dtype=float)
    pred = model_rg(nc, layers, args.oxide) - np.array([[r_off], [g_off]])

    print(f"oxide {args.oxide:.2f} nm, offsets {'fitted' if not args.no_offsets else 'pinned 0'}")
    print(f"n = {nc.real:.3f} - {-nc.imag:.3f}j   r_off {r_off:+.4f}  g_off {g_off:+.4f}   rms {rms:.4f}")
    print(" L      measured (R, G)      fitted (R, G)         resid")
    for j, (n, r, g, _b) in enumerate(ANCHORS_285_IDENTITY_10X):
        dr, dg = pred[0, j] - r, pred[1, j] - g
        print(f"  {n}  ({r:+.3f}, {g:+.3f})   ({pred[0, j]:+.3f}, {pred[1, j]:+.3f})   ({dr:+.3f}, {dg:+.3f})")

    rows = table_points(nc, r_off, g_off, args.oxide)
    print("\n L     table (R, G)      spacing")
    prev = None
    for n, r, g in rows:
        cur = np.array([r, g])
        sp = "" if prev is None else f"   {float(np.hypot(*(cur - prev))):.3f}"
        prev = cur
        print(f"  {n:2d}  ({r:+.3f}, {g:+.3f}){sp}")

    if args.table:
        print("\n# Graphene 1-10L on 285 nm SiO2, identity CCM, raw contrast space.")
        print("# 1-5L: empirical anchors (Scan Notebook 2026-08-31); 6-10L extrapolated")
        print(f"# via scripts/fit_graphene_285_anchors.py: n={nc.real:.3f}-{-nc.imag:.3f}j,")
        print(f"# oxide={args.oxide:.2f}, NA=0.25, r_off={r_off:+.3f}, g_off={g_off:+.3f}, rms={rms:.4f}.")
        print("GRAPHENE_1_10L_285NM_CAL_POINTS: tuple[CalPointRG, ...] = (")
        for n, r, g in rows:
            print(f"    CalPointRG(layers={n}, r={r:.4f}, g={g:.4f}),")
        print(")")


if __name__ == "__main__":
    main()
