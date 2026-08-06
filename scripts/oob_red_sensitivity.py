"""Out-of-band (NIR) sensitivity analysis for the hBN R/G contrast model.

Question: can camera response beyond 700 nm — unmodeled in
scripts/hbn_contrast.py, whose IMX183 response data stops at 700 nm —
produce the additive R-contrast offset the empirical cal sets require?

The IMX183's Bayer dyes become transparent in the NIR (channels converge
around 800-850 nm), so the R channel has real response past 700 nm.  No
public digitized curve exists past 700 nm and the K5C's internal IR-cut
edge is unknown, so the tail is parametrized instead of measured:

    response_R(λ > 700) = tail × peak(in-band R)   for λ ≤ cut, else 0

swept over `--tails` and `--cuts`.  ΔC(t) is the contrast change vs the
in-band-only baseline (tail = 0) computed on the same grid.

The illuminant matters as much as the sensor: a 3200 K halogen *rises*
into the NIR, while the scope's actual white LED (Elijah trace, measured
to 1011 nm) is down to ~2% of peak by 700-750 nm.  Both are computed.

Usage:
    uv run python scripts/oob_red_sensitivity.py
    uv run python scripts/oob_red_sensitivity.py --tails 0.1,0.3 --cuts 750,850
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hbn_contrast import (
    _IMX183_GREEN,
    _IMX183_RED,
    _IMX183_WAVELENGTHS,
    _blackbody,
    n_hbn_constant,
    reflectance,
)

_GRID = np.arange(400.0, 1010.0, 10.0)
_LED_CSV = Path(__file__).resolve().parents[1] / "calibration" / "elijah_led_spectrum.csv"

# Bands used in the summary table
_ANCHOR_BAND = (45.0, 58.0)  # Toghrul AFM 10x anchors
_PREV_BAND = (4.0, 46.0)  # prev-scope 50x AFM cal set


def _extended_response(base: np.ndarray, tail: float, cut: float) -> np.ndarray:
    """In-band response on _GRID with a flat NIR tail out to `cut` nm."""
    resp = np.zeros_like(_GRID)
    in_band = _GRID <= 700.0
    resp[in_band] = np.interp(_GRID[in_band], _IMX183_WAVELENGTHS, base)
    resp[(_GRID > 700.0) & (cut >= _GRID)] = tail * base.max()
    return resp


def _channel_contrast(r_flake: np.ndarray, r_sub: np.ndarray, resp: np.ndarray, illum: np.ndarray) -> np.ndarray:
    """Per-thickness channel contrast for response × illuminant weights."""
    w = resp * illum
    v_sub = np.trapezoid(r_sub * w, _GRID)
    v_flake = np.trapezoid(r_flake * w[None, :], _GRID, axis=1)
    return (v_flake - v_sub) / v_sub


def main():
    ap = argparse.ArgumentParser(description="Out-of-band (NIR) sensitivity analysis for the hBN contrast model")
    ap.add_argument("--n-hbn", type=float, default=2.2175, help="hBN refractive index (constant)")
    ap.add_argument("--oxide", type=float, default=90.2, help="SiO2 thickness (nm)")
    ap.add_argument("--na", type=float, default=0.25, help="objective NA")
    ap.add_argument("--tails", default="0.1,0.2,0.4", help="R tail levels, fraction of in-band R peak")
    ap.add_argument("--cuts", default="750,850,1000", help="IR-cut hard-edge wavelengths (nm)")
    ap.add_argument("--g-tail", type=float, default=0.0, help="G tail level (fraction of G peak), 0 = R-only")
    ap.add_argument("--tmax", type=int, default=100, help="max thickness (nm)")
    ap.add_argument("--output", default="/tmp/oob_red_sensitivity.png", help="output plot path")
    args = ap.parse_args()

    tails = [float(x) for x in args.tails.split(",")]
    cuts = [float(x) for x in args.cuts.split(",")]

    n_fn = n_hbn_constant(args.n_hbn)
    layer_nm = 0.333
    t_nm = np.arange(1.0, args.tmax + 1.0)

    # Reflectance is scenario-independent: compute once on the extended grid.
    r_sub = reflectance(_GRID, n_fn, 0, args.oxide, na=args.na)
    r_flake = np.stack([reflectance(_GRID, n_fn, int(round(t / layer_nm)), args.oxide, na=args.na) for t in t_nm])

    led_raw = np.loadtxt(_LED_CSV, delimiter=",")
    illums = {
        "halogen 3200K": _blackbody(_GRID, 3200.0),
        "Elijah LED": np.interp(_GRID, led_raw[:, 0], led_raw[:, 1]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    band_masks = {
        f"anchors {_ANCHOR_BAND[0]:.0f}-{_ANCHOR_BAND[1]:.0f}nm": (t_nm >= _ANCHOR_BAND[0]) & (t_nm <= _ANCHOR_BAND[1]),
        f"prev cal {_PREV_BAND[0]:.0f}-{_PREV_BAND[1]:.0f}nm": (t_nm >= _PREV_BAND[0]) & (t_nm <= _PREV_BAND[1]),
    }

    for ax, (illum_name, illum) in zip(axes, illums.items(), strict=True):
        base_r = _channel_contrast(r_flake, r_sub, _extended_response(_IMX183_RED, 0.0, 700.0), illum)
        base_g = _channel_contrast(r_flake, r_sub, _extended_response(_IMX183_GREEN, 0.0, 700.0), illum)

        print(f"\n=== {illum_name} (n={args.n_hbn}, oxide={args.oxide}nm, NA={args.na}) ===")
        header = f"{'tail':>5} {'cut':>5} |"
        for band in band_masks:
            header += f" ΔC_R {band}: mean±std |"
        header += " ΔC_R flatness std (4-100nm)"
        print(header)

        cmap = plt.get_cmap("viridis")
        for ci, cut in enumerate(cuts):
            for ti, tail in enumerate(tails):
                c_r = _channel_contrast(r_flake, r_sub, _extended_response(_IMX183_RED, tail, cut), illum)
                d_r = c_r - base_r
                if args.g_tail > 0:
                    c_g = _channel_contrast(r_flake, r_sub, _extended_response(_IMX183_GREEN, args.g_tail, cut), illum)
                    d_g = c_g - base_g
                else:
                    d_g = np.zeros_like(d_r)

                row = f"{tail:>5.2f} {cut:>5.0f} |"
                for _, mask in band_masks.items():
                    row += f" {d_r[mask].mean():+.3f} ± {d_r[mask].std():.3f}          |"
                flat_mask = t_nm >= 4.0
                row += f" {d_r[flat_mask].std():.3f}"
                if args.g_tail > 0:
                    row += f"   (ΔC_G anchors: {d_g[band_masks[list(band_masks)[0]]].mean():+.3f})"
                print(row)

                ax.plot(
                    t_nm,
                    d_r,
                    color=cmap(ci / max(len(cuts) - 1, 1)),
                    ls=["-", "--", ":"][ti % 3],
                    lw=1.5,
                    label=f"tail={tail:.2f}, cut={cut:.0f}nm",
                )

        for band, mask in band_masks.items():
            span_color = "tab:blue" if "anchor" in band else "tab:orange"
            ax.axvspan(t_nm[mask].min(), t_nm[mask].max(), alpha=0.08, color=span_color)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("hBN thickness (nm)")
        ax.set_title(f"ΔC_R from NIR leakage · {illum_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("ΔC_R (leakage − in-band-only)")

    fig.suptitle("Out-of-band R response sensitivity — shaded: anchor band (blue), prev cal band (orange)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nPlot: {args.output}")


if __name__ == "__main__":
    main()
