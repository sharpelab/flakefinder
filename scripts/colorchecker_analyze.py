#!/usr/bin/env python
"""Analyze ColorChecker calibration capture series.

Reads a frames/ directory of captures named <condition>_f<N>.png, where
condition encodes the sweep point:

    exp<MS>          exposure ramp frame (lamp 100)
    driftA_exp10     drift anchor (repeated reference condition)
    dark_exp<MS>     lamp-0 dark (ambient + electronic pedestal)
    lamp<PCT>_exp10  lamp ramp frame

Per condition: mean RGB over the central ROI, averaged across frames.
Outputs patches.json (per-condition stats), summary.json (fingerprint
ratios, linearity fit + residuals, pedestal), and diagnostic plots.

Usage:
    uv run python scripts/colorchecker_analyze.py calibration/colorchecker_dm6m_20260806
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = "RGB"
COLORS = {"R": "tab:red", "G": "tab:green", "B": "tab:blue"}


def roi_stats(path: Path, roi: int) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    rgb = img[:, :, ::-1].astype(float)
    h, w = rgb.shape[:2]
    core = rgb[h // 2 - roi // 2 : h // 2 + roi // 2, w // 2 - roi // 2 : w // 2 + roi // 2]
    return core.mean(axis=(0, 1)), core.std(axis=(0, 1))


def load_conditions(frames_dir: Path, roi: int) -> dict[str, dict]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(frames_dir.glob("*.png")):
        m = re.match(r"(.+)_f(\d+)$", p.stem)
        if not m:
            continue
        groups[m.group(1)].append(p)

    conditions = {}
    for name, paths in groups.items():
        means, stds = zip(*(roi_stats(p, roi) for p in paths), strict=True)
        means = np.array(means)
        conditions[name] = {
            "n_frames": len(paths),
            "mean_rgb": means.mean(axis=0).round(3).tolist(),
            "frame_std_rgb": means.std(axis=0).round(3).tolist(),  # across frames
            "pixel_std_rgb": np.array(stds).mean(axis=0).round(3).tolist(),  # within ROI
        }
    return conditions


def fit_linear_region(exp_ms: np.ndarray, counts: np.ndarray, lo: float, hi: float):
    """Fit counts = a*exp + b on points whose counts lie in [lo, hi]."""
    mask = (counts >= lo) & (counts <= hi)
    if mask.sum() < 3:
        return None
    a, b = np.polyfit(exp_ms[mask], counts[mask], 1)
    pred = a * exp_ms + b
    resid = counts - pred
    rms_in = float(np.sqrt(np.mean(resid[mask] ** 2)))
    return {
        "slope_per_ms": float(a),
        "intercept": float(b),
        "fit_lo": lo,
        "fit_hi": hi,
        "n_fit_points": int(mask.sum()),
        "rms_residual_in_fit": rms_in,
        "pred": pred,
        "resid": resid,
        "mask": mask,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cal_dir", type=Path, help="calibration dir containing frames/")
    ap.add_argument("--roi", type=int, default=300, help="central ROI size in px (default 300)")
    ap.add_argument("--fit-lo", type=float, default=10, help="min counts for linear fit region (default 10)")
    ap.add_argument("--fit-hi", type=float, default=120, help="max counts for linear fit region (default 120)")
    args = ap.parse_args()

    frames_dir = args.cal_dir / "frames"
    conditions = load_conditions(frames_dir, args.roi)
    (args.cal_dir / "patches.json").write_text(json.dumps(conditions, indent=2))
    print(f"{len(conditions)} conditions from {frames_dir}")

    # --- split condition types ---
    ramp = {}  # exp_ms -> rgb
    lamp_ramp = {}  # lamp_pct -> rgb
    darks = {}  # exp_ms -> rgb
    anchors = {}  # name -> rgb
    for name, c in conditions.items():
        rgb = np.array(c["mean_rgb"])
        if m := re.fullmatch(r"exp([\d.]+)", name):
            ramp[float(m.group(1))] = rgb
        elif m := re.fullmatch(r"dark_exp([\d.]+)", name):
            darks[float(m.group(1))] = rgb
        elif m := re.fullmatch(r"lamp([\d.]+)_exp10(?:_repeat)?", name):
            lamp_ramp[float(m.group(1)) + (0.001 if name.endswith("repeat") else 0)] = rgb
        elif "drift" in name:
            anchors[name] = rgb

    summary: dict = {"roi_px": args.roi, "n_conditions": len(conditions)}

    # --- exposure ramp linearity ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True, height_ratios=[2, 1])
    exp = np.array(sorted(ramp))
    # exclude conditions where ANY channel is near clip: saturated photosites
    # bloom into neighbors and contaminate the unclipped channels
    clean = np.array([max(ramp[e]) < 250 for e in exp])
    fits = {}
    for ci, ch in enumerate(CHANNELS):
        counts = np.array([ramp[e][ci] for e in exp])
        fit = fit_linear_region(exp[clean], counts[clean], args.fit_lo, args.fit_hi)
        if fit:  # re-project fit onto the full exposure axis for plotting
            a, b = fit["slope_per_ms"], fit["intercept"]
            fit["pred"] = a * exp + b
            fit["resid"] = counts - fit["pred"]
        fits[ch] = (counts, fit)
        axes[0].plot(exp, counts, "o-", ms=4, color=COLORS[ch], label=f"{ch}")
        if fit:
            axes[0].plot(exp, fit["pred"], "--", lw=1, color=COLORS[ch], alpha=0.5)
            pct = 100 * fit["resid"] / np.maximum(fit["pred"], 1)
            axes[1].plot(exp, pct, "o-", ms=4, color=COLORS[ch])
            fit_keys = ("slope_per_ms", "intercept", "fit_lo", "fit_hi", "n_fit_points", "rms_residual_in_fit")
            summary[f"linearity_{ch}"] = {k: fit[k] for k in fit_keys}
    axes[0].axhline(255, color="k", lw=0.5)
    axes[0].set_ylabel("ROI mean (counts)")
    axes[0].legend()
    axes[0].set_title("Exposure ramp — counts vs exposure (dashed: linear fit from low-count region)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("residual vs linear (%)")
    axes[1].set_xlabel("exposure (ms)")
    axes[1].set_ylim(-25, 10)
    fig.tight_layout()
    ramp_plot = args.cal_dir / "exposure_ramp.png"
    fig.savefig(ramp_plot, dpi=120)
    print(f"wrote {ramp_plot}")

    # --- lamp ramp ---
    if lamp_ramp:
        fig2, ax = plt.subplots(figsize=(8, 5))
        lp = np.array(sorted(lamp_ramp))
        for ci, ch in enumerate(CHANNELS):
            counts = np.array([lamp_ramp[p][ci] for p in lp])
            ax.plot(lp, counts, "o-", color=COLORS[ch], label=ch)
        ax.set_xlabel("lamp (%)")
        ax.set_ylabel("ROI mean (counts) @ 10ms")
        ax.set_title("Lamp drive curve")
        ax.legend()
        fig2.tight_layout()
        lamp_plot = args.cal_dir / "lamp_ramp.png"
        fig2.savefig(lamp_plot, dpi=120)
        print(f"wrote {lamp_plot}")
        summary["lamp_ramp"] = {str(p): lamp_ramp[p].round(2).tolist() for p in lp}

    # --- darks ---
    summary["darks"] = {str(e): darks[e].round(3).tolist() for e in sorted(darks)}

    # --- drift anchors ---
    summary["drift_anchors"] = {k: v.round(2).tolist() for k, v in sorted(anchors.items())}

    # --- fingerprint at reference condition (10ms from ramp) ---
    if 10.0 in ramp:
        r, g, b = ramp[10.0]
        summary["reference_10ms_rgb"] = [round(r, 2), round(g, 2), round(b, 2)]
        summary["cross_channel_ratios"] = {
            "G_over_R": round(g / r, 4),
            "G_over_B": round(g / b, 4),
            "R_over_B": round(r / b, 4),
        }
        mx = max(r, g, b)
        summary["wb_gains_rgb"] = [round(mx / r, 4), round(mx / g, 4), round(mx / b, 4)]

    (args.cal_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.cal_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
