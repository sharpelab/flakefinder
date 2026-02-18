"""Analyze autofocus_demo.py output and produce visualizations.

Loads summary.json from an autofocus output directory, plots sharpness curves,
and generates mosaics from debug frames if available.
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mosaic_util import make_mosaic
from PIL import Image, ImageDraw, ImageFont


def load_summary(dir_path: Path) -> dict:
    """Load summary.json from autofocus output directory.

    Handles two formats:
    - autofocus_demo format: has "params", "before", "after" keys
    - focus_map debug format: has "initial", "final_sharpness", no "params"/"before"/"after"

    Returns the raw dict with a "_format" key ("demo" or "focus_map") for callers
    to branch on.
    """
    summary_path = dir_path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {dir_path}")
    with open(summary_path) as f:
        data = json.load(f)

    if "params" in data:
        data["_format"] = "demo"
    else:
        data["_format"] = "focus_map"
    return data


def normalize_summary(summary: dict) -> dict:
    """Return a dict with normalized keys so plotting code can be format-agnostic.

    Adds/aliases:
    - "before" -> from "initial" in focus_map format
    - "after"  -> synthesized from selected + final_sharpness in focus_map format
    - "params" -> synthesized from scan metadata in focus_map format
    """
    fmt = summary.get("_format", "demo")
    if fmt == "demo":
        return summary

    # Focus map format: build compatible "before" and "after" dicts
    out = dict(summary)

    # "before" <- "initial"
    if "initial" in summary and "before" not in summary:
        out["before"] = summary["initial"]

    # "after" <- synthesized from selected Z + final_sharpness
    if "after" not in summary:
        selected = summary.get("selected", {})
        final_s = summary.get("final_sharpness")
        if selected and final_s is not None:
            out["after"] = {
                "z_um": selected["z_um"],
                "sharpness": final_s,
            }
        elif selected:
            out["after"] = dict(selected)

    # "params" <- synthesized from scan metadata
    if "params" not in summary:
        scan = summary.get("scan", {})
        initial = summary.get("initial", {})
        out["params"] = {
            "z_initial_um": initial.get("z_um"),
            "range_um": scan.get("z_range_um"),
            "objective_position": scan.get("objective_position"),
            "fine_pass": "fine" in summary,
            "super_fine_pass": "super_fine" in summary,
            # x_um/y_um not available in focus_map debug summaries
        }

    return out


def find_debug_dir(dir_path: Path, explicit: str | None) -> Path | None:
    """Auto-detect debug frames directory.

    Search order:
    1. Explicit --debug-dir if given
    2. dir_path/coarse/ (debug-dir was set to same as output dir with --fine)
    3. frame_*.png directly in dir_path (no fine pass, debug-dir = output dir)
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    # Check for frames/ subdir (autofocus_demo.py convention: --debug-dir frames/)
    frames_dir = dir_path / "frames"
    if frames_dir.exists():
        if (frames_dir / "coarse").exists():
            return frames_dir
        if list(frames_dir.glob("frame_*.png")):
            return frames_dir

    # Check for coarse/ subdir directly (debug-dir = output dir with --fine)
    if (dir_path / "coarse").exists():
        return dir_path

    # Check for frames directly in dir (no fine pass, debug-dir = output dir)
    if list(dir_path.glob("frame_*.png")):
        return dir_path

    return None


def parse_frame_filename(filename: str) -> dict | None:
    """Parse frame_NNN_z_ZZZZ.Z_s_SS.S.{jpg,png} into components."""
    m = re.match(r"frame_(\d+)_z_([\d.]+)_s_([\d.]+)\.png", filename)
    if not m:
        return None
    return {
        "frame": int(m.group(1)),
        "z_um": float(m.group(2)),
        "sharpness": float(m.group(3)),
    }


def collect_debug_frames(debug_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Collect coarse, fine, and super_fine frame paths from debug directory.

    Returns (coarse_paths, fine_paths, super_fine_paths) sorted by frame number.
    """
    coarse_dir = debug_dir / "coarse"
    fine_dir = debug_dir / "fine"
    super_fine_dir = debug_dir / "super_fine"

    if coarse_dir.exists():
        coarse = sorted(coarse_dir.glob("frame_*.png"))
    else:
        # Frames directly in debug_dir (no fine pass)
        coarse = sorted(debug_dir.glob("frame_*.png"))

    fine = sorted(fine_dir.glob("frame_*.png")) if fine_dir.exists() else []
    super_fine = sorted(super_fine_dir.glob("frame_*.png")) if super_fine_dir.exists() else []

    return coarse, fine, super_fine


def interpolate_sharpness(z: float, curve: list[dict]) -> float | None:
    """Linearly interpolate sharpness at a given Z from curve data."""
    if not curve:
        return None

    # Sort by Z
    sorted_curve = sorted(curve, key=lambda r: r["z_um"])
    z_vals = [r["z_um"] for r in sorted_curve]
    s_vals = [r["sharpness"] for r in sorted_curve]

    if z <= z_vals[0]:
        return s_vals[0]
    if z >= z_vals[-1]:
        return s_vals[-1]

    # Find bracketing points
    for i in range(len(z_vals) - 1):
        if z_vals[i] <= z <= z_vals[i + 1]:
            alpha = (z - z_vals[i]) / (z_vals[i + 1] - z_vals[i])
            return s_vals[i] + alpha * (s_vals[i + 1] - s_vals[i])

    return None


## -- Sharpness metrics for multi-metric comparison --


def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def metric_tenengrad(img: np.ndarray) -> float:
    gray = _to_gray(img)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.mean(cv2.magnitude(sx, sy))[0]


def metric_laplacian(img: np.ndarray) -> float:
    gray = _to_gray(img)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def metric_brenner(img: np.ndarray) -> float:
    gray = _to_gray(img)
    g = gray.astype(np.float64)
    diff = g[:, 2:] - g[:, :-2]
    return np.mean(diff**2)


def metric_normalized_variance(img: np.ndarray) -> float:
    gray = _to_gray(img)
    mu = gray.mean()
    return gray.var() / mu if mu > 0 else 0


def metric_vollath_f4(img: np.ndarray) -> float:
    gray = _to_gray(img)
    g = gray.astype(np.float64)
    auto1 = np.mean(g[:, :-1] * g[:, 1:])
    auto2 = np.mean(g[:, :-2] * g[:, 2:])
    return auto1 - auto2


METRICS = [
    ("Tenengrad", metric_tenengrad),
    ("Laplacian", metric_laplacian),
    ("Brenner", metric_brenner),
    ("NormVar", metric_normalized_variance),
    ("Vollath F4", metric_vollath_f4),
]


def compute_metrics_from_frames(
    frame_paths: list[Path],
) -> tuple[list[float], dict[str, list[float]]]:
    """Load frames and compute all metrics.

    Returns (z_values, {metric_name: [values...]}).
    """
    z_values = []
    results = {name: [] for name, _ in METRICS}

    for p in sorted(frame_paths):
        info = parse_frame_filename(p.name)
        if info is None:
            continue
        z_values.append(info["z_um"])
        img = cv2.imread(p)
        assert img is not None, f"Failed to read image: {p}"
        for name, fn in METRICS:
            results[name].append(fn(img))

    return z_values, results


def plot_metrics_comparison(
    coarse_paths: list[Path],
    fine_paths: list[Path],
    output_path: Path,
    pick_z: float | None = None,
    summary: dict | None = None,
    quiet: bool = False,
    super_fine_paths: list[Path] | None = None,
) -> None:
    """Compute multiple sharpness metrics on debug frames and plot comparison.

    If summary contains inline metrics (from --all-metrics), also shows
    live-computed values alongside the recomputed-from-disk values.
    """
    # Compute metrics on all available frames (coarse + fine + super_fine combined)
    all_paths = list(coarse_paths) + list(fine_paths) + list(super_fine_paths or [])
    if not all_paths:
        return

    if not quiet:
        print(f"Computing metrics on {len(all_paths)} frames...")

    z_vals, raw_metrics = compute_metrics_from_frames(all_paths)
    z_arr = np.array(z_vals)

    # Extract live metrics from summary.json if available
    live_metrics = None
    if summary:
        curves = summary.get("sharpness_curve", [])
        fine_curve = summary.get("fine_sharpness_curve", [])
        sf_curve = summary.get("super_fine_sharpness_curve", [])
        all_curves = curves + (fine_curve or []) + (sf_curve or [])
        if all_curves and "metrics" in all_curves[0]:
            # Map from summary metric names to display names
            name_map = {
                "tenengrad": "Tenengrad",
                "laplacian": "Laplacian",
                "brenner": "Brenner",
                "normalized_variance": "NormVar",
                "vollath_f4": "Vollath F4",
            }
            live_metrics = {display: [] for display in name_map.values()}
            live_z = []
            for r in all_curves:
                live_z.append(r["z_um"])
                for key, display in name_map.items():
                    live_metrics[display].append(r["metrics"].get(key, 0))
            live_z_arr = np.array(live_z)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Style: tenengrad (the default metric) is thicker/bolder
    styles = {
        "Tenengrad": {"color": "#4488cc", "linewidth": 2.5, "marker": "o", "markersize": 5},
        "Laplacian": {"color": "#ee8833", "linewidth": 1.5, "marker": "s", "markersize": 4},
        "Brenner": {"color": "#44bb66", "linewidth": 1.5, "marker": "^", "markersize": 4},
        "NormVar": {"color": "#cc4488", "linewidth": 1.5, "marker": "D", "markersize": 4},
        "Vollath F4": {"color": "#8844cc", "linewidth": 1.5, "marker": "v", "markersize": 4},
    }

    best_zs = {}
    dynamic_ranges = {}

    for name, _ in METRICS:
        vals = np.array(raw_metrics[name])
        v_min, v_max = vals.min(), vals.max()
        drange = v_max - v_min
        dynamic_ranges[name] = (drange / v_max * 100) if v_max > 0 else 0

        # Normalize to [0, 1]
        if drange > 0:
            normed = (vals - v_min) / drange
        else:
            normed = np.zeros_like(vals)

        best_idx = np.argmax(normed)
        best_zs[name] = z_arr[best_idx]

        style = styles.get(name, {"color": "gray", "linewidth": 1, "marker": ".", "markersize": 3})
        suffix = " *" if name == "Tenengrad" else ""
        ax.plot(
            z_arr,
            normed,
            "-",
            label=f"{name}{suffix}  (DR={dynamic_ranges[name]:.0f}%, best={best_zs[name]:.0f})",
            **style,  # type: ignore[arg-type]  # matplotlib stub limitation
            alpha=0.85,
            zorder=3,
        )

    # Mark best Z per metric with thin vertical lines
    for name, bz in best_zs.items():
        style = styles.get(name, {"color": "gray"})
        ax.axvline(bz, color=style["color"], linestyle=":", linewidth=0.8, alpha=0.5)

    # Highlight disagreements: find metrics whose best Z differs by >10µm
    best_z_list = list(best_zs.values())
    z_spread = max(best_z_list) - min(best_z_list)
    if z_spread > 10:
        ax.text(
            0.02,
            0.97,
            f"Best Z spread: {z_spread:.0f} µm (>10 µm disagreement)",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    # Pick-Z marker
    if pick_z is not None:
        ax.axvline(
            pick_z,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            label=f"Pick Z = {pick_z:.1f}",
            zorder=4,
        )

    ax.set_xlabel("Z position (µm)")
    ax.set_ylabel("Normalized sharpness [0-1]")
    ax.set_title("Multi-metric sharpness comparison (* = default metric)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Compute live best-Z values if available
    live_best_zs = {}
    if live_metrics is not None:
        for name in live_metrics:
            vals = np.array(live_metrics[name])
            live_best_zs[name] = live_z_arr[np.argmax(vals)]

    if not quiet:
        print(f"Metrics comparison saved: {output_path}")
        # Print table
        print()
        if live_best_zs:
            print(f"  {'Metric':<12} {'Dyn.Range':>9} {'Best Z':>10} {'Live Best Z':>12}")
            print(f"  {'-' * 12} {'-' * 9} {'-' * 10} {'-' * 12}")
            for name, _ in METRICS:
                marker = " *" if name == "Tenengrad" else ""
                live_str = f"{live_best_zs[name]:>11.1f}" if name in live_best_zs else "         n/a"
                diff = abs(best_zs[name] - live_best_zs.get(name, best_zs[name]))
                warn = " !" if diff > 10 else ""
                print(f"  {name:<12} {dynamic_ranges[name]:>8.1f}% {best_zs[name]:>9.1f}{marker} {live_str}{warn}")
        else:
            print(f"  {'Metric':<12} {'Dyn.Range':>9} {'Best Z':>10}")
            print(f"  {'-' * 12} {'-' * 9} {'-' * 10}")
            for name, _ in METRICS:
                marker = " *" if name == "Tenengrad" else ""
                print(f"  {name:<12} {dynamic_ranges[name]:>8.1f}% {best_zs[name]:>9.1f}{marker}")
        if z_spread > 10:
            print(f"\n  Best Z spread: {z_spread:.0f} µm (metrics disagree >10 µm)")
        if live_best_zs:
            diffs = {n: abs(best_zs[n] - live_best_zs[n]) for n in best_zs if n in live_best_zs}
            if any(d > 10 for d in diffs.values()):
                print("  ! = recomputed vs live best Z differ >10 µm (JPEG compression artifact on flat curves)")


def plot_sharpness_curve(
    summary: dict,
    dir_path: Path,
    output_path: Path,
    pick_z: float | None = None,
    quiet: bool = False,
) -> None:
    """Generate sharpness curve plot with markers and inset images."""
    norm = normalize_summary(summary)
    coarse_curve = norm.get("sharpness_curve", [])
    fine_curve = norm.get("fine_sharpness_curve", [])
    super_fine_curve = norm.get("super_fine_sharpness_curve", [])
    params = norm["params"]
    before = norm["before"]
    best = norm["selected"]  # new: "selected", legacy: "best"
    after = norm["after"]

    # Load inset images — try both .jpg (demo) and .png (focus_map) extensions
    inset_images = {}
    inset_candidates = [
        ("before", ["before.jpg", "initial.png"]),
        ("best_scan_frame", ["best_scan_frame.jpg"]),
        ("after", ["after.jpg", "after.png", "final.png"]),
    ]
    for key, filenames in inset_candidates:
        for name in filenames:
            img_path = dir_path / name
            if img_path.exists():
                inset_images[key] = Image.open(img_path)
                break

    has_insets = len(inset_images) > 0
    fig_width = 14 if has_insets else 10

    fig = plt.figure(figsize=(fig_width, 7))

    if has_insets:
        # Main plot on left ~65%, inset panel on right ~35%
        ax = fig.add_axes((0.08, 0.12, 0.55, 0.80))
        inset_left = 0.66
        inset_width = 0.31
    else:
        ax = fig.add_axes((0.08, 0.12, 0.88, 0.80))

    # Plot curves, scan ranges, and markers (shared with analyze_focus_map --curves)
    from flakefinder.commands.analyze_focus_map import plot_af_curves

    plot_af_curves(
        ax,
        coarse_curve=coarse_curve,
        fine_curve=fine_curve,
        super_fine_curve=super_fine_curve,
        initial_z_um=before["z_um"],
        selected_z_um=best["z_um"],
        selected_sharpness=best["sharpness"],
        initial_label=f"Initial Z = {before['z_um']:.1f}",
        selected_label=f"Best Z = {best['z_um']:.1f}",
    )

    # Only show final Z line if it differs from best Z
    if abs(after["z_um"] - best["z_um"]) > 0.1:
        ax.axvline(
            after["z_um"],
            color="purple",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"After Z = {after['z_um']:.1f}",
            zorder=2,
        )

    # Pick-Z marker
    if pick_z is not None:
        all_curves = coarse_curve + fine_curve + super_fine_curve
        interp_s = interpolate_sharpness(pick_z, all_curves)
        label = f"Pick Z = {pick_z:.1f}"
        if interp_s is not None:
            label += f" (~S={interp_s:.1f})"
        ax.axvline(pick_z, color="green", linestyle="--", linewidth=1.5, alpha=0.9, label=label, zorder=4)

    ax.set_xlabel("Z position (µm)")
    ax.set_ylabel("Sharpness")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Title
    z_adj = best["z_um"] - before["z_um"]
    title = f"Autofocus: Z adjustment {z_adj:+.1f} µm"
    passes = ["coarse"]
    if params.get("fine_pass"):
        passes.append("fine")
    if params.get("super_fine_pass"):
        passes.append("super fine")
    if len(passes) > 1:
        title += f" ({' + '.join(passes)})"
    ax.set_title(title)

    # Inset images panel
    if has_insets:
        inset_order = [
            ("before", f"Before (Z={before['z_um']:.1f}, S={before['sharpness']:.1f})"),
            ("best_scan_frame", f"Best scan (Z={best['z_um']:.1f}, S={best['sharpness']:.1f})"),
            ("after", f"After (Z={after['z_um']:.1f}, S={after['sharpness']:.1f})"),
        ]
        available = [(key, label) for key, label in inset_order if key in inset_images]
        n_insets = len(available)

        if n_insets > 0:
            inset_h = 0.80 / n_insets - 0.02
            for i, (key, label) in enumerate(available):
                img = inset_images[key]
                y_pos = 0.12 + (n_insets - 1 - i) * (inset_h + 0.02)
                ax_img = fig.add_axes((inset_left, y_pos, inset_width, inset_h))
                ax_img.imshow(np.array(img))
                ax_img.set_title(label, fontsize=8)
                ax_img.axis("off")

    plt.savefig(output_path, dpi=150)
    plt.close()
    if not quiet:
        print(f"Sharpness curve saved: {output_path}")


def make_frame_mosaic(
    frame_paths: list[Path],
    output_path: Path,
    pass_name: str,
    quiet: bool = False,
) -> None:
    """Generate labeled mosaic from debug frame images."""
    if not frame_paths:
        return

    # Parse info from filenames and build labels
    parsed = []
    for p in frame_paths:
        info = parse_frame_filename(p.name)
        if info:
            parsed.append((p, info))

    if not parsed:
        if not quiet:
            print(f"No parseable frame filenames in {pass_name}, skipping mosaic.")
        return

    # Sort by frame number
    parsed.sort(key=lambda x: x[1]["frame"])
    paths = [p for p, _ in parsed]

    # Determine grid layout: prefer roughly square
    n = len(paths)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Generate mosaic without labels first, then overlay our custom labels
    canvas = make_mosaic(
        paths,
        rows=rows,
        cols=cols,
        max_dim=5000,
        margin=2,
        bg_color=(30, 30, 30),
        labels=False,
    )

    # Compute thumbnail size to draw labels at correct positions
    first_img = Image.open(paths[0])
    aspect = first_img.width / first_img.height
    margin = 2
    thumb_w_from_w = (5000 - (cols - 1) * margin) / cols
    thumb_h_from_h = (5000 - (rows - 1) * margin) / rows
    if thumb_w_from_w / (thumb_h_from_h if thumb_h_from_h else 1) <= aspect:
        thumb_w = int(thumb_w_from_w)
        thumb_h = int(thumb_w / aspect)
    else:
        thumb_h = int(thumb_h_from_h)
        thumb_w = int(thumb_h * aspect)

    font_size = max(10, min(36, int(thumb_h * 0.035)))
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    pad = 2
    for i, (_, info) in enumerate(parsed):
        if i >= rows * cols:
            break
        row_i = i // cols
        col_i = i % cols
        x = col_i * (thumb_w + margin)
        y = row_i * (thumb_h + margin)

        label = f"#{info['frame']:03d} Z={info['z_um']:.1f} S={info['sharpness']:.1f}"
        tx = x + pad
        ty = y + pad
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=(0, 0, 0, 180),
        )
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    canvas.save(str(output_path), quality=95)
    if not quiet:
        print(f"{pass_name} mosaic ({n} frames) saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze autofocus output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dir",
        type=Path,
        help="Autofocus output directory (must contain summary.json), "
        "or focus map JSON when used with --focus-map-point",
    )
    parser.add_argument(
        "--focus-map-point",
        type=str,
        default=None,
        metavar="LABEL",
        help="Extract a specific point from a focus map JSON (e.g. g07, c05)",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Path to debug frames directory (default: auto-detect)",
    )
    parser.add_argument(
        "--pick-z",
        type=float,
        default=None,
        help="Mark a user-picked Z on the sharpness curve (green dashed line)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for generated files (default: same as DIR)",
    )
    parser.add_argument(
        "--no-mosaic",
        action="store_true",
        help="Skip mosaic generation",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Recompute sharpness with multiple metrics and plot comparison (requires debug frames)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    input_path = args.dir.resolve()

    if not input_path.exists():
        print(f"Error: Not found: {input_path}")
        return 1

    # Load summary — either from focus map point or directory
    if args.focus_map_point:
        # Extract point from focus map JSON
        if not input_path.is_file():
            print(f"Error: --focus-map-point requires a JSON file, got directory: {input_path}")
            return 1
        with open(input_path) as f:
            focus_map = json.load(f)
        points = focus_map.get("sample_points", [])
        label = args.focus_map_point.lower()
        summary = None
        for p in points:
            pt_label = f"{p['type'][0]}{p['index']:02d}"
            if pt_label == label:
                summary = dict(p)
                break
        if summary is None:
            available = [f"{p['type'][0]}{p['index']:02d}" for p in points]
            print(f"Error: Point '{label}' not found. Available: {', '.join(available)}")
            return 1
        if "error" in summary:
            print(f"Error: Point '{label}' failed autofocus (no data)")
            return 1
        summary["_format"] = "focus_map"
        dir_path = input_path.parent
    else:
        dir_path = input_path
        if not dir_path.is_dir():
            print(f"Error: Expected a directory (or use --focus-map-point with a JSON file): {dir_path}")
            return 1
        try:
            summary = load_summary(dir_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    output_dir = (args.output or dir_path).resolve()

    # Find debug frames
    debug_dir = find_debug_dir(dir_path, args.debug_dir)

    # Normalize summary so rest of code can use consistent keys
    norm = normalize_summary(summary)

    if not args.quiet:
        print("Autofocus Analysis")
        print("=" * 50)
        print(f"Input: {dir_path}/")
        print(f"Summary: {dir_path / 'summary.json'}")
        print(f"Format: {summary.get('_format', 'unknown')}")
        print(f"Debug frames: {debug_dir or '(not found)'}")
        print()

        params = norm["params"]
        print("Params:")
        if params.get("x_um") is not None and params.get("y_um") is not None:
            print(f"  Position: X={params['x_um']:.1f}, Y={params['y_um']:.1f}")
        if params.get("objective_position"):
            print(f"  Objective: position {params['objective_position']}")
        if params.get("range_um") is not None:
            fine_str = "yes" if params.get("fine_pass") else "no"
            sf_str = ", super_fine: yes" if params.get("super_fine_pass") else ""
            print(f"  Range: {params['range_um']:.1f} µm (fine: {fine_str}{sf_str})")
        print()

        before = norm["before"]
        best = norm["selected"]
        after = norm["after"]

        print("Results:")
        print(f"  Initial Z:  {before['z_um']:.2f} µm  (sharpness: {before['sharpness']:.2f})")
        print(f"  Selected Z: {best['z_um']:.2f} µm  (sharpness: {best['sharpness']:.2f})")
        print(f"  After Z:    {after['z_um']:.2f} µm  (sharpness: {after['sharpness']:.2f})")

        z_adj = best["z_um"] - before["z_um"]
        print(f"  Adjustment: {z_adj:+.2f} µm")

        if before["sharpness"] > 0:
            improvement = after["sharpness"] - before["sharpness"]
            pct = (after["sharpness"] / before["sharpness"] - 1) * 100
            print(f"  Improvement: {improvement:+.2f} ({pct:+.1f}%)")

        if summary.get("stayed_at_initial"):
            print("  (Stayed at initial position)")

        if args.pick_z is not None:
            all_curves = (
                norm.get("sharpness_curve", [])
                + norm.get("fine_sharpness_curve", [])
                + norm.get("super_fine_sharpness_curve", [])
            )
            interp_s = interpolate_sharpness(args.pick_z, all_curves)
            s_str = f" (interpolated sharpness: ~{interp_s:.2f})" if interp_s is not None else ""
            print(f"  Pick Z:     {args.pick_z:.2f} µm{s_str}")

        print()

        scan = norm["scan"]
        coarse_n = len(norm.get("sharpness_curve", []))
        fine_n = len(norm.get("fine_sharpness_curve", []))
        sf_n = len(norm.get("super_fine_sharpness_curve", []))
        print("Scan stats:")
        print(f"  Duration: {scan['duration_s']:.2f}s")
        parts = [f"{coarse_n} coarse"]
        if fine_n > 0:
            parts.append(f"{fine_n} fine")
        if sf_n > 0:
            parts.append(f"{sf_n} super_fine")
        total = coarse_n + fine_n + sf_n
        if len(parts) > 1:
            print(f"  Frames: {' + '.join(parts)} = {total} total")
        else:
            print(f"  Frames: {coarse_n}")
        print(f"  Z samples: {scan['z_sample_count']}")
        print()

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate sharpness curve plot
    plot_path = output_dir / "autofocus_analysis.png"
    plot_sharpness_curve(summary, dir_path, plot_path, pick_z=args.pick_z, quiet=args.quiet)

    # Generate mosaics from debug frames
    if not args.no_mosaic and debug_dir:
        coarse_paths, fine_paths, sf_paths = collect_debug_frames(debug_dir)

        if coarse_paths:
            coarse_mosaic_path = output_dir / "coarse_mosaic.jpg"
            make_frame_mosaic(coarse_paths, coarse_mosaic_path, "Coarse", quiet=args.quiet)

        if fine_paths:
            fine_mosaic_path = output_dir / "fine_mosaic.jpg"
            make_frame_mosaic(fine_paths, fine_mosaic_path, "Fine", quiet=args.quiet)

        if sf_paths:
            sf_mosaic_path = output_dir / "super_fine_mosaic.jpg"
            make_frame_mosaic(sf_paths, sf_mosaic_path, "Super fine", quiet=args.quiet)

        if not coarse_paths and not fine_paths and not sf_paths and not args.quiet:
            print("No debug frames found, skipping mosaics.")
    elif not args.no_mosaic and not debug_dir and not args.quiet:
        print("No debug frames found, skipping mosaics.")

    # Multi-metric comparison
    if args.metrics:
        if debug_dir:
            coarse_paths, fine_paths, sf_paths = collect_debug_frames(debug_dir)
            if coarse_paths or fine_paths or sf_paths:
                metrics_path = output_dir / "metrics_comparison.png"
                plot_metrics_comparison(
                    coarse_paths,
                    fine_paths,
                    metrics_path,
                    pick_z=args.pick_z,
                    summary=summary,
                    quiet=args.quiet,
                    super_fine_paths=sf_paths,
                )
            elif not args.quiet:
                print("No debug frames found, skipping metrics comparison.")
        elif not args.quiet:
            print("No debug frames found, skipping metrics comparison.")

    return 0


if __name__ == "__main__":
    exit(main())
