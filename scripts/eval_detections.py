"""Evaluate flake detector against reference datasets with labeled masks.

Usage:
    python scripts/eval_detections.py downloads/flakes_scan129/ \
        --flatfield calibration/flatfield_20x_bin3.npy \
        -o /tmp/eval_output/

    python scripts/eval_detections.py downloads/flakes_scan129/ \
        --flatfield calibration/flatfield_20x_bin3.npy \
        --contrast-offset 15 --min-size 1000 \
        -o /tmp/eval_output/
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mosaic_util import make_mosaic

from flakefinder.scan_utils import apply_flatfield
from flakefinder.segmentation import (
    CurveDetectorConfig,
    Detection,
    DetectorConfig,
    draw_scale_bar,
    segment_frame,
)

# R-G calibration curve: R = 0.193*G^2 - 0.217*G - 0.604
# Polynomial coefficients (G^2, G, constant)
CAL_POLY = np.array([0.193, -0.217, -0.604])

# AFM-verified calibration data from bn_thickness_calibration.md
# (R_contrast, G_contrast, thickness_nm or None)
CAL_DATA = [
    (-0.600, 0.286, 4.6),
    (-0.671, 0.079, 5.6),
    (-0.597, 0.219, 6.3),
    (-0.645, 0.313, 7.0),
    (-0.649, 0.376, 8.1),
    (-0.670, 0.409, 8.6),
    (-0.673, 0.526, 10.2),
    (-0.692, 0.948, 14.1),
    (-0.481, 1.613, 18.0),
    (0.421, 2.907, 26.0),
    (-0.026, 2.356, None),
    (2.520, 4.636, 46.0),
]

# Classification colors (BGR for cv2)
COLOR_THIN = (0, 255, 0)  # green
COLOR_MEDIUM = (0, 200, 255)  # orange
COLOR_THICK = (0, 100, 255)  # dark orange
COLOR_POSSIBLE = (255, 255, 0)  # cyan
COLOR_NON_HBN = (128, 128, 128)  # gray
COLOR_TARGET = (255, 255, 255)  # white outline for target matches

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "thin": COLOR_THIN,
    "medium": COLOR_MEDIUM,
    "thick": COLOR_THICK,
    "possible": COLOR_POSSIBLE,
}


def classify_detection(r: float, g: float, b: float, config: DetectorConfig) -> tuple[str | None, tuple[int, int, int]]:
    """Classify a detection by its contrast triple.

    Returns (label, bgr_color).
    """
    label = config.classify(r, g, b)
    color = _LABEL_COLORS.get(label, COLOR_NON_HBN) if label is not None else COLOR_NON_HBN
    return label, color


def mask_centroid(mask_path: Path) -> tuple[float, float] | None:
    """Compute centroid of nonzero pixels in a grayscale mask. Returns (cx, cy) or None."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def match_detection_to_target(
    detections: list[Detection],
    target_center: tuple[float, float],
    match_radius: float,
) -> int | None:
    """Find the detection closest to target_center within match_radius. Returns index or None."""
    best_idx = None
    best_dist = float("inf")
    tx, ty = target_center
    for i, det in enumerate(detections):
        cx, cy = det["center"]
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if dist < match_radius and dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def draw_eval_frame(
    raw_image: np.ndarray,
    detections: list[Detection],
    classifications: list[tuple[str, tuple[int, int, int]]],
    target_idx: int | None,
    um_per_px: float,
    config: DetectorConfig,
) -> np.ndarray:
    """Draw color-coded detection boxes with classification labels on raw image."""
    vis = raw_image.copy()

    for i, (det, (cls_label, color)) in enumerate(zip(detections, classifications, strict=True)):
        bx, by, bw, bh = det["bbox"]
        r_contrast, g_contrast, b_contrast = det["contrast_rgb"]
        d = config.cal_projection(r_contrast, g_contrast, b_contrast).dist

        # Draw filled bbox
        thickness = 2
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, thickness)

        # White outline for target match
        if i == target_idx:
            cv2.rectangle(vis, (bx - 3, by - 3), (bx + bw + 3, by + bh + 3), COLOR_TARGET, 2)

        # Label above: #idx classification d=X.XX
        label_top = f"#{i} {cls_label} d={d:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thick_text = 1
        cv2.putText(vis, label_top, (bx, by - 8), font, font_scale, (0, 0, 0), thick_text + 2)
        cv2.putText(vis, label_top, (bx, by - 8), font, font_scale, color, thick_text)

        # Label below: R=X.XX G=X.XX
        label_bot = f"R={r_contrast:.2f} G={g_contrast:.2f}"
        cv2.putText(vis, label_bot, (bx, by + bh + 14), font, font_scale, (0, 0, 0), thick_text + 2)
        cv2.putText(vis, label_bot, (bx, by + bh + 14), font, font_scale, color, thick_text)

    if um_per_px > 0:
        draw_scale_bar(vis, um_per_px)

    return vis


def discover_frames(ref_dir: Path) -> list[dict]:
    """Find all Chip_N/Flake_M dirs with raw_img.png and flake_mask.png."""
    frames = []
    for chip_dir in sorted(ref_dir.iterdir()):
        if not chip_dir.is_dir() or not chip_dir.name.startswith("Chip_"):
            continue
        for flake_dir in sorted(chip_dir.iterdir()):
            if not flake_dir.is_dir() or not flake_dir.name.startswith("Flake_"):
                continue
            raw_path = flake_dir / "raw_img.png"
            mask_path = flake_dir / "flake_mask.png"
            if raw_path.exists() and mask_path.exists():
                frames.append(
                    {
                        "raw_path": raw_path,
                        "mask_path": mask_path,
                        "label": f"{chip_dir.name}/{flake_dir.name}",
                    }
                )
    return frames


def generate_rg_scatter(
    all_detections: list[tuple[float, float, str, bool]],
    output_path: Path,
    config: CurveDetectorConfig,
) -> None:
    """Generate R vs G scatter plot with calibration curve.

    all_detections: list of (R, G, classification, is_target_match)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot calibration curve
    g_range = np.linspace(-0.2, 5.0, 200)
    r_curve = np.polyval(config.cal_poly, g_range)
    ax.plot(g_range, r_curve, "k-", linewidth=2, label="Calibration curve", zorder=1)

    # Plot calibration data points as diamonds
    cal_g = [d[1] for d in CAL_DATA]
    cal_r = [d[0] for d in CAL_DATA]
    ax.scatter(
        cal_g,
        cal_r,
        marker="D",
        s=60,
        c="goldenrod",
        edgecolors="black",
        linewidths=0.8,
        label="AFM calibration data",
        zorder=3,
    )

    # Color map for classifications
    cls_colors = {
        "thin": "#00ff00",
        "medium": "#ffa500",
        "thick": "#ff6400",
        "possible": "#00ffff",
        config.non_match_label: "#808080",
    }

    # Plot detection points
    for cls_name, hex_color in cls_colors.items():
        non_target = [(r, g) for r, g, c, t in all_detections if c == cls_name and not t]
        target = [(r, g) for r, g, c, t in all_detections if c == cls_name and t]

        if non_target:
            gs, rs = zip(*[(g, r) for r, g in non_target], strict=True)
            ax.scatter(gs, rs, c=hex_color, s=40, edgecolors="black", linewidths=0.5, zorder=4)

        if target:
            gs, rs = zip(*[(g, r) for r, g in target], strict=True)
            ax.scatter(
                gs,
                rs,
                c=hex_color,
                s=120,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                label=f"{cls_name} (target)" if target else None,
                zorder=5,
            )

    # Legend entries for classification colors (non-target dots)
    for cls_name, hex_color in cls_colors.items():
        count = sum(1 for _, _, c, _ in all_detections if c == cls_name)
        if count > 0:
            ax.scatter([], [], c=hex_color, s=40, edgecolors="black", linewidths=0.5, label=f"{cls_name} ({count})")

    ax.set_xlabel("G normalized contrast", fontsize=12)
    ax.set_ylabel("R normalized contrast", fontsize=12)
    ax.set_title("Detection R-G Contrast with Calibration Curve")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate flake detector against reference datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ref_dir", type=Path, help="Reference directory with Chip_N/Flake_M structure")
    parser.add_argument("--flatfield", type=Path, default=None, help="Flatfield .npy file")
    parser.add_argument(
        "--material",
        default="hbn_medium",
        choices=DetectorConfig.material_names(),
        help="Material preset",
    )
    parser.add_argument("--pixel-size", type=float, default=0.36, help="µm per pixel (20x bin3)")
    parser.add_argument("--contrast-offset", type=float, default=None, help="Override contrast offset from preset")
    parser.add_argument("--min-size-um", type=float, default=None, help="Override min detection area (µm²)")
    parser.add_argument("--match-radius", type=int, default=100, help="Target matching radius (px)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--show", action="store_true", help="Open results after generation")
    args = parser.parse_args()

    from dataclasses import replace as _replace

    config = DetectorConfig.from_material(args.material)
    _overrides = {}
    if args.contrast_offset is not None:
        _overrides["contrast_offset"] = args.contrast_offset
    if args.min_size_um is not None:
        _overrides["min_size_um2"] = args.min_size_um
    if _overrides:
        config = _replace(config, **_overrides)

    # Discover frames
    frames = discover_frames(args.ref_dir)
    if not frames:
        print(f"No frames found in {args.ref_dir}")
        return 1
    print(f"Found {len(frames)} frames")

    # Load flatfield
    flatfield = None
    if args.flatfield:
        ff = np.load(str(args.flatfield)).astype(np.float32)
        flatfield = ff[:, :, ::-1]  # RGB (numpy save convention) -> BGR (cv2)

    args.output.mkdir(parents=True, exist_ok=True)

    # Process each frame
    annotated_paths: list[str] = []
    annotated_labels: list[str] = []
    all_scatter_points: list[tuple[float, float, str, bool]] = []  # (R, G, cls, is_target)
    summary_lines: list[str] = []

    total_targets = 0
    total_matched = 0
    cls_counts: dict[str, int] = {}
    target_cls_counts: dict[str, int] = {}

    for frame_info in frames:
        raw = cv2.imread(frame_info["raw_path"])
        if raw is None:
            print(f"  Failed to read {frame_info['raw_path']}")
            continue

        # Apply flatfield
        if flatfield is not None:
            corrected = apply_flatfield(raw, flatfield)
        else:
            corrected = raw

        # Run detector
        detections = segment_frame(corrected, config, args.pixel_size).detections

        # Compute target centroid
        target_center = mask_centroid(frame_info["mask_path"])
        total_targets += 1

        # Match detections to target
        target_idx = None
        if target_center is not None:
            target_idx = match_detection_to_target(detections, target_center, args.match_radius)

        if target_idx is not None:
            total_matched += 1

        # Classify all detections
        classifications = []
        for det in detections:
            r, g, b = det["contrast_rgb"]
            cls_label, color = classify_detection(r, g, b, config)
            classifications.append((cls_label, color))
            key = cls_label or "unknown"
            cls_counts[key] = cls_counts.get(key, 0) + 1

        # Track target classification
        if target_idx is not None:
            target_cls = classifications[target_idx][0] or "unknown"
            target_cls_counts[target_cls] = target_cls_counts.get(target_cls, 0) + 1

        # Scatter points
        for i, (det, (cls_label, _)) in enumerate(zip(detections, classifications, strict=True)):
            r, g, _ = det["contrast_rgb"]
            all_scatter_points.append((r, g, cls_label, i == target_idx))

        # Draw annotated frame
        vis = draw_eval_frame(raw, detections, classifications, target_idx, args.pixel_size, config)

        # Save annotated frame to temp file
        ann_path = args.output / f"{frame_info['label'].replace('/', '_')}.jpg"
        cv2.imwrite(ann_path, vis)
        annotated_paths.append(str(ann_path))
        annotated_labels.append(frame_info["label"])

        # Per-frame summary
        match_str = f"MATCH #{target_idx}" if target_idx is not None else "NO MATCH"
        frame_line = f"{frame_info['label']}: {len(detections)} detections, {match_str}"
        if target_idx is not None:
            det = detections[target_idx]
            r, g, _ = det["contrast_rgb"]
            cls_label = classifications[target_idx][0]
            frame_line += f" [{cls_label}, R={r:.2f} G={g:.2f}]"
        summary_lines.append(frame_line)
        print(f"  {frame_line}")

    # Generate mosaic
    if annotated_paths:
        print(f"\nGenerating mosaic ({len(annotated_paths)} frames)...")
        mosaic_path = args.output / "filtered_mosaic.jpg"
        n = len(annotated_paths)
        rows = 4
        cols = max(1, (n + rows - 1) // rows)
        mosaic = make_mosaic(
            annotated_paths,
            rows=rows,
            cols=cols,
            max_dim=5000,
            margin=4,
            bg_color=(30, 30, 30),
            labels=annotated_labels,
            label_size=14,
            label_color=(255, 255, 255),
            label_bg=(0, 0, 0, 180),
        )
        mosaic.save(str(mosaic_path), quality=95)
        print(f"Saved mosaic: {mosaic_path}")

    # Generate R-G scatter (only meaningful for curve-based calibrations)
    if all_scatter_points and isinstance(config, CurveDetectorConfig):
        print("Generating R-G scatter plot...")
        scatter_path = args.output / "rg_scatter.png"
        generate_rg_scatter(all_scatter_points, scatter_path, config)
        print(f"Saved scatter: {scatter_path}")

    # Generate summary
    total_detections = sum(cls_counts.values())
    summary_path = args.output / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Flake Detection Evaluation ===\n\n")
        f.write(f"Reference: {args.ref_dir}\n")
        f.write(f"Flatfield: {args.flatfield or 'none'}\n")
        f.write(
            f"Parameters: contrast_offset={args.contrast_offset}, "
            f"min_size_um={args.min_size_um}, match_radius={args.match_radius}\n\n"
        )

        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Targets: {total_targets}\n")
        f.write(
            f"Target matches: {total_matched}/{total_targets} (recall={total_matched / max(total_targets, 1):.1%})\n\n"
        )

        f.write("--- Classification breakdown (all detections) ---\n")
        for cls_name in ["thin", "medium", "thick", "possible", config.non_match_label]:
            count = cls_counts.get(cls_name, 0)
            f.write(f"  {cls_name}: {count}\n")

        f.write("\n--- Target match classifications ---\n")
        for cls_name in ["thin", "medium", "thick", "possible", config.non_match_label]:
            count = target_cls_counts.get(cls_name, 0)
            f.write(f"  {cls_name}: {count}\n")

        # Recall at various classification thresholds
        f.write("\n--- Recall by classification threshold ---\n")
        cum = 0
        for cls_name in ["thin", "medium", "thick", "possible", config.non_match_label]:
            cum += target_cls_counts.get(cls_name, 0)
            f.write(f"  Including {cls_name}: {cum}/{total_targets} ({cum / max(total_targets, 1):.1%})\n")

        f.write("\n--- Per-frame results ---\n")
        for line in summary_lines:
            f.write(f"  {line}\n")

    print(f"Saved summary: {summary_path}")

    # Print summary to stdout too
    with open(summary_path) as f:
        print(f"\n{f.read()}")

    if args.show:
        for path in [
            args.output / "filtered_mosaic.jpg",
            args.output / "rg_scatter.png",
        ]:
            if path.exists():
                subprocess.run(["xdg-open", str(path)], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
