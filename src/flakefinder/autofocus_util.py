"""Autofocus utility functions for saving debug frames and sharpness data."""

import os
from pathlib import Path

from PIL import Image as PILImage

from .leica.autofocus import AutofocusFrame, ALL_SHARPNESS_METRICS

METRIC_NAMES = list(ALL_SHARPNESS_METRICS.keys())


def save_pass_frames(
    frames: list[AutofocusFrame],
    curve: list[dict],
    out_dir: str | Path,
) -> None:
    """Save autofocus frames as PNGs and sharpness curve as CSV.

    Args:
        frames: List of AutofocusFrame with images.
        curve: Sharpness curve dicts from AutofocusResult.
        out_dir: Directory to save into (created if needed).
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        if frame.image is not None:
            fname = f"frame_{i:03d}_z_{frame.z_um:.1f}_s_{frame.sharpness:.1f}.png"
            PILImage.fromarray(frame.image).save(os.path.join(out_dir, fname))

    has_metrics = curve and "metrics" in curve[0]

    csv_path = os.path.join(out_dir, "sharpness_curve.csv")
    with open(csv_path, "w") as f:
        header = "frame,z_um,sharpness"
        if has_metrics:
            header += "," + ",".join(METRIC_NAMES)
        f.write(header + "\n")
        for r in curve:
            line = f"{r['frame']},{r['z_um']:.2f},{r['sharpness']:.2f}"
            if has_metrics and "metrics" in r:
                line += "," + ",".join(f"{r['metrics'][m]:.4f}" for m in METRIC_NAMES)
            f.write(line + "\n")
