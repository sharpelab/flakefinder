"""Autofocus utility functions for saving debug frames and sharpness data."""

import json
import os
from pathlib import Path

from PIL import Image as PILImage

from .leica.autofocus import ALL_SHARPNESS_METRICS, AutofocusFrame, AutofocusResult, SharpnessSample

METRIC_NAMES = list(ALL_SHARPNESS_METRICS.keys())


def save_pass_frames(
    frames: list[AutofocusFrame],
    curve: list[SharpnessSample],
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


def save_debug_frames(
    af: AutofocusResult,
    out_dir: str | Path,
    verbose: bool = False,
) -> None:
    """Save all debug frames from an autofocus result.

    Handles coarse/fine/super_fine subdirectory layout, plus
    initial.png, final.png, and summary.json.

    Args:
        af: AutofocusResult with store_frames=True data.
        out_dir: Base directory to save into.
        verbose: If True, print progress messages.
    """
    if not af.frames:
        return

    out_dir = Path(out_dir)
    has_fine = af.fine_frames is not None and len(af.fine_frames) > 0
    has_super_fine = af.super_fine_frames is not None and len(af.super_fine_frames) > 0

    if has_fine or has_super_fine:
        coarse_dir = out_dir / "coarse"
        if verbose:
            print(f"  Saving {len(af.frames)} coarse frames to {coarse_dir}/...")
        save_pass_frames(af.frames, af.sharpness_curve, coarse_dir)
        if af.fine_frames is not None and len(af.fine_frames) > 0:
            fine_dir = out_dir / "fine"
            if verbose:
                print(f"  Saving {len(af.fine_frames)} fine frames to {fine_dir}/...")
            save_pass_frames(af.fine_frames, af.fine_sharpness_curve, fine_dir)
        if af.super_fine_frames is not None and len(af.super_fine_frames) > 0:
            sf_dir = out_dir / "super_fine"
            if verbose:
                print(f"  Saving {len(af.super_fine_frames)} super fine frames to {sf_dir}/...")
            save_pass_frames(af.super_fine_frames, af.super_fine_sharpness_curve, sf_dir)
    else:
        if verbose:
            print(f"  Saving {len(af.frames)} frames to {out_dir}/...")
        save_pass_frames(af.frames, af.sharpness_curve, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    if af.initial_image is not None:
        PILImage.fromarray(af.initial_image).save(str(out_dir / "initial.png"))
    if af.final_image is not None:
        PILImage.fromarray(af.final_image).save(str(out_dir / "final.png"))

    with open(str(out_dir / "summary.json"), "w") as f:
        json.dump(af.to_dict(), f, indent=2)
