"""Continuous Z-scan autofocus demo.

Performs autofocus by scanning Z downward while capturing frames,
computing sharpness for each frame, and moving to the Z position
with maximum sharpness.

This demo script provides CLI interface, before/after photos, and debug
output. The core autofocus logic is in flakefinder.leica.autofocus.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from PIL import Image as PILImage

from flakefinder.autofocus_util import save_debug_frames
from flakefinder.leica import Microscope, wait_all
from flakefinder.leica.autofocus import (
    ALL_SHARPNESS_METRICS,
    continuous_autofocus,
    sharpness,
)
from flakefinder.scan_utils import build_microscope_meta, get_git_version, parse_white_balance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous Z-scan autofocus demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--objective-mag",
        type=str,
        default=None,
        help="Objective magnification (e.g. 2.5, 5x, 20)",
    )

    parser.add_argument("--x", type=float, default=None, help="Stage X position (um), default: current")
    parser.add_argument("--y", type=float, default=None, help="Stage Y position (um), default: current")
    parser.add_argument("--z", type=float, default=None, help="Initial Z position (um), default: current")
    parser.add_argument("--range", type=float, default=None, help="Z scan range (um), default: auto from objective")
    parser.add_argument(
        "--z-speed",
        type=float,
        default=None,
        help="Z axis speed (um/s), default: use current. Slower = more frames.",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory for photos (optional)")
    parser.add_argument("--debug-dir", type=str, default=None, help="Save all scan frames to this directory")
    parser.add_argument("--fine", action="store_true", help="Two-pass: coarse scan then fine 50um scan")
    parser.add_argument(
        "--fine-speed-factor",
        type=float,
        default=0.25,
        help="Speed multiplier for fine pass (default: 0.25 = 1/4 speed)",
    )
    parser.add_argument(
        "--super-fine",
        action="store_true",
        help="Three-pass (implies --fine): coarse + fine + 10um at 20um/s",
    )
    parser.add_argument(
        "--sharpness-method",
        choices=list(ALL_SHARPNESS_METRICS.keys()),
        default="tenengrad",
        help="Sharpness metric for autofocus",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.2,
        help="Settle time in seconds after final Z move (default: 0.2)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without moving")
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Compute all 5 sharpness metrics per frame (diagnostic)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output/debug directories before running",
    )
    parser.add_argument("--exposure-ms", type=float, default=1.0, help="Exposure time in ms")
    parser.add_argument("--gain", type=float, default=None, help="Camera gain (default: unchanged)")
    parser.add_argument(
        "--white-balance",
        type=parse_white_balance,
        default=None,
        help="White balance as B,G,R gains (e.g., 2.51,1.02,1.41)",
    )
    parser.add_argument("--gamma", type=float, default=None, help="Gamma correction (default: unchanged)")
    parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=0.20,
        help="Unused, kept for CLI compatibility",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Single-line output: 'AF: Z=... (adj ...), sharpness=sel/after, DR=..., Ns, N frames'",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    def vprint(*a, **kw):
        if not args.quiet:
            print(*a, **kw)

    vprint("Continuous Autofocus Demo")
    vprint("=" * 50)

    # Handle dry-run mode (no hardware connection)
    if args.dry_run:
        if args.x is None or args.y is None or args.z is None:
            print("Error: --dry-run requires --x, --y, --z to be specified")
            print("       (cannot read current position without hardware)")
            return 1

        scan_range = args.range if args.range is not None else 500
        z_start = args.z + scan_range / 2
        z_end = args.z - scan_range / 2

        print(f"Target position: X={args.x:.1f} um, Y={args.y:.1f} um")
        print(f"Initial Z: {args.z:.1f} um")
        print(f"Scan range: {scan_range:.1f} um" + (" (auto)" if args.range is None else ""))
        print(f"Scan: Z={z_start:.1f} -> {z_end:.1f} um (downward)")
        if args.output:
            print(f"Output: {args.output}/")
        print()
        print("[DRY RUN] Would perform:")
        print(f"  1. Move stage to X={args.x:.1f}, Y={args.y:.1f}")
        print(f"  2. Move Z to {args.z:.1f} um")
        print("  3. Capture 'before' photo")
        print(f"  4. Scan Z from {z_start:.1f} to {z_end:.1f} um")
        print("     - Capture frames continuously")
        print("     - Poll Z position at ~64 Hz")
        print("     - Compute Sobel sharpness for each frame")
        print("  5. Find Z with maximum sharpness")
        print("  6. Move to best Z")
        print("  7. Capture 'after' photo")
        if args.output:
            print(f"  8. Save summary to {args.output}/")
        return 0

    # Validate/clean output directories BEFORE connecting to hardware
    for d in [args.output, args.debug_dir]:
        if not d:
            continue
        if args.clean and os.path.exists(d):
            shutil.rmtree(d)
            print(f"Removed existing: {d}")
        elif os.path.exists(d):
            print(f"Error: Directory '{d}' already exists (use --clean to remove)")
            return 1

    with Microscope() as scope:
        # Switch objective if requested
        if args.objective_mag:
            scope.switch_objective_mag(args.objective_mag.lower().rstrip("x"))

        stage = scope.stage

        # Read current positions
        current_x, current_y = stage.position_um
        current_z = scope.z.position_um

        vprint(f"Current position: X={current_x:.1f} um, Y={current_y:.1f} um, Z={current_z:.1f} um")
        vprint()

        # Use current positions as defaults
        target_x = args.x if args.x is not None else current_x
        target_y = args.y if args.y is not None else current_y
        target_z = args.z if args.z is not None else current_z

        vprint(f"Target position: X={target_x:.1f} um, Y={target_y:.1f} um")
        vprint(f"Initial Z: {target_z:.1f} um")
        vprint(f"Scan range: {args.range:.1f} um" if args.range else "Scan range: auto (from objective)")
        if args.output:
            vprint(f"Output: {args.output}/")
        vprint()

        # Create output directory if specified
        if args.output:
            os.makedirs(args.output)

        # Lighting and camera
        scope.light_on()
        camera = scope.camera

        # Configure camera for fast capture
        camera.trigger_mode = 0  # CONTINUOUS
        camera.binning = 2  # 3x3 binning for speed
        camera.exposure_time = args.exposure_ms / 1000.0
        if args.gain is not None:
            camera.gain = args.gain
        if args.white_balance is not None:
            camera.gain_rgb = args.white_balance
        if args.gamma is not None:
            camera.gamma = args.gamma

        exp_ms = camera.exposure_time * 1000 if camera.exposure_time else 1.0
        vprint(f"Camera: {camera.name}")
        vprint(f"  Binning: 3x3, Exposure: {exp_ms:.2f}ms")
        vprint(f"Lamp: {scope.lamp.intensity_pct:.0f}% ({scope.lamp.intensity}/{scope.lamp.max_intensity})")
        vprint()

        # === Step 1: Move to XY position ===
        vprint(f"Moving to X={target_x:.1f}, Y={target_y:.1f}...")
        hx, hy = stage.move_to_async(target_x, target_y)
        wait_all([hx, hy])

        # === Step 2: Move to initial Z ===
        # Set Z speed for positioning (restores sane default if a previous crash left it slow)
        scope.z.set_velocity_um_s(args.z_speed if args.z_speed else scope.z.max_velocity_um_s)
        vprint(f"Moving Z to {target_z:.1f} um...")
        scope.z.move_to_corrected(target_z)

        # === Step 3: Capture 'before' photo ===
        vprint("Capturing 'before' photo...")
        before_img = camera.capture()
        if before_img is not None:
            before_sharpness = sharpness(before_img)
            if args.output:
                before_path = os.path.join(args.output, "before.jpg")
                PILImage.fromarray(before_img).save(before_path, quality=95)
                vprint(f"  Saved: {before_path}")
            vprint(f"  Sharpness: {before_sharpness:.2f}")
        else:
            vprint("  Warning: Failed to capture before image")
            before_sharpness = 0.0

        # === Step 4: Run autofocus using library ===
        vprint()
        vprint("Starting autofocus scan...")

        try:
            af_result = continuous_autofocus(
                scope,
                z_range_um=args.range,
                z_start_um=target_z,
                z_speed_um_s=args.z_speed,
                fine_pass=args.fine,
                fine_speed_factor=args.fine_speed_factor,
                super_fine_pass=args.super_fine,
                sharpness_method=args.sharpness_method,
                store_frames=bool(args.debug_dir or args.output),
                compute_all_metrics=args.all_metrics,
                settle_time_s=args.settle_time,
                min_dynamic_range=args.min_dynamic_range,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        vprint(f"  Scan completed in {af_result.scan_duration_s:.2f}s")
        vprint(f"  Frames captured: {af_result.frame_count}")
        vprint(f"  Z samples: {af_result.z_sample_count}")
        vprint()

        # Sharpness stats (combine coarse + fine + super_fine)
        all_sharpness = (
            af_result.sharpness_curve + af_result.fine_sharpness_curve + af_result.super_fine_sharpness_curve
        )
        sharpness_values = [r["sharpness"] for r in all_sharpness]
        min_sharpness = min(sharpness_values) if sharpness_values else 0
        max_sharpness = max(sharpness_values) if sharpness_values else 0
        mean_sharpness = sum(sharpness_values) / len(sharpness_values) if sharpness_values else 0

        vprint(f"  Sharpness range: {min_sharpness:.2f} - {max_sharpness:.2f}")
        vprint(f"  Mean sharpness: {mean_sharpness:.2f}")
        vprint(f"  Selected Z: {af_result.selected_z_um:.2f} um")
        vprint(f"  Selected sharpness: {af_result.selected_sharpness:.2f}")
        vprint(f"  Dynamic range: {af_result.dynamic_range:.4f}")
        if af_result.peak_near_edge:
            vprint("  ** Peak near edge of coarse scan window **")

        # Find best frame for saving — check coarse and fine separately
        best_scan_path = None
        best_frame = None
        if af_result.frames and af_result.sharpness_curve:
            coarse_best = max(af_result.sharpness_curve, key=lambda r: r["sharpness"])
            best_frame = af_result.frames[coarse_best["frame"]]
        if af_result.fine_frames and af_result.fine_sharpness_curve:
            fine_best = max(af_result.fine_sharpness_curve, key=lambda r: r["sharpness"])
            fine_frame = af_result.fine_frames[fine_best["frame"]]
            if best_frame is None or fine_frame.sharpness > best_frame.sharpness:
                best_frame = fine_frame
        if af_result.super_fine_frames and af_result.super_fine_sharpness_curve:
            sf_best = max(af_result.super_fine_sharpness_curve, key=lambda r: r["sharpness"])
            sf_frame = af_result.super_fine_frames[sf_best["frame"]]
            if best_frame is None or sf_frame.sharpness > best_frame.sharpness:
                best_frame = sf_frame

        # === Step 5: Capture 'after' photo ===
        # Z is already at best position (library moved it there)
        # Capture immediately before any debug I/O to minimize delay after AF
        vprint()
        vprint("Capturing 'after' photo...")
        after_img = camera.capture()

        # Save best frame from scan (if we have frames stored and output specified)
        if args.output and best_frame is not None and best_frame.image is not None:
            best_scan_path = os.path.join(args.output, "best.jpg")
            PILImage.fromarray(best_frame.image).save(best_scan_path, quality=95)
            vprint(f"  Saved best scan frame: {best_scan_path}")

        # Save debug frames if requested
        if args.debug_dir and af_result.frames:
            save_debug_frames(af_result, args.debug_dir, verbose=True)
        after_path = None
        if after_img is not None:
            after_sharpness = sharpness(after_img)
            if args.output:
                after_path = os.path.join(args.output, "after.jpg")
            else:
                after_path = "autofocus_after.jpg"
            PILImage.fromarray(after_img).save(after_path, quality=95)
            vprint(f"  Saved: {after_path}")
            vprint(f"  Sharpness: {after_sharpness:.2f}")

        else:
            vprint("  Warning: Failed to capture after image")
            after_sharpness = af_result.final_sharpness

        # Save before/after + initial/final to debug dir
        if args.debug_dir:
            for name, img in [
                ("before", before_img),
                ("initial", af_result.initial_image),
                ("final", af_result.final_image),
                ("after", after_img),
            ]:
                if img is not None:
                    path = os.path.join(args.debug_dir, f"{name}.png")
                    PILImage.fromarray(img).save(path)
                    vprint(f"  Saved {path}")

        # === Step 6: Save summary (if output specified) ===
        summary_path = None
        if args.output:
            summary = af_result.to_dict()
            summary["git_version"] = get_git_version()
            summary["timestamp"] = datetime.now().isoformat()
            summary["command"] = sys.argv
            summary["microscope"] = dict(build_microscope_meta(scope))
            summary["params"] = {
                "x_um": target_x,
                "y_um": target_y,
                "z_initial_um": target_z,
                "range_um": af_result.z_range_um,
                "fine_pass": args.fine,
                "super_fine_pass": args.super_fine,
                "objective_position": af_result.objective_position,
            }
            summary["before"] = {
                "z_um": target_z,
                "sharpness": before_sharpness,
            }
            summary["after"] = {
                "z_um": af_result.selected_z_um,
                "sharpness": after_sharpness,
            }
            summary["sharpness_stats"] = {
                "min": min_sharpness,
                "max": max_sharpness,
                "mean": mean_sharpness,
            }
            if best_frame is not None:
                summary["best_frame"] = {
                    "z_um": best_frame.z_um,
                    "sharpness": best_frame.sharpness,
                }

            summary_path = os.path.join(args.output, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            vprint()
            vprint(f"Saved summary: {summary_path}")

        # === Print summary ===
        if args.quiet:
            # Single-line output for programmatic use
            z_adj = af_result.selected_z_um - target_z
            edge_str = " EDGE" if af_result.peak_near_edge else ""
            print(
                f"AF: Z={af_result.selected_z_um:.1f} (adj {z_adj:+.1f}), "
                f"sharpness={af_result.selected_sharpness:.1f}/{after_sharpness:.1f}, "
                f"DR={af_result.dynamic_range:.3f}{edge_str}, "
                f"{af_result.scan_duration_s:.1f}s, {af_result.frame_count} frames"
            )
        else:
            vprint()
            vprint("=" * 50)
            vprint("AUTOFOCUS SUMMARY")
            vprint("=" * 50)
            vprint(f"Initial Z:    {target_z:.2f} um (sharpness: {before_sharpness:.2f})")
            vprint(f"Selected Z:   {af_result.selected_z_um:.2f} um (sharpness: {af_result.selected_sharpness:.2f})")
            vprint(f"After Z:      {af_result.selected_z_um:.2f} um (sharpness: {after_sharpness:.2f})")
            vprint(f"Z adjustment: {af_result.selected_z_um - target_z:+.2f} um")
            if before_sharpness > 0:
                improvement = after_sharpness - before_sharpness
                pct = (after_sharpness / before_sharpness - 1) * 100
                vprint(f"Sharpness improvement: {improvement:+.2f} ({pct:+.1f}%)")
            if args.output:
                vprint()
                vprint("Output files:")
                vprint(f"  {os.path.join(args.output, 'before.jpg')}")
                vprint(f"  {os.path.join(args.output, 'after.jpg')}")
                if best_scan_path:
                    vprint(f"  {best_scan_path}")
                vprint(f"  {summary_path}")
            vprint()
            vprint("Done.")

        return 0


if __name__ == "__main__":
    exit(main())
