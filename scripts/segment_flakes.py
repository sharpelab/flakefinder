"""Segment flakes using flatfield correction + contrast threshold.

Usage:
    python segment_flakes.py frame.jpg --flatfield flatfield.npy
    python segment_flakes.py frame.jpg --flatfield flatfield.npy --material graphene
    python segment_flakes.py frame.jpg --flatfield flatfield.npy --save-plot
    python segment_flakes.py frame.jpg --flatfield flatfield.npy -o output.jpg
"""

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from flakefinder.scan_utils import apply_flatfield
from flakefinder.segmentation import (
    DetectorConfig,
    compute_dark_frac,
    draw_detections,
    histogram_mode,
    save_plot,
    segment_frame,
)


def main():
    parser = argparse.ArgumentParser(description="Segment flakes via flatfield + contrast threshold")
    parser.add_argument("input", type=Path, help="Frame image")
    parser.add_argument("--flatfield", type=Path, default=None, help="Flatfield .npy file")
    parser.add_argument(
        "--material",
        default="hbn",
        choices=DetectorConfig.material_names(),
        help="Material preset",
    )
    parser.add_argument("--contrast-offset", type=float, default=None, help="Override contrast offset from preset")
    parser.add_argument("--min-size-um", type=float, default=None, help="Override min detection area (µm²)")
    parser.add_argument("--edge-margin", type=int, default=None, help="Override edge margin (px)")
    parser.add_argument(
        "--dark-frac-cutoff",
        type=float,
        default=0.05,
        help="Skip frame if dark pixel fraction exceeds this (edge/off-chip filter)",
    )
    parser.add_argument(
        "--perim-ratio",
        type=float,
        default=0.0,
        help="Classify detections with perim_ratio >= this as tape (0 to disable)",
    )
    parser.add_argument(
        "--perim-ratio-kill",
        type=float,
        default=0.0,
        help="Remove detections with perim_ratio >= this entirely (0 to disable)",
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output image path")
    parser.add_argument("--save-plot", action="store_true", help="Also save matplotlib analysis plot")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.36,
        help="µm per pixel (default: 0.36 for 20x bin3)",
    )
    args = parser.parse_args()

    config = DetectorConfig.from_material(args.material)
    overrides = {}
    if args.contrast_offset is not None:
        overrides["contrast_offset"] = args.contrast_offset
    if args.min_size_um is not None:
        overrides["min_size_um2"] = args.min_size_um
    if args.edge_margin is not None:
        overrides["edge_margin_px"] = args.edge_margin
    if overrides:
        config = replace(config, **overrides)

    raw = cv2.imread(args.input)
    if raw is None:
        print(f"Failed to read {args.input}")
        return 1

    if args.flatfield:
        ff = np.load(str(args.flatfield)).astype(np.float32)
        ff = ff[:, :, ::-1]  # RGB (numpy save convention) -> BGR (cv2)
        corrected = apply_flatfield(raw, ff)
    else:
        corrected = raw

    # Edge/off-chip detection
    dark_frac = compute_dark_frac(corrected)
    skipped = dark_frac > args.dark_frac_cutoff

    if skipped:
        dets = []
        print(f"{args.input.name}: SKIPPED (dark_frac={dark_frac:.3f} > {args.dark_frac_cutoff})")
    else:
        dets = segment_frame(corrected, config, args.pixel_size, args.perim_ratio)
        if args.perim_ratio_kill > 0:
            before = len(dets)
            dets = [d for d in dets if d["perim_ratio"] < args.perim_ratio_kill]
            killed = before - len(dets)
        else:
            killed = 0
        print(f"{args.input.name}: {len(dets)} detections (dark_frac={dark_frac:.3f}, killed={killed})")
        for i, d in enumerate(dets):
            R, G = d["contrast_rgb"][0], d["contrast_rgb"][1]
            print(
                f"  #{i}: {d['size_px']}px  R={R:+.2f} G={G:+.2f}  "
                f"grad={d['grad_energy']:.1f}  entropy={d.get('entropy', d['g_entropy']):.2f}  "
                f"pr={d['perim_ratio']:.2f}  r_std={d['r_std']:.3f}"
            )

    # Output image — draw on raw to preserve true colors
    out = args.output or args.input.with_suffix(".seg.jpg")
    vis = draw_detections(raw, dets, um_per_px=args.pixel_size)
    cv2.imwrite(out, vis)
    print(f"Saved: {out}")

    # Output metadata
    bg_modes = [histogram_mode(corrected, c) for c in range(3)]
    meta = {
        "timestamp": datetime.now().isoformat(),
        "command": ["segment_flakes.py"] + sys.argv[1:],
        "input": str(args.input),
        "output_image": str(out),
        "frame_shape": list(raw.shape),
        "flatfield": str(args.flatfield) if args.flatfield else None,
        "params": {
            "material": args.material,
            "contrast_offset": config.contrast_offset,
            "min_size_um2": config.min_size_um2,
            "min_size_px": int(config.min_size_um2 / (args.pixel_size**2)),
            "edge_margin_px": config.edge_margin_px,
            "dark_frac_cutoff": args.dark_frac_cutoff,
            "perim_ratio_thresh": args.perim_ratio,
        },
        "dark_frac": round(dark_frac, 4),
        "skipped": skipped,
        "bg_modes_bgr": [round(m, 1) for m in bg_modes],
        "detection_count": len(dets),
        "detections": dets,
    }

    meta_path = Path(str(out).rsplit(".", 1)[0] + ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    if args.save_plot:
        plot_path = args.input.with_suffix(".seg.plot.png")
        save_plot(corrected, dets, args.input.name, plot_path, config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
