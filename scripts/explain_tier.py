"""Explain why detections did not reach tier 1, using the live preset gates.

Replays the tier-1 / tier-2 predicates from the material's DetectorConfig over
the detections in a chip's seg summary, and reports which gate each detection
failed. Answers "why was this flake missed?" against the real thresholds
instead of guesswork.

Usage:
    uv run python scripts/explain_tier.py <summary.json> --material graphene_thin_90nm
    uv run python scripts/explain_tier.py <summary.json> -m graphene_thin_90nm --min-size 2000
    uv run python scripts/explain_tier.py <summary.json> -m graphene_thin_90nm \
        --near 66901,34631 --radius 3000
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from flakefinder.segmentation import DetectorConfig

# Each entry: gate name -> (predicate on the detection view, threshold text).
# Mirrors _score_graphene's t1_shape / t1_color conjunctions.
SHAPE_GATES = ("perim_ratio", "aspect_ratio", "solidity", "circularity", "size_um2")
COLOR_GATES = ("cal_dist", "r_max", "r_min", "g_max", "br_ratio", "entropy")


def gate_status(cfg: DetectorConfig, det: dict) -> dict[str, tuple[bool, float, str]]:
    """Return {gate: (passed, value, threshold)} for the tier-1 predicates."""
    r, g, b = det["contrast_rgb"]
    br_ratio = b / r if abs(r) > 0.01 else 99.0
    ent = det.get("entropy", det.get("g_entropy", 99.0))
    return {
        "perim_ratio": (det["perim_ratio"] < cfg.tier1_perim_ratio, det["perim_ratio"], f"< {cfg.tier1_perim_ratio}"),
        "aspect_ratio": (
            det.get("aspect_ratio", 1.0) < cfg.tier1_aspect_ratio,
            det.get("aspect_ratio", 1.0),
            f"< {cfg.tier1_aspect_ratio}",
        ),
        "solidity": (det["solidity"] >= cfg.tier1_solidity_min, det["solidity"], f">= {cfg.tier1_solidity_min}"),
        "circularity": (
            det["circularity"] >= cfg.tier1_circularity_min,
            det["circularity"],
            f">= {cfg.tier1_circularity_min}",
        ),
        "size_um2": (det["size_um2"] >= cfg.tier1_min_size_um2, det["size_um2"], f">= {cfg.tier1_min_size_um2}"),
        "cal_dist": (det["cal_dist"] < cfg.tier1_cal_dist, det["cal_dist"], f"< {cfg.tier1_cal_dist}"),
        "r_max": (r < cfg.tier1_r_max, r, f"< {cfg.tier1_r_max}"),
        "r_min": (r > cfg.tier1_r_min, r, f"> {cfg.tier1_r_min}"),
        "g_max": (g < cfg.tier1_g_max, g, f"< {cfg.tier1_g_max}"),
        "br_ratio": (br_ratio < cfg.tier1_br_ratio_max, br_ratio, f"< {cfg.tier1_br_ratio_max}"),
        "entropy": (ent < cfg.tier1_entropy_max, ent, f"< {cfg.tier1_entropy_max}"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", type=Path, help="Path to a chip's seg/summary.json")
    ap.add_argument("-m", "--material", required=True, choices=DetectorConfig.material_names())
    ap.add_argument("--min-size", type=float, default=0.0, help="Only detections at least this many µm²")
    ap.add_argument("--near", default=None, metavar="X,Y", help="Stage position in µm to search around")
    ap.add_argument("--radius", type=float, default=2000.0, help="Radius in µm for --near (default 2000)")
    ap.add_argument("--tier", type=int, default=None, help="Only detections at this tier")
    ap.add_argument("--top", type=int, default=25, help="How many detections to list (default 25)")
    args = ap.parse_args()

    cfg = DetectorConfig.from_material(args.material)
    with open(args.summary) as f:
        data = json.load(f)
    dets = [d for frame in data["detections_by_frame"].values() for d in frame]

    if args.near:
        cx, cy = (float(v) for v in args.near.split(","))
        dets = [d for d in dets if math.hypot(d["stage_x"] - cx, d["stage_y"] - cy) <= args.radius]
    dets = [d for d in dets if d["size_um2"] >= args.min_size]
    if args.tier is not None:
        dets = [d for d in dets if d["tier"] == args.tier]
    if not dets:
        print("no detections match the filters")
        return 1

    dets.sort(key=lambda d: -d["size_um2"])
    print(f"{len(dets)} detections match  (material {args.material})\n")

    binding: Counter[str] = Counter()
    for d in dets:
        status = gate_status(cfg, d)
        failed = [name for name, (ok, _, _) in status.items() if not ok]
        if d["tier"] != 1:
            binding.update(failed if failed else ["(passes tier-1 gates)"])

    print(f"{'size_um2':>9} {'tier':>4} {'layers':>6}  failed tier-1 gates")
    for d in dets[: args.top]:
        status = gate_status(cfg, d)
        failed = [f"{n}={status[n][1]:.3g} (need {status[n][2]})" for n, (ok, _, _) in status.items() if not ok]
        detail = "; ".join(failed) if failed else "— none —"
        print(f"{d['size_um2']:9.1f} {d['tier']:4} {str(d['layers']):>6}  {detail}")

    print("\nbinding gates across all non-tier-1 detections shown:")
    for name, n in binding.most_common():
        print(f"  {name:16} {n:6}  ({100 * n / max(1, sum(1 for d in dets if d['tier'] != 1)):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
