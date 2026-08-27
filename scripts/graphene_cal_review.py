"""Audit an RG point-calibration set against the detections it is used to gate.

``cal_dist`` for an ``RGPointDetectorConfig`` is the Euclidean distance in
(R, G) contrast space to the *nearest reference vertex*.  That conflates two
independent quantities:

* **perpendicular** offset — how far the detection sits off the material's
  contrast locus.  This is the physically meaningful "is this the material?"
  signal the tier gates are trying to read.
* **along-curve** offset — how far along the locus the detection sits from the
  nearest integer-layer vertex.  This is pure lattice quantization: a flake
  lying exactly on the locus, halfway between two vertices, still books a
  distance of half the vertex spacing.

When the vertex spacing exceeds twice ``tier1_cal_dist`` the second term alone
can veto a perfectly on-locus flake, and the gate stops measuring what it was
meant to measure.  This script separates the two terms over real seg summaries,
and re-tiers offline against a densified point set to show what the lattice is
costing.

``cal_dist`` is a pure function of ``contrast_rgb`` and the reference points,
so every number here is recomputed from stored summaries — no re-segmentation
and no microscope time.

Usage:
    uv run python scripts/graphene_cal_review.py <run_dir>... -m graphene_thin_90nm
    uv run python scripts/graphene_cal_review.py <run_dir> -m graphene_thin_90nm --subdiv 4
    uv run python scripts/graphene_cal_review.py <run_dir> -m graphene_thin_90nm \
        --near 64990,34694 --radius 100
    uv run python scripts/graphene_cal_review.py <run_dir> -m graphene_thin_90nm --plot out.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np

from flakefinder.segmentation import CalPointRG, Detection, DetectorConfig, RGPointDetectorConfig


def densify(points: tuple[CalPointRG, ...], subdiv: int) -> tuple[CalPointRG, ...]:
    """Subdivide each segment of the point polyline into ``subdiv`` pieces.

    The locus is unchanged — only its sampling density.  Interpolated vertices
    inherit the layer count of the nearest integer-layer endpoint, so
    classification stays integer-valued.  Half-integer positions round up
    (``floor(x + 0.5)``, not :func:`round`, whose banker's rounding would label
    2.5L as 2 but 3.5L as 4).
    """
    if subdiv <= 1:
        return points
    out: list[CalPointRG] = []
    for a, b in zip(points, points[1:], strict=False):
        for i in range(subdiv):
            t = i / subdiv
            out.append(
                CalPointRG(
                    layers=math.floor(a.layers + t * (b.layers - a.layers) + 0.5),
                    r=a.r + t * (b.r - a.r),
                    g=a.g + t * (b.g - a.g),
                )
            )
    out.append(points[-1])
    return tuple(out)


def decompose(points: tuple[CalPointRG, ...], r: float, g: float) -> tuple[float, float, float]:
    """Split distance-to-nearest-vertex into (perpendicular, along-curve, vertex).

    ``perpendicular`` is the distance to the polyline through the points —
    the locus-offset term.  ``along`` is the residual quantization term such
    that perp² + along² == vertex², where ``vertex`` is ``cal_dist`` itself.
    Points outside the polyline's span project to an endpoint, so ``along``
    goes to zero and ``vertex`` is honest extrapolation distance.
    """
    p = np.array([r, g])
    vertex = min(math.hypot(pt.r - r, pt.g - g) for pt in points)
    perp = np.inf
    for a, b in zip(points, points[1:], strict=False):
        av = np.array([a.r, a.g])
        bv = np.array([b.r, b.g])
        ab = bv - av
        t = float(np.clip(np.dot(p - av, ab) / np.dot(ab, ab), 0.0, 1.0))
        perp = min(perp, float(np.linalg.norm(p - (av + t * ab))))
    along = math.sqrt(max(vertex**2 - perp**2, 0.0))
    return perp, along, vertex


def load_detections(run_dirs: list[Path]) -> list[dict]:
    """Collect every detection from every chip seg summary under the run dirs."""
    dets: list[dict] = []
    for run in run_dirs:
        summaries = sorted(run.glob("chip_*/seg/summary.json")) or sorted(run.glob("**/seg/summary.json"))
        for s in summaries:
            with open(s) as f:
                data = json.load(f)
            chip = s.parent.parent.name
            for frame in data["detections_by_frame"].values():
                for d in frame:
                    d["chip"] = chip
                    d["run"] = run.name
                    dets.append(d)
    return dets


def spacing_report(points: tuple[CalPointRG, ...], t1: float, t2: float) -> None:
    """Print vertex spacing against the tier thresholds it has to clear."""
    print("Lattice geometry — worst-case cal_dist for a flake exactly ON the locus:")
    print(f"{'gap':>10} {'spacing':>9} {'half':>8}   tier1<{t1:<5} tier2<{t2}")
    worst = 0.0
    for a, b in zip(points, points[1:], strict=False):
        d = math.hypot(a.r - b.r, a.g - b.g)
        worst = max(worst, d / 2)
        print(
            f"{f'{a.layers}L-{b.layers}L':>10} {d:9.4f} {d / 2:8.4f}   "
            f"{'PASS' if d / 2 < t1 else 'FAIL':<11} {'PASS' if d / 2 < t2 else 'FAIL'}"
        )
    print(f"\n  worst-case on-locus cal_dist = {worst:.4f}  (tier1 budget {t1})")
    if worst >= t1:
        print("  -> an on-locus flake between vertices CANNOT reach tier 1 on cal_dist alone.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path, help="Run directories containing chip_*/seg/summary.json")
    ap.add_argument("-m", "--material", required=True, choices=DetectorConfig.material_names())
    ap.add_argument("--subdiv", type=int, default=4, help="Densification factor to evaluate (default 4)")
    ap.add_argument("--min-size", type=float, default=350.0, help="Only detections at least this many µm²")
    ap.add_argument("--near", default=None, metavar="X,Y", help="Stage position in µm to search around")
    ap.add_argument("--radius", type=float, default=2000.0, help="Radius in µm for --near (default 2000)")
    ap.add_argument(
        "--band",
        type=float,
        default=0.15,
        help="Locus branch selection: keep |R-G| < band, both channels dark (default 0.15)",
    )
    ap.add_argument("--plot", type=Path, default=None, help="Write comparison plot to this path")
    args = ap.parse_args()

    cfg = DetectorConfig.from_material(args.material)
    if not isinstance(cfg, RGPointDetectorConfig):
        print(f"{args.material} is not an RG point preset (got {type(cfg).__name__})")
        return 1
    points = cfg.cal_reference_points

    spacing_report(points, cfg.tier1_cal_dist, cfg.tier2_cal_dist)

    dets = load_detections(args.run_dirs)
    if args.near:
        cx, cy = (float(v) for v in args.near.split(","))
        dets = [d for d in dets if math.hypot(d["stage_x"] - cx, d["stage_y"] - cy) <= args.radius]
    dets = [d for d in dets if d["size_um2"] >= args.min_size]
    if not dets:
        print("no detections match the filters")
        return 1
    print(f"{len(dets)} detections loaded (>= {args.min_size} µm²) from {len(args.run_dirs)} run(s)\n")

    dense = densify(points, args.subdiv)
    cfg_dense = replace(cfg, cal_reference_points=dense)

    rows = []
    for d in dets:
        r, g, _ = d["contrast_rgb"]
        perp, along, vertex = decompose(points, r, g)
        dense_dist = cfg_dense.cal_projection(r, g, 0.0).dist
        t_now = d["tier"]
        t_new = cfg_dense.score_detection(cast(Detection, {**d, "cal_dist": dense_dist}))[0]
        rows.append((d, r, g, perp, along, vertex, dense_dist, t_now, t_new))

    # Locus branch: both channels darker than background, near the R=G ridge.
    branch = [x for x in rows if x[1] < -0.05 and x[2] < -0.05 and abs(x[1] - x[2]) < args.band]
    lo = min(p.r for p in points)
    hi = max(p.r for p in points)
    inspan = [x for x in branch if lo <= x[1] <= hi]

    def stats(vals: list[float]) -> str:
        a = np.array(vals)
        return f"median {np.median(a):.4f}  p75 {np.percentile(a, 75):.4f}  p90 {np.percentile(a, 90):.4f}"

    print(f"Locus branch (|R-G| < {args.band}, both channels dark), in cal-point span: n={len(inspan)}")
    if inspan:
        print(f"  perpendicular (locus offset) : {stats([x[3] for x in inspan])}")
        print(f"  along-curve  (quantization)  : {stats([x[4] for x in inspan])}")
        print(f"  cal_dist     (as shipped)    : {stats([x[5] for x in inspan])}")
        print(f"  cal_dist     (subdiv {args.subdiv})      : {stats([x[6] for x in inspan])}")
        t1 = cfg.tier1_cal_dist
        print(f"\n  fraction under tier1_cal_dist ({t1}):")
        print(f"    perpendicular only : {100 * np.mean([x[3] < t1 for x in inspan]):5.1f}%")
        print(f"    as shipped         : {100 * np.mean([x[5] < t1 for x in inspan]):5.1f}%")
        print(f"    subdiv {args.subdiv}           : {100 * np.mean([x[6] < t1 for x in inspan]):5.1f}%")

    promoted = [x for x in rows if x[7] != 1 and x[8] == 1]
    demoted = [x for x in rows if x[7] == 1 and x[8] != 1]
    print(f"\nOffline re-tier with subdiv {args.subdiv} (locus unchanged, sampling denser):")
    print(f"  tier 1 now      : {sum(1 for x in rows if x[7] == 1)}")
    print(f"  tier 1 densified: {sum(1 for x in rows if x[8] == 1)}")
    print(f"  promoted to t1  : {len(promoted)}")
    print(f"  demoted from t1 : {len(demoted)}")

    if promoted:
        promoted.sort(key=lambda x: -x[0]["size_um2"])
        print(f"\n  largest promotions:\n{'  size_um2':>11} {'chip':>7} {'frame':>11} {'cal→dense':>16} {'L':>3}")
        for d, _r, _g, _perp, _al, vx, dd, _tn, _tw in promoted[:15]:
            print(f"{d['size_um2']:11.1f} {d['chip']:>7} {d['frame']:>11} {vx:8.4f}→{dd:7.4f} {str(d['layers']):>3}")

    if args.plot:
        _plot(args.plot, points, dense, rows, inspan, cfg)
        print(f"\nwrote {args.plot}")
    return 0


def _plot(path: Path, points, dense, rows, inspan, cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    t1 = cfg.tier1_cal_dist

    ax = axes[0]
    allr = [x[1] for x in rows]
    allg = [x[2] for x in rows]
    ax.scatter(allr, allg, s=4, c="0.82", label="all detections", zorder=1)
    ax.scatter([x[1] for x in inspan], [x[2] for x in inspan], s=10, c="tab:blue", label="locus branch", zorder=2)
    pr = [p.r for p in points]
    pg = [p.g for p in points]
    ax.plot(pr, pg, "-", c="tab:red", lw=1.6, zorder=3, label="locus (polyline)")
    ax.scatter(pr, pg, s=90, c="tab:red", marker="o", zorder=4, label="cal points (shipped)")
    ax.scatter([p.r for p in dense], [p.g for p in dense], s=8, c="tab:orange", zorder=3, label="densified")
    for p in points:
        ax.add_patch(plt.Circle((p.r, p.g), t1, fill=False, ec="tab:red", ls=":", lw=1.0, zorder=3))
        ax.annotate(f"{p.layers}L", (p.r, p.g), textcoords="offset points", xytext=(7, 6), fontsize=9, color="tab:red")
    ax.set_xlim(-0.75, 0.05)
    ax.set_ylim(-0.75, 0.05)
    ax.set_xlabel("R contrast")
    ax.set_ylabel("G contrast")
    ax.set_title(f"(R,G) space — dotted circles are the tier-1 cal_dist gate ({t1})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter([x[3] for x in inspan], [x[4] for x in inspan], s=14, c="tab:blue")
    ax.axvline(t1, c="tab:red", ls="--", lw=1.2, label=f"tier1_cal_dist {t1}")
    th = np.linspace(0, np.pi / 2, 100)
    ax.plot(t1 * np.cos(th), t1 * np.sin(th), c="tab:red", lw=1.6, label="cal_dist = tier1 (shipped)")
    ax.set_xlabel("perpendicular — locus offset (real signal)")
    ax.set_ylabel("along-curve — lattice quantization (artifact)")
    ax.set_title("what cal_dist is actually made of")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0, 0.12)

    ax = axes[2]
    bins = np.linspace(0, 0.2, 45)
    ax.hist([x[5] for x in inspan], bins=bins, alpha=0.6, label="cal_dist as shipped", color="tab:red")
    ax.hist([x[6] for x in inspan], bins=bins, alpha=0.6, label="cal_dist densified", color="tab:green")
    ax.hist([x[3] for x in inspan], bins=bins, histtype="step", lw=1.8, label="perpendicular only", color="tab:blue")
    ax.axvline(t1, c="k", ls="--", lw=1.2, label=f"tier1 gate {t1}")
    ax.set_xlabel("distance")
    ax.set_ylabel("detections")
    ax.set_title("locus-branch distance distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
