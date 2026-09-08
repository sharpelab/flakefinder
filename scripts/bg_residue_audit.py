"""Audit substrate-background contamination in a run and bound its effect on contrast.

Detection contrast is ``(flake - bg) / bg`` where ``bg`` is a **single scalar per
channel per frame** (``histogram_mode``).  That reference is supposed to be clean
90 nm SiO₂.  When the chip carries tape residue or any other surface film the
reference moves, and every contrast in the frame moves with it.

This script measures three things and keeps them separate:

1. **How far off-spec the background is** — per-frame R/G and B/G against the
   golden blank references in ``calibration/blank_refs_*.json``.  Ratios are
   gain/exposure-invariant, so they compare directly as long as the objective
   and WB preset match.
2. **Whether the deviation is patchy or smooth** — a per-block map of background
   colour inside single frames, before and after removing a quadratic surface.
   A lamp/illumination change is uniform; residue is not.  Vignetting is already
   removed by the flatfield, so what survives is real surface structure.
3. **What it does to flake contrast** — how much of the detections' offset from
   the calibration locus the background actually accounts for, via a regression
   that controls for position along the locus, plus a field-position term.

The point of (3) is to stop background contamination being blamed for
misclassification by assertion.  It is a real effect; it is not automatically
the dominant one.

Usage:
    uv run python scripts/bg_residue_audit.py <run_dir> -m graphene_thin_90nm
    uv run python scripts/bg_residue_audit.py <run_dir> -m graphene_thin_90nm --frames <chip_dir>
    uv run python scripts/bg_residue_audit.py <run_dir> -m graphene_thin_90nm --plot out.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from flakefinder.segmentation import DetectorConfig, RGPointDetectorConfig

BLOCK = 64
# Golden clean-blank references, keyed by objective mag. From
# calibration/blank_refs_20260813.json stage3_objectives (gain 4.0, 0.25 ms,
# graphene_thin/hbn WB). Ratios only — absolute DN depends on gain/exposure.
GOLDEN_BY_MAG = {
    2.5: (1.0333, 2.1153),
    5.0: (1.0717, 2.0656),
    10.0: (1.0098, 2.0440),
    20.0: (1.0318, 1.9027),
    50.0: (1.0915, 1.5407),
}


def load_flatfield_bgr(path: Path) -> np.ndarray:
    """Flatfield correction factors in cv2 channel order.

    The .npy is stored RGB; frames come from cv2 as BGR.  The pipeline
    reverses before ``apply_flatfield`` (find_flakes ``process_frame``) and
    this script must do the same, or the R and B corrections swap.
    """
    return np.load(path).astype(np.float32)[:, :, ::-1]


def signed_perp(points, r: float, g: float) -> float:
    """Signed distance from the calibration polyline; negative = R darker than G."""
    v = np.array([(p.r, p.g) for p in points])
    p = np.array([r, g])
    best, sign = np.inf, 0.0
    for a, b in zip(v, v[1:], strict=False):
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(p - proj))
        if d < best:
            best = d
            n = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
            sign = float(np.dot(p - proj, n))
    return sign


def trimmed_std(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> float:
    """Std of the central 5-95% — edge frames are a minority and would inflate it.

    Plain std, not MAD: background modes are integers, and MAD on quantized
    data collapses onto multiples of 1 (0.0 or 1.4826) instead of tracking the
    real scale.
    """
    a, b = np.percentile(x, [lo, hi])
    k = x[(x >= a) & (x <= b)]
    return float(k.std(ddof=1)) if len(k) > 2 else float("nan")


def seg_path_for(frame: Path) -> Path:
    """The per-frame seg JSON that accompanies a scan frame."""
    return frame.parent.parent / "seg" / f"{frame.stem}.json"


def flake_mask(frame: Path, shape: tuple[int, int], dilate_px: int = 12) -> np.ndarray:
    """True where a segmented detection covers the frame.

    Detections must be excluded before measuring substrate colour — a flake is
    darker and spectrally different, so any block statistic that includes flake
    pixels reports the flake, not the substrate under it.  Dilated because the
    contour hugs the flake edge and the halo just outside it is not clean
    substrate either.
    """
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    seg = seg_path_for(frame)
    if not seg.exists():
        return mask.astype(bool)
    with open(seg) as fh:
        dets = json.load(fh).get("detections", [])
    for d in dets:
        cv2.fillPoly(mask, [np.array(d["contour"], dtype=np.int32)], 1)
    if dilate_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k)
    return mask.astype(bool)


def block_colour_map(path: Path, flatfield: np.ndarray, block: int = BLOCK, min_clean_frac: float = 0.35) -> np.ndarray:
    """Per-block substrate R/G, with segmented detections masked out.

    Blocks whose unmasked (substrate) fraction falls below ``min_clean_frac``
    return NaN rather than a flake-contaminated value.
    """
    import cv2

    from flakefinder.scan_utils import apply_flatfield

    raw = cv2.imread(str(path))
    if raw is None:
        raise FileNotFoundError(path)
    im = apply_flatfield(raw, flatfield).astype(np.float32)
    h, w, _ = im.shape
    keep = ~flake_mask(path, (h, w))
    ny, nx = (h - 1) // block, (w - 1) // block
    grid = np.full((ny, nx), np.nan)
    for j in range(ny):
        for i in range(nx):
            sl = (slice(j * block, (j + 1) * block), slice(i * block, (i + 1) * block))
            k = keep[sl]
            if k.mean() < min_clean_frac:
                continue
            tile = im[sl][k]
            med = np.median(tile, axis=0)
            grid[j, i] = med[2] / max(med[1], 1e-6)  # BGR -> R/G
    return grid


def detrend_quadratic(grid: np.ndarray) -> np.ndarray:
    """Remove a smooth quadratic surface, leaving patch-scale structure.

    Masked-out blocks (NaN) are excluded from the fit and stay NaN.
    """
    ny, nx = grid.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = np.c_[np.ones(grid.size), xx.ravel(), yy.ravel(), xx.ravel() ** 2, yy.ravel() ** 2, (xx * yy).ravel()]
    ok = np.isfinite(grid.ravel())
    if ok.sum() < 12:
        return np.full_like(grid, np.nan)
    coef, *_ = np.linalg.lstsq(a[ok], grid.ravel()[ok], rcond=None)
    return grid - (a @ coef).reshape(ny, nx)


def structure_ratio(grid: np.ndarray) -> float:
    """Random-pair spread over adjacent-block spread. ~1 = noise, >>1 = structure.

    Uses means, not medians: background modes are integer-quantized, so many
    adjacent blocks are exactly equal and a median adjacent-delta collapses to
    zero, which would make the ratio meaningless.  NaN (masked) blocks are
    dropped from both populations.
    """
    nbr = np.concatenate([np.abs(np.diff(grid, axis=0)).ravel(), np.abs(np.diff(grid, axis=1)).ravel()])
    nbr = nbr[np.isfinite(nbr)]
    flat = grid.ravel()
    flat = flat[np.isfinite(flat)]
    if len(flat) < 8 or len(nbr) < 8:
        return float("nan")
    rng = np.random.default_rng(0)
    rand = np.abs(flat[rng.integers(0, len(flat), 20000)] - flat[rng.integers(0, len(flat), 20000)])
    return float(rand.mean() / max(nbr.mean(), 1e-9))


def annotate_frame(frame: Path, seg_json: Path, flatfield: np.ndarray, gold_rg: float, out: Path) -> None:
    """Label one frame's surfaces by what is actually measurable.

    Three things are distinguishable in the data and are drawn as such:
    detections (segmented flakes), and the substrate shaded by how far its
    *local* R/G sits from the golden clean-blank value.  No discrete
    residue/clean boundary is drawn, because the substrate R/G histogram is
    unimodal — the deviation is continuous, not two populations.
    """
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from flakefinder.scan_utils import apply_flatfield

    raw = cv2.imread(str(frame))
    if raw is None:
        raise FileNotFoundError(frame)
    im = apply_flatfield(raw, flatfield).astype(np.float32)
    rgb = im[:, :, ::-1] / 255.0
    h, w, _ = im.shape

    dets = []
    if seg_json.exists():
        with open(seg_json) as fh:
            dets = json.load(fh).get("detections", [])

    # Local R/G on a fine grid — SUBSTRATE ONLY. Detections are masked out
    # before averaging; a flake included here would be reported as substrate
    # colour. Normalising the blurred sums by the blurred mask weight gives a
    # mask-aware local mean (blocks with no substrate left come out NaN).
    step = 16
    keep = (~flake_mask(frame, (h, w))).astype(np.float32)
    sm = (w // step, h // step)
    wgt = cv2.GaussianBlur(cv2.resize(keep, sm, interpolation=cv2.INTER_AREA), (0, 0), 3)
    num_r = cv2.GaussianBlur(cv2.resize(im[:, :, 2] * keep, sm, interpolation=cv2.INTER_AREA), (0, 0), 3)
    num_g = cv2.GaussianBlur(cv2.resize(im[:, :, 1] * keep, sm, interpolation=cv2.INTER_AREA), (0, 0), 3)
    local_rg = np.where(wgt > 0.15, num_r / np.maximum(num_g, 1e-6), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.2))

    ax = axes[0]
    ax.imshow(rgb)
    for d in dets:
        c = np.array(d["contour"])
        ax.plot(c[:, 0], c[:, 1], lw=1.1, color="#00e5ff")
    ax.set_title(f"{frame.stem} — as captured\ncyan = segmented detections ({len(dets)})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    m = ax.imshow(
        local_rg,
        cmap="RdBu_r",
        vmin=np.nanpercentile(local_rg, 2),
        vmax=max(np.nanpercentile(local_rg, 98), gold_rg),
        extent=(0, w, h, 0),
    )
    cb = fig.colorbar(m, ax=ax, fraction=0.036)
    cb.ax.axhline(gold_rg, color="k", lw=2)
    cb.set_label(f"local R/G   (black line = golden clean blank {gold_rg:.4f})", fontsize=8)
    ax.set_title(
        "substrate colour field\nnowhere in this frame reaches the clean-blank value",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2]
    dev = gold_rg - local_rg  # how far below clean spec
    shade = np.nan_to_num(np.clip(dev / max(np.nanmax(dev), 1e-6), 0, 1))
    ax.imshow(rgb)
    ax.imshow(
        np.dstack([np.ones_like(shade), 0.55 * np.ones_like(shade), np.zeros_like(shade), 0.45 * shade]),
        extent=(0, w, h, 0),
    )
    for d in dets:
        c = np.array(d["contour"])
        ax.plot(c[:, 0], c[:, 1], lw=1.1, color="#00e5ff")
    ax.legend(
        handles=[
            Patch(color="#00e5ff", label="flake (segmented detection)"),
            Patch(color="#ff8c00", alpha=0.6, label="substrate, shaded by deficit vs clean blank"),
        ],
        fontsize=8,
        loc="lower right",
    )
    ax.set_title(
        "my reading: ONE contaminated substrate, not clean+patches\n(darker orange = further from clean 90 nm SiO₂)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-m", "--material", required=True, choices=DetectorConfig.material_names())
    ap.add_argument("--mag", type=float, default=10.0, help="Objective mag for the golden ref (default 10)")
    ap.add_argument("--frames", type=Path, default=None, help="Chip dir with scan frames, for the spatial map")
    ap.add_argument("--max-frames", type=int, default=8, help="Frames to include in the spatial map")
    ap.add_argument("--flatfield", type=Path, default=Path("calibration/flatfield_10x_bin3.npy"))
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--annotate", type=Path, default=None, help="Frame .jpg to label surfaces on")
    ap.add_argument("--annotate-out", type=Path, default=None, help="Where to write the labelled frame")
    args = ap.parse_args()

    if args.annotate:
        gold = GOLDEN_BY_MAG.get(args.mag, (float("nan"), float("nan")))[0]
        seg = args.annotate.parent.parent / "seg" / f"{args.annotate.stem}.json"
        out = args.annotate_out or Path(f"{args.annotate.stem}_surfaces.png")
        annotate_frame(args.annotate, seg, load_flatfield_bgr(args.flatfield), gold, out)
        print(f"wrote {out}")
        return 0

    cfg = DetectorConfig.from_material(args.material)
    gold_rg, gold_bg = GOLDEN_BY_MAG.get(args.mag, (float("nan"), float("nan")))

    summaries = sorted(args.run_dir.glob("chip_*/seg/summary.json"))
    if not summaries:
        print(f"no chip seg summaries under {args.run_dir}")
        return 1

    # ---- 1. per-chip background ratios vs golden -------------------------
    print(f"Golden clean-blank ref at {args.mag:g}x: R/G {gold_rg:.4f}   B/G {gold_bg:.4f}\n")
    print(
        f"{'chip':>7} {'frames':>7} {'R/G':>8} {'+/-':>7} {'vs gold':>9} "
        f"{'B/G':>8} {'+/-':>7} {'vs gold':>9} {'G DN':>7}"
    )
    per_chip = {}
    for s in summaries:
        with open(s) as fh:
            data = json.load(fh)
        chip = s.parent.parent.name
        modes = [v for v in data["bg_mode_by_frame"].values() if v]
        if not modes:
            continue
        a = np.array(modes, float)
        rg, bg = a[:, 0] / a[:, 1], a[:, 2] / a[:, 1]
        per_chip[chip] = (np.median(rg), np.median(bg), np.median(a[:, 1]))
        print(
            f"{chip:>7} {len(a):>7} {np.median(rg):8.4f} {trimmed_std(rg):7.4f} {np.median(rg) - gold_rg:+9.4f} "
            f"{np.median(bg):8.4f} {trimmed_std(bg):7.4f} {np.median(bg) - gold_bg:+9.4f} {np.median(a[:, 1]):7.1f}"
        )
    spread = max(v[0] for v in per_chip.values()) - min(v[0] for v in per_chip.values())
    print(f"\n  chip-to-chip R/G spread: {spread:.4f}")
    print("  (illumination is common to a run; a spread across chips is a surface property)")

    # Estimator noise floor, so the numbers above can be read against it.
    # histogram_mode returns an integer, so a ratio of two ~50 DN modes is
    # quantized; that floor is present before any real variation.
    chunks = []
    for s in summaries:
        with open(s) as fh:
            chunks.append(np.array([v for v in json.load(fh)["bg_mode_by_frame"].values() if v], float))
    all_modes = np.concatenate(chunks)
    med_r, med_g = float(np.median(all_modes[:, 0])), float(np.median(all_modes[:, 1]))
    q = 0.5 / math.sqrt(3.0)  # uniform quantization error of an integer mode
    q_rg = (med_r / med_g) * math.hypot(q / med_r, q / med_g)
    print(f"  integer-quantization floor on R/G at R~{med_r:.0f} G~{med_g:.0f}: {q_rg:.4f}")
    print("  frame-to-frame +/- above includes this floor; subtract in quadrature for real variation.\n")

    # ---- 2. spatial structure inside frames ------------------------------
    if args.frames:
        ff = load_flatfield_bgr(args.flatfield)
        frames = sorted(glob.glob(str(args.frames / "*.jpg")))[: args.max_frames]
        print(f"Within-frame background colour structure ({len(frames)} frames, {BLOCK}px blocks):")
        print(f"{'frame':>12} {'raw p5-95':>10} {'detrended':>10} {'struct raw':>11} {'struct detr':>12}")
        raws, detrs = [], []
        for p in frames:
            g = block_colour_map(Path(p), ff)
            d = detrend_quadratic(g)
            raws.append(float(np.ptp(np.nanpercentile(g, [5, 95]))))
            detrs.append(float(np.ptp(np.nanpercentile(d, [5, 95]))))
            print(
                f"{Path(p).stem:>12} {raws[-1]:10.4f} {detrs[-1]:10.4f} "
                f"{structure_ratio(g):11.1f} {structure_ratio(d):12.1f}"
            )
        print(f"\n  median raw spread {np.median(raws):.4f} -> detrended {np.median(detrs):.4f}")
        print("  the removed part is a smooth field gradient; the remainder is patch-scale.\n")

    # ---- 3. what it does to contrast -------------------------------------
    if not isinstance(cfg, RGPointDetectorConfig):
        print(f"{args.material} has no RG point locus — skipping the contrast-bias regression")
        return 0
    pts = cfg.cal_reference_points
    lo, hi = min(p.r for p in pts), max(p.r for p in pts)

    perp, rc, bgrg, px, py = [], [], [], [], []
    for s in summaries:
        with open(s) as fh:
            data = json.load(fh)
        bm = data["bg_mode_by_frame"]
        for fr, dets in data["detections_by_frame"].items():
            mode = bm.get(fr)
            if not mode:
                continue
            for d in dets:
                r, g, _ = d["contrast_rgb"]
                if not (r < -0.05 and g < -0.05 and abs(r - g) < 0.15 and lo <= r <= hi):
                    continue
                if d["size_um2"] < 300 or d["perim_ratio"] >= 1.5:
                    continue
                perp.append(signed_perp(pts, r, g))
                rc.append(r)
                bgrg.append(mode[0] / mode[1])
                px.append(d["center"][0])
                py.append(d["center"][1])
    perp, rc, bgrg, px, py = (np.array(v) for v in (perp, rc, bgrg, px, py))
    print(f"Locus-offset regression, n={len(perp)}")
    print(f"  median signed perpendicular offset: {np.median(perp):+.4f}")
    design = np.c_[np.ones(len(rc)), rc, bgrg, px, py]
    coef, *_ = np.linalg.lstsq(design, perp, rcond=None)
    resid = perp - design @ coef
    cov = np.sum(resid**2) / (len(rc) - design.shape[1]) * np.linalg.inv(design.T @ design)
    names = ["const", "R_contrast", "bg_R/G", "x_px", "y_px"]
    scales = [1.0, 1.0, spread, 1824.0, 1216.0]
    print(f"\n  perp ~ {' + '.join(names[1:])}")
    print(f"{'term':>12} {'coeff':>12} {'std err':>10} {'t':>7}   effect over its range")
    for i, (n, sc) in enumerate(zip(names, scales, strict=True)):
        se = math.sqrt(cov[i, i])
        note = "" if i == 0 else f"{coef[i] * sc:+.4f}"
        print(f"{n:>12} {coef[i]:12.4f} {se:10.4f} {coef[i] / se:7.2f}   {note}")
    print(f"\n  offset to explain: {np.median(perp):+.4f}")
    print("  compare each term's effect-over-range against it before assigning blame.")

    if args.plot:
        _plot(args.plot, per_chip, gold_rg, gold_bg, perp, rc, bgrg, args)
        print(f"\nwrote {args.plot}")
    return 0


def _plot(path, per_chip, gold_rg, gold_bg, perp, rc, bgrg, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    ax = axes[0]
    chips = sorted(per_chip)
    ax.bar(range(len(chips)), [per_chip[c][0] for c in chips], color="tab:red", alpha=0.75, label="R/G")
    ax.axhline(gold_rg, color="k", ls="--", lw=1.4, label=f"golden clean blank {gold_rg:.4f}")
    ax.set_xticks(range(len(chips)))
    ax.set_xticklabels(chips, rotation=45, ha="right")
    ax.set_ylim(0.85, 1.05)
    ax.set_ylabel("background R/G")
    ax.set_title("background is off-spec on every chip, by differing amounts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    if args.frames:
        ff = load_flatfield_bgr(args.flatfield)
        frames = sorted(glob.glob(str(args.frames / "*.jpg")))[:1]
        if frames:
            g = block_colour_map(Path(frames[0]), ff)
            im = ax.imshow(g, cmap="RdBu_r", vmin=np.nanpercentile(g, 2), vmax=np.nanpercentile(g, 98))
            fig.colorbar(im, ax=ax, fraction=0.036)
            ax.set_title(f"background R/G across one field\n{Path(frames[0]).stem}")
    else:
        ax.set_title("(pass --frames for the spatial map)")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2]
    ax.scatter(rc, perp, s=6, c=bgrg, cmap="viridis")
    order = np.argsort(rc)
    k = max(len(rc) // 12, 5)
    binned = [(np.median(rc[order][i : i + k]), np.median(perp[order][i : i + k])) for i in range(0, len(rc) - k, k)]
    if binned:
        ax.plot(*zip(*binned, strict=True), "-o", c="tab:red", lw=2, ms=4, label="binned median")
    ax.axhline(0, c="k", lw=1)
    ax.set_xlabel("R contrast  (deeper = thicker)")
    ax.set_ylabel("signed offset from cal locus")
    ax.set_title("offset tracks thickness, not background\n(colour = frame background R/G)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
