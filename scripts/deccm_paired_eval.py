"""Paired legacy-vs-identity scan evaluation for the de-CCM cutover.

Two chip scans of the same chips, same hulls/planes/camera params, differing
only in colour matrix (5800K legacy vs identity). Frame grids match to ±1
frame but not by index — frames pair by stage position.

Stages (run independently):
  --pair-only          build + report the frame pairing map
  --pixel              pixel-level parity: deccm(legacy frames) vs identity frames
                       (requires --deccm-dir with de-CCM'd frames from deccm_frames.py)

Usage:
    uv run python scripts/deccm_paired_eval.py scans/run_20260807_1106 \
        scans/run_20260807_1106_identity --chips 0 1 --pair-only
    uv run python scripts/deccm_paired_eval.py scans/run_20260807_1106 \
        scans/run_20260807_1106_identity --deccm-dir scans/run_20260807_1106_deccm \
        --chips 0 1 --pixel -o /tmp/deccm_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

FEATURE_G_DELTA = 8.0  # counts from frame G median -> "feature" (flake-ish) pixel
N_HBN_V4 = 2.060  # pipeline-model fit vs Toghrul AFM anchors (offsets = 0)
OXIDE_NM = 90.0
NA_10X = 0.25
MATCH_RADIUS_UM = 25.0


class FramePair(NamedTuple):
    legacy_n: int
    identity_n: int
    line: int
    x_dir: int  # +1 / -1 scan direction of the row
    dx_um: float  # identity - legacy stage x at capture
    dy_um: float

    def to_dict(self) -> dict:
        return self._asdict()


def load_meta(scan_dir: Path) -> dict:
    return json.loads((scan_dir / "scan_meta.json").read_text())


def row_direction(frames: list[dict], line: int) -> int:
    xs = [f["x_um"] for f in frames if f["line"] == line]
    return 1 if len(xs) < 2 or xs[-1] >= xs[0] else -1


def pair_frames(legacy: dict, identity: dict) -> tuple[list[FramePair], list[int], list[int]]:
    """1:1 nearest-stage-position pairing, per scan line.

    Returns (pairs, unmatched_legacy_n, unmatched_identity_n).
    """
    lf, idf = legacy["frames"], identity["frames"]
    pairs: list[FramePair] = []
    un_l: list[int] = []
    un_i: list[int] = []
    lines = sorted({f["line"] for f in lf} | {f["line"] for f in idf})
    for line in lines:
        a = [f for f in lf if f["line"] == line]
        b = [f for f in idf if f["line"] == line]
        x_dir = row_direction(lf, line)
        # Greedy globally-closest 1:1 matching within the row.
        cands = sorted(
            ((abs(fb["x_um"] - fa["x_um"]), i, j) for i, fa in enumerate(a) for j, fb in enumerate(b)),
            key=lambda t: t[0],
        )
        used_a: set[int] = set()
        used_b: set[int] = set()
        for _, i, j in cands:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            fa, fb = a[i], b[j]
            pairs.append(
                FramePair(
                    legacy_n=fa["n"],
                    identity_n=fb["n"],
                    line=line,
                    x_dir=x_dir,
                    dx_um=fb["x_um"] - fa["x_um"],
                    dy_um=fb["y_um"] - fa["y_um"],
                )
            )
        un_l += [a[i]["n"] for i in range(len(a)) if i not in used_a]
        un_i += [b[j]["n"] for j in range(len(b)) if j not in used_b]
    pairs.sort(key=lambda p: p.legacy_n)
    return pairs, un_l, un_i


def report_pairing(chip: int, pairs: list[FramePair], un_l: list[int], un_i: list[int]) -> dict:
    dx = np.array([p.dx_um for p in pairs])
    out = {
        "chip": chip,
        "n_pairs": len(pairs),
        "unmatched_legacy": un_l,
        "unmatched_identity": un_i,
        "dx_um": {"mean": float(dx.mean()), "rms": float(np.sqrt((dx**2).mean())), "max_abs": float(np.abs(dx).max())},
    }
    print(f"chip {chip}: {len(pairs)} pairs; unmatched legacy {un_l or '-'}, identity {un_i or '-'}")
    for d in (1, -1):
        sel = dx[np.array([p.x_dir for p in pairs]) == d]
        if len(sel):
            print(
                f"   {'+X' if d > 0 else '-X'} rows: dx mean {sel.mean():+7.1f} um,"
                f" rms {np.sqrt((sel**2).mean()):6.1f}, max |dx| {np.abs(sel).max():6.1f} (n={len(sel)})"
            )
        out[f"dx_um_dir{d:+d}"] = {"mean": float(sel.mean()), "max_abs": float(np.abs(sel).max())} if len(sel) else None
    return out


class PixelStats(NamedTuple):
    legacy_n: int
    identity_n: int
    shift_px: tuple[float, float]  # refined (dx, dy) image shift, identity -> deccm alignment
    shift_resid_px: float  # |refined - stage-predicted| (x)
    align_rms: float  # gray RMS on the central crop at the refined shift
    overlap_frac: float
    feature_frac: float
    mean_d: tuple[float, float, float]  # per-channel mean(deccm - identity), overlap region
    rms_d: tuple[float, float, float]
    p99_abs_d: tuple[float, float, float]
    rms_d_substrate: tuple[float, float, float]
    rms_d_feature: tuple[float, float, float]

    def to_dict(self) -> dict:
        return self._asdict()


def _shift_rms(ag: np.ndarray, bg: np.ndarray, sx: float, sy: float) -> float:
    """Gray RMS between a and b warped by (sx, sy), on the shift-valid overlap interior."""
    h, w = ag.shape
    m = np.array([[1, 0, -sx], [0, 1, -sy]], dtype=np.float32)
    bw = cv2.warpAffine(bg, m, (w, h), flags=cv2.INTER_LINEAR)
    x0, x1 = max(0, -int(np.ceil(sx))) + 8, w + min(0, -int(np.floor(sx))) - 8
    y0, y1 = max(0, -int(np.ceil(sy))) + 8, h + min(0, -int(np.floor(sy))) - 8
    if x1 - x0 < 200 or y1 - y0 < 200:
        return float("inf")
    return float(np.sqrt(((ag[y0:y1, x0:x1] - bw[y0:y1, x0:x1]) ** 2).mean()))


def _refine_shift(ag: np.ndarray, bg: np.ndarray, sx0: float, sy0: float) -> tuple[float, float, float]:
    """Two-stage local grid search around (sx0, sy0); returns (sx, sy, rms)."""
    best = (sx0, sy0, _shift_rms(ag, bg, sx0, sy0))
    for step, span in ((1.0, 3), (0.25, 3)):
        cx, cy = best[0], best[1]
        for dx in np.arange(-span, span + 0.001) * step:
            for dy in np.arange(-span, span + 0.001) * step:
                if dx == 0 and dy == 0:
                    continue
                r = _shift_rms(ag, bg, cx + dx, cy + dy)
                if r < best[2]:
                    best = (cx + dx, cy + dy, r)
    return best


@lru_cache(maxsize=2)
def _load_ff_bgr(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)[:, :, ::-1]  # RGB (npy convention) -> BGR


def compare_pair(
    deccm_fp: Path, identity_fp: Path, pair: FramePair, px_um: float, min_overlap: float, ff_path: str | None
) -> tuple[PixelStats, np.ndarray] | None:
    """Subpixel-align identity frame onto the de-CCM'd legacy frame and diff.

    Returns (stats, hist) where hist is a per-channel 2D histogram of
    (identity value // 8, delta clipped to ±31) on subsampled pixels.
    """
    a = cv2.imread(str(deccm_fp))  # BGR
    b = cv2.imread(str(identity_fp))
    if a is None or b is None:
        return None
    if ff_path:
        from flakefinder.scan_utils import apply_flatfield

        ff = _load_ff_bgr(ff_path)
        a = apply_flatfield(a, ff)
        b = apply_flatfield(b, ff)
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = ag.shape
    pred_px = pair.dx_um / px_um
    if abs(pred_px) > w / 4:
        # Large known offset (adjacent-frame control): phase correlation is
        # unreliable at low overlap; start from the stage prediction, trying
        # both image-x sign conventions.
        cands = [(s * pred_px, 0.0) for s in (-1, 1)]
    else:
        win = cv2.createHanningWindow(ag.shape[::-1], cv2.CV_32F)
        (psx, psy), _resp = cv2.phaseCorrelate(ag, bg, win)
        cands = [(psx, psy)]
    sx, sy, rms = min((_refine_shift(ag, bg, cx, cy) for cx, cy in cands), key=lambda t: t[2])
    resid = min(abs(sx - pred_px), abs(sx + pred_px))
    # Warp full colour frame by the refined subpixel shift; crop shift-valid interior.
    m = np.array([[1, 0, -sx], [0, 1, -sy]], dtype=np.float32)
    bw = cv2.warpAffine(b, m, (w, h), flags=cv2.INTER_LINEAR)
    x0, x1 = max(0, -int(np.ceil(sx))) + 8, w + min(0, -int(np.floor(sx))) - 8
    y0, y1 = max(0, -int(np.ceil(sy))) + 8, h + min(0, -int(np.floor(sy))) - 8
    if (x1 - x0) * (y1 - y0) < min_overlap * w * h:
        return None
    ca = a[y0:y1, x0:x1].astype(np.int16)
    cb = bw[y0:y1, x0:x1].astype(np.int16)
    d = ca - cb  # BGR
    g_med = float(np.median(cb[:, :, 1]))
    feature = np.abs(cb[:, :, 1].astype(np.float32) - g_med) > FEATURE_G_DELTA
    feat_frac = float(feature.mean())

    def ch_stats(mask: np.ndarray | None) -> tuple[tuple, tuple, tuple]:
        means, rmss, p99s = [], [], []
        for c in (2, 1, 0):  # RGB order
            dc = d[:, :, c][mask] if mask is not None else d[:, :, c]
            means.append(float(dc.mean()))
            rmss.append(float(np.sqrt((dc.astype(np.float64) ** 2).mean())))
            p99s.append(float(np.percentile(np.abs(dc), 99)))
        return tuple(means), tuple(rmss), tuple(p99s)

    mean_d, rms_d, p99_d = ch_stats(None)
    _, rms_sub, _ = ch_stats(~feature)
    _, rms_feat, _ = ch_stats(feature) if feat_frac > 0 else (None, (0.0, 0.0, 0.0), None)

    # (value, delta) joint histogram on 4x-subsampled pixels: 32 value bins x 63 delta bins
    hist = np.zeros((3, 32, 63), dtype=np.int64)
    vs = cb[::4, ::4]
    ds = np.clip(d[::4, ::4], -31, 31)
    for k, c in enumerate((2, 1, 0)):
        np.add.at(hist[k], (vs[:, :, c] // 8, ds[:, :, c] + 31), 1)

    stats = PixelStats(
        legacy_n=pair.legacy_n,
        identity_n=pair.identity_n,
        shift_px=(round(sx, 2), round(sy, 2)),
        shift_resid_px=round(resid, 2),
        align_rms=round(rms, 3),
        overlap_frac=float((x1 - x0) * (y1 - y0) / (w * h)),
        feature_frac=round(feat_frac, 4),
        mean_d=mean_d,
        rms_d=rms_d,
        p99_abs_d=p99_d,
        rms_d_substrate=rms_sub,
        rms_d_feature=rms_feat,
    )
    return stats, hist


def _pixel_worker(args_tuple) -> tuple[dict, np.ndarray] | None:
    deccm_fp, identity_fp, pair, px_um, min_overlap, ff_path = args_tuple
    res = compare_pair(deccm_fp, identity_fp, pair, px_um, min_overlap, ff_path)
    if res is None:
        return None
    stats, hist = res
    return stats.to_dict(), hist


def adjacent_pairs(meta: dict) -> list[FramePair]:
    """Consecutive-frame pairs within each scan line of a single scan (noise-floor control)."""
    pairs = []
    frames = sorted(meta["frames"], key=lambda f: f["n"])
    for fa, fb in zip(frames, frames[1:], strict=False):
        if fa["line"] != fb["line"]:
            continue
        pairs.append(
            FramePair(
                legacy_n=fa["n"],
                identity_n=fb["n"],
                line=fa["line"],
                x_dir=1 if fb["x_um"] >= fa["x_um"] else -1,
                dx_um=fb["x_um"] - fa["x_um"],
                dy_um=fb["y_um"] - fa["y_um"],
            )
        )
    return pairs


def run_pixel(
    deccm_scan: Path,
    identity_scan: Path,
    pairs: list[FramePair],
    px_um: float,
    jobs: int,
    min_overlap: float = 0.45,
    ff_path: str | None = None,
) -> tuple[list[dict], np.ndarray]:
    tasks = [
        (
            deccm_scan / f"frame_{p.legacy_n:04d}.jpg",
            identity_scan / f"frame_{p.identity_n:04d}.jpg",
            p,
            px_um,
            min_overlap,
            ff_path,
        )
        for p in pairs
    ]
    tasks = [t for t in tasks if t[0].exists() and t[1].exists()]
    all_stats: list[dict] = []
    hist_total = np.zeros((3, 32, 63), dtype=np.int64)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, res in enumerate(ex.map(_pixel_worker, tasks, chunksize=8), 1):
            if res is None:
                continue
            stats, hist = res
            all_stats.append(stats)
            hist_total += hist
            if i % 100 == 0:
                print(f"  [{i}/{len(tasks)}]")
    return all_stats, hist_total


def summarize_pixel(chip: int, stats: list[dict]) -> dict:
    def agg(key: str) -> list[float]:
        arr = np.array([s[key] for s in stats])
        return [round(float(v), 3) for v in np.sqrt((arr**2).mean(axis=0))]

    mean_d = np.array([s["mean_d"] for s in stats])
    out = {
        "chip": chip,
        "n_frames": len(stats),
        "mean_delta_rgb": [round(float(v), 3) for v in mean_d.mean(axis=0)],
        "mean_delta_rgb_spread": [round(float(v), 3) for v in mean_d.std(axis=0)],
        "rms_delta_rgb": agg("rms_d"),
        "rms_delta_rgb_substrate": agg("rms_d_substrate"),
        "rms_delta_rgb_feature": agg("rms_d_feature"),
        "p99_abs_delta_rgb": [round(float(np.mean([s["p99_abs_d"][c] for s in stats])), 2) for c in range(3)],
        "shift_resid_px_p95": round(float(np.percentile([s["shift_resid_px"] for s in stats], 95)), 2),
        "worst_frames_by_rms_g": sorted(stats, key=lambda s: -s["rms_d"][1])[:10],
    }
    print(f"chip {chip}: {len(stats)} compared frames")
    print(f"   mean delta (R,G,B): {out['mean_delta_rgb']}  (frame-to-frame spread {out['mean_delta_rgb_spread']})")
    print(f"   RMS delta  (R,G,B): {out['rms_delta_rgb']}   substrate {out['rms_delta_rgb_substrate']}")
    print(f"   feature-px RMS     : {out['rms_delta_rgb_feature']}   p99|d| {out['p99_abs_delta_rgb']}")
    return out


# ---------------------------------------------------------------------------
# Segmentation stage: raw-space hbn_medium config + three-way det comparison
# ---------------------------------------------------------------------------


def median_bg_rgb(seg_dir: Path) -> np.ndarray:
    s = json.loads((seg_dir / "summary.json").read_text())
    modes = np.array([m for m in s["bg_mode_by_frame"].values() if m is not None])
    return np.median(modes, axis=0)


def model_b_contrast_mixed(thicknesses_nm: np.ndarray) -> np.ndarray:
    """Pipeline-model mixed-space B contrast at the given thicknesses (10x chain)."""
    from colorchecker_cal import WB_REQUESTS_HBN_SCAN, derive_pipeline
    from hbn_contrast import _IMX183_BLUE, _IMX183_GREEN, _IMX183_RED, _IMX183_WAVELENGTHS
    from hbn_contrast_widget import compute_rg

    pipe = derive_pipeline(_IMX183_WAVELENGTHS)
    op = pipe.objectives["10x"]
    mix = pipe.ccm @ np.diag(WB_REQUESTS_HBN_SCAN)
    perm = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # row 0 <- B
    b_all, _, t_all = compute_rg(
        complex(N_HBN_V4),
        OXIDE_NM,
        NA_10X,
        max_layers=200,
        red_lit=_IMX183_RED * op.chain_spd,
        green_lit=_IMX183_GREEN * op.chain_spd,
        blue_lit=_IMX183_BLUE * op.chain_spd,
        glare_f=np.array(op.glare_f_raw),
        mix=perm @ mix,
    )
    return np.interp(thicknesses_nm, t_all, b_all)


def transform_contrast(c_out: np.ndarray, bg_out: np.ndarray, m_inv: np.ndarray, pedestal: float) -> np.ndarray:
    """Exact per-channel raw-space contrast for output-space contrast c_out against bg_out.

    f_raw - b_raw = M^-1 (bg*c) exactly; b_raw = M^-1 (bg + p) - p.
    """
    b_raw = m_inv @ (bg_out + pedestal) - pedestal
    return (m_inv @ (bg_out * c_out)) / b_raw


def build_raw_hbn_medium(bg_out: np.ndarray, quiet: bool = False):
    """Raw-space hbn_medium: committed empirical cal points exactly transformed via M^-1.

    B contrast (needed by the mixing transform, absent from the R/G cal
    points) is filled in from the v4 pipeline model at each point's AFM
    thickness. Tier G/R box gates migrate at constant thickness through the
    curve; distance-like gates scale with the band arc-length ratio;
    contrast_offset (counts) scales with the G count-delta ratio.
    """
    from deccm_frames import load_inverse_ccm

    from flakefinder.segmentation import CurveDetectorConfig, DetectorConfig

    base = DetectorConfig.from_material("hbn_medium")
    assert isinstance(base, CurveDetectorConfig)
    m_inv, ped = load_inverse_ccm()

    pts = np.array(base.cal_points)  # (R, G, t_nm)
    b_fill = model_b_contrast_mixed(pts[:, 2])
    raw_pts = []
    for (r, g, t), b in zip(pts, b_fill, strict=True):
        c_raw = transform_contrast(np.array([r, g, b]), bg_out, m_inv, ped)
        raw_pts.append((round(float(c_raw[0]), 4), round(float(c_raw[1]), 4), t))
    raw_arr = np.array(raw_pts)

    def arc_len(a: np.ndarray) -> float:
        return float(np.sum(np.hypot(*np.diff(a[:, :2], axis=0).T)))

    arc_scale = arc_len(raw_arr) / arc_len(pts)
    # G count-delta scale at the mid-band point (contrast_offset is in counts).
    mid = len(pts) // 2
    c_mid = np.array([pts[mid, 0], pts[mid, 1], b_fill[mid]])
    count_scale = float((m_inv @ (bg_out * c_mid))[1] / (bg_out[1] * c_mid[1]))

    # Map G/R box gates at constant thickness: mixed curve value -> t -> raw
    # curve value. Curves are non-monotonic in places, so find the crossing on
    # a dense t grid (crossings of interest — G=3, R=0 — are unique in-band).
    # Bounds outside the curve's reach scale proportionally through the
    # nearest endpoint (0 maps to 0: zero contrast is space-invariant).
    t_grid = np.linspace(pts[0, 2], pts[-1, 2], 800)

    def mixed_to_raw(col: int, val: float) -> float:
        mixed_c = np.interp(t_grid, pts[:, 2], pts[:, col])
        raw_c = np.interp(t_grid, raw_arr[:, 2], raw_arr[:, col])
        cross = np.flatnonzero(np.diff(np.sign(mixed_c - val)))
        if len(cross):
            return float(raw_c[cross[-1]])
        k = 0 if abs(val - mixed_c[0]) < abs(val - mixed_c[-1]) else -1
        return float(val * raw_c[k] / mixed_c[k])

    g_lo, g_hi = base.cal_g_range
    cfg = replace(
        base,
        cal_points=tuple(raw_pts),
        cal_g_range=(mixed_to_raw(1, g_lo), mixed_to_raw(1, g_hi)),
        contrast_offset=base.contrast_offset * count_scale,
        subseg_min_std=base.subseg_min_std * arc_scale,
        subseg_min_range=base.subseg_min_range * arc_scale,
        cal_dist_match=base.cal_dist_match * arc_scale,
        cal_dist_possible=base.cal_dist_possible * arc_scale,
        tier1_cal_dist=base.tier1_cal_dist * arc_scale,
        tier2_cal_dist=base.tier2_cal_dist * arc_scale,
        tier1_g_min=mixed_to_raw(1, base.tier1_g_min),
        tier1_g_max=mixed_to_raw(1, base.tier1_g_max),
        tier1_r_max=mixed_to_raw(0, base.tier1_r_max),
    )
    if not quiet:
        print(f"raw hbn_medium: arc scale {arc_scale:.4f}, count scale {count_scale:.4f}")
        print(f"   contrast_offset {cfg.contrast_offset:.2f} counts, cal_dist match {cfg.cal_dist_match:.3f}")
        print(
            f"   tier1 box: G [{cfg.tier1_g_min:.3f}, {cfg.tier1_g_max:.3f}), R < {cfg.tier1_r_max:.3f}"
            f"   cal_g_range ({cfg.cal_g_range[0]:.3f}, {cfg.cal_g_range[1]:.3f})"
        )
    return cfg, {"arc_scale": arc_scale, "count_scale": count_scale, "raw_cal_points": raw_pts}


def load_dets(seg_dir: Path, meta: dict) -> list[dict]:
    from flakefinder.data_utils import add_stage_coords

    s = json.loads((seg_dir / "summary.json").read_text())
    dets = []
    for frame, frame_dets in s["detections_by_frame"].items():
        for d in frame_dets:
            d["frame"] = frame
            dets.append(d)
    add_stage_coords(dets, meta)  # ty: ignore[invalid-argument-type]  # summary dicts are Detection-shaped
    return dets


def match_by_stage(a: list[dict], b: list[dict], radius_um: float = MATCH_RADIUS_UM) -> list[tuple[dict, dict]]:
    """Greedy globally-closest 1:1 matching on stage coordinates."""
    bx = np.array([d["stage_x"] for d in b])
    by = np.array([d["stage_y"] for d in b])
    cands = []
    for i, d in enumerate(a):
        dist = np.hypot(bx - d["stage_x"], by - d["stage_y"])
        for j in np.flatnonzero(dist <= radius_um):
            cands.append((float(dist[j]), i, int(j)))
    cands.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for _, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((a[i], b[j]))
    return pairs


def tier_counts(dets: list[dict]) -> dict[int, int]:
    t = {1: 0, 2: 0, 3: 0}
    for d in dets:
        t[d.get("tier", 3)] += 1
    return t


def compare_dets(name: str, a: list[dict], b: list[dict], out: dict) -> None:
    ta, tb = tier_counts(a), tier_counts(b)
    pairs = match_by_stage(a, b)
    print(f"== {name}")
    print(f"   A: total {len(a)}, tiers {ta[1]}/{ta[2]}/{ta[3]}   B: total {len(b)}, tiers {tb[1]}/{tb[2]}/{tb[3]}")
    a12 = [d for d in a if d.get("tier", 3) <= 2]
    b12 = [d for d in b if d.get("tier", 3) <= 2]
    p12 = match_by_stage(a12, b12)
    print(f"   matched: {len(pairs)}/{len(a)} of A ({len(b)} B); tier1+2: {len(p12)}/{len(a12)} of A ({len(b12)} B)")
    conf: dict[str, int] = {}
    dt = []
    for da, db in pairs:
        key = f"{da.get('tier', 3)}->{db.get('tier', 3)}"
        conf[key] = conf.get(key, 0) + 1
        if da.get("thickness_nm") is not None and db.get("thickness_nm") is not None:
            dt.append(db["thickness_nm"] - da["thickness_nm"])
    print(f"   tier confusion: {dict(sorted(conf.items()))}")
    if dt:
        arr = np.array(dt)
        print(
            f"   thickness delta (B-A): mean {arr.mean():+.2f} nm, rms {np.sqrt((arr**2).mean()):.2f} nm,"
            f" median |d| {np.median(np.abs(arr)):.2f} nm, n={len(arr)}"
        )
    out[name] = {
        "a_total": len(a),
        "b_total": len(b),
        "a_tiers": ta,
        "b_tiers": tb,
        "matched": len(pairs),
        "matched_t12": len(p12),
        "a_t12": len(a12),
        "b_t12": len(b12),
        "tier_confusion": conf,
        "thickness_delta_mean": float(np.mean(dt)) if dt else None,
        "thickness_delta_rms": float(np.sqrt(np.mean(np.array(dt) ** 2))) if dt else None,
        "pairs": [
            {
                "a": {k: da.get(k) for k in ("frame", "tier", "thickness_nm", "contrast_rgb", "cal_dist", "size_um2")},
                "b": {k: db.get(k) for k in ("frame", "tier", "thickness_nm", "contrast_rgb", "cal_dist", "size_um2")},
                "stage_x": da["stage_x"],
                "stage_y": da["stage_y"],
            }
            for da, db in pairs
        ],
    }


def raw_model_b_curve() -> tuple[np.ndarray, np.ndarray]:
    """(t_nm, raw-space model B contrast) from the v4 pipeline chain (10x)."""
    from colorchecker_cal import derive_pipeline
    from hbn_contrast import _IMX183_BLUE, _IMX183_GREEN, _IMX183_RED, _IMX183_WAVELENGTHS
    from hbn_contrast_widget import compute_rg

    pipe = derive_pipeline(_IMX183_WAVELENGTHS)
    op = pipe.objectives["10x"]
    perm = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b_all, _, t_all = compute_rg(
        complex(N_HBN_V4),
        OXIDE_NM,
        NA_10X,
        max_layers=200,
        red_lit=_IMX183_RED * op.chain_spd,
        green_lit=_IMX183_GREEN * op.chain_spd,
        blue_lit=_IMX183_BLUE * op.chain_spd,
        glare_f=np.array(op.glare_f_raw),
        mix=perm @ np.eye(3),
    )
    return t_all, b_all


def run_revisit_stage(args: argparse.Namespace) -> None:
    """Paired revisit comparison: de-CCM legacy revisit PNGs in memory vs identity PNGs.

    Same filenames both sides; stationary captures with per-point AF, so
    alignment is phase-corr + subpixel refine from zero. Lossless PNGs:
    no JPEG floor in the comparison.
    """
    from deccm_frames import deccm_image, load_inverse_ccm

    m_inv, ped = load_inverse_ccm()
    for mag in ("20x", "50x"):
        all_stats = []
        clip_l = []
        clip_i = []
        for chip in args.chips:
            leg_dir = args.legacy_run / f"chip_{chip}" / f"revisit_{mag}"
            id_dir = args.identity_run / f"chip_{chip}" / f"revisit_{mag}"
            if not leg_dir.is_dir() or not id_dir.is_dir():
                continue
            for fp in sorted(leg_dir.glob("*.png")):
                fp_i = id_dir / fp.name
                if not fp_i.exists():
                    continue
                a_bgr = cv2.imread(str(fp))
                b = cv2.imread(str(fp_i))
                if a_bgr is None or b is None:
                    continue
                a_rgb, cf = deccm_image(a_bgr[:, :, ::-1], m_inv, ped)
                clip_l.append(cf)
                clip_i.append(float(np.any((b == 0) | (b == 255), axis=-1).mean()))
                a = a_rgb[:, :, ::-1]
                ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
                bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
                win = cv2.createHanningWindow(ag.shape[::-1], cv2.CV_32F)
                (psx, psy), _ = cv2.phaseCorrelate(ag, bg, win)
                sx, sy, rms = _refine_shift(ag, bg, psx, psy)
                h, w = ag.shape
                m = np.array([[1, 0, -sx], [0, 1, -sy]], dtype=np.float32)
                bw = cv2.warpAffine(b, m, (w, h), flags=cv2.INTER_LINEAR)
                x0, x1 = max(0, -int(np.ceil(sx))) + 8, w + min(0, -int(np.floor(sx))) - 8
                y0, y1 = max(0, -int(np.ceil(sy))) + 8, h + min(0, -int(np.floor(sy))) - 8
                d = a[y0:y1, x0:x1].astype(np.int16) - bw[y0:y1, x0:x1].astype(np.int16)
                g_med = float(np.median(bw[y0:y1, x0:x1, 1]))
                feat = np.abs(bw[y0:y1, x0:x1, 1].astype(np.float32) - g_med) > FEATURE_G_DELTA
                rec = {"file": f"c{chip}/{fp.name}", "shift": (round(sx, 2), round(sy, 2)), "align_rms": round(rms, 2)}
                for label, mask in (("all", None), ("substrate", ~feat), ("feature", feat)):
                    dm = d if mask is None else d[mask]
                    if dm.size == 0:
                        continue
                    rec[f"mean_{label}"] = [round(float(dm[..., c].mean()), 3) for c in (2, 1, 0)]
                    rec[f"rms_{label}"] = [
                        round(float(np.sqrt((dm[..., c].astype(np.float64) ** 2).mean())), 3) for c in (2, 1, 0)
                    ]
                all_stats.append(rec)
        if not all_stats:
            continue
        print(f"== revisit {mag}: {len(all_stats)} pairs")
        print(
            f"   clip frac: legacy mean {np.mean(clip_l):.3%} max {np.max(clip_l):.2%},"
            f" identity mean {np.mean(clip_i):.3%} max {np.max(clip_i):.2%}"
        )
        for label in ("all", "substrate", "feature"):
            rms = np.array([r[f"rms_{label}"] for r in all_stats if f"rms_{label}" in r])
            mean = np.array([r[f"mean_{label}"] for r in all_stats if f"mean_{label}" in r])
            print(
                f"   {label:9s} mean delta RGB {mean.mean(axis=0).round(3)}"
                f"  RMS {np.sqrt((rms**2).mean(axis=0)).round(2)}"
            )
        shifts = np.array([np.hypot(*r["shift"]) for r in all_stats])
        print(f"   |shift| px: median {np.median(shifts):.1f}, p95 {np.percentile(shifts, 95):.1f}")
        (args.out / f"revisit_{mag}_pixel.json").write_text(json.dumps(all_stats, indent=1))


def build_hbn_medium_v2():
    """From-scratch identity-space hbn_medium: model-curve cal + shape gates only.

    Cal: v4 pipeline model raw-space (R,G)(t), n=2.060, oxide 90, NA 0.25,
    zero offsets, at 0.1 nm thickness resolution (RGPoint quantization
    ~0.003 vs poly2 curve-fit residual up to 0.10 — points win).
    Gates: cal_dist + thickness window + shape (perim, aspect, size).
    Colour-sensitive gates (R/G boxes, entropy, bg_ratio demotions)
    deliberately wide open — to be worked in one at a time against the
    labeled populations.
    """
    from colorchecker_cal import derive_pipeline
    from hbn_contrast import _IMX183_BLUE, _IMX183_GREEN, _IMX183_RED, _IMX183_WAVELENGTHS
    from hbn_contrast_widget import compute_rg

    from flakefinder.segmentation import (
        CalPointRG,
        ContrastMode,
        GainRGB,
        RGPointDetectorConfig,
        Substrate,
        _score_hbn_thick_50_100,
    )

    pipe = derive_pipeline(_IMX183_WAVELENGTHS)
    op = pipe.objectives["10x"]
    r_m, g_m, t_m = compute_rg(
        complex(N_HBN_V4),
        OXIDE_NM,
        NA_10X,
        max_layers=200,
        red_lit=_IMX183_RED * op.chain_spd,
        green_lit=_IMX183_GREEN * op.chain_spd,
        blue_lit=_IMX183_BLUE * op.chain_spd,
        glare_f=np.array(op.glare_f_raw),
    )
    step_nm = 0.1
    t_grid = np.arange(3.0, 50.0 + step_nm / 2, step_nm)
    points = tuple(
        CalPointRG(
            layers=int(round(t / step_nm)),
            r=round(float(np.interp(t, t_m, r_m)), 4),
            g=round(float(np.interp(t, t_m, g_m)), 4),
        )
        for t in t_grid
    )
    return RGPointDetectorConfig(
        name="hBN 5-20nm v2 (identity)",
        substrate=Substrate.SI_90NM,
        contrast_mode=ContrastMode.ABOVE,
        contrast_offset=14.0,
        min_size_um2=400.0,
        edge_margin_px=50,
        morph_kernel_size=5,
        entropy_threshold=0.4,
        subseg_min_std=0.69,
        subseg_min_range=0.43,
        cal_reference_points=points,
        layer_spacing_nm=step_nm,  # 0.1 nm pseudo-layers: thickness resolution, not literal layers
        classify_nm=True,
        cal_dist_match=0.10,
        cal_dist_possible=0.20,
        non_match_label="non-hBN",
        white_balance=GainRGB(red=1.41, green=1.02, blue=2.51),
        score_fn=_score_hbn_thick_50_100,
        tier1_perim_ratio=1.50,
        tier1_cal_dist=0.065,
        tier1_g_min=-99.0,
        tier1_g_max=99.0,
        tier1_r_max=99.0,
        tier1_r_min=-99.0,
        tier1_entropy_max=99.0,
        tier1_min_size_um2=500.0,
        tier1_br_ratio_max=99.0,
        tier1_aspect_ratio=6.0,
        tier1_solidity_min=0.0,
        tier1_circularity_min=0.0,
        tier1_grad_energy_max=99.0,  # crap gates applied post-hoc (V2_GRAD_ENERGY_MAX/V2_B_MIN); production scorer TBD
        tier1_thickness_window_nm=(4.5, 26.0),
        tier2_perim_ratio=1.50,
        tier2_cal_dist=0.10,
        tier2_entropy_max=99.0,
    )


# Crap gates from labeled paired-scan data (good n=122 / crap n=87 inside v2 T1,
# ~90% joint recall -> 20% crap leak): texture + a B floor. Tape residue is
# B-deficient in identity space; model B >= ~0.44 across the 4.5-26nm window,
# so the floor costs nothing in-band. Applied as a post-tier filter here;
# a production _score_hbn_medium_v2 would carry them natively.
V2_GRAD_ENERGY_MAX = 80.0
V2_B_MIN = 0.41


def v2_post_filter(dets: list[dict]) -> int:
    """Demote T1 dets failing the crap gates to T2. Returns demotion count."""
    n = 0
    for d in dets:
        if d.get("tier", 3) != 1:
            continue
        if d.get("grad_energy", 0.0) > V2_GRAD_ENERGY_MAX or d["contrast_rgb"][2] < V2_B_MIN:
            d["tier"] = 2
            n += 1
    return n


def run_v2_stage(args: argparse.Namespace) -> None:
    import segment_chip_scan

    cfg = build_hbn_medium_v2()
    print(f"hbn_medium_v2: {len(cfg.cal_reference_points)} cal points, tier1 cal_dist {cfg.tier1_cal_dist}")
    ff_deccm = args.deccm_dir / "flatfield_10x_bin3_deccm.npy"
    for chip in args.chips:
        ident_chip = args.identity_run / f"chip_{chip}"
        seg_v2 = ident_chip / "seg_v2"
        if args.segment:
            print(f"== segment identity chip {chip} (hbn_medium_v2)")
            segment_chip_scan.run(
                ident_chip / "scan_10x", seg_v2, flatfield=ff_deccm, jobs=args.jobs, quiet=True, config=cfg
            )
        dets_a = load_dets(
            args.legacy_run / f"chip_{chip}" / "seg", load_meta(args.legacy_run / f"chip_{chip}" / "scan_10x")
        )
        dets_v = load_dets(seg_v2, load_meta(ident_chip / "scan_10x"))
        if args.v2_gates:
            n = v2_post_filter(dets_v)
            print(f"chip {chip}: post-filter demoted {n} T1 (grad_energy>{V2_GRAD_ENERGY_MAX} or B<{V2_B_MIN})")
        out: dict = {"chip": chip}
        compare_dets(f"chip{chip} legacy(committed) vs identity(v2) [floor check]", dets_a, dets_v, out)
        (args.out / f"chip{chip}_v2_compare.json").write_text(json.dumps(out, indent=1))


def run_b3d_stage(args: argparse.Namespace) -> None:
    """3D (R,G,B) curve distance on the identity raw-space segs: counterfactual tiers.

    Curve: R,G from the exactly-transformed empirical cal points; B from the
    raw-space pipeline model at the same thickness. Threshold is data-driven:
    p95 of the 3D distance of legacy-T1-keeps (dets that were T1 in both the
    committed legacy seg and the raw-config identity seg).
    """
    t_b, b_model = raw_model_b_curve()
    for chip in args.chips:
        legacy_chip = args.legacy_run / f"chip_{chip}"
        bg = median_bg_rgb(legacy_chip / "seg")
        raw_cfg, info = build_raw_hbn_medium(bg, quiet=True)
        pts = np.array(info["raw_cal_points"])  # (R, G, t)
        t_grid = np.linspace(pts[0, 2], pts[-1, 2], 600)
        curve = np.column_stack(
            [
                np.interp(t_grid, pts[:, 2], pts[:, 0]),
                np.interp(t_grid, pts[:, 2], pts[:, 1]),
                np.interp(t_grid, t_b, b_model),
            ]
        )

        def dist3(rgb: list[float]) -> tuple[float, float]:
            d = np.sqrt(((curve - np.array(rgb)) ** 2).sum(axis=1))
            j = int(d.argmin())
            return float(d[j]), float(t_grid[j])

        dets_a = load_dets(legacy_chip / "seg", load_meta(legacy_chip / "scan_10x"))
        ident_chip = args.identity_run / f"chip_{chip}"
        dets_c = load_dets(ident_chip / "seg_raw", load_meta(ident_chip / "scan_10x"))
        for d in dets_c:
            d["dist3"], d["t3"] = dist3(d["contrast_rgb"])
        pairs = match_by_stage(dets_a, dets_c)
        keeps = [c for a, c in pairs if a.get("tier", 3) == 1 and c.get("tier", 3) == 1]
        lifts = [c for a, c in pairs if a.get("tier", 3) == 3 and c.get("tier", 3) == 1]
        thr = float(np.percentile([c["dist3"] for c in keeps], 95))
        c_t1 = [d for d in dets_c if d.get("tier", 3) == 1]
        print(f"chip {chip}: 3D threshold (p95 of {len(keeps)} T1-keeps) = {thr:.3f}")
        print(f"   identity raw-config T1: {len(c_t1)}; pass 3D gate: {sum(d['dist3'] < thr for d in c_t1)}")
        print(
            f"   lifts (legacy T3 -> T1): {len(lifts)}; re-rejected by 3D: {sum(c['dist3'] >= thr for c in lifts)}"
            f" ({sum(c['dist3'] >= thr for c in lifts) / max(len(lifts), 1):.0%})"
        )
        # 3D-tier counterfactual over ALL identity dets: 3D gate + size/shape quality only
        # (perim + size from the preset; no 2D box, no entropy/bg_ratio demotions).
        t1_3d = [
            d
            for d in dets_c
            if d["dist3"] < thr
            and d["perim_ratio"] < raw_cfg.tier1_perim_ratio
            and d["size_um2"] >= raw_cfg.tier1_min_size_um2
            and d.get("aspect_ratio", 1.0) < raw_cfg.tier1_aspect_ratio
        ]
        print(f"   counterfactual 3D-T1 (3D gate + perim/size/aspect only): {len(t1_3d)}")
        (args.out / f"chip{chip}_b3d.json").write_text(
            json.dumps(
                {
                    "chip": chip,
                    "threshold": thr,
                    "t1_raw_config": len(c_t1),
                    "t1_3d": len(t1_3d),
                    "lifts_rerejected": sum(c["dist3"] >= thr for c in lifts),
                    "lifts_total": len(lifts),
                    "t1_3d_list": [
                        {
                            "frame": d["frame"],
                            "det_id": d["det_id"],
                            "score": d.get("score"),
                            "dist3": round(d["dist3"], 4),
                            "t3": round(d["t3"], 1),
                            "tier_2d": d.get("tier", 3),
                        }
                        for d in sorted(t1_3d, key=lambda x: -x.get("score", 0))
                    ],
                },
                indent=1,
            )
        )


def run_plot_b_stage(args: argparse.Namespace) -> None:
    """Blue-channel figure: B vs t with model + clip ceiling, mixed-vs-raw B, B-deviation split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_b, b_model = raw_model_b_curve()
    c_t1, c_t2, c_t3, c_model = "#d62728", "#ff9f40", "0.75", "#1f77b4"

    dets_c_all: list[dict] = []
    pairs_all: list[tuple[dict, dict]] = []
    bg_b = []
    for chip in args.chips:
        legacy_chip = args.legacy_run / f"chip_{chip}"
        ident_chip = args.identity_run / f"chip_{chip}"
        dets_a = load_dets(legacy_chip / "seg", load_meta(legacy_chip / "scan_10x"))
        dets_c = load_dets(ident_chip / "seg_raw", load_meta(ident_chip / "scan_10x"))
        for d in dets_c:
            d["chip"] = chip
        dets_c_all += dets_c
        pairs_all += match_by_stage(dets_a, dets_c)
        bg_b.append(median_bg_rgb(ident_chip / "seg_raw")[2])
    clip_ceiling = float((255.0 - np.mean(bg_b)) / np.mean(bg_b))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax = axes[0, 0]
    for tier, (c, s, z) in {3: (c_t3, 3, 1), 2: (c_t2, 10, 2), 1: (c_t1, 14, 3)}.items():
        sel = [d for d in dets_c_all if d.get("tier", 3) == tier and d.get("thickness_nm") is not None]
        ax.scatter(
            [d["thickness_nm"] for d in sel],
            [d["contrast_rgb"][2] for d in sel],
            s=s,
            c=c,
            zorder=z,
            alpha=0.6,
            label=f"tier {tier} (n={len(sel)})",
        )
    sel_t = (t_b >= 3) & (t_b <= 50)
    ax.plot(t_b[sel_t], b_model[sel_t], "-", color=c_model, lw=2, zorder=4, label="pipeline model B(t)")
    ax.axhline(clip_ceiling, color="k", ls="--", lw=1, alpha=0.6)
    ax.text(46, clip_ceiling + 0.04, f"8-bit clip ceiling ({clip_ceiling:.2f})", ha="right", fontsize=8)
    ax.set_xlim(0, 50)
    ax.set_xlabel("projected thickness (nm)")
    ax.set_ylabel("B contrast (identity space)")
    ax.set_title("Identity-space B vs thickness — data vs model")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    for tier, (c, s, z) in {3: (c_t3, 3, 1), 2: (c_t2, 10, 2), 1: (c_t1, 14, 3)}.items():
        pp = [(a, b) for a, b in pairs_all if b.get("tier", 3) == tier]
        ax.scatter(
            [a["contrast_rgb"][2] for a, b in pp],
            [b["contrast_rgb"][2] for a, b in pp],
            s=s,
            c=c,
            zorder=z,
            alpha=0.5,
        )
    lim = (-0.6, 2.0)
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("B contrast, legacy (mixed 5800K)")
    ax.set_ylabel("B contrast, identity (matched det)")
    ax.set_title("Same flake, both spaces (matched detections)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    for tier, (c, s, z) in {3: (c_t3, 3, 1), 2: (c_t2, 10, 2), 1: (c_t1, 14, 3)}.items():
        sel = [d for d in dets_c_all if d.get("tier", 3) == tier]
        ax.scatter(
            [d["contrast_rgb"][1] for d in sel],
            [d["contrast_rgb"][2] for d in sel],
            s=s,
            c=c,
            zorder=z,
            alpha=0.5,
        )
    g_curve_pts = []
    for chip in args.chips:
        _, info = build_raw_hbn_medium(median_bg_rgb(args.legacy_run / f"chip_{chip}" / "seg"), quiet=True)
        g_curve_pts = info["raw_cal_points"]
    g_arr = np.array(g_curve_pts)
    tt = np.linspace(g_arr[0, 2], g_arr[-1, 2], 300)
    ax.plot(
        np.interp(tt, g_arr[:, 2], g_arr[:, 1]),
        np.interp(tt, t_b, b_model),
        "-",
        color=c_model,
        lw=2,
        zorder=4,
        label="model (G, B)(t)",
    )
    ax.axhline(clip_ceiling, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("G contrast (identity space)")
    ax.set_ylabel("B contrast (identity space)")
    ax.set_title("G-B plane (identity space) with model locus")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]

    def bdevs(sel_pairs):
        out = []
        for _a, b in sel_pairs:
            t = b.get("thickness_nm")
            if t is not None:
                out.append(b["contrast_rgb"][2] - float(np.interp(t, t_b, b_model)))
        return np.array(out)

    keeps = bdevs([(a, b) for a, b in pairs_all if a.get("tier", 3) == 1 and b.get("tier", 3) == 1])
    lifts = bdevs([(a, b) for a, b in pairs_all if a.get("tier", 3) == 3 and b.get("tier", 3) == 1])
    bins = np.linspace(-0.5, 0.5, 41)
    ax.hist(keeps, bins=bins, color=c_t1, alpha=0.65, label=f"T1 keeps (n={len(keeps)})", density=True)
    ax.hist(lifts, bins=bins, color="#6a4c93", alpha=0.55, label=f"T3→T1 lifts (n={len(lifts)})", density=True)
    ax.axvline(0, color="k", lw=1, alpha=0.5)
    for x in (-0.15, 0.15):
        ax.axvline(x, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("B deviation from model B(t)")
    ax.set_ylabel("density")
    ax.set_title("B-deviation: legacy-kept T1 vs lifted-from-T3 (junk candidates)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.suptitle("Blue channel in identity space — paired scans run_20260807_1106 (chips 0+1)", fontsize=13)
    fig.tight_layout()
    out = args.out / "b_channel.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def run_segment_stage(args: argparse.Namespace) -> None:
    import segment_chip_scan

    ff_deccm = args.deccm_dir / "flatfield_10x_bin3_deccm.npy"
    for chip in args.chips:
        legacy_chip = args.legacy_run / f"chip_{chip}"
        bg = median_bg_rgb(legacy_chip / "seg")
        print(f"chip {chip}: legacy median bg RGB {bg.round(1)}")
        raw_cfg, info = build_raw_hbn_medium(bg)
        deccm_scan = args.deccm_dir / f"chip_{chip}" / "scan_10x"
        ident_scan = args.identity_run / f"chip_{chip}" / "scan_10x"
        seg_b = args.deccm_dir / f"chip_{chip}" / "seg_raw"
        seg_c = args.identity_run / f"chip_{chip}" / "seg_raw"
        if args.segment:
            print(f"== segment de-CCM chip {chip} (raw config)")
            segment_chip_scan.run(deccm_scan, seg_b, flatfield=ff_deccm, jobs=args.jobs, quiet=True, config=raw_cfg)
            print(f"== segment identity chip {chip} (raw config)")
            segment_chip_scan.run(ident_scan, seg_c, flatfield=ff_deccm, jobs=args.jobs, quiet=True, config=raw_cfg)

        legacy_meta = load_meta(legacy_chip / "scan_10x")
        dets_a = load_dets(legacy_chip / "seg", legacy_meta)  # scope-time baseline seg
        dets_b = load_dets(seg_b, load_meta(deccm_scan))
        dets_c = load_dets(seg_c, load_meta(ident_scan))
        out: dict = {"chip": chip, "raw_config": info}
        compare_dets(f"chip{chip} deccm(raw) vs identity(raw) [de-CCM parity]", dets_b, dets_c, out)
        compare_dets(f"chip{chip} legacy(committed) vs identity(raw) [migration]", dets_a, dets_c, out)
        (args.out / f"chip{chip}_det_compare.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired legacy-vs-identity de-CCM evaluation")
    parser.add_argument("legacy_run", type=Path)
    parser.add_argument("identity_run", type=Path)
    parser.add_argument("--deccm-dir", type=Path, help="De-CCM'd run dir (deccm_frames.py output)")
    parser.add_argument("--chips", type=int, nargs="+", required=True)
    parser.add_argument("--pair-only", action="store_true")
    parser.add_argument("--pixel", action="store_true")
    parser.add_argument(
        "--self-adjacent",
        action="store_true",
        help="Noise-floor control: diff adjacent-frame overlaps within legacy_run itself",
    )
    parser.add_argument("--segment", action="store_true", help="Run raw-config segmentations (b, c)")
    parser.add_argument("--det-compare", action="store_true", help="Compare detections (needs seg outputs)")
    parser.add_argument("--flatfield", type=Path, help="Flatfield .npy applied to both frames before pixel diff")
    parser.add_argument("--b3d", action="store_true", help="3D (R,G,B) curve-distance counterfactual on identity seg")
    parser.add_argument("--plot-b", action="store_true", help="Blue-channel figure (identity space vs model)")
    parser.add_argument("--v2", action="store_true", help="Segment identity scans with hbn_medium_v2 + floor check")
    parser.add_argument("--revisit", action="store_true", help="Paired 20x/50x revisit pixel comparison")
    parser.add_argument("--v2-gates", action="store_true", help="Apply v2 crap gates (grad_energy + B floor) post-tier")
    parser.add_argument("-o", "--out", type=Path, default=Path("/tmp/deccm_eval"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.plot_b:
        run_plot_b_stage(args)
        return
    if args.v2:
        if not args.deccm_dir:
            parser.error("--v2 requires --deccm-dir (for the de-CCM'd flatfield)")
        run_v2_stage(args)
        return
    if args.revisit:
        run_revisit_stage(args)
        return
    if args.b3d:
        run_b3d_stage(args)
        return
    if args.segment or args.det_compare:
        if not args.deccm_dir:
            parser.error("--segment/--det-compare require --deccm-dir")
        run_segment_stage(args)
        return
    px_um = 0.720703125  # 10x, bin 3 (matches seg params)
    for chip in args.chips:
        legacy = load_meta(args.legacy_run / f"chip_{chip}" / "scan_10x")
        ff_path = str(args.flatfield) if args.flatfield else None
        if args.self_adjacent:
            scan = args.legacy_run / f"chip_{chip}" / "scan_10x"
            stats, hist = run_pixel(
                scan, scan, adjacent_pairs(legacy), px_um, args.jobs, min_overlap=0.20, ff_path=ff_path
            )
            summary = summarize_pixel(chip, stats)
            (args.out / f"chip{chip}_selfadj_pixel.json").write_text(
                json.dumps({"summary": summary, "frames": stats}, indent=1)
            )
            continue
        identity = load_meta(args.identity_run / f"chip_{chip}" / "scan_10x")
        pairs, un_l, un_i = pair_frames(legacy, identity)
        rep = report_pairing(chip, pairs, un_l, un_i)
        (args.out / f"chip{chip}_pairing.json").write_text(
            json.dumps({**rep, "pairs": [p.to_dict() for p in pairs]}, indent=1)
        )
        if args.pair_only or not args.pixel:
            continue
        if not args.deccm_dir:
            parser.error("--pixel requires --deccm-dir")
        stats, hist = run_pixel(
            args.deccm_dir / f"chip_{chip}" / "scan_10x",
            args.identity_run / f"chip_{chip}" / "scan_10x",
            pairs,
            px_um,
            args.jobs,
            ff_path=ff_path,
        )
        summary = summarize_pixel(chip, stats)
        (args.out / f"chip{chip}_pixel.json").write_text(json.dumps({"summary": summary, "frames": stats}, indent=1))
        np.save(args.out / f"chip{chip}_value_delta_hist.npy", hist)
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
