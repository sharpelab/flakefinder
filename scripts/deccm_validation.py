"""De-CCM validation: does identity-space segmentation reproduce mixed-space behavior?

Pre-registration for the identity-colour-matrix cutover (see
scripts/deccm_frames.py). Pipeline:

1. Segment the ORIGINAL frames with the committed mixed-space preset
   (baseline; also sanity-checks local seg against the scope-time summary).
2. Segment the DE-CCM'd frames with a raw-space config: same preset, cal
   points regenerated from the validated pipeline model minus the CCM/WB
   mix step, contrast-space gates scaled by the model's band arc-length
   ratio (raw space compresses contrast, so distance gates shrink with it).
3. Match detections frame-by-frame by center position and compare: tier
   confusion, thickness agreement, contrast scatter, and the AFM anchor
   flakes' positions in the raw-space R/G plane.

Usage:
    uv run python scripts/deccm_validation.py scans/run_20260804_1354/chip_5 \
        --deccm-dir scans/run_20260804_1354_deccm/chip_5 [--segment] [--plot out.png]

--segment runs the two segmentations (skip to reuse existing seg output).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import segment_chip_scan
from colorchecker_cal import derive_pipeline
from hbn_contrast import _IMX183_BLUE, _IMX183_GREEN, _IMX183_RED, _IMX183_WAVELENGTHS
from hbn_contrast_widget import compute_rg

from flakefinder.data_utils import add_stage_coords
from flakefinder.segmentation import CalPointRG, DetectorConfig

N_HBN_V4 = 2.060  # pipeline-model fit vs Toghrul AFM anchors (offsets = 0)
OXIDE_NM = 90.0
NA_10X = 0.25
BAND_NM = (40, 110)  # match HBN_THICK_50_100_90NM_CAL_POINTS extent, 2 nm steps
MATCH_RADIUS_PX = 15.0

# AFM anchors: flake_id -> (afm_mid_nm, stage_x, stage_y) filled from flakes_meta.
# downloads/ is a local (gitignored) dir — when running from a worktree at
# <main>/.worktrees/<name>, fall back to the main repo's copy.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FLAKES_META = next(
    p
    for p in (
        _REPO_ROOT / "downloads" / "flakes_scan287" / "flakes_meta.json",
        _REPO_ROOT.parents[1] / "downloads" / "flakes_scan287" / "flakes_meta.json",
    )
    if p.exists()
)


def model_points(mix: bool, n_hbn: float = N_HBN_V4) -> list[tuple[int, float, float, float]]:
    """(layers, r, g, t_nm) along the band arc from the v4 pipeline chain.

    mix=True reproduces the mixed-space v4 model (CCM @ hBN scan WB);
    mix=False is the same physical chain in raw channel space.
    """
    pipe = derive_pipeline(_IMX183_WAVELENGTHS)
    op = pipe.objectives["10x"]
    mix_matrix = None
    if mix:
        from colorchecker_cal import WB_REQUESTS_HBN_SCAN

        mix_matrix = pipe.ccm @ np.diag(WB_REQUESTS_HBN_SCAN)
    r_all, g_all, _t_all = compute_rg(
        complex(n_hbn),
        OXIDE_NM,
        NA_10X,
        max_layers=340,
        red_lit=_IMX183_RED * op.chain_spd,
        green_lit=_IMX183_GREEN * op.chain_spd,
        blue_lit=_IMX183_BLUE * op.chain_spd,
        glare_f=np.array(op.glare_f_raw),
        mix=mix_matrix,
    )
    points = []
    for t_nm in range(BAND_NM[0], BAND_NM[1] + 1, 2):
        layers = round(t_nm / 0.333)
        idx = layers - 1
        points.append((layers, float(r_all[idx]), float(g_all[idx]), t_nm))
    return points


def build_raw_config(n_raw: float = N_HBN_V4) -> tuple[DetectorConfig, float]:
    """Raw-space variant of hbn_thick_50_100_90nm + the contrast scale factor."""
    base = DetectorConfig.from_material("hbn_thick_50_100_90nm")

    raw_pts = model_points(mix=False, n_hbn=n_raw)
    mixed_pts = model_points(mix=True)

    # Contrast-space scale: band arc length ratio raw/mixed. Distance-like
    # gates (cal_dist, contrast_offset, subseg thresholds) scale with it.
    def arc_len(pts):
        a = np.array([(p[1], p[2]) for p in pts])
        return float(np.sum(np.hypot(*np.diff(a, axis=0).T)))

    scale = arc_len(raw_pts) / arc_len(mixed_pts)

    band = [(r, g) for _, r, g, t in raw_pts if 50.0 <= t <= 100.0]
    rs, gs = np.array([p[0] for p in band]), np.array([p[1] for p in band])
    pad = base.cal_dist_match * scale

    cfg = replace(
        base,
        cal_reference_points=tuple(CalPointRG(layers=lay, r=round(r, 4), g=round(g, 4)) for lay, r, g, _ in raw_pts),
        contrast_offset=base.contrast_offset * scale,
        subseg_min_std=base.subseg_min_std * scale,
        subseg_min_range=base.subseg_min_range * scale,
        cal_dist_match=base.cal_dist_match * scale,
        cal_dist_possible=base.cal_dist_possible * scale,
        tier1_cal_dist=base.tier1_cal_dist * scale,
        tier2_cal_dist=base.tier2_cal_dist * scale,
        tier1_g_min=float(gs.min() - pad),
        tier1_g_max=float(gs.max() + pad),
        tier1_r_min=float(rs.min() - pad),
        tier1_r_max=float(rs.max() + pad),
    )
    return cfg, scale


def load_summary(seg_dir: Path) -> list[dict]:
    s = json.loads((seg_dir / "summary.json").read_text())
    dets = []
    for frame, frame_dets in s["detections_by_frame"].items():
        for d in frame_dets:
            d["frame"] = frame
            dets.append(d)
    return dets


def match_detections(a: list[dict], b: list[dict]) -> list[tuple[dict, dict]]:
    """Greedy per-frame center matching within MATCH_RADIUS_PX."""
    by_frame_b: dict[str, list[dict]] = {}
    for d in b:
        by_frame_b.setdefault(d["frame"], []).append(d)
    pairs = []
    for d in a:
        cands = by_frame_b.get(d["frame"], [])
        if not cands:
            continue
        ax, ay = d["center"]
        dists = [np.hypot(ax - c["center"][0], ay - c["center"][1]) for c in cands]
        j = int(np.argmin(dists))
        if dists[j] <= MATCH_RADIUS_PX:
            pairs.append((d, cands[j]))
            cands.pop(j)
    return pairs


def anchor_records() -> list[dict]:
    """The 9 AFM anchors: id, afm_mid_nm, stage um, mixed-space (R, G)."""
    d = json.loads(_FLAKES_META.read_text())
    flakes = d if isinstance(d, list) else d.get("flakes", [])
    out = []
    for f in flakes:
        note = f.get("flake_note") or ""
        if not note.strip():
            continue
        nums = [float(x) for x in note.replace("nm", "").replace("-", " ").split() if x.replace(".", "").isdigit()]
        mid = float(np.mean(nums[:2])) if nums else float("nan")
        out.append(
            {
                "flake_id": f["flake_id"],
                "afm_nm": mid,
                # Website positions are plain stage mm; detections carry stage um.
                "x_um": f["flake_position_x"] * 1000.0,
                "y_um": f["flake_position_y"] * 1000.0,
                "r": f["flake_mean_r"],
                "g": f["flake_mean_g"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate de-CCM'd segmentation against mixed-space baseline")
    parser.add_argument("chip_dir", type=Path, help="Original chip dir (with scan_10x/)")
    parser.add_argument("--deccm-dir", type=Path, required=True, help="De-CCM'd chip dir (with scan_10x/)")
    parser.add_argument("--deccm-flatfield", type=Path, required=True, help="De-CCM'd flatfield .npy")
    parser.add_argument("--segment", action="store_true", help="Run both segmentations (else reuse seg output)")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--plot", type=Path, help="Save R/G-plane comparison figure")
    parser.add_argument(
        "--n-raw",
        type=float,
        default=N_HBN_V4,
        help="hBN n for the raw-space cal table (fit it against de-CCM'd anchors on a second pass)",
    )
    args = parser.parse_args()

    raw_cfg, scale = build_raw_config(args.n_raw)
    print(f"contrast scale (raw/mixed band arc length): {scale:.4f}")
    print(
        f"raw-space gates: contrast_offset {raw_cfg.contrast_offset:.2f}, cal_dist match {raw_cfg.cal_dist_match:.3f}"
    )
    print(
        f"raw-space band box: G [{raw_cfg.tier1_g_min:.3f}, {raw_cfg.tier1_g_max:.3f}]"
        f"  R [{raw_cfg.tier1_r_min:.3f}, {raw_cfg.tier1_r_max:.3f}]"
    )

    base_seg = args.chip_dir / "seg_baseline"
    deccm_seg = args.deccm_dir / "seg_raw"
    if args.segment:
        print("== segment baseline (original frames, committed preset)")
        segment_chip_scan.run(
            args.chip_dir / "scan_10x", base_seg, material="hbn_thick_50_100_90nm", jobs=args.jobs, quiet=True
        )
        print("== segment de-CCM (raw frames, raw-space config)")
        segment_chip_scan.run(
            args.deccm_dir / "scan_10x",
            deccm_seg,
            flatfield=args.deccm_flatfield,
            jobs=args.jobs,
            quiet=True,
            config=raw_cfg,
        )

    dets_a = load_summary(base_seg)
    dets_b = load_summary(deccm_seg)

    def tiers(dets):
        t = {1: 0, 2: 0, 3: 0}
        for d in dets:
            t[d.get("tier", 3)] += 1
        return t

    stored = json.loads((args.chip_dir / "seg" / "summary.json").read_text())["stats"]
    print(
        f"scope-time summary: total {stored['total_detections']},"
        f" tiers {stored['tier_1']}/{stored['tier_2']}/{stored['tier_3']}"
    )
    ta, tb = tiers(dets_a), tiers(dets_b)
    print(f"baseline (local):   total {len(dets_a)}, tiers {ta[1]}/{ta[2]}/{ta[3]}")
    print(f"de-CCM raw-space:   total {len(dets_b)}, tiers {tb[1]}/{tb[2]}/{tb[3]}")

    pairs = match_detections(dets_a, dets_b)
    print(f"matched pairs (<= {MATCH_RADIUS_PX:.0f} px): {len(pairs)} of {len(dets_a)} baseline")

    conf: dict[tuple[int, int], int] = {}
    dt = []
    dt_t12 = []
    for a, b in pairs:
        conf[(a.get("tier", 3), b.get("tier", 3))] = conf.get((a.get("tier", 3), b.get("tier", 3)), 0) + 1
        if a.get("thickness_nm") is not None and b.get("thickness_nm") is not None:
            delta = b["thickness_nm"] - a["thickness_nm"]
            dt.append(delta)
            if a.get("tier", 3) <= 2 and b.get("tier", 3) <= 2:
                dt_t12.append(delta)
    print("tier confusion (baseline -> deccm):")
    for (i, j), n in sorted(conf.items()):
        print(f"   {i} -> {j}: {n}")
    if dt:
        dt_arr = np.array(dt)
        print(
            f"thickness delta (deccm - baseline): mean {dt_arr.mean():+.2f} nm,"
            f" rms {np.sqrt((dt_arr**2).mean()):.2f} nm, n={len(dt_arr)}"
        )
    if dt_t12:
        t12 = np.array(dt_t12)
        print(
            f"thickness delta, both tier<=2: mean {t12.mean():+.2f} nm,"
            f" rms {np.sqrt((t12**2).mean()):.2f} nm, median |d| {np.median(np.abs(t12)):.2f} nm, n={len(t12)}"
        )

    # AFM anchors: locate in the BASELINE seg (position within 500 um of the
    # mirror-corrected website coords + best mixed-space contrast match),
    # then follow the pair mapping to the raw-space partner.
    meta = json.loads((args.chip_dir / "scan_10x" / "scan_meta.json").read_text())
    add_stage_coords(dets_a, meta)  # ty: ignore[invalid-argument-type]  # summary dicts are Detection-shaped
    partner = {id(a): b for a, b in pairs}
    print("AFM anchors (baseline located by position+contrast; raw values via pair match):")
    print(
        f"   {'flake':>6} {'afm nm':>7} {'(R, G) mixed':>18} {'(R, G) raw':>18} {'t_mix':>6} {'t_raw':>6} {'tier':>4}"
    )
    for a_rec in anchor_records():
        near = [
            d
            for d in dets_a
            if "stage_x" in d and np.hypot(d["stage_x"] - a_rec["x_um"], d["stage_y"] - a_rec["y_um"]) < 500
        ]
        if not near:
            print(f"   {a_rec['flake_id']:>6} {a_rec['afm_nm']:>7.1f}  -- no baseline detection within 500 um")
            continue
        cd = [np.hypot(d["contrast_rgb"][0] - a_rec["r"], d["contrast_rgb"][1] - a_rec["g"]) for d in near]
        j = int(np.argmin(cd))
        if cd[j] > 0.15:
            print(f"   {a_rec['flake_id']:>6} {a_rec['afm_nm']:>7.1f}  -- no contrast match (best d={cd[j]:.2f})")
            continue
        a = near[j]
        b = partner.get(id(a))
        rg_m = f"({a['contrast_rgb'][0]:+.3f}, {a['contrast_rgb'][1]:+.3f})"
        if b is None:
            print(f"   {a_rec['flake_id']:>6} {a_rec['afm_nm']:>7.1f} {rg_m:>18} {'-- unmatched --':>18}")
            continue
        rg_r = f"({b['contrast_rgb'][0]:+.3f}, {b['contrast_rgb'][1]:+.3f})"
        print(
            f"   {a_rec['flake_id']:>6} {a_rec['afm_nm']:>7.1f} {rg_m:>18} {rg_r:>18}"
            f" {a.get('thickness_nm')!s:>6} {b.get('thickness_nm')!s:>6} {b.get('tier', 3):>4}"
        )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_m, ax_r) = plt.subplots(1, 2, figsize=(14, 7))
        for ax, dets, pts, title in (
            (ax_m, dets_a, model_points(mix=True), "Mixed space (baseline seg, committed preset)"),
            (ax_r, dets_b, model_points(mix=False, n_hbn=args.n_raw), "Raw space (de-CCM seg, raw config)"),
        ):
            t123 = {1: ("tab:red", 12), 2: ("tab:orange", 7), 3: ("0.7", 3)}
            for tier, (c, s) in t123.items():
                sel = [d for d in dets if d.get("tier", 3) == tier]
                ax.scatter(
                    [d["contrast_rgb"][1] for d in sel],
                    [d["contrast_rgb"][0] for d in sel],
                    s=s,
                    c=c,
                    label=f"tier {tier} ({len(sel)})",
                    alpha=0.6,
                )
            arc = np.array([(p[2], p[1]) for p in pts])
            ax.plot(arc[:, 0], arc[:, 1], "b-", lw=1.5, label="model arc 40-110 nm")
            ax.set_xlabel("Green contrast")
            ax.set_ylabel("Red contrast")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"De-CCM validation — chip 5, run_20260804_1354 (scale {scale:.3f})")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
