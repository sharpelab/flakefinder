"""Sweep LOO-fit-RMSE-ratio outlier detection across many focus map JSONs.

For each focus map, replay the prefilter + iterative drop-and-refit process.
At each iteration, compute the "improvement ratio" for the best candidate
(baseline_rmse / rmse_with_candidate_dropped) and the gap to the next-best
(rmse_2nd / rmse_best). Aggregate stats so we can pick a sensible threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def fit_rmse(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> float:
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    c, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
    res = zs - A @ c
    return float(np.sqrt(np.mean(res**2)))


def prefilter(pts: list[dict], reject_sharpness: float = 10.0, min_dr: float = 0.05) -> np.ndarray:
    n = len(pts)
    sharp = np.array([p["selected"]["sharpness"] for p in pts])
    dr = np.array([p.get("dynamic_range", 0) for p in pts])
    mi = np.array([p.get("mean_intensity", float("inf")) for p in pts])
    z = np.array([p["selected"]["z_um"] for p in pts])
    coarse = np.array([p.get("coarse", {}).get("best_z_um", p["selected"]["z_um"]) for p in pts])
    rd = np.abs(coarse - z)
    pne = np.array([p.get("peak_near_edge", False) for p in pts])
    mask = (sharp >= reject_sharpness) & (dr >= min_dr) & (mi >= 5.0) & (rd > 0) & ~(pne & (dr < min_dr))
    if mask.sum() < 3:
        mask = np.ones(n, dtype=bool)
    return mask


def replay_loo(
    pts: list[dict],
    ratio_threshold: float,
    min_baseline_rmse: float,
    min_drop_floor: float,
    min_points_after: int = 4,
) -> list[dict]:
    """Replay iterative LOO-fit-improvement rejection. Returns per-iteration log."""
    mask = prefilter(pts)
    x = np.array([p["x_um"] for p in pts])
    y = np.array([p["y_um"] for p in pts])
    z = np.array([p["selected"]["z_um"] for p in pts])

    log = []
    iteration = 0
    while True:
        idxs = np.where(mask)[0]
        n = len(idxs)
        if n <= min_points_after:
            log.append({"iter": iteration, "n": n, "stop": "min_points"})
            break
        xs, ys, zs = x[mask], y[mask], z[mask]
        baseline = fit_rmse(xs, ys, zs)

        rmse_drops = np.zeros(n)
        for j in range(n):
            m = np.ones(n, bool)
            m[j] = False
            rmse_drops[j] = fit_rmse(xs[m], ys[m], zs[m])

        order = np.argsort(rmse_drops)
        best_j = int(order[0])
        rmse_best = rmse_drops[best_j]
        rmse_2nd = rmse_drops[order[1]] if n > 1 else float("inf")

        ratio = baseline / rmse_best if rmse_best > 0 else float("inf")
        gap = rmse_2nd / rmse_best if rmse_best > 0 else float("inf")

        entry = {
            "iter": iteration,
            "n": n,
            "baseline_rmse": baseline,
            "best_drop_idx": int(idxs[best_j]),
            "rmse_best_drop": float(rmse_best),
            "rmse_2nd_drop": float(rmse_2nd),
            "ratio": float(ratio),
            "gap": float(gap),
        }

        # Reject criteria:
        # (a) ratio big: baseline / rmse_best > K
        # (b) baseline must exceed a noise floor (don't chase already-clean fits)
        # (c) the resulting RMSE must be plausibly tight (drop must "land")
        fires = ratio > ratio_threshold and baseline > min_baseline_rmse and rmse_best < min_drop_floor

        if fires and (n - 1) >= min_points_after:
            entry["action"] = "reject"
            log.append(entry)
            mask[idxs[best_j]] = False
            iteration += 1
            continue
        else:
            entry["action"] = "stop"
            entry["stop"] = (
                "ratio_too_low"
                if ratio <= ratio_threshold
                else "baseline_below_floor"
                if baseline <= min_baseline_rmse
                else "drop_rmse_too_high"
                if rmse_best >= min_drop_floor
                else "min_points"
            )
            log.append(entry)
            break

    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="focus_map_*.json files or dirs to glob")
    ap.add_argument("--ratio", type=float, default=3.0, help="rejection ratio threshold")
    ap.add_argument("--min-baseline", type=float, default=5.0, help="don't reject if baseline RMSE already < this (µm)")
    ap.add_argument(
        "--min-drop-floor", type=float, default=20.0, help="resulting RMSE must be < this to count as a real fix (µm)"
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.rglob("focus_map_chip*.json")))
        elif pp.is_file():
            files.append(pp)
        else:
            files.extend(sorted(Path().glob(p)))

    files = [f for f in files if "_plane" not in f.name and "_meta" not in f.name]
    print(f"Sweeping {len(files)} focus maps")
    print(
        f"Params: ratio_threshold={args.ratio}, min_baseline={args.min_baseline}, min_drop_floor={args.min_drop_floor}"
    )
    print()

    all_iter_entries: list[dict] = []
    rejection_stats: list[tuple[str, int, int, float, float]] = []
    # (file, n_initial, n_rejected, baseline_initial, baseline_final)

    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        pts = data.get("sample_points")
        if not pts or len(pts) < 4:
            continue
        log = replay_loo(
            pts,
            ratio_threshold=args.ratio,
            min_baseline_rmse=args.min_baseline,
            min_drop_floor=args.min_drop_floor,
        )
        first = log[0]
        n_initial = first["n"]
        baseline_initial = first.get("baseline_rmse", float("nan"))
        n_rejected = sum(1 for e in log if e.get("action") == "reject")
        # final fit baseline (last entry has n after rejections)
        last = log[-1]
        baseline_final = last.get("baseline_rmse", baseline_initial)

        rejection_stats.append((str(f), n_initial, n_rejected, baseline_initial, baseline_final))
        for e in log:
            if "ratio" in e:
                all_iter_entries.append({**e, "file": str(f)})

        if args.verbose:
            print(
                f"{f.parent.parent.name}/{f.parent.name}: "
                f"n={n_initial}, dropped {n_rejected}, "
                f"RMSE {baseline_initial:.2f} -> {baseline_final:.2f} µm"
            )
            for e in log:
                if "ratio" in e:
                    print(
                        f"    iter{e['iter']}: drop c{e['best_drop_idx']} "
                        f"ratio={e['ratio']:.2f} gap={e['gap']:.2f} "
                        f"base={e['baseline_rmse']:.2f} "
                        f"drop_rmse={e['rmse_best_drop']:.2f} -> {e['action']}"
                    )

    # Aggregate stats
    print()
    print("=" * 70)
    print("PER-ITERATION RATIO DISTRIBUTION (sorted, all candidates considered):")
    print("=" * 70)
    if all_iter_entries:
        ratios = np.array([e["ratio"] for e in all_iter_entries])
        gaps = np.array([e["gap"] for e in all_iter_entries])
        baselines = np.array([e["baseline_rmse"] for e in all_iter_entries])
        drop_rmses = np.array([e["rmse_best_drop"] for e in all_iter_entries])

        print(
            f"  ratio  : min={ratios.min():.2f} p25={np.percentile(ratios, 25):.2f} "
            f"med={np.median(ratios):.2f} p75={np.percentile(ratios, 75):.2f} "
            f"p90={np.percentile(ratios, 90):.2f} p95={np.percentile(ratios, 95):.2f} "
            f"max={ratios.max():.2f}"
        )
        print(
            f"  gap    : min={gaps.min():.2f} p25={np.percentile(gaps, 25):.2f} "
            f"med={np.median(gaps):.2f} p75={np.percentile(gaps, 75):.2f} "
            f"p90={np.percentile(gaps, 90):.2f} p95={np.percentile(gaps, 95):.2f} "
            f"max={gaps.max():.2f}"
        )
        print(
            f"  base   : min={baselines.min():.2f} med={np.median(baselines):.2f} "
            f"p90={np.percentile(baselines, 90):.2f} max={baselines.max():.2f}"
        )
        print(
            f"  drop_r : min={drop_rmses.min():.2f} med={np.median(drop_rmses):.2f} "
            f"p90={np.percentile(drop_rmses, 90):.2f} max={drop_rmses.max():.2f}"
        )

        # Histogram of ratios bucketed
        print()
        print("Ratio histogram (across all evaluated iterations):")
        edges = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1e9]
        labels = ["1-1.2", "1.2-1.5", "1.5-2", "2-3", "3-5", "5-10", "10-20", "20-50", ">50"]
        counts = np.zeros(len(labels), dtype=int)
        for r in ratios:
            for i in range(len(edges) - 1):
                if edges[i] <= r < edges[i + 1]:
                    counts[i] += 1
                    break
        for lab, c in zip(labels, counts, strict=True):
            bar = "#" * min(60, c)
            print(f"  {lab:>8}: {c:4d} {bar}")

    # Per-file summary
    print()
    print("=" * 70)
    print("PER-FILE OUTCOMES (with current thresholds):")
    print("=" * 70)
    n_no_drops = sum(1 for _, _, nr, _, _ in rejection_stats if nr == 0)
    n_one_drop = sum(1 for _, _, nr, _, _ in rejection_stats if nr == 1)
    n_multi_drop = sum(1 for _, _, nr, _, _ in rejection_stats if nr >= 2)
    print(f"  Total files:        {len(rejection_stats)}")
    print(f"  No drops:           {n_no_drops}")
    print(f"  1 drop:             {n_one_drop}")
    print(f"  ≥2 drops:           {n_multi_drop}")
    print()
    print("Files with ≥1 drop (ranked by initial RMSE):")
    drops = [(f, ni, nr, bi, bf) for (f, ni, nr, bi, bf) in rejection_stats if nr > 0]
    for f, ni, nr, bi, bf in sorted(drops, key=lambda r: -r[3]):
        short = "/".join(Path(f).parts[-3:-1])
        print(f"  {short:30s}  n={ni:2d} dropped={nr}  RMSE {bi:7.2f} -> {bf:6.2f} µm")


if __name__ == "__main__":
    main()
