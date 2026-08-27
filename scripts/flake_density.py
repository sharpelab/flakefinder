"""Flake yield per unit chip area, joined from the flakes server and run dirs.

Detection counts come from flakes.sharpelab.science; chip areas come from the
overview chip-detection JSON in each run directory (convex-hull area, the
measured chip surface). Runs are matched to server scans by operator and
upload time.

Usage:
    uv run python scripts/flake_density.py --runs scans
    uv run python scripts/flake_density.py --runs scans --material Graphene --by user
    uv run python scripts/flake_density.py --runs scans --since 2026-08-01 --by scan
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

from flakefinder.flakes_api import api_get

RUN_RE = re.compile(r"run_(\d{8})_(\d{4})")
# Run directories are stamped in the microscope PC's local (Pacific) time; the
# server records absolute upload time. Upload trails the run start by the scan
# duration, so the window is generous and matching is resolved one-to-one.
SCOPE_TZ = ZoneInfo("America/Los_Angeles")
MATCH_WINDOW_S = 6 * 3600


def run_local_time(run_dir: Path) -> dt.datetime | None:
    m = RUN_RE.search(run_dir.name)
    if not m:
        return None
    naive = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
    return naive.replace(tzinfo=SCOPE_TZ)


def load_runs(roots: list[Path]) -> list[dict]:
    """Collect (time, operator, material, git_version, chip areas) per run dir."""
    runs = []
    for root in roots:
        for path in sorted(glob.glob(str(root / "run_*"))):
            run_dir = Path(path)
            ckpt = run_dir / "checkpoint.json"
            if not ckpt.exists():
                continue
            with open(ckpt) as f:
                ck = json.load(f)
            chips_files = sorted(glob.glob(str(run_dir / "overview_*_stitch_chips.json")))
            if not chips_files:
                continue
            with open(chips_files[0]) as f:
                chips = json.load(f).get("chips", [])
            args = ck.get("args", {})
            seg = {"total_detections": 0, "tier_1": 0, "tier_2": 0, "tier_3": 0}
            seg_chips = 0
            for summary in sorted(run_dir.glob("chip_*/seg/summary.json")):
                with open(summary) as f:
                    stats = json.load(f).get("stats", {})
                if not stats or "error" in stats:
                    continue
                seg_chips += 1
                for k in seg:
                    seg[k] += stats.get(k, 0)
            runs.append(
                {
                    "run": run_dir.name,
                    "seg": seg if seg_chips else None,
                    "time": run_local_time(run_dir),
                    "operator": (ck.get("operator") or "").lower(),
                    "material": args.get("material"),
                    "git_version": args.get("git_version"),
                    "areas_mm2": [c["area_um2"] / 1e6 for c in chips],
                }
            )
    return runs


def match_runs_to_scans(scans: list[dict], runs: list[dict]) -> dict[str, dict]:
    """One-to-one run/scan assignment: same operator, nearest in time.

    A scan is claimed by at most one run, so sibling runs on the same day
    cannot collapse onto a single scan. Upload must follow the run start.
    """
    candidates = []
    for run in runs:
        if run["time"] is None:
            continue
        for scan in scans:
            if (scan.get("scan_user") or "").lower() != run["operator"]:
                continue
            delta = scan["scan_time"] - run["time"].timestamp()
            if 0 <= delta <= MATCH_WINDOW_S:
                candidates.append((delta, run["run"], scan))
    candidates.sort(key=lambda c: c[0])
    matched: dict[str, dict] = {}
    claimed: set[int] = set()
    for _, run_name, scan in candidates:
        if run_name in matched or scan["scan_id"] in claimed:
            continue
        matched[run_name] = scan
        claimed.add(scan["scan_id"])
    return matched


# Categorical slots 1 and 2 of the validated default palette (all-pairs clean).
SERIES_COLOURS = ("#2a78d6", "#eb6834")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"
# The 2026-08-13 illumination incident (docs/illum_incident_20260813.md).
INCIDENT = dt.datetime(2026, 8, 13, tzinfo=SCOPE_TZ)


def plot_density(rows: list[dict], dest: Path, ylabel: str, title: str) -> None:
    """Detection density over time, one panel per material."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    materials = sorted({str(r["material"]) for r in rows})
    users = sorted({r["user"] for r in rows})
    colours = {u: SERIES_COLOURS[i % len(SERIES_COLOURS)] for i, u in enumerate(users)}

    fig, axes = plt.subplots(
        len(materials), 1, figsize=(11, 3.1 * len(materials)), sharex=True, sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    ymax = max(r["density"] for r in rows) * 1.18

    for ax, material in zip(axes[:, 0], materials, strict=True):
        ax.set_facecolor(SURFACE)
        ax.axvline(INCIDENT, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for user in users:
            pts = [r for r in rows if r["user"] == user and str(r["material"]) == material]
            if not pts:
                continue
            ax.plot(
                [r["date"] for r in pts],
                [r["density"] for r in pts],
                marker="o",
                markersize=8,
                lw=2,
                color=colours[user],
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                label=user,
                zorder=3,
            )
        ax.set_title(material, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)

    top = axes[0, 0]
    top.annotate(
        "illumination incident 8/13",
        xy=(INCIDENT, ymax * 0.94),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
        va="top",
    )
    top.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", ncol=len(users))

    bottom = axes[-1, 0]
    bottom.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))

    fig.suptitle(
        title,
        x=0.008,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_tiers(rows: list[dict], dest: Path) -> None:
    """Tier composition over time: junk rate, keep rate, and the ratio between them.

    A classifier that is mis-tiering real flakes shows a rising tier-3 rate and a
    falling tier-1 share at the same time; a genuinely barren chip drops both.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("tiers")]
    if not rows:
        raise SystemExit("no rows carry a seg tier breakdown")
    users = sorted({r["user"] for r in rows})
    colours = {u: SERIES_COLOURS[i % len(SERIES_COLOURS)] for i, u in enumerate(users)}

    def t1_share(r: dict) -> float:
        return 100 * r["tiers"]["tier_1"] / r["n_flakes"] if r["n_flakes"] else 0.0

    panels = [
        ("tier-3 detections / mm²", lambda r: r["tiers"]["tier_3"] / r["area_mm2"]),
        ("tier-1 detections / mm²", lambda r: r["tiers"]["tier_1"] / r["area_mm2"]),
        ("tier-1 share of all detections (%)", t1_share),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.9 * len(panels)), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (label, fn) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ax.axvline(INCIDENT, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for user in users:
            pts = [r for r in rows if r["user"] == user]
            if not pts:
                continue
            ax.plot(
                [r["date"] for r in pts],
                [fn(r) for r in pts],
                marker="o",
                markersize=8,
                lw=2,
                color=colours[user],
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                label=user,
                zorder=3,
            )
        ax.set_title(label, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    axes[0].annotate(
        "illumination incident 8/13",
        xy=(INCIDENT, axes[0].get_ylim()[1] * 0.94),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
        va="top",
    )
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", ncol=len(users))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))

    fig.suptitle("Detection tier composition per scan, by operator", x=0.008, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", default=[Path("scans")], help="Run-dir roots")
    ap.add_argument("--material", default=None, help="Substring filter on server scan_material (e.g. Graph)")
    ap.add_argument("--since", default=None, help="Only runs on/after this date (YYYY-MM-DD)")
    ap.add_argument("--by", choices=["scan", "user", "material", "week"], default="scan")
    ap.add_argument("--tier", type=int, default=None, help="Count only flakes at this tier")
    ap.add_argument(
        "--counts",
        choices=["server", "seg"],
        default="server",
        help=(
            "server: tier-1 flakes uploaded to flakes.sharpelab.science (default). "
            "seg: raw counts from each run's seg summary, which include every tier."
        ),
    )
    ap.add_argument("--users", nargs="+", default=None, help="Only these server users (case-insensitive)")
    ap.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Fold one user label into another (repeatable), e.g. --alias benalex=Ben",
    )
    ap.add_argument("--plot", type=Path, default=None, help="Write a density-over-time chart to this path")
    ap.add_argument(
        "--plot-tiers",
        type=Path,
        default=None,
        help="Write a tier-composition chart (needs --counts seg): tier-3 rate, tier-1 rate, tier-1 share",
    )
    args = ap.parse_args()

    aliases = {}
    for spec in args.alias:
        src, _, dst = spec.partition("=")
        aliases[src.strip().lower()] = dst.strip()

    since = dt.datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=SCOPE_TZ) if args.since else None

    scans = api_get("scans")
    assert isinstance(scans, list)
    runs = load_runs(args.runs)

    matched = match_runs_to_scans(scans, runs)
    rows = []
    for run in runs:
        if since and (run["time"] is None or run["time"] < since):
            continue
        scan = matched.get(run["run"])
        if scan is None:
            continue
        if args.material and args.material.lower() not in (scan.get("scan_material") or "").lower():
            continue
        if args.counts == "seg":
            if run["seg"] is None:
                continue
            key = f"tier_{args.tier}" if args.tier is not None else "total_detections"
            n_flakes = run["seg"][key]
            n_tier1 = run["seg"]["tier_1"]
            tiers = run["seg"]
        else:
            flakes = api_get("flakes", {"scan_id": scan["scan_id"]})
            assert isinstance(flakes, list)
            if args.tier is not None:
                flakes = [f for f in flakes if f.get("flake_tier") == args.tier]
            n_flakes = len(flakes)
            n_tier1 = sum(1 for f in flakes if f.get("flake_tier") == 1)
            tiers = None
        area = sum(run["areas_mm2"])
        user = scan.get("scan_user") or run["operator"]
        user = aliases.get(user.lower(), user)
        if args.users and user.lower() not in {u.lower() for u in args.users}:
            continue
        rows.append(
            {
                "run": run["run"],
                "date": run["time"],
                "user": user,
                "material": scan.get("scan_material"),
                "preset": run["material"],
                "git": run["git_version"],
                "scan_id": scan["scan_id"],
                "n_chips": len(run["areas_mm2"]),
                "area_mm2": area,
                "n_flakes": n_flakes,
                "n_tier1": n_tier1,
                "tiers": tiers,
                "density": n_flakes / area if area else float("nan"),
                "density_t1": n_tier1 / area if area else float("nan"),
            }
        )

    rows.sort(key=lambda r: r["date"])

    if args.plot is not None:
        if args.counts == "seg":
            what = f"tier-{args.tier}" if args.tier is not None else "all-tier"
            ylabel, title = f"{what} detections / mm²", f"Segmentation {what} detection density per scan, by operator"
        else:
            ylabel, title = "tier-1 flakes / mm²", "Tier-1 flake density per scan, by operator"
        plot_density(rows, args.plot, ylabel, title)
        print(f"wrote {args.plot}")

    if args.plot_tiers is not None:
        if args.counts != "seg":
            print("--plot-tiers needs --counts seg (tier 2/3 are never uploaded to the server)")
            return 2
        plot_tiers(rows, args.plot_tiers)
        print(f"wrote {args.plot_tiers}")

    if args.by == "scan":
        header = (
            f"{'run':20} {'user':9} {'material':13} {'scan':>5} "
            f"{'chips':>5} {'area mm2':>9} {'flakes':>6} {'/mm2':>7}  git"
        )
        print(header)
        for r in rows:
            print(
                f"{r['run']:20} {r['user'][:9]:9} {str(r['material'])[:13]:13} {r['scan_id']:5} "
                f"{r['n_chips']:5} {r['area_mm2']:9.1f} {r['n_flakes']:6} "
                f"{r['density']:7.3f}  {r['git']}"
            )
        return 0

    def key(r: dict):
        if args.by == "user":
            return r["user"].lower()
        if args.by == "material":
            return str(r["material"])
        iso = r["date"].isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)

    print(f"{args.by:16} {'runs':>5} {'chips':>5} {'area mm2':>9} {'flakes':>6} {'/mm2':>7}")
    for k in sorted(groups):
        g = groups[k]
        area = sum(r["area_mm2"] for r in g)
        nf = sum(r["n_flakes"] for r in g)
        print(f"{k[:16]:16} {len(g):5} {sum(r['n_chips'] for r in g):5} {area:9.1f} {nf:6} {nf / area:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
