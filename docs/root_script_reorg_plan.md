# Root Script Reorganization Plan

**Status**: Phase 1 complete

## Goal

Clean up the root directory from 21 Python scripts to 11. Root keeps only pipeline core and live microscope tools. Analysis, plotting, and utilities move to `scripts/`. Superseded scripts get archived or deleted.

Long-term (v2): move all root scripts into `commands/` with typed `run()` signatures. `find_flakes.py` calls them in-process with a shared `Microscope` connection, saving ~5s per subprocess call (2+2N connections per pipeline run).

---

## Phase 0: Add `AreaRect` type + parser

Follow the established `GainRGB`/`parse_white_balance` pattern.

**Add to `src/flakefinder/types.py`:**
```python
class AreaRect(NamedTuple):
    """Stage-coordinate rectangle (x_min, x_max, y_min, y_max) in um."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
```

**Add to `src/flakefinder/scan_utils.py`:**
```python
def parse_area_rect(s: str) -> AreaRect:
    """Parse 'x_min,x_max,y_min,y_max' string into AreaRect.

    Intended as an argparse type= callback.
    Sorts min/max if swapped. Validates non-zero extent.
    """
```

**Replace duplicates** (3 scripts have their own copy):
- `scan_area_v1.py`: local `parse_area_rect` → `from flakefinder.scan_utils import parse_area_rect`
- `scan_area_with_focus.py` (archive): same
- `stitch_area.py`: hand-rolled crop parsing → use `parse_area_rect`

Can be done independently, no downtime needed.

---

## Phase 1: File moves + cleanup

### Actions

### Keep in root (11 scripts)

| Script | Role |
|--------|------|
| `find_flakes.py` | Pipeline orchestrator |
| `scan_area_v1.py` | Core scanning engine |
| `stitch_area.py` | Stitch scan frames |
| `find_chips.py` | Chip detection |
| `stage_util.py` | Stage/objective control |
| `focus_map.py` | Focus map sampling |
| `analyze_focus_map.py` | Plane fit + export (used by pipeline) |
| `scan_chip_v2.py` | Chip scanner with Z tracking |
| `capture_util.py` | Single image capture |
| `autofocus_demo.py` | Autofocus CLI |
| `detect_flakes.py` | MaskTerial flake detection (Phase 5) |

### Move to scripts/ (6 scripts)

| Script | Reason |
|--------|--------|
| `mosaic_util.py` | Utility script, not part of live scanning |
| `analyze_autofocus.py` | Post-hoc analysis of autofocus runs |
| `analyze_scan.py` → `analyze_chip_scan.py` | Post-hoc analysis of completed scans |
| `plot_frame_pacing.py` | Diagnostic plotting |
| `plot_zjump_polling.py` | Diagnostic plotting |
| `plot_zjump_timing.py` | Diagnostic plotting |

### Move to src/flakefinder/ (1 module)

| Script | Reason |
|--------|--------|
| `cli_utils.py` | Library module (provides `report_status`, `park_microscope`), not a standalone script |

### Archive (2 scripts)

| Script | Reason |
|--------|--------|
| `scan_area_with_focus.py` | Superseded by scan_area_v1 + focus_map workflow. Docstring says "Unmaintained". Keep for reference. |
| `analyze_focus_scan.py` | Companion analysis for scan_area_with_focus. Keep for reference. |

Move to `archive/` directory.

### Delete (1 script)

| Script | Reason |
|--------|--------|
| `scan_chip.py` | Fully superseded by `scan_chip_v2.py`. `find_flakes.py` only calls v2. |

## Cross-reference fixes

These must be done atomically with the moves:

1. **`cli_utils` import path** (2 files):
   - `capture_util.py`: `from cli_utils import report_status` → `from flakefinder.cli_utils import report_status`
   - `stage_util.py`: `from cli_utils import park_microscope, report_status` → `from flakefinder.cli_utils import park_microscope, report_status`

2. **`analyze_scan` path** (1 file):
   - `scripts/process_chip_scan.py` line 55: `REPO_DIR / "analyze_scan.py"` → `REPO_DIR / "scripts" / "analyze_scan.py"`

3. **`analyze_autofocus` → `mosaic_util` import**: No change needed. Both move to `scripts/`, and when running `uv run python scripts/analyze_autofocus.py`, Python adds `scripts/` to `sys.path` so `from mosaic_util import make_mosaic` resolves.

4. **Docs and commands** (path updates):
   - `.claude/commands/scan.md`: update paths for `analyze_scan.py` and `mosaic_util.py` to `scripts/`
   - `README.md`: remove `scan_area_with_focus.py` from directory tree listing
   - `CLAUDE.md`: no changes needed (only lists scripts that stay in root)

## Execution checklist

- [ ] Create `archive/` directory
- [ ] Move 2 scripts to `archive/`
- [ ] Move 6 scripts to `scripts/`
- [ ] Move `cli_utils.py` to `src/flakefinder/cli_utils.py`
- [ ] Update 2 import statements (capture_util, stage_util)
- [ ] Update subprocess path in process_chip_scan.py
- [ ] Update .claude/commands/scan.md paths
- [ ] Update README.md directory tree
- [ ] Delete `scan_chip.py`
- [ ] Run `uv run ruff check --fix . && uv run ruff format .`
- [ ] Test: `uv run python find_flakes.py --dry-run`
- [ ] Git pull on microscope

---

## Phase 2 (v2): In-process pipeline with typed commands

Move root scripts into `commands/` as importable modules. `find_flakes.py` calls them in-process with a shared `Microscope` connection.

### Architecture

Each command exposes a `run()` with typed keyword args:

```python
# commands/scan.py

def run(
    scope: Microscope | None = None,
    *,
    output: Path,
    objective_mag: float = 5.0,
    area_rect: AreaRect,
    z: float | None = None,
    downsample: int = 1,
    white_balance: GainRGB | None = None,
    clean: bool = False,
    quiet: bool = False,
):
    """Core scan logic. Creates own Microscope if scope is None."""
    ...

if __name__ == "__main__":
    args = parser.parse_args()
    run(
        output=args.output,
        objective_mag=float(args.objective_mag.rstrip("x")),
        area_rect=args.area_rect,  # argparse type=parse_area_rect
        ...
    )
```

- `run()` accepts typed values only — no string parsing
- Standalone: `uv run python commands/scan.py --area-rect 8000,95000,0,78000`
- Pipeline: `find_flakes` imports `run()` directly, passes shared `scope`
- String parsing (`parse_area_rect`, `parse_white_balance`) stays in argparse layer

### Naming cleanup

| Current | New (`commands/`) |
|---------|-------------------|
| `scan_area_v1.py` | `scan.py` |
| `scan_chip_v2.py` | `chip_scan.py` |
| `autofocus_demo.py` | `autofocus.py` |
| `stage_util.py` | `stage.py` |
| `capture_util.py` | `capture.py` |
| `mosaic_util.py` | `mosaic.py` |
| `stitch_area.py` | `stitch.py` |
| `find_chips.py` | `find_chips.py` |
| `focus_map.py` | `focus_map.py` |
| `analyze_focus_map.py` | `analyze_focus_map.py` |
| `detect_flakes.py` | `detect_flakes.py` |

### Pipeline connection savings

`find_flakes.py` currently shells out via subprocess, each hardware script creates a new `Microscope()` connection (~5s camera init).

Formula: `(2 + 2N) × 5s` overhead, where N = chip count.

| Chips | Connections | Overhead saved |
|-------|-------------|----------------|
| 1 | 4 | 20s |
| 5 | 12 | 60s |
| 10 | 22 | 110s |
| 20 | 42 | 210s (3.5 min) |

With in-process calls: 1 connection total, 5s.

### find_flakes.py pipeline (sketch)

```python
from commands import scan, stitch, find_chips, stage, focus_map, analyze_focus_map, chip_scan

with Microscope() as scope:
    # Step 1: 5x overview
    scan.run(scope, output=overview_dir, objective_mag=5.0,
             z=args.initial_z, area_rect=args.area_rect,
             downsample=4, white_balance=args.white_balance, clean=True)

    # Step 2-3: stitch + detect (no hardware)
    stitch.run(scan_dir=overview_dir, quiet=True)
    chips = find_chips.run(stitch_path)

    # Step 4: switch to 20x
    stage.run(scope, objective_mag=20.0)

    # Per-chip loop
    for chip_idx in chip_indices:
        focus_map.run(scope, chips_meta=chips_json, chip=chip_idx, ...)
        analyze_focus_map.run(focus_map_path, export_plane=plane_path, ...)
        chip_scan.run(scope, output=scan_dir, chips_meta=chips_json,
                      chip=chip_idx, plane=plane_path, ...)
```

### Execution (incremental)

Can migrate one script at a time:
1. Move script to `commands/`, extract `run()`, keep `if __name__ == "__main__"` working
2. Update `find_flakes.py` to call `run()` for that one step (keep subprocess for rest)
3. Test both standalone and pipeline
4. Repeat for next script
