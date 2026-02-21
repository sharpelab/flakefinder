# Run Viewer (`run-viewer`)

Tkinter GUI for browsing completed find-flakes runs. Designed for collaborators to review results on the microscope PC without CLI knowledge.

```
uv run run-viewer                       # default: scans/ directory
uv run run-viewer --scans-dir path/to/  # custom scans directory
```

## Layout

Run list auto-hides when a run is selected. Click "← Runs" to restore it.

```
┌──────────────────────────────────────────────────┐
│ [← Runs]                                         │
│                                                   │
│ ┌─ run_20260220_1543 ──────────────────────────┐  │
│ │ Date / Preset / Chips / Duration             │  │
│ │ Notes: Operator: Sandesh ...                 │  │
│ └──────────────────────────────────────────────┘  │
│                                                   │
│ ┌─ Overview ───────────────────────────────────┐  │
│ │ (chip detection overlay image)               │  │
│ └──────────────────────────────────────────────┘  │
│                                                   │
│ ┌─ Filters ───────────────────────────────────┐   │
│ │ perim_ratio ≤ [===] cal_dist ≤ [===] ...    │   │
│ │ G min ≥ [===] G max ≤ [===] R max ≤ [===]  │   │
│ │ entropy ≤ [===] grad ≤ [===] ar ≤ [===]    │   │
│ │ kurtosis ≤ [===]                            │   │
│ │ 845 / 49123 pass       Top N: [20] [Reset] │   │
│ └──────────────────────────────────────────────┘  │
│                                                   │
│ ┌─ Chips ──────────────────────────────────────┐  │
│ │ [Chip 0] [Chip 1] [▼Chip 2▼] [Chip 3] ...  │   │
│ └──────────────────────────────────────────────┘  │
│                                                   │
│ ┌─ Top Detections ─────────────────────────────┐  │
│ │ (clickable frame thumbnails with bbox)       │  │
│ │ #1 C2 R=+0.04 G=+0.03 580µm²               │  │
│ │ e=2.1 g=3.4 k=12                            │  │
│ └──────────────────────────────────────────────┘  │
│                                                   │
│ ┌─ Scoring Data ──────────────────────────────┐   │
│ │ # Chip T Frame      Size  R     G  Score ...│   │  ← sortable columns
│ │ 1  2   1 frame_0403 7890 +0.04 +0.03  12.3 │   │
│ └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## Architecture

### Data loading

`_load_all_detections(run_dir, n_chips)` loads all chips' detections at once from each `chip_N/seg/summary.json`. Each detection dict gets a `chip_idx` key. Returns `(all_detections, chip_scan_dirs, um_per_px)`.

Stage coordinates (`stage_x`, `stage_y`) are included in `summary.json` since the segmentation step (both `segment_chip_scan.py` and `find_flakes.py`'s background seg call `add_stage_coords()`).

### Chip toggle buttons

Chip buttons are toggle filters, not selectors:
- **All off = all shown** (default on run load)
- Click a chip → toggles it on (SUNKEN relief). Only that chip's detections are visible.
- Click more chips → union of selected chips shown.
- Click an active chip → deactivates it. When last one is deactivated, back to showing all.
- Reset button clears chip selection too.

Implementation: `_active_chips: set[int]` tracks toggled chips. `_get_visible_chips()` returns `None` (all) when set is empty. `_apply_filters()` applies chip filter before slider gates.

### Filter pipeline

`_apply_filters()` runs on every slider change (debounced 200ms) and chip toggle:

1. **Chip filter**: if any chips toggled, keep only detections with matching `chip_idx`
2. **Slider gates**: perim_ratio, cal_dist, G range, R max, entropy, min size, grad_energy, aspect_ratio, kurtosis
3. **Sort** by score descending
4. **Top N** truncation
5. **Update** table and frame thumbnails

Slider defaults come from `DetectorConfig.from_material(preset)` tier1 gates. Reset restores these defaults.

### Scoring table

`_populate_detection_table` renders a sortable `Treeview` with columns: rank, chip, tier, frame, det, size, R, G, score, cal_dist, grad, entropy, kurtosis, perim_ratio, aspect_ratio. Click any column header to sort ascending/descending (arrow indicator in header text).

### Frame cache

`_frame_cache: dict[tuple[int, str], Image.Image]` keyed by `(chip_idx, frame_name)` since frame names (e.g. `frame_0001`) can collide across chips. Each chip's frames live in its own `chip_N/scan_*/` directory, mapped by `_chip_scan_dirs: dict[int, Path | None]`. LRU eviction at 30 entries.

### Image popups

- **Thumbnail click** → `ZoomableFramePopup`: full frame with detection bbox. Scroll to zoom (centered on mouse), drag to pan. Initial view shows whole frame, centered on detection. Dismiss with Escape.
- **Overview/image click** → `ImagePopup`: resizable contain-scaled view. Dismiss with Escape or click.

### Mousewheel scrolling

Mousewheel bound once at startup via `bind_all`. A `_scroll_active` boolean flag, toggled by `<Enter>`/`<Leave>` on the detail canvas, gates whether scroll events actually scroll. This avoids `unbind_all` which caused visual flashing when popups opened.

## Data Sources

| Section | Source file | Key fields |
|---------|-----------|------------|
| Run list | `checkpoint.json` | `args.preset`, `n_chips`, `step_timing`, `notes` |
| Run list detections | `chip_N/seg/summary.json` | `stats.total_detections` |
| Overview image | `overview_*_stitch_chips_detected.png` | — |
| Detection data | `chip_N/seg/summary.json` | `detections_by_frame` (includes `stage_x/y`) |
| Frame images | `chip_N/scan_*/frame_NNNN.jpg` | Raw scan frames |

## Filter Slider Ranges

| Slider | Range | Default | Resolution |
|--------|-------|---------|------------|
| perim_ratio | 1.0–3.0 | from preset | 0.05 |
| cal_dist | 0.0–2.0 | from preset | 0.05 |
| min µm² | 0–2000 | 0 | 10 |
| G min | -2.0–6.0 | from preset | 0.1 |
| G max | -2.0–6.0 | from preset | 0.1 |
| R max | -3.0–6.0 | from preset | 0.1 |
| entropy | 0.0–8.0 | from preset | 0.1 |
| grad_energy | 0.0–50.0 | 50.0 (open) | 0.5 |
| aspect_ratio | 1.0–6.0 | 6.0 (open) | 0.5 |
| kurtosis | -2.0–50.0 | 50.0 (open) | 1.0 |

## Platform Notes

- **Windows**: Segoe UI font, native tkinter rendering
- **Linux**: DejaVu Sans font (TrueType, antialiased), `clam` ttk theme
- No hardware dependencies — reads only from disk
