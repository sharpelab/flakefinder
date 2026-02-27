# Quick Scan — Interactive Stage Viewer

Interactive GUI for ad-hoc microscope operation. Canvas-based stage viewer with viewport tracking, image stitching, and targeted high-res scans.

## Status

**Phase 1 (done):** Canvas foundation — PySide6 QGraphicsScene stage viewer with pan/zoom, coordinate grid, viewport tracking, and microscope connection.

**Phase 2 (done):** Live camera + controls — left dock with exposure (log slider), white balance (R/G/B), light (shutter + intensity), objective buttons, camera preview. Live camera overlay on canvas at viewport position. Double-click-to-move. ScopeManager background thread handles all SDK calls.

**Phase 3 (next):** Image display — load frames from existing scan directories onto the canvas, capture button.

**Phase 4 (future):** ROI drawing, quick scan, pretty scan.

## Running

```bash
# Offline mode (canvas only, no microscope)
uv run --group gui quick-scan

# With live microscope position tracking
uv run --group gui quick-scan --connect
```

On the microscope PC, the `quick-scan.exe` in `.venv/Scripts/` can be a desktop shortcut target (no console window — uses `pythonw.exe` via `gui-scripts`).

## Controls

| Input | Action |
|-------|--------|
| Scroll wheel | Zoom (centered on cursor) |
| Right-click drag | Pan |
| Middle-click drag | Pan |
| Home | Fit full stage in view |
| F | Center on microscope viewport (~5× FOV) |
| Double-click | Move stage to clicked position |

## Architecture

```
src/quick_scan/
    __init__.py
    __main__.py         # python -m quick_scan
    app.py              # QMainWindow, toolbar, dock panel, signal wiring
    stage_canvas.py     # QGraphicsView/Scene — pan, zoom, grid, viewport rect, camera overlay
    scope_manager.py    # Background thread: connection, position polling, camera stream, command queue
    control_panel.py    # Left dock: exposure, white balance, light, camera preview, objectives
```

Sibling package to `flakefinder` in `src/`. Imports `flakefinder.data_utils` and `flakefinder.leica` (the latter only when connected).

**Dependencies:** PySide6 (in `gui` dependency group). `uv sync --group gui` to install.

**Threading:** `ScopeManager` runs all SDK interaction on a single background thread. Camera frames arrive via `FrameStream` (SDK-managed thread), are drained each poll cycle, and emitted as Qt signals. Command queue dispatches stage moves, objective switches, and property changes on the same thread. GUI thread does UI only.

## Design

### Canvas

The scene uses stage coordinates directly (µm). The view transform handles zoom (scale) and pan (scroll). QGraphicsScene manages item visibility/culling automatically.

- **Stage area:** 0–95,172 × 0–85,103 µm
- **Grid:** Auto-adapts spacing (100 µm to 50 mm tiers) based on zoom level, labels in mm
- **Viewport rect:** Red outline + translucent fill showing current FOV at current objective
- **Coordinate convention:** +Y down (matches stage and screen)

### FOV Sizes (3×3 binning)

| Objective | FOV Width × Height (µm) |
|-----------|--------------------------|
| 2.5x | 5,253 × 3,502 |
| 5x | 2,627 × 1,751 |
| 10x | 1,313 × 876 |
| 20x | 657 × 438 |
| 50x | 263 × 175 |
| 150x | 88 × 58 |

## Roadmap

### Phase 3 — Image Display

- Load images from existing scan directories (`scan_meta.json` → frame positions)
- Place as `QGraphicsPixmapItem` at stage coordinates
- LOD: downsample based on zoom level, only render visible tiles
- "Capture" button → single frame at current position, placed on canvas

### Phase 4 — ROI & Scans

- Rubber-band rectangle drawing (left-click drag on empty canvas)
- Store ROIs as `AreaRect` in stage coordinates
- Quick Scan: fast low-mag sweep of a drawn region
- Pretty Scan: select region → magnification dialog → focus map + chip scan + stitch
- Output: `scans/quickscan_YYYYMMDD_HHMM/`

### Future

- Scale bar on canvas (auto-adapting to zoom level)
- Load previous stitched images as background layer
- Flake detection overlay
