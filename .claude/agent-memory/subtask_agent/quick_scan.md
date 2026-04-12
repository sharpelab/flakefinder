# Quick Scan GUI Context

## What It Is
Interactive PySide6 stage viewer for ad-hoc microscope operation. Sibling package to flakefinder at `src/quick_scan/`. Full architecture doc at `docs/quick_scan.md` — read it first.

## Current State (Phase 2 complete, on master, synced to microscope)

### Files
- `src/quick_scan/app.py` — QMainWindow, wires ScopeManager ↔ ControlPanel ↔ StageCanvas via Qt signals
- `src/quick_scan/stage_canvas.py` — QGraphicsView/Scene in µm coords, zoom, pan, grid, viewport rect, live camera overlay, persistent frame stamping, double-click-to-move
- `src/quick_scan/scope_manager.py` — single background thread owns Microscope, polls position, drains FrameStream, processes command queue. All SDK on this thread, GUI thread does zero SDK calls.
- `src/quick_scan/control_panel.py` — left dock: exposure (log slider), white balance (R/G/B + reset), light (shutter checkbox + intensity), mag buttons (2×3), camera preview (double-click-to-move)
- `docs/quick_scan.md` — full architecture doc with threading diagram, signal flow, z-order, gotchas

### Entry points
- `[project.gui-scripts] quick-scan` → `quick_scan.app:main` (pythonw.exe, no console)
- `[project.gui-scripts] find-flakes-gui` → `flakefinder.commands.launch_gui:main`

### Dep
- `PySide6-Essentials` in `[dependency-groups] gui`, included in `[tool.uv] default-groups`
- Essentials ships `.pyi` stubs so ty resolves imports. Do NOT use full `PySide6` (causes module conflict warnings).

## Key Patterns
- **Threading:** `ScopeManager` background thread does ALL SDK calls. GUI communicates via Qt signals (updates) and command queue (actions). Methods named `open()`/`close()` not `connect()`/`disconnect()` (shadows QObject).
- **Scene rect ±500k µm** so scrollbar pan always works. Items only in stage area.
- **Grid labels:** `ItemIgnoresTransformations` flag = fixed screen size at any zoom.
- **Zoom on cursor:** `mapToScene → scale → mapToScene → translate(delta)`, `NoAnchor`.
- **Camera overlay:** `QGraphicsPixmapItem` z=50, `SmoothTransformation`, scaled `fov_w/px_width`.
- **Frame stamping:** When viewport moves >30% FOV, live frame copied as persistent pixmap (z=10). Cap 500.
- **Preview label:** `SizePolicy.Ignored` prevents dock growth feedback loop from pixmap updates.
- **`_updating` guard:** All `set_*()` methods in ControlPanel set `_updating=True` to suppress signal emission during programmatic updates.
- **numpy→QImage:** Must `.copy()` to detach from numpy buffer before emitting.

## What's Next
- **Phase 3:** Load existing scan data onto canvas (frames from `scan_meta.json`), capture button
- **Phase 4:** ROI drawing, quick scan, pretty scan
- **Backlog:** Scale bar, stitched image layers, flake overlay, app icons

## Known Issues / Gaps
- FrameStream (`flakefinder.leica.camera`) is old/lightly-used — works for live view but may need hardening
- No app icon yet (gen_icon.py exists for flakefinder, need a quick-scan variant)
- Camera stream was showing black initially — likely shutter/lamp state on connect. Controls now sync from hardware state but user must manually open shutter.
