# Quick Scan — Interactive Stage Viewer

Interactive PySide6 GUI for ad-hoc microscope operation. Live camera feed on a pannable stage canvas, hardware controls, and persistent frame stamping.

## Status & Roadmap

**Phase 1 (done):** Canvas — QGraphicsScene stage viewer, pan/zoom, coordinate grid, viewport tracking.

**Phase 2 (done):** Live camera + controls — left dock with exposure, white balance, light, objectives, camera preview. Live camera overlay on canvas. Persistent frame stamping. Double-click-to-move (canvas and preview). ScopeManager background thread.

**Phase 3 (next):** Load existing scan data — place frames from `scan_meta.json` onto canvas, capture button.

**Phase 4 (future):** ROI drawing, quick scan, pretty scan.

**Backlog:** Scale bar, stitched image background layers, flake detection overlay.

## Running

```bash
uv run quick-scan              # offline (canvas only)
uv run quick-scan --connect    # connect to microscope
```

On the microscope PC, `.venv/Scripts/quick-scan.exe` (console-free via `gui-scripts`) can be a desktop shortcut target.

## File Map

```
src/quick_scan/
    __init__.py
    __main__.py           # python -m quick_scan
    app.py                # QuickScanWindow — wires everything together
    stage_canvas.py       # StageCanvas — QGraphicsView with pan/zoom/grid/camera
    scope_manager.py      # ScopeManager — background thread, SDK bridge
    control_panel.py      # ControlPanel — dock widget with all controls
```

Sibling package to `flakefinder` in `src/`. Imports `flakefinder.leica` and `flakefinder.data_utils` only when connected.

**Dep:** `PySide6-Essentials` in the `gui` dependency group. `default-groups` in `[tool.uv]` includes it, so `uv sync` installs it. The Essentials package ships `.pyi` stubs so ty resolves PySide6 imports.

## Architecture

### Threading Model

The GUI thread does **zero** SDK calls. All hardware interaction is on a single background thread managed by `ScopeManager`:

```
┌─────────────────────────────────────────────────────────────────┐
│ GUI Thread                                                      │
│                                                                 │
│  QuickScanWindow                                                │
│    ├── StageCanvas          (QGraphicsView)                     │
│    ├── ControlPanel         (QWidget in left dock)              │
│    └── signals ←──────────── ScopeManager (QObject)             │
│              Qt AutoConnection (auto-queued cross-thread)       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Qt Signals (position, frames,
                                │ hw_state, errors)
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│ ScopeManager._run() — background daemon thread                  │
│                                                                 │
│  1. Creates Microscope(), enters context                        │
│  2. Reads initial HardwareState → emits hw_state_ready          │
│  3. Starts FrameStream (camera.stream().start())                │
│  4. Main loop (~20 Hz):                                         │
│     a. Drain command queue → execute each cmd(scope)            │
│     b. Poll position → emit position_updated                    │
│     c. Drain FrameStream → take latest → emit frame_ready      │
│     d. sleep 50ms                                               │
│                                                                 │
│  FrameStream runs AcquireContinuous on its own internal thread  │
│  (SDK-managed). Frames land in a queue; we drain() each cycle.  │
└─────────────────────────────────────────────────────────────────┘
```

**Command queue pattern:** GUI thread calls convenience methods like `scope.move_to(x, y)` which push a `Callable[[Microscope], None]` onto a `queue.Queue`. The background thread drains the queue each cycle and executes commands synchronously. This keeps all SDK access on one thread.

**Qt signal safety:** Signals emitted from the background thread use `Qt.AutoConnection`, which automatically becomes `QueuedConnection` across threads — slots run on the GUI thread's event loop.

### Signal Flow

```
User drags exposure slider
  → ControlPanel.exposure_changed(ms)
    → ScopeManager.set_exposure_ms(ms)        [queues command]
      → background thread: camera.exposure_time = ms/1000

Background thread polls position
  → ScopeManager.position_updated(x, y, z, mag, pos, fov_w, fov_h)
    → app._on_scope_position()
      → StageCanvas.set_viewport(ViewportInfo)
      → ControlPanel.set_objective(mag)
      → status bar text update

Background thread drains camera frame
  → ScopeManager.frame_ready(QImage)
    → app._on_frame()
      → StageCanvas.update_camera_frame(qimg)   [canvas overlay + stamping]
      → ControlPanel.update_preview(qimg)        [dock preview]
```

### StageCanvas Internals

**Coordinate system:** Scene uses stage µm directly. The view transform handles zoom (scale) and pan. `+Y` is down (matches stage and screen convention).

**Scene rect:** Set to ±500,000 µm (much larger than the ~95×85 mm stage). This ensures scrollbar range is always nonzero for pan-via-scrollbar to work. Items only exist in the stage area so the extra extent costs nothing.

**Z-order layers:**
| Z | Content |
|---|---------|
| 100 | Viewport rect (red outline + translucent fill) |
| 50 | Live camera frame (current position) |
| 10 | Placed/stamped frames (persistent mosaic) |
| −9 | Grid labels |
| −10 | Grid lines |

**Zoom:** Scroll wheel with factor 1.15×, clamped to 0.01–200 µm/px. Anchored on cursor position via `mapToScene → scale → mapToScene → translate(delta)`. Must use `NoAnchor` transformation anchor.

**Pan:** Right/middle-click drag. Uses `horizontalScrollBar().setValue()` offset, NOT `translate()` on the transform (which doesn't work reliably with the internal scene rect).

**Grid:** Auto-adapting spacing tiers from 100 µm to 50 mm. Regenerated only when the tier changes. Labels use `ItemIgnoresTransformations` flag so text stays screen-readable at any zoom.

**Frame stamping:** When the viewport moves >30% of FOV from the last stamp, the current live camera frame is copied as a new `QGraphicsPixmapItem` at its position. Builds up a mosaic. Caps at 500 items, evicts oldest. `clear_placed_frames()` available for cleanup.

**Double-click-to-move:** `mouseDoubleClickEvent` maps click to scene coords, emits `move_requested(x_um, y_um)`.

### ControlPanel Layout

Top-to-bottom in the left dock:

1. **Exposure** — log-scale slider (0.1 ms → 10,000 ms, 5 decades) + editable spinbox. Mapping: `ms = 10^(slider/1000*5 - 1)`.
2. **White Balance** — R/G/B sliders (0.50–4.00) + spinboxes + Reset button.
3. **Light** — checkbox toggle ("Shutter Open/Closed") + intensity slider (0 to `lamp.max_intensity`).
4. **Mag** — 2×3 grid of toggle buttons (2.5x, 5x, 10x, 20x, 50x, 150x). Active objective highlighted.
5. **Camera Preview** — QLabel showing scaled live frame. Double-click maps click position to stage coords for move.

**`_updating` guard:** All `set_*()` methods set `self._updating = True` before programmatic widget changes to suppress signal emission, preventing feedback loops with the hardware.

**Preview double-click:** Uses `eventFilter` on the QLabel. Maps click position within the scaled pixmap to stage coordinates using tracked viewport info and pixmap rect within the label.

### ScopeManager API

| Method | Effect |
|--------|--------|
| `open()` / `close()` | Start/stop background thread (named to avoid shadowing `QObject.connect/disconnect`) |
| `move_to(x, y)` | Queue async XY stage move |
| `switch_objective(mag)` | Queue nosepiece change |
| `set_exposure_ms(ms)` | Queue exposure change |
| `set_gain_rgb(r, g, b)` | Queue white balance change |
| `set_shutter(bool)` | Queue shutter open/close |
| `set_lamp_intensity(int)` | Queue lamp intensity change |

| Signal | Payload |
|--------|---------|
| `position_updated` | x, y, z, mag, obj_pos, fov_w, fov_h |
| `frame_ready` | QImage (RGB888, owned copy) |
| `hw_state_ready` | HardwareState NamedTuple |
| `scope_connected` | (none) |
| `scope_disconnected` | (none) |
| `scope_error` | error message string |

### Key Types

```python
class ViewportInfo(NamedTuple):   # stage_canvas.py
    x_um: float                   # viewport center X
    y_um: float                   # viewport center Y
    fov_w_um: float               # field of view width
    fov_h_um: float               # field of view height

class HardwareState(NamedTuple): # scope_manager.py
    exposure_ms: float
    gain_rgb: tuple[float, float, float]
    shutter_open: bool
    lamp_intensity: int
    lamp_max: int
```

## Controls

| Input | Action |
|-------|--------|
| Scroll wheel | Zoom (centered on cursor) |
| Right/middle-click drag | Pan |
| Home | Fit full stage in view |
| F | Center on microscope viewport (~5× FOV) |
| Double-click (canvas) | Move stage to clicked position |
| Double-click (preview) | Move stage to clicked position in FOV |

## Gotchas & Lessons

- **Scene rect must be huge** (±500k µm) for scrollbar-based pan to work. If `fitInView()` makes the scene fit exactly, scrollbar range is 0 and pan breaks.
- **`ItemIgnoresTransformations`** is how grid labels stay readable at any zoom — they're positioned in scene coords but rendered at fixed screen size.
- **Cosmetic pens** `QPen(color, 0)` = always 1px on screen regardless of zoom. Used for grid, stage border, viewport rect.
- **`SizePolicy.Ignored`** on the camera preview QLabel prevents a feedback loop where pixmap → label grows → dock grows → bigger pixmap → repeat.
- **`open()`/`close()`** not `connect()`/`disconnect()` — the latter shadow `QObject` static methods and ty catches the LSP violation.
- **numpy → QImage:** Must `.copy()` to detach from numpy buffer, otherwise QImage holds a dangling pointer after the numpy array is GC'd.
- **FrameStream** (`flakefinder.leica.camera`) is old/lightly-used code. It works for live view but may need work for higher-performance scenarios.

## FOV Sizes (3×3 binning)

| Objective | FOV (µm) | Frame (px) |
|-----------|-----------|------------|
| 2.5x | 5,253 × 3,502 | 1,824 × 1,216 |
| 5x | 2,627 × 1,751 | 1,824 × 1,216 |
| 10x | 1,313 × 876 | 1,824 × 1,216 |
| 20x | 657 × 438 | 1,824 × 1,216 |
| 50x | 263 × 175 | 1,824 × 1,216 |
| 150x | 88 × 58 | 1,824 × 1,216 |
