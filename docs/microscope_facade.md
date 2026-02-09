# `Microscope` Facade Class

## Current Implementation State

**Step 0 (done):** `Microscope` class created in `src/flakefinder/leica/microscope.py`. Exported from `leica/__init__.py`.

Subsystems initialized eagerly on `__enter__` (all required, fail hard if missing):
- `stage`, `z`, `nosepiece`, `shutter`, `lamp`
- `camera` and `acquisition` are lazy (first access triggers UCAPI registration)

High-level methods:
- `switch_objective_pos(pos)` / `switch_objective_mag(mag)` — Z-speed-max logic hoisted from `Nosepiece.set_position()` (now deprecated)
- `light_on(intensity_pct=100)` / `light_off()` — shutter + lamp combined
- `create_acquisition_context()` — UCAPI factory

Supporting changes:
- `Lamp.intensity_pct` property added (0-100 scale, maps to native 0-255)
- `Point2F`/`Point3F` type aliases applied to `Stage.position_um`, `Microscope.position`, `Frame.position`

**Step 1 (done):** `stage_util.py` and `capture_util.py` migrated. All None guards removed.

**Step 2 (in progress):** `focus_map.py`, `scan_chip.py`, and `scan_area_v1.py` migrated. Also extracted `sdk_image_to_numpy` to `src/flakefinder/image_utils.py` (was `Camera._image_to_numpy` private static). `scan_area_with_focus.py`, `capture_flatfield.py` still use raw `LeicaConnection` + `Nosepiece.set_position()`.

**Step 3 (not started):** `continuous_autofocus()` signature refactor.

---

## 1. Original Audit

### Setup boilerplate duplicated across scripts

Every script that touches hardware follows the same pattern. Here's the common sequence, with which scripts include each step:

| Step | scan_chip | scan_area_v1 | focus_map | autofocus_demo | capture_util | stage_util |
|------|:---------:|:------------:|:---------:|:--------------:|:------------:|:----------:|
| `LeicaConnection()` context manager | x | x | x | x | x | x |
| `Extensions.ExUCAPI.Register()` | x | x | x | x | - | - |
| `Stage.from_connection(conn)` | x | x | x | x | x | x |
| `ZDrive.from_connection(conn)` | x | x | x | x | x | x |
| Nosepiece init (try/except) | x | x | - | - | x | x |
| Objective switching logic | x | x | - | - | - | x |
| `Shutter.from_connection` + open | x | x | x | x | x | x |
| `Lamp.from_connection` + full | x | x | x | x | x | x |
| `Camera.from_connection(conn)` | x | x | x | x | x | - |
| Camera config (trigger, binning, exposure, gain, wb, gamma) | x | x | x | x | x | - |
| `get_interface_required(camera._unit, UCAPI_IID.IID_IMAGE_ACQUISITION)` | x | x | x | x | - | - |
| Acquisition context factory | x | x | x | x | - | - |
| Frame size computation from camera props | x | x | - | - | - | - |

### Concrete line counts of setup boilerplate

| Script | Setup lines | What those lines do |
|--------|-------------|---------------------|
| `scan_chip.py` | ~143 (L384-527) | Connect, stage/z/nosepiece/shutter/lamp/camera, objective switch, camera config, frame size validation, acquisition context |
| `scan_area_v1.py` | ~233 (L271-504) | Same as scan_chip + velocity helpers + autofocus position + more printing |
| `focus_map.py` | ~54 (L509-563) | Connect, stage/z/shutter/lamp/camera, camera config, acquisition context |
| `autofocus_demo.py` | ~96 (L155-251) | Connect, stage/z/shutter/lamp/camera, conditional camera config, acquisition context |
| `capture_util.py` | ~64 (L128-192) | Connect, lamp/shutter/camera, camera config, status reporting |
| `stage_util.py` | ~5 (L248-253) | Connect, stage/z (other peripherals accessed ad-hoc) |

### Specific repeated patterns

**Pattern A: "Safe peripheral init" (5/6 scripts)**
```python
try:
    shutter = Shutter.from_connection(conn)
    shutter.open()
except LookupError:
    pass
```
Repeated for shutter, lamp, nosepiece. ~6-12 lines per script, identical everywhere.

**Pattern B: "Objective switching" (3/6 scripts)**
```python
# ~30 lines: parse mag string OR validate position number,
# check nosepiece exists, compare to current position,
# call nosepiece.set_position(target, z=z_drive)
```
Copy-pasted across `scan_chip.py`, `scan_area_v1.py`, `stage_util.py` with minor variations.

**Pattern C: "Camera + acquisition setup" (4/6 scripts)**
```python
camera = Camera.from_connection(conn)
acquisition = get_interface_required(camera._unit, UCAPI_IID.IID_IMAGE_ACQUISITION)
camera.trigger_mode = 0
camera.binning = binning_idx
camera.exposure_time = args.exposure_ms / 1000.0
camera.gain = args.gain
camera.gain_rgb = (wb_red, wb_green, wb_blue)
camera.gamma = args.gamma
context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
```
Every scanning script does this. The `acquisition` and `context` objects are low-level SDK details that leak into every script.

**Pattern D: "UCAPI registration" (4/6 scripts)**
```python
from LeicaMicrosystems.HardwareModel import Extensions
Extensions.ExUCAPI.Register()
```
Required before any camera access. Easy to forget; scripts that skip it crash.

**Pattern E: "Nosepiece.set_position needs ZDrive" (3/6 scripts)**
```python
nosepiece.set_position(target_pos, z=z_drive)
```
The `Nosepiece.set_position` TODO comment (units.py L737) already says: "TODO: Move to a higher-level Microscope class that owns both the nosepiece and ZDrive, so callers don't need to pass z explicitly."

---

## 2. Design Decisions

### Subsystem ownership

`Microscope` owns the `LeicaConnection` lifecycle. Every script currently creates its own connection; there's no use case for sharing. Escape hatch: `scope.conn` for raw access.

### All subsystems required

All hardware (stage, z, nosepiece, shutter, lamp, camera) is present on this DM6M. No `| None` types, no try/except guards. If hardware is missing, fail hard on `__enter__`.

Camera stays lazy (avoids UCAPI import cost for `stage_util.py` and similar non-camera scripts) but fails hard when accessed.

### Objective switching split

Two explicit methods instead of `str | int` dispatch:
- `switch_objective_pos(pos: int)` — validates position, handles z-speed-max, does the switch
- `switch_objective_mag(mag: str)` — parses magnification, delegates to `switch_objective_pos()`

### What stays out of `Microscope`

- **Camera configuration** (exposure, gain, wb, gamma, binning) — scan-specific, no useful default
- **Autofocus** — complex enough for its own module
- **Scan planning / row iteration** — application logic
- **Z velocity management** — deeply context-dependent
- **Parking** — free function taking `Microscope`
- **Metadata construction** — serialization concern
- **Frame size computation** — already in `data_utils`

---

## 3. Migration Example: `focus_map.py`

### Before (~35 lines of setup)

```python
from flakefinder.leica import Lamp, LeicaConnection, Shutter, Stage, ZDrive
from flakefinder.leica.camera import Camera
from flakefinder.leica.core import get_interface_required
from flakefinder.leica.enums import UCAPI_IID

with LeicaConnection() as conn:
    from LeicaMicrosystems.HardwareModel import Extensions
    Extensions.ExUCAPI.Register()

    stage = Stage.from_connection(conn)
    z_drive = ZDrive.from_connection(conn)
    try:
        shutter = Shutter.from_connection(conn)
        shutter.open()
    except LookupError:
        pass
    try:
        lamp = Lamp.from_connection(conn)
        lamp.full()
    except LookupError:
        lamp = None
    camera = Camera.from_connection(conn)
    acquisition = get_interface_required(camera._unit, UCAPI_IID.IID_IMAGE_ACQUISITION)
    camera.trigger_mode = 0
    camera.binning = 2
    camera.exposure_time = 0.001
    context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
```

### After (~10 lines)

```python
from flakefinder.leica import Microscope

with Microscope() as scope:
    scope.light_on()
    scope.camera.trigger_mode = 0
    scope.camera.binning = 2
    scope.camera.exposure_time = 0.001
    context = scope.create_acquisition_context()
```

---

## 4. Remaining Migration Plan

### Step 2: Migrate scan scripts

4. **`focus_map.py`** — refactor `run_focus_map()` to take `Microscope` instead of 5 separate subsystem args.
5. **`scan_area_v1.py`** — larger but the setup section simplifies dramatically.
6. **`scan_chip.py`** — most complex; benefits the most from reduced boilerplate.
7. **`scan_area_with_focus.py`** / **`capture_flatfield.py`** — same pattern.

### Step 3: Refactor autofocus library

Change `continuous_autofocus()` signature from `(conn, camera, acquisition, context, ...)` to `(scope, context, ...)`.

### Step 4: Facade-owned acquisition context

`create_acquisition_context()` currently returns a raw factory that callers must manage and dispose. Make `Microscope` own a default context via lazy `scope.context` property, auto-disposed in `__exit__`.

This works for single-shot capture (`camera.capture()`) and autofocus (`continuous_autofocus`), which share one context. `FrameStream` and `DeferredFrameStream` already create their own per-thread contexts in `_acquire_loop` — they must, since each needs independent cancellation. So facade-owned context = default for non-streaming use; streams self-manage as today.

Once Step 3 lands (autofocus takes `scope`), `context` disappears from `run_focus_map` and `continuous_autofocus` signatures entirely — the facade provides it implicitly.

### Step 5: Discovered improvements

- **`Stage.wait_all()` is a misplaced static method.** `run_focus_map` and scan scripts import `Stage` just for this utility. Should be a standalone function (e.g. in `units.py` module-level or a `utils` module).
- **`desc` (MicroscopeDescription) used inconsistently.** `scan_area_v1.py` guards `desc` with `if desc:` for pre-validation but later uses `desc.camera` unconditionally for frame size computation. Low risk (JSON always exists) but should either fail hard early if `desc` is None or guard consistently.

### What doesn't need to change

- `src/flakefinder/leica/units.py` — `Stage`, `ZDrive`, `Nosepiece`, etc. remain as-is.
- `src/flakefinder/leica/camera.py` — `Camera` keeps its existing API.
- `src/flakefinder/leica/core.py` — `LeicaConnection` still exists; `Microscope` wraps it.
- `stitch_area.py`, `find_chips.py`, `analyze_focus_map.py` — no hardware access, unaffected.

### Testing strategy

Since this runs on real hardware only, the migration is verifiable by:
1. Running `stage_util.py` after migration — confirms subsystem init works
2. Running `capture_util.py` — confirms camera works through facade
3. Running `autofocus_demo.py` — confirms acquisition pipeline works
4. Full scan test — confirms nothing regressed in the hot path

Each script can be migrated independently. No big-bang required.
