# SDK Bumps

Issues and improvement ideas for Phase 2.5 evaluation.

---

## Binning uses magic numbers

**Current:**
```python
camera.binning_level = 2  # what does 2 mean?
```

**Should be:**
```python
class BinningMode(IntEnum):
    NONE = 0
    TWO_BY_TWO = 1
    THREE_BY_THREE = 2

camera.binning = BinningMode.THREE_BY_THREE
```

---

## Missing velocity and framerate controls

SDK has these in enums but they're not wired up in the driver:

**Stage velocity:**
- `IID_BASIC_CONTROL_VALUE_VELOCITY` (264)
- `IID_DIRECTED_CONTROL_VALUE_ASYNC_VELOCITY` (276)

**Camera framerate:**
- `PROP_ACQUISITION_FRAMERATE` (16502)
- `PROP_ACQUISITION_FRAMERATE_ENABLED` (16503)

Currently only `move_abs`/`move_rel` exposed (position control, blocking). For continuous scanning need:

```python
# Stage
stage.x_axis.velocity = 60.0  # mm/s
stage.x_axis.move_continuous(direction=1)  # non-blocking
stage.x_axis.stop()

# Camera
camera.framerate = 32
camera.start_continuous()
camera.stop_continuous()
# or callback-based: camera.on_frame(callback)
```

---
