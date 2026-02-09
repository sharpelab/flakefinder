# scan_chip_v2 Refactor Plan

**Status**: Implemented. All three phases complete in `scan_chip_v2.py`.

## Motivation

During chip scans, Z velocity tracking starts 400-660ms before X motion begins
(warmup frames happen in between). Combined with a ~170-300ms camera delay on
the second frame, Z accumulates 2-8 um of error before the first useful frame.

Fix: use an X lead-in region. Start X first, delay Z tracking until X
approaches the real chip edge. The capture loop itself triggers Z start,
eliminating the timing gap entirely.

## Phase 0: Simplify Options

Remove experimental/diagnostic flags that complicate the row logic. Hardcode
the best-known defaults:

| Flag | Current | Action |
|------|---------|--------|
| `--z-preposition-mode` | corrected/raw | Remove. Always use corrected. |
| `--z-tracking-mode` | directed/async | Remove. Always use directed. |
| `--persistent-polling` | flag | Remove. Always per-row polling. |
| `--debug-polling` | flag | Remove. Position data is in scan_meta.json. |
| `--padding` | float, default 0 | Remove. Replaced by `--lead-in-um`. |

Remove all code branches for these options: persistent polling thread
management, async Z tracking mode, raw Z preposition, debug polling JSON dump.

## Phase 1: Refactor Row Startup

### New CLI args

```
--lead-in-um    float  default=2000  Lead-in distance before chip edge (um)
--z-lead-ms     float  default=50    Start Z tracking this many ms before X reaches chip edge
```

### Row geometry

The scan plan is computed with `padding=0` (hull-clipped, unpadded rows).
Lead-in is applied manually per row:

```
direction=+1:
  chip_edge_x  = row_x_min                    (real hull boundary)
  x_start_pos  = row_x_min - lead_in_um       (lead-in position)
  x_end_pos    = row_x_max                    (hull end, no tail pad)

direction=-1:
  chip_edge_x  = row_x_max
  x_start_pos  = row_x_max + lead_in_um
  x_end_pos    = row_x_min

z_start = plane_z(chip_edge_x, row_y)         (Z where tracking begins)
z_end   = plane_z(x_end_pos, row_y)           (Z at scan end)
z_vel   = plane_a * direction * x_speed        (unchanged constant)
```

### New per-row sequence

```
1. Parallel preposition
   - Y async, X async, Z corrected (blocking, runs while X/Y move)
   - wait_all([hx, hy]), dispose
   Z corrected has no async variant, but Z moves between rows are tiny
   (few um) so it finishes before X/Y. Total time = max(Y, X).

2. Settle (--row-settle, default 100ms)

3. Start per-row X/Z polling threads

4. Warmup camera (--warmup-frames, default 3)

5. Start X motion (async to x_end_pos at scan speed)

6. Capture loop (owns Z start):
   - Acquire frame
   - Read latest polled X position
   - If Z not started and X within z_lead of chip edge:
       start Z directed velocity (towards_max or towards_min)
       record t_z_started and z_start_x_um
   - Position-based frame save/skip (unchanged)
   - Repeat until X handle completes

7. Cleanup: halt Z, stop polling, collect stats
```

### Z start condition inside capture loop

```python
if not z_started and x_now is not None:
    signed_dist = (chip_edge_x - x_now) * direction  # positive = still approaching
    if signed_dist / x_speed_um_s <= z_lead_s:
        z_drive.start_towards_max(abs(z_vel)) if z_vel > 0 else ...start_towards_min(...)
        z_started = True
        t_z_started = perf_counter()
        z_start_x_um = x_now
```

When `signed_dist <= 0`, X has passed the edge, so the condition fires
immediately. Lead-in frames captured before Z starts get saved/skipped by
the normal position-advance logic (first frame saves at wherever X is,
subsequent frames skip until advance threshold met).

## Phase 2: Extract `scan_row` Helper

### Analysis

The per-row body is currently ~190 lines inside `main()`. After phase 0+1
cleanup it shrinks to ~100 lines, but it's still the core logic buried
inside a 1000-line function. Extracting it has clear benefits:

**Pros:**
- `main()` loop body becomes ~15 lines (call + accumulate results + print)
- Row logic is independently readable and eventually testable
- Clean separation: main() owns scan lifecycle, scan_row() owns row lifecycle
- Typed return (RowResult dataclass) replaces implicit state mutation

**Cons:**
- Needs ~10 hardware/config params passed in — manageable with 2 dataclasses
- save_queue crossing the boundary is slightly awkward (function puts to
  caller's queue) — but this is a clean producer interface

### Proposed interface

```python
@dataclass
class ScanHardware:
    """Hardware interfaces needed for row scanning."""
    stage: Stage
    z_drive: ZDrive
    acquisition: Any          # SDK acquisition context
    acq_context: Any          # SDK image acquisition context
    current_image: list       # [None] mutable ref for image callback
    x_bcv: Any                # fast X position reader
    x_converter: Any          # native -> um converter
    z_converter: Any          # native -> um converter
    z_bcv_hysteresis: Any     # hysteresis-corrected Z reader

@dataclass
class RowConfig:
    """Scan configuration for a single row."""
    plane_a: float
    plane_b: float
    plane_c: float
    x_speed_um_s: float
    move_speed_um_s: float
    target_advance_um: float
    lead_in_um: float
    z_lead_s: float
    row_settle_s: float
    warmup_frames: int
    z_max: float

@dataclass
class RowResult:
    """Results from scanning a single row."""
    frames_saved: int
    frames_skipped: int
    skipped: bool             # True if row was safety-skipped
    timing_s: dict            # timestamp dict (see Timing Metadata below)
    position_samples: list    # [{t_before, t_after, x_um, row}, ...]
    z_start_x_um: float | None

def scan_row(
    row_idx: int,
    row_y: float,
    row_x_min: float,
    row_x_max: float,
    direction: int,
    *,
    hw: ScanHardware,
    cfg: RowConfig,
    save_queue: queue.Queue,
    global_frame_idx: int,    # starting frame index for this row
    scan_t0: float,           # perf_counter at scan start (for timestamps)
) -> RowResult:
```

### main() loop after extraction

```python
for row_idx, (row_y, row_x_min, row_x_max) in enumerate(plan.rows):
    direction = 1 if row_idx % 2 == 0 else -1

    result = scan_row(
        row_idx, row_y, row_x_min, row_x_max, direction,
        hw=hw, cfg=cfg, save_queue=save_queue,
        global_frame_idx=global_frame_idx, scan_t0=total_scan_start,
    )

    if not result.skipped:
        global_frame_idx += result.frames_saved
        all_position_samples.extend(result.position_samples)
        row_timings.append(result.timing_s)

    # Console summary
    print(f"  {result.frames_saved} saved, {result.frames_skipped} skipped, ...")
```

### Recommendation

Do it. The function boundary is clean, the dataclasses are small, and the
main loop becomes trivially readable. The save_queue pattern (function puts
to caller's queue) is standard producer/consumer and doesn't need any
special handling.

## Timing Metadata

Timestamps relative to scan start (t0), stored in seconds. Same reference
frame as `position_stream[].t_before` in scan_meta.json, so you can
directly overlay position data on row timing phases.

Added to each entry in `rows_meta[]`:

```json
{
    "row_idx": 0,
    "y_um": 45000,
    "x_min_um": 30000,
    "x_max_um": 38000,
    "direction": 1,
    "frame_start": 0,
    "frame_end": 42,
    "position_samples": 312,
    "timing_s": {
        "preposition_start": 0.000,
        "preposition_end": 0.045,
        "settle_end": 0.145,
        "warmup_end": 0.298,
        "x_started": 0.299,
        "z_started": 0.496,
        "z_start_x_um": 31050.3,
        "capture_end": 1.548
    }
}
```

Notes:
- `z_started` is null if z_vel is negligible (flat plane in X)
- `z_start_x_um` is the polled X position when Z tracking was triggered
- Durations can be computed as differences: e.g., warmup = warmup_end - settle_end
- Polling thread start time is not tracked separately (sub-ms, not interesting)
- `duration_s` field (currently always null) is replaced by
  `timing_s.capture_end - timing_s.preposition_start`

## CLI Summary (after refactor)

### Removed
```
--padding
--z-preposition-mode
--z-tracking-mode
--persistent-polling
--debug-polling
```

### Added
```
--lead-in-um     float  default=2000  Lead-in distance before chip edge (um)
--z-lead-ms      float  default=50    Start Z tracking N ms before chip edge
```

### Unchanged
```
-o/--output, --chips-meta, --chip, --plane
--x-overlap-percent, --dry-run, --row-limit, --row-settle
--objective-mag, --objective-pos
--speed-mm, --move-speed-mm
--exposure-ms, --gain, --binning, --white-balance, --gamma
--downsample, --warmup-frames, --y-overlap-percent
--z-max, --clean, --write-threads, --compress
```
