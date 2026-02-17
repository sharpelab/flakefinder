# Poll Throttling Plan: 1ms FTDI Safety

## Problem

At 1ms FTDI (cmusb.dll), tight position polling loops saturate the
per-axis USB bus resource, preventing `AsyncResult.GetState()` from
returning COMPLETED — async moves hang forever. The current codebase
works at 16ms FTDI because polling is slow enough (~63 Hz) that moves
naturally get bus time between polls.

See `docs/sdk_threading_investigation.md` for experimental results.

## Phase 1: `timeBeginPeriod(1)` in Microscope (done)

Add Windows timer resolution boost to `Microscope.__enter__()` /
`__exit__()` so every script gets 1ms `time.sleep()` precision for
free. Without this, rate-limited sleeps round up to ~15ms.

## Phase 2: `poll_position()` helper

New module `src/flakefinder/leica/polling.py` with a single function:

```python
def poll_position(
    bcv, converter, samples: list[PositionSample], stop: threading.Event,
    *, target_hz: float = DEFAULT_POLL_HZ,
) -> None:
```

- Thread-target function: poll, convert, append PositionSample, sleep
  to stay at target_hz
- DEFAULT_POLL_HZ = 100.0
- At cmusb (~4.8ms/call): ~5ms sleep, ~100 Hz
- At 16ms FTDI: call exceeds target period, no sleep, ~63 Hz unchanged
- Export from `flakefinder.leica.__init__`

## Phase 3: Update polling loops

Replace the 4 inline poll loops with `poll_position()` calls:

1. `commands/scan.py` x_poll_thread — same-axis X poll during X move ✅
2. `commands/chip_scan.py` x_poll_fn — same-axis X poll during X move ✅
3. `commands/chip_scan.py` z_poll_fn — Z poll during X move + Z directed vel ✅
4. `src/flakefinder/leica/autofocus.py` z_poll_thread — same-axis Z poll during Z move ✅

All 4 now use `start_motion_polling` / `start_polling` from `polling.py`.
Autofocus also replaced `is_complete` loop termination with position-based
termination (stop margin 0.1µm for fine Z scans).

### Sample format migration

- scan.py x_samples: `list[tuple]` → `list[PositionSample]` ✅
- chip_scan.py z_samples: `list[tuple[float, float]]` → `list[PositionSample]` ✅
- autofocus.py z_samples: `list[tuple[float, float, float]]` → `list[PositionSample]` ✅
  — `interpolate_position` works unchanged (NamedTuple supports index access)

## Starvation risks addressed

| Loop | Axis polled | Concurrent move | Risk | Fix |
|------|-------------|-----------------|------|-----|
| scan.py X | X | X async | HANG | 100 Hz throttle |
| chip_scan.py X | X | X async | HANG | 100 Hz throttle |
| chip_scan.py Z | Z | X async + Z directed | Low (halt delay) | 100 Hz throttle |
| autofocus.py Z | Z | Z async | HANG | 100 Hz throttle |

## Already safe (no changes needed)

- `wait_all()` / `MoveHandle.wait()`: built-in 10ms sleep
- Capture loop `is_complete` checks: replaced with position-based termination
  in scan.py, chip_scan.py, and autofocus.py
- `focus_map.py`: no concurrent polling during moves
- `find_flakes.py`: pure orchestrator
- Camera fps: independent of polling (confirmed experimentally)
