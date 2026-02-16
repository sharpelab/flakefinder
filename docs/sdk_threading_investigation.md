# SDK Threading Investigation

## Goal

100 Hz position polling on all three axes (X, Y, Z) simultaneously,
full ~60 fps camera capture, and no broken SDK functionality (async
move waits must still complete). Currently chip_scan runs 2 poll
threads (X + Z) + capture at ~20 fps. Adding Y polling and pushing
camera to 60 fps increases SDK contention — need to find the limits.

## Why

Chip scan currently tracks X and Z but assumes Y is static per row.
Supporting X+Y stage moves (diagonal or curved paths) requires Y
position data at the same rate as X. And 60 fps capture at 3x3
binning is the camera's hardware limit — we want to use all of it.

## Known Constraints

- SDK `GetControlValue()` call takes ~16 ms (measured 2026-02-05)
- 2 threads polling: 124 Hz aggregate, ~62 Hz each, 100% efficient
- `timeBeginPeriod(1)` required for good timer resolution on Windows
- **Starvation bug**: saturating the SDK with many polling threads
  prevents `AsyncResult.GetState()` from returning COMPLETED — async
  move waits hang forever. Threshold unknown.
- Camera `Acquire()` is a blocking SDK call that also competes for
  whatever internal resource the polling uses.

## Characterization Plan

### Experiment 1: Polling throughput vs thread count

N threads all calling `GetControlValue()` in tight loops on X axis.
Measure aggregate rate and per-thread rate for N = 1, 2, 4, 8.
Determines: does throughput plateau (shared resource) or scale?

### Experiment 2: Move completion under polling load

Start N polling threads, then `move_to_async` + `wait(timeout)`.
Find the N where moves stop completing. Tests same-axis (poll X,
move X) first.

### Experiment 3: Cross-axis interference

Poll axis A while moving axis B. Does polling X prevent Y moves
from completing? Does polling Z interfere with X moves?

### Experiment 4: Sleep between polls (starvation mitigation)

If move completion breaks at N threads, does adding `time.sleep(K)`
between polls fix it? Find minimum K. This distinguishes pure
starvation (fixable with rate limiting) from fundamental limits.

### Experiment 5: Camera + polling combined

The real workload: N position poll threads + camera `Acquire()` in a
loop. Does capture interfere with polling? Does polling prevent
capture? What's the max sustainable fps with 3 poll threads running?

### Experiment 6: Combined full workload

3 poll threads (X, Y, Z) + continuous capture + async move.
The actual chip_scan scenario. Does it work? What are the rates?

### Test Script

`scripts/sdk_threading_test.py` — runs experiments 1-4.
Experiments 5-6 TBD (need camera integration).

```
uv run python scripts/sdk_threading_test.py
uv run python scripts/sdk_threading_test.py --experiments 1,2 --threads 0,1,2,4
```

---

## Log

*(append-only — newest entries at bottom)*
