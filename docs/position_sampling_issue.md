# Position Sampling Issue

## Summary

The position data in `scan_meta.json` has jitter that causes stitching misalignment. The timestamps are consistent but the position values lag/jump unpredictably.

## Evidence

From `test_scan/scan_meta.json`, analyzing position samples during constant-velocity scanning (~40 mm/s):

| Sample | t (s) | dt (ms) | x (µm) | dx (µm) |
|--------|-------|---------|--------|---------|
| 31 | 0.510846 | - | 10081.0 | - |
| 32 | 0.526827 | 16.0 | 10696.1 | 615 |
| 33 | 0.542903 | 16.1 | 11303.9 | 608 |
| 34 | 0.558888 | 16.0 | 11932.2 | 628 |
| 35 | 0.574928 | 16.0 | 12403.2 | **471** ← low |
| 36 | 0.590805 | 15.9 | 13095.5 | 692 |
| 37 | 0.606796 | 16.0 | 13782.6 | 687 |
| 38 | 0.622809 | 16.0 | 14543.8 | **761** ← high |
| 39 | 0.638775 | 16.0 | 14949.0 | **405** ← low |
| 40 | 0.654816 | 16.0 | 15628.3 | 679 |
| 41 | 0.670741 | 15.9 | 16307.7 | 679 |
| 42 | 0.686803 | 16.1 | 16991.2 | 683 |
| 43 | 0.702831 | 16.0 | 17678.7 | 688 |
| 44 | 0.718844 | 16.0 | 18124.8 | **446** ← low |
| 45 | 0.734741 | 15.9 | 18836.9 | 712 |

**Key observations:**
- **Timestamps are rock solid**: dt = 16.0ms ± 0.1ms
- **Position deltas are jittery**: dx ranges from 405-761 µm (expected ~640 µm at 40 mm/s)
- **Pattern**: Low dx often followed by high dx (lag then catch-up)

## Impact on Stitching

Frame positions are interpolated from these samples. When a position sample is wrong, the interpolated frame position is wrong, causing visible misalignment in overlapping regions even though the actual stage motion is smooth.

Example: Frames 39-41 don't stitch well because they span samples 38-39 which have the 761→405 µm jump.

## Likely Cause

The position readout (`x_bcv.GetControlValue()` → `x_converter.GetMetricsValue()`) has variable latency. When polled, it sometimes returns a slightly stale position, then on the next poll returns a position that "jumps ahead" to compensate.

## Questions to Investigate

1. Is there a way to get timestamped position data from the stage controller (position + when it was actually sampled)?
2. Is there a higher-frequency position readout available?
3. Is there a way to synchronize position reads with frame captures more tightly?
4. Does the SDK offer position interpolation or velocity data directly?

## Current Sampling Code

From `scan_row_v1.py`:

```python
def x_poll_thread():
    """Background thread: busy-poll X position."""
    while not stop_polling.is_set():
        t = time.perf_counter()
        x_native = x_bcv.GetControlValue()
        x_um = x_converter.GetMetricsValue(x_native)
        x_samples.append((t, x_um))
```

The timestamp is taken *before* the position read, so any latency in `GetControlValue()` isn't accounted for.

## Update: t_before/t_after Analysis

New scan data now records `t_before` and `t_after` for each position sample.

| Sample | t_before   | t_after    | read_ms |   x (µm)   |  dx (µm) |
|--------|------------|------------|---------|------------|----------|
|   31   | 0.511030   | 0.527001   |  16.0   |    10178.4 |      -   |
|   32   | 0.527055   | 0.542996   |  15.9   |    10595.6 |   417 ←  |
|   33   | 0.543047   | 0.558988   |  15.9   |    11237.0 |   641    |
|   34   | 0.559051   | 0.574974   |  15.9   |    11866.9 |   630    |
|   35   | 0.575045   | 0.591016   |  16.0   |    12589.1 |   722    |
|   36   | 0.591081   | 0.606976   |  15.9   |    12986.0 |   397 ←  |
|   ...  | ...        | ...        |  ...    |    ...     |   ...    |

**Key findings:**

1. **Read time is rock solid**: 15.8-16.1ms (avg 15.9ms)
2. **Position jitter persists**: dx ranges 397-722 µm (expected ~640 µm)
3. **No correlation** between read time and dx jitter
4. **Using t_before, t_after, or midpoint makes no difference** - velocity std dev is 10.3 mm/s regardless

**Conclusion:** The jitter is in the position values returned by the SDK, not our timing. The SDK's `GetControlValue()` blocks for ~16ms (suspiciously close to camera readout time of 15ms) and returns a position that was sampled at some unknown point during that window.

## Possible Solutions

1. **SDK-level**: Find a different API that returns timestamped position data
2. **Hardware-level**: Check if stage controller has a position stream/buffer feature
3. **Workaround**: Use frame-to-frame feature matching to refine positions post-capture
4. **Reduce overlap**: Accept some stitching error and use blending to hide seams
