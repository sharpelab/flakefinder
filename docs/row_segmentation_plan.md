# Row-Level Segmentation Overlap

## Problem

The find-flakes pipeline currently scans an entire chip, then segments all
frames at once. Segmentation is CPU-bound (~100ms/frame) and takes 30-60s per
chip. During that time the microscope is either idle or already scanning the
next chip while seg queues up. If we start segmenting frames as they hit disk
during the scan, the seg cost is hidden behind microscope motion time.

## Current Flow

```
chip_scan.run()          ──────────────────────────────►  scan_meta.json written
                                                          │
                                                          ▼
_run_chip_seg()          ──────────────────────────────►  summary.json written
  glob frame_*.jpg
  ProcessPoolExecutor(process_frame × all_frames)
  add_stage_coords (needs scan_meta.json)
  dedup_detections
  write summary.json
  generate revisits
```

Seg starts only after the full scan completes. For a 10-chip run, this adds
5-10 minutes of pure CPU wait.

## Key Insight

`process_frame()` is **100% per-frame** with zero cross-frame dependencies:

- Loads one JPEG, applies pre-computed flatfield
- Computes per-frame background modes via `histogram_mode()`
- Thresholds, labels connected components, analyzes each blob
- Classifies and scores using fixed `DetectorConfig` constants

No shared statistics, no cross-frame normalization, no global thresholds.

The only chip-wide steps are cheap post-processing:

| Step | Cost | Needs |
|------|------|-------|
| `add_stage_coords()` | <0.5s | `scan_meta.json` frame positions |
| `dedup_detections()` | <0.5s | stage coords from all detections |
| `summary.json` | <0.5s | all `FrameResult`s |
| `_generate_revisits()` | <0.5s | all detections + plane JSON |

## Proposed Flow

```
chip_scan.run() begins
  │
  ├─ row 0 complete ──► saver threads flush ──► submit row 0 frames to seg pool
  ├─ row 1 complete ──► saver threads flush ──► submit row 1 frames to seg pool
  ├─ ...                                        (seg workers run in parallel
  ├─ row N complete ──►                          with microscope scanning)
  │
  ▼
  scan_meta.json written
  │
  ▼
  Merge phase (<2s):
    wait for all seg futures
    add_stage_coords (from scan_meta.json)
    dedup_detections
    write summary.json + per-frame JSONs
    generate revisit JSONs
```

## Implementation

### 1. Row-complete callback in chip_scan.run()

Add an optional callback parameter:

```python
def run(
    scope: Microscope,
    *,
    ...
    on_row_complete: Callable[[int, int, int], None] | None = None,
    #                         row_idx, frame_start, frame_end
) -> None:
```

After each `scan_row()` returns, drain the save queue and fire the callback:

```python
# In the scan loop, after scan_row() returns:
if not result.skipped:
    global_frame_idx += result.frames_saved
    ...

# Drain saver queue so frames are on disk before signaling
save_queue.join()

if on_row_complete is not None:
    on_row_complete(row_idx, row_frame_start, global_frame_idx)
```

`save_queue.join()` blocks until all queued frames are written. This adds
~100-500ms per row but overlaps with the next row's preposition time (the
stage is moving to the next Y while savers flush).

### 2. Split _run_chip_seg into two phases

**Phase A — per-frame (during scan):**

```python
def _seg_submit_row(
    pool: ProcessPoolExecutor,
    scan_dir: Path,
    frame_start: int,
    frame_end: int,
    flatfield_str: str | None,
    config: DetectorConfig,
    pixel_size: float,
    dark_frac_cutoff: float,
) -> list[Future[FrameResult]]:
    """Submit one row's frames to the seg worker pool."""
    futures = []
    for i in range(frame_start, frame_end):
        fp = scan_dir / f"frame_{i:04d}.jpg"
        fut = pool.submit(process_frame, str(fp), flatfield_str, config, pixel_size, dark_frac_cutoff)
        futures.append(fut)
    return futures
```

**Phase B — chip merge (after scan_meta.json exists):**

```python
def _seg_merge(
    futures: list[Future[FrameResult]],
    scan_dir: Path,
    seg_dir: Path,
    ...
) -> _SegResult:
    """Collect frame results, add stage coords, build summary."""
    # Wait for all frame futures
    results = {fut.result().frame_name: fut.result() for fut in futures}

    # Write per-frame JSONs (geometry only)
    ...

    # Load scan_meta.json for stage coord mapping
    # add_stage_coords, dedup, summary.json, revisits
    ...
```

### 3. Integration in find_flakes.py

In the per-chip loop, replace the single `_run_chip_seg` submission:

```python
# Before chip scan — set up seg pool
seg_pool = ProcessPoolExecutor(max_workers=seg_jobs)
seg_frame_futures: list[Future[FrameResult]] = []

def on_row_complete(row_idx, frame_start, frame_end):
    """Callback from chip_scan — submit row's frames to seg."""
    new_futures = _seg_submit_row(seg_pool, chip_scan_dir, frame_start, frame_end, ...)
    seg_frame_futures.extend(new_futures)

# Chip scan with callback
chip_scan.run(scope, ..., on_row_complete=on_row_complete)

# Merge phase (scan_meta.json now exists)
seg_result = _seg_merge(seg_frame_futures, chip_scan_dir, seg_dir, ...)
```

### 4. pixel_size availability

`process_frame()` needs `pixel_size` (µm/px). Currently read from
`scan_meta.json`, but that doesn't exist yet during the scan. Two options:

- **Compute from microscope description** — `compute_frame_size_um()` is
  already used in `chip_scan._plan()` to get frame dimensions. Divide by
  frame pixel count to get µm/px. This is what `scan_meta.json` would
  contain anyway.
- **Read from chip_scan's `_Preflight`** — expose pixel_size on the preflight
  object so find_flakes can pass it to the callback.

The second is cleaner since `_plan()` already computes everything needed.

## Validation Strategy

1. **Unit test**: Run seg on a saved chip scan directory both ways (all-at-once
   vs row-by-row), compare `summary.json` outputs. Detection counts, tiers,
   and scores should be identical. Stage coords will be identical since they
   come from the same scan_meta.json in the merge phase.

2. **Timing comparison**: Run `find_flakes --seg-wait` (current blocking mode)
   vs the new overlap mode on the same chip. Measure wall-clock per-chip time.
   Expect seg time to be nearly hidden.

3. **Edge cases**:
   - Skipped rows (Z safety) — callback fires with frame_start == frame_end
   - Single-frame rows — still works, just one future
   - `--no-segment` — callback is None, no change
   - `--seg-wait` — could still use per-row submission but block on merge
     before next chip

## Risks

- **CPU contention**: Seg workers competing with camera acquisition thread.
  At `jobs=4` on an 8+ core machine this should be fine. The camera thread
  is I/O-bound (USB transfer) not CPU-bound. Monitor for increased frame
  skip rates.

- **save_queue.join() latency**: If saver threads are slow (disk I/O),
  join() could delay the next row's preposition. In practice saver threads
  write ~2MB JPEGs at 95% quality — well under the row transition time
  (~500ms preposition + settle). Profile on the microscope PC to confirm.

- **Memory**: Accumulating `Future[FrameResult]` objects for all frames in
  a chip. Each `FrameResult` holds detection dicts with geometry (hull,
  contour points). For a large chip (1000 frames, 500 detections) this is
  ~50MB — fine.
