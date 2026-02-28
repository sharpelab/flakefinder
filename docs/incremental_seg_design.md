# Incremental Segmentation Design

## Status: Draft (investigation complete, not yet implemented)

## Problem

The find-flakes pipeline currently runs segmentation as a batch after each
chip scan completes. Segmentation sits idle during scanning, and uploads
can't start until everything finishes. We want to overlap seg with scanning
and eventually stream results to the upload server.

## Current Architecture

```
chip_scan.run()
  Row 0: capture → saver threads write frame_0000..frame_0015.jpg
  Row 1: capture → saver threads write frame_0016..frame_0031.jpg
  ...
  Row N: last row
  flush saver queue
  smooth_frame_positions()  ← spline fit, already per-row internally
  write scan_meta.json      ← all frames, positions, Z stats
     ↓ scan complete
_run_chip_seg()  ← background ThreadPoolExecutor(1)
  read scan_meta.json for pixel_size + frame positions
  ProcessPoolExecutor(4) → process_frame() per frame
  add_stage_coords()  ← maps pixel → stage using scan_meta.frames
  write summary.json + per-frame JSONs + revisit JSONs
     ↓ seg complete
upload / revisit captures
```

## Key Insight: Everything Is Per-Row

Investigation found that all the "global" post-scan steps are actually
per-row internally:

- **`smooth_frame_positions()`**: Loops `for line in lines:`, fits a spline
  per-row using only that row's position samples. Zero cross-row state.

- **`add_stage_coords()`**: Looks up each detection's frame in
  `scan_meta["frames"]` for `(x_um, y_um, x_vel_um_s)`. These are
  computed by the saver thread using `interpolate_position(t_start,
  row_x_samples)` — row-local data.

- **`segment_frame()`**: Fully independent per frame. Takes image array,
  DetectorConfig, um_per_px. No inter-frame state.

- **Pixel size, camera params, flatfield, DetectorConfig**: All known
  before scan starts.

The only truly cross-row operation is `dedup_detections()` (spatial NMS
with 50µm radius using stage coords).

## V1: Per-Row Pipelining

**Goal:** Start segmentation during the chip scan, per-row, so seg is
~95% done when the scan finishes.

### Changes

#### 1. Extract `smooth_row_positions()` from `smooth_frame_positions()`

File: `src/flakefinder/scan_utils.py`

Extract the per-row loop body into a standalone function:

```python
def smooth_row_positions(
    frames: list[FrameMeta],
    position_samples: list[PositionSample],
) -> int:
    """Smooth x_um and x_vel_um_s for frames using position samples.

    Modifies frames in place. Returns count of frames updated.
    All inputs must belong to the same row.
    """
    # (body of the existing per-line loop)
```

`smooth_frame_positions()` becomes a thin wrapper calling this per row.

#### 2. Add `on_row_complete` callback to `chip_scan.run()`

File: `src/flakefinder/commands/chip_scan.py`

New parameter:

```python
def run(
    scope, *, ...,
    on_row_complete: Callable[[RowCompleteEvent], None] | None = None,
) -> None:
```

Where `RowCompleteEvent` is:

```python
class RowCompleteEvent(NamedTuple):
    row_idx: int
    frame_paths: list[Path]          # written JPEGs for this row
    frame_meta: list[dict]           # saved_frames_meta entries (with smoothed x_um)
    position_samples: list[dict]     # raw position stream for this row
    micro_meta: dict                 # camera/optics (same every row, for convenience)
```

After each row's capture loop, before moving to the next row:
1. `save_queue.join()` — ensure this row's frames are written to disk
2. Call `smooth_row_positions()` on this row's frame_meta entries
3. Fire `on_row_complete(event)`

The `save_queue.join()` call is the key synchronization point. It's
cheap — the saver threads run during capture and usually finish before
the row ends. Worst case it adds a few ms wait.

#### 3. Refactor `_run_chip_seg` into streaming model

File: `src/flakefinder/commands/find_flakes.py`

Split into two pieces:

```python
class _ChipSegState:
    """Accumulates segmentation results during a chip scan."""
    pool: ProcessPoolExecutor
    futures: list[Future]
    results: dict[str, FrameResult]
    frame_meta_by_row: dict[int, list[dict]]  # for stage coords
    ...

    def on_row_complete(self, event: RowCompleteEvent) -> None:
        """Submit row's frames to seg pool."""
        self.frame_meta_by_row[event.row_idx] = event.frame_meta
        for path in event.frame_paths:
            fut = self.pool.submit(process_frame, ...)
            self.futures.append(fut)

    def finalize(self, scan_meta_path: Path) -> _SegResult:
        """Wait for remaining futures, build summary, dedup."""
        # Wait for any in-flight frames
        # add_stage_coords using accumulated frame_meta
        # dedup_detections
        # Write summary.json, per-frame JSONs, revisit JSONs
```

The `chip_scan.run()` call becomes:

```python
seg_state = _ChipSegState(config, flatfield, pixel_size, ...)
chip_scan.run(scope, ..., on_row_complete=seg_state.on_row_complete)
result = seg_state.finalize(scan_meta_path)
```

#### 4. Threading model

No change to the high-level threading: `_run_chip_seg` still runs in the
background `ThreadPoolExecutor(1)`. The difference is that its
`ProcessPoolExecutor(4)` starts receiving frames during the scan instead
of after.

The `on_row_complete` callback runs in chip_scan's main thread (the scan
loop), which is the background seg thread from find_flakes' perspective.
The callback just submits work to the process pool — it's fast.

### Time Savings (V1)

For a typical chip: 60s scan, 20s seg (4 workers, ~150 frames).

- **Current**: seg starts at t=60s, finishes at t=80s. 20s tail.
- **V1**: seg starts at t≈1s (after row 1), by t=60s only last 1-2 rows
  remain. Seg finishes at t≈62s. **~18s saved per chip.**
- **Latency to first T1**: current ~80s, V1 ~3s after first row.

For last chip in an 8-chip run (no next chip to overlap with), this
saves the full ~18s. For intermediate chips, the existing background
overlap already hides most seg time, so savings are smaller (~5s).

## V2: Incremental Upload with Overlap-Band Dedup

**Goal:** Emit finalized detections (with stage coords, tier, score) as
soon as they clear dedup, enabling streaming upload.

### Spatial Dedup Analysis

- Y step = frame_height × (1 - y_overlap/100) ≈ 877µm at 20x bin3
- Dedup radius = 50µm
- Frame height ≈ 997µm at 20x bin3
- Y overlap band = ±50µm around row boundary = 100µm total

A detection's stage_y is approximately:
  `row_y + (det_center_py - frame_h/2) * um_per_px`

Detections in the center ~876µm of each row (the "safe zone") cannot
overlap with any detection from adjacent rows. Only detections within
50µm of the row's Y boundary could duplicate across rows.

### Design

```python
class _IncrementalDedup:
    """Row-by-row dedup with overlap-band buffering."""

    def __init__(self, dedup_radius_um: float, row_y_step_um: float):
        self.radius = dedup_radius_um
        self.y_step = row_y_step_um
        self._pending: dict[int, list[Detection]] = {}  # row_idx → held dets
        self._emitted: list[Detection] = []

    def add_row(self, row_idx: int, detections: list[Detection]) -> list[Detection]:
        """Add a completed row's detections. Returns newly cleared detections.

        Detections in the safe zone (>radius from row boundary) emit
        immediately. Detections in the overlap band are held until the
        next row arrives, then deduped against it.
        """
        # 1. Classify each detection as safe or overlap-band
        safe, held = self._split_by_band(row_idx, detections)

        # 2. Dedup held detections from previous row against this row's dets
        newly_cleared = []
        prev_held = self._pending.pop(row_idx - 1, [])
        if prev_held:
            # NMS: prev_held sorted by (tier, -score), suppress against
            # current row's detections within radius
            newly_cleared = self._nms_cross_row(prev_held, detections)

        # 3. Hold this row's overlap-band dets for next row
        self._pending[row_idx] = held

        # 4. Emit safe + newly cleared
        emit = safe + newly_cleared
        self._emitted.extend(emit)
        return emit

    def flush(self) -> list[Detection]:
        """Emit any remaining held detections (call after last row)."""
        remaining = []
        for dets in self._pending.values():
            remaining.extend(dets)
        self._pending.clear()
        self._emitted.extend(remaining)
        return remaining
```

### Integration with V1

The `on_row_complete` callback gains a second phase:

```
on_row_complete(row N):
  1. Submit frames to seg pool
  2. As frames complete → add_stage_coords per detection
  3. Feed detections to _IncrementalDedup.add_row(N, ...)
  4. Emit cleared detections → on_flakes_ready callback
```

The `on_flakes_ready` callback is the hook for incremental upload:

```python
def on_flakes_ready(detections: list[Detection]) -> None:
    """Called with finalized, deduped detections ready for upload."""
    # Could: stream to server, update operator dashboard, write to disk
```

### Upload Hook Points

| Hook | Trigger | Data Available |
|------|---------|----------------|
| `on_flakes_ready` | Per-row (safe zone) + per-row (prev overlap cleared) | Deduped detections with stage coords, tier, score |
| `on_revisit_captured` | After each revisit point | Revisit PNG + detection metadata |
| `on_chip_complete` | After last row flushed + revisits done | Full chip summary, all images |

### Edge Cases

- **First row**: No previous overlap to resolve. Safe-zone dets emit,
  overlap-band dets held.
- **Last row**: `flush()` emits all remaining held dets (no next row to
  dedup against). Slightly lower dedup quality at chip bottom edge —
  acceptable since chip edges have fewer good flakes anyway.
- **Detection straddles safe/overlap boundary**: Classify by detection
  center (stage_y). A large flake whose contour crosses the boundary but
  whose center is in the safe zone emits immediately — its center can't
  match a detection centered in the next row.

## Implementation Order

1. **V1 Phase 1**: Extract `smooth_row_positions()`, add `on_row_complete`
   to chip_scan, wire into `_run_chip_seg`. Dedup stays at finalization.
2. **V1 Phase 2**: Test on real scan data — verify per-row smoothed
   positions match batch-smoothed positions (they should be identical).
3. **V2 Phase 1**: `_IncrementalDedup` class + `on_flakes_ready` callback.
4. **V2 Phase 2**: Upload streaming — per-chip or per-batch upload to
   flakes.sharpelab.science as detections clear.
5. **V2 Phase 3**: Per-revisit-image upload hook in `revisit.run()`.
