# Z Lead-In Ramp Plan

Eliminate the blocking Z preposition during chip scan rows by ramping Z to the correct position and velocity during the X lead-in traverse.

## Problem

Each chip scan row has ~670ms of overhead before useful data capture:

| Phase | Duration | Notes |
|-------|----------|-------|
| Z preposition (blocking) | ~400ms | `move_to_corrected()` blocks even for 1µm moves |
| Settle | 100ms | `sleep(row_settle)` |
| Lead-in traverse | 400ms | 2mm at 5mm/s — Z sits idle |

The blocking Z preposition dominates overhead. Meanwhile, the 400ms lead-in traverse is wasted time where Z could be actively ramping to the correct height.

## Approach

Replace the blocking Z preposition + idle lead-in with active Z control during lead-in, in three phases.

### Phase 1: Async Z preposition

Replace `z_drive.move_to_corrected(z_start)` (blocking, ~400ms) with `z_drive.move_to_async(z_leadin_target)` (non-blocking, ~50ms) running concurrent with X/Y moves.

`z_leadin_target` is the focus plane extrapolated to the lead-in start position:
```
z_leadin_target = plane_a * x_start_pos + plane_b * row_y + plane_c
```

This gets Z within ~1-2µm of where it needs to be.

### Phase 2: Active Z control loop during lead-in

After X starts, run a proportional + feedforward controller on each capture loop iteration while X is in the lead-in region:

```
z_target = compute_plane_z(plane_a, plane_b, plane_c, x_now, row_y)
z_error = z_target - z_actual
v_total = z_vel_tracking + K_p * z_error
```

- **Feedforward** (`z_vel_tracking = plane_a * direction * x_speed`): steady-state tracking velocity (~2.5 µm/s)
- **Proportional** (`K_p * z_error`): closes the position gap from preposition error
- Updates every ~25ms (camera frame rate) → ~16 corrections during 400ms lead-in
- K_p = 10 s⁻¹ → settles 1µm error in ~200ms, well within lead-in time
- Velocity capped at 50 µm/s (conservative; typical values 2-15 µm/s)

### Phase 3: Chip edge handoff

At the `z_lead_s` trigger (30ms before chip edge):
1. Measure actual X speed (existing `measure_x_cruise_speed()`)
2. Switch from P+FF controller to pure feedforward at corrected tracking velocity
3. Z tracking continues as before — no change to chip traverse behavior

### Phase 4: Eliminate settle

Change `--row-settle` default from 0.1s to 0.0s. The active Z control during lead-in provides settling. Users can still override via CLI.

## Key Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **Frame 1 Z error** | z_error of first chip (non-lead-in) frame | Lead-in ramp positioning accuracy |
| **Z drift** | abs(mean(first 5 chip z_errors) - mean(last 5 chip z_errors)) | Tracking quality over the row |

## Baseline (run_20260211_1756, 42 rows at 5mm/s)

| Metric | Mean | Max | P95 |
|--------|------|-----|-----|
| Frame 1 Z error | 0.045 µm | 0.127 µm | 0.124 µm |
| Z drift | 0.258 µm | 0.556 µm | 0.499 µm |
| Overall Z tracking | 0.008 µm mean, 0.219 std | 0.680 µm | 0.414 µm |

Current per-row overhead: ~670ms (prepos ~470ms + settle 100ms). Target: ~100ms.

## Expected Savings

| Phase | Per-row savings | Notes |
|-------|----------------|-------|
| Phase 1 (async Z) | ~350ms | 470ms blocking → ~50ms async (concurrent with X/Y) |
| Phase 4 (no settle) | ~100ms | Default 0.1s → 0.0s |
| **Total** | **~450ms/row** | |

For 42 rows: ~19s saved. Phases 2-3 add no time (run during existing lead-in).

## Implementation Plan

### Step 1: Baseline test
5-row chip scan with current code to establish per-test baseline metrics.

### Step 2: Phase 1 + 4 (kill delays)
Async Z preposition + remove settle. Run 5-row test. Validates timing savings and checks whether Z accuracy degrades without the blocking corrected move.

### Step 3: Phase 2 + 3 (add control loop)
Active Z control during lead-in + chip edge handoff. Run 5-row test. Should recover any accuracy lost in step 2.

## Code Changes

All in `commands/chip_scan.py`:

| Area | Change |
|------|--------|
| `RowConfig` | Add `z_kp: float` (proportional gain, default 10.0) |
| `RowResult` | Add `z_frame1_error_um`, `z_drift_um` |
| `scan_row()` preposition | `move_to_async(z_leadin_target)` replacing `move_to_corrected(z_start)` |
| `scan_row()` after X start | Start Z directed velocity at tracking speed immediately |
| `scan_row()` capture loop | P+FF control loop during lead-in; chip-edge trigger becomes velocity correction |
| `run()` summary | Compute and print frame1/drift metrics per-row and aggregate |
| `_build_parser()` | `--row-settle` default 0.1→0.0; add `--z-kp` (default 10.0) |

## Reference Values

From current scan session:
- Focus plane: `Z = 1.047*X_mm + -2.013*Y_mm + 24717.05`
- Typical dZ/dX: ~1.0 µm/mm → tracking velocity ~5.0 µm/s at 5mm/s X speed
- Z between rows: ~1-2 µm (snake scan, adjacent sides)
- 20x DOF: ~1.7 µm
- Z speed limit: 5000 µm/s max

---

*Plan date: 2026-02-12*
