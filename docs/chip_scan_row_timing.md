# Chip Scan Row Timing Analysis

Analysis of per-row overhead in `commands/chip_scan.py` based on scan data from `chip0_20x_0211b` (42 rows, 3498 frames, 397s at 5 mm/s).

## Per-Row Time Budget (steady-state, rows 4–38)

| Phase | Duration | Notes |
|-------|----------|-------|
| Cleanup (prev row end) | ~130ms | `z_drive.halt()`, thread joins, `set_velocity` |
| **Z preposition (blocking)** | **~400ms** | `move_to_corrected()` blocks ~400ms even for 1µm moves |
| X/Y preposition | ~50ms | Async, hidden behind Z block |
| Settle | 100ms | Hard-coded `sleep(row_settle)` |
| Warmup | ~30ms | 3 camera frames |
| X start | ~10ms | `set_velocity` + `move_to_async` |
| **Lead-in traverse** | **400ms** | 2mm at 5mm/s |
| **Chip traverse** | **8160ms** | 40.8mm at 5mm/s (physics) |
| Accel/decel | ~360ms | Stage ramp-up/down (physics) |
| **Total** | **~9640ms** | Observed avg: 9700ms |

**Current overhead: 670ms/row (7.0% of row time)**

## Findings

### 1. Blocking Z preposition dominates overhead (~400ms/row)

`z_drive.move_to_corrected()` uses `SetControlValue` on the hysteresis-corrected BCV interface, which blocks for ~400ms even when moving only ~1µm between rows. The SDK waits for the piezo to settle and confirm position.

`ZDrive` inherits `move_to_async()` from `Axis`, but `move_to_corrected` has no async variant — the hysteresis BCV only exposes blocking `SetControlValue`, not `SetControlValueAsync`.

**Possible mitigation:** Use regular `move_to_async()` for inter-row repositioning instead of `move_to_corrected()`. Hysteresis correction matters for tracking accuracy during continuous motion, not for coarse positioning before a row starts. This could drop Z preposition to ~50ms (matching X/Y), making overhead negligible. Needs validation that Z tracking start isn't affected.

### 2. Settle time (100ms) is likely redundant

The blocking Z move already includes ~400ms of settling. The subsequent 100ms `sleep` is on top of that. `scan.py` uses zero settle time. The 2mm lead-in provides additional damping time before any real chip data is captured.

### 3. Lead-in (2mm) is generous

The lead-in serves two purposes: (a) reach cruise speed (~50–100ms at 5mm/s) and (b) measure actual X speed for Z velocity correction (~5 samples needed). 1000µm at 5mm/s (200ms) would likely suffice for both.

### 4. Cleanup gap (~130ms) between rows

After capture: `z_drive.halt()` (blocking SDK call), `stop_polling.set()`, two `thread.join(1.0)` calls (each waits up to ~16ms for current SDK poll), and `set_velocity_um_s()`. Variable 66–282ms.

### 5. Accel/decel (~360ms) is physics

Stage starts from rest, accelerates to 5mm/s, then decelerates to rest. Consistent across rows, not optimizable.

## Comparison with `scan.py`

`scan.py` achieves ~3.5s/row (26 rows, 90s) at 40mm/s because:

- **No Z tracking** — no Z move, no Z halt, no lead-in
- **No settle** — 0ms
- **Speed = move speed** — no velocity switching
- **Simple reposition** — `move_to_async(x_start, row_y)` + `wait_all` + warmup

## Optimization Opportunities

### Parameter changes (no code restructuring)

| Change | Savings/row | Total (42 rows) |
|--------|-------------|------------------|
| Reduce settle 100ms → 0ms | 100ms | ~4s |
| Reduce lead-in 2000 → 1000µm | 200ms | ~8s |

### Code changes

| Change | Savings/row | Total (42 rows) | Notes |
|--------|-------------|------------------|-------|
| Thread Z `move_to_corrected`, overlap with settle+warmup | ~260ms | ~11s | settle+warmup+cleanup run during Z's 400ms block |
| Use `move_to_async` instead of `move_to_corrected` for preposition | ~350ms | ~15s | Drops Z preposition to ~50ms; needs validation |
| Overlap cleanup with next row's preposition | ~80ms | ~3s | Restructure row loop |

### Net effect

| Scenario | Overhead/row | Overhead % | Scan time |
|----------|-------------|------------|-----------|
| Current | 670ms | 7.0% | 397s |
| Thread Z + drop settle + overlap cleanup | 410ms | 4.4% | ~386s |
| Use async Z + drop settle + reduce lead-in | ~100ms | 1.1% | ~375s |

The dominant cost is physics: 40.8mm ÷ 5mm/s = 8.16s. Even eliminating all overhead saves ~28s (7%). The real lever for scan speed is increasing X velocity — but that requires validating Z tracking accuracy at higher speeds (see `docs/continuous_autofocus_plan.md`).

---

*Analysis date: 2026-02-11. Data: `scans/chip0_20x_0211b/scan_meta.json`.*
