# Autofocus Speed Improvements

## Per-Point Timing Breakdown

Typical focus map point with two-pass AF (500µm range, 5mm/s Z speed, fine pass enabled):

| Step | What | Est. time | Configurable? |
|------|------|-----------|---------------|
| 1 | XY stage move + wait | 0.1–1.0s | (depends on distance) |
| 2 | `move_settle` sleep | 0s | `--move-settle` |
| 3 | Z move to initial_z | ~0.05s | — |
| 4 | Pre-initial sleep | 0.05s | **hardcoded** |
| 5 | Flush capture (throwaway) | ~0.03s | — |
| 6 | Initial capture | ~0.03s | — |
| 7 | Z move to coarse scan start (+250µm) | ~0.05s | — |
| 8 | Pre-scan settle | **0.1s** | **hardcoded** |
| 9 | Coarse scan (500µm at 5mm/s) | ~0.1s | `--z-speed`, `--range` |
| 10 | Sharpness computation (all frames) | ~0.01s | — |
| 11 | Z move to fine scan start | ~0.06s | — |
| 12 | Pre-scan settle (fine) | **0.1s** | **hardcoded** |
| 13 | Fine scan (50µm at 1.25mm/s) | ~0.04s | `--fine-speed-factor`, `--fine-range` |
| 14 | Sharpness computation | ~0.01s | — |
| 15 | Overshoot move (+100µm) | ~0.05s | `--backlash-overshoot` |
| 16 | Post-overshoot sleep | 0.05s | **hardcoded** |
| 17 | move_to_corrected (best Z) | ~0.02s | — |
| 18 | Post-move settle | **0.2s** | `--settle-time` |
| 19 | Flush capture (throwaway) | ~0.03s | — |
| 20 | Final capture | ~0.03s | — |
| 21 | After capture (if --save-images) | ~0.03s | — |
| | **Total** | **~1.0–1.5s** | |

### Fixed sleep budget: ~500ms per point

| Sleep | Location | Default | Purpose |
|-------|----------|---------|---------|
| 50ms | Before initial capture | `autofocus.py:527` | Settle after Z move to initial |
| 100ms | Before coarse scan | `_run_z_scan:372` | Settle at scan start |
| 100ms | Before fine scan | `_run_z_scan:372` | Settle at fine scan start |
| 50ms | After overshoot move | `autofocus.py:638` | Settle before corrected move |
| 200ms | After corrected move | `autofocus.py:640` | Settle before final capture |

With 33 points, that's **~16.5s of sleeping** out of a ~40s focus map.

## Test Plan

All tests at c09 (36280, 30210), z=24700, with `--debug-dir` + `-o` for before/after comparison.

### Baseline
```bash
uv run python autofocus_demo.py --x 36280 --y 30210 --z 24700 --fine --debug-dir af_test/baseline -o af_test/baseline -q
```

### Test 1: Settle time (200ms → 50ms, 0ms)
```bash
uv run python autofocus_demo.py --x 36280 --y 30210 --z 24700 --fine --settle-time 0.05 --debug-dir af_test/settle_50ms -o af_test/settle_50ms -q
uv run python autofocus_demo.py --x 36280 --y 30210 --z 24700 --fine --settle-time 0 --debug-dir af_test/settle_0ms -o af_test/settle_0ms -q
```
**Hypothesis**: With flush capture already handling stale buffers, 200ms settle may be unnecessary. The corrected move command waits for completion internally.

### Test 2: Backlash overshoot (100µm → 50µm, 0µm)
```bash
uv run python autofocus_demo.py --x 36280 --y 30210 --z 24700 --fine --backlash-overshoot 50 --debug-dir af_test/overshoot_50 -o af_test/overshoot_50 -q
uv run python autofocus_demo.py --x 36280 --y 30210 --z 24700 --fine --backlash-overshoot 0 --debug-dir af_test/overshoot_0 -o af_test/overshoot_0 -q
```
**Hypothesis**: `move_to_corrected` already uses SDK hysteresis compensation. The overshoot may be redundant — it was added before the corrected move was implemented.

### Test 3: No fine pass
```bash
uv run python autofocus_demo.py --x 36280 --y 30210 --z 24700 --debug-dir af_test/no_fine -o af_test/no_fine -q
```
**Hypothesis**: Coarse pass at 5mm/s with 1ms exposure gives ~2µm Z sampling. Fine pass adds ~250ms for marginal improvement. May not matter for focus maps where the plane fit averages out per-point noise anyway.

### Test 4: Hardcoded sleeps (requires code changes)

Cut the three hardcoded sleeps and run:

| Change | From | To | Saves |
|--------|------|----|-------|
| Pre-initial sleep | 50ms | 0ms | 50ms × 1 |
| Pre-scan settle | 100ms | 0ms | 100ms × 2 |
| Post-overshoot | 50ms | 0ms | 50ms × 1 |

```bash
uv run python autofocus_demo.py --x 36280 --y 30210 --z 24700 --fine --debug-dir af_test/no_sleeps -o af_test/no_sleeps -q
```
**Hypothesis**: The flush capture before initial/final already handles stale buffers. The pre-scan settle was added conservatively early on — the scan itself starts with a Z move which has its own completion wait. The post-overshoot 50ms is before another blocking `move_to_corrected`.

## Expected Savings

| Optimization | Per-point savings | 33-point savings |
|-------------|-------------------|------------------|
| Settle 200ms → 0ms | 200ms | 6.6s |
| Overshoot 100 → 0µm | ~100ms (move + sleep) | 3.3s |
| No fine pass | ~250ms | 8.3s |
| Cut hardcoded sleeps | ~250ms | 8.3s |
| **All combined** | **~800ms** | **~26s** |

Best case: focus map drops from ~45s to ~20s.

## Evaluation Criteria

For each test, compare:
1. `selected_sharpness` — did AF find the same best Z?
2. `final_sharpness` vs `after_sharpness` — is the final capture reliable?
3. Visual comparison of `initial.png`, `final.png`, `after.png` — any blur?
4. Z position consistency across repeated runs at same point
