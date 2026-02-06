# Continuous Focus Tracking During X Motion Scanning

## Overview

This document explores approaches for maintaining focus during continuous X motion scanning, using a pre-computed focus map approximated as a plane.

**Current state:**
- `focus_map.py` samples Z focus at grid points, `analyze_focus_map.py` fits a plane: `Z = aX + bY + c`
- `scan_area_v1.py` does continuous X motion capture but uses **constant Z** throughout (single autofocus at start)
- This works for flat samples but fails when sample tilt exceeds depth of field (~4µm at 20x)

**Goal:** Continuously adjust Z during X scanning to follow the focus plane.

---

## 1. Hardware Capabilities

### Speed Constraints

| Axis | Max Speed | Notes |
|------|-----------|-------|
| X    | 40 mm/s   | Used for scanning |
| Y    | 40 mm/s   | Used for row positioning |
| Z    | 5 mm/s    | Focus adjustment |

### Required Z Velocity During X Scan

From the chip0 focus map analysis:
- Tilt magnitude: ~1.5 µm/mm (typical)
- At X=40mm/s scanning speed: **Z velocity needed = 40mm/s × 1.5µm/mm = 60 µm/s = 0.06 mm/s**

This is **83x below Z max speed** (5mm/s), so Z can easily keep up with X motion.

### Position Polling

From MEMORY.md learnings:
- Position polling: ~124 Hz with 2 threads (SDK call ~16ms)
- Position jitter: 40-100µm raw, smoothable with Savitzky-Golay filter
- Frame timing: use `t_start` for position (rolling shutter)

### Concurrent Motion

The SDK supports concurrent async moves on multiple axes:
```python
handle_x = stage.x.move_to_async(x_end)
handle_z = z_drive.move_to_async(z_end)  # Can run simultaneously
```

---

## 2. Focus Plane Model

### Plane Equation

From `analyze_focus_map.py`:
```python
Z = a*X + b*Y + c
# coefficients in µm units
```

For a row at fixed Y:
```python
Z(X) = a*X + (b*Y_row + c)
     = a*X + Z_row_offset
```

This is a simple linear ramp in Z as X changes.

### Typical Values (from chip0 data)

- Z range across chip: ~85µm (24671-24756µm)
- Chip X span: ~27mm
- Tilt: ~3 µm/mm along diagonal
- Plane R²: typically >0.95 (good fit)

### Per-Row Z Change

For a 90mm X scan at 1.5µm/mm tilt:
- Total Z change per row: ~135µm
- Direction alternates with snake pattern (+X row: +ΔZ, -X row: -ΔZ)

---

## 3. Implementation Approaches

### Approach A: Synchronized Linear Ramp (Recommended)

**Concept:** Start both X and Z moves simultaneously with velocities calculated so they finish together.

```
Row scan time = X_distance / X_velocity
Z_velocity = Z_change / Row_scan_time
```

**Procedure:**
1. Before each row, calculate:
   - `X_start`, `X_end` (row endpoints)
   - `Z_start = a*X_start + b*Y_row + c`
   - `Z_end = a*X_end + b*Y_row + c`
2. Set Z velocity proportional to X velocity: `Z_vel = X_vel * a`
3. Start both async moves simultaneously
4. Poll `handle_x.is_complete` to detect row end
5. Capture frames throughout

**Pros:**
- Simple to implement
- No closed-loop feedback needed
- Uses existing async move infrastructure

**Cons:**
- Assumes perfect velocity tracking (no slip)
- Doesn't adapt to actual position

**Code sketch:**
```python
# Per-row setup
z_start = plane_a * x_start + plane_b * y_row + plane_c
z_end = plane_a * x_end + plane_b * y_row + plane_c

# Set Z velocity to match X ramp
z_vel = abs(scan_speed_um_s * plane_a)  # µm/s
z_drive.set_velocity_um_s(z_vel)

# Start synchronized moves
z_drive.move_to(z_start)  # Position at row start
handle_x = stage.x.move_to_async(x_end)
handle_z = z_drive.move_to_async(z_end)

# Capture during motion (both axes moving)
while not handle_x.is_complete:
    capture_frame()
```

### Approach B: Closed-Loop Position Tracking

**Concept:** Continuously poll X position, compute desired Z, send incremental Z commands.

```
while scanning:
    x_current = stage.x.position_um
    z_target = a*x_current + b*y_row + c
    z_drive.move_to(z_target)  # Non-blocking incremental
```

**Pros:**
- Adapts to actual X position
- Handles velocity variations

**Cons:**
- Requires very fast command loop
- SDK latency (~16ms per call) may cause lag
- Risk of Z oscillation if commands pile up
- More complex implementation

**Timing analysis:**
- X at 40mm/s moves 640µm in 16ms
- At 1.5µm/mm tilt, Z needs to change 0.96µm per 16ms
- With DOF of 4µm at 20x, this could work but margins are tight

### Approach C: Segmented Rows

**Concept:** Break each row into segments, adjust Z at segment boundaries.

```
X row: [-------segment 1-------][-------segment 2-------][...etc...]
              ↓ stop, adjust Z        ↓ stop, adjust Z
```

**Pros:**
- Uses proven stop-and-move pattern
- Guaranteed Z accuracy at segment starts

**Cons:**
- Interrupts continuous motion (slower)
- More complex capture logic
- Segment boundaries may show discontinuities in images

### Approach D: Predictive Feed-Forward

**Concept:** Like Approach A, but add small corrections based on position feedback.

```
z_commanded = z_ramp(t) + k*(x_expected - x_actual) * plane_a
```

**Pros:**
- Best of both worlds: mostly open-loop with corrections

**Cons:**
- More complex
- Requires tuning gain k
- Overkill if Approach A works well

---

## 4. Timing Considerations

### Frame-to-Position Association

Current approach in `scan_area_v1.py`:
1. Poll X position in background thread → `x_samples[]`
2. Capture frame → timestamp `t_start`, `t_end`
3. Post-process: interpolate X at `t_start` using `x_samples`

For focus tracking, we also need:
- Associate each frame with its (X, Z) position
- Decide: use commanded Z or polled Z?

**Recommendation:**
- Poll both X and Z during scan
- Record actual positions for metadata accuracy
- Small deviations from commanded Z don't affect image quality (within DOF)

### Timing Jitter

From MEMORY.md:
- SDK hiccup at ~60% through motion at high speeds
- Startup warmup causes anomalies in first few frames

**Mitigations:**
- Continue using warmup frames before each row
- Accept that some frames may be slightly out of focus
- Post-filter frames with duration outside mean ± 2*std

---

## 5. Recommended Implementation Plan

### Phase 1: Synchronized Linear Ramp (Approach A)

1. **Add focus plane to scan parameters:**
   ```python
   --focus-plane a,b,c    # From analyze_focus_map.py
   --focus-map PATH       # Load from focus_map JSON directly
   ```

2. **Modify row scan loop:**
   - Calculate Z endpoints from plane equation
   - Set Z velocity to `X_velocity * |plane_a|`
   - Start X and Z moves simultaneously
   - Poll both X and Z during capture

3. **Update metadata:**
   - Record actual Z positions per frame
   - Store focus plane parameters

4. **Test with moderate tilt:**
   - Start with slow X speed (10mm/s)
   - Verify Z tracking visually
   - Increase speed incrementally

### Phase 2: Validation and Refinement

1. **Compare focus quality:**
   - Scan same area with/without tracking
   - Measure sharpness across frame sequence
   - Check for systematic defocus patterns

2. **Add closed-loop correction (if needed):**
   - Only if Phase 1 shows systematic drift
   - Implement Approach D with small correction gain

### Phase 3: Integration

1. **Auto-derive focus plane from focus map:**
   ```python
   # Load focus map, fit plane, use directly in scan
   focus_plane = fit_plane_from_focus_map(focus_map_path)
   ```

2. **Integrate with chip detection workflow:**
   - 5x overview scan (no focus tracking needed, large DOF)
   - Detect chips
   - Per-chip focus map
   - 20x scan with focus tracking

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Z can't keep up with X | Very Low | High | Calculate before scan; Z has 83x headroom |
| Timing mismatch between axes | Low | Medium | Use synchronized start; verify with position logging |
| Plane model inadequate (curved surface) | Low | Medium | Check R² before scan; fall back to segmented if <0.9 |
| SDK concurrent motion issues | Low | Medium | Test incrementally; have fallback to segmented |

---

## 7. Open Questions

1. **Does the SDK support setting Z velocity independently of move commands?**
   - `units.py` shows `set_velocity_um_s()` exists
   - Need to verify it affects subsequent `move_to_async()` calls

2. **What happens if Z move finishes before X move?**
   - Should be fine (Z will hold position)
   - But we want them synchronized; may need velocity tuning

3. **Should we poll Z position or trust commanded position?**
   - For metadata: poll actual
   - For control: commanded is fine (no feedback loop)

4. **How to handle focus plane with Y component during snake scan?**
   - Each row has different Y, so different Z_offset
   - Already handled: `Z = a*X + (b*Y_row + c)`

---

## 8. Appendix: Focus Map Data Summary

From `scans/focus_map_chip0.json`:
- 26 sample points (8 contour + 18 grid)
- X range: 11,070 - 37,901 µm (~27mm)
- Y range: 7,050 - 21,797 µm (~15mm)
- Z range: 24,671 - 24,756 µm (~85µm variation)
- Plane fit coefficients (from analysis): ~1.5 µm/mm tilt

At 20x objective:
- DOF: ~4µm
- Z variation (85µm) is ~21x DOF
- **Focus tracking is essential for acceptable image quality**
