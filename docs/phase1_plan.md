# Phase 1: Continuous Overview - Detailed Plan

**Goal:** Full wafer overview in < 1 minute (currently ~15 min with stop-and-shoot)

## Background

The current 2DMatGMM-System uses stop-and-shoot scanning: move to position, wait for settle, capture, repeat. With 1,700 positions, motor settle time dominates.

Continuous motion scanning keeps the stage moving at constant velocity while the camera captures at a fixed rate. This should reduce overview time from ~15 min to ~1 min.

## Hardware

| Component | Model | Key Specs |
|-----------|-------|-----------|
| Stage | Märzhäuser SCAN 100x100 | 60/120/240 mm/s (depends on ball screw pitch) |
| Camera | Leica K5C | 7 fps full-res, 32 fps @ 3×3 binning, **rolling shutter** |
| Objective | 5× | For overview scan |

**Note:** Wiki mentioned DFK 33UX174 (162 fps, global shutter) but microscope has K5C. May be a second camera - confirm with Aaron.

## Approach

Build SDK features incrementally as needed, rather than patching the existing driver.

### Step 1: Safety First

Chat with Aaron about microscope no-nos:
- [ ] Stage limits - how to avoid ramming endstops
- [ ] Z-drive limits - objective collision risks
- [ ] Command behavior while moving - queue? abort? error?
- [ ] Objective switching clearance - position restrictions?
- [ ] Power-on calibration/homing requirements
- [ ] Anything else that can damage hardware or samples

### Step 2: SDK - Velocity Control

Expose stage velocity control. The enums exist but aren't wired up:
- `IID_BASIC_CONTROL_VALUE_VELOCITY` (264)
- `IID_DIRECTED_CONTROL_VALUE_ASYNC_VELOCITY` (276)

**Target API:**
```python
stage.x_axis.velocity = 60.0  # mm/s
stage.x_axis.move_continuous(direction=1)
stage.x_axis.stop()
```

**Test:** Move stage at known velocity, time it, verify speed.

### Step 3: SDK - Continuous Capture

Expose camera continuous acquisition. Enums exist:
- `PROP_ACQUISITION_FRAMERATE` (16502)
- `PROP_ACQUISITION_FRAMERATE_ENABLED` (16503)

**Target API:**
```python
camera.binning = BinningMode.THREE_BY_THREE  # 32 fps
camera.framerate = 32
camera.start_continuous(callback=on_frame)
camera.stop_continuous()
```

**Test:** Capture frames at 32 fps, verify timing and image quality.

### Step 4: Movement Tests

Implement snake raster pattern:
```
→ → → → → → → → → →
                    ↓
← ← ← ← ← ← ← ← ← ←
↓
→ → → → → → → → → →
```

- Minimizes turnaround time vs. always returning to start
- Time a full wafer sweep at various speeds
- Find max speed before we hit issues

### Step 5: Single Line Capture

Combine continuous motion + continuous capture on a single line:
- Stage moves at constant velocity
- Camera captures at fixed interval
- Check for rolling shutter distortion
- Tune speed vs. quality tradeoff

**Using binning:** 3×3 binning gives 32 fps with lower resolution (~2 MP). For 5× overview finding chips on a wafer, this should be sufficient.

### Step 6: Full Raster Capture

Extend single line to full snake raster:
- Coordinate line direction with capture
- Handle turnarounds (stop capture? discard frames?)
- Verify coverage and overlap

### Step 7: Stitching

Assemble captured frames into single overview image.

**Simple approach:** Grid placement trusting stage positions. The stage is accurate enough that frames should align well.

**Better approach:** Overlap + blend to handle vignetting artifacts (light falloff at frame corners seen in existing stitches on microscope PC).

Options:
- Flat-field correction: capture blank image, divide out illumination pattern
- Feather blending: overlap frames and blend at edges
- Both

OpenCV has multiband blending that handles this well.

### Step 8: Chip Detection

From stitched overview, detect chip boundaries:
- Chips are distinct rectangles on the wafer
- Simple thresholding or edge detection should work
- Output: mask of chip regions for Phase 2 scanning

---

## Open Questions

- [ ] Ball screw pitch? (determines max stage speed)
- [ ] Is there a 2.5× objective? (wider FOV = fewer rows)
- [ ] Second camera (DFK 33UX174) - does it exist?
- [ ] What's the actual FOV at 5× with binning?

## Success Criteria

- [ ] Full wafer overview in < 2 minutes (stretch: < 1 min)
- [ ] Stitched image with no visible seams
- [ ] Chip boundaries detected reliably
- [ ] No damage to microscope or samples

## Dependencies

- Aaron available for safety walkthrough and first hardware tests
- Access to microscope PC
