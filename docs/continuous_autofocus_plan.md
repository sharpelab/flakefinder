# Continuous Z-Scan Autofocus Plan

## Overview

Instead of step-and-capture (move Z, stop, capture, measure, repeat), continuously move Z while capturing frames and find the sharpness peak. Potentially much faster than discrete stepping.

## Process Flow

```
1. Start at current Z (or specified start position)
2. Begin continuous Z motion downward through focus range
3. Capture frames continuously during motion (like X-axis scanning)
4. Poll Z position with timestamps (same as X polling)
5. Compute sharpness (Sobel/Tenengrad) for each frame
6. Interpolate Z position for each frame's capture time
7. Find Z with maximum sharpness
8. Move to best Z
```

## Crash Prevention

**The danger:** Moving Z too far up (toward sample) crashes objective into sample.

**Safeguards:**

| Method | Description |
|--------|-------------|
| **Software limits** | Z axis has min/max from SDK - never exceed |
| **Upper bound parameter** | User specifies max safe Z (or use current + margin) |
| **Direction** | Always scan downward (away from sample) first, then up |
| **Sharpness monitoring** | If sharpness drops to ~0 and stays there, we've gone past focus - stop early |
| **Known starting point** | Start from a Z where sample is approximately in focus (manual or previous focus) |

**Safe scanning approach:**
```
1. User provides Z_max (known safe upper limit) or we use current_Z + small_margin
2. Scan from Z_max downward to Z_min
3. Find peak sharpness
4. Single targeted move to best Z (upward, known safe)
```

## Time Cost Estimate

**Parameters (verified from SDK):**
- Z travel speed: **5000 µm/s (5 mm/s)** max
- Focus range for 5x objective: ~50-100 µm depth of field, scan maybe 500 µm to be safe
- Frame rate: 30-60 fps at 3x3 binning

**Calculation:**
```
Z range: 500 µm
Z speed: 5000 µm/s (max)
Scan time: 500/5000 = 0.1s

Frames captured: 0.1s × 40fps = 4 frames (may need slower speed)
At 2000 µm/s: 0.25s, ~10 frames
Plus move to start + move to final: ~0.1s each

Total: ~0.5s (at max speed) to ~1.5s (at slower speed for more samples)
```

**Compared to step-and-capture (2DMatGMM style):**
```
44 images × (move + settle + capture)
44 × (0.1s + 0.05s + 0.03s) ≈ 8 seconds
```

## Potential Issues

1. **Motion blur**: If Z moves too fast relative to exposure (1ms exposure, 2000 µm/s = 2µm travel per frame - probably fine)

2. **Z polling**: Need to verify Z axis supports same fast polling as X axis

3. **Coarse peak**: Single pass might not find precise peak - may need coarse + fine passes

4. **Sharpness computation speed**: Need to compute sharpness fast enough to not fall behind capture rate (can do in background thread)

## Z Axis Capabilities (Verified)

| Property | Value |
|----------|-------|
| Position range | 0 - 6,799,210 native (~68mm) |
| Velocity range | 1 - 8,820,000 native |
| Polling rate | **64 Hz** |
| Async moves | SetControlValueAsync available |
| Halt | IID_HALT_CONTROL_VALUE available |

**Interfaces available:**
- IID_BASIC_CONTROL_VALUE (position read/write)
- IID_BASIC_CONTROL_VALUE_ASYNC (async moves)
- IID_BASIC_CONTROL_VALUE_VELOCITY (velocity control)
- IID_DIRECTED_CONTROL_VALUE_ASYNC
- IID_HALT_CONTROL_VALUE (emergency stop)

**Measured performance (from SDK velocity converter):**
- Max velocity: **5000 µm/s (5 mm/s)**
- 500µm focus scan at max speed: 0.1s
- 500µm focus scan at 2000 µm/s: 0.25s, ~10 frames

## Open Questions

1. What's a safe Z range for each objective? (need to measure working distances)

## Sharpness Metric

Using Tenengrad (Sobel gradient magnitude) - same as 2DMatGMM:

```python
def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.mean(cv2.magnitude(sobel_x, sobel_y))[0]
```

## Implementation Phases

### Phase 1: Z Axis Exploration
- Check Z axis interfaces (velocity, polling, limits)
- Measure Z move speed and position polling rate
- Verify async move support

### Phase 2: Basic Implementation
- Single-pass continuous scan
- Capture + poll + compute sharpness
- Find and move to peak

### Phase 3: Refinements (if needed)
- Coarse + fine two-pass approach
- Early termination when past focus
- Integration with scan scripts
