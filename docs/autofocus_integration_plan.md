# Continuous Autofocus Integration Plan

## Overview

Integrate the continuous Z-scan autofocus from `autofocus_demo.py` into the flakefinder library and wire it up to `scan_area_v1.py`'s `--auto-focus-pos` parameter.

## Current State

- **Demo**: `autofocus_demo.py` - working standalone CLI with full autofocus implementation
- **Library**: `src/flakefinder/leica/` - well-structured package with units.py, camera.py, etc.
- **Scan script**: `scan_area_v1.py:431-438` - has `--auto-focus-pos X,Y` parameter, currently just records in metadata with TODO comment

## Files to Create/Modify

### 1. Create: `src/flakefinder/leica/autofocus.py`

New module containing reusable autofocus logic extracted from `autofocus_demo.py`.

**Key components:**

```python
# Sharpness metric (from demo lines 20-36)
def sharpness(image: np.ndarray) -> float:
    """Tenengrad sharpness using Sobel gradients."""

# Position interpolation (from demo lines 39-66)
def interpolate_position(t: float, samples: list[tuple[float, float, float]]) -> float | None:
    """Interpolate Z position at time t from polled samples."""

@dataclass
class AutofocusFrame:
    """Single frame from autofocus scan."""
    z_um: float
    sharpness: float
    image: np.ndarray | None  # Only populated if store_frames=True

@dataclass
class AutofocusResult:
    """Result from autofocus operation."""
    best_z_um: float
    best_sharpness: float
    initial_z_um: float
    initial_sharpness: float
    final_sharpness: float
    z_range_um: float              # Actual range used (for logging)
    objective_position: int | None # Queried from microscope
    scan_duration_s: float
    frame_count: int
    z_sample_count: int
    sharpness_curve: list[dict]    # Always populated: [{z_um, sharpness}, ...]
    frames: list[AutofocusFrame] | None  # Only if store_frames=True
```

**Implementation approach:**
- Extract core scan loop from demo lines 259-315
- Return structured result instead of printing/saving files
- Sharpness curve always included for analysis/plotting
- Actual images optionally stored via `store_frames=True`
- Use demo's proven patterns (polling thread, async move, frame capture)

### 2. Modify: `src/flakefinder/leica/__init__.py`

Add exports:
```python
from .autofocus import (
    sharpness,
    interpolate_position,
    AutofocusFrame,
    AutofocusResult,
    continuous_autofocus,
    WORKING_DISTANCES_UM,
)
```

### 3. Modify: `scan_area_v1.py`

Replace the TODO comment (lines 431-438) with actual autofocus call:

```python
# Handle autofocus position if specified
auto_focus_pos = None
af_result = None
if args.auto_focus_pos:
    try:
        auto_focus_pos = parse_xy_position(args.auto_focus_pos)
        print(f"Autofocus position: ({auto_focus_pos[0]:.0f}, {auto_focus_pos[1]:.0f}) µm")

        # Move to autofocus position
        print(f"Moving to autofocus position...")
        stage.x.move_to(auto_focus_pos[0])
        stage.y.move_to(auto_focus_pos[1])

        # Run autofocus - queries microscope directly for objective safety
        from flakefinder.leica.autofocus import continuous_autofocus
        af_result = continuous_autofocus(
            conn=conn,
            camera=camera,
            acquisition=acquisition,
            context=context,
            # z_range_um auto-calculated from current objective
            fine_pass=True,
        )
        print(f"Autofocus: Z {af_result.initial_z_um:.1f} -> {af_result.best_z_um:.1f} µm")
        print(f"  Range: {af_result.z_range_um:.0f}µm, Sharpness: {af_result.initial_sharpness:.1f} -> {af_result.best_sharpness:.1f}")

    except ValueError as e:
        print(f"Autofocus error: {e}")
        return 1
```

Also add autofocus result to metadata:
```python
"autofocus": {
    "position_um": list(auto_focus_pos) if auto_focus_pos else None,
    "best_z_um": af_result.best_z_um if 'af_result' in dir() else None,
    "initial_z_um": af_result.initial_z_um if 'af_result' in dir() else None,
    "sharpness_improvement": ...,
} if auto_focus_pos else None,
```

### 4. Update: `autofocus_demo.py`

Refactor to use the new library module:
```python
from flakefinder.leica.autofocus import continuous_autofocus, sharpness, AutofocusResult

# In main():
af_result = continuous_autofocus(
    conn=conn,
    camera=camera,
    acquisition=acquisition,
    context=context,
    z_range_um=args.range,
    fine_pass=args.fine,
    store_frames=bool(args.debug_dir),  # Only store if saving debug output
)

# Save debug frames if requested
if args.debug_dir and af_result.frames:
    os.makedirs(args.debug_dir)
    for i, frame in enumerate(af_result.frames):
        fname = f"frame_{i:03d}_z_{frame.z_um:.1f}_s_{frame.sharpness:.1f}.jpg"
        PILImage.fromarray(frame.image).save(os.path.join(args.debug_dir, fname))
```

Keep CLI interface and file output, but delegate core logic to library.

## Safety Design

### The Problem

+Z = closer to sample. Scanning "up" (increasing Z) risks crashing objective into sample.
Working distances vary dramatically by objective:

| Position | Mag  | Working Distance |
|----------|------|------------------|
| 1        | 5x   | 12,700 µm        |
| 2        | 10x  | 11,000 µm        |
| 3        | 20x  | 1,900 µm         |
| 4        | 50x  | 380 µm           |
| 5        | 150x | 210 µm           |
| 6        | 2.5x | 15,000 µm        |

A 500µm scan range would exceed the working distance of 50x and 150x objectives!

### Safety Constraints

1. **Objective-dependent default range**:
   - Default scan range = `min(working_distance / 3, 500)` µm
   - 5x: 500µm, 20x: 500µm, 50x: 126µm, 150x: 70µm

2. **Z axis limit validation**:
   - `z_start <= z_axis.max_um`
   - `z_end >= z_axis.min_um`

3. **Working distance validation**:
   - Refuse to move Z above `current_z + safe_margin` where safe_margin is based on working distance
   - Or: require explicit `z_max_safe` parameter for high-mag objectives

4. **Scan direction**: Always scan downward (decreasing Z = away from sample)
   - Start at `current_z` or `current_z + small_margin` (never large margin for high-mag)
   - End at `current_z - range`

### Updated Function Signature

```python
# Working distances (from stage_util.py)
WORKING_DISTANCES_UM: dict[int, float] = {
    1: 12700, 2: 11000, 3: 1900, 4: 380, 5: 210, 6: 15000,
}

def continuous_autofocus(
    conn: LeicaConnection,           # Query microscope directly for safety
    camera: Camera,
    acquisition,
    context,
    z_range_um: float | None = None,         # None = auto from objective
    z_start_um: float | None = None,         # None = current_z (safe default)
    z_max_safe_um: float | None = None,      # Hard upper limit (overrides all)
    fine_pass: bool = False,
    fine_range_um: float = 50,
    store_frames: bool = False,              # Store images in result (for debug)
) -> AutofocusResult:
    """
    Safety behavior:
    - Queries nosepiece directly to get current objective position
    - Auto-calculates safe range from objective's working distance
    - If z_range_um exceeds safe range for objective: raises ValueError
    - If z_start_um would exceed z_max_safe_um: raises ValueError
    - If z_start_um would exceed z_axis.max_um: raises ValueError

    Frame storage:
    - sharpness_curve always populated (z_um, sharpness pairs)
    - If store_frames=True, frames list includes actual images
    - Demo uses store_frames=True for --debug-dir feature

    Internal:
    - Creates ZDrive from conn
    - Creates Nosepiece from conn to query objective
    """
```

**Key change**: Function takes `LeicaConnection` instead of individual axes. This lets it:
1. Query the nosepiece directly for current objective (don't trust caller)
2. Create ZDrive internally
3. Single source of truth for safety calculations

### Validation Logic

```python
def _get_safe_range(conn: LeicaConnection, z_range_um: float | None) -> tuple[float, int | None]:
    """Query microscope and calculate safe Z range.

    Returns:
        (safe_range_um, objective_position)
    """
    # Query current objective from microscope
    objective_position = None
    working_distance = None
    try:
        nosepiece = Nosepiece.from_connection(conn)
        objective_position = nosepiece.position
        working_distance = WORKING_DISTANCES_UM.get(objective_position)
    except LookupError:
        pass  # No nosepiece available

    if z_range_um is not None:
        # Explicit range provided - validate against objective
        if working_distance and z_range_um > working_distance / 2:
            raise ValueError(
                f"Z range {z_range_um}µm exceeds safe limit for "
                f"objective position {objective_position} (working distance: {working_distance}µm)"
            )
        return z_range_um, objective_position

    # Auto-calculate from objective
    if working_distance:
        return min(working_distance / 3, 500), objective_position

    # Unknown objective - use conservative default
    return 200, objective_position  # Safe for all objectives

def _validate_z_limits(
    z_axis: Axis,
    z_start: float,
    z_end: float,
    z_max_safe: float | None,
) -> None:
    """Validate Z positions against axis and safety limits."""
    if z_start > z_axis.max_um:
        raise ValueError(f"Z start {z_start}µm exceeds axis max {z_axis.max_um}µm")
    if z_end < z_axis.min_um:
        raise ValueError(f"Z end {z_end}µm below axis min {z_axis.min_um}µm")
    if z_max_safe is not None and z_start > z_max_safe:
        raise ValueError(f"Z start {z_start}µm exceeds safe max {z_max_safe}µm")
```

## Key Design Decisions

1. **Minimal API**: `continuous_autofocus()` is a single function, not a class
   - Most use cases just need "focus here"
   - Result dataclass captures all info needed for logging/debugging

2. **Caller provides SDK objects**: acquisition/context passed in
   - Avoids library needing to know about SDK registration (Extensions.ExUCAPI.Register())
   - Scan script already has these set up

3. **Safe defaults**:
   - Default range auto-calculated from objective (conservative 1/3 of working distance)
   - Default z_start = current position (no upward movement)
   - Fine pass optional, uses smaller 50µm range

4. **No file I/O in library**:
   - Library function returns data, caller decides what to save
   - Demo script handles --debug-dir, --output

5. **Optional frame storage** (`store_frames=True`):
   - Sharpness curve (z, sharpness pairs) always returned
   - Actual images only stored when requested (for demo's --debug-dir)
   - Keeps memory usage low for scan_area_v1.py integration

6. **Fail-safe validation**:
   - Explicit errors for dangerous parameters
   - Queries microscope directly for objective (don't trust caller)
   - Unknown objective = conservative 200µm default

## Dependencies

- `cv2` (OpenCV) - for Sobel sharpness calculation
- `numpy` - array handling
- Already used in autofocus_demo.py, likely already installed

## Implementation Phases

### Phase 1: Create Library Module
Create `src/flakefinder/leica/autofocus.py` with:
- `sharpness()` function
- `interpolate_position()` function
- `AutofocusFrame` and `AutofocusResult` dataclasses
- `WORKING_DISTANCES_UM` dict
- `_get_safe_range()` and `_validate_z_limits()` helpers
- `continuous_autofocus()` main function

Update `src/flakefinder/leica/__init__.py` with exports.

**PAUSE for review before Phase 2.**

### Phase 2: Update autofocus_demo.py
Refactor to use the new library module:
- Replace inline sharpness/interpolation with imports
- Replace scan loop with `continuous_autofocus()` call
- Use `store_frames=True` for --debug-dir feature
- Keep CLI interface, before/after photo capture, summary output

**PAUSE for review before Phase 3.**

### Phase 3: Integrate with scan_area_v1.py
Wire up `--auto-focus-pos` parameter:
- Move to XY position before autofocus
- Call `continuous_autofocus()`
- Add autofocus results to scan metadata

**PAUSE for final review.**

## Testing

1. **Unit test the library module**:
   ```bash
   # Test sharpness function with sample images
   # Test interpolate_position with mock data
   ```

2. **Test autofocus_demo.py still works**:
   ```bash
   python autofocus_demo.py -o test_af --x 50000 --y 35000 --fine
   ```

3. **Test scan_area_v1.py integration**:
   ```bash
   # Small area scan with autofocus
   python scan_area_v1.py -o test_scan --area-rect 45000,55000,30000,40000 \
       --auto-focus-pos 50000,35000 --objective-mag 5x
   ```

4. **Verify metadata includes autofocus info**:
   ```bash
   cat test_scan/scan_meta.json | jq .autofocus
   ```

## Verification Checklist

- [ ] `continuous_autofocus()` returns correct best Z
- [ ] Z axis ends at best focus position after call
- [ ] Scan script moves to XY position before autofocus
- [ ] Scan script continues scanning after autofocus
- [ ] Metadata includes autofocus parameters and results
- [ ] Demo script works with refactored library code
- [ ] No regression in scan timing or image quality
