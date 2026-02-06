# Z Axis Feedforward Control Investigation

## Context

We're building continuous focus tracking for a Leica DM6M microscope. During horizontal (X) scans, the Z axis must follow a pre-computed surface profile to maintain focus. The surface has:
- A planar tilt (~1.47 µm/mm in X)
- Residual bumps/valleys (±6 µm deviations from plane)

## Current Implementation

See `test_z_curve_tracking.py` for the full implementation. Key control loop logic:

```python
# In control_z() thread, running at ~50 Hz:

# 1. Read current positions
current_x = ...  # from SDK
current_z = ...

# 2. Compute actual X velocity from position change
actual_x_vel = (current_x - last_x) / dt  # µm/s

# 3. Apply feedforward (CURRENT ISSUE)
feedforward_s = feedforward_ms / 1000.0
x_predicted = current_x + actual_x_vel * feedforward_s

# 4. Compute desired Z velocity
dzdx = dzdx_func(x_predicted)  # slope at predicted position
z_vel_desired = dzdx * actual_x_vel  # dZ/dt = dZ/dX * dX/dt

# 5. Command Z velocity
z_drive.start_towards_max(z_vel_desired)  # or start_towards_min
```

The `dzdx_func` returns the derivative of the surface profile (plane slope + spline derivative of residuals).

## The Problem

Empirically, we found that **negative feedforward** works better:

| Feedforward | Mean Error | Max Error | Result |
|-------------|------------|-----------|--------|
| +50 ms | -4.20 µm | 5.14 µm | FAIL |
| 0 ms | -2.12 µm | 4.40 µm | FAIL |
| **-50 ms** | **-0.82 µm** | **3.85 µm** | **PASS** |

This is counter-intuitive. Standard feedforward should predict where X *will be* when Z responds (positive offset), not where X *was* (negative offset).

## Z Axis Characteristics (from step response test)

- Rise time (10-90%): 32 ms
- Settling time (±2%): 90 ms
- Overshoot: 0% (critically damped)
- SDK read latency: ~16 ms per call

## System Timing

- Control loop: 50 Hz (20 ms intervals)
- X position read: ~16 ms
- Z velocity command: ~16 ms
- Z mechanical response: ~32 ms rise time

## Relevant Files

- `test_z_curve_tracking.py` - Main test script with control loop
- `test_z_step_response.py` - Z step response characterization
- `src/flakefinder/leica/units.py` - Axis control classes (`start_towards_max`, etc.)

## Data Files

Recent test results in the repo root:
- `z_curve_top_10mms_20260205_053209.json` - 10mm/s with -50ms feedforward (PASS)
- `z_curve_top_10mms_20260205_053140.json` - 10mm/s with 0ms feedforward
- `z_curve_top_5mms_20260205_052650.json` - 5mm/s with 0ms feedforward (PASS)
- `z_step_response_20260205_050649.json` - Step response data

## Your Task

1. **Research**: What is the standard approach to feedforward in velocity tracking control? Look at:
   - Feedforward in motion control systems
   - Predictive control for tracking applications
   - Phase lead compensation

2. **Analyze**: Why might negative feedforward work in our case? Consider:
   - Where delays occur in our control loop (read, compute, command, response)
   - Whether we're compensating for the right thing
   - Sign conventions in our velocity computation

3. **Propose**: Show me a plan for fixing the feedforward implementation properly:
   - Should feedforward be positive or negative, and why?
   - What is the correct formula?
   - Should we change the default from -50ms to something else?

## Questions to Answer

1. In standard feedforward control, do you predict where the setpoint *will be* or compensate for where the measurement *was*?

2. Is our dZ/dt = dZ/dX × dX/dt formulation correct for velocity feedforward?

3. Should feedforward time be positive (predict ahead) or negative (compensate for lag), and under what circumstances?

4. Are we accidentally double-compensating somewhere?

## Don't

- Don't run microscope commands without explicit approval
- Don't rewrite the whole control loop - propose targeted fixes
- Do research first, then propose a plan
