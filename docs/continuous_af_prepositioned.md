# continuous_autofocus: pre_positioned optimization

All three `_run_z_scan()` calls in `continuous_autofocus()` redundantly move to z_start internally despite the caller already positioning there. Each costs ~400ms (blocking move + 100ms settle).

## Fine and super-fine passes

Already do `move_to_corrected(z_start)` immediately before calling `_run_z_scan(z_start=...)`. Just add `pre_positioned=True`.

## Coarse pass

Positions at `initial_z` (center) for initial sharpness capture, then relies on `_run_z_scan` to move to `z_start` (center + range/2). Fix: after the initial capture, add `z_axis.move_to_corrected(z_start)` and pass `pre_positioned=True`.

## Expected savings

~400ms per pass. For a 3-pass AF (coarse + fine + super-fine), ~1.2s total.
