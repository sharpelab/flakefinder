"""Shared CLI utilities for microscope scripts."""

from flakefinder.data_utils import require_microscope_description
from flakefinder.leica import Microscope, wait_all


def report_status(scope: Microscope, *, verbose: bool = False) -> None:
    """Print current microscope status (stage, optics, camera).

    Args:
        scope: Connected Microscope instance.
        verbose: If True, also print velocity limits.
    """
    # Stage XY
    x, y = scope.stage.position_um
    print(f"Stage X: {x:.1f} µm ({scope.stage.x.min_um:.0f} - {scope.stage.x.max_um:.0f})")
    print(f"Stage Y: {y:.1f} µm ({scope.stage.y.min_um:.0f} - {scope.stage.y.max_um:.0f})")

    # Z axis
    print(f"Stage Z: {scope.z.position_um:.1f} µm ({scope.z.min_um:.0f} - {scope.z.max_um:.0f})")

    # Velocity info (verbose mode)
    if verbose:
        print()
        print("Velocity (from SDK converter):")
        print(
            f"  X: {scope.stage.x.velocity_um_s / 1000:.1f} mm/s "
            f"(max: {scope.stage.x.max_velocity_um_s / 1000:.1f} mm/s)"
        )
        print(
            f"  Y: {scope.stage.y.velocity_um_s / 1000:.1f} mm/s "
            f"(max: {scope.stage.y.max_velocity_um_s / 1000:.1f} mm/s)"
        )
        print(f"  Z: {scope.z.velocity_um_s / 1000:.1f} mm/s (max: {scope.z.max_velocity_um_s / 1000:.1f} mm/s)")

    # Nosepiece/objective
    mag = scope.nosepiece.magnification
    if mag:
        print(f"Objective: position {scope.nosepiece.position} ({mag}x)")
    else:
        print(f"Objective: position {scope.nosepiece.position}")

    # Lamp
    print(f"Lamp: {scope.lamp.intensity_pct:.0f}% ({scope.lamp.intensity}/{scope.lamp.max_intensity})")

    # Shutter
    print(f"Shutter: {'open' if scope.shutter.is_open else 'closed'}")

    # Aperture
    ap = scope.aperture
    label = " (fully open)" if ap.value == ap.max_value else ""
    print(f"Aperture: {ap.value}/{ap.max_value}{label}")

    # Camera settings
    camera = scope.camera
    desc = require_microscope_description()
    binning_str = desc.camera.binning_levels[camera.binning].name
    w, h = camera.frame_size_px
    r, g, b = camera.gain_rgb
    exp_ms = camera.exposure_time * 1000
    print(
        f"Camera: {w}x{h} @ {binning_str} binning, {exp_ms:.1f}ms exposure,"
        f" saturation={camera.saturation}, gamma={camera.gamma:.2f}"
    )
    print(f"White balance: R={r:.2f} G={g:.2f} B={b:.2f}")


def park_microscope(scope: Microscope) -> None:
    """Put microscope in a safe idle state.

    Operations in order:
    1. Move Z to standard 5x focus position (24690 µm)
    2. Switch to 5x objective
    3. Move XY to origin (0, 0)
    4. Turn lamp off + close shutter
    """
    print("Parking microscope...")

    # 1. Z to safe position first
    z_target = 24690.0
    scope.z.move_to(z_target)
    print(f"  Z -> {scope.z.position_um:.0f} um [ok]")

    # 2. Switch to 5x objective
    target_pos = None
    for pos, mag in scope.nosepiece.magnifications.items():
        if mag == 5.0:
            target_pos = pos
            break
    if target_pos is not None:
        scope.switch_objective_pos(target_pos)
    print(f"  Objective -> {scope.objective_mag}x [ok]")

    # 3. XY to origin
    hx, hy = scope.stage.move_to_async(0.0, 0.0)
    wait_all([hx, hy])
    print(f"  X -> {scope.stage.x.position_um:.0f} um [ok]")
    print(f"  Y -> {scope.stage.y.position_um:.0f} um [ok]")

    # 4. Lights off
    scope.light_off()
    print("  Lamp off [ok]")
    print("  Shutter closed [ok]")

    print("Parked.")
