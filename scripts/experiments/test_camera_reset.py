"""Test: does the SDK reset camera settings on reconnect?

Session 1: read defaults, set unusual values, verify they took.
Session 2: reconnect, read — did unit.Init() reset them?

Usage:
    uv run python scripts/experiments/test_camera_reset.py
"""

from flakefinder.leica import Microscope


def dump(cam, label: str) -> None:
    r, g, b = cam.gain_rgb
    print(f"  {label}:")
    print(f"    binning={cam.binning}  gamma={cam.gamma:.2f}  exposure={cam.exposure_time * 1000:.1f}ms")
    print(f"    gain={cam.gain}  saturation={cam.saturation}  wb=R{r:.2f}/G{g:.2f}/B{b:.2f}")


# Session 1: read defaults, then set unusual values
with Microscope() as scope:
    cam = scope.camera
    dump(cam, "Session 1 — initial (SDK defaults)")

    cam.binning = 0  # 1x1 (we never use this)
    cam.gamma = 0.50
    cam.exposure_time = 0.1  # 100ms
    cam.gain = 2.0
    cam.saturation = 50
    cam.gain_rgb = (1.5, 1.5, 1.5)
    dump(cam, "Session 1 — after setting unusual values")

# Session 2: reconnect and read
with Microscope() as scope:
    cam = scope.camera
    dump(cam, "Session 2 — after reconnect")

    changed = []
    if cam.binning != 0:
        changed.append(f"binning: 0 -> {cam.binning}")
    if abs(cam.gamma - 0.50) > 0.01:
        changed.append(f"gamma: 0.50 -> {cam.gamma:.2f}")
    if abs(cam.exposure_time - 0.1) > 0.001:
        changed.append(f"exposure: 100ms -> {cam.exposure_time * 1000:.1f}ms")
    if abs(cam.gain - 2.0) > 0.01:
        changed.append(f"gain: 2.0 -> {cam.gain}")
    if cam.saturation != 50:
        changed.append(f"saturation: 50 -> {cam.saturation}")

    print()
    if changed:
        print(">> SDK RESET these settings on reconnect:")
        for c in changed:
            print(f"   {c}")
    else:
        print(">> Settings PERSISTED across reconnect (no SDK reset)")
