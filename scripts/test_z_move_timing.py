"""Time Z axis move_to_async vs move_to_corrected for small moves.

Moves Z ±5µm from current position and times each method.
"""

import time

from flakefinder.leica import Microscope


def main():
    with Microscope() as scope:
        z = scope.z
        z_pos = z.position_um

        print(f"Z position: {z_pos:.1f} µm")
        print(f"Z velocity: {z.velocity_um_s:.1f} µm/s (current)")
        print(f"Z max velocity: {z.max_velocity_um_s:.1f} µm/s")
        print()

        deltas = [-5, +5, -2, +2, -1, +1]

        # --- move_to_corrected (blocking) ---
        print("=== move_to_corrected (blocking, hysteresis) ===")
        for d in deltas:
            target = z_pos + d
            t0 = time.perf_counter()
            z.move_to_corrected(target)
            dt = time.perf_counter() - t0
            actual = z.position_um
            print(f"  {d:+.0f}µm: {dt * 1000:.0f}ms  (landed at {actual:.1f}, err={actual - target:+.2f})")
        z.move_to_corrected(z_pos)
        print()

        # --- move_to_async (default velocity) ---
        print("=== move_to_async (default velocity) ===")
        for d in deltas:
            target = z_pos + d
            t0 = time.perf_counter()
            h = z.move_to_async(target)
            t_dispatch = time.perf_counter() - t0
            h.wait()
            dt = time.perf_counter() - t0
            h.dispose()
            actual = z.position_um
            err = actual - target
            print(f"  {d:+.0f}µm: dispatch={t_dispatch * 1000:.0f}ms, total={dt * 1000:.0f}ms  (err={err:+.2f})")
        z.move_to_corrected(z_pos)
        print()

        # --- move_to_async (max velocity) ---
        print("=== move_to_async (max velocity) ===")
        z.set_velocity_um_s(z.max_velocity_um_s)
        print(f"  Z velocity set to: {z.velocity_um_s:.1f} µm/s")
        for d in deltas:
            target = z_pos + d
            t0 = time.perf_counter()
            h = z.move_to_async(target)
            t_dispatch = time.perf_counter() - t0
            h.wait()
            dt = time.perf_counter() - t0
            h.dispose()
            actual = z.position_um
            err = actual - target
            print(f"  {d:+.0f}µm: dispatch={t_dispatch * 1000:.0f}ms, total={dt * 1000:.0f}ms  (err={err:+.2f})")
        z.move_to_corrected(z_pos)
        print()

        # --- move_to (blocking, regular BCV) ---
        print("=== move_to (blocking, regular BCV) ===")
        for d in deltas:
            target = z_pos + d
            t0 = time.perf_counter()
            z.move_to(target)
            dt = time.perf_counter() - t0
            actual = z.position_um
            print(f"  {d:+.0f}µm: {dt * 1000:.0f}ms  (landed at {actual:.1f}, err={actual - target:+.2f})")
        z.move_to_corrected(z_pos)

        print("\nDone. Z returned to original position.")


if __name__ == "__main__":
    main()
