"""Time X/Y stage moves at max velocity for distances matching chip scan preposition.

Distances tested: 2mm, 6mm, 14mm (matching row transitions in 5-row test scans).
Tests both single-axis and parallel X+Y moves.
"""

import time

from flakefinder.leica import Microscope, wait_all


def main():
    with Microscope() as scope:
        stage = scope.stage
        x = stage.x
        y = stage.y

        x_pos = x.position_um
        y_pos = y.position_um

        print(f"Position: X={x_pos:.0f}, Y={y_pos:.0f} µm")
        print(f"X velocity: {x.velocity_um_s:.0f} µm/s (current), max={x.max_velocity_um_s:.0f}")
        print(f"Y velocity: {y.velocity_um_s:.0f} µm/s (current), max={y.max_velocity_um_s:.0f}")
        print()

        # Set max velocity
        x.set_velocity_um_s(x.max_velocity_um_s)
        y.set_velocity_um_s(y.max_velocity_um_s)
        print(f"Set X/Y to max velocity: {x.velocity_um_s:.0f} µm/s")
        print()

        x_deltas = [+2000, -2000, +6000, -6000, +14000, -14000]

        # --- X only, async ---
        print("=== X async (max velocity) ===")
        for d in x_deltas:
            target = x_pos + d
            t0 = time.perf_counter()
            h = x.move_to_async(target)
            t_dispatch = time.perf_counter() - t0
            h.wait()
            dt = time.perf_counter() - t0
            h.dispose()
            actual = x.position_um
            err = actual - target
            print(
                f"  {d / 1000:+5.0f}mm: dispatch={t_dispatch * 1000:.0f}ms, total={dt * 1000:.0f}ms  (err={err:+.1f}µm)"
            )
        x.move_to(x_pos)
        print()

        # --- X only, blocking ---
        print("=== X blocking (max velocity) ===")
        for d in x_deltas:
            target = x_pos + d
            t0 = time.perf_counter()
            x.move_to(target)
            dt = time.perf_counter() - t0
            actual = x.position_um
            err = actual - target
            print(f"  {d / 1000:+5.0f}mm: {dt * 1000:.0f}ms  (err={err:+.1f}µm)")
        x.move_to(x_pos)
        print()

        # --- Parallel X+Y (simulating row transition) ---
        # Y step is ~386µm (typical row step), X varies
        y_delta = 386
        print(f"=== Parallel X+Y async (Y step={y_delta}µm) ===")
        for dx in [2000, 6000, 14000]:
            for x_sign in [+1, -1]:
                d = dx * x_sign
                x_target = x_pos + d
                y_target = y_pos + y_delta

                t0 = time.perf_counter()
                hx = x.move_to_async(x_target)
                hy = y.move_to_async(y_target)
                wait_all([hx, hy])
                dt = time.perf_counter() - t0

                x_actual = x.position_um
                y_actual = y.position_um
                x_err = x_actual - x_target
                y_err = y_actual - y_target
                print(
                    f"  X={d / 1000:+5.0f}mm Y=+{y_delta}µm: {dt * 1000:.0f}ms  "
                    f"(xerr={x_err:+.1f}, yerr={y_err:+.1f}µm)"
                )

                # Return
                hx = x.move_to_async(x_pos)
                hy = y.move_to_async(y_pos)
                wait_all([hx, hy])

        print()
        print(f"Done. Returned to X={x.position_um:.0f}, Y={y.position_um:.0f}")


if __name__ == "__main__":
    main()
