"""Time X/Y stage moves at max velocity for distances matching chip scan preposition.

Distances tested: 2mm, 6mm, 14mm (matching row transitions in 5-row test scans).
Tests both single-axis and parallel X+Y moves. Writes results to JSON.
"""

import json
import time

from flakefinder.leica import Microscope, wait_all

OUTPUT = "scans/xy_move_timing.json"


def main():
    with Microscope() as scope:
        stage = scope.stage
        x = stage.x
        y = stage.y

        # Move to stage center for safe ± moves
        x_center = (x.min_um + x.max_um) / 2
        y_center = (y.min_um + y.max_um) / 2
        print(f"Moving to stage center: X={x_center:.0f}, Y={y_center:.0f} µm...")
        x.move_to(x_center)
        y.move_to(y_center)

        x_pos = x.position_um
        y_pos = y.position_um
        x_max_vel = x.max_velocity_um_s
        y_max_vel = y.max_velocity_um_s

        print(f"Position: X={x_pos:.0f}, Y={y_pos:.0f} µm")
        print(f"Max velocity: X={x_max_vel:.0f}, Y={y_max_vel:.0f} µm/s")
        print()

        x.set_velocity_um_s(x_max_vel)
        y.set_velocity_um_s(y_max_vel)

        results = {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "x_max_vel_um_s": x_max_vel,
            "y_max_vel_um_s": y_max_vel,
            "x_async": [],
            "x_blocking": [],
            "parallel_xy": [],
        }

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
            print(f"  {d / 1000:+5.0f}mm: {dt * 1000:.0f}ms  (err={err:+.1f}µm)")
            results["x_async"].append(
                {
                    "delta_um": d,
                    "dispatch_ms": round(t_dispatch * 1000, 1),
                    "total_ms": round(dt * 1000, 1),
                    "err_um": round(err, 2),
                }
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
            results["x_blocking"].append(
                {
                    "delta_um": d,
                    "total_ms": round(dt * 1000, 1),
                    "err_um": round(err, 2),
                }
            )
            x.move_to(x_pos)
        print()

        # --- Parallel X+Y (simulating row transition) ---
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

                x_err = x.position_um - x_target
                y_err = y.position_um - y_target
                print(f"  X={d / 1000:+5.0f}mm Y=+{y_delta}µm: {dt * 1000:.0f}ms")
                results["parallel_xy"].append(
                    {
                        "x_delta_um": d,
                        "y_delta_um": y_delta,
                        "total_ms": round(dt * 1000, 1),
                        "x_err_um": round(x_err, 2),
                        "y_err_um": round(y_err, 2),
                    }
                )

                hx = x.move_to_async(x_pos)
                hy = y.move_to_async(y_pos)
                wait_all([hx, hy])

        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
