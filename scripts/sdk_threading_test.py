"""Characterize SDK threading constraints.

Experiments:
  1. Poll throughput vs thread count — how does GetControlValue() rate
     scale with 1, 2, 4, 8 concurrent polling threads?
  2. Move completion under load — does move_to_async + wait() complete
     when N threads are polling? At what N does it break?
  3. Cross-axis interference — poll axis A while moving axis B.
  4. Sleep between polls — does yielding between calls fix starvation?

Usage:
    uv run python scripts/sdk_threading_test.py
    uv run python scripts/sdk_threading_test.py --threads 1,2,4
    uv run python scripts/sdk_threading_test.py --skip-move     # polling only
    uv run python scripts/sdk_threading_test.py --experiments 1,2  # select experiments
"""

import argparse
import contextlib
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from flakefinder.leica import Microscope

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PollResult:
    thread_count: int
    duration_s: float
    total_calls: int
    per_thread_calls: list[int]
    aggregate_hz: float
    per_thread_hz: float
    mean_call_us: float
    poll_axis: str
    sleep_ms: float = 0


@dataclass
class MoveTestResult:
    thread_count: int
    completed: bool
    move_time_s: float | None
    timeout_s: float
    poll_hz_during: float
    poll_axis: str
    move_axis: str
    move_distance_um: float
    sleep_ms: float = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_axis(scope: Microscope, name: str):
    """Get axis object by name ('x', 'y', 'z')."""
    axes = {"x": scope.stage.x, "y": scope.stage.y, "z": scope.z}
    if name not in axes:
        raise ValueError(f"Unknown axis: {name}")
    return axes[name]


def _poll_loop(
    bcv,
    stop: threading.Event,
    count_out: list[int],
    idx: int,
    latencies_out: list[float] | None = None,
    sleep_s: float = 0,
):
    """Tight polling loop for one thread."""
    local_count = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        bcv.GetControlValue()
        t1 = time.perf_counter()
        local_count += 1
        # Sample latencies from thread 0 only (first 200 calls)
        if latencies_out is not None and idx == 0 and local_count <= 200:
            latencies_out.append(t1 - t0)
        if sleep_s > 0:
            time.sleep(sleep_s)
    count_out[idx] = local_count


# ---------------------------------------------------------------------------
# Experiment 1: Pure polling throughput
# ---------------------------------------------------------------------------


def experiment_poll_throughput(
    scope: Microscope,
    thread_counts: list[int],
    duration_s: float = 3.0,
    axis_name: str = "x",
) -> list[PollResult]:
    """Measure polling throughput vs thread count."""
    axis = _get_axis(scope, axis_name)
    bcv = axis.bcv
    results = []

    for n in thread_counts:
        counts = [0] * n
        latencies: list[float] = []
        stop = threading.Event()

        threads = []
        for i in range(n):
            t = threading.Thread(
                target=_poll_loop,
                args=(bcv, stop, counts, i, latencies if i == 0 else None),
                daemon=True,
            )
            threads.append(t)

        start = time.perf_counter()
        for t in threads:
            t.start()

        time.sleep(duration_s)
        stop.set()

        for t in threads:
            t.join(timeout=2.0)

        elapsed = time.perf_counter() - start
        total = sum(counts)
        mean_us = sum(latencies) / len(latencies) * 1e6 if latencies else 0

        results.append(
            PollResult(
                thread_count=n,
                duration_s=elapsed,
                total_calls=total,
                per_thread_calls=list(counts),
                aggregate_hz=total / elapsed,
                per_thread_hz=total / elapsed / n,
                mean_call_us=mean_us,
                poll_axis=axis_name,
            )
        )

        print(
            f"  {n} thread(s): {total / elapsed:.0f} Hz aggregate, "
            f"{total / elapsed / n:.0f} Hz/thread, "
            f"call={mean_us:.0f} µs"
        )

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Move completion under polling load
# ---------------------------------------------------------------------------


def experiment_move_under_load(
    scope: Microscope,
    thread_counts: list[int],
    poll_axis_name: str = "x",
    move_axis_name: str = "x",
    move_distance_um: float = 2000,
    timeout_s: float = 10.0,
) -> list[MoveTestResult]:
    """Test whether async moves complete with N polling threads running."""
    poll_axis = _get_axis(scope, poll_axis_name)
    move_axis = _get_axis(scope, move_axis_name)
    poll_bcv = poll_axis.bcv
    results = []

    for n in thread_counts:
        # Start polling threads
        counts = [0] * max(n, 1)
        stop = threading.Event()
        poll_threads = []

        if n > 0:
            for i in range(n):
                t = threading.Thread(
                    target=_poll_loop,
                    args=(poll_bcv, stop, counts, i),
                    daemon=True,
                )
                t.start()
                poll_threads.append(t)

        # Small settle after starting pollers
        time.sleep(0.1)

        # Do async move
        poll_start = time.perf_counter()
        handle = move_axis.move_rel_async(move_distance_um)
        completed = handle.wait(timeout=timeout_s)
        move_time = time.perf_counter() - poll_start if completed else None

        # Stop polling
        stop.set()
        for t in poll_threads:
            t.join(timeout=2.0)

        poll_elapsed = time.perf_counter() - poll_start
        total_polls = sum(counts)
        poll_hz = total_polls / poll_elapsed if poll_elapsed > 0 else 0

        handle.dispose()

        # Move back to starting position
        if completed:
            rev = move_axis.move_rel_async(-move_distance_um)
            rev.wait(timeout=10.0)
            rev.dispose()
        else:
            # If move didn't complete, halt and try to move back
            move_axis.halt()
            time.sleep(0.5)
            with contextlib.suppress(Exception):
                move_axis.move_rel(-move_distance_um)

        result = MoveTestResult(
            thread_count=n,
            completed=completed,
            move_time_s=move_time,
            timeout_s=timeout_s,
            poll_hz_during=poll_hz,
            poll_axis=poll_axis_name,
            move_axis=move_axis_name,
            move_distance_um=move_distance_um,
        )
        results.append(result)

        status = f"{move_time:.3f}s" if completed else f"HUNG (timeout {timeout_s}s)"
        print(f"  {n} poll thread(s): move {status}, poll={poll_hz:.0f} Hz")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Cross-axis interference
# ---------------------------------------------------------------------------


def experiment_cross_axis(
    scope: Microscope,
    thread_counts: list[int],
    move_distance_um: float = 2000,
    timeout_s: float = 10.0,
) -> list[MoveTestResult]:
    """Poll axis A while moving axis B. Test all axis combos."""
    combos = [
        ("x", "y"),  # poll X, move Y
        ("y", "x"),  # poll Y, move X
        ("z", "x"),  # poll Z, move X
        ("x", "z"),  # poll X, move Z (Z is slower, use smaller distance)
    ]
    results = []

    for poll_ax, move_ax in combos:
        dist = 500 if move_ax == "z" else move_distance_um
        print(f"  Poll {poll_ax.upper()}, move {move_ax.upper()} ({dist} µm):")

        for n in thread_counts:
            poll_axis = _get_axis(scope, poll_ax)
            move_axis = _get_axis(scope, move_ax)
            poll_bcv = poll_axis.bcv

            counts = [0] * max(n, 1)
            stop = threading.Event()
            poll_threads = []

            if n > 0:
                for i in range(n):
                    t = threading.Thread(
                        target=_poll_loop,
                        args=(poll_bcv, stop, counts, i),
                        daemon=True,
                    )
                    t.start()
                    poll_threads.append(t)

            time.sleep(0.1)

            poll_start = time.perf_counter()
            handle = move_axis.move_rel_async(dist)
            completed = handle.wait(timeout=timeout_s)
            move_time = time.perf_counter() - poll_start if completed else None

            stop.set()
            for t in poll_threads:
                t.join(timeout=2.0)

            poll_elapsed = time.perf_counter() - poll_start
            total_polls = sum(counts)
            poll_hz = total_polls / poll_elapsed if poll_elapsed > 0 else 0

            handle.dispose()

            if completed:
                rev = move_axis.move_rel_async(-dist)
                rev.wait(timeout=10.0)
                rev.dispose()
            else:
                move_axis.halt()
                time.sleep(0.5)
                with contextlib.suppress(Exception):
                    move_axis.move_rel(-dist)

            result = MoveTestResult(
                thread_count=n,
                completed=completed,
                move_time_s=move_time,
                timeout_s=timeout_s,
                poll_hz_during=poll_hz,
                poll_axis=poll_ax,
                move_axis=move_ax,
                move_distance_um=dist,
            )
            results.append(result)

            status = f"{move_time:.3f}s" if completed else "HUNG"
            print(f"    {n} thread(s): {status}, poll={poll_hz:.0f} Hz")

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Sleep between polls (starvation mitigation)
# ---------------------------------------------------------------------------


def experiment_poll_with_sleep(
    scope: Microscope,
    thread_count: int = 4,
    sleep_values_ms: list[float] | None = None,
    move_distance_um: float = 2000,
    timeout_s: float = 10.0,
    axis_name: str = "x",
) -> list[MoveTestResult]:
    """Test if sleeping between polls prevents move starvation."""
    if sleep_values_ms is None:
        sleep_values_ms = [0, 0.5, 1, 2, 5, 10]

    axis = _get_axis(scope, axis_name)
    bcv = axis.bcv
    results = []

    for sleep_ms in sleep_values_ms:
        sleep_s = sleep_ms / 1000

        counts = [0] * thread_count
        stop = threading.Event()
        poll_threads = []

        for i in range(thread_count):
            t = threading.Thread(
                target=_poll_loop,
                args=(bcv, stop, counts, i, None, sleep_s),
                daemon=True,
            )
            t.start()
            poll_threads.append(t)

        time.sleep(0.1)

        poll_start = time.perf_counter()
        handle = axis.move_rel_async(move_distance_um)
        completed = handle.wait(timeout=timeout_s)
        move_time = time.perf_counter() - poll_start if completed else None

        stop.set()
        for t in poll_threads:
            t.join(timeout=2.0)

        poll_elapsed = time.perf_counter() - poll_start
        total_polls = sum(counts)
        poll_hz = total_polls / poll_elapsed if poll_elapsed > 0 else 0

        handle.dispose()

        if completed:
            rev = axis.move_rel_async(-move_distance_um)
            rev.wait(timeout=10.0)
            rev.dispose()
        else:
            axis.halt()
            time.sleep(0.5)
            with contextlib.suppress(Exception):
                axis.move_rel(-move_distance_um)

        result = MoveTestResult(
            thread_count=thread_count,
            completed=completed,
            move_time_s=move_time,
            timeout_s=timeout_s,
            poll_hz_during=poll_hz,
            poll_axis=axis_name,
            move_axis=axis_name,
            move_distance_um=move_distance_um,
            sleep_ms=sleep_ms,
        )
        results.append(result)

        status = f"{move_time:.3f}s" if completed else "HUNG"
        print(f"  sleep={sleep_ms:.1f}ms: {status}, poll={poll_hz:.0f} Hz")

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    poll_results: list[PollResult],
    move_results: list[MoveTestResult],
    cross_results: list[MoveTestResult],
    sleep_results: list[MoveTestResult],
):
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if poll_results:
        print()
        print("Experiment 1: Polling throughput vs thread count")
        print(f"  {'Threads':>8} {'Aggregate Hz':>14} {'Per-thread Hz':>14} {'Call µs':>10}")
        print(f"  {'--------':>8} {'--------------':>14} {'--------------':>14} {'----------':>10}")
        for r in poll_results:
            print(f"  {r.thread_count:>8} {r.aggregate_hz:>14.0f} {r.per_thread_hz:>14.0f} {r.mean_call_us:>10.0f}")

    if move_results:
        print()
        pa, ma = move_results[0].poll_axis.upper(), move_results[0].move_axis.upper()
        print(f"Experiment 2: Move completion under load (poll {pa}, move {ma})")
        print(f"  {'Threads':>8} {'Completed':>10} {'Move time':>12} {'Poll Hz':>10}")
        print(f"  {'--------':>8} {'----------':>10} {'------------':>12} {'----------':>10}")
        for r in move_results:
            t = f"{r.move_time_s:.3f}s" if r.move_time_s is not None else "---"
            print(f"  {r.thread_count:>8} {'YES' if r.completed else 'NO':>10} {t:>12} {r.poll_hz_during:>10.0f}")

    if cross_results:
        print()
        print("Experiment 3: Cross-axis interference")
        print(f"  {'Poll':>5} {'Move':>5} {'Threads':>8} {'Completed':>10} {'Move time':>12} {'Poll Hz':>10}")
        print(f"  {'-----':>5} {'-----':>5} {'--------':>8} {'----------':>10} {'------------':>12} {'----------':>10}")
        for r in cross_results:
            t = f"{r.move_time_s:.3f}s" if r.move_time_s is not None else "---"
            print(
                f"  {r.poll_axis.upper():>5} {r.move_axis.upper():>5} "
                f"{r.thread_count:>8} {'YES' if r.completed else 'NO':>10} "
                f"{t:>12} {r.poll_hz_during:>10.0f}"
            )

    if sleep_results:
        print()
        n = sleep_results[0].thread_count
        print(f"Experiment 4: Sleep between polls ({n} threads, {sleep_results[0].poll_axis.upper()} axis)")
        print(f"  {'Sleep ms':>10} {'Completed':>10} {'Move time':>12} {'Poll Hz':>10}")
        print(f"  {'----------':>10} {'----------':>10} {'------------':>12} {'----------':>10}")
        for r in sleep_results:
            t = f"{r.move_time_s:.3f}s" if r.move_time_s is not None else "---"
            print(f"  {r.sleep_ms:>10.1f} {'YES' if r.completed else 'NO':>10} {t:>12} {r.poll_hz_during:>10.0f}")

    # Diagnosis
    print()
    print("-" * 70)

    if move_results:
        failed = [r for r in move_results if not r.completed]
        passed = [r for r in move_results if r.completed]
        if not failed:
            print("Move completion: ALL PASSED (no starvation observed)")
        elif not passed:
            print("Move completion: ALL FAILED (even 0 poll threads?!)")
        else:
            threshold = min(r.thread_count for r in failed)
            max_ok = max(r.thread_count for r in passed)
            print(f"Move starvation threshold: {threshold} threads (last OK: {max_ok})")

    if sleep_results:
        first_ok = next((r for r in sleep_results if r.completed), None)
        if first_ok and first_ok.sleep_ms > 0:
            print(f"Minimum sleep to fix starvation: {first_ok.sleep_ms:.1f} ms")
        elif first_ok and first_ok.sleep_ms == 0:
            print("No starvation observed even without sleep")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Characterize SDK threading constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="0,1,2,4,8",
        help="Comma-separated thread counts to test (default: 0,1,2,4,8)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,3,4",
        help="Comma-separated experiment numbers to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--poll-duration",
        type=float,
        default=3.0,
        help="Duration for pure polling test in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--move-timeout",
        type=float,
        default=10.0,
        help="Timeout for move completion in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--move-distance",
        type=float,
        default=2000,
        help="Move distance in µm for move tests (default: 2000)",
    )
    parser.add_argument(
        "--sleep-threads",
        type=int,
        default=4,
        help="Thread count for sleep experiment (default: 4)",
    )
    parser.add_argument(
        "--sleep-values",
        type=str,
        default="0,0.5,1,2,5,10",
        help="Comma-separated sleep values in ms for experiment 4 (default: 0,0.5,1,2,5,10)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/sdk_threading_YYYYMMDD_HHMM.json)",
    )
    args = parser.parse_args()

    thread_counts = [int(x) for x in args.threads.split(",")]
    experiments = [int(x) for x in args.experiments.split(",")]
    sleep_values = [float(x) for x in args.sleep_values.split(",")]

    # Filter: experiments 2-4 need thread counts > 0 for the poll side,
    # but experiment 2 also tests 0 (baseline with no polling)
    poll_thread_counts = [n for n in thread_counts if n >= 1]

    print("SDK Threading Characterization")
    print("=" * 70)
    print(f"Thread counts: {thread_counts}")
    print(f"Experiments: {experiments}")
    print(f"Poll duration: {args.poll_duration}s")
    print(f"Move timeout: {args.move_timeout}s")
    print(f"Move distance: {args.move_distance} µm")
    print()

    with Microscope() as scope:
        poll_results: list[PollResult] = []
        move_results: list[MoveTestResult] = []
        cross_results: list[MoveTestResult] = []
        sleep_results: list[MoveTestResult] = []

        if 1 in experiments:
            print("--- Experiment 1: Polling throughput ---")
            poll_results = experiment_poll_throughput(
                scope,
                poll_thread_counts,
                duration_s=args.poll_duration,
            )
            print()

        if 2 in experiments:
            print("--- Experiment 2: Move completion under load ---")
            move_results = experiment_move_under_load(
                scope,
                thread_counts,  # include 0 as baseline
                move_distance_um=args.move_distance,
                timeout_s=args.move_timeout,
            )
            print()

        if 3 in experiments:
            print("--- Experiment 3: Cross-axis interference ---")
            # Use smaller thread set for cross-axis (many combos)
            cross_threads = [n for n in thread_counts if n in (0, 2, 4)]
            if not cross_threads:
                cross_threads = [0, 2]
            cross_results = experiment_cross_axis(
                scope,
                cross_threads,
                move_distance_um=args.move_distance,
                timeout_s=args.move_timeout,
            )
            print()

        if 4 in experiments:
            print(f"--- Experiment 4: Sleep between polls ({args.sleep_threads} threads) ---")
            sleep_results = experiment_poll_with_sleep(
                scope,
                thread_count=args.sleep_threads,
                sleep_values_ms=sleep_values,
                move_distance_um=args.move_distance,
                timeout_s=args.move_timeout,
            )
            print()

        print_summary(poll_results, move_results, cross_results, sleep_results)

        # Save results to JSON
        output_path = args.output
        if output_path is None:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_path = str(results_dir / f"sdk_threading_{timestamp}.json")

        results_json = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "thread_counts": thread_counts,
                "experiments": experiments,
                "poll_duration_s": args.poll_duration,
                "move_timeout_s": args.move_timeout,
                "move_distance_um": args.move_distance,
                "sleep_threads": args.sleep_threads,
                "sleep_values_ms": sleep_values,
            },
            "experiment_1_poll_throughput": [asdict(r) for r in poll_results],
            "experiment_2_move_under_load": [asdict(r) for r in move_results],
            "experiment_3_cross_axis": [asdict(r) for r in cross_results],
            "experiment_4_sleep_mitigation": [asdict(r) for r in sleep_results],
        }

        with open(output_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
