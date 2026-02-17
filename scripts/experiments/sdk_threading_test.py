"""Characterize SDK threading constraints.

Experiments:
  1. Poll throughput vs thread count — how does GetControlValue() rate
     scale with 1, 2, 4, 8 concurrent polling threads?
  2. Move completion under load — does move_to_async + wait() complete
     when N threads are polling? At what N does it break?
  3. Cross-axis interference — poll axis A while moving axis B.
  4. Sleep between polls — does yielding between calls fix starvation?
  5. Camera + polling — does continuous Acquire() degrade polling or
     vice versa? Does a concurrent move complete?
  6. Full combined workload — multi-axis polling + camera + async move
     (the actual chip_scan scenario).
  7. Poll stagger analysis — do 2-thread polls cluster or interleave?
     Tests with and without explicit stagger offset.
  8. Coordinated 2-thread polling — threads coordinate via shared
     timestamps so each waits target_gap after peer's last completion.
     Continuous drift correction to maintain interleaving.
  9. Multi-axis throughput — 1 thread per axis polling concurrently.
     Tests 2-axis (X+Y) and 3-axis (X+Y+Z) to measure per-axis Hz
     when sharing the USB bus.
 10. Continuous velocity + halt under polling load — does halt() get
     through when polling saturates the bus?
 11. Sync move in thread + halt under polling load — same question
     but with blocking SetControlValue in a separate thread.

Usage:
    uv run python scripts/sdk_threading_test.py
    uv run python scripts/sdk_threading_test.py --experiments 1,2  # select experiments
    uv run python scripts/sdk_threading_test.py --experiments 5,6,7  # camera + stagger
    uv run python scripts/sdk_threading_test.py --experiments 5 --camera-threads 0,1,3
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


@dataclass
class CameraPollResult:
    poll_thread_count: int
    poll_sleep_ms: float
    duration_s: float
    camera_fps: float
    camera_frames: int
    poll_hz: float
    poll_total_calls: int
    move_completed: bool | None  # None = no move tested
    move_time_s: float | None
    poll_axis: str = "x"
    move_axis: str = "x"
    move_distance_um: float = 0
    notes: str = ""


@dataclass
class FullWorkloadResult:
    config: str
    poll_threads_per_axis: int
    total_poll_threads: int
    poll_sleep_ms: float
    duration_s: float
    camera_fps: float
    camera_frames: int
    poll_hz_x: float
    poll_hz_y: float
    poll_hz_z: float
    poll_hz_total: float
    move_completed: bool
    move_time_s: float | None
    move_axis: str = "x"
    move_distance_um: float = 2000


@dataclass
class StaggerResult:
    thread_count: int
    sleep_ms: float
    stagger_ms: float
    duration_s: float
    total_samples: int
    aggregate_hz: float
    gap_mean_ms: float
    gap_std_ms: float
    gap_min_ms: float
    gap_max_ms: float
    gap_p10_ms: float
    gap_p50_ms: float
    gap_p90_ms: float
    clustered_count: int  # gaps < 2ms
    clustered_pct: float


@dataclass
class MultiAxisResult:
    axes: list[str]
    duration_s: float
    per_axis_calls: dict[str, int]
    per_axis_hz: dict[str, float]
    aggregate_hz: float
    per_axis_mean_call_us: dict[str, float]
    sleep_ms: float = 0


@dataclass
class HaltUnderLoadResult:
    mode: str  # "continuous_velocity" or "sync_move_thread"
    poll_hz: float
    halt_latency_s: float
    stage_stopped: bool
    move_duration_s: float
    distance_traveled_um: float
    poll_axis: str = "x"
    move_axis: str = "x"
    notes: str = ""


@dataclass
class CoordinatedResult:
    target_gap_ms: float
    thread_count: int
    duration_s: float
    total_samples: int
    aggregate_hz: float
    gap_mean_ms: float
    gap_std_ms: float
    gap_min_ms: float
    gap_max_ms: float
    gap_p10_ms: float
    gap_p50_ms: float
    gap_p90_ms: float
    clustered_count: int  # gaps < 2ms
    clustered_pct: float
    move_completed: bool | None = None  # None = no move tested
    move_time_s: float | None = None


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


def _poll_loop_ts_pos(
    bcv,
    stop: threading.Event,
    ts_out: list[float],
    pos_out: list[int],
    sleep_s: float = 0,
):
    """Polling loop that records timestamps and position values."""
    while not stop.is_set():
        val = bcv.GetControlValue()
        ts_out.append(time.perf_counter())
        pos_out.append(val)
        if sleep_s > 0:
            time.sleep(sleep_s)


def _poll_loop_timestamped(
    bcv,
    stop: threading.Event,
    ts_out: list,
    idx: int,
    sleep_s: float = 0,
    stagger_s: float = 0,
):
    """Polling loop that records per-sample timestamps."""
    if stagger_s > 0:
        time.sleep(stagger_s)
    local_ts: list[float] = []
    while not stop.is_set():
        bcv.GetControlValue()
        local_ts.append(time.perf_counter())
        if sleep_s > 0:
            time.sleep(sleep_s)
    ts_out[idx] = local_ts


def _poll_loop_coordinated(
    bcv,
    stop: threading.Event,
    ts_out: list,
    idx: int,
    completions: list[float],
    lock: threading.Lock,
    target_gap_s: float,
    initial_delay_s: float = 0,
):
    """Coordinated polling: wait target_gap after peer's last completion.

    Two threads alternate, each waiting until target_gap has elapsed since
    the other thread's most recent sample. This prevents both threads from
    polling simultaneously (clustering) by continuously correcting drift.
    """
    if initial_delay_s > 0:
        time.sleep(initial_delay_s)

    peer_idx = 1 - idx
    local_ts: list[float] = []

    while not stop.is_set():
        # Wait until target_gap after peer's last completion
        with lock:
            peer_last = completions[peer_idx]

        if peer_last > 0:
            wait_until = peer_last + target_gap_s
            now = time.perf_counter()
            if now < wait_until:
                time.sleep(wait_until - now)

        # Poll
        bcv.GetControlValue()
        now = time.perf_counter()
        local_ts.append(now)

        with lock:
            completions[idx] = now

    ts_out[idx] = local_ts


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
# Experiment 5: Camera + polling combined
# ---------------------------------------------------------------------------


def experiment_camera_polling(
    scope: Microscope,
    thread_counts: list[int],
    duration_s: float = 5.0,
    sleep_ms: float = 8.0,
    axis_name: str = "x",
    move_distance_um: float = 2000,
    timeout_s: float = 10.0,
) -> list[CameraPollResult]:
    """Test camera continuous capture + position polling combined."""
    axis = _get_axis(scope, axis_name)
    bcv = axis.bcv
    sleep_s = sleep_ms / 1000
    results = []

    camera = scope.camera
    camera.trigger_mode = 0  # CONTINUOUS

    for n in thread_counts:
        # --- Phase 1: camera + polling, no move ---
        counts = [0] * max(n, 1)
        stop = threading.Event()
        threads = []

        if n > 0:
            for i in range(n):
                t = threading.Thread(
                    target=_poll_loop,
                    args=(bcv, stop, counts, i, None, sleep_s),
                    daemon=True,
                )
                threads.append(t)

        start = time.perf_counter()
        for t in threads:
            t.start()

        with camera.stream() as stream:
            time.sleep(duration_s)
            cam_fps = stream.frame_rate
            cam_frames = stream.frames_captured

        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        elapsed = time.perf_counter() - start
        total_polls = sum(counts) if n > 0 else 0
        poll_hz = total_polls / elapsed if elapsed > 0 and n > 0 else 0

        results.append(
            CameraPollResult(
                poll_thread_count=n,
                poll_sleep_ms=sleep_ms if n > 0 else 0,
                duration_s=elapsed,
                camera_fps=cam_fps,
                camera_frames=cam_frames,
                poll_hz=poll_hz,
                poll_total_calls=total_polls,
                move_completed=None,
                move_time_s=None,
                poll_axis=axis_name,
                move_axis=axis_name,
            )
        )

        print(f"  {n} poll thread(s): cam={cam_fps:.1f} fps, poll={poll_hz:.0f} Hz")

        # --- Phase 2: camera + polling + move ---
        counts2 = [0] * max(n, 1)
        stop2 = threading.Event()
        threads2 = []

        if n > 0:
            for i in range(n):
                t = threading.Thread(
                    target=_poll_loop,
                    args=(bcv, stop2, counts2, i, None, sleep_s),
                    daemon=True,
                )
                threads2.append(t)

        with camera.stream() as stream:
            for t in threads2:
                t.start()
            time.sleep(0.1)  # settle

            move_start = time.perf_counter()
            handle = axis.move_rel_async(move_distance_um)
            completed = handle.wait(timeout=timeout_s)
            move_end = time.perf_counter()
            move_time = move_end - move_start if completed else None

            cam_fps2 = stream.frame_rate
            cam_frames2 = stream.frames_captured

        stop2.set()
        for t in threads2:
            t.join(timeout=2.0)

        move_elapsed = move_end - move_start
        total_polls2 = sum(counts2) if n > 0 else 0
        poll_hz2 = total_polls2 / move_elapsed if move_elapsed > 0 and n > 0 else 0

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

        results.append(
            CameraPollResult(
                poll_thread_count=n,
                poll_sleep_ms=sleep_ms if n > 0 else 0,
                duration_s=move_elapsed,
                camera_fps=cam_fps2,
                camera_frames=cam_frames2,
                poll_hz=poll_hz2,
                poll_total_calls=total_polls2,
                move_completed=completed,
                move_time_s=move_time,
                poll_axis=axis_name,
                move_axis=axis_name,
                move_distance_um=move_distance_um,
                notes="with_move",
            )
        )

        status = f"{move_time:.3f}s" if completed else "HUNG"
        print(f"    + move: {status}, cam={cam_fps2:.1f} fps, poll={poll_hz2:.0f} Hz")

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Full combined workload
# ---------------------------------------------------------------------------


def experiment_full_workload(
    scope: Microscope,
    move_distance_um: float = 2000,
    timeout_s: float = 10.0,
) -> list[FullWorkloadResult]:
    """Full chip_scan workload: multi-axis polling + camera + move."""
    camera = scope.camera
    camera.trigger_mode = 0  # CONTINUOUS

    configs = [
        ("1/axis_no_sleep", 1, 0.0),
        ("2/axis_8ms_sleep", 2, 8.0),
    ]

    results = []

    for config_name, threads_per_axis, sleep_ms in configs:
        sleep_s = sleep_ms / 1000
        stop = threading.Event()

        # Start per-axis poll threads
        axis_counts: dict[str, list[int]] = {}
        all_threads: list[threading.Thread] = []

        for ax_name in ("x", "y", "z"):
            ax = _get_axis(scope, ax_name)
            counts = [0] * threads_per_axis
            axis_counts[ax_name] = counts
            for i in range(threads_per_axis):
                t = threading.Thread(
                    target=_poll_loop,
                    args=(ax.bcv, stop, counts, i, None, sleep_s),
                    daemon=True,
                )
                all_threads.append(t)

        # Start camera + poll threads, then move
        with camera.stream() as stream:
            for t in all_threads:
                t.start()
            time.sleep(0.1)  # settle

            t_start = time.perf_counter()
            handle = scope.stage.x.move_rel_async(move_distance_um)
            completed = handle.wait(timeout=timeout_s)
            t_end = time.perf_counter()

            cam_fps = stream.frame_rate
            cam_frames = stream.frames_captured

        stop.set()
        for t in all_threads:
            t.join(timeout=2.0)

        elapsed = t_end - t_start
        move_time = elapsed if completed else None

        handle.dispose()

        if completed:
            rev = scope.stage.x.move_rel_async(-move_distance_um)
            rev.wait(timeout=10.0)
            rev.dispose()
        else:
            scope.stage.x.halt()
            time.sleep(0.5)
            with contextlib.suppress(Exception):
                scope.stage.x.move_rel(-move_distance_um)

        hz_x = sum(axis_counts["x"]) / elapsed if elapsed > 0 else 0
        hz_y = sum(axis_counts["y"]) / elapsed if elapsed > 0 else 0
        hz_z = sum(axis_counts["z"]) / elapsed if elapsed > 0 else 0
        hz_total = hz_x + hz_y + hz_z

        result = FullWorkloadResult(
            config=config_name,
            poll_threads_per_axis=threads_per_axis,
            total_poll_threads=threads_per_axis * 3,
            poll_sleep_ms=sleep_ms,
            duration_s=elapsed,
            camera_fps=cam_fps,
            camera_frames=cam_frames,
            poll_hz_x=hz_x,
            poll_hz_y=hz_y,
            poll_hz_z=hz_z,
            poll_hz_total=hz_total,
            move_completed=completed,
            move_time_s=move_time,
            move_axis="x",
            move_distance_um=move_distance_um,
        )
        results.append(result)

        status = f"{move_time:.3f}s" if completed else "HUNG"
        print(
            f"  {config_name}: cam={cam_fps:.1f} fps, "
            f"X={hz_x:.0f} Y={hz_y:.0f} Z={hz_z:.0f} (total={hz_total:.0f} Hz), "
            f"move={status}"
        )

    return results


# ---------------------------------------------------------------------------
# Experiment 7: Poll stagger analysis
# ---------------------------------------------------------------------------


def experiment_stagger(
    scope: Microscope,
    duration_s: float = 3.0,
    sleep_ms: float = 8.0,
    axis_name: str = "x",
) -> list[StaggerResult]:
    """Analyze temporal distribution of 2-thread polling with/without stagger."""
    axis = _get_axis(scope, axis_name)
    bcv = axis.bcv
    sleep_s = sleep_ms / 1000

    # Optimal stagger: half the per-thread period (call_time + sleep)
    estimated_call_ms = 16.0
    optimal_stagger_ms = (estimated_call_ms + sleep_ms) / 2

    configs = [
        ("no_stagger", 2, 0.0),
        (f"stagger_{optimal_stagger_ms:.0f}ms", 2, optimal_stagger_ms),
        ("1_thread_ref", 1, 0.0),
    ]

    results = []

    for label, n_threads, stagger_ms in configs:
        ts_lists: list[list[float]] = [[] for _ in range(n_threads)]
        stop = threading.Event()
        threads = []

        for i in range(n_threads):
            t = threading.Thread(
                target=_poll_loop_timestamped,
                args=(
                    bcv,
                    stop,
                    ts_lists,
                    i,
                    sleep_s,
                    stagger_ms / 1000 if i > 0 else 0,
                ),
                daemon=True,
            )
            threads.append(t)

        for t in threads:
            t.start()
        time.sleep(duration_s)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        # Merge and sort all timestamps
        all_ts: list[float] = []
        for ts in ts_lists:
            all_ts.extend(ts)
        all_ts.sort()

        if len(all_ts) < 2:
            continue

        # Compute inter-sample gaps
        gaps_ms = [(all_ts[i + 1] - all_ts[i]) * 1000 for i in range(len(all_ts) - 1)]
        n_gaps = len(gaps_ms)
        mean_gap = sum(gaps_ms) / n_gaps
        std_gap = (sum((g - mean_gap) ** 2 for g in gaps_ms) / n_gaps) ** 0.5

        gaps_sorted = sorted(gaps_ms)
        clustered = sum(1 for g in gaps_ms if g < 2.0)

        elapsed = all_ts[-1] - all_ts[0]
        hz = len(all_ts) / elapsed if elapsed > 0 else 0

        result = StaggerResult(
            thread_count=n_threads,
            sleep_ms=sleep_ms,
            stagger_ms=stagger_ms,
            duration_s=elapsed,
            total_samples=len(all_ts),
            aggregate_hz=hz,
            gap_mean_ms=mean_gap,
            gap_std_ms=std_gap,
            gap_min_ms=gaps_sorted[0],
            gap_max_ms=gaps_sorted[-1],
            gap_p10_ms=gaps_sorted[int(n_gaps * 0.1)],
            gap_p50_ms=gaps_sorted[n_gaps // 2],
            gap_p90_ms=gaps_sorted[int(n_gaps * 0.9)],
            clustered_count=clustered,
            clustered_pct=clustered / n_gaps * 100,
        )
        results.append(result)

        print(
            f"  {label}: {hz:.0f} Hz, gap={mean_gap:.1f}\u00b1{std_gap:.1f}ms, "
            f"clustered={clustered}/{n_gaps} ({clustered / n_gaps * 100:.0f}%)"
        )

    return results


# ---------------------------------------------------------------------------
# Experiment 8: Coordinated 2-thread polling
# ---------------------------------------------------------------------------


def experiment_coordinated_stagger(
    scope: Microscope,
    duration_s: float = 3.0,
    target_gaps_ms: list[float] | None = None,
    axis_name: str = "x",
    move_distance_um: float = 2000,
    timeout_s: float = 10.0,
) -> list[CoordinatedResult]:
    """Test coordinated 2-thread polling with continuous drift correction."""
    if target_gaps_ms is None:
        target_gaps_ms = [6.0, 8.0, 10.0]

    axis = _get_axis(scope, axis_name)
    bcv = axis.bcv
    results = []

    for target_gap_ms in target_gaps_ms:
        target_gap_s = target_gap_ms / 1000

        # --- Phase 1: steady-state gap analysis ---
        ts_lists: list[list[float]] = [[], []]
        completions = [0.0, 0.0]
        lock = threading.Lock()
        stop = threading.Event()

        threads = []
        for i in range(2):
            t = threading.Thread(
                target=_poll_loop_coordinated,
                args=(bcv, stop, ts_lists, i, completions, lock, target_gap_s),
                kwargs={"initial_delay_s": target_gap_s * i},
                daemon=True,
            )
            threads.append(t)

        for t in threads:
            t.start()
        time.sleep(duration_s)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        # Merge and sort timestamps
        all_ts = sorted([t for ts in ts_lists for t in ts])
        if len(all_ts) < 2:
            continue

        # Compute gap stats
        gaps_ms = [(all_ts[i + 1] - all_ts[i]) * 1000 for i in range(len(all_ts) - 1)]
        n_gaps = len(gaps_ms)
        mean_gap = sum(gaps_ms) / n_gaps
        std_gap = (sum((g - mean_gap) ** 2 for g in gaps_ms) / n_gaps) ** 0.5
        gaps_sorted = sorted(gaps_ms)
        clustered = sum(1 for g in gaps_ms if g < 2.0)

        elapsed = all_ts[-1] - all_ts[0]
        hz = len(all_ts) / elapsed if elapsed > 0 else 0

        print(
            f"  gap={target_gap_ms:.0f}ms: {hz:.0f} Hz, "
            f"gap={mean_gap:.1f}\u00b1{std_gap:.1f}ms, "
            f"clustered={clustered}/{n_gaps} ({clustered / n_gaps * 100:.0f}%)"
        )

        # --- Phase 2: move safety ---
        ts_lists2: list[list[float]] = [[], []]
        completions2 = [0.0, 0.0]
        lock2 = threading.Lock()
        stop2 = threading.Event()

        threads2 = []
        for i in range(2):
            t = threading.Thread(
                target=_poll_loop_coordinated,
                args=(bcv, stop2, ts_lists2, i, completions2, lock2, target_gap_s),
                kwargs={"initial_delay_s": target_gap_s * i},
                daemon=True,
            )
            threads2.append(t)

        for t in threads2:
            t.start()
        time.sleep(0.2)  # let coordination stabilize

        move_start = time.perf_counter()
        handle = axis.move_rel_async(move_distance_um)
        completed = handle.wait(timeout=timeout_s)
        move_time = time.perf_counter() - move_start if completed else None

        stop2.set()
        for t in threads2:
            t.join(timeout=2.0)

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

        status = f"{move_time:.3f}s" if completed else "HUNG"
        print(f"    + move: {status}")

        results.append(
            CoordinatedResult(
                target_gap_ms=target_gap_ms,
                thread_count=2,
                duration_s=elapsed,
                total_samples=len(all_ts),
                aggregate_hz=hz,
                gap_mean_ms=mean_gap,
                gap_std_ms=std_gap,
                gap_min_ms=gaps_sorted[0],
                gap_max_ms=gaps_sorted[-1],
                gap_p10_ms=gaps_sorted[int(n_gaps * 0.1)],
                gap_p50_ms=gaps_sorted[n_gaps // 2],
                gap_p90_ms=gaps_sorted[int(n_gaps * 0.9)],
                clustered_count=clustered,
                clustered_pct=clustered / n_gaps * 100,
                move_completed=completed,
                move_time_s=move_time,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Experiment 9: Multi-axis throughput (1 thread per axis)
# ---------------------------------------------------------------------------


def experiment_multi_axis_throughput(
    scope: Microscope,
    duration_s: float = 3.0,
    sleep_ms: float = 0,
) -> list[MultiAxisResult]:
    """Poll multiple axes concurrently, 1 thread per axis."""
    sleep_s = sleep_ms / 1000
    results = []

    configs: list[tuple[str, list[str]]] = [
        ("X+Y", ["x", "y"]),
        ("X+Y+Z", ["x", "y", "z"]),
    ]

    for label, axis_names in configs:
        axes = {name: _get_axis(scope, name) for name in axis_names}
        counts = {name: [0] for name in axis_names}
        latencies: dict[str, list[float]] = {name: [] for name in axis_names}
        stop = threading.Event()
        threads = []

        for name in axis_names:
            t = threading.Thread(
                target=_poll_loop,
                args=(axes[name].bcv, stop, counts[name], 0, latencies[name], sleep_s),
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

        per_axis_calls = {name: counts[name][0] for name in axis_names}
        per_axis_hz = {name: counts[name][0] / elapsed for name in axis_names}
        per_axis_us = {}
        for name in axis_names:
            lats = latencies[name]
            per_axis_us[name] = sum(lats) / len(lats) * 1e6 if lats else 0
        total = sum(per_axis_calls.values())

        result = MultiAxisResult(
            axes=axis_names,
            duration_s=elapsed,
            per_axis_calls=per_axis_calls,
            per_axis_hz=per_axis_hz,
            aggregate_hz=total / elapsed,
            per_axis_mean_call_us=per_axis_us,
            sleep_ms=sleep_ms,
        )
        results.append(result)

        hz_parts = ", ".join(f"{n.upper()}={per_axis_hz[n]:.0f}" for n in axis_names)
        us_parts = ", ".join(f"{n.upper()}={per_axis_us[n]:.0f}" for n in axis_names)
        print(f"  {label}: {hz_parts} Hz (total={total / elapsed:.0f}), call={us_parts} µs")

    return results


# ---------------------------------------------------------------------------
# Experiment 10: Continuous velocity + halt under polling load
# ---------------------------------------------------------------------------


def experiment_velocity_halt(
    scope: Microscope,
    move_duration_s: float = 1.0,
    velocity_um_s: float = 5000,
    axis_name: str = "x",
    sleep_ms: float = 0,
) -> HaltUnderLoadResult:
    """Start continuous velocity move, poll at given rate, then halt."""
    axis = _get_axis(scope, axis_name)
    bcv = axis.bcv
    sleep_s = sleep_ms / 1000

    pos_before = axis.position_um
    counts = [0]
    stop = threading.Event()

    poll_t = threading.Thread(
        target=_poll_loop,
        args=(bcv, stop, counts, 0, None, sleep_s),
        daemon=True,
    )

    # Start continuous velocity move
    axis.start_towards_max(velocity_um_s)
    t_start = time.perf_counter()
    poll_t.start()

    # Let it run
    time.sleep(move_duration_s)

    # Halt and measure latency
    t_halt_start = time.perf_counter()
    axis.halt()
    t_halt_end = time.perf_counter()

    # Check if stage actually stopped
    time.sleep(0.05)
    pos_after_halt = axis.position_um
    time.sleep(0.1)
    pos_check = axis.position_um
    stage_stopped = abs(pos_check - pos_after_halt) < 1.0  # <1 µm drift = stopped

    stop.set()
    poll_t.join(timeout=2.0)

    elapsed = t_halt_end - t_start
    poll_hz = counts[0] / elapsed if elapsed > 0 else 0
    halt_latency = t_halt_end - t_halt_start
    distance = abs(pos_after_halt - pos_before)

    # Move back
    axis.move_to(pos_before)

    result = HaltUnderLoadResult(
        mode="continuous_velocity",
        poll_hz=poll_hz,
        halt_latency_s=halt_latency,
        stage_stopped=stage_stopped,
        move_duration_s=elapsed,
        distance_traveled_um=distance,
        poll_axis=axis_name,
        move_axis=axis_name,
    )

    print(
        f"  Continuous velocity: poll={poll_hz:.0f} Hz, "
        f"halt={halt_latency * 1000:.1f} ms, "
        f"stopped={'YES' if stage_stopped else 'NO'}, "
        f"traveled={distance:.0f} µm"
    )

    return result


# ---------------------------------------------------------------------------
# Experiment 11: Sync move in thread under polling load
# ---------------------------------------------------------------------------


def experiment_sync_move_under_load(
    scope: Microscope,
    move_distance_um: float = 2000,
    timeout_s: float = 10.0,
    axis_name: str = "x",
) -> MoveTestResult:
    """Blocking move_rel in a thread while polling same axis at max rate."""
    axis = _get_axis(scope, axis_name)
    bcv = axis.bcv

    counts = [0]
    stop = threading.Event()
    move_done = threading.Event()
    move_exception: list[Exception | None] = [None]

    def sync_move():
        try:
            axis.move_rel(move_distance_um)
        except Exception as e:
            move_exception[0] = e
        finally:
            move_done.set()

    # Start polling thread (no sleep, max rate)
    poll_t = threading.Thread(
        target=_poll_loop,
        args=(bcv, stop, counts, 0),
        daemon=True,
    )

    # Start sync move in thread + polling
    t_start = time.perf_counter()
    move_t = threading.Thread(target=sync_move, daemon=True)
    move_t.start()
    poll_t.start()

    # Wait for move to complete or timeout
    completed = move_done.wait(timeout=timeout_s)
    move_time = time.perf_counter() - t_start if completed else None

    stop.set()
    poll_t.join(timeout=2.0)
    move_t.join(timeout=2.0)

    poll_elapsed = time.perf_counter() - t_start
    poll_hz = counts[0] / poll_elapsed if poll_elapsed > 0 else 0

    notes = ""
    if move_exception[0]:
        notes = f"exception: {move_exception[0]}"

    # Move back if completed
    if completed:
        axis.move_rel(-move_distance_um)
    else:
        axis.halt()
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            axis.move_rel(-move_distance_um)

    result = MoveTestResult(
        thread_count=1,
        completed=completed,
        move_time_s=move_time,
        timeout_s=timeout_s,
        poll_hz_during=poll_hz,
        poll_axis=axis_name,
        move_axis=axis_name,
        move_distance_um=move_distance_um,
        notes=f"sync_move {notes}".strip(),
    )

    status = f"{move_time:.3f}s" if completed else f"HUNG (timeout {timeout_s}s)"
    print(f"  Sync move: {status}, poll={poll_hz:.0f} Hz" + (f", {notes}" if notes else ""))

    return result


# ---------------------------------------------------------------------------
# Experiment 12: Jitter + position smoothness during velocity move
# ---------------------------------------------------------------------------


@dataclass
class JitterResult:
    sleep_ms: float
    poll_hz: float
    total_samples: int
    duration_s: float
    gap_mean_ms: float
    gap_std_ms: float
    gap_min_ms: float
    gap_max_ms: float
    gap_p10_ms: float
    gap_p50_ms: float
    gap_p90_ms: float
    velocity_um_s: float  # linear fit
    pos_residual_std_um: float  # position smoothness (residual from linear fit)
    pos_residual_max_um: float
    delta_mean_um: float  # inter-sample position deltas
    delta_std_um: float
    delta_min_um: float
    delta_max_um: float
    delta_zero_count: int  # duplicate readings (delta=0)
    delta_zero_pct: float
    axis: str = "x"


def experiment_jitter(
    scope: Microscope,
    duration_s: float = 3.0,
    velocity_um_s: float = 5000,
    sleep_ms: float = 6.0,
    axis_name: str = "x",
) -> JitterResult:
    """Continuous velocity move with timestamped position polling. Reports timing jitter and position smoothness."""
    axis = _get_axis(scope, axis_name)
    bcv = axis.bcv
    sleep_s = sleep_ms / 1000

    pos_before = axis.position_um

    ts_list: list[float] = []
    pos_list: list[int] = []
    stop = threading.Event()

    poll_t = threading.Thread(
        target=_poll_loop_ts_pos,
        args=(bcv, stop, ts_list, pos_list, sleep_s),
        daemon=True,
    )

    axis.start_towards_max(velocity_um_s)
    poll_t.start()
    time.sleep(duration_s)
    stop.set()
    poll_t.join(timeout=2.0)
    axis.halt()
    time.sleep(0.1)

    # Move back
    axis.move_to(pos_before)

    n = len(ts_list)
    if n < 2:
        raise RuntimeError("No samples collected")

    # Convert positions to µm
    conv = axis.converter
    positions_um = [conv.GetMetricsValue(p) for p in pos_list]

    # Timing gaps
    gaps_ms = [(ts_list[i + 1] - ts_list[i]) * 1000 for i in range(n - 1)]
    n_gaps = len(gaps_ms)
    mean_gap = sum(gaps_ms) / n_gaps
    std_gap = (sum((g - mean_gap) ** 2 for g in gaps_ms) / n_gaps) ** 0.5
    gaps_sorted = sorted(gaps_ms)

    elapsed = ts_list[-1] - ts_list[0]
    hz = n / elapsed

    # Linear fit for velocity and residuals
    t0 = ts_list[0]
    times = [t - t0 for t in ts_list]
    t_mean = sum(times) / n
    p_mean = sum(positions_um) / n
    num = sum((t - t_mean) * (p - p_mean) for t, p in zip(times, positions_um, strict=True))
    den = sum((t - t_mean) ** 2 for t in times)
    slope = num / den if den > 0 else 0  # velocity in µm/s
    intercept = p_mean - slope * t_mean

    residuals = [p - (slope * t + intercept) for t, p in zip(times, positions_um, strict=True)]
    res_std = (sum(r**2 for r in residuals) / n) ** 0.5
    res_max = max(abs(r) for r in residuals)

    # Position deltas
    deltas_um = [positions_um[i + 1] - positions_um[i] for i in range(n - 1)]
    n_deltas = len(deltas_um)
    abs_deltas = [abs(d) for d in deltas_um]
    d_mean = sum(abs_deltas) / n_deltas
    d_std = (sum((d - d_mean) ** 2 for d in abs_deltas) / n_deltas) ** 0.5
    d_min = min(abs_deltas)
    d_max = max(abs_deltas)
    d_zeros = sum(1 for d in abs_deltas if d < 0.1)  # <0.1 µm = duplicate

    result = JitterResult(
        sleep_ms=sleep_ms,
        poll_hz=hz,
        total_samples=n,
        duration_s=elapsed,
        gap_mean_ms=mean_gap,
        gap_std_ms=std_gap,
        gap_min_ms=gaps_sorted[0],
        gap_max_ms=gaps_sorted[-1],
        gap_p10_ms=gaps_sorted[int(n_gaps * 0.1)],
        gap_p50_ms=gaps_sorted[n_gaps // 2],
        gap_p90_ms=gaps_sorted[int(n_gaps * 0.9)],
        velocity_um_s=slope,
        pos_residual_std_um=res_std,
        pos_residual_max_um=res_max,
        delta_mean_um=d_mean,
        delta_std_um=d_std,
        delta_min_um=d_min,
        delta_max_um=d_max,
        delta_zero_count=d_zeros,
        delta_zero_pct=d_zeros / n_deltas * 100,
        axis=axis_name,
    )

    abs_deltas_sorted = sorted(abs_deltas)
    print(
        f"  {hz:.0f} Hz, gap={mean_gap:.1f}±{std_gap:.1f}ms "
        f"[{gaps_sorted[0]:.1f}, {gaps_sorted[int(n_gaps * 0.1)]:.1f}, "
        f"{gaps_sorted[n_gaps // 2]:.1f}, {gaps_sorted[int(n_gaps * 0.9)]:.1f}, "
        f"{gaps_sorted[-1]:.1f}]"
    )
    print(
        f"  velocity={slope:.0f} µm/s (target {velocity_um_s:.0f}), "
        f"pos residual std={res_std:.2f} µm, max={res_max:.2f} µm"
    )
    print(
        f"  pos delta={d_mean:.1f}±{d_std:.1f} µm "
        f"[{abs_deltas_sorted[0]:.1f}, {abs_deltas_sorted[int(n_deltas * 0.1)]:.1f}, "
        f"{abs_deltas_sorted[n_deltas // 2]:.1f}, {abs_deltas_sorted[int(n_deltas * 0.9)]:.1f}, "
        f"{abs_deltas_sorted[-1]:.1f}], "
        f"zeros={d_zeros}/{n_deltas} ({d_zeros / n_deltas * 100:.0f}%)"
    )

    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    poll_results: list[PollResult],
    move_results: list[MoveTestResult],
    cross_results: list[MoveTestResult],
    sleep_results: list[MoveTestResult],
    *,
    camera_results: list[CameraPollResult] | None = None,
    workload_results: list[FullWorkloadResult] | None = None,
    stagger_results: list[StaggerResult] | None = None,
    coordinated_results: list[CoordinatedResult] | None = None,
    multi_axis_results: list[MultiAxisResult] | None = None,
    halt_results: list[HaltUnderLoadResult] | None = None,
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

    if camera_results:
        no_move = [r for r in camera_results if r.notes != "with_move"]
        with_move = [r for r in camera_results if r.notes == "with_move"]

        print()
        sleep_val = no_move[0].poll_sleep_ms if no_move else 0
        print(f"Experiment 5: Camera + polling combined ({sleep_val:.0f}ms sleep)")

        if no_move:
            print("  No-move test:")
            print(f"    {'Polls':>6} {'Cam fps':>10} {'Poll Hz':>10}")
            print(f"    {'------':>6} {'----------':>10} {'----------':>10}")
            for r in no_move:
                print(f"    {r.poll_thread_count:>6} {r.camera_fps:>10.1f} {r.poll_hz:>10.0f}")

        if with_move:
            dist = with_move[0].move_distance_um
            ax = with_move[0].move_axis.upper()
            print(f"  With move ({ax}, {dist:.0f} µm):")
            print(f"    {'Polls':>6} {'Cam fps':>10} {'Poll Hz':>10} {'Move':>6} {'Time':>10}")
            print(f"    {'------':>6} {'----------':>10} {'----------':>10} {'------':>6} {'----------':>10}")
            for r in with_move:
                ok = "YES" if r.move_completed else "NO"
                t = f"{r.move_time_s:.3f}s" if r.move_time_s is not None else "---"
                print(f"    {r.poll_thread_count:>6} {r.camera_fps:>10.1f} {r.poll_hz:>10.0f} {ok:>6} {t:>10}")

    if workload_results:
        print()
        print("Experiment 6: Full combined workload")
        print(
            f"  {'Config':<22} {'Cam fps':>8} {'X Hz':>6} {'Y Hz':>6} {'Z Hz':>6} {'Total':>6} {'Move':>5} {'Time':>8}"
        )
        print(
            f"  {'----------------------':<22} {'--------':>8} {'------':>6} {'------':>6} "
            f"{'------':>6} {'------':>6} {'-----':>5} {'--------':>8}"
        )
        for r in workload_results:
            ok = "YES" if r.move_completed else "NO"
            t = f"{r.move_time_s:.3f}s" if r.move_time_s is not None else "---"
            print(
                f"  {r.config:<22} {r.camera_fps:>8.1f} {r.poll_hz_x:>6.0f} {r.poll_hz_y:>6.0f} "
                f"{r.poll_hz_z:>6.0f} {r.poll_hz_total:>6.0f} {ok:>5} {t:>8}"
            )

    if stagger_results:
        print()
        print("Experiment 7: Poll stagger analysis")
        print(
            f"  {'Config':<18} {'Hz':>5} {'Gap mean±std (ms)':>20} {'P10':>6} {'P50':>6} {'P90':>6} {'Clustered':>10}"
        )
        print(
            f"  {'------------------':<18} {'-----':>5} {'--------------------':>20} "
            f"{'------':>6} {'------':>6} {'------':>6} {'----------':>10}"
        )
        for r in stagger_results:
            label = f"{r.thread_count}T"
            if r.stagger_ms > 0:
                label += f" stg={r.stagger_ms:.0f}ms"
            else:
                label += " no stagger"
            gap_str = f"{r.gap_mean_ms:.1f}\u00b1{r.gap_std_ms:.1f}"
            clust_str = f"{r.clustered_count} ({r.clustered_pct:.0f}%)"
            print(
                f"  {label:<18} {r.aggregate_hz:>5.0f} {gap_str:>20} "
                f"{r.gap_p10_ms:>6.1f} {r.gap_p50_ms:>6.1f} {r.gap_p90_ms:>6.1f} {clust_str:>10}"
            )

    if coordinated_results:
        print()
        print("Experiment 8: Coordinated 2-thread polling")
        print(
            f"  {'Gap(ms)':>8} {'Hz':>5} {'Gap mean\u00b1std (ms)':>20} "
            f"{'P10':>6} {'P50':>6} {'P90':>6} {'Clustered':>10} {'Move':>5} {'Time':>8}"
        )
        print(
            f"  {'--------':>8} {'-----':>5} {'--------------------':>20} "
            f"{'------':>6} {'------':>6} {'------':>6} {'----------':>10} {'-----':>5} {'--------':>8}"
        )
        for r in coordinated_results:
            gap_str = f"{r.gap_mean_ms:.1f}\u00b1{r.gap_std_ms:.1f}"
            clust_str = f"{r.clustered_count} ({r.clustered_pct:.0f}%)"
            ok = "YES" if r.move_completed else ("NO" if r.move_completed is not None else "---")
            t = f"{r.move_time_s:.3f}s" if r.move_time_s is not None else "---"
            print(
                f"  {r.target_gap_ms:>8.0f} {r.aggregate_hz:>5.0f} {gap_str:>20} "
                f"{r.gap_p10_ms:>6.1f} {r.gap_p50_ms:>6.1f} {r.gap_p90_ms:>6.1f} "
                f"{clust_str:>10} {ok:>5} {t:>8}"
            )

    if multi_axis_results:
        print()
        print("Experiment 9: Multi-axis throughput (1 thread/axis)")
        for r in multi_axis_results:
            label = "+".join(n.upper() for n in r.axes)
            hz_parts = "  ".join(f"{n.upper()}={r.per_axis_hz[n]:.0f}" for n in r.axes)
            us_parts = "  ".join(f"{n.upper()}={r.per_axis_mean_call_us[n]:.0f}" for n in r.axes)
            sleep_str = f" (sleep {r.sleep_ms:.0f}ms)" if r.sleep_ms > 0 else ""
            print(f"  {label}{sleep_str}: {hz_parts} Hz, total={r.aggregate_hz:.0f} Hz, call={us_parts} µs")

    if halt_results:
        print()
        print("Experiments 10/11: Halt under polling load")
        print(f"  {'Mode':<22} {'Poll Hz':>8} {'Halt ms':>8} {'Stopped':>8} {'Distance':>10}")
        print(f"  {'----------------------':<22} {'--------':>8} {'--------':>8} {'--------':>8} {'----------':>10}")
        for r in halt_results:
            print(
                f"  {r.mode:<22} {r.poll_hz:>8.0f} {r.halt_latency_s * 1000:>8.1f} "
                f"{'YES' if r.stage_stopped else 'NO':>8} {r.distance_traveled_um:>10.0f}"
                + (f"  {r.notes}" if r.notes else "")
            )

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
        "--camera-threads",
        type=str,
        default="0,1,2,3,6",
        help="Comma-separated poll thread counts for camera experiments (default: 0,1,2,3,6)",
    )
    parser.add_argument(
        "--camera-duration",
        type=float,
        default=5.0,
        help="Duration for camera+polling steady-state test in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--camera-sleep",
        type=float,
        default=8.0,
        help="Sleep between polls in ms for camera experiments (default: 8.0)",
    )
    parser.add_argument(
        "--coord-gaps",
        type=str,
        default="6,8,10",
        help="Comma-separated target gap values in ms for experiment 8 (default: 6,8,10)",
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
    camera_thread_counts = [int(x) for x in args.camera_threads.split(",")]
    coord_gaps_ms = [float(x) for x in args.coord_gaps.split(",")]

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
    if any(e in experiments for e in [5, 6]):
        print(f"Camera poll threads: {camera_thread_counts}")
        print(f"Camera duration: {args.camera_duration}s")
        print(f"Camera poll sleep: {args.camera_sleep}ms")
    print()

    with Microscope() as scope:
        poll_results: list[PollResult] = []
        move_results: list[MoveTestResult] = []
        cross_results: list[MoveTestResult] = []
        sleep_results: list[MoveTestResult] = []
        camera_results: list[CameraPollResult] = []
        workload_results: list[FullWorkloadResult] = []
        stagger_results: list[StaggerResult] = []
        coordinated_results: list[CoordinatedResult] = []
        multi_axis_results: list[MultiAxisResult] = []
        halt_results: list[HaltUnderLoadResult] = []

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

        if 5 in experiments:
            print(f"--- Experiment 5: Camera + polling ({args.camera_sleep}ms sleep) ---")
            camera_results = experiment_camera_polling(
                scope,
                camera_thread_counts,
                duration_s=args.camera_duration,
                sleep_ms=args.camera_sleep,
                move_distance_um=args.move_distance,
                timeout_s=args.move_timeout,
            )
            print()

        if 6 in experiments:
            print("--- Experiment 6: Full combined workload ---")
            workload_results = experiment_full_workload(
                scope,
                move_distance_um=args.move_distance,
                timeout_s=args.move_timeout,
            )
            print()

        if 7 in experiments:
            print("--- Experiment 7: Poll stagger analysis ---")
            stagger_results = experiment_stagger(
                scope,
                duration_s=args.poll_duration,
            )
            print()

        if 8 in experiments:
            print("--- Experiment 8: Coordinated 2-thread polling ---")
            coordinated_results = experiment_coordinated_stagger(
                scope,
                duration_s=args.poll_duration,
                target_gaps_ms=coord_gaps_ms,
                move_distance_um=args.move_distance,
                timeout_s=args.move_timeout,
            )
            print()

        if 9 in experiments:
            sleep_9 = float(args.camera_sleep) if args.camera_sleep else 0
            print(f"--- Experiment 9: Multi-axis throughput (sleep={sleep_9}ms) ---")
            multi_axis_results = experiment_multi_axis_throughput(
                scope,
                duration_s=args.poll_duration,
                sleep_ms=sleep_9,
            )
            print()

        if 10 in experiments:
            sleep_val = float(args.camera_sleep) if args.camera_sleep else 0
            print(f"--- Experiment 10: Continuous velocity + halt under polling (sleep={sleep_val}ms) ---")
            halt_results.append(experiment_velocity_halt(scope, sleep_ms=sleep_val))
            print()

        if 11 in experiments:
            print("--- Experiment 11: Sync move in thread under polling ---")
            experiment_sync_move_under_load(scope)
            print()

        if 12 in experiments:
            sleep_12 = float(args.camera_sleep) if args.camera_sleep else 6.0
            print(f"--- Experiment 12: Jitter + smoothness during velocity move (sleep={sleep_12}ms) ---")
            experiment_jitter(
                scope,
                duration_s=args.poll_duration,
                sleep_ms=sleep_12,
            )
            print()

        print_summary(
            poll_results,
            move_results,
            cross_results,
            sleep_results,
            camera_results=camera_results,
            workload_results=workload_results,
            stagger_results=stagger_results,
            coordinated_results=coordinated_results,
            multi_axis_results=multi_axis_results,
            halt_results=halt_results,
        )

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
                "camera_thread_counts": camera_thread_counts,
                "camera_duration_s": args.camera_duration,
                "camera_sleep_ms": args.camera_sleep,
                "coord_gaps_ms": coord_gaps_ms,
            },
            "experiment_1_poll_throughput": [asdict(r) for r in poll_results],
            "experiment_2_move_under_load": [asdict(r) for r in move_results],
            "experiment_3_cross_axis": [asdict(r) for r in cross_results],
            "experiment_4_sleep_mitigation": [asdict(r) for r in sleep_results],
            "experiment_5_camera_polling": [asdict(r) for r in camera_results],
            "experiment_6_full_workload": [asdict(r) for r in workload_results],
            "experiment_7_stagger_analysis": [asdict(r) for r in stagger_results],
            "experiment_8_coordinated": [asdict(r) for r in coordinated_results],
            "experiment_9_multi_axis": [asdict(r) for r in multi_axis_results],
            "experiment_10_11_halt": [asdict(r) for r in halt_results],
        }

        with open(output_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
