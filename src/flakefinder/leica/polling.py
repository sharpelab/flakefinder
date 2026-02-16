"""Rate-limited position polling for scan threads.

Provides a reusable poll-sleep loop that keeps position sampling at a
target Hz, preventing SDK bus starvation when using the fast cmusb (D2XX)
transport (~4.8 ms/call, ~206 Hz unthrottled).  At the older 16 ms VCP
transport the call itself exceeds the target period, so no sleep occurs
and the natural ~63 Hz rate is preserved.

See docs/poll_throttling_plan.md for background.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..types import PositionSample

DEFAULT_POLL_HZ: float = 100.0
STARTUP_POLL_HZ: float = 10.0
_MOTION_THRESHOLD_UM: float = 1.0


@dataclass
class PollingHandle:
    """Handle returned by start_polling(). Owns the samples, stop event, and thread."""

    samples: list[PositionSample] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    # Private — stored for deferred thread creation (paused=True)
    _bcv: Any = field(default=None, repr=False)
    _converter: Any = field(default=None, repr=False)
    _target_hz: float = field(default=DEFAULT_POLL_HZ, repr=False)
    _startup_hz: float | None = field(default=None, repr=False)

    def start(self) -> None:
        """Start the polling thread (for use after paused=True)."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=poll_position,
            args=(self._bcv, self._converter, self.samples, self.stop),
            kwargs={"target_hz": self._target_hz, "startup_hz": self._startup_hz},
            daemon=True,
        )
        self.thread.start()

    def join(self, timeout: float = 1.0) -> None:
        """Signal stop and join the polling thread."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)


def poll_position(
    bcv,
    converter,
    samples: list[PositionSample],
    stop: threading.Event,
    *,
    target_hz: float = DEFAULT_POLL_HZ,
    startup_hz: float | None = None,
) -> None:
    """Thread-target: poll position, convert, append PositionSample, sleep to target rate.

    When startup_hz is provided, starts at that rate and ramps to target_hz
    once motion is detected (>1 µm from initial position).

    Args:
        bcv: BasicControlValue interface (or hysteresis-corrected variant).
        converter: MetricsConverter for native -> microns.
        samples: Shared list to append PositionSample results.
        stop: Event to signal thread shutdown.
        target_hz: Target polling rate in Hz.
        startup_hz: If set, initial polling rate before motion detected.
    """
    if startup_hz is not None:
        period = 1.0 / startup_hz
        initial_pos = samples[0].axis_um if samples else None
        ramped = False
    else:
        period = 1.0 / target_hz
        initial_pos = None
        ramped = True

    while not stop.is_set():
        t_before = time.perf_counter()
        native = bcv.GetControlValue()
        t_after = time.perf_counter()
        um = converter.GetMetricsValue(native)
        samples.append(PositionSample(t_before, t_after, um))

        if not ramped and abs(um - initial_pos) > _MOTION_THRESHOLD_UM:
            period = 1.0 / target_hz
            ramped = True

        elapsed = t_after - t_before
        if elapsed < period:
            time.sleep(period - elapsed)


def start_polling(
    bcv,
    converter,
    *,
    target_hz: float = DEFAULT_POLL_HZ,
    startup_hz: float | None = None,
    paused: bool = False,
) -> PollingHandle:
    """Start a position polling thread.

    Always captures one initial position sample before returning.
    With paused=True, the thread is not started — call handle.start()
    after issuing any bus commands (e.g. move_to_async) that would
    compete with polling on the same axis.

    When startup_hz is provided, the polling thread starts at that rate
    and ramps to target_hz once motion is detected (>1 µm from initial sample).

    Args:
        bcv: BasicControlValue interface (or hysteresis-corrected variant).
        converter: MetricsConverter for native -> microns.
        target_hz: Target polling rate in Hz.
        startup_hz: If set, initial polling rate before motion detected.
        paused: If True, capture initial sample but don't start thread.

    Returns:
        PollingHandle with .samples, .stop, .thread, .start(), and .join().
    """
    handle = PollingHandle(_bcv=bcv, _converter=converter, _target_hz=target_hz, _startup_hz=startup_hz)

    # Capture initial position before thread starts
    t_before = time.perf_counter()
    native = bcv.GetControlValue()
    t_after = time.perf_counter()
    handle.samples.append(PositionSample(t_before, t_after, converter.GetMetricsValue(native)))

    if not paused:
        handle.start()
    return handle


def start_motion_polling(
    bcv,
    converter,
    *,
    target_hz: float = DEFAULT_POLL_HZ,
    startup_hz: float = STARTUP_POLL_HZ,
) -> PollingHandle:
    """Start polling for use during a move: 10 Hz ramp to 100 Hz on motion.

    Convenience wrapper around start_polling with adaptive Hz defaults.
    No paused mode — 10 Hz startup is gentle enough to coexist with
    move_to_async on the bus.
    """
    return start_polling(bcv, converter, target_hz=target_hz, startup_hz=startup_hz)
