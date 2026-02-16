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

    def start(self) -> None:
        """Start the polling thread (for use after paused=True)."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=poll_position,
            args=(self._bcv, self._converter, self.samples, self.stop),
            kwargs={"target_hz": self._target_hz},
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
) -> None:
    """Thread-target: poll position, convert, append PositionSample, sleep to target rate.

    Args:
        bcv: BasicControlValue interface (or hysteresis-corrected variant).
        converter: MetricsConverter for native -> microns.
        samples: Shared list to append PositionSample results.
        stop: Event to signal thread shutdown.
        target_hz: Target polling rate in Hz.
    """
    period = 1.0 / target_hz
    while not stop.is_set():
        t_before = time.perf_counter()
        native = bcv.GetControlValue()
        t_after = time.perf_counter()
        um = converter.GetMetricsValue(native)
        samples.append(PositionSample(t_before, t_after, um))
        elapsed = t_after - t_before
        if elapsed < period:
            time.sleep(period - elapsed)


def start_polling(
    bcv,
    converter,
    *,
    target_hz: float = DEFAULT_POLL_HZ,
    paused: bool = False,
) -> PollingHandle:
    """Start a position polling thread.

    Always captures one initial position sample before returning.
    With paused=True, the thread is not started — call handle.start()
    after issuing any bus commands (e.g. move_to_async) that would
    compete with polling on the same axis.

    Args:
        bcv: BasicControlValue interface (or hysteresis-corrected variant).
        converter: MetricsConverter for native -> microns.
        target_hz: Target polling rate in Hz.
        paused: If True, capture initial sample but don't start thread.

    Returns:
        PollingHandle with .samples, .stop, .thread, .start(), and .join().
    """
    handle = PollingHandle(_bcv=bcv, _converter=converter, _target_hz=target_hz)

    # Capture initial position before thread starts
    t_before = time.perf_counter()
    native = bcv.GetControlValue()
    t_after = time.perf_counter()
    handle.samples.append(PositionSample(t_before, t_after, converter.GetMetricsValue(native)))

    if not paused:
        handle.start()
    return handle
