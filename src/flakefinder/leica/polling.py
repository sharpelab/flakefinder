"""Rate-limited position polling for scan threads.

Provides a reusable poll-sleep loop that keeps position sampling at a
target Hz, preventing SDK bus starvation when using the fast cmusb (D2XX)
transport (~4.8 ms/call, ~206 Hz unthrottled).  At the older 16 ms VCP
transport the call itself exceeds the target period, so no sleep occurs
and the natural ~63 Hz rate is preserved.

See docs/sdk_threading_investigation.md for background.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..types import PositionSample
from .types import BasicControlValue, MetricsConverter

DEFAULT_POLL_HZ: float = 100.0
STARTUP_POLL_HZ: float = 10.0
_MOTION_THRESHOLD_UM: float = 0.1


@dataclass
class PollingHandle:
    """Handle returned by start_polling(). Owns the samples, stop event, and thread."""

    samples: list[PositionSample] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    # Private — stored for deferred thread creation (paused=True)
    _bcv: BasicControlValue | None = field(default=None, repr=False)
    _converter: MetricsConverter | None = field(default=None, repr=False)
    _target_hz: float = field(default=DEFAULT_POLL_HZ, repr=False)
    _startup_hz: float | None = field(default=None, repr=False)
    _motion_threshold_um: float = field(default=_MOTION_THRESHOLD_UM, repr=False)

    def start(self) -> None:
        """Start the polling thread (for use after paused=True)."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=poll_position,
            args=(self._bcv, self._converter, self.samples, self.stop),
            kwargs={
                "target_hz": self._target_hz,
                "startup_hz": self._startup_hz,
                "motion_threshold_um": self._motion_threshold_um,
            },
            daemon=True,
        )
        self.thread.start()

    def join(self, timeout: float = 1.0) -> None:
        """Signal stop and join the polling thread."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)


def poll_position(
    bcv: BasicControlValue,
    converter: MetricsConverter,
    samples: list[PositionSample],
    stop: threading.Event,
    *,
    target_hz: float = DEFAULT_POLL_HZ,
    startup_hz: float | None = None,
    motion_threshold_um: float = _MOTION_THRESHOLD_UM,
) -> None:
    """Thread-target: poll position, convert, append PositionSample, sleep to target rate.

    When startup_hz is provided, starts at that rate and ramps to target_hz
    once motion is detected (position moves more than motion_threshold_um
    from initial position).

    Args:
        bcv: BasicControlValue interface (or hysteresis-corrected variant).
        converter: MetricsConverter for native -> microns.
        samples: Shared list to append PositionSample results.
        stop: Event to signal thread shutdown.
        target_hz: Target polling rate in Hz.
        startup_hz: If set, initial polling rate before motion detected.
        motion_threshold_um: Distance from initial position to trigger ramp.
    """
    if startup_hz is not None:
        period = 1.0 / startup_hz
        # start_polling always seeds samples[0] before spawning this thread
        initial_pos = samples[0].axis_um
        ramped = False
    else:
        period = 1.0 / target_hz
        initial_pos = 0.0  # unused — ramped is already True
        ramped = True

    while not stop.is_set():
        t_before = time.perf_counter()
        native = bcv.GetControlValue()
        t_after = time.perf_counter()
        um = converter.GetMetricsValue(native)
        samples.append(PositionSample(t_before, t_after, um))

        if not ramped and abs(um - initial_pos) > motion_threshold_um:
            period = 1.0 / target_hz
            ramped = True

        elapsed = t_after - t_before
        if elapsed < period:
            time.sleep(period - elapsed)


def start_polling(
    bcv: BasicControlValue,
    converter: MetricsConverter,
    *,
    target_hz: float = DEFAULT_POLL_HZ,
    startup_hz: float | None = None,
    motion_threshold_um: float = _MOTION_THRESHOLD_UM,
    paused: bool = False,
) -> PollingHandle:
    """Start a position polling thread.

    Always captures one initial position sample before returning.
    With paused=True, the thread is not started — call handle.start()
    after issuing any bus commands (e.g. move_to_async) that would
    compete with polling on the same axis.

    When startup_hz is provided, the polling thread starts at that rate
    and ramps to target_hz once motion is detected.

    Args:
        bcv: BasicControlValue interface (or hysteresis-corrected variant).
        converter: MetricsConverter for native -> microns.
        target_hz: Target polling rate in Hz.
        startup_hz: If set, initial polling rate before motion detected.
        motion_threshold_um: Distance from initial position to trigger ramp.
        paused: If True, capture initial sample but don't start thread.

    Returns:
        PollingHandle with .samples, .stop, .thread, .start(), and .join().
    """
    handle = PollingHandle(
        _bcv=bcv,
        _converter=converter,
        _target_hz=target_hz,
        _startup_hz=startup_hz,
        _motion_threshold_um=motion_threshold_um,
    )

    # Capture initial position before thread starts
    t_before = time.perf_counter()
    native = bcv.GetControlValue()
    t_after = time.perf_counter()
    handle.samples.append(PositionSample(t_before, t_after, converter.GetMetricsValue(native)))

    if not paused:
        handle.start()
    return handle


def start_motion_polling(
    bcv: BasicControlValue,
    converter: MetricsConverter,
    *,
    target_hz: float = DEFAULT_POLL_HZ,
    startup_hz: float = STARTUP_POLL_HZ,
    motion_threshold_um: float = _MOTION_THRESHOLD_UM,
) -> PollingHandle:
    """Start polling for use during a move: slow ramp to target Hz on motion.

    Convenience wrapper around start_polling with adaptive Hz defaults.
    No paused mode — low startup Hz is gentle enough to coexist with
    move_to_async on the bus.
    """
    return start_polling(
        bcv,
        converter,
        target_hz=target_hz,
        startup_hz=startup_hz,
        motion_threshold_um=motion_threshold_um,
    )
