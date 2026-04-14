"""Microscope connection manager with camera streaming and command dispatch.

All SDK interaction runs on a single background thread.  The GUI thread
communicates via Qt signals (updates) and a command queue (actions).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage


class HardwareState(NamedTuple):
    """Snapshot of camera/light settings for syncing controls on connect."""

    exposure_ms: float
    gain: float
    gain_rgb: tuple[float, float, float]
    shutter_open: bool
    lamp_intensity: int
    lamp_max: int


def numpy_rgb_to_qimage(arr: np.ndarray) -> QImage:
    """Convert RGB uint8 numpy array to QImage.  Returns an owned copy."""
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    # .copy() detaches from the numpy buffer so the QImage owns its data
    return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class ScopeManager(QObject):
    """Manages microscope connection on a background thread.

    Emits Qt signals for position, camera frames, and hardware state.
    Accepts commands via a thread-safe queue.
    """

    # Background thread → GUI thread
    position_updated = Signal(float, float, float, object, int, float, float)
    frame_ready = Signal(QImage)
    hw_state_ready = Signal(object)  # HardwareState
    scope_connected = Signal()
    scope_disconnected = Signal()
    scope_error = Signal(str)  # fatal — connection lost
    command_error = Signal(str)  # non-fatal — command failed, scope still alive
    autofocus_finished = Signal(float, float)  # best_z_um, best_sharpness
    scan_finished = Signal()
    scan_row_started = Signal(int, int)  # current_row (0-based), total_rows

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cmd_queue: queue.Queue[Callable] = queue.Queue()
        self._poll_interval = 0.05  # 50 ms → ~20 Hz
        self._stream = None  # set on background thread
        self._scope = None  # set on background thread
        self._desc = None  # MicroscopeDescription, set on background thread

    @property
    def is_connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Lifecycle ────────────────────────────────────────────────

    def open(self) -> None:
        """Start background thread and connect to microscope."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Signal the background thread to stop."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    # ── Command dispatch ─────────────────────────────────────────

    def send_command(self, cmd: Callable) -> None:
        """Queue a callable to run on the background thread.

        The callable receives the Microscope instance as its only argument.
        """
        self._cmd_queue.put(cmd)

    def move_to(self, x_um: float, y_um: float) -> None:
        def _move(scope):
            from flakefinder.leica.units import wait_all

            hx, hy = scope.stage.move_to_async(x_um, y_um)
            wait_all([hx, hy])

        self.send_command(_move)

    def switch_objective(self, mag: str) -> None:
        self.send_command(lambda scope: scope.switch_objective_mag(mag))

    def set_exposure_ms(self, ms: float) -> None:
        self.send_command(lambda scope: setattr(scope.camera, "exposure_time", ms / 1000.0))

    def set_gain(self, value: float) -> None:
        self.send_command(lambda scope: setattr(scope.camera, "gain", value))

    def set_gain_rgb(self, r: float, g: float, b: float) -> None:
        self.send_command(lambda scope: setattr(scope.camera, "gain_rgb", (r, g, b)))

    def set_shutter(self, is_open: bool) -> None:
        self.send_command(lambda scope: scope.shutter.open() if is_open else scope.shutter.close())

    def set_lamp_intensity(self, value: int) -> None:
        self.send_command(lambda scope: setattr(scope.lamp, "intensity", value))

    def autofocus(self) -> None:
        def _af(scope):
            from flakefinder.leica.autofocus import continuous_autofocus

            if self._stream is not None:
                self._stream.stop()
            try:
                result = continuous_autofocus(scope)
                self.autofocus_finished.emit(result.selected_z_um, result.selected_sharpness)
            except Exception as e:
                self.command_error.emit(f"Autofocus failed: {e}")
            finally:
                if self._stream is not None:
                    self._stream.start()
                self.hw_state_ready.emit(self._read_hw_state(scope))

        self.send_command(_af)

    def quick_scan(self, row_y_positions: list[float], x_min: float, x_max: float, speed_mm: float) -> None:
        """Sweep the stage in a snake pattern over the given rows.

        The FrameStream stays running — position and frame updates are
        emitted during the sweep so the canvas builds up the mosaic live.
        """

        def _scan(scope):
            total = len(row_y_positions)
            target_um_s = speed_mm * 1000
            for axis in (scope.stage.x, scope.stage.y):
                axis.set_velocity_um_s(min(target_um_s, axis.max_velocity_um_s))

            try:
                for i, y in enumerate(row_y_positions):
                    if self._stop.is_set():
                        break
                    self.scan_row_started.emit(i, total)

                    # Snake: even rows go +X, odd rows go -X
                    x_start = x_min if i % 2 == 0 else x_max
                    x_end = x_max if i % 2 == 0 else x_min

                    # Move to row start (poll while waiting)
                    handles = list(scope.stage.move_to_async(x_start, y))
                    while not all(h.is_complete for h in handles):
                        self._poll_and_emit(scope)
                    for h in handles:
                        h.dispose()

                    # Sweep the row (poll while waiting — this is where frames build up)
                    hx = scope.stage.x.move_to_async(x_end)
                    while not hx.is_complete:
                        self._poll_and_emit(scope)
                    hx.dispose()
            except Exception as e:
                self.command_error.emit(f"Scan failed: {e}")
            else:
                self.scan_finished.emit()

        self.send_command(_scan)

    # ── Background thread ────────────────────────────────────────

    @staticmethod
    def _read_hw_state(scope) -> HardwareState:
        r, g, b = scope.camera.gain_rgb
        return HardwareState(
            exposure_ms=scope.camera.exposure_time * 1000.0,
            gain=scope.camera.gain,
            gain_rgb=(r, g, b),
            shutter_open=scope.shutter.is_open,
            lamp_intensity=scope.lamp.intensity,
            lamp_max=scope.lamp.max_intensity,
        )

    def _poll_and_emit(self, scope) -> None:
        """Poll position + drain camera frames, emit signals. Sleeps one interval."""
        from flakefinder.data_utils import compute_frame_size_um

        x, y = scope.stage.position_um
        z = scope.z.position_um
        mag = scope.nosepiece.magnification
        obj_pos = scope.nosepiece.position

        fov_w, fov_h = 0.0, 0.0
        if mag is not None and self._desc is not None:
            size = compute_frame_size_um(self._desc.camera, mag, binning_idx=2)
            if size is not None:
                fov_w, fov_h = size

        self.position_updated.emit(x, y, z, mag, obj_pos, fov_w, fov_h)

        if self._stream is not None:
            latest = None
            for frame in self._stream.drain():
                latest = frame
            if latest is not None:
                qimg = numpy_rgb_to_qimage(latest.image)
                self.frame_ready.emit(qimg)

        self._stop.wait(self._poll_interval)

    def _run(self) -> None:
        """Main loop for the background thread."""
        try:
            from flakefinder.data_utils import require_microscope_description
            from flakefinder.leica.microscope import Microscope

            self._desc = require_microscope_description()
            scope = Microscope()
            scope.__enter__()
            try:
                self.scope_connected.emit()
                self.hw_state_ready.emit(self._read_hw_state(scope))

                # Start camera streaming
                self._stream = scope.camera.stream()
                self._stream.start()
                try:
                    while not self._stop.is_set():
                        # --- Commands ---
                        while not self._cmd_queue.empty():
                            try:
                                cmd = self._cmd_queue.get_nowait()
                                cmd(scope)
                            except Exception as e:
                                self.command_error.emit(f"Command error: {e}")

                        self._poll_and_emit(scope)
                finally:
                    self._stream.stop()
                    self._stream = None
            finally:
                scope.__exit__(None, None, None)
        except Exception as e:
            self.scope_error.emit(str(e))
        finally:
            self._desc = None
            self.scope_disconnected.emit()
