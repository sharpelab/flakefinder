"""Hardware unit abstractions with sync and async control.

This module provides high-level classes for controlling microscope components
with both blocking (sync) and non-blocking (async) operations.
"""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING, Callable

from .enums import TID, IID, EMetricsId, MoveState
from .core import get_interface, get_interface_required, find_unit

if TYPE_CHECKING:
    from .types import (
        Unit,
        BasicControlValue,
        BasicControlValueAsync,
        HaltControlValue,
        BasicControlState,
        BasicControlValueVelocity,
        AutoCalibration,
        MetricsConverter,
        AsyncResult,
    )


class MoveHandle:
    """Handle to track and control an async move operation.

    Usage:
        handle = axis.move_to_async(1000.0)

        # Option 1: Block until complete
        handle.wait()

        # Option 2: Poll
        while handle.state == MoveState.IN_PROGRESS:
            print(f"Position: {axis.position_um}")
            time.sleep(0.05)

        # Option 3: Cancel
        handle.cancel()
    """

    # SDK AsyncResult.EState values (from ahwbasic.h)
    # enum eState { INPROGRESS, COMPLETED, STOPPED, OVERRIDDEN }
    _SDK_INPROGRESS = 0
    _SDK_COMPLETED = 1
    _SDK_STOPPED = 2
    _SDK_OVERRIDDEN = 3

    def __init__(
        self,
        async_result: "AsyncResult",
        halt_interface: "HaltControlValue | None" = None,
    ):
        """Initialize move handle.

        Args:
            async_result: SDK AsyncResult from SetControlValueAsync.
            halt_interface: Optional halt interface for cancellation.
        """
        self._result = async_result
        self._halt = halt_interface
        self._disposed = False

    @property
    def state(self) -> MoveState:
        """Get current move state."""
        if self._disposed:
            return MoveState.COMPLETE

        sdk_state = self._result.GetState()
        if sdk_state == self._SDK_INPROGRESS:
            return MoveState.IN_PROGRESS
        elif sdk_state == self._SDK_COMPLETED:
            return MoveState.COMPLETE
        elif sdk_state == self._SDK_STOPPED:
            return MoveState.CANCELLED
        elif sdk_state == self._SDK_OVERRIDDEN:
            return MoveState.CANCELLED  # Treat overridden as cancelled
        else:
            return MoveState.ERROR

    @property
    def state_raw(self) -> int:
        """Get raw SDK state value (for debugging)."""
        if self._disposed:
            return -1
        return self._result.GetState()

    @property
    def is_complete(self) -> bool:
        """Check if move is complete (success or cancelled)."""
        return self.state in (MoveState.COMPLETE, MoveState.CANCELLED, MoveState.ERROR)

    def wait(self, timeout: float | None = None, poll_interval: float = 0.01) -> bool:
        """Block until move completes.

        Args:
            timeout: Maximum seconds to wait (None = forever).
            poll_interval: Seconds between state checks.

        Returns:
            True if move completed, False if timed out.
        """
        start = time.monotonic()
        while not self.is_complete:
            if timeout is not None and (time.monotonic() - start) > timeout:
                return False
            time.sleep(poll_interval)
        return True

    def cancel(self) -> None:
        """Cancel the move by halting the axis."""
        if self._halt is not None and not self.is_complete:
            self._halt.Halt()

    def dispose(self) -> None:
        """Release SDK resources. Called automatically when move completes."""
        if not self._disposed:
            try:
                self._result.Dispose()
            except Exception:
                pass
            self._disposed = True

    def __del__(self):
        self.dispose()


class Axis:
    """Single-axis motion control (X, Y, or Z).

    Provides both synchronous (blocking) and asynchronous (non-blocking)
    movement methods, plus velocity control and calibration.

    Usage:
        # Sync move (blocks until complete)
        axis.move_to(1000.0)

        # Async move (returns immediately)
        handle = axis.move_to_async(2000.0)
        # ... do other work ...
        handle.wait()

        # Read position
        print(f"Position: {axis.position_um} µm")

        # Emergency stop
        axis.halt()
    """

    def __init__(self, unit: "Unit"):
        """Initialize axis from SDK unit.

        Args:
            unit: SDK Unit object (must support BasicControlValue).

        Raises:
            LookupError: If required interfaces not found.
        """
        self._unit = unit
        self._name = unit.GetName()

        # Required interfaces
        self._bcv: "BasicControlValue" = get_interface_required(
            unit, IID.IID_BASIC_CONTROL_VALUE
        )

        # Get metrics converter for µm
        converters = self._bcv.GetMetricsConverters()
        self._converter: "MetricsConverter" = converters.FindMetricsConverter(
            int(EMetricsId.METRICS_MICRONS)
        )
        if self._converter is None:
            raise LookupError(f"No microns converter for {self._name}")

        # Optional interfaces
        self._bcv_async: "BasicControlValueAsync | None" = get_interface(
            unit, IID.IID_BASIC_CONTROL_VALUE_ASYNC
        )
        self._halt: "HaltControlValue | None" = get_interface(
            unit, IID.IID_HALT_CONTROL_VALUE
        )
        self._state: "BasicControlState | None" = get_interface(
            unit, IID.IID_BASIC_CONTROL_STATE
        )
        self._velocity: "BasicControlValueVelocity | None" = get_interface(
            unit, IID.IID_BASIC_CONTROL_VALUE_VELOCITY
        )
        self._calibration: "AutoCalibration | None" = get_interface(
            unit, IID.IID_AUTO_CALIBRATION
        )

        # Cache limits
        self._min_native = self._bcv.MinControlValue()
        self._max_native = self._bcv.MaxControlValue()

    @property
    def name(self) -> str:
        """Axis name from SDK."""
        return self._name

    @property
    def unit(self) -> "Unit":
        """Underlying SDK unit."""
        return self._unit

    # --- Position ---

    @property
    def position_native(self) -> int:
        """Current position in native units (steps)."""
        return self._bcv.GetControlValue()

    @property
    def position_um(self) -> float:
        """Current position in microns."""
        return self._converter.GetMetricsValue(self.position_native)

    @property
    def min_um(self) -> float:
        """Minimum position in microns."""
        return self._converter.GetMetricsValue(self._min_native)

    @property
    def max_um(self) -> float:
        """Maximum position in microns."""
        return self._converter.GetMetricsValue(self._max_native)

    def _um_to_native(self, um: float) -> int:
        """Convert microns to native units."""
        return self._converter.GetControlValue(um)

    # --- Sync Movement ---

    def move_to(self, position_um: float) -> None:
        """Move to absolute position (blocking).

        Args:
            position_um: Target position in microns.
        """
        native = self._um_to_native(position_um)
        self._bcv.SetControlValue(native)

    def move_rel(self, delta_um: float) -> None:
        """Move relative to current position (blocking).

        Args:
            delta_um: Distance to move in microns (positive or negative).
        """
        current = self.position_native
        delta_native = self._um_to_native(delta_um) - self._um_to_native(0)
        self._bcv.SetControlValue(current + delta_native)

    # --- Async Movement ---

    @property
    def supports_async(self) -> bool:
        """Check if axis supports async moves."""
        return self._bcv_async is not None

    def move_to_async(self, position_um: float) -> MoveHandle:
        """Move to absolute position (non-blocking).

        Args:
            position_um: Target position in microns.

        Returns:
            MoveHandle to track/cancel the move.

        Raises:
            RuntimeError: If axis doesn't support async moves.
        """
        if self._bcv_async is None:
            raise RuntimeError(f"Axis {self._name} doesn't support async moves")

        native = self._um_to_native(position_um)
        result = self._bcv_async.SetControlValueAsync(native)
        return MoveHandle(result, self._halt)

    def move_rel_async(self, delta_um: float) -> MoveHandle:
        """Move relative to current position (non-blocking).

        Args:
            delta_um: Distance to move in microns.

        Returns:
            MoveHandle to track/cancel the move.

        Raises:
            RuntimeError: If axis doesn't support async moves.
        """
        if self._bcv_async is None:
            raise RuntimeError(f"Axis {self._name} doesn't support async moves")

        current = self.position_native
        delta_native = self._um_to_native(delta_um) - self._um_to_native(0)
        target = current + delta_native
        result = self._bcv_async.SetControlValueAsync(target)
        return MoveHandle(result, self._halt)

    # --- State ---

    @property
    def is_moving(self) -> bool:
        """Check if axis is currently moving."""
        if self._state is not None:
            return self._state.IsChanging()
        return False

    @property
    def supports_halt(self) -> bool:
        """Check if axis supports halt."""
        return self._halt is not None

    def halt(self) -> None:
        """Emergency stop - halt axis movement immediately.

        Raises:
            RuntimeError: If axis doesn't support halt.
        """
        if self._halt is None:
            raise RuntimeError(f"Axis {self._name} doesn't support halt")
        self._halt.Halt()

    # --- Velocity ---

    @property
    def supports_velocity(self) -> bool:
        """Check if axis supports velocity control."""
        return self._velocity is not None

    @property
    def velocity_native(self) -> int | None:
        """Current velocity in native units, or None if not supported."""
        if self._velocity is None:
            return None
        return self._velocity.GetControlValue()

    def set_velocity_native(self, velocity: int) -> None:
        """Set velocity in native units.

        Args:
            velocity: Velocity in native units.

        Raises:
            RuntimeError: If axis doesn't support velocity control.
        """
        if self._velocity is None:
            raise RuntimeError(f"Axis {self._name} doesn't support velocity control")
        self._velocity.SetControlValue(velocity)

    @property
    def min_velocity_native(self) -> int | None:
        """Minimum velocity in native units."""
        if self._velocity is None:
            return None
        return self._velocity.MinControlValue()

    @property
    def max_velocity_native(self) -> int | None:
        """Maximum velocity in native units."""
        if self._velocity is None:
            return None
        return self._velocity.MaxControlValue()

    # --- Calibration ---

    @property
    def supports_calibration(self) -> bool:
        """Check if axis supports calibration."""
        return self._calibration is not None

    @property
    def is_calibrated(self) -> bool:
        """Check if axis is calibrated."""
        if self._calibration is None:
            return True  # Assume calibrated if no interface
        return self._calibration.IsCalibrated()

    def calibrate(self) -> None:
        """Run axis calibration (homing).

        Raises:
            RuntimeError: If axis doesn't support calibration.
        """
        if self._calibration is None:
            raise RuntimeError(f"Axis {self._name} doesn't support calibration")
        self._calibration.Calibrate()

    def __repr__(self) -> str:
        return f"Axis({self._name}, pos={self.position_um:.1f}µm)"


class Shutter:
    """Shutter control (open/close).

    Usage:
        shutter = Shutter.from_connection(conn)
        shutter.open()
        shutter.close()
    """

    def __init__(self, unit: "Unit"):
        """Initialize shutter from SDK unit.

        Args:
            unit: SDK Unit object (must support BasicControlValue).

        Raises:
            LookupError: If required interface not found.
        """
        self._unit = unit
        self._name = unit.GetName()
        self._bcv: "BasicControlValue" = get_interface_required(
            unit, IID.IID_BASIC_CONTROL_VALUE
        )

    @classmethod
    def from_connection(cls, conn: "LeicaConnection", tid: TID = TID.MICROSCOPE_IL_SHUTTER) -> "Shutter":
        """Create Shutter from a LeicaConnection.

        Args:
            conn: Active LeicaConnection.
            tid: Shutter type ID (default: IL shutter).

        Returns:
            Shutter instance.

        Raises:
            LookupError: If shutter unit not found.
        """
        shutter_unit = find_unit(conn.root, tid)
        if shutter_unit is None:
            raise LookupError(f"Shutter unit not found: {tid.name}")
        return cls(shutter_unit)

    @property
    def name(self) -> str:
        """Shutter name from SDK."""
        return self._name

    @property
    def is_open(self) -> bool:
        """Check if shutter is open."""
        return self._bcv.GetControlValue() == 1

    def open(self) -> None:
        """Open the shutter."""
        self._bcv.SetControlValue(1)

    def close(self) -> None:
        """Close the shutter."""
        self._bcv.SetControlValue(0)

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"Shutter({self._name}, {state})"


class Lamp:
    """Lamp intensity control.

    Usage:
        lamp = Lamp.from_connection(conn)
        lamp.intensity = lamp.max_intensity  # Full brightness
        lamp.intensity = 0  # Off
    """

    def __init__(self, unit: "Unit"):
        """Initialize lamp from SDK unit.

        Args:
            unit: SDK Unit object (must support BasicControlValue).

        Raises:
            LookupError: If required interface not found.
        """
        self._unit = unit
        self._name = unit.GetName()
        self._bcv: "BasicControlValue" = get_interface_required(
            unit, IID.IID_BASIC_CONTROL_VALUE
        )
        self._min = self._bcv.MinControlValue()
        self._max = self._bcv.MaxControlValue()

    @classmethod
    def from_connection(cls, conn: "LeicaConnection") -> "Lamp":
        """Create Lamp from a LeicaConnection.

        Args:
            conn: Active LeicaConnection.

        Returns:
            Lamp instance.

        Raises:
            LookupError: If lamp unit not found.
        """
        lamp_unit = find_unit(conn.root, TID.MICROSCOPE_LAMP)
        if lamp_unit is None:
            raise LookupError("Lamp unit not found")
        return cls(lamp_unit)

    @property
    def name(self) -> str:
        """Lamp name from SDK."""
        return self._name

    @property
    def intensity(self) -> int:
        """Current lamp intensity."""
        return self._bcv.GetControlValue()

    @intensity.setter
    def intensity(self, value: int) -> None:
        self._bcv.SetControlValue(value)

    @property
    def min_intensity(self) -> int:
        """Minimum intensity value."""
        return self._min

    @property
    def max_intensity(self) -> int:
        """Maximum intensity value."""
        return self._max

    def off(self) -> None:
        """Turn lamp off."""
        self.intensity = self._min

    def full(self) -> None:
        """Set lamp to full brightness."""
        self.intensity = self._max

    def __repr__(self) -> str:
        return f"Lamp({self._name}, intensity={self.intensity}/{self._max})"


class Stage:
    """XY stage control combining X and Y axes.

    Provides coordinated XY movement and convenience methods.

    Usage:
        stage = Stage.from_connection(conn)

        # Move X and Y together (async)
        hx, hy = stage.move_to_async(1000, 2000)
        stage.wait_all([hx, hy])

        # Read position
        x, y = stage.position_um
    """

    def __init__(self, x_axis: Axis, y_axis: Axis):
        """Initialize stage from X and Y axis objects.

        Args:
            x_axis: X axis controller.
            y_axis: Y axis controller.
        """
        self.x = x_axis
        self.y = y_axis

    @classmethod
    def from_connection(cls, conn: "LeicaConnection") -> "Stage":
        """Create Stage from a LeicaConnection.

        Args:
            conn: Active LeicaConnection.

        Returns:
            Stage instance.

        Raises:
            LookupError: If X or Y unit not found.
        """
        from .core import LeicaConnection

        x_unit = conn.find_unit(TID.MICROSCOPE_X_UNIT)
        y_unit = conn.find_unit(TID.MICROSCOPE_Y_UNIT)

        if x_unit is None:
            raise LookupError("X axis unit not found")
        if y_unit is None:
            raise LookupError("Y axis unit not found")

        return cls(Axis(x_unit), Axis(y_unit))

    @property
    def position_um(self) -> tuple[float, float]:
        """Current (X, Y) position in microns."""
        return (self.x.position_um, self.y.position_um)

    @property
    def position_native(self) -> tuple[int, int]:
        """Current (X, Y) position in native units."""
        return (self.x.position_native, self.y.position_native)

    def move_to(self, x_um: float, y_um: float) -> None:
        """Move to absolute XY position (blocking).

        Note: Moves X then Y sequentially. For parallel moves, use move_to_async.

        Args:
            x_um: Target X position in microns.
            y_um: Target Y position in microns.
        """
        self.x.move_to(x_um)
        self.y.move_to(y_um)

    def move_to_async(
        self, x_um: float, y_um: float
    ) -> tuple[MoveHandle, MoveHandle]:
        """Move to absolute XY position (non-blocking, parallel).

        Args:
            x_um: Target X position in microns.
            y_um: Target Y position in microns.

        Returns:
            Tuple of (x_handle, y_handle) to track moves.
        """
        hx = self.x.move_to_async(x_um)
        hy = self.y.move_to_async(y_um)
        return (hx, hy)

    def move_rel(self, dx_um: float, dy_um: float) -> None:
        """Move relative to current position (blocking).

        Args:
            dx_um: X distance in microns.
            dy_um: Y distance in microns.
        """
        self.x.move_rel(dx_um)
        self.y.move_rel(dy_um)

    def move_rel_async(
        self, dx_um: float, dy_um: float
    ) -> tuple[MoveHandle, MoveHandle]:
        """Move relative to current position (non-blocking, parallel).

        Args:
            dx_um: X distance in microns.
            dy_um: Y distance in microns.

        Returns:
            Tuple of (x_handle, y_handle) to track moves.
        """
        hx = self.x.move_rel_async(dx_um)
        hy = self.y.move_rel_async(dy_um)
        return (hx, hy)

    def halt(self) -> None:
        """Emergency stop both axes."""
        if self.x.supports_halt:
            self.x.halt()
        if self.y.supports_halt:
            self.y.halt()

    @staticmethod
    def wait_all(
        handles: list[MoveHandle],
        timeout: float | None = None,
        poll_interval: float = 0.01,
    ) -> bool:
        """Wait for multiple moves to complete.

        Args:
            handles: List of MoveHandles to wait for.
            timeout: Maximum seconds to wait (None = forever).
            poll_interval: Seconds between state checks.

        Returns:
            True if all completed, False if timed out.
        """
        start = time.monotonic()
        while not all(h.is_complete for h in handles):
            if timeout is not None and (time.monotonic() - start) > timeout:
                return False
            time.sleep(poll_interval)
        return True

    def __repr__(self) -> str:
        x, y = self.position_um
        return f"Stage(x={x:.1f}µm, y={y:.1f}µm)"
