"""Hardware unit abstractions with sync and async control.

This module provides high-level classes for controlling microscope components
with both blocking (sync) and non-blocking (async) operations.
"""

import contextlib
import re
import time
import warnings

from flakefinder.types import Point2F

from .core import find_unit, get_interface, get_interface_required
from .enums import IID, TID, EMetricsId, MoveState
from .types import (
    AsyncResult,
    AutoCalibration,
    BasicControlState,
    BasicControlValue,
    BasicControlValueAsync,
    BasicControlValueVelocity,
    HaltControlValue,
    MetricsConverter,
    Unit,
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
            with contextlib.suppress(Exception):
                self._result.Dispose()
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
        self._bcv: BasicControlValue = get_interface_required(unit, IID.IID_BASIC_CONTROL_VALUE)

        # Get metrics converter for µm
        converters = self._bcv.GetMetricsConverters()
        self._converter: MetricsConverter = converters.FindMetricsConverter(int(EMetricsId.METRICS_MICRONS))
        if self._converter is None:
            raise LookupError(f"No microns converter for {self._name}")

        # Optional interfaces
        self._bcv_async: BasicControlValueAsync | None = get_interface(unit, IID.IID_BASIC_CONTROL_VALUE_ASYNC)
        self._halt: HaltControlValue | None = get_interface(unit, IID.IID_HALT_CONTROL_VALUE)
        self._state: BasicControlState | None = get_interface(unit, IID.IID_BASIC_CONTROL_STATE)
        self._velocity: BasicControlValueVelocity | None = get_interface(unit, IID.IID_BASIC_CONTROL_VALUE_VELOCITY)
        self._velocity_converter: MetricsConverter | None = None
        if self._velocity is not None:
            try:
                vel_converters = self._velocity.GetMetricsConverters()
                self._velocity_converter = vel_converters.FindMetricsConverter(
                    int(EMetricsId.METRICS_MICRONS_PER_SECOND)
                )
            except Exception:
                pass  # No velocity converter available

        self._calibration: AutoCalibration | None = get_interface(unit, IID.IID_AUTO_CALIBRATION)
        self._directed_velocity = get_interface(unit, IID.IID_DIRECTED_CONTROL_VALUE_ASYNC_VELOCITY)
        # Use regular velocity converter for directed velocity (same native units)

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

    def read_position_timed(self) -> tuple[float, float, float]:
        """Fast position read with high-precision timestamps.

        Useful for position interpolation during scanning. The timestamps
        bracket the actual SDK call, allowing sub-millisecond timing accuracy.

        Returns:
            (t_before, t_after, position_um) tuple where times are from
            time.perf_counter().
        """
        t_before = time.perf_counter()
        native = self._bcv.GetControlValue()
        t_after = time.perf_counter()
        return (t_before, t_after, self._converter.GetMetricsValue(native))

    @property
    def bcv(self) -> "BasicControlValue":
        """Direct BasicControlValue interface for fast polling.

        Use this for tight polling loops where you need maximum performance.
        Call bcv.GetControlValue() directly and convert with self.converter.
        """
        return self._bcv

    @property
    def converter(self) -> "MetricsConverter":
        """Microns converter for manual position conversion.

        Use with bcv for fast polling:
            native = axis.bcv.GetControlValue()
            um = axis.converter.GetMetricsValue(native)
        """
        return self._converter

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

    @property
    def velocity_um_s(self) -> float | None:
        """Current velocity in µm/s, or None if not supported."""
        if self._velocity is None or self._velocity_converter is None:
            return None
        native = self._velocity.GetControlValue()
        return self._velocity_converter.GetMetricsValue(native)

    @property
    def max_velocity_um_s(self) -> float | None:
        """Maximum velocity in µm/s, or None if not supported."""
        if self._velocity is None or self._velocity_converter is None:
            return None
        native = self._velocity.MaxControlValue()
        return self._velocity_converter.GetMetricsValue(native)

    @property
    def min_velocity_um_s(self) -> float | None:
        """Minimum velocity in µm/s, or None if not supported."""
        if self._velocity is None or self._velocity_converter is None:
            return None
        native = self._velocity.MinControlValue()
        return self._velocity_converter.GetMetricsValue(native)

    def set_velocity_um_s(self, velocity_um_s: float) -> None:
        """Set velocity in µm/s.

        Args:
            velocity_um_s: Velocity in microns per second.

        Raises:
            RuntimeError: If axis doesn't support velocity control or conversion.
        """
        if self._velocity is None:
            raise RuntimeError(f"Axis {self._name} doesn't support velocity control")
        if self._velocity_converter is None:
            raise RuntimeError(f"Axis {self._name} doesn't have velocity converter")
        native = self._velocity_converter.GetControlValue(velocity_um_s)
        self._velocity.SetControlValue(native)

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

    # --- Directed Velocity Movement ---

    @property
    def supports_directed_velocity(self) -> bool:
        """Check if axis supports directed velocity movement."""
        return self._directed_velocity is not None

    def start_towards_max(self, velocity_um_s: float) -> None:
        """Start moving towards max position at specified velocity.

        Motion continues until halt() is called or limit is reached.

        Args:
            velocity_um_s: Velocity in microns per second.

        Raises:
            RuntimeError: If axis doesn't support directed velocity.
        """
        if self._directed_velocity is None:
            raise RuntimeError(f"Axis {self._name} doesn't support directed velocity")
        if self._velocity_converter is None:
            raise RuntimeError(f"Axis {self._name} doesn't have velocity converter")
        native_vel = self._velocity_converter.GetControlValue(velocity_um_s)
        self._directed_velocity.StartTowardsMaxVelocity(native_vel)

    def start_towards_min(self, velocity_um_s: float) -> None:
        """Start moving towards min position at specified velocity.

        Motion continues until halt() is called or limit is reached.

        Args:
            velocity_um_s: Velocity in microns per second.

        Raises:
            RuntimeError: If axis doesn't support directed velocity.
        """
        if self._directed_velocity is None:
            raise RuntimeError(f"Axis {self._name} doesn't support directed velocity")
        if self._velocity_converter is None:
            raise RuntimeError(f"Axis {self._name} doesn't have velocity converter")
        native_vel = self._velocity_converter.GetControlValue(velocity_um_s)
        self._directed_velocity.StartTowardsMinVelocity(native_vel)

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
        self._bcv: BasicControlValue = get_interface_required(unit, IID.IID_BASIC_CONTROL_VALUE)

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
        self._bcv: BasicControlValue = get_interface_required(unit, IID.IID_BASIC_CONTROL_VALUE)
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

    @property
    def intensity_pct(self) -> float:
        """Current intensity as percentage (0-100)."""
        range_ = self._max - self._min
        if range_ == 0:
            return 0.0
        return (self.intensity - self._min) / range_ * 100

    @intensity_pct.setter
    def intensity_pct(self, value: float) -> None:
        """Set intensity as percentage (0-100), clamped to valid range."""
        value = max(0.0, min(100.0, value))
        range_ = self._max - self._min
        native = self._min + round(value / 100 * range_)
        self.intensity = native

    def off(self) -> None:
        """Turn lamp off."""
        self.intensity = self._min

    def full(self) -> None:
        """Set lamp to full brightness."""
        self.intensity = self._max

    def __repr__(self) -> str:
        return f"Lamp({self._name}, intensity={self.intensity}/{self._max})"


class Nosepiece:
    """Objective turret (nosepiece) control.

    Usage:
        nosepiece = Nosepiece.from_connection(conn)
        print(f"Objective: {nosepiece.magnification}x")
        nosepiece.position = 3  # Switch to position 3
    """

    # Default magnification lookup for Sharpe Lab DM6M configuration.
    # Position 1-indexed. Override by setting magnifications property.
    DEFAULT_MAGNIFICATIONS: dict[int, float] = {
        1: 5,
        2: 10,
        3: 20,
        4: 50,
        5: 150,
        6: 2.5,
    }

    def __init__(self, unit: "Unit", magnifications: dict[int, float] | None = None):
        """Initialize nosepiece from SDK unit.

        Args:
            unit: SDK Unit object (must support BasicControlValue).
            magnifications: Optional dict mapping position (1-indexed) to
                magnification. If None, uses DEFAULT_MAGNIFICATIONS.

        Raises:
            LookupError: If required interface not found.
        """
        self._unit = unit
        self._name = unit.GetName()
        self._bcv: BasicControlValue = get_interface_required(unit, IID.IID_BASIC_CONTROL_VALUE)
        self._magnifications = magnifications or self.DEFAULT_MAGNIFICATIONS.copy()

    @classmethod
    def from_connection(
        cls,
        conn: "LeicaConnection",
        magnifications: dict[int, float] | None = None,
    ) -> "Nosepiece":
        """Create Nosepiece from a LeicaConnection.

        Args:
            conn: Active LeicaConnection.
            magnifications: Optional magnification lookup table.

        Returns:
            Nosepiece instance.

        Raises:
            LookupError: If nosepiece unit not found.
        """
        nosepiece_unit = find_unit(conn.root, TID.MICROSCOPE_NOSEPIECE)
        if nosepiece_unit is None:
            raise LookupError("Nosepiece unit not found")
        return cls(nosepiece_unit, magnifications)

    @property
    def name(self) -> str:
        """Nosepiece name from SDK."""
        return self._name

    @property
    def position(self) -> int:
        """Current objective position (1-indexed)."""
        return self._bcv.GetControlValue()

    @position.setter
    def position(self, value: int) -> None:
        """Set objective position (1-indexed)."""
        self._bcv.SetControlValue(value)

    def set_position(self, value: int, z: "ZDrive") -> None:
        """Set objective position, temporarily maxing Z speed to avoid SDK timeout.

        .. deprecated::
            Use ``Microscope.switch_objective_pos()`` instead, which owns
            both the nosepiece and ZDrive internally.

        Args:
            value: Target position (1-indexed).
            z: ZDrive instance for temporary velocity override.
        """
        warnings.warn(
            "Nosepiece.set_position() is deprecated, use Microscope.switch_objective_pos()",
            DeprecationWarning,
            stacklevel=2,
        )
        saved = z.velocity_native
        z.set_velocity_native(z.max_velocity_native)
        try:
            self._bcv.SetControlValue(value)
        finally:
            z.set_velocity_native(saved)

    @property
    def magnification(self) -> float | None:
        """Current objective magnification from lookup table.

        Returns:
            Magnification value (e.g., 5, 10, 20), or None if position
            not in magnifications table.
        """
        return self._magnifications.get(self.position)

    @property
    def magnifications(self) -> dict[int, float]:
        """Position-to-magnification lookup table."""
        return self._magnifications

    @magnifications.setter
    def magnifications(self, value: dict[int, float]) -> None:
        """Set the magnification lookup table."""
        self._magnifications = value

    @property
    def min_position(self) -> int:
        """Minimum objective position."""
        return self._bcv.MinControlValue()

    @property
    def max_position(self) -> int:
        """Maximum objective position."""
        return self._bcv.MaxControlValue()

    def parse_magnification(self, value: str) -> int:
        """Parse magnification string (e.g. '5', '5x', '20X', '2.5') to position number.

        Returns:
            Position number (1-indexed).

        Raises:
            ValueError: If value cannot be parsed or doesn't match known objectives.
        """
        match = re.match(r"^(\d+(?:\.\d+)?)[xX]?$", value.strip())
        if match:
            mag = float(match.group(1))
            for pos, obj_mag in self._magnifications.items():
                if obj_mag == mag:
                    return pos

        valid = [f"{mag}x (pos {pos})" for pos, mag in sorted(self._magnifications.items())]
        raise ValueError(f"Unknown magnification '{value}'. Available: {', '.join(valid)}")

    def validate_position(self, pos: int) -> int:
        """Validate turret position is in range, return it.

        Raises:
            ValueError: If position is out of range.
        """
        if self.min_position <= pos <= self.max_position:
            return pos
        raise ValueError(f"Position {pos} out of range ({self.min_position}-{self.max_position})")

    def __repr__(self) -> str:
        mag = self.magnification
        mag_str = f"{mag}x" if mag else f"pos={self.position}"
        return f"Nosepiece({self._name}, {mag_str})"


class ZDrive(Axis):
    """Z-axis (focus) drive control.

    Convenience wrapper providing easy access to the Z drive axis.
    Inherits all Axis functionality (move_to, move_to_async, position_um, etc.)

    Adds hysteresis-corrected position reading for more accurate Z position
    during motion. The Z axis has ~45 µm of mechanical backlash that the
    hysteresis-corrected interface compensates for.

    Usage:
        z = ZDrive.from_connection(conn)
        print(f"Z position: {z.position_um} µm")
        print(f"Z corrected: {z.position_um_hysteresis_corrected} µm")
        z.move_to(25000)  # blocking
        handle = z.move_to_async(24000)  # non-blocking
        handle.wait()
    """

    def __init__(self, unit: "Unit"):
        """Initialize ZDrive from SDK unit.

        Args:
            unit: SDK Unit object for Z drive.
        """
        super().__init__(unit)

        # Hysteresis-corrected interface for accurate position during motion
        self._bcv_hysteresis = get_interface(unit, IID.IID_BASIC_CONTROL_VALUE_HYSTERESIS_CORRECTED)

    @property
    def position_um_hysteresis_corrected(self) -> float | None:
        """Current Z position with hysteresis correction (more accurate during motion).

        The Z axis has ~45 µm of mechanical backlash. During motion:
        - Regular position_um lags by ~20-25 µm
        - This corrected value compensates based on motion direction

        Returns:
            Position in microns, or None if interface not available.
        """
        if self._bcv_hysteresis is None:
            return None
        native = self._bcv_hysteresis.GetControlValue()
        return self._converter.GetMetricsValue(native)

    @property
    def bcv_hysteresis(self):
        """Direct hysteresis-corrected BCV interface for fast polling, or None."""
        return self._bcv_hysteresis

    @property
    def supports_hysteresis_correction(self) -> bool:
        """Check if hysteresis-corrected position is available."""
        return self._bcv_hysteresis is not None

    def move_to_corrected(self, position_um: float) -> None:
        """Move to position using hysteresis-corrected interface.

        The SDK's hysteresis-corrected interface compensates for ~45 µm
        of mechanical backlash based on motion history/direction.
        Falls back to regular move_to if not available.

        Args:
            position_um: Target position in microns.
        """
        if self._bcv_hysteresis is not None:
            native = self._um_to_native(position_um)
            self._bcv_hysteresis.SetControlValue(native)
        else:
            self.move_to(position_um)

    @classmethod
    def from_connection(cls, conn: "LeicaConnection") -> "ZDrive":
        """Create ZDrive from a LeicaConnection.

        Args:
            conn: Active LeicaConnection.

        Returns:
            ZDrive instance.

        Raises:
            LookupError: If Z drive unit not found.
        """

        z_unit = conn.find_unit(TID.MICROSCOPE_ZDRIVE)
        if z_unit is None:
            raise LookupError("Z drive unit not found")
        return cls(z_unit)

    def __repr__(self) -> str:
        return f"ZDrive(pos={self.position_um:.1f}µm)"


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

        x_unit = conn.find_unit(TID.MICROSCOPE_X_UNIT)
        y_unit = conn.find_unit(TID.MICROSCOPE_Y_UNIT)

        if x_unit is None:
            raise LookupError("X axis unit not found")
        if y_unit is None:
            raise LookupError("Y axis unit not found")

        return cls(Axis(x_unit), Axis(y_unit))

    @property
    def position_um(self) -> Point2F:
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

    def move_to_async(self, x_um: float, y_um: float) -> tuple[MoveHandle, MoveHandle]:
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

    def move_rel_async(self, dx_um: float, dy_um: float) -> tuple[MoveHandle, MoveHandle]:
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
