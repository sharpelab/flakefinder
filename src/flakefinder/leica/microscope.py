"""High-level microscope facade.

Consolidates hardware access behind a single context manager, eliminating
per-script boilerplate for connection, subsystem lookup, and UCAPI registration.
"""

import contextlib

from flakefinder.types import MicroscopeDescription, Point3F

from .camera import Camera
from .core import LeicaConnection
from .units import Lamp, Nosepiece, Shutter, Stage, ZDrive


class Microscope:
    """High-level facade for the Leica DM6M microscope.

    Owns the LeicaConnection lifecycle and provides single-point access
    to all hardware subsystems. Acts as a context manager.

    Usage:
        with Microscope() as scope:
            scope.light_on()
            scope.switch_objective_mag("20x")
            image = scope.camera.capture()

    All subsystems (stage, z, nosepiece, shutter, lamp) are initialized
    eagerly on __enter__ and raise LookupError if not found. Camera and
    acquisition are lazily initialized on first access to avoid UCAPI
    import cost for scripts that don't need them.
    """

    def __init__(self) -> None:
        """Initialize (does NOT connect yet — use as context manager)."""
        self._conn: LeicaConnection | None = None
        self._stage: Stage | None = None
        self._z: ZDrive | None = None
        self._nosepiece: Nosepiece | None = None
        self._shutter: Shutter | None = None
        self._lamp: Lamp | None = None
        self._camera: Camera | None = None
        self._camera_initialized = False
        self._context = None  # Lazy acquisition context

    def __enter__(self) -> "Microscope":
        """Connect to hardware and initialize subsystems."""
        self._conn = LeicaConnection()
        self._conn.connect()
        try:
            self._stage = Stage.from_connection(self._conn)
            self._z = ZDrive.from_connection(self._conn)
            self._nosepiece = Nosepiece.from_connection(self._conn)
            self._shutter = Shutter.from_connection(self._conn)
            self._lamp = Lamp.from_connection(self._conn)
        except Exception:
            self._conn.disconnect()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Dispose acquisition context, camera, and disconnect."""
        if self._context is not None:
            with contextlib.suppress(Exception):
                self._context.Dispose()
        if self._camera is not None:
            with contextlib.suppress(Exception):
                self._camera.dispose()
        if self._conn is not None:
            self._conn.disconnect()

    # --- Subsystem properties ---

    @property
    def conn(self) -> LeicaConnection:
        """Underlying LeicaConnection (escape hatch for advanced use)."""
        if self._conn is None:
            raise RuntimeError("Not connected")
        return self._conn

    @property
    def stage(self) -> Stage:
        """XY stage."""
        if self._stage is None:
            raise RuntimeError("Not connected")
        return self._stage

    @property
    def z(self) -> ZDrive:
        """Z drive (focus axis)."""
        if self._z is None:
            raise RuntimeError("Not connected")
        return self._z

    @property
    def nosepiece(self) -> Nosepiece:
        """Nosepiece/turret."""
        if self._nosepiece is None:
            raise RuntimeError("Not connected")
        return self._nosepiece

    @property
    def shutter(self) -> Shutter:
        """IL shutter."""
        if self._shutter is None:
            raise RuntimeError("Not connected")
        return self._shutter

    @property
    def lamp(self) -> Lamp:
        """Lamp."""
        if self._lamp is None:
            raise RuntimeError("Not connected")
        return self._lamp

    # --- Camera (lazy init) ---

    def _init_camera(self) -> None:
        """Initialize camera (lazily on first access).

        Camera.__init__ handles UCAPI registration and acquires the
        acquisition interface internally — no need to duplicate that here.
        """
        if self._camera_initialized:
            return
        self._camera_initialized = True
        self._camera = Camera.from_connection(self._conn)

    @property
    def camera(self) -> Camera:
        """Camera (lazily initialized on first access).

        First access triggers UCAPI registration and camera init.
        """
        self._init_camera()
        assert self._camera is not None
        return self._camera

    @property
    def acquisition(self):
        """Raw image acquisition interface for scan loops.

        Use with create_acquisition_context() for direct capture control.
        Delegates to Camera's own acquisition interface to avoid a redundant
        GetObject() call on the SDK interface node.
        """
        self._init_camera()
        assert self._camera is not None
        return self._camera._acquisition

    @property
    def context(self):
        """Shared acquisition context (lazily created, auto-disposed on exit).

        Use for single-shot captures and autofocus. Streaming classes
        (FrameStream, DeferredFrameStream) create their own per-thread
        contexts and should NOT use this one.
        """
        if self._context is None:
            self._context = self.create_acquisition_context()
        return self._context

    # --- Computed properties ---

    @property
    def objective_mag(self) -> float | None:
        """Current objective magnification, or None if position unknown."""
        return self.nosepiece.magnification

    @property
    def position(self) -> Point3F:
        """Current (X, Y, Z) position in microns."""
        x, y = self.stage.position_um
        return Point3F(x, y, self.z.position_um)

    # --- High-level operations ---

    def switch_objective_pos(self, pos: int) -> bool:
        """Switch objective by turret position (1-6).

        Temporarily maxes Z speed for the SDK's internal z-hop.
        The SDK's nosepiece SetControlValue performs an internal z-hop
        (retract, rotate, return). If Z speed is set low (e.g. from
        autofocus), this can exceed the SDK's internal timeout.

        No-op if already at the target position.

        Args:
            pos: Turret position (1-6).

        Returns:
            True if the objective was switched, False if already at target.

        Raises:
            ValueError: If position is out of range.
        """
        nosepiece = self.nosepiece
        target_pos = nosepiece.validate_position(pos)
        if target_pos == nosepiece.position:
            return False
        z = self.z
        saved = z.velocity_native
        z.set_velocity_native(z.max_velocity_native)
        try:
            nosepiece.position = target_pos
        finally:
            z.set_velocity_native(saved)
        return True

    def switch_objective_mag(self, mag: str) -> bool:
        """Switch objective by magnification string (e.g. '20x', '5', '2.5').

        Parses the magnification and delegates to switch_objective_pos().

        Args:
            mag: Magnification string (e.g. '20x', '5', '2.5x').

        Returns:
            True if the objective was switched, False if already at target.

        Raises:
            ValueError: If magnification is unknown.
        """
        target_pos = self.nosepiece.parse_magnification(mag)
        return self.switch_objective_pos(target_pos)

    def light_on(self, intensity_pct: float = 100) -> None:
        """Open shutter and set lamp intensity.

        Args:
            intensity_pct: Lamp intensity as percentage (0-100).
                          Default 100 (full brightness).
        """
        self.shutter.open()
        self.lamp.intensity_pct = intensity_pct

    def light_off(self) -> None:
        """Close shutter and turn lamp off."""
        self.shutter.close()
        self.lamp.off()

    def validate_description(self, desc: MicroscopeDescription) -> None:
        """Validate a microscope description against live hardware.

        Call after connecting to ensure pre-connection checks (area bounds,
        frame size estimates) used trustworthy values.

        Args:
            desc: Parsed MicroscopeDescription to validate.

        Raises:
            ValueError: If any description value doesn't match live hardware.
        """
        errors: list[str] = []
        tol = 1.0  # µm tolerance for float conversion differences

        # Stage axes
        for name, desc_axis, live_axis in [
            ("X", desc.stage.x, self.stage.x),
            ("Y", desc.stage.y, self.stage.y),
            ("Z", desc.stage.z, self.z),
        ]:
            if abs(desc_axis.min_um - live_axis.min_um) > tol:
                errors.append(f"{name} min: desc={desc_axis.min_um:.1f}, live={live_axis.min_um:.1f}")
            if abs(desc_axis.max_um - live_axis.max_um) > tol:
                errors.append(f"{name} max: desc={desc_axis.max_um:.1f}, live={live_axis.max_um:.1f}")

        # Nosepiece objectives
        live_mags = self.nosepiece.magnifications
        for pos, obj in desc.objectives.items():
            live_mag = live_mags.get(pos)
            if live_mag is None:
                errors.append(f"Objective pos {pos}: in description but not on nosepiece")
            elif obj.magnification != live_mag:
                errors.append(f"Objective pos {pos}: desc={obj.magnification}x, live={live_mag}x")

        if errors:
            raise ValueError("Microscope description does not match live hardware:\n  " + "\n  ".join(errors))

    def create_acquisition_context(self):
        """Create a new CancellableImageAcquisitionContext.

        The caller owns the returned context and must dispose it when done.

        Returns:
            CancellableImageAcquisitionContext factory instance.
        """
        # Access camera to ensure it's initialized
        self.camera  # noqa: B018
        from LeicaMicrosystems.HardwareModel import Extensions

        return Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory

    def __repr__(self) -> str:
        parts = []
        if self._conn is not None and self._conn.connected:
            try:
                x, y, z_pos = self.position
                parts.append(f"pos=({x:.0f}, {y:.0f}, {z_pos:.0f})µm")
            except Exception:
                pass
            mag = self.nosepiece.magnification
            if mag:
                parts.append(f"{mag}x")
        else:
            parts.append("disconnected")
        return f"Microscope({', '.join(parts)})"
