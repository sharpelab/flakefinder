"""High-level microscope facade.

Consolidates hardware access behind a single context manager, eliminating
per-script boilerplate for connection, subsystem lookup, and UCAPI registration.
"""

import contextlib

from flakefinder.types import Point3F

from .camera import Camera
from .core import LeicaConnection, get_interface_required
from .enums import UCAPI_IID
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
        self._acquisition = None
        self._camera_initialized = False

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
        """Dispose camera and disconnect."""
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
        """Initialize camera and acquisition interface.

        Called lazily on first access to camera or acquisition properties.
        Camera.__init__ handles UCAPI registration internally.
        """
        if self._camera_initialized:
            return
        self._camera_initialized = True
        self._camera = Camera.from_connection(self._conn)
        self._acquisition = get_interface_required(self._camera._unit, UCAPI_IID.IID_IMAGE_ACQUISITION)

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
        """
        self._init_camera()
        return self._acquisition

    # --- Computed properties ---

    @property
    def objective_mag(self) -> float | None:
        """Current objective magnification, or None if position unknown."""
        return self.nosepiece.magnification

    @property
    def position(self) -> Point3F:
        """Current (X, Y, Z) position in microns."""
        x, y = self.stage.position_um
        return (x, y, self.z.position_um)

    # --- High-level operations ---

    def switch_objective_pos(self, pos: int) -> None:
        """Switch objective by turret position (1-6).

        Temporarily maxes Z speed for the SDK's internal z-hop.
        The SDK's nosepiece SetControlValue performs an internal z-hop
        (retract, rotate, return). If Z speed is set low (e.g. from
        autofocus), this can exceed the SDK's internal timeout.

        No-op if already at the target position.

        Args:
            pos: Turret position (1-6).

        Raises:
            ValueError: If position is out of range.
        """
        nosepiece = self.nosepiece
        target_pos = nosepiece.validate_position(pos)
        if target_pos != nosepiece.position:
            z = self.z
            saved = z.velocity_native
            z.set_velocity_native(z.max_velocity_native)
            try:
                nosepiece.position = target_pos
            finally:
                z.set_velocity_native(saved)

    def switch_objective_mag(self, mag: str) -> None:
        """Switch objective by magnification string (e.g. '20x', '5', '2.5').

        Parses the magnification and delegates to switch_objective_pos().

        Args:
            mag: Magnification string (e.g. '20x', '5', '2.5x').

        Raises:
            ValueError: If magnification is unknown.
        """
        target_pos = self.nosepiece.parse_magnification(mag)
        self.switch_objective_pos(target_pos)

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
