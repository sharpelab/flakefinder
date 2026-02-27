"""Leica microscope control API.

This package provides a clean Python interface to the Leica AHM SDK
for controlling DM6M microscopes.

Quick start:
    from flakefinder.leica import LeicaConnection, Stage, ZDrive, Camera, Nosepiece

    with LeicaConnection() as conn:
        # High-level wrappers
        z = ZDrive.from_connection(conn)
        z.move_to(1000.0)  # blocking
        handle = z.move_to_async(2000.0)  # non-blocking
        handle.wait()

        stage = Stage.from_connection(conn)
        stage.move_to_async(5000, 5000)

        # Fast position reading for scanning
        t_before, t_after, pos_um = stage.x.read_position_timed()

        # Camera capture
        camera = Camera.from_connection(conn)
        camera.trigger_mode = 0  # CONTINUOUS
        print(f"Frame size: {camera.frame_size_px}")
        image = camera.capture()

        # Objective info
        nosepiece = Nosepiece.from_connection(conn)
        print(f"Objective: {nosepiece.magnification}x")

        # Continuous streaming
        with camera.stream(stage) as stream:
            frame = stream.get_frame(timeout=1.0)

Modules:
    core: Connection management and unit discovery
    enums: TID, IID, and other SDK enumerations
    units: Axis, Stage, MoveHandle, Nosepiece classes
    camera: Camera, FrameStream classes
    events: Event subscription system
    types: Protocol definitions for type hints
    utils: Conversion utilities and helpers
"""

import contextlib

# Windows: boost process-wide timer resolution from ~15ms to ~1ms.
# Without this, time.sleep() rounds up to the next 15ms tick, which
# makes rate-limited polling loops (~5ms target sleep) inaccurate.
# No-op on Linux (ctypes.windll doesn't exist). No cleanup needed —
# Windows ref-counts per-process and cleans up on exit.
with contextlib.suppress(Exception):
    import ctypes

    ctypes.windll.winmm.timeBeginPeriod(1)  # type: ignore[attr-defined]

from .autofocus import (
    AF_DEFAULTS,
    WORKING_DISTANCES_UM,
    AFDefaults,
    AutofocusFrame,
    AutofocusResult,
    continuous_autofocus,
    interpolate_position,
    sharpness,
)
from .camera import Camera, Frame, FrameStream
from .core import (
    LeicaConnection,
    find_all_units,
    find_unit,
    get_interface,
    get_interface_required,
    has_interface,
    leica_connection,
)
from .enums import (
    IID,
    TID,
    UCAPI_IID,
    UCAPI_PROP,
    UCAPI_TID,
    EErrorClass,
    EErrorCode,
    EEventType,
    EMetricsId,
    EState,
    MoveState,
)
from .events import AxisEvents, EventQueue, PositionMonitor, Subscription
from .microscope import Microscope
from .polling import PollingHandle, start_motion_polling, start_polling
from .units import Aperture, Axis, Lamp, MoveHandle, Nosepiece, Shutter, Stage, ZDrive, wait_all
from .utils import (
    UnitConverter,
    get_metrics_converter,
    microns_to_native,
    native_to_microns,
    print_unit_tree,
)

__all__ = [
    # Connection
    "LeicaConnection",
    "leica_connection",
    # Unit discovery
    "find_unit",
    "find_all_units",
    # Interface access
    "get_interface",
    "get_interface_required",
    "has_interface",
    # Enums
    "TID",
    "IID",
    "EMetricsId",
    "MoveState",
    "EState",
    "EErrorClass",
    "EErrorCode",
    "EEventType",
    "UCAPI_TID",
    "UCAPI_IID",
    "UCAPI_PROP",
    # Microscope facade
    "Microscope",
    # Units
    "Aperture",
    "Axis",
    "Stage",
    "ZDrive",
    "MoveHandle",
    "Lamp",
    "Shutter",
    "Nosepiece",
    "wait_all",
    # Camera
    "Camera",
    "FrameStream",
    "Frame",
    # Events
    "Subscription",
    "AxisEvents",
    "PositionMonitor",
    "EventQueue",
    # Polling
    "PollingHandle",
    "start_polling",
    "start_motion_polling",
    # Autofocus
    "AF_DEFAULTS",
    "AFDefaults",
    "sharpness",
    "interpolate_position",
    "AutofocusFrame",
    "AutofocusResult",
    "continuous_autofocus",
    "WORKING_DISTANCES_UM",
    # Utils
    "UnitConverter",
    "get_metrics_converter",
    "native_to_microns",
    "microns_to_native",
    "print_unit_tree",
]
