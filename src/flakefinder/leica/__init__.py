"""Leica microscope control API.

This package provides a clean Python interface to the Leica AHM SDK
for controlling DM6M microscopes.

Quick start:
    from flakefinder.leica import LeicaConnection, TID, Axis, Stage

    with LeicaConnection() as conn:
        # Low-level: find units directly
        zdrive = conn.find_unit(TID.MICROSCOPE_ZDRIVE)

        # High-level: use Axis/Stage wrappers
        z = Axis(conn.find_unit_required(TID.MICROSCOPE_ZDRIVE))
        z.move_to(1000.0)  # blocking
        handle = z.move_to_async(2000.0)  # non-blocking
        handle.wait()

        stage = Stage.from_connection(conn)
        stage.move_to_async(5000, 5000)

Modules:
    core: Connection management and unit discovery
    enums: TID, IID, and other SDK enumerations
    units: Axis, Stage, MoveHandle classes
    types: Protocol definitions for type hints
    utils: Conversion utilities and helpers
"""

from .core import (
    LeicaConnection,
    leica_connection,
    find_unit,
    find_all_units,
    get_interface,
    get_interface_required,
    has_interface,
)
from .enums import TID, IID, EMetricsId, MoveState
from .units import Axis, Stage, MoveHandle
from .utils import (
    UnitConverter,
    get_metrics_converter,
    native_to_microns,
    microns_to_native,
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
    # Units
    "Axis",
    "Stage",
    "MoveHandle",
    # Utils
    "UnitConverter",
    "get_metrics_converter",
    "native_to_microns",
    "microns_to_native",
    "print_unit_tree",
]
