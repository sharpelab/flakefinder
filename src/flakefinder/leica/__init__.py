"""Leica microscope control API.

This package provides a clean Python interface to the Leica AHM SDK
for controlling DM6M microscopes.

Quick start:
    from flakefinder.leica import LeicaConnection, TID

    with LeicaConnection() as conn:
        zdrive = conn.find_unit(TID.MICROSCOPE_ZDRIVE)
        # ... control the microscope ...

Modules:
    core: Connection management and unit discovery
    enums: TID, IID, and other SDK enumerations
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
    # Utils
    "UnitConverter",
    "get_metrics_converter",
    "native_to_microns",
    "microns_to_native",
    "print_unit_tree",
]
