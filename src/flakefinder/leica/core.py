"""Core Leica SDK connection and unit discovery.

This module provides the low-level connection to the Leica hardware model
and utilities for finding units in the device tree.
"""

import contextlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from .enums import IID, TID, UCAPI_IID, UCAPI_TID
from .types import HardwareModel, Unit

# DLLs are in shared location: flakefinder/dlls/
_DLL_DIR = os.path.join(os.path.dirname(__file__), "..", "dlls")

# Flag to track if SDK has been initialized
_sdk_initialized = False


def _init_sdk() -> None:
    """Initialize the Leica SDK by loading required DLLs.

    This is called lazily on first connection attempt.
    """
    global _sdk_initialized
    if _sdk_initialized:
        return

    import clr

    clr.AddReference(os.path.join(_DLL_DIR, "hwmodel2.dll"))  # ty: ignore[unresolved-attribute]
    clr.AddReference(os.path.join(_DLL_DIR, "hwmodel2exucapi.dll"))  # ty: ignore[unresolved-attribute]
    clr.AddReference("System")  # ty: ignore[unresolved-attribute]

    _sdk_initialized = True


def _get_hardware_model(config_dir: str | None = None) -> "HardwareModel":
    """Get the HardwareModel singleton.

    Args:
        config_dir: Optional path to configuration directory.
                   If None, uses default SDK configuration.

    Returns:
        HardwareModel singleton instance.
    """
    _init_sdk()

    from LeicaMicrosystems.HardwareModel import HardwareModel

    if config_dir:
        return HardwareModel.TheHardwareModelInDirectory(config_dir)
    else:
        return HardwareModel.TheHardwareModel()


class LeicaConnection:
    """Context manager for Leica SDK lifecycle.

    Handles initialization and cleanup of SDK resources.

    Usage:
        with LeicaConnection() as conn:
            zdrive = conn.find_unit(TID.MICROSCOPE_ZDRIVE)
            # ... use zdrive ...
        # SDK resources automatically cleaned up

    Or without context manager (manual cleanup):
        conn = LeicaConnection()
        conn.connect()
        try:
            # ... use conn ...
        finally:
            conn.disconnect()
    """

    def __init__(self, config_dir: str | None = None):
        """Initialize connection wrapper.

        Args:
            config_dir: Optional path to SDK configuration directory.
        """
        self._config_dir = config_dir
        self._hwm: HardwareModel | None = None
        self._root: Unit | None = None

    def connect(self) -> "LeicaConnection":
        """Establish connection to microscope.

        Returns:
            self for method chaining.

        Raises:
            RuntimeError: If already connected.
            Exception: If SDK initialization fails.
        """
        if self._hwm is not None:
            raise RuntimeError("Already connected")

        self._hwm = _get_hardware_model(self._config_dir)
        self._root = self._hwm.GetUnit("")
        return self

    def disconnect(self) -> None:
        """Disconnect and release SDK resources."""
        if self._root is not None:
            with contextlib.suppress(Exception):  # Ignore cleanup errors
                self._root.Dispose()
            self._root = None

        if self._hwm is not None:
            with contextlib.suppress(Exception):  # Ignore cleanup errors
                self._hwm.Dispose()
            self._hwm = None

    @property
    def connected(self) -> bool:
        """Check if connection is active."""
        return self._hwm is not None

    @property
    def root(self) -> "Unit":
        """Get the root microscope unit.

        Raises:
            RuntimeError: If not connected.
        """
        if self._root is None:
            raise RuntimeError("Not connected")
        return self._root

    def find_unit(self, tid: TID | UCAPI_TID) -> "Unit | None":
        """Find a unit by type ID in the device tree.

        Args:
            tid: Type ID to search for.

        Returns:
            First matching unit, or None if not found.

        Raises:
            RuntimeError: If not connected.
        """
        if self._root is None:
            raise RuntimeError("Not connected")
        return find_unit(self._root, tid)

    def find_unit_required(self, tid: TID | UCAPI_TID) -> "Unit":
        """Find a unit by type ID, raising if not found.

        Args:
            tid: Type ID to search for.

        Returns:
            The matching unit.

        Raises:
            RuntimeError: If not connected.
            LookupError: If unit not found.
        """
        unit = self.find_unit(tid)
        if unit is None:
            raise LookupError(f"Unit not found: {tid.name}")
        return unit

    def __enter__(self) -> "LeicaConnection":
        """Enter context manager, establishing connection."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, cleaning up resources."""
        self.disconnect()


@contextmanager
def leica_connection(config_dir: str | None = None) -> Iterator[LeicaConnection]:
    """Context manager function for Leica connection.

    Alternative to using LeicaConnection directly:

        with leica_connection() as conn:
            # ... use conn ...

    Args:
        config_dir: Optional configuration directory path.

    Yields:
        Connected LeicaConnection instance.
    """
    conn = LeicaConnection(config_dir)
    conn.connect()
    try:
        yield conn
    finally:
        conn.disconnect()


def find_unit(root: "Unit", tid: TID | UCAPI_TID) -> "Unit | None":
    """Recursively search for a unit by type ID.

    Args:
        root: Root unit to start search from.
        tid: Type ID to search for.

    Returns:
        First matching unit, or None if not found.
    """

    def search(unit: "Unit") -> "Unit | None":
        # Check this unit
        if unit.GetUnitType().IsA(int(tid)):
            return unit

        # Search children
        children = unit.GetUnits()
        if children is None:
            return None

        for i in range(children.NumUnits()):
            child = children.GetUnit(i)
            if child is None:
                continue
            found = search(child)
            if found is not None:
                return found

        return None

    return search(root)


def find_all_units(root: "Unit", tid: TID) -> list["Unit"]:
    """Find all units matching a type ID.

    Args:
        root: Root unit to start search from.
        tid: Type ID to search for.

    Returns:
        List of all matching units (may be empty).
    """
    results: list[Unit] = []

    def search(unit: "Unit") -> None:
        if unit.GetUnitType().IsA(int(tid)):
            results.append(unit)

        children = unit.GetUnits()
        if children is None:
            return

        for i in range(children.NumUnits()):
            child = children.GetUnit(i)
            if child is not None:
                search(child)

    search(root)
    return results


T = TypeVar("T")


def get_interface[T](unit: "Unit", iid: IID | UCAPI_IID, expected_type: type[T] | None = None) -> T | None:
    """Get an interface from a unit by interface ID.

    Args:
        unit: Unit to get interface from.
        iid: Interface ID to retrieve.
        expected_type: Optional type for documentation (not enforced at runtime).

    Returns:
        Interface object, or None if not found.
    """
    interfaces = unit.GetInterfaces()
    if interfaces is None:
        return None

    iface = interfaces.FindInterface(int(iid))
    if iface is None:
        return None

    return iface.GetObject()


def get_interface_required[T](unit: "Unit", iid: IID | UCAPI_IID, expected_type: type[T] | None = None) -> T:
    """Get an interface from a unit, raising if not found.

    Args:
        unit: Unit to get interface from.
        iid: Interface ID to retrieve.
        expected_type: Optional type for documentation (not enforced at runtime).

    Returns:
        Interface object.

    Raises:
        LookupError: If interface not found.
    """
    iface = get_interface(unit, iid, expected_type)
    if iface is None:
        raise LookupError(f"Interface not found: {iid.name} on unit {unit.GetName()}")
    return iface


def has_interface(unit: "Unit", iid: IID) -> bool:
    """Check if a unit has a specific interface.

    Args:
        unit: Unit to check.
        iid: Interface ID to look for.

    Returns:
        True if unit has the interface.
    """
    interfaces = unit.GetInterfaces()
    if interfaces is None:
        return False
    return interfaces.HasInterface(int(iid))
