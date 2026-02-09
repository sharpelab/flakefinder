"""Utility functions for Leica SDK operations."""

from .core import get_interface
from .enums import IID, EMetricsId
from .types import BasicControlValue, MetricsConverter, Unit


def get_metrics_converter(
    unit: "Unit",
    metrics_id: EMetricsId = EMetricsId.METRICS_MICRONS,
) -> "MetricsConverter | None":
    """Get a metrics converter for a unit.

    Args:
        unit: Unit to get converter from (must have BasicControlValue interface).
        metrics_id: Type of metrics conversion (default: microns).

    Returns:
        MetricsConverter for the specified metrics, or None if not available.
    """
    bcv: BasicControlValue | None = get_interface(unit, IID.IID_BASIC_CONTROL_VALUE)
    if bcv is None:
        return None

    converters = bcv.GetMetricsConverters()
    if converters is None:
        return None

    return converters.FindMetricsConverter(int(metrics_id))


def native_to_microns(converter: "MetricsConverter", native_value: int) -> float:
    """Convert native (step) value to microns.

    Args:
        converter: MetricsConverter configured for microns.
        native_value: Value in native units (steps).

    Returns:
        Value in microns.
    """
    return converter.GetMetricsValue(native_value)


def microns_to_native(converter: "MetricsConverter", microns: float) -> int:
    """Convert microns to native (step) value.

    Args:
        converter: MetricsConverter configured for microns.
        microns: Value in microns.

    Returns:
        Value in native units (steps).
    """
    return converter.GetControlValue(microns)


class UnitConverter:
    """Convenience wrapper for unit conversion.

    Caches the metrics converter and provides simple conversion methods.

    Usage:
        conv = UnitConverter(zdrive_unit)
        pos_um = conv.to_microns(native_pos)
        native = conv.from_microns(100.5)
    """

    def __init__(
        self,
        unit: "Unit",
        metrics_id: EMetricsId = EMetricsId.METRICS_MICRONS,
    ):
        """Initialize converter for a unit.

        Args:
            unit: Unit to convert values for.
            metrics_id: Type of metrics conversion.

        Raises:
            ValueError: If unit doesn't support the requested metrics.
        """
        converter = get_metrics_converter(unit, metrics_id)
        if converter is None:
            raise ValueError(f"Unit {unit.GetName()} does not support metrics {metrics_id.name}")
        self._converter = converter
        self._metrics_id = metrics_id

    def to_microns(self, native_value: int) -> float:
        """Convert native value to microns."""
        return self._converter.GetMetricsValue(native_value)

    def from_microns(self, microns: float) -> int:
        """Convert microns to native value."""
        return self._converter.GetControlValue(microns)

    @property
    def metrics_id(self) -> EMetricsId:
        """Get the metrics type this converter uses."""
        return self._metrics_id


def print_unit_tree(unit: "Unit", indent: int = 0) -> None:
    """Print the unit tree structure for debugging.

    Args:
        unit: Root unit to print from.
        indent: Current indentation level (internal use).
    """
    prefix = "  " * indent
    unit_type = unit.GetUnitType()
    type_id = unit_type.ExposedTypeId()
    name = unit.GetName()

    print(f"{prefix}- {name} (TID={type_id})")

    # Print interfaces
    interfaces = unit.GetInterfaces()
    if interfaces is not None:
        for i in range(interfaces.NumInterfaces()):
            try:
                iface = interfaces.GetInterface(i)
                iid = iface.GetInterfaceId()
                print(f"{prefix}    [IID={iid}]")
            except Exception:
                pass

    # Recurse to children
    children = unit.GetUnits()
    if children is not None:
        for i in range(children.NumUnits()):
            child = children.GetUnit(i)
            if child is not None:
                print_unit_tree(child, indent + 1)
