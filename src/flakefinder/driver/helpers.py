from .enums import TID
from .interfaces import Unit


def find_unit_by_type(unit: Unit, tid: TID) -> Unit | None:
    """Find a unit of a specific type within a microscope unit's hierarchy.
    Args:
        microscope_unit (Unit): The root unit to start the search from.
        tid (TID): The type ID of the unit to find.
    Returns:
        Unit: The first unit found that matches the specified type ID,
            or None if no such unit is found.
    """

    def search_recursively(current_parent_unit: Unit) -> Unit | None:
        # This helper function searches for a unit of type 'tid'
        # among the direct children of 'current_parent_unit',
        # and then recursively in their children.

        if current_parent_unit is None:
            return None

        subunits_collection = current_parent_unit.GetUnits()
        if subunits_collection is None:
            # No subunits to search
            return None

        num_subunits = subunits_collection.NumUnits()
        for i in range(num_subunits):
            # Get the child unit
            child_unit = subunits_collection.GetUnit(i)

            if child_unit is None:
                # Skip if the subunit itself is None (defensive check)
                continue

            # Check if this child_unit is the one we're looking for
            if child_unit.GetUnitType().IsA(tid):
                return child_unit  # Found it!

            # If not, recurse: search within this child_unit's own subunits
            found_in_descendants = search_recursively(child_unit)
            if found_in_descendants:
                # Found in a deeper level
                return found_in_descendants

        # Not found among the children (or their descendants) of current_parent_unit
        return None

    # Start the search: look for units within the subunits of the initial 'microscope_unit'
    return search_recursively(unit)
