"""Event subscription system for real-time hardware updates.

This module provides safe Python wrappers for the Leica SDK's event system.
Events fire on .NET thread pool threads, so this module handles threading
concerns and provides a clean Python interface.

Key classes:
- Subscription: RAII wrapper for event subscriptions
- PositionMonitor: Tracks position changes on an axis
"""

import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

from .core import get_interface
from .enums import IID
from .types import EventSource, Unit

# Type alias for position callback: (position_native: int) -> None
PositionCallback = Callable[[int], None]

# Type alias for raw event callback: (sender: Unit, iid: int, value: int) -> None
RawEventCallback = Callable[["Unit", int, int], None]


class Subscription:
    """RAII wrapper for event subscriptions.

    Automatically unsubscribes when the subscription goes out of scope
    or when unsubscribe() is called.

    Usage:
        # Context manager (recommended)
        with axis_events.subscribe_position(callback) as sub:
            # ... callback will fire while in this block ...
        # Automatically unsubscribed

        # Manual management
        sub = axis_events.subscribe_position(callback)
        # ... later ...
        sub.unsubscribe()
    """

    def __init__(
        self,
        event_source: "EventSource",
        handler: Any,
        unsubscribe_method: str = "UnSubscribe",
    ):
        """Initialize subscription.

        Args:
            event_source: SDK EventSource interface.
            handler: The delegate/handler that was subscribed.
            unsubscribe_method: Method name to call for unsubscribing.
        """
        self._event_source = event_source
        self._handler = handler
        self._unsubscribe_method = unsubscribe_method
        self._active = True

    @property
    def active(self) -> bool:
        """Check if subscription is still active."""
        return self._active

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if self._active and self._event_source is not None:
            try:
                unsub = getattr(self._event_source, self._unsubscribe_method)
                unsub(self._handler)
            except Exception:
                pass  # Ignore errors during unsubscribe
            self._active = False

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unsubscribe()

    def __del__(self):
        self.unsubscribe()


class AxisEvents:
    """Event subscriptions for a single axis.

    Wraps the SDK's EventSource interface and provides Python-friendly
    callbacks for position changes.

    Usage:
        events = AxisEvents(axis_unit)

        # Subscribe to position changes
        def on_position(pos_native: int):
            print(f"Position: {pos_native}")

        with events.subscribe_position(on_position):
            # Move the axis - callback fires during movement
            axis.move_to_async(1000).wait()

    Threading note:
        Callbacks fire on .NET thread pool threads, not the main Python thread.
        Keep callbacks fast and thread-safe. For complex processing, consider
        using a queue to pass data to the main thread.
    """

    def __init__(self, unit: "Unit"):
        """Initialize axis events.

        Args:
            unit: SDK Unit object (must support EventSource interface).

        Raises:
            LookupError: If unit doesn't have EventSource interface.
        """
        self._unit = unit
        event_source: EventSource | None = get_interface(unit, IID.IID_EVENT_SOURCE)
        if event_source is None:
            raise LookupError(f"Unit {unit.GetName()} doesn't support events")
        self._event_source: EventSource = event_source

        # Keep references to prevent GC of callbacks
        self._handlers: list[Any] = []

    def subscribe_raw(self, callback: RawEventCallback) -> Subscription:
        """Subscribe to raw value change events.

        Args:
            callback: Function called with (sender, iid, value) on each event.

        Returns:
            Subscription that can be used to unsubscribe.
        """
        # Import the delegate type from .NET
        from LeicaMicrosystems.HardwareModel import EventSource as ESType

        # Create wrapper that calls the Python callback
        def handler(sender: "Unit", iid: int, value: int) -> None:
            with contextlib.suppress(Exception):  # Don't let exceptions propagate to .NET
                callback(sender, iid, value)

        # Create .NET delegate explicitly (pythonnet requires this)
        delegate = ESType.ValueChangedEventHandler(handler)

        # Keep references to prevent GC (both handler and delegate)
        self._handlers.append((handler, delegate))

        # Subscribe with the delegate
        self._event_source.Subscribe(delegate)

        return Subscription(self._event_source, delegate, "UnSubscribe")

    def subscribe_position(self, callback: PositionCallback) -> Subscription:
        """Subscribe to position change events only.

        Filters events to only call back when IID_BASIC_CONTROL_VALUE changes.

        Args:
            callback: Function called with position (native units) on each change.

        Returns:
            Subscription that can be used to unsubscribe.
        """

        def filtered_handler(sender: "Unit", iid: int, value: int) -> None:
            if iid == int(IID.IID_BASIC_CONTROL_VALUE):
                callback(value)

        return self.subscribe_raw(filtered_handler)


class PositionMonitor:
    """Monitor position changes with thread-safe value access.

    Subscribes to position events and maintains the latest position,
    accessible from any thread.

    Usage:
        with PositionMonitor(axis_unit) as monitor:
            handle = axis.move_to_async(target)
            while not handle.is_complete:
                print(f"Position: {monitor.position}")
                time.sleep(0.05)
    """

    def __init__(self, unit: "Unit"):
        """Initialize position monitor.

        Args:
            unit: SDK Unit object to monitor.
        """
        self._unit = unit
        self._events = AxisEvents(unit)
        self._position: int = 0
        self._lock = threading.Lock()
        self._subscription: Subscription | None = None
        self._update_count = 0

    def _on_position(self, pos_native: int) -> None:
        """Handle position update from event."""
        with self._lock:
            self._position = pos_native
            self._update_count += 1

    @property
    def position(self) -> int:
        """Get latest position (native units). Thread-safe."""
        with self._lock:
            return self._position

    @property
    def update_count(self) -> int:
        """Get number of position updates received. Thread-safe."""
        with self._lock:
            return self._update_count

    def start(self) -> "PositionMonitor":
        """Start monitoring. Returns self for chaining."""
        if self._subscription is None:
            # Get initial position
            from .core import get_interface

            bcv = get_interface(self._unit, IID.IID_BASIC_CONTROL_VALUE)
            if bcv:
                self._position = bcv.GetControlValue()

            self._subscription = self._events.subscribe_position(self._on_position)
        return self

    def stop(self) -> None:
        """Stop monitoring."""
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None

    def __enter__(self) -> "PositionMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class EventQueue:
    """Queue-based event handler for processing events on main thread.

    Use this when you need to process events on the main thread rather
    than on .NET's thread pool.

    Usage:
        eq = EventQueue()
        sub = axis_events.subscribe_position(eq.handler)

        # In main loop
        while running:
            for pos in eq.drain():
                print(f"Position update: {pos}")
            time.sleep(0.01)
    """

    def __init__(self, maxsize: int = 1000):
        """Initialize event queue.

        Args:
            maxsize: Maximum queue size. Older events dropped if full.
        """
        self._queue: queue.Queue[int] = queue.Queue(maxsize=maxsize)

    def handler(self, value: int) -> None:
        """Event handler that queues values. Use with subscribe_position."""
        try:
            self._queue.put_nowait(value)
        except queue.Full:
            # Drop oldest and add new
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(value)
            except queue.Empty:
                pass

    def drain(self) -> list[int]:
        """Get all queued values, clearing the queue."""
        values = []
        while True:
            try:
                values.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return values

    def get(self, timeout: float | None = None) -> int | None:
        """Get next value, blocking if necessary.

        Args:
            timeout: Max seconds to wait (None = forever).

        Returns:
            Position value, or None if timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def pending(self) -> int:
        """Number of events waiting in queue."""
        return self._queue.qsize()
