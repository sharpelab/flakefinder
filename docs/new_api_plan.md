# New Leica API Plan: Continuous Scanning Support

## Goal
Build a new API layer that supports continuous scanning - coordinated stage movement with image acquisition without stop-and-go blocking operations.

## Current State

The existing `flakefinder/driver/` uses pythonnet to call Leica's `hwmodel2.dll` directly. Problems:

1. **All moves are blocking** - `SetControlValue()` waits until the stage stops before returning
2. **No event subscriptions** - can't get real-time position updates during movement
3. **No async move support** - `BasicControlValueAsync` interface isn't used
4. **No halt capability** - can't emergency stop a movement
5. **Tight coupling to .NET objects** - hard to test, hard to mock

## Proposed Architecture

```
flakefinder/
├── driver/                    # Existing (keep for reference, deprecate)
│   └── ...
└── leica/                     # New API
    ├── __init__.py
    ├── core.py                # Low-level .NET wrappers
    ├── units.py               # Unit abstractions (Stage, ZDrive, etc.)
    ├── events.py              # Event subscription system
    ├── scanning.py            # High-level scanning orchestration
    └── mock.py                # Mock implementations for testing
```

### Layer 1: Core (.NET wrapper)

Thin pythonnet wrapper that exposes SDK primitives with proper cleanup:

```python
# core.py
class LeicaConnection:
    """Context manager for SDK lifecycle."""
    def __enter__(self) -> 'LeicaConnection': ...
    def __exit__(self, ...): ...  # Proper Dispose() calls

    @property
    def root(self) -> Unit: ...

    def find_unit(self, tid: TID) -> Unit | None: ...
```

### Layer 2: Units (Component abstractions)

Each hardware component gets a class with:
- **Sync** methods (blocking, for setup/calibration)
- **Async** methods (non-blocking, returns handle)
- **Event** subscription for real-time updates
- **Halt** for emergency stop

```python
# units.py
class Axis:
    """Single-axis control (X, Y, or Z)."""

    # Sync (blocking)
    def move_to(self, pos_um: float) -> None: ...
    def move_rel(self, delta_um: float) -> None: ...

    # Async (non-blocking)
    def move_to_async(self, pos_um: float) -> MoveHandle: ...

    # Status
    @property
    def position_um(self) -> float: ...
    @property
    def is_moving(self) -> bool: ...

    # Control
    def halt(self) -> None: ...
    def set_velocity(self, um_per_sec: float) -> None: ...

    # Events
    def subscribe(self, callback: Callable[[float], None]) -> Subscription: ...

class MoveHandle:
    """Handle to track async move completion."""
    def wait(self, timeout: float = None) -> bool: ...
    def cancel(self) -> None: ...
    @property
    def state(self) -> MoveState: ...  # PENDING, IN_PROGRESS, COMPLETE, CANCELLED
```

### Layer 3: Events (Real-time updates)

Event system that bridges .NET events to Python callbacks:

```python
# events.py
class Subscription:
    """RAII subscription handle."""
    def __enter__(self): ...
    def __exit__(self, ...): ...  # Auto-unsubscribe
    def unsubscribe(self): ...

class EventBus:
    """Central event routing."""
    def subscribe_position(self, axis: Axis, cb: Callable[[float], None]) -> Subscription: ...
    def subscribe_all_axes(self, cb: Callable[[str, float], None]) -> Subscription: ...
```

### Layer 4: Scanning (Orchestration)

High-level scanning operations:

```python
# scanning.py
class ContinuousScanner:
    """Coordinated stage movement + image acquisition."""

    def __init__(self, stage: Stage, camera: Camera): ...

    # Scan patterns
    def line_scan(
        self,
        start: tuple[float, float],  # (x_um, y_um)
        end: tuple[float, float],
        velocity: float,  # um/s
        frame_callback: Callable[[np.ndarray, tuple[float, float]], None],
    ) -> ScanHandle: ...

    def raster_scan(
        self,
        bounds: tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max)
        line_spacing: float,
        velocity: float,
        frame_callback: Callable[[np.ndarray, tuple[float, float]], None],
    ) -> ScanHandle: ...

class ScanHandle:
    """Handle to monitor/control running scan."""
    def wait(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def cancel(self) -> None: ...
    @property
    def progress(self) -> float: ...  # 0.0-1.0
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `leica/` package structure
2. Implement `LeicaConnection` with proper resource management
3. Add unit finder utilities
4. Add TID/IID enums (can copy from existing driver)

### Phase 2: Async Axis Control
1. Implement `Axis` class with sync methods first
2. Add `BasicControlValueAsync` wrapper for async moves
3. Implement `MoveHandle` with completion tracking
4. Add `HaltControlValue` wrapper for halt
5. Test with Z-drive (simplest single axis)

### Phase 3: Event System
1. Implement `EventSource` wrapper with pythonnet delegate handling
2. Create `Subscription` RAII class
3. Implement position event callbacks
4. Handle threading (events fire on .NET thread pool)

### Phase 4: Stage Integration
1. Implement `Stage` class wrapping X and Y axes
2. Add coordinated XY moves
3. Add velocity control

### Phase 5: Continuous Scanning
1. Implement `ContinuousScanner` with line scan
2. Add camera trigger coordination
3. Implement raster scan pattern
4. Add position-tagged frame capture

### Phase 6: Migration
1. Update CLI to use new API
2. Add deprecation warnings to old driver
3. Document migration path

## Key SDK Interfaces to Wrap

| Interface | Purpose | Priority |
|-----------|---------|----------|
| BasicControlValue | Sync get/set position | P1 |
| BasicControlValueAsync | Async moves | P1 |
| HaltControlValue | Emergency stop | P1 |
| EventSource | Position updates | P1 |
| BasicControlState | Check if moving | P2 |
| BasicControlValueVelocity | Speed control | P2 |
| MetricsConverter | Steps ↔ microns | P1 |
| ImageAcquisition.AcquireContinuous | Streaming frames | P2 |

## Threading Considerations

The Leica SDK fires events on .NET thread pool threads. Need to:
1. Use pythonnet's `Py.GIL()` in callbacks
2. Consider queue-based pattern: events push to queue, Python thread consumes
3. Ensure callbacks don't block (could deadlock SDK)

```python
# Pattern for safe event handling
import queue
import threading

class EventBridge:
    def __init__(self):
        self._queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._process, daemon=True)
        self._thread.start()

    def _on_dotnet_event(self, sender, iid, value):
        # Called on .NET thread - minimal work here
        self._queue.put((sender, iid, value))

    def _process(self):
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                # Process in Python thread
                self._dispatch(event)
            except queue.Empty:
                continue
```

## Test Strategy

1. **Unit tests with mocks** - `mock.py` provides fake implementations
2. **Integration tests on hardware** - require microscope connection
3. **Continuous scanning smoke test** - verify frames come in at expected rate

## Files Created

```
flakefinder/src/flakefinder/
├── dlls/                    # MOVED: shared DLL location
│   └── README.md
├── driver/
│   ├── dlls/README.md       # Points to ../dlls/
│   └── microscope.py        # Updated to use ../dlls/
└── leica/                   # NEW API
    ├── __init__.py          # Package exports
    ├── core.py              # LeicaConnection, unit discovery
    ├── enums.py             # TID, IID, EMetricsId, MoveState, UCAPI_*
    ├── types.py             # Protocol classes for type hints
    ├── utils.py             # UnitConverter, helpers
    ├── units.py             # Axis, Stage, MoveHandle (Phase 2)
    ├── events.py            # AxisEvents, EventQueue, Subscription (Phase 3)
    └── camera.py            # Camera, FrameStream, Frame (Phase 4)
```

## Progress

- [x] Phase 1: Core Infrastructure
- [x] Phase 2: Async Axis Control (includes Stage)
- [x] Phase 3: Event System
- [x] Phase 4: Camera Integration
- [ ] Phase 5: Continuous Scanning Orchestrator

## Questions to Resolve

1. **Camera coordination**: Does UCAPI support triggered acquisition synced to stage position? Or do we poll position when frame arrives?
2. **Velocity limits**: What are the min/max stage velocities? Need to validate user input.
3. **Event frequency**: How often do position events fire during movement? If too slow, may need to poll.
4. **Existing code**: Keep old `driver/` or refactor in place? Recommend keeping for now, deprecate later.

## Deploy Instructions

On microscope PC, after pulling:

```bash
# Move DLLs from old location to shared location
cd C:\Users\GGG-Leica-DM6M\flakefinder\src\flakefinder
mkdir dlls
move driver\dlls\*.dll dlls\
move driver\dlls\*.cti dlls\
move driver\dlls\*.xml dlls\
```

Then test both APIs still work:
```python
# Test old driver still works
from flakefinder.driver import Microscope

# Test new leica API
from flakefinder.leica import LeicaConnection, TID
with LeicaConnection() as conn:
    print(conn.root.GetName())
```

## Next Steps

1. ~~Create the `leica/` package skeleton~~ ✓
2. ~~Implement `LeicaConnection` context manager~~ ✓
3. ~~Port over enums from existing driver~~ ✓
4. **Deploy to microscope PC and move DLLs**
5. Implement `Axis` class with sync + async methods (Phase 2)
6. Test async moves on the microscope PC
