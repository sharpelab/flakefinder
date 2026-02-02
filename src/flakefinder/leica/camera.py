"""Camera control with single-shot and streaming acquisition.

This module provides camera control for the Leica microscope using the UCAPI SDK.
Supports both single-shot capture and continuous streaming for scanning operations.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from .enums import IID, UCAPI_TID, UCAPI_IID, UCAPI_PROP
from .core import get_interface, get_interface_required, find_unit

if TYPE_CHECKING:
    from .types import Unit


@dataclass
class Frame:
    """A captured image frame with metadata."""

    image: np.ndarray
    timestamp: float  # time.monotonic() when frame was received
    position: tuple[float, float] | None = None  # (x_um, y_um) if stage provided
    frame_number: int = 0


class Camera:
    """Camera control with property management and single-shot capture.

    Usage:
        camera = Camera.from_connection(conn)

        # Configure
        camera.exposure_time = 0.05  # 50ms
        camera.binning = 2

        # Capture
        image = camera.capture()

        # Streaming (for scanning)
        with camera.stream() as stream:
            for _ in range(100):
                frame = stream.get_frame(timeout=1.0)
                process(frame.image)
    """

    def __init__(self, unit: "Unit"):
        """Initialize camera from SDK unit.

        Args:
            unit: SDK Unit object (must be UCAPI_CAMERA type).

        Raises:
            LookupError: If required interfaces not found.
        """
        self._unit = unit
        self._name = unit.GetName()

        # Register UCAPI extensions (required for camera interfaces)
        from LeicaMicrosystems.HardwareModel import Extensions
        try:
            Extensions.ExUCAPI.Register()
        except Exception:
            pass  # May already be registered

        # Camera requires initialization
        unit.Init()

        # Get acquisition interface
        self._acquisition = get_interface_required(unit, UCAPI_IID.IID_IMAGE_ACQUISITION)

        # Get properties interface
        self._properties = get_interface_required(unit, IID.IID_PROPERTIES)

        # Set up acquisition context (lazy init on first use)
        self._context = None
        self._current_image: np.ndarray | None = None
        self._image_ready = threading.Event()

        # Cache property value objects for fast access
        self._prop_cache: dict[int, object] = {}

        # Set reasonable defaults
        self._init_defaults()

    @classmethod
    def from_connection(cls, conn: "LeicaConnection") -> "Camera":
        """Create Camera from a LeicaConnection.

        Args:
            conn: Active LeicaConnection.

        Returns:
            Camera instance.

        Raises:
            LookupError: If camera unit not found.
        """
        camera_unit = find_unit(conn.root, UCAPI_TID.UCAPI_CAMERA)
        if camera_unit is None:
            raise LookupError("Camera unit not found")
        return cls(camera_unit)

    def _init_defaults(self) -> None:
        """Set default camera settings."""
        self.auto_brightness = False
        self.exposure_time = 0.01  # 10ms
        self.gain = 1.0
        self.gain_rgb = (1.0, 1.0, 1.0)
        self.saturation = 100  # int, not float
        self.gamma = 1.0
        self.binning = 1  # 0=1x1, 1=2x2, 2=4x4

    def _get_property(self, prop_id: int):
        """Get a property value object, with caching."""
        if prop_id not in self._prop_cache:
            prop = self._properties.FindProperty(prop_id)
            if prop is not None:
                self._prop_cache[prop_id] = prop.GetValue()
            else:
                self._prop_cache[prop_id] = None
        return self._prop_cache[prop_id]

    # --- Properties ---

    @property
    def name(self) -> str:
        """Camera name from SDK."""
        return self._name

    @property
    def exposure_time(self) -> float:
        """Exposure time in seconds."""
        prop = self._get_property(UCAPI_PROP.PROP_EXPOSURE_TIME)
        return prop.GetValue() if prop else 0.0

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_EXPOSURE_TIME)
        if prop:
            prop.SetValue(value)

    @property
    def gain(self) -> float:
        """Overall gain multiplier."""
        prop = self._get_property(UCAPI_PROP.PROP_GAIN)
        return prop.GetValue() if prop else 1.0

    @gain.setter
    def gain(self, value: float) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_GAIN)
        if prop:
            prop.SetValue(value)

    @property
    def gain_rgb(self) -> tuple[float, float, float]:
        """Per-channel gain (red, green, blue)."""
        r = self._get_property(UCAPI_PROP.PROP_GAIN_RED)
        g = self._get_property(UCAPI_PROP.PROP_GAIN_GREEN)
        b = self._get_property(UCAPI_PROP.PROP_GAIN_BLUE)
        return (
            r.GetValue() if r else 1.0,
            g.GetValue() if g else 1.0,
            b.GetValue() if b else 1.0,
        )

    @gain_rgb.setter
    def gain_rgb(self, value: tuple[float, float, float]) -> None:
        r, g, b = value
        prop_r = self._get_property(UCAPI_PROP.PROP_GAIN_RED)
        prop_g = self._get_property(UCAPI_PROP.PROP_GAIN_GREEN)
        prop_b = self._get_property(UCAPI_PROP.PROP_GAIN_BLUE)
        if prop_r:
            prop_r.SetValue(r)
        if prop_g:
            prop_g.SetValue(g)
        if prop_b:
            prop_b.SetValue(b)

    @property
    def auto_brightness(self) -> bool:
        """Auto brightness/exposure enabled."""
        prop = self._get_property(UCAPI_PROP.PROP_AUTO_BRIGHTNESS_ENABLED)
        return prop.GetValue() if prop else False

    @auto_brightness.setter
    def auto_brightness(self, value: bool) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_AUTO_BRIGHTNESS_ENABLED)
        if prop:
            prop.SetValue(value)

    @property
    def saturation(self) -> int:
        """Color saturation (0-100+)."""
        prop = self._get_property(UCAPI_PROP.PROP_COLOUR_SATURATION)
        return prop.GetValue() if prop else 100

    @saturation.setter
    def saturation(self, value: int) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_COLOUR_SATURATION)
        if prop:
            prop.SetValue(int(value))

    @property
    def gamma(self) -> float:
        """Gamma correction level."""
        prop = self._get_property(UCAPI_PROP.PROP_GAMMA_LEVEL)
        return prop.GetValue() if prop else 1.0

    @gamma.setter
    def gamma(self, value: float) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_GAMMA_LEVEL)
        if prop:
            prop.SetValue(value)

    @property
    def binning(self) -> int:
        """Binning level (0=1x1, 1=2x2, 2=4x4)."""
        prop = self._get_property(UCAPI_PROP.PROP_BINNING_LEVEL)
        return prop.GetIndex() if prop else 0

    @binning.setter
    def binning(self, value: int) -> None:
        if value not in (0, 1, 2):
            raise ValueError(f"Invalid binning level: {value}. Must be 0, 1, or 2.")
        prop = self._get_property(UCAPI_PROP.PROP_BINNING_LEVEL)
        if prop:
            prop.SetIndex(value)

    # --- Acquisition ---

    def _ensure_context(self) -> None:
        """Ensure acquisition context is initialized."""
        if self._context is not None:
            return

        # Import .NET types for context creation
        from LeicaMicrosystems.HardwareModel import Extensions

        self._context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
        self._context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(
            self._on_image_acquired
        )

    def _on_image_acquired(self, image) -> None:
        """Callback when image is acquired."""
        self._current_image = self._image_to_numpy(image)
        image.Dispose()
        self._image_ready.set()

    @staticmethod
    def _image_to_numpy(image) -> np.ndarray:
        """Convert SDK Image to numpy array."""
        import System

        image.LockPixelData()
        try:
            fmt = image.Format()
            width = fmt.Width()
            height = fmt.Height()
            buffer_size = fmt.PixelBufferSize()

            # Copy from .NET memory to Python
            bytes_array = System.Array[System.Byte](buffer_size)
            System.Runtime.InteropServices.Marshal.Copy(
                image.PixelData(), bytes_array, 0, buffer_size
            )

            # Convert to numpy
            return np.frombuffer(bytes_array, dtype=np.uint8).reshape((height, width, -1))
        finally:
            image.UnlockPixelData()

    def capture(self) -> np.ndarray:
        """Capture a single image (blocking).

        Returns:
            Image as numpy array (H, W, C).
        """
        self._ensure_context()
        self._image_ready.clear()
        self._acquisition.Acquire(self._context, None)
        return self._current_image

    def stream(self, stage: "Stage | None" = None) -> "FrameStream":
        """Start continuous frame acquisition.

        Args:
            stage: Optional Stage for position tagging.

        Returns:
            FrameStream context manager.

        Usage:
            with camera.stream() as stream:
                while scanning:
                    frame = stream.get_frame(timeout=1.0)
        """
        return FrameStream(self, stage)

    def dispose(self) -> None:
        """Release camera resources."""
        if self._context is not None:
            self._context.IsCancelled = True
            try:
                self._context.Dispose()
            except Exception:
                pass
            self._context = None
        if self._unit is not None:
            try:
                self._unit.Dispose()
            except Exception:
                pass
            self._unit = None

    def __del__(self):
        self.dispose()

    def __repr__(self) -> str:
        return f"Camera({self._name}, exp={self.exposure_time:.3f}s, bin={self.binning})"


class FrameStream:
    """Continuous frame acquisition running in background thread.

    Wraps AcquireContinuous() which is blocking, running it in a background
    thread and providing a queue-based interface for frame retrieval.

    Usage:
        with camera.stream() as stream:
            stream.start()
            while not done:
                frame = stream.get_frame(timeout=0.5)
                if frame:
                    process(frame.image, frame.position)
    """

    def __init__(
        self,
        camera: Camera,
        stage: "Stage | None" = None,
        max_buffer: int = 100,
    ):
        """Initialize frame stream.

        Args:
            camera: Camera to stream from.
            stage: Optional Stage for position tagging.
            max_buffer: Maximum frames to buffer (older dropped if full).
        """
        self._camera = camera
        self._stage = stage
        self._max_buffer = max_buffer

        # Thread management
        self._thread: threading.Thread | None = None
        self._running = False
        self._context = None

        # Frame queue
        self._frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=max_buffer)

        # Stats
        self._frames_captured = 0
        self._frames_dropped = 0
        self._start_time: float | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start background acquisition thread."""
        if self._running:
            return

        self._running = True
        self._frames_captured = 0
        self._frames_dropped = 0
        self._start_time = time.monotonic()

        self._thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop acquisition.

        Args:
            timeout: Max seconds to wait for thread to finish.
        """
        if not self._running:
            return

        self._running = False

        # Signal cancellation
        if self._context is not None:
            self._context.IsCancelled = True

        # Wait for thread
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _acquire_loop(self) -> None:
        """Background thread: run continuous acquisition."""
        from LeicaMicrosystems.HardwareModel import Extensions

        # Create context for this thread
        self._context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
        self._context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(
            self._on_frame
        )

        try:
            # This blocks until IsCancelled is set
            self._camera._acquisition.AcquireContinuous(self._context, None)
        except Exception:
            pass  # Expected when cancelled
        finally:
            try:
                self._context.Dispose()
            except Exception:
                pass
            self._context = None

    def _on_frame(self, image) -> None:
        """Callback for each frame (runs on .NET thread)."""
        timestamp = time.monotonic()

        # Get position if stage provided
        position = None
        if self._stage is not None:
            try:
                position = self._stage.position_um
            except Exception:
                pass

        # Convert image
        try:
            img_array = Camera._image_to_numpy(image)
        finally:
            image.Dispose()

        with self._lock:
            frame_num = self._frames_captured
            self._frames_captured += 1

        frame = Frame(
            image=img_array,
            timestamp=timestamp,
            position=position,
            frame_number=frame_num,
        )

        # Queue frame (drop oldest if full)
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()  # Drop oldest
                self._frame_queue.put_nowait(frame)
                with self._lock:
                    self._frames_dropped += 1
            except queue.Empty:
                pass

    def get_frame(self, timeout: float | None = None) -> Frame | None:
        """Get next frame from buffer.

        Args:
            timeout: Max seconds to wait (None = forever).

        Returns:
            Frame or None if timeout.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[Frame]:
        """Get all buffered frames, clearing the buffer."""
        frames = []
        while True:
            try:
                frames.append(self._frame_queue.get_nowait())
            except queue.Empty:
                break
        return frames

    @property
    def frames_captured(self) -> int:
        """Total frames captured since start."""
        with self._lock:
            return self._frames_captured

    @property
    def frames_dropped(self) -> int:
        """Frames dropped due to full buffer."""
        with self._lock:
            return self._frames_dropped

    @property
    def frame_rate(self) -> float:
        """Average frame rate (fps) since start."""
        with self._lock:
            if self._start_time is None or self._frames_captured == 0:
                return 0.0
            elapsed = time.monotonic() - self._start_time
            return self._frames_captured / elapsed if elapsed > 0 else 0.0

    @property
    def pending(self) -> int:
        """Number of frames waiting in buffer."""
        return self._frame_queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if streaming is active."""
        return self._running

    def __enter__(self) -> "FrameStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"FrameStream(captured={self.frames_captured}, rate={self.frame_rate:.1f}fps)"
