"""Camera control with single-shot and streaming acquisition.

This module provides camera control for the Leica microscope using the UCAPI SDK.
Supports both single-shot capture and continuous streaming for scanning operations.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from flakefinder.types import ColourMatrix, GainRGB, Point2F, RGBImage

from .core import LeicaConnection, find_unit, get_interface_required
from .enums import IID, UCAPI_CCM, UCAPI_IID, UCAPI_PROP, UCAPI_TID
from .types import Image as SdkImage
from .types import Unit
from .units import Stage

# App-level ColourMatrix <-> SDK enum index, translated only at this boundary.
_COLOUR_MATRIX_TO_CCM: dict[ColourMatrix, UCAPI_CCM] = {
    ColourMatrix.IDENTITY: UCAPI_CCM.IDENTITY,
    ColourMatrix.CCM_4500K: UCAPI_CCM.K4500,
    ColourMatrix.CCM_5800K: UCAPI_CCM.K5800,
    ColourMatrix.CCM_6600K: UCAPI_CCM.K6600,
}
_CCM_TO_COLOUR_MATRIX: dict[UCAPI_CCM, ColourMatrix] = {v: k for k, v in _COLOUR_MATRIX_TO_CCM.items()}


@dataclass
class Frame:
    """A captured image frame with metadata."""

    image: RGBImage
    timestamp: float  # time.monotonic() when frame was received
    position: Point2F | None = None  # (x_um, y_um) if stage provided
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

    def __init__(self, unit: Unit):
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

        with contextlib.suppress(Exception):  # May already be registered
            Extensions.ExUCAPI.Register()

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

        self._init_defaults()

    def _init_defaults(self) -> None:
        """Override SDK defaults for settings that unit.Init() resets.

        unit.Init() resets binning, gamma, saturation, and white balance
        to SDK defaults (2x2, 0.45, 100, R1.93/G1.00/B1.94).
        Exposure and gain are NOT reset by the SDK.

        colour_matrix (CCM selector) is pinned explicitly so the colour
        space is always known regardless of what LAS X or a prior session
        left behind.
        """
        self.binning = 2  # 3x3 (SDK default: 1 / 2x2)
        self.gamma = 1.0  # linear (SDK default: 0.45)
        self.saturation = 100  # same as SDK, but explicit
        self.gain_rgb = (1.0, 1.0, 1.0)  # neutral (SDK default: R1.93/G1.00/B1.94)
        self.colour_matrix = ColourMatrix.IDENTITY  # raw channels, no CCM mixing (5800K = legacy scan space)

    @classmethod
    def from_connection(cls, conn: LeicaConnection) -> Camera:
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
    def gain_rgb(self) -> GainRGB:
        """Per-channel gain (red, green, blue)."""
        r = self._get_property(UCAPI_PROP.PROP_GAIN_RED)
        g = self._get_property(UCAPI_PROP.PROP_GAIN_GREEN)
        b = self._get_property(UCAPI_PROP.PROP_GAIN_BLUE)
        return GainRGB(
            red=r.GetValue() if r else 1.0,
            green=g.GetValue() if g else 1.0,
            blue=b.GetValue() if b else 1.0,
        )

    @gain_rgb.setter
    def gain_rgb(self, value: GainRGB | tuple[float, float, float]) -> None:
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
        """Binning level (0=1x1, 1=2x2, 2=3x3)."""
        prop = self._get_property(UCAPI_PROP.PROP_BINNING_LEVEL)
        return prop.GetIndex() if prop else 0

    @binning.setter
    def binning(self, value: int) -> None:
        if value not in (0, 1, 2):
            raise ValueError(f"Invalid binning level: {value}. Must be 0, 1, or 2.")
        prop = self._get_property(UCAPI_PROP.PROP_BINNING_LEVEL)
        if prop:
            prop.SetIndex(value)

    @property
    def colour_temperature(self) -> int | None:
        """Raw CCM selection index (SDK enum), or None if unavailable.

        Selects the host-side colour correction matrix applied after white
        balance. K5C options (per driver logs): UserDefinedMatrix, 4500K,
        5800K, 6600K [Standard]; SDK default is 5800K (index 2). The CCM is
        what couples WB channel gains into a 3x3 mixing matrix.

        SDK-level access for probe/calibration tooling — application code
        should use ``colour_matrix`` instead.
        """
        prop = self._get_property(UCAPI_PROP.PROP_COLOUR_TEMPERATURE)
        return prop.GetIndex() if prop else None

    @colour_temperature.setter
    def colour_temperature(self, index: int) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_COLOUR_TEMPERATURE)
        if prop:
            prop.SetIndex(int(index))

    @property
    def colour_matrix(self) -> ColourMatrix | None:
        """Colour-correction matrix mode, or None if unavailable/unknown.

        App-level view of ``colour_temperature`` — translates the SDK enum
        index at this boundary (same pattern as gain_rgb/GainRGB).
        """
        index = self.colour_temperature
        if index is None:
            return None
        try:
            return _CCM_TO_COLOUR_MATRIX[UCAPI_CCM(index)]
        except ValueError:
            return None

    @colour_matrix.setter
    def colour_matrix(self, mode: ColourMatrix | str) -> None:
        self.colour_temperature = _COLOUR_MATRIX_TO_CCM[ColourMatrix(mode)]

    @property
    def pixel_type(self) -> int | None:
        """Delivered pixel type index (enum), or None if unavailable.

        K5C default index 0 = PIXEL_TYPE_BGR. Other options (if exposed)
        may include raw Bayer / mono delivery which bypasses the host
        colour pipeline. See UCAPI_PIXEL_TYPE for option codes.
        """
        prop = self._get_property(UCAPI_PROP.PROP_PIXEL_TYPE)
        return prop.GetIndex() if prop else None

    @pixel_type.setter
    def pixel_type(self, index: int) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_PIXEL_TYPE)
        if prop:
            prop.SetIndex(int(index))

    @property
    def pixel_depth(self) -> int | None:
        """Pixel depth index (enum), or None if unavailable.

        K5C default index 0 (8 bits/channel). The sensor ADC is 12-bit;
        the probe determines whether a 12-bit option is exposed.
        """
        prop = self._get_property(UCAPI_PROP.PROP_PIXEL_DEPTH)
        return prop.GetIndex() if prop else None

    @pixel_depth.setter
    def pixel_depth(self, index: int) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_PIXEL_DEPTH)
        if prop:
            prop.SetIndex(int(index))

    @property
    def sharpening_enabled(self) -> bool | None:
        """Host-side 5x5 sharpening filter enabled, or None if unavailable."""
        prop = self._get_property(UCAPI_PROP.PROP_SHARPENING_ENABLED)
        return bool(prop.GetValue()) if prop else None

    @sharpening_enabled.setter
    def sharpening_enabled(self, value: bool) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_SHARPENING_ENABLED)
        if prop:
            prop.SetValue(bool(value))

    @property
    def trigger_mode(self) -> int:
        """Trigger mode index (0=CONTINUOUS, 1=SOFT, etc.)."""
        prop = self._get_property(UCAPI_PROP.PROP_IMAGE_TRIGGER_MODE)
        return prop.GetIndex() if prop else 0

    @trigger_mode.setter
    def trigger_mode(self, value: int) -> None:
        prop = self._get_property(UCAPI_PROP.PROP_IMAGE_TRIGGER_MODE)
        if prop:
            prop.SetIndex(value)

    # --- Read-only Metadata Properties ---

    @property
    def frame_size_px(self) -> tuple[int, int]:
        """Frame (width, height) in pixels after binning."""
        w = self._get_property(UCAPI_PROP.PROP_LOGICAL_XRESOLUTION)
        h = self._get_property(UCAPI_PROP.PROP_LOGICAL_YRESOLUTION)
        return (
            int(w.GetValue()) if w else 0,
            int(h.GetValue()) if h else 0,
        )

    @property
    def sensor_size_px(self) -> tuple[int, int]:
        """Physical sensor (width, height) in pixels."""
        w = self._get_property(UCAPI_PROP.PROP_SENSOR_XRESOLUTION)
        h = self._get_property(UCAPI_PROP.PROP_SENSOR_YRESOLUTION)
        return (
            int(w.GetValue()) if w else 0,
            int(h.GetValue()) if h else 0,
        )

    @property
    def pixel_size_um(self) -> tuple[float, float]:
        """Logical (x, y) pixel size in µm (accounts for binning).

        Note: This is the SDK's logical pixel size, which may not account
        for objective magnification. For sample-plane pixel size, divide
        by objective magnification.
        """
        x = self._get_property(UCAPI_PROP.PROP_LOGICAL_PIXEL_XSIZE)
        y = self._get_property(UCAPI_PROP.PROP_LOGICAL_PIXEL_YSIZE)
        # SDK returns meters, convert to µm
        return (
            x.GetValue() * 1e6 if x else 0.0,
            y.GetValue() * 1e6 if y else 0.0,
        )

    @property
    def physical_pixel_size_um(self) -> tuple[float, float]:
        """Physical sensor (x, y) pixel size in µm (before binning)."""
        x = self._get_property(UCAPI_PROP.PROP_PHYSICAL_PIXEL_XSIZE)
        y = self._get_property(UCAPI_PROP.PROP_PHYSICAL_PIXEL_YSIZE)
        # SDK returns meters, convert to µm
        return (
            x.GetValue() * 1e6 if x else 0.0,
            y.GetValue() * 1e6 if y else 0.0,
        )

    @property
    def readout_time_s(self) -> float | None:
        """Image readout time in seconds, or None if unavailable."""
        prop = self._get_property(UCAPI_PROP.PROP_IMAGE_READOUT_TIME)
        return prop.GetValue() if prop else None

    # --- Acquisition ---

    def _ensure_context(self) -> None:
        """Ensure acquisition context is initialized."""
        if self._context is not None:
            return

        # Import .NET types for context creation
        from LeicaMicrosystems.HardwareModel import Extensions

        self._context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
        self._context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(self._on_image_acquired)

    def _on_image_acquired(self, image) -> None:
        """Callback when image is acquired."""
        self._current_image = self._image_to_numpy(image)
        image.Dispose()
        self._image_ready.set()

    @staticmethod
    def _image_to_numpy(image) -> RGBImage:
        """Convert SDK Image to RGB numpy array.

        Delegates to flakefinder.image_utils.sdk_image_to_numpy.
        Kept for backward compatibility with callers using Camera._image_to_numpy.
        """
        from flakefinder.image_utils import sdk_image_to_numpy

        return sdk_image_to_numpy(image)

    def capture_maybe(self) -> np.ndarray | None:
        """Capture a single image (blocking), returning None on failure.

        Returns:
            Image as numpy array (H, W, C), or None if acquisition failed.
        """
        self._ensure_context()
        self._image_ready.clear()
        self._acquisition.Acquire(self._context, None)
        return self._current_image

    def capture(self) -> np.ndarray:
        """Capture a single image (blocking).

        Returns:
            Image as numpy array (H, W, C).

        Raises:
            RuntimeError: If acquisition failed to produce an image.
        """
        image = self.capture_maybe()
        if image is None:
            raise RuntimeError("Capture failed: no image acquired")
        return image

    def stream(self, stage: Stage | None = None) -> FrameStream:
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

    def deferred_stream(self, max_frames: int = 1000) -> DeferredFrameStream:
        """Start deferred frame acquisition (keeps images in .NET memory).

        Higher fps by avoiding per-frame numpy conversion. Convert all frames
        after acquisition stops with get_all_frames().

        Args:
            max_frames: Maximum frames to buffer.

        Returns:
            DeferredFrameStream context manager.
        """
        return DeferredFrameStream(self, max_frames)

    def dispose(self) -> None:
        """Release camera resources.

        Note: Only disposes the context we created, NOT the unit.
        The unit is owned by LeicaConnection and disposed when it closes.
        """
        if self._context is not None:
            self._context.IsCancelled = True
            with contextlib.suppress(Exception):
                self._context.Dispose()
            self._context = None
        # Don't dispose _unit - it's owned by LeicaConnection

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
        stage: Stage | None = None,
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
        self._context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(self._on_frame)

        try:
            # This blocks until IsCancelled is set
            self._camera._acquisition.AcquireContinuous(self._context, None)
        except Exception:
            pass  # Expected when cancelled
        finally:
            with contextlib.suppress(Exception):
                self._context.Dispose()
            self._context = None

    def _on_frame(self, image) -> None:
        """Callback for each frame (runs on .NET thread)."""
        timestamp = time.monotonic()

        # Get position if stage provided
        position = None
        if self._stage is not None:
            with contextlib.suppress(Exception):
                position = self._stage.position_um

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

    def __enter__(self) -> FrameStream:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"FrameStream(captured={self.frames_captured}, rate={self.frame_rate:.1f}fps)"


class DeferredFrameStream:
    """Deferred frame acquisition - keeps images in .NET memory until retrieval.

    This avoids the Marshal.Copy overhead during acquisition, potentially
    achieving higher frame rates. Images are converted to numpy only when
    get_all_frames() is called after acquisition stops.

    Usage:
        with camera.deferred_stream() as stream:
            # Do scanning - images accumulate in .NET memory
            stage.move_async(...)
            handle.wait()

        # After context exits, convert and process
        for timestamp, image in stream.get_all_frames():
            save(image)
    """

    def __init__(self, camera: Camera, max_frames: int = 1000):
        """Initialize deferred stream.

        Args:
            camera: Camera to stream from.
            max_frames: Maximum frames to buffer (older dropped if exceeded).
        """
        self._camera = camera
        self._max_frames = max_frames

        # Thread management
        self._thread: threading.Thread | None = None
        self._running = False
        self._context = None

        # Store raw .NET images with timestamps
        self._raw_frames: list[tuple[float, SdkImage]] = []  # (timestamp, .NET Image)
        self._lock = threading.Lock()

        # Stats
        self._frames_captured = 0
        self._frames_dropped = 0
        self._start_time: float | None = None

    def start(self) -> None:
        """Start background acquisition thread."""
        if self._running:
            return

        self._running = True
        self._frames_captured = 0
        self._frames_dropped = 0
        self._raw_frames = []
        self._start_time = time.monotonic()

        self._thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop acquisition."""
        if not self._running:
            return

        self._running = False

        if self._context is not None:
            self._context.IsCancelled = True

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _acquire_loop(self) -> None:
        """Background thread: run continuous acquisition."""
        from LeicaMicrosystems.HardwareModel import Extensions

        self._context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
        self._context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(self._on_frame)

        try:
            self._camera._acquisition.AcquireContinuous(self._context, None)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                self._context.Dispose()
            self._context = None

    def _on_frame(self, image) -> None:
        """Callback for each frame - keep in .NET memory."""
        timestamp = time.monotonic()

        with self._lock:
            if len(self._raw_frames) >= self._max_frames:
                # Drop oldest
                old_ts, old_img = self._raw_frames.pop(0)
                with contextlib.suppress(Exception):
                    old_img.Dispose()
                self._frames_dropped += 1

            # Keep reference to .NET image (don't dispose yet)
            self._raw_frames.append((timestamp, image))
            self._frames_captured += 1

    def get_all_frames(self) -> list[tuple[float, np.ndarray]]:
        """Convert all captured frames to numpy arrays.

        Call this after stopping acquisition. Disposes .NET images after conversion.

        Returns:
            List of (timestamp, numpy_array) tuples.
        """
        results = []
        with self._lock:
            for timestamp, image in self._raw_frames:
                try:
                    arr = Camera._image_to_numpy(image)
                    results.append((timestamp, arr))
                finally:
                    with contextlib.suppress(Exception):
                        image.Dispose()
            self._raw_frames = []
        return results

    @property
    def frames_captured(self) -> int:
        """Total frames captured since start."""
        with self._lock:
            return self._frames_captured

    @property
    def frames_dropped(self) -> int:
        """Frames dropped due to buffer full."""
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

    def __enter__(self) -> DeferredFrameStream:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
