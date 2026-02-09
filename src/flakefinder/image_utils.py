"""Image conversion utilities for Leica SDK images."""

import numpy as np


def sdk_image_to_numpy(image) -> np.ndarray:
    """Convert a Leica SDK Image to a numpy array.

    Copies pixel data from .NET memory into a numpy array.
    The caller is responsible for disposing the SDK image afterward.

    Args:
        image: Leica SDK Image object (must support LockPixelData/PixelData/Format).

    Returns:
        Image as numpy array (H, W, C), dtype uint8.
    """
    import System

    image.LockPixelData()
    try:
        fmt = image.Format()
        width = fmt.Width()
        height = fmt.Height()
        buffer_size = fmt.PixelBufferSize()

        # Copy from .NET memory to Python
        bytes_array = System.Array[System.Byte](buffer_size)
        System.Runtime.InteropServices.Marshal.Copy(image.PixelData(), bytes_array, 0, buffer_size)

        # Convert to numpy
        return np.frombuffer(bytes_array, dtype=np.uint8).reshape((height, width, -1))
    finally:
        image.UnlockPixelData()
