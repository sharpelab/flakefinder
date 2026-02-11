"""Image conversion utilities for Leica SDK images.

Convention: all images in flakefinder are RGB uint8 numpy arrays.
The Leica SDK returns BGR; this module converts at the boundary so all
downstream code can assume RGB.
"""

import numpy as np

from .types import RGBImage


def sdk_image_to_numpy(image) -> RGBImage:
    """Convert a Leica SDK Image to a numpy array.

    Copies pixel data from .NET memory into a numpy array and converts
    from the SDK's BGR channel order to RGB.

    The caller is responsible for disposing the SDK image afterward.

    Args:
        image: Leica SDK Image object (must support LockPixelData/PixelData/Format).

    Returns:
        Image as numpy array (H, W, 3), dtype uint8, RGB channel order.
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

        # SDK returns BGR; convert to RGB (project convention: all images are RGB)
        bgr = np.frombuffer(bytes_array, dtype=np.uint8).reshape((height, width, -1))
        return bgr[:, :, ::-1].copy()
    finally:
        image.UnlockPixelData()
