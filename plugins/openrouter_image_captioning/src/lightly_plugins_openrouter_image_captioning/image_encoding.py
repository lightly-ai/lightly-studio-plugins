"""Encode local image files as base64 JPEG data URLs for vision model requests."""

from __future__ import annotations

import base64
import io

import PIL.Image
import PIL.ImageOps

_JPEG_MIME_PREFIX = "data:image/jpeg;base64,"


class ImageEncodingError(RuntimeError):
    """Raised when a local image cannot be read or encoded."""


def encode_image_as_data_url(
    *, file_path: str, max_edge: int, jpeg_quality: int
) -> str:
    """Read, orient, downscale and JPEG-encode an image as a base64 data URL.

    Args:
        file_path: Absolute path to the image file.
        max_edge: Maximum length of the longest edge in pixels. 0 disables resizing.
        jpeg_quality: JPEG quality between 1 and 95.

    Returns:
        A ``data:image/jpeg;base64,...`` URL usable as an OpenAI ``image_url``.

    Raises:
        ImageEncodingError: If the file cannot be opened, decoded or encoded.
    """
    try:
        with PIL.Image.open(file_path) as opened:
            # exif_transpose applies the EXIF orientation tag so portrait photos are not
            # sent to the model sideways. It returns None for images without EXIF data.
            oriented = PIL.ImageOps.exif_transpose(opened)
            # convert("RGB") also detaches the pixel buffer from the file handle, so the
            # thumbnail() call below stays valid after the context manager closes.
            image = (oriented if oriented is not None else opened).convert("RGB")
        if max_edge > 0 and max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), PIL.Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception as exc:
        raise ImageEncodingError(
            f"Could not encode image '{file_path}': {exc}"
        ) from exc
    return _JPEG_MIME_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")
