"""Utilities for post-processing YOLOE model outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.bounding_box import BoundingBox
from numpy.typing import NDArray


def clamp_xywh(
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Convert a center-form box to (x, y, w, h), clamped to the image bounds.

    YOLOE boxes can extend a pixel or two outside the image on edge detections, which would
    otherwise produce a bounding box that does not bound the mask.

    Args:
        x_center: Box center along x, in pixels.
        y_center: Box center along y, in pixels.
        w: Box width in pixels.
        h: Box height in pixels.
        image_size: (width, height) of the source image.

    Returns:
        The box as (x, y, width, height) with a width and height of at least 1.
    """
    img_w, img_h = image_size
    x1 = max(0, min(round(x_center - w / 2), img_w - 1))
    y1 = max(0, min(round(y_center - h / 2), img_h - 1))
    x2 = max(x1 + 1, min(round(x_center + w / 2), img_w))
    y2 = max(y1 + 1, min(round(y_center + h / 2), img_h))
    return x1, y1, x2 - x1, y2 - y1


def encode_mask_rle(mask: Any, box: tuple[int, int, int, int]) -> list[int]:
    """Return the row-wise run-length encoding of a binary mask.

    The mask must already match the source image resolution, which YOLOE does when it is run
    with `retina_masks=True`.

    Args:
        mask: Tensor (H, W) with values in {0, 1}.
        box: The (x, y, width, height) box bounding the mask.

    Returns:
        The mask as a row-wise run-length encoding.
    """
    x, y, w, h = box
    binary_mask: NDArray[np.int_] = mask.cpu().numpy().astype(np.int_)
    bounding_box = BoundingBox(
        xmin=float(x),
        ymin=float(y),
        xmax=float(x + w),
        ymax=float(y + h),
    )
    return BinaryMaskSegmentation.from_binary_mask(binary_mask, bounding_box).get_rle()
