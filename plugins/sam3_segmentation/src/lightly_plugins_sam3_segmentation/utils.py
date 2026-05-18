"""Utilities for post-processing SAM3 model outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.bounding_box import BoundingBox


def _clamp_xyxy_to_xywh(
    box: Any,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert an xyxy box to (x, y, w, h), clamped to image bounds."""
    x1 = max(0, min(int(box[0]), width - 1))
    y1 = max(0, min(int(box[1]), height - 1))
    x2 = max(x1 + 1, min(int(box[2]), width))
    y2 = max(y1 + 1, min(int(box[3]), height))
    return x1, y1, x2 - x1, y2 - y1


def prepare_segmentation_entries(
    boxes: Any,
    masks: Any,
    scores: Any,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Convert SAM3 post-processed outputs to annotation-ready entries.

    Args:
        boxes: Tensor (N, 4) absolute-pixel xyxy coordinates.
        masks: Tensor (N, H, W) boolean masks.
        scores: Tensor (N,) confidence scores.
        image_size: (width, height) of the source image.

    Returns:
        List of dicts with keys 'box' (x, y, w, h), 'score' (float),
        and 'rle' (list[int] row-wise run-length encoding).
    """
    img_w, img_h = image_size
    entries = []
    for box, mask, score in zip(boxes, masks, scores):
        x, y, w, h = _clamp_xyxy_to_xywh(box, img_w, img_h)
        binary_mask: NDArray[np.int_] = mask.cpu().numpy().astype(np.int_)
        bounding_box = BoundingBox(
            xmin=float(x),
            ymin=float(y),
            xmax=float(x + w),
            ymax=float(y + h),
        )
        seg = BinaryMaskSegmentation.from_binary_mask(binary_mask, bounding_box)
        entries.append(
            {"box": (x, y, w, h), "score": float(score), "rle": seg.get_rle()}
        )
    return entries
