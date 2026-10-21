"""Utilities for post-processing SAM3 model outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.bounding_box import BoundingBox
from numpy.typing import NDArray


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
    include_rle: bool = True,
) -> list[dict[str, Any]]:
    """Convert SAM3 post-processed outputs to annotation-ready entries.

    Args:
        boxes: Tensor (N, 4) absolute-pixel xyxy coordinates.
        masks: Tensor (N, H, W) boolean masks.
        scores: Tensor (N,) confidence scores.
        image_size: (width, height) of the source image.
        include_rle: Whether to encode the masks. Set to False when only bounding
            boxes are stored, to skip the run-length encoding of every mask.

    Returns:
        List of dicts with keys 'box' (x, y, w, h), 'score' (float), and 'rle'
        (list[int] row-wise run-length encoding, None when `include_rle` is False).
        When `include_rle` is True, the box is the tight box around the mask, and
        detections with an empty mask are skipped.
    """
    img_w, img_h = image_size
    entries = []
    for box, mask, score in zip(boxes, masks, scores):
        if not include_rle:
            # SAM3 predicts the box and the mask with separate heads, so they
            # do not always agree. Without a mask, the predicted box is used.
            entries.append(
                {
                    "box": _clamp_xyxy_to_xywh(box, img_w, img_h),
                    "score": float(score),
                    "rle": None,
                }
            )
            continue

        binary_mask: NDArray[np.int_] = mask.cpu().numpy().astype(np.int_)
        mask_box = _mask_to_xywh(binary_mask=binary_mask)
        if mask_box is None:
            continue
        x, y, w, h = mask_box
        bounding_box = BoundingBox(
            xmin=float(x),
            ymin=float(y),
            xmax=float(x + w),
            ymax=float(y + h),
        )
        seg = BinaryMaskSegmentation.from_binary_mask(binary_mask, bounding_box)
        entries.append({"box": mask_box, "score": float(score), "rle": seg.get_rle()})
    return entries


def _mask_to_xywh(binary_mask: NDArray[np.int_]) -> tuple[int, int, int, int] | None:
    """Get the tight (x, y, w, h) box around the non-zero pixels of a mask.

    Returns:
        The box, or None if the mask has no non-zero pixels.
    """
    ys, xs = np.nonzero(binary_mask)
    if len(xs) == 0:
        return None
    x = int(xs.min())
    y = int(ys.min())
    return x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1
