"""YOLOE inference operator for open vocabulary object detection auto-labeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from lightly_studio.models.annotation.annotation_base import (
    AnnotationCreate,
    AnnotationType,
)
from lightly_studio.models.annotation_label import AnnotationLabelCreate
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.plugins.parameter import (
    BaseParameter,
    BoolParameter,
    FloatParameter,
    StringParameter,
    TableParameter,
)
from lightly_studio.resolvers import (
    annotation_label_resolver,
    annotation_resolver,
    collection_resolver,
    image_resolver,
)
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter
from sqlmodel import Session
from ultralytics import YOLO  # type: ignore[attr-defined]

from lightly_plugins_yoloe_object_detection.utils import clamp_xywh, encode_mask_rle

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yoloe-26n-seg.pt"
DEFAULT_CONFIDENCE = 0.25

PARAM_MODEL = "model_path"
PARAM_CONFIDENCE = "confidence"
PARAM_CLASSES = "classes"
PARAM_INSTANCE_SEGMENTATION = "instance_segmentation"
PARAM_ANNOTATION_SOURCE = "annotation_source"

COLUMN_CLASS_NAME = "class_name"

_WRITE_BATCH_SIZE = 100


@dataclass
class YoloEObjectDetectionOperator(BaseOperator):
    """Runs YOLOE inference to auto-label images with boxes or segmentation masks."""

    name: str = "YOLOE Open Vocabulary Object Detection"
    description: str = (
        "Runs YOLOE inference and adds bounding box or instance segmentation "
        "annotations to unlabeled images."
    )

    @property
    def parameters(self) -> list[BaseParameter]:
        """Return the list of parameters this operator expects."""
        return [
            StringParameter(
                name=PARAM_MODEL,
                required=True,
                default=DEFAULT_MODEL,
                description=(
                    "YOLOE model weights path or Ultralytics model name "
                    "(e.g. yoloe-26n-seg.pt)."
                ),
            ),
            FloatParameter(
                name=PARAM_CONFIDENCE,
                required=True,
                default=DEFAULT_CONFIDENCE,
                description="Minimum confidence threshold for keeping a prediction.",
            ),
            TableParameter(
                name=PARAM_CLASSES,
                required=True,
                description="Open vocabulary classes to detect. One class per row.",
                columns=[
                    StringParameter(
                        name=COLUMN_CLASS_NAME,
                        description="Class name to detect.",
                    ),
                ],
                default=[{COLUMN_CLASS_NAME: "person"}],
            ),
            BoolParameter(
                name=PARAM_INSTANCE_SEGMENTATION,
                required=False,
                default=False,
                description=(
                    "Store instance segmentation masks instead of bounding boxes."
                ),
            ),
            StringParameter(
                name=PARAM_ANNOTATION_SOURCE,
                required=False,
                description=(
                    "Target annotation source name where predictions will be stored. "
                    "Defaults to yoloe_auto_label__{model_path}."
                ),
            ),
        ]

    @property
    def supported_scopes(self) -> list[OperatorScope]:
        """Return the list of scopes this operator can be triggered from."""
        return [OperatorScope.IMAGE]

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        """Execute the operator with the given parameters."""
        model_path = str(parameters.get(PARAM_MODEL, DEFAULT_MODEL))
        _annotation_source = parameters.get(PARAM_ANNOTATION_SOURCE)
        collection_name = (
            str(_annotation_source).strip()
            if _annotation_source is not None and str(_annotation_source).strip()
            else f"yoloe_auto_label__{model_path}"
        )
        confidence = float(parameters.get(PARAM_CONFIDENCE, DEFAULT_CONFIDENCE))
        instance_segmentation = bool(parameters.get(PARAM_INSTANCE_SEGMENTATION, False))

        if not 0.0 <= confidence <= 1.0:
            return OperatorResult(
                success=False,
                message="confidence must be between 0 and 1",
            )

        class_names, class_names_error = _parse_class_names(
            rows=parameters.get(PARAM_CLASSES)
        )
        if class_names_error is not None:
            return OperatorResult(success=False, message=class_names_error)

        try:
            model = YOLO(model_path)
        except Exception as e:
            logger.error("Failed to load YOLOE model '%s': %s", model_path, e)
            return OperatorResult(
                success=False,
                message=f"Failed to load YOLOE model '{model_path}': {e}",
            )

        try:
            model.set_classes(class_names)
        except Exception as e:
            logger.error("Failed to set classes on model '%s': %s", model_path, e)
            return OperatorResult(
                success=False,
                message=(
                    f"Failed to set classes on model '{model_path}': {e}. "
                    "Make sure it is an open vocabulary YOLOE checkpoint."
                ),
            )

        label_map = _get_or_create_label_map(
            session=session,
            root_collection_id=context.collection_id,
            class_map=model.names,
        )

        context_filter = None
        if context.context_filter:
            if isinstance(context.context_filter, SampleFilter):
                context_filter = ImageFilter(sample_filter=context.context_filter)
            elif isinstance(context.context_filter, ImageFilter):
                context_filter = context.context_filter

        samples_result = image_resolver.get_all_by_collection_id(
            session=session,
            collection_id=context.collection_id,
            filters=context_filter,
        )
        samples = list(samples_result.samples)
        if not samples:
            return OperatorResult(
                success=True,
                message="No samples found for current view.",
            )

        annotation_type = (
            AnnotationType.SEGMENTATION_MASK
            if instance_segmentation
            else AnnotationType.OBJECT_DETECTION
        )

        annotations_to_create: list[AnnotationCreate] = []
        total_annotations_created = 0
        for i, image_entry in enumerate(samples, start=1):
            try:
                results = model(
                    image_entry.file_path_abs,
                    conf=confidence,
                    verbose=False,
                    # Masks are letterboxed to the model input size by default. Ask for
                    # masks at the source resolution so they can be stored as-is.
                    retina_masks=instance_segmentation,
                )[0]
            except Exception as e:
                logger.error(
                    "Failed to run inference on '%s': %s",
                    image_entry.file_path_abs,
                    e,
                )
                return OperatorResult(
                    success=False,
                    message=f"Failed to run inference on '{image_entry.file_path_abs}': {e}",
                )

            image_size = (image_entry.width, image_entry.height)
            for detection_index, box in enumerate(results.boxes):
                category_id = int(box.cls)
                label_id = label_map.get(category_id)
                if label_id is None:
                    continue
                x_center, y_center, w, h = box.xywh[0].tolist()
                x, y, width, height = clamp_xywh(
                    x_center=x_center,
                    y_center=y_center,
                    w=w,
                    h=h,
                    image_size=image_size,
                )
                segmentation_mask = None
                if instance_segmentation:
                    if results.masks is None:
                        return OperatorResult(
                            success=False,
                            message=(
                                f"Model '{model_path}' returned no segmentation masks. "
                                "Use a YOLOE segmentation checkpoint, or untick "
                                "instance segmentation to store bounding boxes."
                            ),
                        )
                    segmentation_mask = encode_mask_rle(
                        mask=results.masks.data[detection_index],
                        box=(x, y, width, height),
                    )
                annotations_to_create.append(
                    AnnotationCreate(
                        annotation_label_id=label_id,
                        annotation_type=annotation_type,
                        parent_sample_id=image_entry.sample_id,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        confidence=float(box.conf),
                        segmentation_mask=segmentation_mask,
                    )
                )

            if i % _WRITE_BATCH_SIZE == 0 and annotations_to_create:
                created = annotation_resolver.create_many(
                    session=session,
                    parent_collection_id=context.collection_id,
                    annotations=annotations_to_create,
                    collection_name=collection_name,
                )
                total_annotations_created += len(created)
                annotations_to_create = []

        if annotations_to_create:
            created = annotation_resolver.create_many(
                session=session,
                parent_collection_id=context.collection_id,
                annotations=annotations_to_create,
                collection_name=collection_name,
            )
            total_annotations_created += len(created)

        return OperatorResult(
            success=True,
            message=f"Auto-labeled {len(samples)} samples with {total_annotations_created} annotations.",
        )


def _parse_class_names(rows: Any) -> tuple[list[str], str | None]:
    """Read the class names from the rows of the classes table parameter.

    The rows are not validated against the declared parameters before they reach the operator,
    so they are read defensively.

    Args:
        rows: The value of the classes table parameter.

    Returns:
        A tuple of the class names and an error message, the latter being None on success.
    """
    if not isinstance(rows, list):
        return [], "classes parameter is required and cannot be empty."

    class_names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        class_name = str(row.get(COLUMN_CLASS_NAME, "")).strip()
        # `YOLOE.set_classes` asserts that " " is not among the class names, as it uses
        # that as its background sentinel. Stripping and skipping empty names covers it.
        if not class_name:
            continue
        if class_name not in class_names:
            class_names.append(class_name)

    if not class_names:
        return [], "classes parameter must contain at least one valid class name."
    return class_names, None


def _get_or_create_label_map(
    *,
    session: Session,
    root_collection_id: UUID,
    class_map: dict[int, str],
) -> dict[int, UUID]:
    """Ensure labels exist for all class names and return {category_id: label_id}."""
    collection = collection_resolver.get_by_id(
        session=session,
        collection_id=root_collection_id,
    )
    if collection is None:
        raise ValueError(f"Collection {root_collection_id} doesn't exist")
    dataset_id = collection.dataset_id

    label_map: dict[int, UUID] = {}
    for category_id, label_name in class_map.items():
        label = annotation_label_resolver.get_by_label_name(
            session=session,
            dataset_id=dataset_id,
            label_name=label_name,
        )
        if label is None:
            label = annotation_label_resolver.create(
                session=session,
                label=AnnotationLabelCreate(
                    dataset_id=dataset_id,
                    annotation_label_name=label_name,
                ),
            )
        label_map[category_id] = label.annotation_label_id

    return label_map
