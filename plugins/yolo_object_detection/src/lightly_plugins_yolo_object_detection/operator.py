"""YOLO inference operator for object detection auto-labeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlmodel import Session
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.engine.results import Results

from lightly_studio.models.annotation.annotation_base import (
    AnnotationCreate,
    AnnotationType,
)
from lightly_studio.models.annotation_label import AnnotationLabelCreate
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.plugins.parameter import (
    BaseParameter,
    FloatParameter,
    StringParameter,
)
from lightly_studio.resolvers import (
    annotation_label_resolver,
    annotation_resolver,
    collection_resolver,
    image_resolver,
)
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25

PARAM_MODEL = "model_path"
PARAM_CONFIDENCE = "confidence"
PARAM_ANNOTATION_SOURCE = "annotation_source"

_WRITE_BATCH_SIZE = 100


@dataclass
class YoloObjectDetectionOperator(BaseOperator):
    """Runs YOLO inference to auto-label images with bounding box annotations."""

    name: str = "YOLO Object Detection"
    description: str = (
        "Runs YOLO inference and adds bounding box annotations to unlabeled images."
    )

    @property
    def parameters(self) -> list[BaseParameter]:
        """Return the list of parameters this operator expects."""
        return [
            StringParameter(
                name=PARAM_MODEL,
                required=True,
                default=DEFAULT_MODEL,
                description="YOLO model weights path or Ultralytics model name (e.g. yolov8n.pt).",
            ),
            FloatParameter(
                name=PARAM_CONFIDENCE,
                required=True,
                default=DEFAULT_CONFIDENCE,
                description="Minimum confidence threshold for keeping a prediction.",
            ),
            StringParameter(
                name=PARAM_ANNOTATION_SOURCE,
                required=False,
                description="Target annotation source name where predictions will be stored. Defaults to yolo_auto_label__{model_path}.",
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
            else f"yolo_auto_label__{model_path}"
        )
        confidence = float(parameters.get(PARAM_CONFIDENCE, DEFAULT_CONFIDENCE))

        if not 0.0 <= confidence <= 1.0:
            return OperatorResult(
                success=False,
                message="confidence must be between 0 and 1",
            )

        try:
            model = YOLO(model_path)
        except Exception as e:
            logger.error("Failed to load YOLO model '%s': %s", model_path, e)
            return OperatorResult(
                success=False,
                message=f"Failed to load YOLO model '{model_path}': {e}",
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

        annotations_to_create: list[AnnotationCreate] = []
        total_annotations_created = 0
        for i, image_entry in enumerate(samples, start=1):
            try:
                result = list(
                    model(image_entry.file_path_abs, conf=confidence, verbose=False)
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
            # A single image always yields one `Results`; `embed=` is never passed.
            assert isinstance(result, Results)
            boxes = result.boxes
            if boxes is None:
                logger.warning(
                    "No boxes returned for '%s'; is '%s' a detection model?",
                    image_entry.file_path_abs,
                    model_path,
                )
                continue
            for box_index in range(len(boxes)):
                category_id = int(boxes.cls[box_index])
                label_id = label_map.get(category_id)
                if label_id is None:
                    continue
                x_center, y_center, w, h = boxes.xywh[box_index].tolist()
                annotations_to_create.append(
                    AnnotationCreate(
                        annotation_label_id=label_id,
                        annotation_type=AnnotationType.OBJECT_DETECTION,
                        parent_sample_id=image_entry.sample_id,
                        x=round(x_center - w / 2),
                        y=round(y_center - h / 2),
                        width=max(1, round(w)),
                        height=max(1, round(h)),
                        confidence=float(boxes.conf[box_index]),
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
