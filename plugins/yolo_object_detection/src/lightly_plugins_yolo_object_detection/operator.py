"""YOLO inference operator for object detection auto-labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlmodel import Session

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

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25

PARAM_MODEL = "model_path"
PARAM_CONFIDENCE = "confidence"


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
        from ultralytics import YOLO

        model_path = str(parameters.get(PARAM_MODEL, DEFAULT_MODEL))
        confidence = float(parameters.get(PARAM_CONFIDENCE, DEFAULT_CONFIDENCE))

        if not 0.0 <= confidence <= 1.0:
            return OperatorResult(
                success=False,
                message="confidence must be between 0 and 1",
            )

        model = YOLO(model_path)
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
        for image_entry in samples:
            results = model(image_entry.file_path_abs, conf=confidence, verbose=False)[0]
            for box in results.boxes:
                category_id = int(box.cls)
                label_id = label_map.get(category_id)
                if label_id is None:
                    continue
                x_center, y_center, w, h = box.xywh[0].tolist()
                annotations_to_create.append(
                    AnnotationCreate(
                        annotation_label_id=label_id,
                        annotation_type=AnnotationType.OBJECT_DETECTION,
                        parent_sample_id=image_entry.sample_id,
                        x=round(x_center - w / 2),
                        y=round(y_center - h / 2),
                        width=round(w),
                        height=round(h),
                        confidence=float(box.conf),
                    )
                )

        if annotations_to_create:
            annotation_resolver.create_many(
                session=session,
                parent_collection_id=context.collection_id,
                annotations=annotations_to_create,
                collection_name=f"yolo_auto_label__{model_path}",
            )

        return OperatorResult(
            success=True,
            message=f"Auto-labeled {len(samples)} samples.",
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
