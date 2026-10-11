"""YOLOE inference operator for open vocabulary object detection auto-labeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple
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
PARAM_PROMPTS = "prompts"
PARAM_INSTANCE_SEGMENTATION = "instance_segmentation"
PARAM_ANNOTATION_SOURCE = "annotation_source"

COLUMN_PROMPT = "prompt"
COLUMN_LABEL = "label"

_WRITE_BATCH_SIZE = 100


@dataclass
class YoloEObjectDetectionOperator(BaseOperator):
    """Runs YOLOE inference to auto-label images with boxes or segmentation masks."""

    name: str = "YOLOE Open Vocabulary Object Detection"
    description: str = (
        "Runs YOLOE inference from a table of text prompts and adds bounding box or "
        "instance segmentation annotations to unlabeled images."
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
                name=PARAM_PROMPTS,
                required=True,
                description=(
                    "One prompt per row. Detections are annotated with the row's "
                    "label, or with the prompt if the label is empty."
                ),
                columns=[
                    StringParameter(
                        name=COLUMN_PROMPT,
                        description="What to detect, e.g. 'person'.",
                    ),
                    StringParameter(
                        name=COLUMN_LABEL,
                        description=(
                            "Annotation label for this prompt's detections. "
                            "Leave empty to use the prompt."
                        ),
                        required=False,
                    ),
                ],
                default=[{COLUMN_PROMPT: "person", COLUMN_LABEL: "person"}],
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

        prompt_rows, prompt_rows_error = _parse_prompt_rows(
            rows=parameters.get(PARAM_PROMPTS)
        )
        if prompt_rows_error is not None:
            return OperatorResult(success=False, message=prompt_rows_error)

        try:
            model = YOLO(model_path)
        except Exception as e:
            logger.error("Failed to load YOLOE model '%s': %s", model_path, e)
            return OperatorResult(
                success=False,
                message=f"Failed to load YOLOE model '{model_path}': {e}",
            )

        try:
            model.set_classes([row.prompt for row in prompt_rows])
        except Exception as e:
            logger.error("Failed to set prompts on model '%s': %s", model_path, e)
            return OperatorResult(
                success=False,
                message=(
                    f"Failed to set prompts on model '{model_path}': {e}. "
                    "Make sure it is an open vocabulary YOLOE checkpoint."
                ),
            )

        label_map = _get_or_create_label_map(
            session=session,
            root_collection_id=context.collection_id,
            prompt_rows=prompt_rows,
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


class PromptRow(NamedTuple):
    """One prompt-table row: what to detect and the label its detections get."""

    prompt: str
    label: str


def _parse_prompt_rows(rows: Any) -> tuple[list[PromptRow], str | None]:
    """Read the prompt table, which reaches the operator unvalidated.

    `TableParameter` only checks cell types and column names, so blank and duplicate rows
    are rejected here rather than silently dropped.

    Args:
        rows: The value of the prompts table parameter.

    Returns:
        A tuple of the parsed rows and an error message, the latter being None on success.
    """
    if not isinstance(rows, list) or not rows:
        return [], "Please provide at least one prompt."

    parsed: list[PromptRow] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return [], "Please provide at least one prompt."
        prompt = str(row.get(COLUMN_PROMPT, "")).strip()
        # `YOLOE.set_classes` asserts that " " is not among the prompts, as it uses that
        # as its background sentinel, so a blank prompt can never be passed through.
        if not prompt:
            return [], "Every row needs a prompt. Remove empty rows and try again."
        if prompt in seen:
            return [], (
                "Each prompt can only appear once. "
                "Remove the duplicate rows and try again."
            )
        seen.add(prompt)
        label = str(row.get(COLUMN_LABEL, "")).strip() or prompt
        parsed.append(PromptRow(prompt=prompt, label=label))

    return parsed, None


def _get_or_create_label(
    *, session: Session, dataset_id: UUID, label_name: str
) -> UUID:
    """Return the id of the label with this name, creating it if needed."""
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
    return label.annotation_label_id


def _get_or_create_label_map(
    *,
    session: Session,
    root_collection_id: UUID,
    prompt_rows: list[PromptRow],
    class_map: dict[int, str],
) -> dict[int, UUID]:
    """Ensure labels exist for all rows and return {category_id: label_id}.

    `class_map` is `model.names` read back after `set_classes`, mapping each category id to
    its prompt. Going through it rather than the row order is deliberate: `set_classes`
    keeps the checkpoint's own ordering when the prompts are a permutation of its existing
    names. Rows sharing a label resolve to the same label id.

    Args:
        session: The database session.
        root_collection_id: The collection the annotations belong to.
        prompt_rows: The parsed rows of the prompts table parameter.
        class_map: The model's {category_id: prompt} mapping.

    Returns:
        The {category_id: label_id} mapping for the model's classes.
    """
    collection = collection_resolver.get_by_id(
        session=session,
        collection_id=root_collection_id,
    )
    if collection is None:
        raise ValueError(f"Collection {root_collection_id} doesn't exist")
    dataset_id = collection.dataset_id

    label_ids = {
        row.label: _get_or_create_label(
            session=session,
            dataset_id=dataset_id,
            label_name=row.label,
        )
        for row in prompt_rows
    }
    labels_by_prompt = {row.prompt: row.label for row in prompt_rows}
    return {
        category_id: label_ids[labels_by_prompt[prompt]]
        for category_id, prompt in class_map.items()
        if prompt in labels_by_prompt
    }
