"""SAM3 instance segmentation plugin for Lightly Studio."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, NamedTuple
from uuid import UUID

import PIL.Image
import torch
from sqlmodel import Session
from transformers import Sam3Model, Sam3Processor

from lightly_studio.models.annotation.annotation_base import (
    AnnotationCreate,
    AnnotationType,
)
from lightly_studio.models.annotation_label import AnnotationLabelCreate
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter
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

from lightly_plugins_sam3_segmentation.utils import prepare_segmentation_entries

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "facebook/sam3"

PARAM_MODEL_ID = "model_id"
PARAM_PROMPTS = "prompts"
PARAM_CONFIDENCE_THRESHOLD = "confidence_threshold"
PARAM_BOUNDING_BOXES_ONLY = "bounding_boxes_only"
PARAM_COLLECTION_NAME = "collection_name"

COLUMN_PROMPT = "prompt"
COLUMN_LABEL = "label"


class PromptRow(NamedTuple):
    """One prompt-table row: what to segment and the label its detections get."""

    prompt: str
    label: str


def _parse_prompt_rows(rows: Any) -> list[PromptRow]:
    """Read the prompt table, which reaches the operator unvalidated."""
    if not isinstance(rows, list):
        return []

    parsed: list[PromptRow] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get(COLUMN_PROMPT, "")).strip()
        if prompt:
            label = str(row.get(COLUMN_LABEL, "")).strip() or prompt
            parsed.append(PromptRow(prompt=prompt, label=label))

    return parsed


def _select_device() -> str:
    """Pick the fastest device SAM3 can run on."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_or_create_label(session: Session, dataset_id: UUID, label_name: str) -> UUID:
    label = annotation_label_resolver.get_by_label_name(
        session=session, dataset_id=dataset_id, label_name=label_name
    )
    if label is None:
        label = annotation_label_resolver.create(
            session=session,
            label=AnnotationLabelCreate(
                dataset_id=dataset_id, annotation_label_name=label_name
            ),
        )
    return label.annotation_label_id


@dataclass
class SAM3SegmentationOperator(BaseOperator):
    """Instance segmentation using SAM3 driven by a table of prompts and labels."""

    name: str = "SAM3"
    description: str = (
        "Automatic instance segmentation using SAM3 (facebook/sam3) from a table of text "
        "prompts. Requires HuggingFace access — authenticate with `hf auth login` first."
    )
    _model: Any = dataclasses.field(default=None, init=False, repr=False)
    _processor: Any = dataclasses.field(default=None, init=False, repr=False)
    _model_device: str = dataclasses.field(default="", init=False, repr=False)
    _loaded_model_id: str = dataclasses.field(default="", init=False, repr=False)

    @property
    def parameters(self) -> list[BaseParameter]:
        return [
            StringParameter(
                name=PARAM_MODEL_ID,
                required=True,
                default=_DEFAULT_MODEL_ID,
                description="HuggingFace model ID (e.g. 'facebook/sam3' or 'facebook/sam3.1')",
            ),
            TableParameter(
                name=PARAM_PROMPTS,
                required=True,
                description=(
                    "Text prompts to segment with and the label to assign to each "
                    "prompt's detections. One prompt per row."
                ),
                columns=[
                    StringParameter(
                        name=COLUMN_PROMPT,
                        description="What to segment, e.g. 'person'.",
                    ),
                    StringParameter(
                        name=COLUMN_LABEL,
                        description=(
                            "Label assigned to this prompt's detections. Defaults to "
                            "the prompt itself."
                        ),
                        required=False,
                    ),
                ],
                default=[
                    {COLUMN_PROMPT: "person", COLUMN_LABEL: "person"},
                    {COLUMN_PROMPT: "car", COLUMN_LABEL: "car"},
                ],
            ),
            FloatParameter(
                name=PARAM_CONFIDENCE_THRESHOLD,
                required=False,
                default=0.5,
                description="Minimum confidence score for keeping a prediction",
            ),
            BoolParameter(
                name=PARAM_BOUNDING_BOXES_ONLY,
                required=False,
                default=False,
                description="Store bounding boxes instead of segmentation masks.",
            ),
            StringParameter(
                name=PARAM_COLLECTION_NAME,
                required=True,
                default="SAM3_auto_label",
                description="The target annotation collection name.",
            ),
        ]

    @property
    def supported_scopes(self) -> list[OperatorScope]:
        return [OperatorScope.IMAGE]

    def _load_model(self, model_id: str, device: str) -> None:
        if (
            self._model is not None
            and self._model_device == device
            and self._loaded_model_id == model_id
        ):
            return

        logger.info("Loading SAM3 model (%s) on device: %s", model_id, device)
        # `facebook/sam3` may resolve to the video processor through AutoProcessor in
        # recent `transformers` builds. Load the image SAM3 classes explicitly so
        # text-prompted image segmentation stays on the correct code path.
        self._model = Sam3Model.from_pretrained(model_id).to(device).eval()  # type: ignore[arg-type]
        self._processor = Sam3Processor.from_pretrained(model_id)
        self._model_device = device
        self._loaded_model_id = model_id

    def _segment_image(
        self,
        *,
        image: PIL.Image.Image,
        prompt_rows: list[PromptRow],
        image_size: tuple[int, int],
        confidence_threshold: float,
        include_rle: bool,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Segment one image with every prompt, returning (label, detection) pairs."""
        width, height = image_size
        device = self._model_device
        detections: list[tuple[str, dict[str, Any]]] = []

        with torch.no_grad():
            image_inputs = self._processor(images=image, return_tensors="pt").to(device)
            vision_embeds = self._model.get_vision_features(
                pixel_values=image_inputs["pixel_values"]
            )

            for row in prompt_rows:
                text_inputs = self._processor(text=row.prompt, return_tensors="pt").to(
                    device
                )
                outputs = self._model(vision_embeds=vision_embeds, **text_inputs)
                result = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=confidence_threshold,
                    target_sizes=[(height, width)],
                )[0]
                entries = prepare_segmentation_entries(
                    boxes=result["boxes"],
                    masks=result["masks"],
                    scores=result["scores"],
                    image_size=image_size,
                    include_rle=include_rle,
                )
                detections += [(row.label, entry) for entry in entries]

        return detections

    def _build_runtime_error_result(self, exc: Exception) -> OperatorResult:
        logger.exception("SAM3 segmentation failed: %s", exc)
        return OperatorResult(
            success=False,
            message=(
                "SAM3 segmentation failed. Verify HuggingFace access for the selected "
                "model, run `hf auth login`, and check the logs for details."
            ),
        )

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        model_id: str = parameters.get(PARAM_MODEL_ID, _DEFAULT_MODEL_ID)
        prompt_rows = _parse_prompt_rows(parameters.get(PARAM_PROMPTS))
        if not prompt_rows:
            return OperatorResult(
                success=False,
                message="Please provide at least one prompt.",
            )
        confidence_threshold: float = parameters.get(PARAM_CONFIDENCE_THRESHOLD, 0.5)

        bounding_boxes_only = bool(parameters.get(PARAM_BOUNDING_BOXES_ONLY, False))

        device = _select_device()
        collection_name_value = parameters.get(PARAM_COLLECTION_NAME)
        if collection_name_value is None:
            return OperatorResult(
                success=False,
                message="Please provide a collection name.",
            )
        collection_name: str = collection_name_value

        collection = collection_resolver.get_by_id(
            session=session, collection_id=context.collection_id
        )
        if collection is None:
            return OperatorResult(success=False, message="Collection not found.")

        context_filter: ImageFilter | None = None
        if isinstance(context.context_filter, SampleFilter):
            context_filter = ImageFilter(sample_filter=context.context_filter)
        elif isinstance(context.context_filter, ImageFilter):
            context_filter = context.context_filter

        result = image_resolver.get_all_by_collection_id(
            session=session, collection_id=context.collection_id, filters=context_filter
        )

        samples = list(result.samples)
        if not samples:
            return OperatorResult(
                success=True,
                message="No samples found for current view.",
            )

        try:
            self._load_model(model_id, device)
        except Exception as exc:
            return self._build_runtime_error_result(exc)

        detections: list[tuple[Any, str, dict[str, Any]]] = []
        for sample in samples:
            try:
                with PIL.Image.open(sample.file_path_abs) as opened_image:
                    image = opened_image.convert("RGB")
            except Exception:
                logger.warning(
                    "Could not open image: %s — skipping.", sample.file_path_abs
                )
                continue

            try:
                image_detections = self._segment_image(
                    image=image,
                    prompt_rows=prompt_rows,
                    image_size=(sample.width, sample.height),
                    confidence_threshold=confidence_threshold,
                    include_rle=not bounding_boxes_only,
                )
            except Exception as exc:
                return self._build_runtime_error_result(exc)

            detections += [(sample, label, entry) for label, entry in image_detections]

        if not detections:
            return OperatorResult(
                success=True,
                message="Segmentation complete. No annotations created.",
            )

        label_ids = {
            row.label: _get_or_create_label(
                session=session, dataset_id=collection.dataset_id, label_name=row.label
            )
            for row in prompt_rows
        }

        # A segmentation annotation carries its bounding box as well, so storing
        # masks already covers both. Boxes-only drops the mask on purpose.
        annotation_type = (
            AnnotationType.OBJECT_DETECTION
            if bounding_boxes_only
            else AnnotationType.SEGMENTATION_MASK
        )

        annotation_creates: list[AnnotationCreate] = []
        for sample, label, entry in detections:
            x, y, w, h = entry["box"]
            annotation_creates.append(
                AnnotationCreate(
                    annotation_label_id=label_ids[label],
                    annotation_type=annotation_type,
                    parent_sample_id=sample.sample_id,
                    confidence=entry["score"],
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    segmentation_mask=entry["rle"],
                )
            )

        annotation_resolver.create_many(
            session=session,
            parent_collection_id=context.collection_id,
            annotations=annotation_creates,
            collection_name=collection_name,
        )
        return OperatorResult(
            success=True,
            message=f"Segmentation complete. Created {len(annotation_creates)} annotations.",
        )
