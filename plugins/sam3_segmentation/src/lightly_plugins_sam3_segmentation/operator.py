"""SAM3 instance segmentation plugin for Lightly Studio."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import PIL.Image
import torch
from sqlmodel import Session
from transformers.models.sam3.modeling_sam3 import Sam3Model
from transformers.models.sam3.processing_sam3 import Sam3Processor

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
    """Instance segmentation using SAM3 driven by a text prompt."""

    name: str = "SAM3 Segmentation"
    description: str = (
        "Automatic instance segmentation using SAM3 (facebook/sam3). "
        "Requires HuggingFace access — authenticate with `hf auth login` first."
    )
    _model: Any = dataclasses.field(default=None, init=False, repr=False)
    _processor: Any = dataclasses.field(default=None, init=False, repr=False)
    _model_device: str = dataclasses.field(default="", init=False, repr=False)
    _loaded_model_id: str = dataclasses.field(default="", init=False, repr=False)

    @property
    def parameters(self) -> list[BaseParameter]:
        return [
            StringParameter(
                name="model_id",
                required=True,
                default=_DEFAULT_MODEL_ID,
                description="HuggingFace model ID (e.g. 'facebook/sam3' or 'facebook/sam3.1')",
            ),
            StringParameter(
                name="prompt",
                required=True,
                default="person",
                description="Text prompt describing what to segment (e.g. 'person', 'car')",
            ),
            FloatParameter(
                name="confidence_threshold",
                required=False,
                default=0.5,
                description="Minimum confidence score for keeping a prediction",
            ),
            BoolParameter(
                name="use_gpu",
                required=False,
                default=False,
                description="Run inference on GPU (CUDA). Falls back to CPU if unavailable.",
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
        self._model = Sam3Model.from_pretrained(model_id).to(device).eval()
        self._processor = Sam3Processor.from_pretrained(model_id)  # type: ignore[no-untyped-call]
        self._model_device = device
        self._loaded_model_id = model_id

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        model_id: str = parameters.get("model_id", _DEFAULT_MODEL_ID)
        prompt: str = parameters.get("prompt", "person")
        confidence_threshold: float = parameters.get("confidence_threshold", 0.5)
        use_gpu: bool = parameters.get("use_gpu", False)
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        self._load_model(model_id, device)

        collection = collection_resolver.get_by_id(
            session=session, collection_id=context.collection_id
        )
        if collection is None:
            return OperatorResult(success=False, message="Collection not found.")

        label_id = _get_or_create_label(
            session=session, dataset_id=collection.dataset_id, label_name=prompt
        )

        context_filter: ImageFilter | None = None
        if isinstance(context.context_filter, SampleFilter):
            context_filter = ImageFilter(sample_filter=context.context_filter)
        elif isinstance(context.context_filter, ImageFilter):
            context_filter = context.context_filter

        result = image_resolver.get_all_by_collection_id(
            session=session, collection_id=context.collection_id, filters=context_filter
        )

        annotation_creates: list[AnnotationCreate] = []
        for sample in result.samples:
            try:
                image = PIL.Image.open(sample.file_path_abs).convert("RGB")
            except Exception:
                logger.warning(
                    "Could not open image: %s — skipping.", sample.file_path_abs
                )
                continue

            inputs = self._processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            post_results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence_threshold,
                target_sizes=[(sample.height, sample.width)],
            )
            detections = post_results[0]

            entries = prepare_segmentation_entries(
                boxes=detections["boxes"],
                masks=detections["masks"],
                scores=detections["scores"],
                image_size=(sample.width, sample.height),
            )
            for entry in entries:
                x, y, w, h = entry["box"]
                annotation_creates.append(
                    AnnotationCreate(
                        annotation_label_id=label_id,
                        annotation_type=AnnotationType.INSTANCE_SEGMENTATION,
                        parent_sample_id=sample.sample_id,
                        confidence=entry["score"],
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        segmentation_mask=entry["rle"],
                    )
                )

        if not annotation_creates:
            return OperatorResult(
                success=True,
                message="Segmentation complete. No annotations created.",
            )

        annotation_resolver.create_many(
            session=session,
            parent_collection_id=context.collection_id,
            annotations=annotation_creates,
        )
        return OperatorResult(
            success=True,
            message=f"Segmentation complete. Created {len(annotation_creates)} annotations.",
        )
