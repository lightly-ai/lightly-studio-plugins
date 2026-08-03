"""CLIP zero-shot image classification operator for Lightly Studio."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import PIL.Image
import torch
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
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "openai/clip-vit-base-patch16"
_DEFAULT_CONFIDENCE = 0.0
_DEFAULT_COLLECTION_NAME = "clip_zero_shot"

# Number of images encoded in a single CLIP forward pass.
_INFERENCE_BATCH_SIZE = 32
# Number of annotations buffered before they are written to the database.
_WRITE_BATCH_SIZE = 100

PARAM_MODEL_ID = "model_id"
PARAM_PROMPTS = "prompts"
PARAM_CONFIDENCE = "confidence_threshold"
PARAM_ANNOTATION_SOURCE = "annotation_source"

COLUMN_PROMPT = "prompt"
COLUMN_LABEL = "label"


def _get_or_create_label(session: Session, dataset_id: UUID, label_name: str) -> UUID:
    """Return the id of the label with the given name, creating it if necessary."""
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


def _parse_prompt_rows(rows: Any) -> tuple[list[str], list[str]]:
    """Extract the prompt and label of every usable row of the prompts table.

    The rows are not validated against the declared parameters before they reach the
    operator, so they are read defensively here. Rows that are malformed or carry an empty
    prompt are skipped. An empty label falls back to the prompt itself, since the label
    column is optional.

    Args:
        rows: The raw value of the prompts table parameter.

    Returns:
        The prompts and their labels as two parallel lists.
    """
    prompts: list[str] = []
    labels: list[str] = []
    if not isinstance(rows, list):
        return prompts, labels

    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get(COLUMN_PROMPT, "")).strip()
        if not prompt:
            continue
        label = str(row.get(COLUMN_LABEL, "")).strip() or prompt
        prompts.append(prompt)
        labels.append(label)

    return prompts, labels


def _select_device() -> str:
    """Return the best available torch device, preferring CUDA, then Apple Silicon."""
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _normalized_embeddings(features: Any) -> torch.Tensor:
    """Turn the output of a CLIP feature call into L2-normalized embeddings.

    `get_text_features` and `get_image_features` return a bare tensor on transformers 4.x
    but a `BaseModelOutputWithPooling` on 5.x, where the embeddings are the pooler output.
    Note that indexing such an output yields `last_hidden_state`, not the embeddings, so the
    pooler output is read explicitly.

    Args:
        features: The value returned by a CLIP feature call.

    Returns:
        The embeddings of shape (batch_size, embedding_dim), L2-normalized along the last
        dimension.
    """
    embeds = features if isinstance(features, torch.Tensor) else features.pooler_output
    normalized: torch.Tensor = embeds / embeds.norm(dim=-1, keepdim=True)
    return normalized


def _label_row_indices(labels: list[str]) -> dict[str, list[int]]:
    """Group the row indices of the prompts table by label name.

    Several rows may carry the same label to ensemble multiple prompts for one class, so a
    label maps to every row index that declares it.

    Args:
        labels: The label of every prompt, in row order.

    Returns:
        A mapping from label name to the row indices carrying that label.
    """
    indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        indices.setdefault(label, []).append(index)
    return indices


@dataclass
class ClipZeroShotClassificationOperator(BaseOperator):
    """Zero-shot image classification using CLIP driven by a table of prompts and labels."""

    name: str = "CLIP Zero-Shot Classification"
    description: str = (
        "Classifies images with CLIP against a table of text prompts and the labels to "
        "assign. Several prompts may share a label to ensemble them."
    )
    _model: Any = dataclasses.field(default=None, init=False, repr=False)
    _processor: Any = dataclasses.field(default=None, init=False, repr=False)
    _model_device: str = dataclasses.field(default="", init=False, repr=False)
    _loaded_model_id: str = dataclasses.field(default="", init=False, repr=False)

    @property
    def parameters(self) -> list[BaseParameter]:
        """Return the list of parameters this operator expects."""
        return [
            StringParameter(
                name=PARAM_MODEL_ID,
                required=True,
                default=_DEFAULT_MODEL_ID,
                description=(
                    "HuggingFace CLIP model ID. Larger models are more accurate but "
                    "slower, e.g. 'openai/clip-vit-large-patch14'."
                ),
            ),
            TableParameter(
                name=PARAM_PROMPTS,
                required=True,
                description=(
                    "Text prompt to match against each image and the label to assign when "
                    "it wins. Several rows may share a label to ensemble prompts."
                ),
                columns=[
                    StringParameter(
                        name=COLUMN_PROMPT,
                        description="CLIP text prompt, e.g. 'a photo of a dog'.",
                    ),
                    StringParameter(
                        name=COLUMN_LABEL,
                        description=(
                            "Label to assign when this prompt scores highest. Defaults to "
                            "the prompt itself."
                        ),
                        required=False,
                    ),
                ],
                default=[
                    {COLUMN_PROMPT: "a photo of a dog", COLUMN_LABEL: "dog"},
                    {COLUMN_PROMPT: "a photo of a cat", COLUMN_LABEL: "cat"},
                ],
            ),
            FloatParameter(
                name=PARAM_CONFIDENCE,
                required=False,
                default=_DEFAULT_CONFIDENCE,
                description=(
                    "Minimum score for assigning a label. Images whose best label scores "
                    "below this are left unclassified."
                ),
            ),
            StringParameter(
                name=PARAM_ANNOTATION_SOURCE,
                required=False,
                default=_DEFAULT_COLLECTION_NAME,
                description="Target annotation source name where predictions are stored.",
            ),
        ]

    @property
    def supported_scopes(self) -> list[OperatorScope]:
        """Return the list of scopes this operator can be triggered from."""
        return [OperatorScope.IMAGE]

    def _load_model(self, model_id: str, device: str) -> None:
        """Load the CLIP model and processor, reusing an already loaded model."""
        if (
            self._model is not None
            and self._model_device == device
            and self._loaded_model_id == model_id
        ):
            return

        logger.info("Loading CLIP model (%s) on device: %s", model_id, device)
        self._model = CLIPModel.from_pretrained(model_id).to(device).eval()  # type: ignore[arg-type]
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._model_device = device
        self._loaded_model_id = model_id

    def _encode_prompts(self, prompts: list[str], device: str) -> torch.Tensor:
        """Encode the prompts into normalized CLIP text embeddings.

        Args:
            prompts: The prompts to encode.
            device: Device the model runs on.

        Returns:
            The normalized text embeddings of shape (num_prompts, embedding_dim).
        """
        inputs = self._processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
        return _normalized_embeddings(text_features)

    def _encode_images(
        self, images: list[PIL.Image.Image], device: str
    ) -> torch.Tensor:
        """Encode the images into normalized CLIP image embeddings.

        Args:
            images: The images to encode.
            device: Device the model runs on.

        Returns:
            The normalized image embeddings of shape (num_images, embedding_dim).
        """
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
        return _normalized_embeddings(image_features)

    def _build_runtime_error_result(self, exc: Exception) -> OperatorResult:
        """Log the exception and turn it into a failed operator result."""
        logger.exception("CLIP zero-shot classification failed: %s", exc)
        return OperatorResult(
            success=False,
            message=(
                "CLIP zero-shot classification failed. Verify the model ID is a CLIP "
                "model available on HuggingFace and check the logs for details."
            ),
        )

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        """Classify every image in the current view against the configured prompts.

        Args:
            session: Database session.
            context: Execution context containing collection_id and optional filter.
            parameters: Parameters passed to the operator.

        Returns:
            An OperatorResult summarizing how many images were classified.
        """
        model_id = str(parameters.get(PARAM_MODEL_ID, _DEFAULT_MODEL_ID)).strip()
        if not model_id:
            return OperatorResult(success=False, message="Please provide a model ID.")

        confidence_threshold = float(
            parameters.get(PARAM_CONFIDENCE, _DEFAULT_CONFIDENCE)
        )
        if not 0.0 <= confidence_threshold <= 1.0:
            return OperatorResult(
                success=False,
                message=f"{PARAM_CONFIDENCE} must be between 0 and 1.",
            )

        annotation_source = parameters.get(PARAM_ANNOTATION_SOURCE)
        collection_name = (
            str(annotation_source).strip()
            if annotation_source is not None and str(annotation_source).strip()
            else _DEFAULT_COLLECTION_NAME
        )

        prompts, labels = _parse_prompt_rows(parameters.get(PARAM_PROMPTS))
        if not prompts:
            return OperatorResult(
                success=False,
                message="Please provide at least one prompt.",
            )

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

        device = _select_device()

        try:
            self._load_model(model_id, device)
            text_embeds = self._encode_prompts(prompts, device)
        except Exception as exc:
            return self._build_runtime_error_result(exc)

        row_indices_by_label = _label_row_indices(labels)
        # `logit_scale` is a trainable parameter, so detach it to keep the scores out of the
        # autograd graph.
        logit_scale = self._model.logit_scale.detach().exp()
        label_ids = {
            label_name: _get_or_create_label(
                session=session,
                dataset_id=collection.dataset_id,
                label_name=label_name,
            )
            for label_name in row_indices_by_label
        }
        label_names = list(row_indices_by_label)

        annotations_to_create: list[AnnotationCreate] = []
        total_annotations_created = 0
        skipped_below_threshold = 0
        unreadable_images = 0

        for batch_start in range(0, len(samples), _INFERENCE_BATCH_SIZE):
            batch = samples[batch_start : batch_start + _INFERENCE_BATCH_SIZE]

            batch_samples = []
            images: list[PIL.Image.Image] = []
            for sample in batch:
                try:
                    with PIL.Image.open(sample.file_path_abs) as opened_image:
                        images.append(opened_image.convert("RGB"))
                except Exception:
                    logger.warning(
                        "Could not open image: %s — skipping.", sample.file_path_abs
                    )
                    unreadable_images += 1
                    continue
                batch_samples.append(sample)

            if not images:
                continue

            try:
                image_embeds = self._encode_images(images, device)
                similarities = image_embeds @ text_embeds.T
                # Pool the prompt *similarities* per label by averaging them, which is the
                # standard way to ensemble several prompts for one class. The softmax is
                # then taken over labels rather than over prompts, so the resulting
                # probabilities sum to one across labels and are directly comparable to
                # the confidence threshold. Pooling after a softmax over prompts would
                # instead split a class's probability mass across its own prompts, which
                # both deflates its score and can hand the win to a single-prompt class.
                label_similarities = torch.stack(
                    [
                        similarities[:, row_indices_by_label[label_name]].mean(dim=-1)
                        for label_name in label_names
                    ],
                    dim=-1,
                )
                label_scores = (logit_scale * label_similarities).softmax(dim=-1)
                best_scores, best_indices = label_scores.max(dim=-1)
            except Exception as exc:
                return self._build_runtime_error_result(exc)

            for sample, best_score, best_index in zip(
                batch_samples, best_scores.tolist(), best_indices.tolist()
            ):
                if best_score < confidence_threshold:
                    skipped_below_threshold += 1
                    continue
                annotations_to_create.append(
                    AnnotationCreate(
                        annotation_label_id=label_ids[label_names[best_index]],
                        annotation_type=AnnotationType.CLASSIFICATION,
                        parent_sample_id=sample.sample_id,
                        confidence=float(best_score),
                    )
                )

            if len(annotations_to_create) >= _WRITE_BATCH_SIZE:
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

        message = (
            f"Classified {total_annotations_created} of {len(samples)} samples into "
            f"{len(label_names)} labels."
        )
        if skipped_below_threshold:
            message += f" {skipped_below_threshold} below threshold."
        if unreadable_images:
            message += f" {unreadable_images} could not be read."
        return OperatorResult(success=True, message=message)
