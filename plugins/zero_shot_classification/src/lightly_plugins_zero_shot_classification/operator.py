"""Zero-shot classification operator for Lightly Studio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple
from uuid import UUID

import numpy as np
from lightly_studio.dataset.embedding_manager import (
    EmbeddingManagerProvider,
    TextEmbedQuery,
)
from lightly_studio.models.annotation.annotation_base import (
    AnnotationCreate,
    AnnotationType,
)
from lightly_studio.models.annotation_label import AnnotationLabelCreate
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.plugins.parameter import (
    BaseParameter,
    StringParameter,
    TableParameter,
)
from lightly_studio.resolvers import (
    annotation_label_resolver,
    annotation_resolver,
    collection_resolver,
    sample_embedding_resolver,
)
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter
from lightly_studio.resolvers.video_frame_resolver.video_frame_filter import (
    VideoFrameFilter,
)
from lightly_studio.resolvers.video_resolver.video_filter import VideoFilter
from numpy.typing import NDArray
from sqlmodel import Session

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION_NAME = "zero_shot"

# Number of samples scored in a single matrix multiplication and written per insert.
_BATCH_SIZE = 1024
# Guards against dividing by zero when normalizing a degenerate embedding.
_EPSILON = 1e-12

PARAM_PROMPTS = "prompts"
PARAM_ANNOTATION_SOURCE = "annotation_source"

COLUMN_PROMPT = "prompt"
COLUMN_LABEL = "label"


class PromptRow(NamedTuple):
    """One prompt-table row: the text to match and the label its best match gets."""

    prompt: str
    label: str


@dataclass
class ZeroShotClassificationOperator(BaseOperator):
    """Zero-shot classification driven by a table of prompts and labels."""

    name: str = "Zero-Shot Classification"
    description: str = (
        "Classifies samples against a table of text prompts using the embeddings already "
        "computed for the collection."
    )

    @property
    def parameters(self) -> list[BaseParameter]:
        return [
            TableParameter(
                name=PARAM_PROMPTS,
                required=True,
                description=(
                    "Text prompt to match against each sample and the label to assign "
                    "when it wins."
                ),
                columns=[
                    StringParameter(
                        name=COLUMN_PROMPT,
                        description="Text prompt, e.g. 'a photo of a dog'.",
                        required=True,
                    ),
                    StringParameter(
                        name=COLUMN_LABEL,
                        description=(
                            "Label to assign when this prompt scores highest. Leave "
                            "empty to use the prompt itself."
                        ),
                        required=False,
                    ),
                ],
                default=[
                    {COLUMN_PROMPT: "a photo of a dog", COLUMN_LABEL: "dog"},
                    {COLUMN_PROMPT: "a photo of a cat", COLUMN_LABEL: "cat"},
                ],
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
        return [OperatorScope.IMAGE, OperatorScope.VIDEO_FRAME, OperatorScope.VIDEO]

    def _build_runtime_error_result(self, exc: Exception) -> OperatorResult:
        logger.exception("Zero-shot classification failed: %s", exc)
        return OperatorResult(
            success=False,
            message=("Zero-shot classification failed. Check the logs for details."),
        )

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        annotation_source = parameters.get(PARAM_ANNOTATION_SOURCE)
        collection_name = (
            str(annotation_source).strip()
            if annotation_source is not None and str(annotation_source).strip()
            else _DEFAULT_COLLECTION_NAME
        )

        try:
            prompt_rows = _parse_prompt_rows(parameters.get(PARAM_PROMPTS))
        except ValueError as exc:
            return OperatorResult(success=False, message=str(exc))
        if len(prompt_rows) < 2:
            return OperatorResult(
                success=False,
                message="Please provide at least two prompts.",
            )

        collection = collection_resolver.get_by_id(
            session=session, collection_id=context.collection_id
        )
        if collection is None:
            return OperatorResult(success=False, message="Collection not found.")

        # Registers the collection's default model, which `embed_text` then requires.
        embedding_manager = EmbeddingManagerProvider.get_embedding_manager()
        embedding_model_id = embedding_manager.load_or_get_default_model(
            session=session, collection_id=context.collection_id
        )
        if embedding_model_id is None:
            return OperatorResult(
                success=False,
                message="No embedding model is available for this collection.",
            )

        embedding_rows = sample_embedding_resolver.get_all_by_collection_id(
            session=session,
            collection_id=context.collection_id,
            embedding_model_id=embedding_model_id,
            filters=_as_sample_filter(context.context_filter),
        )
        if not embedding_rows:
            return OperatorResult(
                success=False,
                message=(
                    "No embeddings found for the current view. Embed the collection "
                    "before classifying."
                ),
            )

        try:
            prompt_embeds = _l2_normalize(
                np.stack(
                    [
                        np.asarray(
                            embedding_manager.embed_text(
                                collection_id=context.collection_id,
                                text_query=TextEmbedQuery(
                                    text=row.prompt,
                                    embedding_model_id=embedding_model_id,
                                ),
                            ),
                            dtype=np.float32,
                        )
                        for row in prompt_rows
                    ]
                )
            )
        except Exception as exc:
            return self._build_runtime_error_result(exc)

        embedding_dimension = embedding_rows[0].embedding.shape[-1]
        if embedding_dimension != prompt_embeds.shape[-1]:
            return OperatorResult(
                success=False,
                message=(
                    f"Sample embeddings have {embedding_dimension} dimensions but the "
                    f"prompt embeddings have {prompt_embeds.shape[-1]}."
                ),
            )

        label_ids = {
            row.label: _get_or_create_label(
                session=session, dataset_id=collection.dataset_id, label_name=row.label
            )
            for row in prompt_rows
        }

        total_annotations_created = 0

        for batch_start in range(0, len(embedding_rows), _BATCH_SIZE):
            batch = embedding_rows[batch_start : batch_start + _BATCH_SIZE]

            try:
                sample_embeds = _l2_normalize(
                    np.stack([row.embedding for row in batch]).astype(np.float32)
                )
                # Both sets are L2-normalized, so the dot product is cosine similarity.
                similarities = sample_embeds @ prompt_embeds.T
                # Unscaled softmax: the model's trained logit scale saturates it at ~1.
                scores = _softmax(similarities)
                best_indices = scores.argmax(axis=-1)
                best_scores = scores[np.arange(len(batch)), best_indices]
            except Exception as exc:
                return self._build_runtime_error_result(exc)

            annotations_to_create = [
                AnnotationCreate(
                    annotation_label_id=label_ids[prompt_rows[best_index].label],
                    annotation_type=AnnotationType.CLASSIFICATION,
                    parent_sample_id=row.sample_id,
                    confidence=float(best_score),
                )
                for row, best_score, best_index in zip(
                    batch, best_scores.tolist(), best_indices.tolist()
                )
            ]
            created = annotation_resolver.create_many(
                session=session,
                parent_collection_id=context.collection_id,
                annotations=annotations_to_create,
                collection_name=collection_name,
            )
            total_annotations_created += len(created)

        message = (
            f"Classified {total_annotations_created} samples into "
            f"{len(label_ids)} labels."
        )
        return OperatorResult(success=True, message=message)


def _parse_prompt_rows(rows: Any) -> list[PromptRow]:
    """Read the prompt table, which reaches the operator unvalidated.

    Raises `ValueError` on a row whose prompt is empty, rather than dropping it, so a
    half-filled row is reported instead of silently classifying against fewer prompts.
    """
    if not isinstance(rows, list):
        return []

    parsed: list[PromptRow] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        prompt = str(row.get(COLUMN_PROMPT, "")).strip()
        if not prompt:
            raise ValueError(f"Row {index + 1} has an empty prompt.")
        label = str(row.get(COLUMN_LABEL, "")).strip() or prompt
        parsed.append(PromptRow(prompt=prompt, label=label))

    return parsed


def _as_sample_filter(context_filter: Any) -> SampleFilter | None:
    """Reduce the operator's context filter to the sample filter embeddings accept.

    Only the sample filter survives: the media-specific predicates of the grid filters
    (width, height, fps, duration_s, frame_number) are dropped, so a view narrowed by
    those alone classifies more samples than it shows.
    """
    if isinstance(context_filter, SampleFilter):
        return context_filter
    if isinstance(context_filter, (ImageFilter, VideoFrameFilter, VideoFilter)):
        return context_filter.sample_filter
    return None


def _l2_normalize(matrix: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """L2-normalize row-wise, so a dot product is cosine similarity.

    Stored embeddings are not normalized by every model: some write raw encoder
    output while others normalize.
    """
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    normalized: NDArray[np.floating[Any]] = matrix / np.maximum(norms, _EPSILON)
    return normalized


def _softmax(scores: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Softmax over the prompts, shifted by the row max for numerical stability."""
    exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


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
