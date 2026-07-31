"""OpenRouter image captioning plugin for Lightly Studio.

HTTP work happens in worker threads while every database access stays on the calling
thread, because the SQLModel session is not thread-safe.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import logging
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
import PIL.Image
import PIL.ImageOps
from lightly_studio.models.caption import CaptionCreate
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import (
    AnyFilter,
    ExecutionContext,
    OperatorScope,
)
from lightly_studio.plugins.parameter import (
    BaseParameter,
    IntParameter,
    StringParameter,
)
from lightly_studio.resolvers import caption_resolver, image_resolver
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter
from sqlmodel import Session

from lightly_plugins_openrouter_image_captioning import openrouter_client
from lightly_plugins_openrouter_image_captioning.openrouter_client import (
    OpenRouterError,
    RequestConfig,
)

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
MODELS_URL = "https://openrouter.ai/models"

_DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
_DEFAULT_PROMPT = (
    "Describe this image in one or two concise sentences. Name the main objects, their "
    "notable attributes and the overall scene. Do not begin with 'The image shows'."
)
_DEFAULT_MAX_IMAGE_EDGE = 256
_DEFAULT_MAX_CONCURRENCY = 4
_JPEG_QUALITY = 85
_DB_FLUSH_BATCH_SIZE = 200

_MISSING_KEY_MESSAGE = (
    f"{API_KEY_ENV_VAR} is not set. Create a key at https://openrouter.ai/keys, "
    "export it (or put it in a .env file where you start LightlyStudio) and restart "
    "LightlyStudio."
)


@dataclass(frozen=True)
class CaptionJob:
    """Session-free data needed to caption one image inside a worker thread.

    Worker threads must not touch the session or any ORM object, so everything they
    need is snapshotted into immutable values before the work is submitted.
    """

    sample_id: UUID
    file_path_abs: str


@dataclass
class CaptionTally:
    """Running counts of a captioning run."""

    stored: int = 0
    failed: int = 0
    first_error: str | None = None


@dataclass
class OpenRouterImageCaptioningOperator(BaseOperator):
    """Generate image captions with a vision model served through OpenRouter."""

    name: str = "OpenRouter Image Captioning"
    description: str = (
        "Captions images with a vision model via OpenRouter and stores the result as a "
        "LightlyStudio caption. Requires the OPENROUTER_API_KEY environment variable."
    )

    @property
    def parameters(self) -> list[BaseParameter]:
        return [
            StringParameter(
                name="model",
                required=True,
                default=_DEFAULT_MODEL,
                description=(
                    "OpenRouter model slug of a vision-capable model. Browse and "
                    f"compare models at {MODELS_URL} (filter by image input)."
                ),
            ),
            StringParameter(
                name="prompt",
                required=True,
                default=_DEFAULT_PROMPT,
                description="Instruction sent to the model together with the image.",
            ),
            IntParameter(
                name="max_image_edge",
                required=False,
                default=_DEFAULT_MAX_IMAGE_EDGE,
                description=(
                    "Downscale images so the longest edge is at most this many pixels "
                    "before upload. Lower is cheaper and faster, but fine detail is "
                    "lost; raise it if captions miss small objects or text. 0 uploads "
                    "the image at its original size."
                ),
            ),
            IntParameter(
                name="max_concurrency",
                required=False,
                default=_DEFAULT_MAX_CONCURRENCY,
                description=(
                    "Number of images captioned in parallel. Raise this to speed up "
                    "large runs; lower it if requests start hitting rate limits."
                ),
            ),
        ]

    @property
    def supported_scopes(self) -> list[OperatorScope]:
        return [OperatorScope.IMAGE]

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        try:
            return _run(session=session, context=context, parameters=parameters)
        except Exception:
            logger.exception("OpenRouter captioning failed.")
            return OperatorResult(
                success=False,
                message="OpenRouter captioning failed. Check the server log for details.",
            )


def _run(
    *, session: Session, context: ExecutionContext, parameters: dict[str, Any]
) -> OperatorResult:
    """Caption the images in scope and report what happened."""
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not api_key:
        return OperatorResult(success=False, message=_MISSING_KEY_MESSAGE)

    jobs = _find_jobs(
        session=session,
        context=context,
        image_filter=_resolve_image_filter(context_filter=context.context_filter),
    )
    if not jobs:
        return OperatorResult(
            success=True, message="No images to caption in the current view."
        )

    tally = _caption_images(
        session=session,
        collection_id=context.collection_id,
        jobs=jobs,
        model=_text(parameters, "model", _DEFAULT_MODEL),
        prompt=_text(parameters, "prompt", _DEFAULT_PROMPT),
        max_image_edge=_int(parameters, "max_image_edge", _DEFAULT_MAX_IMAGE_EDGE),
        max_concurrency=_int(parameters, "max_concurrency", _DEFAULT_MAX_CONCURRENCY),
        api_key=api_key,
    )
    return _build_result(tally=tally)


def _text(parameters: Mapping[str, Any], name: str, default: str) -> str:
    """Read a string parameter, falling back to the default when absent or blank."""
    value = parameters.get(name)
    return value.strip() or default if isinstance(value, str) else default


def _int(parameters: Mapping[str, Any], name: str, default: int) -> int:
    """Read a positive int parameter, falling back to the default when unusable.

    `max_image_edge` accepts 0 to mean "do not resize", so 0 is passed through and
    only negative values fall back.
    """
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _resolve_image_filter(*, context_filter: AnyFilter | None) -> ImageFilter | None:
    """Narrow the discriminated filter union down to an image filter.

    Which images to caption is decided in the GUI, so a view filtered to uncaptioned
    images is the way to avoid captioning an image twice.
    """
    if isinstance(context_filter, SampleFilter):
        return ImageFilter(sample_filter=context_filter)
    if isinstance(context_filter, ImageFilter):
        return context_filter
    return None


def _find_jobs(
    *, session: Session, context: ExecutionContext, image_filter: ImageFilter | None
) -> list[CaptionJob]:
    """Load the images in scope as session-free captioning jobs."""
    result = image_resolver.get_all_by_collection_id(
        session=session, collection_id=context.collection_id, filters=image_filter
    )
    return [
        CaptionJob(sample_id=sample.sample_id, file_path_abs=sample.file_path_abs)
        for sample in result.samples
    ]


def _caption_images(
    *,
    session: Session,
    collection_id: UUID,
    jobs: Sequence[CaptionJob],
    model: str,
    prompt: str,
    max_image_edge: int,
    max_concurrency: int,
    api_key: str,
) -> CaptionTally:
    """Caption every job concurrently and persist the captions that succeed."""
    logger.info(
        "Captioning %d image(s) with %s at concurrency %d.",
        len(jobs),
        model,
        max_concurrency,
    )
    config = RequestConfig(api_key=api_key, model=model, prompt=prompt)
    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
    )
    tally = CaptionTally()
    pending: list[CaptionCreate] = []

    with (
        httpx.Client(
            timeout=openrouter_client.REQUEST_TIMEOUT, limits=limits
        ) as client,
        ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="openrouter-caption",
        ) as pool,
    ):
        futures = {
            pool.submit(
                _caption_one,
                job=job,
                client=client,
                config=config,
                max_image_edge=max_image_edge,
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            caption = _result_or_record_failure(future=future, job=job, tally=tally)
            if caption is None:
                continue
            pending.append(CaptionCreate(parent_sample_id=job.sample_id, text=caption))
            if len(pending) >= _DB_FLUSH_BATCH_SIZE:
                tally.stored += _store_captions(
                    session=session, collection_id=collection_id, captions=pending
                )
                pending.clear()

    tally.stored += _store_captions(
        session=session, collection_id=collection_id, captions=pending
    )
    logger.info("Captioned %d/%d image(s).", tally.stored, len(jobs))
    return tally


def _caption_one(
    *,
    job: CaptionJob,
    client: httpx.Client,
    config: RequestConfig,
    max_image_edge: int,
) -> str:
    """Encode one image and return its caption.

    Runs in a worker thread. Encoding happens here rather than up front so that the
    JPEG bytes of all images are never held in memory at once.
    """
    data_url = _encode_image_as_data_url(
        file_path=job.file_path_abs, max_edge=max_image_edge
    )
    return openrouter_client.request_caption(
        client=client, config=config, image_data_url=data_url
    )


def _encode_image_as_data_url(*, file_path: str, max_edge: int) -> str:
    """Read, orient, downscale and JPEG-encode an image as a base64 data URL.

    Raises:
        OpenRouterError: If the file cannot be opened, decoded or encoded.
    """
    try:
        with PIL.Image.open(file_path) as opened:
            oriented = PIL.ImageOps.exif_transpose(opened)
            image = (oriented if oriented is not None else opened).convert("RGB")
        if max_edge > 0 and max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), PIL.Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    except Exception as exc:
        raise OpenRouterError(f"Could not encode image '{file_path}': {exc}") from exc
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _result_or_record_failure(
    *, future: Future[str], job: CaptionJob, tally: CaptionTally
) -> str | None:
    """Return the caption of a finished job, or None after recording its failure.

    Failures are isolated per image so that one unreadable file or one rejected
    request does not abort the whole run.
    """
    try:
        return future.result()
    except OpenRouterError as exc:
        tally.failed += 1
        tally.first_error = tally.first_error or str(exc)
        logger.warning("Captioning failed for %s: %s", job.file_path_abs, exc)
        return None
    except Exception:
        tally.failed += 1
        tally.first_error = tally.first_error or "Unexpected error, see the server log."
        logger.exception("Captioning failed for %s.", job.file_path_abs)
        return None


def _store_captions(
    *, session: Session, collection_id: UUID, captions: Sequence[CaptionCreate]
) -> int:
    if not captions:
        return 0
    caption_resolver.create_many(
        session=session, parent_collection_id=collection_id, captions=captions
    )
    return len(captions)


def _build_result(*, tally: CaptionTally) -> OperatorResult:
    """Turn the run counts into a result for the GUI.

    A run where nothing succeeded is reported as a failure, because that points at a
    misconfiguration such as a wrong model slug or a missing network connection.
    """
    if tally.stored == 0 and tally.failed > 0:
        return OperatorResult(
            success=False,
            message=(
                f"Captioning failed for all {tally.failed} image(s). "
                f"First error: {tally.first_error}"
            ),
        )
    parts = [f"Captioned {tally.stored} image(s)."]
    if tally.failed:
        parts.append(f"{tally.failed} failed (first error: {tally.first_error}).")
    return OperatorResult(success=True, message=" ".join(parts))
