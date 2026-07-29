"""Concurrent captioning of images and persistence of the results.

All HTTP work happens in worker threads while every database access stays on the calling
thread, because the SQLModel session is not thread-safe.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import statistics
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

import httpx
from lightly_studio.models.caption import CaptionCreate
from lightly_studio.resolvers import caption_resolver
from sqlmodel import Session

from lightly_plugins_openrouter_image_captioning import (
    image_encoding,
    openrouter_client,
)
from lightly_plugins_openrouter_image_captioning.image_encoding import (
    ImageEncodingError,
)
from lightly_plugins_openrouter_image_captioning.openrouter_client import (
    OpenRouterError,
    OpenRouterRequestConfig,
)
from lightly_plugins_openrouter_image_captioning.settings import CaptionSettings

logger = logging.getLogger(__name__)

# Kept as a constant rather than a parameter to keep the operator form manageable.
_JPEG_QUALITY = 85
# Captions are flushed in batches so that a cancelled request keeps completed work.
_DB_FLUSH_BATCH_SIZE = 200
_THREAD_NAME_PREFIX = "openrouter-caption"
# Identifies the run in OpenRouter's activity view, next to other tools on the same key.
_SESSION_ID_PREFIX = "lightly-studio-captioning"
_SESSION_ID_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
_SESSION_ID_SUFFIX_CHARS = 6


@dataclass(frozen=True)
class CaptionJob:
    """Session-free data needed to caption one image inside a worker thread.

    Worker threads must not touch the session or any ORM object, so everything they need
    is snapshotted into immutable values before the work is submitted. This also survives
    the commit inside `caption_resolver.create_many`, which expires ORM objects.

    Attributes:
        sample_id: Identifier of the image sample the caption belongs to.
        file_path_abs: Absolute path of the image file on disk.
    """

    sample_id: UUID
    file_path_abs: str


@dataclass(frozen=True)
class _CaptionResult:
    """One finished caption and how long its request took.

    Attributes:
        text: The caption returned by the model.
        duration_s: Time spent encoding the image and awaiting the response.
    """

    text: str
    duration_s: float


@dataclass
class CaptionTally:
    """Running counts and timings of a captioning run.

    Attributes:
        stored: Number of captions written to the database.
        failed: Number of images that could not be captioned.
        first_error: Message of the first failure, used to explain the run to the user.
        elapsed_s: Wall-clock duration of the whole run in seconds.
        request_durations_s: Duration of each successful request in seconds. Failures are
            left out because a timed-out request only measures the timeout.
    """

    stored: int = 0
    failed: int = 0
    first_error: str | None = None
    elapsed_s: float = 0.0
    request_durations_s: list[float] = field(default_factory=list)


def caption_images(
    *,
    session: Session,
    collection_id: UUID,
    jobs: Sequence[CaptionJob],
    settings: CaptionSettings,
    api_key: str,
) -> CaptionTally:
    """Caption every job concurrently and persist the captions that succeed.

    Args:
        session: Database session. Only used on the calling thread.
        collection_id: Collection the captioned images belong to.
        jobs: The images to caption.
        settings: Validated run settings.
        api_key: OpenRouter API key.

    Returns:
        The counts and timings of the run.
    """
    tally = CaptionTally()
    pending: list[CaptionCreate] = []
    session_id = new_session_id()
    logger.info(
        "Captioning %d image(s) with %s at concurrency %d as OpenRouter session '%s'.",
        len(jobs),
        settings.model,
        settings.max_concurrency,
        session_id,
    )
    started_at = time.monotonic()
    with contextlib.ExitStack() as stack:
        client = stack.enter_context(_build_client(settings=settings))
        pool = stack.enter_context(_build_pool(settings=settings))
        futures = _submit(
            pool=pool,
            client=client,
            jobs=jobs,
            settings=settings,
            api_key=api_key,
            session_id=session_id,
        )
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            result = _caption_or_record_failure(future=future, job=job, tally=tally)
            if result is None:
                continue
            pending.append(
                CaptionCreate(parent_sample_id=job.sample_id, text=result.text)
            )
            if len(pending) >= _DB_FLUSH_BATCH_SIZE:
                tally.stored += store_captions(
                    session=session, collection_id=collection_id, captions=pending
                )
                pending.clear()
    tally.stored += store_captions(
        session=session, collection_id=collection_id, captions=pending
    )
    tally.elapsed_s = time.monotonic() - started_at
    _log_summary(tally=tally, total=len(jobs))
    return tally


def new_session_id() -> str:
    """Return an identifier that groups every request of one captioning run.

    The identifier is built from a fixed prefix, a UTC start timestamp and a short random
    suffix, for example `lightly-studio-captioning-20260729T150312Z-a1b2c3`. The prefix
    tells the run apart from other tools sharing the API key, the timestamp makes runs
    sortable and easy to match against when something happened, and the random suffix
    keeps two runs started in the same second from being merged into one session.
    """
    started_at = datetime.now(timezone.utc).strftime(_SESSION_ID_TIMESTAMP_FORMAT)
    suffix = uuid.uuid4().hex[:_SESSION_ID_SUFFIX_CHARS]
    return f"{_SESSION_ID_PREFIX}-{started_at}-{suffix}"


def store_captions(
    *, session: Session, collection_id: UUID, captions: Sequence[CaptionCreate]
) -> int:
    """Persist a batch of captions and return how many were written."""
    if not captions:
        return 0
    caption_resolver.create_many(
        session=session, parent_collection_id=collection_id, captions=captions
    )
    return len(captions)


def _build_client(*, settings: CaptionSettings) -> httpx.Client:
    """Build the shared HTTP client, sized to the requested concurrency."""
    limits = httpx.Limits(
        max_connections=settings.max_concurrency,
        max_keepalive_connections=settings.max_concurrency,
    )
    return httpx.Client(timeout=settings.request_timeout, limits=limits)


def _build_pool(*, settings: CaptionSettings) -> ThreadPoolExecutor:
    """Build the worker pool that issues the HTTP requests."""
    return ThreadPoolExecutor(
        max_workers=settings.max_concurrency, thread_name_prefix=_THREAD_NAME_PREFIX
    )


def _submit(
    *,
    pool: ThreadPoolExecutor,
    client: httpx.Client,
    jobs: Sequence[CaptionJob],
    settings: CaptionSettings,
    api_key: str,
    session_id: str,
) -> dict[Future[_CaptionResult], CaptionJob]:
    """Submit every job and return a mapping from future back to its job.

    The mapping is needed because a failure has to be reported against the image that
    caused it, which `as_completed` alone does not tell us.
    """
    config = _build_request_config(
        settings=settings, api_key=api_key, session_id=session_id
    )
    return {
        pool.submit(
            _caption_one,
            job=job,
            client=client,
            config=config,
            max_image_edge=settings.max_image_edge,
        ): job
        for job in jobs
    }


def _build_request_config(
    *, settings: CaptionSettings, api_key: str, session_id: str
) -> OpenRouterRequestConfig:
    """Translate the run settings into an immutable request configuration."""
    return OpenRouterRequestConfig(
        base_url=settings.base_url,
        api_key=api_key,
        model=settings.model,
        prompt=settings.prompt,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        max_retries=settings.max_retries,
        timeout=settings.request_timeout,
        provider_sort=settings.provider_sort,
        session_id=session_id,
    )


def _caption_one(
    *,
    job: CaptionJob,
    client: httpx.Client,
    config: OpenRouterRequestConfig,
    max_image_edge: int,
) -> _CaptionResult:
    """Encode one image and return its caption. Runs in a worker thread.

    Encoding happens here rather than up front so that the JPEG bytes of all images are
    never held in memory at once.

    Raises:
        ImageEncodingError: If the image cannot be read or encoded.
        OpenRouterError: If OpenRouter returns no usable caption.
    """
    started_at = time.monotonic()
    data_url = image_encoding.encode_image_as_data_url(
        file_path=job.file_path_abs,
        max_edge=max_image_edge,
        jpeg_quality=_JPEG_QUALITY,
    )
    text = openrouter_client.request_caption(
        client=client, config=config, image_data_url=data_url
    )
    return _CaptionResult(text=text, duration_s=time.monotonic() - started_at)


def _caption_or_record_failure(
    *, future: Future[_CaptionResult], job: CaptionJob, tally: CaptionTally
) -> _CaptionResult | None:
    """Return the result of a finished job, or None after recording its failure.

    Failures are isolated per image so that one unreadable file or one rejected request
    does not abort the whole run.
    """
    try:
        result = future.result()
    except (OpenRouterError, ImageEncodingError) as exc:
        tally.failed += 1
        tally.first_error = tally.first_error or str(exc)
        logger.warning("Captioning failed for %s: %s", job.file_path_abs, exc)
        return None
    except Exception:
        tally.failed += 1
        tally.first_error = tally.first_error or "Unexpected error, see the server log."
        logger.exception("Captioning failed for %s.", job.file_path_abs)
        return None
    tally.request_durations_s.append(result.duration_s)
    logger.debug("Captioned %s in %.1fs.", job.file_path_abs, result.duration_s)
    return result


def _log_summary(*, tally: CaptionTally, total: int) -> None:
    """Log how long the run took and how fast the requests were.

    Both numbers are needed to tell a slow provider apart from too little concurrency: if
    the median request is fast but throughput is far below `concurrency / median`, the run
    is bottlenecked somewhere other than the model.
    """
    throughput = tally.stored / tally.elapsed_s if tally.elapsed_s > 0 else 0.0
    summary = (
        f"Captioned {tally.stored}/{total} image(s) in {tally.elapsed_s:.1f}s "
        f"({throughput:.1f} image/s)"
    )
    if not tally.request_durations_s:
        logger.info("%s.", summary)
        return
    logger.info(
        "%s. Request time median %.1fs, slowest %.1fs.",
        summary,
        statistics.median(tally.request_durations_s),
        max(tally.request_durations_s),
    )
