"""OpenRouter image captioning plugin for Lightly Studio."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.plugins.parameter import BaseParameter
from lightly_studio.resolvers import image_resolver
from lightly_studio.resolvers.image_filter import ImageFilter
from sqlmodel import Session

from lightly_plugins_openrouter_image_captioning import (
    captioning,
    image_filters,
    parameter_values,
    settings,
)
from lightly_plugins_openrouter_image_captioning.captioning import (
    CaptionJob,
    CaptionTally,
)
from lightly_plugins_openrouter_image_captioning.settings import CaptionSettings

logger = logging.getLogger(__name__)

_MISSING_KEY_MESSAGE = (
    f"{settings.API_KEY_ENV_VAR} is not set. Create a key at "
    "https://openrouter.ai/keys, export it (or put it in a .env file where you start "
    "LightlyStudio) and restart LightlyStudio."
)
_UNEXPECTED_ERROR_MESSAGE = (
    "OpenRouter captioning failed. Check the server log for details."
)


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
        return settings.build_parameters()

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
            # The API route does not wrap execute(), so anything escaping here would
            # surface as an opaque HTTP 500 instead of a message in the GUI.
            logger.exception("OpenRouter captioning failed.")
            return OperatorResult(success=False, message=_UNEXPECTED_ERROR_MESSAGE)


def _run(
    *,
    session: Session,
    context: ExecutionContext,
    parameters: dict[str, Any],
) -> OperatorResult:
    """Caption the images in scope and report what happened.

    Raises:
        Exception: Any unexpected error, which `execute` turns into a failed result.
    """
    # The credentials are checked first because it is the cheapest failure to detect.
    api_key = os.environ.get(settings.API_KEY_ENV_VAR, "").strip()
    if not api_key:
        return OperatorResult(success=False, message=_MISSING_KEY_MESSAGE)

    try:
        caption_settings = settings.read_settings(parameters=parameters)
    except parameter_values.ParameterError as exc:
        return OperatorResult(success=False, message=str(exc))

    resolution = image_filters.resolve_image_filter(
        context_filter=context.context_filter,
        skip_captioned=caption_settings.skip_captioned,
    )
    if resolution.conflict_message is not None:
        return OperatorResult(success=True, message=resolution.conflict_message)

    jobs = _find_jobs(
        session=session, context=context, image_filter=resolution.image_filter
    )
    if not jobs:
        return OperatorResult(
            success=True, message=_empty_message(caption_settings=caption_settings)
        )

    selected, truncated = _limit(jobs=jobs, max_samples=caption_settings.max_samples)
    tally = captioning.caption_images(
        session=session,
        collection_id=context.collection_id,
        jobs=selected,
        settings=caption_settings,
        api_key=api_key,
    )
    return _result(tally=tally, truncated=truncated)


def _find_jobs(
    *,
    session: Session,
    context: ExecutionContext,
    image_filter: ImageFilter | None,
) -> list[CaptionJob]:
    """Load the images in scope as session-free captioning jobs."""
    result = image_resolver.get_all_by_collection_id(
        session=session, collection_id=context.collection_id, filters=image_filter
    )
    return [
        CaptionJob(sample_id=sample.sample_id, file_path_abs=sample.file_path_abs)
        for sample in result.samples
    ]


def _limit(
    *, jobs: Sequence[CaptionJob], max_samples: int
) -> tuple[Sequence[CaptionJob], int]:
    """Cap the number of jobs and report how many were dropped.

    The cap is applied by slicing rather than through `Paginated`, whose limit cannot
    exceed 100.

    Returns:
        The jobs to run, and the number of images left out.
    """
    if max_samples <= 0 or len(jobs) <= max_samples:
        return jobs, 0
    return jobs[:max_samples], len(jobs) - max_samples


def _empty_message(*, caption_settings: CaptionSettings) -> str:
    """Explain an empty selection, which is usually caused by skipping captioned images."""
    if caption_settings.skip_captioned:
        return "No images to caption in the current view. Every image already has a caption."
    return "No images to caption in the current view."


def _result(*, tally: CaptionTally, truncated: int) -> OperatorResult:
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
    if truncated:
        parts.append(f"{truncated} image(s) skipped by {settings.PARAM_MAX_SAMPLES}.")
    return OperatorResult(success=True, message=" ".join(parts))
