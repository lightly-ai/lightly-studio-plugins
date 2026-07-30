"""Parameter definitions and validated settings for the captioning operator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lightly_studio.plugins.parameter import (
    BaseParameter,
    FloatParameter,
    IntParameter,
    StringParameter,
)

API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
MODELS_URL = "https://openrouter.ai/models"

BASE_URL = "https://openrouter.ai/api/v1"
"""OpenRouter's OpenAI-compatible API root."""

# The settings below are deliberately not exposed as parameters: their defaults work for
# every run we know of, and a user has no basis on which to choose a different value.
PROVIDER_SORT = "throughput"
"""How OpenRouter orders the providers of a model. Bulk captioning wants throughput."""
REQUEST_TIMEOUT = 60.0
"""Per-request timeout in seconds. Generous for a caption; a stall is handled by a retry."""
MAX_RETRIES = 3
"""Retries per image on rate limits, server errors and network errors."""

PARAM_MODEL = "model"
PARAM_PROMPT = "prompt"
PARAM_MAX_TOKENS = "max_tokens"
PARAM_TEMPERATURE = "temperature"
PARAM_MAX_SAMPLES = "max_samples"
PARAM_MAX_IMAGE_EDGE = "max_image_edge"
PARAM_MAX_CONCURRENCY = "max_concurrency"

_DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
_DEFAULT_PROMPT = (
    "Describe this image in one or two concise sentences. Name the main objects, their "
    "notable attributes and the overall scene. Do not begin with 'The image shows'."
)
_DEFAULT_MAX_TOKENS = 200
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_SAMPLES = 200
_DEFAULT_MAX_IMAGE_EDGE = 256
_DEFAULT_MAX_CONCURRENCY = 16

# Below 64 an image carries too little detail to caption; 0 means "do not resize".
_MIN_USEFUL_IMAGE_EDGE = 64


class ParameterError(ValueError):
    """Raised when a supplied parameter value cannot be used."""


@dataclass(frozen=True)
class CaptionSettings:
    """Validated operator parameters for one captioning run.

    Attributes:
        model: OpenRouter slug of a vision-capable model.
        prompt: Instruction sent alongside each image.
        max_tokens: Upper bound on caption length in tokens.
        temperature: Sampling temperature.
        max_samples: Maximum images per run, or 0 for no limit.
        max_image_edge: Longest edge in pixels after downscaling, or 0 to not resize.
        max_concurrency: Number of images captioned in parallel.
    """

    model: str
    prompt: str
    max_tokens: int
    temperature: float
    max_samples: int
    max_image_edge: int
    max_concurrency: int


def build_parameters() -> list[BaseParameter]:
    """Return the operator parameters in the order they are shown in the GUI."""
    return [
        StringParameter(
            name=PARAM_MODEL,
            required=True,
            default=_DEFAULT_MODEL,
            description=(
                "OpenRouter model slug of a vision-capable model. Browse and compare "
                f"models at {MODELS_URL} (filter by image input)."
            ),
        ),
        StringParameter(
            name=PARAM_PROMPT,
            required=True,
            default=_DEFAULT_PROMPT,
            description="Instruction sent to the model together with the image.",
        ),
        IntParameter(
            name=PARAM_MAX_TOKENS,
            required=False,
            default=_DEFAULT_MAX_TOKENS,
            description=(
                "Upper bound on caption length in tokens. This is a cap, not a target, "
                "so lowering it does not speed up a run."
            ),
        ),
        FloatParameter(
            name=PARAM_TEMPERATURE,
            required=False,
            default=_DEFAULT_TEMPERATURE,
            description=(
                "Sampling temperature. Use 0.0 for the most reproducible captions."
            ),
        ),
        IntParameter(
            name=PARAM_MAX_SAMPLES,
            required=False,
            default=_DEFAULT_MAX_SAMPLES,
            description=(
                "Maximum number of images to caption in one run. 0 means no limit."
            ),
        ),
        IntParameter(
            name=PARAM_MAX_IMAGE_EDGE,
            required=False,
            default=_DEFAULT_MAX_IMAGE_EDGE,
            description=(
                "Downscale images so the longest edge is at most this many pixels "
                "before upload. Lower is cheaper and faster, but fine detail is lost; "
                "raise it if captions miss small objects or text. 0 uploads the image "
                "at its original size."
            ),
        ),
        IntParameter(
            name=PARAM_MAX_CONCURRENCY,
            required=False,
            default=_DEFAULT_MAX_CONCURRENCY,
            description=(
                "Number of images captioned in parallel. Raise this to speed up large "
                "runs; lower it if requests start hitting rate limits."
            ),
        ),
    ]


def read_settings(*, parameters: Mapping[str, Any]) -> CaptionSettings:
    """Coerce and range-check the raw parameter dict.

    Args:
        parameters: The unvalidated parameters supplied by the caller.

    Returns:
        The validated settings.

    Raises:
        ParameterError: If a value cannot be used.
    """
    max_image_edge = _number(
        parameters, PARAM_MAX_IMAGE_EDGE, _DEFAULT_MAX_IMAGE_EDGE, int, 0, 4096
    )
    if 0 < max_image_edge < _MIN_USEFUL_IMAGE_EDGE:
        raise ParameterError(
            f"Parameter '{PARAM_MAX_IMAGE_EDGE}' must be at least "
            f"{_MIN_USEFUL_IMAGE_EDGE}, or 0 to upload images unresized."
        )

    return CaptionSettings(
        model=_text(parameters, PARAM_MODEL, _DEFAULT_MODEL),
        prompt=_text(parameters, PARAM_PROMPT, _DEFAULT_PROMPT),
        max_tokens=_number(
            parameters, PARAM_MAX_TOKENS, _DEFAULT_MAX_TOKENS, int, 16, 4096
        ),
        temperature=_number(
            parameters, PARAM_TEMPERATURE, _DEFAULT_TEMPERATURE, float, 0.0, 2.0
        ),
        max_samples=_number(
            parameters, PARAM_MAX_SAMPLES, _DEFAULT_MAX_SAMPLES, int, 0, 100_000
        ),
        max_image_edge=max_image_edge,
        max_concurrency=_number(
            parameters, PARAM_MAX_CONCURRENCY, _DEFAULT_MAX_CONCURRENCY, int, 1, 64
        ),
    )


def _text(parameters: Mapping[str, Any], name: str, default: str) -> str:
    """Read a string parameter, falling back to the default when absent or blank.

    Raises:
        ParameterError: If the value is not text.
    """
    value = parameters.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ParameterError(f"Parameter '{name}' must be text.")
    return value.strip() or default


def _number(
    parameters: Mapping[str, Any],
    name: str,
    default: Any,
    kind: type,
    minimum: Any,
    maximum: Any,
) -> Any:
    """Read an int or float parameter and check it against its allowed range.

    Raises:
        ParameterError: If the value is not of the expected type or is out of range.
    """
    value = parameters.get(name)
    if value is None:
        value = default
    # bool is a subclass of int, so reject it before the isinstance check below.
    elif isinstance(value, bool):
        raise ParameterError(f"Parameter '{name}' must be a number.")
    elif kind is float and isinstance(value, int):
        value = float(value)
    elif not isinstance(value, kind):
        expected = "a whole number" if kind is int else "a number"
        raise ParameterError(f"Parameter '{name}' must be {expected}.")
    if not minimum <= value <= maximum:
        raise ParameterError(
            f"Parameter '{name}' must be between {minimum} and {maximum}."
        )
    return value
