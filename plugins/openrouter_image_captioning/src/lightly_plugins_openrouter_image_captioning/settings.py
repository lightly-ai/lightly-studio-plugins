"""Parameter definitions and validated settings for the captioning operator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lightly_studio.plugins.parameter import (
    BaseParameter,
    BoolParameter,
    FloatParameter,
    IntParameter,
    StringParameter,
)

API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
MODELS_URL = "https://openrouter.ai/models"

PARAM_MODEL = "model"
PARAM_PROMPT = "prompt"
PARAM_BASE_URL = "base_url"
PARAM_MAX_TOKENS = "max_tokens"
PARAM_TEMPERATURE = "temperature"
PARAM_SKIP_CAPTIONED = "skip_captioned"
PARAM_MAX_SAMPLES = "max_samples"
PARAM_MAX_IMAGE_EDGE = "max_image_edge"
PARAM_MAX_CONCURRENCY = "max_concurrency"
PARAM_PROVIDER_SORT = "provider_sort"
PARAM_REQUEST_TIMEOUT = "request_timeout"
PARAM_MAX_RETRIES = "max_retries"

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
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
_DEFAULT_PROVIDER_SORT = "throughput"
_DEFAULT_REQUEST_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 3

# An empty choice means "let OpenRouter decide", keeping its default load balancing.
_PROVIDER_SORTS = ("", "throughput", "latency", "price")
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
        base_url: OpenAI-compatible API base URL.
        max_tokens: Upper bound on caption length in tokens.
        temperature: Sampling temperature.
        skip_captioned: Whether images that already have a caption are skipped.
        max_samples: Maximum images per run, or 0 for no limit.
        max_image_edge: Longest edge in pixels after downscaling, or 0 to not resize.
        max_concurrency: Number of images captioned in parallel.
        provider_sort: How OpenRouter orders providers, or empty for its default.
        request_timeout: Per-request timeout in seconds.
        max_retries: Retries per image on rate limits, server errors and network errors.
    """

    model: str
    prompt: str
    base_url: str
    max_tokens: int
    temperature: float
    skip_captioned: bool
    max_samples: int
    max_image_edge: int
    max_concurrency: int
    provider_sort: str
    request_timeout: float
    max_retries: int


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
        StringParameter(
            name=PARAM_BASE_URL,
            required=False,
            default=_DEFAULT_BASE_URL,
            description=(
                "OpenAI-compatible API base URL. Change only to use a proxy or a "
                "different gateway."
            ),
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
        BoolParameter(
            name=PARAM_SKIP_CAPTIONED,
            required=False,
            default=True,
            description=(
                "Skip images that already have at least one caption. Disable to add an "
                "additional caption to every image."
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
        StringParameter(
            name=PARAM_PROVIDER_SORT,
            required=False,
            default=_DEFAULT_PROVIDER_SORT,
            description=(
                "How OpenRouter picks between providers serving the model: 'throughput' "
                "for bulk speed, 'latency' for fastest first token, 'price' for lowest "
                "cost. Leave empty for OpenRouter's default load balancing."
            ),
        ),
        FloatParameter(
            name=PARAM_REQUEST_TIMEOUT,
            required=False,
            default=_DEFAULT_REQUEST_TIMEOUT,
            description="Per-request timeout in seconds.",
        ),
        IntParameter(
            name=PARAM_MAX_RETRIES,
            required=False,
            default=_DEFAULT_MAX_RETRIES,
            description=(
                "Retries per image on rate limits (429), server errors (5xx) and "
                "network errors."
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

    provider_sort = _text(
        parameters, PARAM_PROVIDER_SORT, _DEFAULT_PROVIDER_SORT, keep_blank=True
    ).lower()
    if provider_sort not in _PROVIDER_SORTS:
        readable = ", ".join(f"'{option}'" for option in _PROVIDER_SORTS if option)
        raise ParameterError(
            f"Parameter '{PARAM_PROVIDER_SORT}' must be one of {readable}, or empty."
        )

    return CaptionSettings(
        model=_text(parameters, PARAM_MODEL, _DEFAULT_MODEL),
        prompt=_text(parameters, PARAM_PROMPT, _DEFAULT_PROMPT),
        base_url=_text(parameters, PARAM_BASE_URL, _DEFAULT_BASE_URL),
        max_tokens=_number(
            parameters, PARAM_MAX_TOKENS, _DEFAULT_MAX_TOKENS, int, 16, 4096
        ),
        temperature=_number(
            parameters, PARAM_TEMPERATURE, _DEFAULT_TEMPERATURE, float, 0.0, 2.0
        ),
        skip_captioned=_boolean(parameters, PARAM_SKIP_CAPTIONED, default=True),
        max_samples=_number(
            parameters, PARAM_MAX_SAMPLES, _DEFAULT_MAX_SAMPLES, int, 0, 100_000
        ),
        max_image_edge=max_image_edge,
        max_concurrency=_number(
            parameters, PARAM_MAX_CONCURRENCY, _DEFAULT_MAX_CONCURRENCY, int, 1, 64
        ),
        provider_sort=provider_sort,
        request_timeout=_number(
            parameters,
            PARAM_REQUEST_TIMEOUT,
            _DEFAULT_REQUEST_TIMEOUT,
            float,
            1.0,
            600.0,
        ),
        max_retries=_number(
            parameters, PARAM_MAX_RETRIES, _DEFAULT_MAX_RETRIES, int, 0, 10
        ),
    )


def _text(
    parameters: Mapping[str, Any],
    name: str,
    default: str,
    *,
    keep_blank: bool = False,
) -> str:
    """Read a string parameter, falling back to the default when absent.

    Raises:
        ParameterError: If the value is not text.
    """
    value = parameters.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ParameterError(f"Parameter '{name}' must be text.")
    stripped = value.strip()
    # A blank choice stays selectable, so it must not fall back to the default.
    return stripped if keep_blank else (stripped or default)


def _boolean(parameters: Mapping[str, Any], name: str, *, default: bool) -> bool:
    """Read a boolean parameter.

    Raises:
        ParameterError: If the value is not a boolean.
    """
    value = parameters.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ParameterError(f"Parameter '{name}' must be true or false.")
    return value


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
