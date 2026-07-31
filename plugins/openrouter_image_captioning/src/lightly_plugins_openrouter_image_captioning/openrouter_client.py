"""Minimal OpenRouter chat-completions client for image captioning."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
PROVIDER_SORT = "throughput"
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 3
MAX_TOKENS = 200
TEMPERATURE = 0.2

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})
_MAX_BACKOFF_SECONDS = 30.0
_ERROR_BODY_CHARS = 300

logger = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter does not return a usable caption."""


@dataclass(frozen=True)
class RequestConfig:
    """Immutable request configuration shared by all worker threads."""

    api_key: str
    model: str
    prompt: str


def request_caption(
    *, client: httpx.Client, config: RequestConfig, image_data_url: str
) -> str:
    """Request a caption for a single image.

    Args:
        client: Shared HTTP client. Safe to use from multiple threads.
        config: Request configuration.
        image_data_url: The image as a ``data:`` URL.

    Returns:
        The caption text.

    Raises:
        OpenRouterError: If the request fails or the response holds no caption.
    """
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config.prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "provider": {"sort": PROVIDER_SORT},
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lightly.ai",
        "X-OpenRouter-Title": "LightlyStudio",
    }
    response = _post_with_retries(
        client=client, headers=headers, payload=payload, model=config.model
    )
    try:
        body = response.json()
    except ValueError as exc:
        raise OpenRouterError(
            f"OpenRouter returned a non-JSON response: {exc}"
        ) from exc
    return _extract_caption(body)


def _post_with_retries(
    *,
    client: httpx.Client,
    headers: dict[str, str],
    payload: dict[str, Any],
    model: str,
) -> httpx.Response:
    """POST the payload, retrying transient failures with capped backoff.

    Raises:
        OpenRouterError: On a non-retryable status or after exhausting retries.
    """
    attempts = MAX_RETRIES + 1
    for attempt in range(attempts):
        is_last = attempt == attempts - 1
        try:
            response = client.post(
                BASE_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if is_last:
                raise OpenRouterError(f"Request to OpenRouter failed: {exc}") from exc
            time.sleep(_retry_delay_seconds(attempt=attempt))
            continue

        if response.status_code == httpx.codes.OK:
            return response
        if response.status_code not in _RETRYABLE_STATUS_CODES:
            raise OpenRouterError(_error_message(response=response, model=model))
        if is_last:
            raise OpenRouterError(
                f"OpenRouter returned HTTP {response.status_code} after "
                f"{attempts} attempt(s): {_excerpt(response)}"
            )
        time.sleep(_retry_delay_seconds(attempt=attempt))

    raise OpenRouterError("Request to OpenRouter failed.")


def _error_message(*, response: httpx.Response, model: str) -> str:
    """Build an actionable message for a status that must not be retried."""
    excerpt = _excerpt(response)
    status = response.status_code
    if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
        return (
            f"OpenRouter rejected the credentials (HTTP {status}). Check that "
            f"OPENROUTER_API_KEY is valid and still active: {excerpt}"
        )
    if status == httpx.codes.PAYMENT_REQUIRED:
        return (
            "OpenRouter reports insufficient credits (HTTP 402). Top up your "
            f"account at https://openrouter.ai/credits: {excerpt}"
        )
    return f"OpenRouter returned HTTP {status} for model '{model}': {excerpt}"


def _excerpt(response: httpx.Response) -> str:
    """Return a short, single-line excerpt of a response body for error messages."""
    return " ".join(response.text.split())[:_ERROR_BODY_CHARS]


def _retry_delay_seconds(*, attempt: int) -> float:
    """Return how long to wait before the next attempt.

    Exponential backoff with jitter, so concurrent workers do not all retry at the
    same instant.
    """
    backoff = min(2.0**attempt, _MAX_BACKOFF_SECONDS)
    return backoff * (0.5 + random.random() / 2.0)


def _extract_caption(body: object) -> str:
    """Extract the caption from a chat-completions response body.

    Raises:
        OpenRouterError: If the body carries an error or holds no usable caption.
    """
    if not isinstance(body, Mapping):
        raise OpenRouterError("Unexpected response shape from OpenRouter.")
    error = body.get("error")
    if error is not None:
        raise OpenRouterError(f"OpenRouter returned an error: {error}")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("OpenRouter returned no choices.")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise OpenRouterError("Unexpected choice shape from OpenRouter.")

    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    text = _content_to_text(content).strip()
    if not text:
        raise OpenRouterError("OpenRouter returned an empty caption.")
    if choice.get("finish_reason") == "length":
        logger.info("Caption was truncated by the MAX_TOKENS cap.")
    return text


def _content_to_text(content: object) -> str:
    """Flatten a message content field into plain text.

    Most providers return a string, but some return a list of typed content parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = (p.get("text") for p in content if isinstance(p, Mapping))
        return "\n".join(p for p in parts if isinstance(p, str))
    return ""
