"""Minimal OpenRouter chat-completions client for image captioning."""

from __future__ import annotations

import email.utils
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_PATH = "/chat/completions"
# Statuses worth retrying: request timeouts, conflicts, "too early", rate limits, and
# transient upstream/gateway failures. 522 and 524 are Cloudflare-specific timeouts.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})
_MAX_BACKOFF_SECONDS = 30.0
_MAX_RETRY_AFTER_SECONDS = 60.0
_ERROR_BODY_CHARS = 300
_REFERER = "https://lightly.ai"
_TITLE = "LightlyStudio"

SESSION_ID_MAX_LENGTH = 256
"""Longest `session_id` OpenRouter accepts."""


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter does not return a usable caption."""


@dataclass(frozen=True)
class OpenRouterRequestConfig:
    """Immutable request configuration shared by all worker threads.

    Attributes:
        base_url: OpenAI-compatible API base URL.
        api_key: OpenRouter API key.
        model: Model slug to caption with.
        prompt: Instruction sent alongside each image.
        max_tokens: Upper bound on caption length in tokens.
        temperature: Sampling temperature.
        max_retries: Retries per request on transient failures.
        timeout: Per-request timeout in seconds.
        provider_sort: How OpenRouter orders providers, or empty for its default.
        session_id: Groups all requests of one run for observability, or empty to send
            no grouping key. At most `SESSION_ID_MAX_LENGTH` characters.
    """

    base_url: str
    api_key: str
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    max_retries: int
    timeout: float
    provider_sort: str = ""
    session_id: str = ""


def request_caption(
    *, client: httpx.Client, config: OpenRouterRequestConfig, image_data_url: str
) -> str:
    """Request a caption for a single image.

    Args:
        client: Shared HTTP client. Safe to use from multiple threads.
        config: Request configuration.
        image_data_url: The image as a ``data:`` URL.

    Returns:
        The caption text.

    Raises:
        OpenRouterError: If the request fails or the response holds no usable caption.
    """
    response = _post_with_retries(
        client=client,
        url=config.base_url.rstrip("/") + _CHAT_COMPLETIONS_PATH,
        headers=_build_headers(api_key=config.api_key),
        payload=_build_payload(config=config, image_data_url=image_data_url),
        config=config,
    )
    try:
        body = response.json()
    except ValueError as exc:
        raise OpenRouterError(
            f"OpenRouter returned a non-JSON response: {exc}"
        ) from exc
    return _extract_caption_text(body)


def _build_payload(
    *, config: OpenRouterRequestConfig, image_data_url: str
) -> dict[str, Any]:
    """Build the OpenAI-compatible chat-completions request body."""
    payload: dict[str, Any] = {
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
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    if config.provider_sort:
        # Many models are served by several providers at very different speeds. Sorting
        # disables load balancing and tries providers in the requested order instead.
        payload["provider"] = {"sort": config.provider_sort}
    if config.session_id:
        # Groups the whole run in OpenRouter's activity view. It doubles as a sticky
        # routing key, which costs nothing here: captions share no cacheable prompt
        # prefix, and `provider.sort` has already disabled load balancing.
        payload["session_id"] = config.session_id[:SESSION_ID_MAX_LENGTH]
    return payload


def _build_headers(*, api_key: str) -> dict[str, str]:
    """Build the request headers, including OpenRouter's attribution headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter; they attribute usage to the caller.
        "HTTP-Referer": _REFERER,
        "X-OpenRouter-Title": _TITLE,
    }


def _post_with_retries(
    *,
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    config: OpenRouterRequestConfig,
) -> httpx.Response:
    """POST the payload, retrying transient failures with capped backoff.

    Args:
        client: Shared HTTP client.
        url: Fully qualified chat-completions URL.
        headers: Request headers.
        payload: Request body.
        config: Request configuration; supplies the retry count and timeout.

    Returns:
        The successful response.

    Raises:
        OpenRouterError: On a non-retryable status or after exhausting all retries.
    """
    total_attempts = config.max_retries + 1
    for attempt in range(total_attempts):
        is_last_attempt = attempt == total_attempts - 1
        try:
            response = client.post(
                url, headers=headers, json=payload, timeout=config.timeout
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if is_last_attempt:
                raise OpenRouterError(f"Request to OpenRouter failed: {exc}") from exc
            time.sleep(_retry_delay_seconds(response=None, attempt=attempt))
            continue

        if response.status_code == httpx.codes.OK:
            return response
        _raise_unless_retryable(
            response=response,
            config=config,
            is_last_attempt=is_last_attempt,
            total_attempts=total_attempts,
        )
        time.sleep(_retry_delay_seconds(response=response, attempt=attempt))

    # Unreachable: the loop either returns or raises on the last attempt.
    raise OpenRouterError("Request to OpenRouter failed.")


def _raise_unless_retryable(
    *,
    response: httpx.Response,
    config: OpenRouterRequestConfig,
    is_last_attempt: bool,
    total_attempts: int,
) -> None:
    """Raise unless the failed response is worth another attempt.

    Raises:
        OpenRouterError: If the status must not be retried, or retries are exhausted.
    """
    if response.status_code not in _RETRYABLE_STATUS_CODES:
        raise OpenRouterError(_non_retryable_message(response=response, config=config))
    if is_last_attempt:
        raise OpenRouterError(
            f"OpenRouter returned HTTP {response.status_code} after "
            f"{total_attempts} attempt(s): {_response_excerpt(response)}"
        )


def _non_retryable_message(
    *, response: httpx.Response, config: OpenRouterRequestConfig
) -> str:
    """Build an actionable message for a status that must not be retried."""
    excerpt = _response_excerpt(response)
    if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
        return (
            f"OpenRouter rejected the credentials (HTTP {response.status_code}). "
            f"Check that OPENROUTER_API_KEY is valid and still active: {excerpt}"
        )
    if response.status_code == httpx.codes.PAYMENT_REQUIRED:
        return (
            "OpenRouter reports insufficient credits (HTTP 402). Top up your account "
            f"at https://openrouter.ai/credits: {excerpt}"
        )
    if response.status_code in (httpx.codes.BAD_REQUEST, httpx.codes.NOT_FOUND):
        return (
            f"OpenRouter rejected the request (HTTP {response.status_code}). Check that "
            f"the model '{config.model}' exists and accepts images: {excerpt}"
        )
    return f"OpenRouter returned HTTP {response.status_code}: {excerpt}"


def _response_excerpt(response: httpx.Response) -> str:
    """Return a short, single-line excerpt of a response body for error messages."""
    return " ".join(response.text.split())[:_ERROR_BODY_CHARS]


def _retry_delay_seconds(*, response: httpx.Response | None, attempt: int) -> float:
    """Return how long to wait before the next attempt.

    Honours the server's own hint when present, otherwise uses exponential backoff with
    full jitter so concurrent workers do not all retry at the same instant.
    """
    if response is not None:
        for header in ("retry-after", "x-ratelimit-reset-after"):
            raw = response.headers.get(header)
            if raw is None:
                continue
            seconds = _parse_retry_after(raw)
            if seconds is not None:
                return min(max(seconds, 0.0), _MAX_RETRY_AFTER_SECONDS)
    backoff = min(2.0**attempt, _MAX_BACKOFF_SECONDS)
    return backoff * (0.5 + random.random() / 2.0)


def _parse_retry_after(raw: str) -> float | None:
    """Parse a Retry-After value given either as seconds or as an HTTP date."""
    value = raw.strip()
    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(timezone.utc)).total_seconds()


def _extract_caption_text(body: object) -> str:
    """Extract the caption from a chat-completions response body.

    Args:
        body: The parsed JSON response.

    Returns:
        The caption text, stripped of surrounding whitespace.

    Raises:
        OpenRouterError: If the body carries an error or holds no usable caption.
    """
    choice = _first_choice(body=body)
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    text = _content_to_text(content).strip()
    if not text:
        raise OpenRouterError("OpenRouter returned an empty caption.")

    if choice.get("finish_reason") == "length":
        logger.info(
            "Caption was truncated by max_tokens; raise it for longer captions."
        )
    return text


def _first_choice(*, body: object) -> Mapping[str, Any]:
    """Return the first choice of a chat-completions body.

    Raises:
        OpenRouterError: If the body carries an error or holds no usable choice.
    """
    if not isinstance(body, Mapping):
        raise OpenRouterError("Unexpected response shape from OpenRouter.")

    # OpenRouter can return HTTP 200 with a top-level error object, for example when a
    # provider blocks the request. Without this check those become empty captions.
    error = body.get("error")
    if error is not None:
        raise OpenRouterError(f"OpenRouter returned an error: {error}")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("OpenRouter returned no choices.")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise OpenRouterError("Unexpected choice shape from OpenRouter.")
    return choice


def _content_to_text(content: object) -> str:
    """Flatten a message content field into plain text.

    Most providers return a string, but some return a list of typed content parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts)
    return ""
