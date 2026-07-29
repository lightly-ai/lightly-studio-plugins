"""Coercion helpers for the untyped parameter dict that operators receive.

Operator parameters arrive as `dict[str, Any]` without any validation, so every value
has to be coerced and range-checked before use. These helpers know nothing about
captioning and raise `ParameterError` with a message meant for the user.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_BOOL_STRINGS = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "yes": True,
    "no": False,
}


class ParameterError(ValueError):
    """Raised when a supplied parameter value cannot be used."""


def get_str(*, parameters: Mapping[str, Any], name: str, default: str) -> str:
    """Read a string parameter, falling back to the default when absent or blank.

    Args:
        parameters: The raw parameter dict.
        name: Name of the parameter to read.
        default: Value to use when the parameter is missing or blank.

    Returns:
        The stripped value, or the default.

    Raises:
        ParameterError: If the value is not a string.
    """
    value = parameters.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ParameterError(f"Parameter '{name}' must be text.")
    return value.strip() or default


def get_bool(*, parameters: Mapping[str, Any], name: str, default: bool) -> bool:
    """Read a boolean parameter, accepting the string forms the GUI may send.

    Args:
        parameters: The raw parameter dict.
        name: Name of the parameter to read.
        default: Value to use when the parameter is missing.

    Returns:
        The parsed boolean, or the default.

    Raises:
        ParameterError: If the value cannot be read as a boolean.
    """
    value = parameters.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        parsed = _BOOL_STRINGS.get(value.strip().lower())
        if parsed is not None:
            return parsed
    raise ParameterError(f"Parameter '{name}' must be true or false.")


def get_int(
    *,
    parameters: Mapping[str, Any],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read an integer parameter and check it against its allowed range.

    Args:
        parameters: The raw parameter dict.
        name: Name of the parameter to read.
        default: Value to use when the parameter is missing.
        minimum: Smallest accepted value, inclusive.
        maximum: Largest accepted value, inclusive.

    Returns:
        The parsed integer, or the default.

    Raises:
        ParameterError: If the value is not a whole number or is out of range.
    """
    value = parameters.get(name)
    number = default if value is None else _to_int(value=value, name=name)
    if not minimum <= number <= maximum:
        raise ParameterError(
            f"Parameter '{name}' must be between {minimum} and {maximum}."
        )
    return number


def get_float(
    *,
    parameters: Mapping[str, Any],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Read a float parameter and check it against its allowed range.

    Args:
        parameters: The raw parameter dict.
        name: Name of the parameter to read.
        default: Value to use when the parameter is missing.
        minimum: Smallest accepted value, inclusive.
        maximum: Largest accepted value, inclusive.

    Returns:
        The parsed float, or the default.

    Raises:
        ParameterError: If the value is not a number or is out of range.
    """
    value = parameters.get(name)
    number = default if value is None else _to_float(value=value, name=name)
    if not minimum <= number <= maximum:
        raise ParameterError(
            f"Parameter '{name}' must be between {minimum} and {maximum}."
        )
    return number


def get_choice(
    *,
    parameters: Mapping[str, Any],
    name: str,
    default: str,
    allowed: Sequence[str],
) -> str:
    """Read a string parameter restricted to a fixed set of values.

    Unlike `get_str`, a blank value is kept rather than replaced by the default, so an
    empty choice stays selectable.

    Args:
        parameters: The raw parameter dict.
        name: Name of the parameter to read.
        default: Value to use when the parameter is missing.
        allowed: The accepted values. Include the empty string to allow a blank choice.

    Returns:
        The chosen value, or the default.

    Raises:
        ParameterError: If the value is not one of the allowed choices.
    """
    value = parameters.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ParameterError(f"Parameter '{name}' must be text.")
    choice = value.strip().lower()
    if choice not in allowed:
        readable = ", ".join(f"'{option}'" for option in allowed if option)
        raise ParameterError(f"Parameter '{name}' must be one of {readable}, or empty.")
    return choice


def _to_int(*, value: Any, name: str) -> int:
    """Coerce a parameter value to a whole number.

    Raises:
        ParameterError: If the value is not a whole number.
    """
    error = ParameterError(f"Parameter '{name}' must be a whole number.")
    # bool is a subclass of int, so reject it before the int check.
    if isinstance(value, bool):
        raise error
    if isinstance(value, int):
        return value
    number = (
        value if isinstance(value, float) else _parse_float(value=value, error=error)
    )
    if not number.is_integer():
        raise error
    return int(number)


def _to_float(*, value: Any, name: str) -> float:
    """Coerce a parameter value to a number.

    Raises:
        ParameterError: If the value is not a number.
    """
    error = ParameterError(f"Parameter '{name}' must be a number.")
    # bool is a subclass of int, so reject it before the int check.
    if isinstance(value, bool):
        raise error
    if isinstance(value, (int, float)):
        return float(value)
    return _parse_float(value=value, error=error)


def _parse_float(*, value: Any, error: ParameterError) -> float:
    """Parse a numeric string.

    Raises:
        ParameterError: The given `error`, if the value is not a numeric string.
    """
    if not isinstance(value, str):
        raise error
    try:
        return float(value.strip())
    except ValueError as exc:
        raise error from exc
