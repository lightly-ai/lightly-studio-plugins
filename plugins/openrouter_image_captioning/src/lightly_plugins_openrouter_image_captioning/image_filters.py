"""Resolve the operator execution context into an image filter."""

from __future__ import annotations

from dataclasses import dataclass

from lightly_studio.plugins.operator_context import AnyFilter
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter

_CONFLICT_MESSAGE = (
    "The current view is filtered to images that already have captions, but "
    "'skip_captioned' is enabled, so there is nothing to do. Turn 'skip_captioned' off "
    "to add another caption to these images."
)


@dataclass(frozen=True)
class FilterResolution:
    """The filter to query with, or a reason the request cannot be honoured.

    Attributes:
        image_filter: The filter to pass to the image resolver, or None for no filtering.
        conflict_message: Set when the request contradicts itself and should not run.
    """

    image_filter: ImageFilter | None
    conflict_message: str | None


def resolve_image_filter(
    *, context_filter: AnyFilter | None, skip_captioned: bool
) -> FilterResolution:
    """Widen the context filter to an image filter, optionally excluding captioned images.

    Args:
        context_filter: The filter supplied by the caller, if any.
        skip_captioned: Whether images that already have captions should be excluded.

    Returns:
        The resolved filter, or a conflict message when the request cannot be honoured.
    """
    image_filter = _widen(context_filter=context_filter)
    if not skip_captioned:
        return FilterResolution(image_filter=image_filter, conflict_message=None)

    sample_filter = image_filter.sample_filter if image_filter is not None else None
    if sample_filter is not None and sample_filter.has_captions is True:
        return FilterResolution(
            image_filter=image_filter, conflict_message=_CONFLICT_MESSAGE
        )
    return FilterResolution(
        image_filter=_exclude_captioned(
            image_filter=image_filter, sample_filter=sample_filter
        ),
        conflict_message=None,
    )


def _widen(*, context_filter: AnyFilter | None) -> ImageFilter | None:
    """Narrow the discriminated filter union down to an image filter.

    The GUI hands over either an already specific `ImageFilter` or the more general
    `SampleFilter`, depending on where the operator was triggered from.
    """
    if isinstance(context_filter, SampleFilter):
        return ImageFilter(sample_filter=context_filter)
    if isinstance(context_filter, ImageFilter):
        return context_filter
    return None


def _exclude_captioned(
    *, image_filter: ImageFilter | None, sample_filter: SampleFilter | None
) -> ImageFilter:
    """Add a "has no captions" condition without discarding the caller's own filters.

    The filters are copied rather than mutated so that every other condition the user
    set in the current view, such as tags or metadata, is preserved untouched.
    """
    updated_sample_filter = (
        SampleFilter(has_captions=False)
        if sample_filter is None
        else sample_filter.model_copy(update={"has_captions": False})
    )
    if image_filter is None:
        return ImageFilter(sample_filter=updated_sample_filter)
    return image_filter.model_copy(update={"sample_filter": updated_sample_filter})
