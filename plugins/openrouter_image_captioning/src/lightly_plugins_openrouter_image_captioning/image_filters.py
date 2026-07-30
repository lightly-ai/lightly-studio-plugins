"""Resolve the operator execution context into an image filter."""

from __future__ import annotations

from lightly_studio.plugins.operator_context import AnyFilter
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter


def resolve_image_filter(*, context_filter: AnyFilter | None) -> ImageFilter | None:
    """Narrow the discriminated filter union down to an image filter.

    The GUI hands over either an already specific `ImageFilter` or the more general
    `SampleFilter`, depending on where the operator was triggered from. The filter is
    passed through untouched: which images to caption is decided in the GUI, so a view
    filtered to uncaptioned images is the way to avoid captioning an image twice.

    Args:
        context_filter: The filter supplied by the caller, if any.

    Returns:
        The filter to pass to the image resolver, or None for no filtering.
    """
    if isinstance(context_filter, SampleFilter):
        return ImageFilter(sample_filter=context_filter)
    if isinstance(context_filter, ImageFilter):
        return context_filter
    return None
