"""Plugin for exporting filtered image samples to KITTI object-detection format."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from labelformat.formats import KittiObjectDetectionOutput
from labelformat.model.image import Image
from labelformat.model.object_detection import ImageObjectDetection
from sqlmodel import Session

from lightly_studio.core.image.image_sample import ImageSample
from lightly_studio.export.lightly_studio_label_input import (
    LightlyStudioObjectDetectionInput,
)
from lightly_studio.models.collection import CollectionTable
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import (
    AnyFilter,
    ExecutionContext,
    OperatorScope,
)
from lightly_studio.plugins.operator_registry import operator_registry
from lightly_studio.plugins.parameter import BaseParameter, StringParameter
from lightly_studio.resolvers import image_resolver
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter


class KittiObjectDetectionInput(LightlyStudioObjectDetectionInput):
    """Object-detection input with KITTI-compatible relative label filenames."""

    def __init__(
        self,
        session: Session,
        dataset_id: UUID,
        samples: Iterable[ImageSample],
        images_root: Path | None = None,
    ) -> None:
        """Initialize the input.

        Args:
            session: The database session.
            dataset_id: The dataset ID for label retrieval.
            samples: The samples to export.
            images_root: Common root path used to preserve nested image folders.
        """
        self._images_root = images_root
        super().__init__(session=session, dataset_id=dataset_id, samples=samples)

    def get_images(self) -> list[Image]:
        """Return images with filenames relative to the KITTI output folder."""
        return [
            Image(
                id=image.id,
                filename=_get_kitti_filename(
                    image_filename=image.filename,
                    images_root=self._images_root,
                ),
                width=image.width,
                height=image.height,
            )
            for image in super().get_images()
        ]

    def get_labels(self) -> list[ImageObjectDetection]:
        """Return labels with filenames relative to the KITTI output folder."""
        return [
            ImageObjectDetection(
                image=Image(
                    id=label.image.id,
                    filename=_get_kitti_filename(
                        image_filename=label.image.filename,
                        images_root=self._images_root,
                    ),
                    width=label.image.width,
                    height=label.image.height,
                ),
                objects=label.objects,
            )
            for label in super().get_labels()
        ]


@dataclass
class ExportKittiOperator(BaseOperator):
    """Export the current image scope to KITTI object-detection format."""

    name: str = "Export to KITTI"
    description: str = "Export filtered image samples to KITTI object-detection format."

    @property
    def parameters(self) -> list[BaseParameter]:
        """Return the operator parameters."""
        return [
            StringParameter(
                name="output_folder",
                description="Destination folder for the KITTI .txt files.",
                required=True,
                default="kitti_export",
            )
        ]

    @property
    def supported_scopes(self) -> list[OperatorScope]:
        """Return the scopes where this operator is available."""
        return [OperatorScope.IMAGE]

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        """Export filtered image samples to KITTI object-detection labels."""
        collection = session.get(CollectionTable, context.collection_id)
        if collection is None:
            return OperatorResult(success=False, message="Collection does not exist.")

        result = image_resolver.get_all_by_collection_id(
            session=session,
            collection_id=context.collection_id,
            filters=_get_image_filter(context_filter=context.context_filter),
        )
        samples = [ImageSample(inner=image_table) for image_table in result.samples]

        output_folder = Path(parameters["output_folder"]).absolute()
        output_folder.mkdir(parents=True, exist_ok=True)

        label_input = KittiObjectDetectionInput(
            session=session,
            dataset_id=collection.dataset_id,
            samples=samples,
            images_root=_get_common_image_root(samples=samples),
        )
        KittiObjectDetectionOutput(output_folder=output_folder).save(
            label_input=label_input
        )

        return OperatorResult(
            success=True,
            message=f"Exported {len(samples)} samples to KITTI format: {output_folder}",
        )


def _get_image_filter(*, context_filter: AnyFilter | None) -> ImageFilter | None:
    if isinstance(context_filter, SampleFilter):
        return ImageFilter(sample_filter=context_filter)
    if isinstance(context_filter, ImageFilter):
        return context_filter
    return None


def _get_common_image_root(*, samples: list[ImageSample]) -> Path | None:
    if not samples:
        return None
    image_parent_paths = [
        str(Path(sample.file_path_abs).resolve(strict=False).parent)
        for sample in samples
    ]
    try:
        return Path(os.path.commonpath(image_parent_paths))
    except ValueError:
        return None


def _get_kitti_filename(*, image_filename: str, images_root: Path | None) -> str:
    image_path = Path(image_filename).resolve(strict=False)
    if images_root is None:
        return image_path.name
    try:
        return image_path.relative_to(images_root).as_posix()
    except ValueError:
        return image_path.name
