"""BBox auto propagation plugin for Lightly-Studio."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import cv2
from sqlmodel import Session, select

from lightly_studio.models.annotation.annotation_base import (
    AnnotationBaseTable,
    AnnotationCreate,
    AnnotationType,
)
from lightly_studio.models.annotation.object_track import ObjectTrackCreate
from lightly_studio.models.collection import SampleType
from lightly_studio.models.video import VideoFrameTable, VideoTable
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.plugins.parameter import BaseParameter, FloatParameter
from lightly_studio.resolvers import (
    annotation_resolver,
    collection_resolver,
    object_track_resolver,
    video_frame_resolver,
)
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter
from lightly_studio.resolvers.video_frame_resolver.video_frame_filter import (
    VideoFrameFilter,
)

logger = logging.getLogger(__name__)

_NANOTRACK_MODEL_DIR = Path.home() / ".lightly_studio" / "models" / "nanotrack"
_NANOTRACK_BASE_URL = (
    "https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2"
)
_BACKBONE_FILENAME = "nanotrack_backbone_sim.onnx"
_NECKHEAD_FILENAME = "nanotrack_head_sim.onnx"

COLLECTION_NAME = "bbox_auto_propagation_nanotrack"


def _ensure_nanotrack_models() -> tuple[Path, Path]:
    """Download NanoTracker model files if they don't exist."""
    _NANOTRACK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backbone_path = _NANOTRACK_MODEL_DIR / _BACKBONE_FILENAME
    neckhead_path = _NANOTRACK_MODEL_DIR / _NECKHEAD_FILENAME

    for filename, path in [
        (_BACKBONE_FILENAME, backbone_path),
        (_NECKHEAD_FILENAME, neckhead_path),
    ]:
        if not path.exists():
            url = f"{_NANOTRACK_BASE_URL}/{filename}"
            logger.info("Downloading NanoTracker model: %s", url)
            urllib.request.urlretrieve(url, path)
            logger.info("Saved to %s", path)

    return backbone_path, neckhead_path


def _create_nano_tracker(backbone_path: Path, neckhead_path: Path) -> cv2.TrackerNano:
    """Create a new TrackerNano instance."""
    params = cv2.TrackerNano.Params()
    params.backbone = str(backbone_path)
    params.neckhead = str(neckhead_path)
    return cv2.TrackerNano.create(parameters=params)


@dataclass
class AutoPropagateOperator(BaseOperator):
    """Propagate bounding box annotations across video frames using NanoTracker.

    Can be triggered from a video frame (propagates all annotations on that frame)
    or from a specific annotation (propagates only that annotation).
    """

    name: str = "Auto-Propagate Annotations"
    description: str = (
        "Track bounding box annotations from the current frame or annotation to other "
        "frames in the video using NanoTracker."
    )

    @property
    def parameters(self) -> list[BaseParameter]:
        return [
            FloatParameter(
                name="backward_seconds",
                required=False,
                default=0.0,
                description="Seconds to propagate backward from the source frame (0 = disabled)",
            ),
            FloatParameter(
                name="forward_seconds",
                required=False,
                default=2.0,
                description="Seconds to propagate forward from the source frame (0 = disabled)",
            ),
        ]

    @property
    def supported_scopes(self) -> list[OperatorScope]:
        return [OperatorScope.VIDEO_FRAME, OperatorScope.ANNOTATION]

    def _resolve_from_frame(
        self,
        session: Session,
        context: ExecutionContext,
    ) -> tuple[VideoFrameTable, list[AnnotationBaseTable]] | OperatorResult:
        """Resolve source frame and all its OD annotations when triggered from a video frame."""
        context_filter = context.context_filter
        if isinstance(context_filter, SampleFilter):
            result = video_frame_resolver.get_all_by_collection_id(
                session=session,
                collection_id=context.collection_id,
                video_frame_filter=VideoFrameFilter(sample_filter=context_filter),
            )
        else:
            return OperatorResult(
                success=False, message="Plugin only compatible with SampleFilter."
            )

        if not result.samples:
            return OperatorResult(success=False, message="No matching frame found.")
        source_frame = result.samples[0]
        all_frame_annotations = annotation_resolver.get_all_by_parent_sample_ids(
            session=session, parent_sample_ids=[source_frame.sample_id]
        )
        annotations = [
            annotation
            for annotation in all_frame_annotations
            if annotation.annotation_type == AnnotationType.OBJECT_DETECTION
        ]
        return source_frame, annotations

    def _resolve_from_annotation(
        self,
        session: Session,
        context: ExecutionContext,
    ) -> tuple[VideoFrameTable, list[AnnotationBaseTable]] | OperatorResult:
        """Resolve source frame and the triggered OD annotation(s) from an annotation context."""
        context_filter = context.context_filter
        if isinstance(context_filter, SampleFilter) and context_filter.sample_ids:
            # UI passes a SampleFilter with the specific annotation sample IDs that were selected.
            raw_annotations = list(
                annotation_resolver.get_by_ids(
                    session=session, annotation_ids=context_filter.sample_ids
                )
            )
        else:
            return OperatorResult(
                success=False, message="Plugin only compatible with SampleFilter."
            )

        annotations = [
            annotation
            for annotation in raw_annotations
            if annotation.annotation_type == AnnotationType.OBJECT_DETECTION
        ]
        if not annotations:
            return OperatorResult(
                success=False, message="No object detection annotations found."
            )
        if len(annotations) > 1:
            return OperatorResult(
                success=False, message="Plugin does not support multiple annotations."
            )

        source_frame = video_frame_resolver.get_by_id(
            session=session, sample_id=annotations[0].parent_sample_id
        )
        if source_frame is None:
            return OperatorResult(success=False, message="Could not find parent frame.")
        return source_frame, annotations

    def _resolve_source(
        self,
        session: Session,
        context: ExecutionContext,
    ) -> tuple[VideoFrameTable, list[AnnotationBaseTable]] | OperatorResult:
        """Resolve the source frame and annotations from the execution context.

        Uses the collection's sample_type to distinguish frame vs annotation trigger.
        Returns a (frame, annotations) tuple, or an OperatorResult on failure.
        """
        triggered_collection = collection_resolver.get_by_id(
            session=session, collection_id=context.collection_id
        )
        if triggered_collection is None:
            return OperatorResult(success=False, message="Collection not found.")
        if triggered_collection.sample_type == SampleType.VIDEO_FRAME:
            return self._resolve_from_frame(session=session, context=context)
        if triggered_collection.sample_type == SampleType.ANNOTATION:
            return self._resolve_from_annotation(session=session, context=context)
        return OperatorResult(
            success=False,
            message=f"Unsupported collection type: {triggered_collection.sample_type}.",
        )

    def _create_tracks_and_bounding_boxes(
        self,
        session: Session,
        dataset_id: UUID,
        source_annotations: list[AnnotationBaseTable],
    ) -> list[tuple[UUID, int, int, int, int, UUID]]:
        """Create one ObjectTrack per annotation, link the source annotation, and return boxes."""
        existing_tracks = object_track_resolver.get_all_by_dataset_id(
            session=session, dataset_id=dataset_id
        )
        next_track_number = (
            max((t.object_track_number for t in existing_tracks), default=0) + 1
        )
        track_ids = object_track_resolver.create_many(
            session=session,
            tracks=[
                ObjectTrackCreate(
                    object_track_number=next_track_number + i,
                    dataset_id=dataset_id,
                )
                for i in range(len(source_annotations))
            ],
        )
        # Read bbox data before any mutations (add_annotation_to_object_track does a
        # delete-and-reinsert which invalidates the Python annotation objects).
        bbox_data: list[tuple[UUID, int, int, int, int] | None] = []
        for annotation in source_annotations:
            det = annotation.object_detection_details
            if det is not None:
                bbox_data.append(
                    (
                        annotation.annotation_label_id,
                        det.x,
                        det.y,
                        det.width,
                        det.height,
                    )
                )
            else:
                bbox_data.append(None)

        bounding_boxes: list[tuple[UUID, int, int, int, int, UUID]] = []
        for annotation, track_id, bbox in zip(source_annotations, track_ids, bbox_data):
            object_track_resolver.add_annotation_to_object_track(
                session=session,
                annotation_id=annotation.sample_id,
                object_track_id=track_id,
            )
            if bbox is not None:
                label_id, x, y, w, h = bbox
                bounding_boxes.append((label_id, x, y, w, h, track_id))
        return bounding_boxes

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        """Propagate bounding box annotations using NanoTracker."""
        backward_seconds: float = parameters.get("backward_seconds", 0.0) or 0.0
        forward_seconds: float = parameters.get("forward_seconds", 2.0) or 0.0
        if backward_seconds <= 0 and forward_seconds <= 0:
            return OperatorResult(
                success=False,
                message="Set backward_seconds or forward_seconds to a value above 0.",
            )

        resolved = self._resolve_source(session=session, context=context)
        if isinstance(resolved, OperatorResult):
            return resolved
        source_frame, source_annotations = resolved

        video = session.exec(
            select(VideoTable).where(
                VideoTable.sample_id == source_frame.parent_sample_id
            )
        ).one_or_none()
        if video is None:
            return OperatorResult(
                success=False, message="Could not find video for frame."
            )

        if source_frame.sample is None:
            return OperatorResult(
                success=False, message="Could not resolve frame sample."
            )

        root_collection = collection_resolver.get_root_collection(
            session=session, collection_id=source_frame.sample.collection_id
        )
        dataset_id = root_collection.dataset_id
        frames_collection_id = collection_resolver.get_or_create_child_collection(
            session=session,
            collection_id=source_frame.sample.collection_id,
            sample_type=SampleType.VIDEO_FRAME,
        )
        bounding_boxes = self._create_tracks_and_bounding_boxes(
            session=session,
            dataset_id=dataset_id,
            source_annotations=source_annotations,
        )
        if not bounding_boxes:
            return OperatorResult(
                success=False, message="No valid bounding boxes found."
            )

        backbone_path, neck_head_path = _ensure_nanotrack_models()

        all_frames = video_frame_resolver.get_all_by_video_ids(
            session=session, video_ids=[video.sample_id]
        )
        frame_by_number: dict[int, VideoFrameTable] = {
            f.frame_number: f for f in all_frames
        }

        max_backward_frames = max(0, int(backward_seconds * video.fps))
        max_forward_frames = max(0, int(forward_seconds * video.fps))

        cap = cv2.VideoCapture(video.file_path_abs)
        if not cap.isOpened():
            return OperatorResult(
                success=False, message=f"Could not open video: {video.file_path_abs}"
            )
        try:
            new_annotation_creates = _track_all_directions(
                cap=cap,
                frame_number=source_frame.frame_number,
                bounding_boxes=bounding_boxes,
                frame_by_number=frame_by_number,
                backbone_path=backbone_path,
                neck_head_path=neck_head_path,
                max_forward_frames=max_forward_frames,
                max_backward_frames=max_backward_frames,
            )
        finally:
            cap.release()

        if not new_annotation_creates:
            return OperatorResult(
                success=True,
                message="Tracking completed but no new annotations were created.",
            )

        annotation_resolver.create_many(
            session=session,
            parent_collection_id=frames_collection_id,
            annotations=new_annotation_creates,
            collection_name=COLLECTION_NAME,
        )

        n_frames = len({a.parent_sample_id for a in new_annotation_creates})
        return OperatorResult(
            success=True,
            message=(
                f"Propagated {len(bounding_boxes)} annotation(s) to {n_frames} frame(s) "
                f"({len(new_annotation_creates)} total new annotations)."
            ),
        )


def _track_all_directions(
    *,
    cap: cv2.VideoCapture,
    frame_number: int,
    bounding_boxes: list[tuple[UUID, int, int, int, int, UUID]],
    frame_by_number: dict[int, VideoFrameTable],
    backbone_path: Path,
    neck_head_path: Path,
    max_forward_frames: int,
    max_backward_frames: int,
) -> list[AnnotationCreate]:
    """Track bounding boxes forward and backward from the source frame."""
    all_frame_numbers = sorted(frame_by_number.keys())

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, source_img = cap.read()
    if not ret:
        return []

    new_annotations = []

    forward_frames = [fn for fn in all_frame_numbers if fn > frame_number][
        :max_forward_frames
    ]
    if forward_frames:
        new_annotations.extend(
            _track_direction(
                cap=cap,
                source_img=source_img,
                bounding_boxes=bounding_boxes,
                target_frame_numbers=forward_frames,
                frame_by_number=frame_by_number,
                backbone_path=backbone_path,
                neck_head_path=neck_head_path,
            )
        )

    backward_frames = [fn for fn in reversed(all_frame_numbers) if fn < frame_number][
        :max_backward_frames
    ]
    if backward_frames:
        new_annotations.extend(
            _track_direction(
                cap=cap,
                source_img=source_img,
                bounding_boxes=bounding_boxes,
                target_frame_numbers=backward_frames,
                frame_by_number=frame_by_number,
                backbone_path=backbone_path,
                neck_head_path=neck_head_path,
            )
        )

    return new_annotations


def _track_direction(
    *,
    cap: cv2.VideoCapture,
    source_img: Any,
    bounding_boxes: list[tuple[UUID, int, int, int, int, UUID]],
    target_frame_numbers: list[int],
    frame_by_number: dict[int, VideoFrameTable],
    backbone_path: Path,
    neck_head_path: Path,
) -> list[AnnotationCreate]:
    """Track bounding boxes in one direction (forward or backward)."""
    trackers: list[tuple[cv2.TrackerNano, UUID, UUID]] = []
    for annotation_label_id, x, y, w, h, track_id in bounding_boxes:
        tracker = _create_nano_tracker(backbone_path, neck_head_path)
        tracker.init(source_img, (x, y, w, h))
        trackers.append((tracker, annotation_label_id, track_id))

    new_annotations = []

    for target_fn in target_frame_numbers:
        if not trackers:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_fn)
        ret, frame_img = cap.read()
        if not ret:
            break

        target_frame = frame_by_number[target_fn]
        active_trackers = []

        for tracker, annotation_label_id, track_id in trackers:
            success, bbox = tracker.update(frame_img)
            if not success:
                continue

            active_trackers.append((tracker, annotation_label_id, track_id))

            bx, by, bw, bh = bbox
            new_annotations.append(
                AnnotationCreate(
                    annotation_label_id=annotation_label_id,
                    annotation_type=AnnotationType.OBJECT_DETECTION,
                    parent_sample_id=target_frame.sample_id,
                    object_track_id=track_id,
                    x=max(0, int(bx)),
                    y=max(0, int(by)),
                    width=max(1, int(bw)),
                    height=max(1, int(bh)),
                )
            )

        trackers = active_trackers

    return new_annotations
