"""
PragyanAI - Tracking and Track Analytics Helpers

IMPORTANT:
Ultralytics YOLO's model.track(..., tracker="bytetrack.yaml", persist=True)
runs the actual ByteTrack algorithm. This module does NOT implement a second
tracker. It maintains application-level statistics from the track IDs returned
by the detection module.

Expected detection dictionary:
{
    "track_id": 1,          # int or None
    "x1": 100,
    "y1": 50,
    "x2": 220,
    "y2": 400,
    "confidence": 0.91,
    "class_id": 0,
    "class_name": "person"
}
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

logger = logging.getLogger("pragyanai.tracking")

DEFAULT_TRACKER_CONFIG = "bytetrack.yaml"


def get_tracker_config(tracker: Optional[str] = None) -> str:
    """Return the configured Ultralytics tracker YAML path/name."""
    selected = (tracker or DEFAULT_TRACKER_CONFIG).strip()
    return selected or DEFAULT_TRACKER_CONFIG


def extract_track_ids(
    detections: Optional[Sequence[Dict[str, Any]]],
    *,
    unique: bool = True,
) -> List[int]:
    """Extract non-null track IDs from detection dictionaries."""
    if not detections:
        return []

    output: List[int] = []
    seen: Set[int] = set()

    for detection in detections:
        raw_id = detection.get("track_id")
        if raw_id is None:
            continue

        try:
            track_id = int(raw_id)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid track ID: %r", raw_id)
            continue

        if unique:
            if track_id in seen:
                continue
            seen.add(track_id)

        output.append(track_id)

    return output


def count_current_people(
    detections: Optional[Sequence[Dict[str, Any]]],
) -> int:
    """Count person detections in the current frame (COCO class ID 0)."""
    if not detections:
        return 0

    count = 0
    for detection in detections:
        class_id = detection.get("class_id")
        class_name = str(detection.get("class_name", "")).lower()

        is_person = (
            class_id is not None and str(class_id) == "0"
        ) or class_name == "person"

        if is_person:
            count += 1

    return count


@dataclass
class TrackRecord:
    """Latest known information for one track ID."""

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen_at: float
    last_seen_at: float
    frames_seen: int = 0
    confidence_sum: float = 0.0
    last_confidence: Optional[float] = None
    last_bbox: Optional[List[int]] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None

    @property
    def average_confidence(self) -> Optional[float]:
        if self.frames_seen <= 0:
            return None
        return self.confidence_sum / self.frames_seen


@dataclass
class TrackingAnalytics:
    """
    Thread-safe, per-video tracking statistics.

    Create ONE instance per video job. Call update() once per processed frame.
    The ByteTrack IDs must come from model.track(...), not be generated here.
    """

    tracker_name: str = DEFAULT_TRACKER_CONFIG
    _tracks: Dict[int, TrackRecord] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _frames_processed: int = field(default=0, init=False)
    _last_frame_number: int = field(default=0, init=False)
    _started_at: float = field(default_factory=time.time, init=False)
    _current_people: int = field(default=0, init=False)
    _current_track_ids: List[int] = field(default_factory=list, init=False)
    _untracked_people_frames: int = field(default=0, init=False)

    def reset(self) -> None:
        """Reset statistics before starting a new video job."""
        with self._lock:
            self._tracks.clear()
            self._frames_processed = 0
            self._last_frame_number = 0
            self._started_at = time.time()
            self._current_people = 0
            self._current_track_ids = []
            self._untracked_people_frames = 0

        logger.info(
            "[TRACKING RESET] tracker=%s",
            self.tracker_name,
        )

    def update(
        self,
        detections: Optional[Sequence[Dict[str, Any]]],
        frame_number: int,
        timestamp_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update track history from one frame's detections.

        Returns a JSON-serializable statistics dictionary.
        """
        detections = detections or []
        now = time.time()
        frame_number = int(frame_number)
        video_timestamp = (
            float(timestamp_seconds)
            if timestamp_seconds is not None
            else None
        )

        with self._lock:
            self._frames_processed += 1
            self._last_frame_number = frame_number

            people = []
            for detection in detections:
                class_id = detection.get("class_id")
                class_name = str(
                    detection.get("class_name", "")
                ).lower()

                if (
                    (class_id is not None and str(class_id) == "0")
                    or class_name == "person"
                ):
                    people.append(detection)

            current_ids: List[int] = []
            seen_this_frame: Set[int] = set()

            for detection in people:
                raw_id = detection.get("track_id")
                if raw_id is None:
                    continue

                try:
                    track_id = int(raw_id)
                except (TypeError, ValueError):
                    continue

                if track_id in seen_this_frame:
                    continue

                seen_this_frame.add(track_id)
                current_ids.append(track_id)

                try:
                    confidence = float(detection.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0

                bbox = [
                    int(detection.get("x1", 0)),
                    int(detection.get("y1", 0)),
                    int(detection.get("x2", 0)),
                    int(detection.get("y2", 0)),
                ]

                record = self._tracks.get(track_id)

                if record is None:
                    record = TrackRecord(
                        track_id=track_id,
                        first_seen_frame=frame_number,
                        last_seen_frame=frame_number,
                        first_seen_at=now,
                        last_seen_at=now,
                    )
                    self._tracks[track_id] = record

                record.last_seen_frame = frame_number
                record.last_seen_at = now
                record.frames_seen += 1
                record.confidence_sum += confidence
                record.last_confidence = confidence
                record.last_bbox = bbox

                class_value = detection.get("class_id")
                record.class_id = (
                    int(class_value)
                    if class_value is not None
                    else 0
                )
                record.class_name = detection.get(
                    "class_name", "person"
                )

            self._current_people = len(people)
            self._current_track_ids = current_ids

            people_without_ids = sum(
                1 for detection in people
                if detection.get("track_id") is None
            )
            if people_without_ids:
                self._untracked_people_frames += people_without_ids

            result = {
                "tracker": self.tracker_name,
                "frame_number": frame_number,
                "timestamp_seconds": video_timestamp,
                "frames_processed": self._frames_processed,
                "current_people": len(people),
                "current_track_ids": list(current_ids),
                "unique_people_so_far": len(self._tracks),
                "tracked_people_in_current_frame": len(current_ids),
                "untracked_people_in_current_frame": people_without_ids,
                "untracked_people_observations": self._untracked_people_frames,
                "total_track_observations": sum(
                    record.frames_seen for record in self._tracks.values()
                ),
                "tracks": self._track_summaries_locked(),
            }

        logger.info(
            "[TRACKING UPDATE] frame=%s | people=%s | "
            "track_ids=%s | unique_tracks=%s | untracked=%s",
            frame_number,
            result["current_people"],
            result["current_track_ids"],
            result["unique_people_so_far"],
            result["untracked_people_in_current_frame"],
        )

        return result

    def _track_summaries_locked(self) -> List[Dict[str, Any]]:
        """Build summaries. Caller must hold self._lock."""
        summaries = []
        for track_id in sorted(self._tracks):
            record = self._tracks[track_id]
            summaries.append({
                "track_id": record.track_id,
                "first_seen_frame": record.first_seen_frame,
                "last_seen_frame": record.last_seen_frame,
                "frames_seen": record.frames_seen,
                "last_confidence": record.last_confidence,
                "average_confidence": record.average_confidence,
                "last_bbox": record.last_bbox,
                "class_id": record.class_id,
                "class_name": record.class_name,
            })
        return summaries

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of current tracking metrics."""
        with self._lock:
            return {
                "tracker": self.tracker_name,
                "frames_processed": self._frames_processed,
                "last_frame_number": self._last_frame_number,
                "current_people": self._current_people,
                "current_track_ids": list(self._current_track_ids),
                "unique_people_so_far": len(self._tracks),
                "untracked_people_observations": self._untracked_people_frames,
                "elapsed_seconds": max(0.0, time.time() - self._started_at),
                "tracks": self._track_summaries_locked(),
            }

    def get_unique_track_ids(self) -> List[int]:
        """Return all unique track IDs observed during this video."""
        with self._lock:
            return sorted(self._tracks.keys())


def update_tracking_statistics(
    state: TrackingAnalytics,
    detections: Optional[Sequence[Dict[str, Any]]],
    frame_number: int,
    timestamp_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for existing video-processing code."""
    if not isinstance(state, TrackingAnalytics):
        raise TypeError("state must be a TrackingAnalytics instance")

    return state.update(
        detections=detections,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
    )


def get_unique_track_ids(
    detections: Optional[Sequence[Dict[str, Any]]] = None,
    state: Optional[TrackingAnalytics] = None,
) -> List[int]:
    """
    Return IDs from a frame, or all IDs from a TrackingAnalytics instance.
    """
    if state is not None:
        return state.get_unique_track_ids()

    return extract_track_ids(detections, unique=True)
