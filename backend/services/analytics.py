"""
PragyanAI People Detection & Counting
=====================================

Frame-level and video-level analytics helpers.

This module is intentionally independent of the YOLO inference runtime. It accepts
the detection dictionaries produced by the project's detection.py module and
maintains:
- Current people count
- Current tracked person IDs
- Unique people seen during the video
- Per-class current counts and total detections
- Frame-by-frame analytics history
- Processing and inference timing statistics
- CSV/JSON-friendly summaries

Expected detection dictionary (extra fields are allowed):
{
    "track_id": 12,             # may be None when tracking is unavailable
    "x1": 100, "y1": 50,
    "x2": 220, "y2": 300,
    "confidence": 0.91,
    "class_id": 0,
    "class_name": "person"
}

Typical integration in video_processor.py:

    from services.analytics import VideoAnalytics

    analytics = VideoAnalytics()
    analytics.reset()

    # Inside the frame loop, after detection:
    frame_stats = analytics.update_frame(
        detections=detections,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        inference_time_ms=inference_ms,
        processing_time_ms=frame_time_ms,
    )

    # Optional: publish this to main.py's job callback:
    progress = {
        "frame_number": frame_stats["frame_number"],
        "total_frames": total_frames,
        "current_people": frame_stats["current_people"],
        "unique_people_so_far": frame_stats["unique_people_so_far"],
        "current_track_ids": frame_stats["current_track_ids"],
        "current_objects": frame_stats["current_objects"],
        "class_counts": frame_stats["class_counts"],
        "inference_time_ms": frame_stats["inference_time_ms"],
        "processing_time_ms": frame_stats["processing_time_ms"],
    }

    # At the end of the video:
    final_report = analytics.finalize(total_frames=total_frames)

NOTE:
ByteTrack itself should be run by the detector, e.g. model.track(...,
tracker="bytetrack.yaml", persist=True). This module only aggregates the
track IDs returned by that detector; it is not a second tracking algorithm.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


Number = Union[int, float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _normalise_class_name(detection: Mapping[str, Any]) -> str:
    name = detection.get("class_name") or detection.get("name")
    if name:
        return str(name).strip().lower()

    class_id = _safe_int(detection.get("class_id"), None)
    if class_id == 0:
        return "person"
    if class_id is not None:
        return f"class_{class_id}"
    return "unknown"


def _is_person(detection: Mapping[str, Any]) -> bool:
    class_id = _safe_int(detection.get("class_id"), None)
    if class_id == 0:
        return True

    name = _normalise_class_name(detection)
    return name in {"person", "people", "human"}


def _track_id(detection: Mapping[str, Any]) -> Optional[Union[int, str]]:
    value = detection.get("track_id", detection.get("id"))
    if value is None:
        return None
    try:
        # Avoid converting non-numeric identifiers.
        return int(value)
    except (TypeError, ValueError, OverflowError):
        value = str(value).strip()
        return value or None


def _bbox(detection: Mapping[str, Any]) -> Optional[List[float]]:
    keys = ("x1", "y1", "x2", "y2")
    if all(key in detection for key in keys):
        return [_safe_float(detection.get(key)) for key in keys]

    # Also accept common xyxy/bbox formats.
    xyxy = detection.get("xyxy")
    if isinstance(xyxy, (list, tuple)) and len(xyxy) >= 4:
        return [_safe_float(v) for v in xyxy[:4]]

    bbox = detection.get("bbox")
    if isinstance(bbox, Mapping):
        return [_safe_float(bbox.get(k)) for k in keys]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [_safe_float(v) for v in bbox[:4]]
    return None


@dataclass
class FrameAnalytics:
    frame_number: int
    timestamp_seconds: float
    current_people: int
    unique_people_so_far: int
    current_track_ids: List[Union[int, str]]
    current_objects: int
    class_counts: Dict[str, int]
    detections: List[Dict[str, Any]]
    inference_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAnalytics:
    """
    Stateful accumulator for one video-processing job.

    Create one instance per job. Do not share one instance across simultaneous
    uploads unless access is externally isolated.
    """

    retain_frame_history: bool = True
    max_history_frames: Optional[int] = None
    tracker_name: str = "bytetrack.yaml"

    started_at: Optional[float] = field(default=None, init=False)
    finished_at: Optional[float] = field(default=None, init=False)
    total_frames_processed: int = field(default=0, init=False)
    total_detections: int = field(default=0, init=False)
    total_person_detections: int = field(default=0, init=False)
    unique_track_ids: set = field(default_factory=set, init=False)
    class_detection_totals: Counter = field(default_factory=Counter, init=False)
    frame_history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    inference_times_ms: List[float] = field(default_factory=list, init=False)
    processing_times_ms: List[float] = field(default_factory=list, init=False)
    peak_people_in_frame: int = field(default=0, init=False)
    peak_objects_in_frame: int = field(default=0, init=False)
    last_frame: Dict[str, Any] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def reset(self) -> None:
        """Reset all metrics before processing a new video."""
        with self._lock:
            self.started_at = time.time()
            self.finished_at = None
            self.total_frames_processed = 0
            self.total_detections = 0
            self.total_person_detections = 0
            self.unique_track_ids = set()
            self.class_detection_totals = Counter()
            self.frame_history = []
            self.inference_times_ms = []
            self.processing_times_ms = []
            self.peak_people_in_frame = 0
            self.peak_objects_in_frame = 0
            self.last_frame = {}

    def update_frame(
        self,
        detections: Optional[Iterable[Mapping[str, Any]]],
        frame_number: Optional[int] = None,
        timestamp_seconds: Optional[Number] = None,
        inference_time_ms: Optional[Number] = None,
        processing_time_ms: Optional[Number] = None,
    ) -> Dict[str, Any]:
        """
        Add analytics for one processed frame and return a JSON-friendly dict.

        Detections without a track_id still contribute to current counts and
        class totals, but cannot contribute to unique tracked-person counts.
        """
        with self._lock:
            if self.started_at is None:
                self.started_at = time.time()

            normalised: List[Dict[str, Any]] = []
            class_counts: Counter = Counter()
            current_track_ids: List[Union[int, str]] = []
            person_track_ids: List[Union[int, str]] = []
            people_count = 0

            for raw in detections or []:
                if not isinstance(raw, Mapping):
                    continue

                item = dict(raw)
                class_name = _normalise_class_name(item)
                item["class_name"] = class_name
                item["confidence"] = _safe_float(item.get("confidence"), 0.0)
                item["track_id"] = _track_id(item)
                box = _bbox(item)
                if box is not None:
                    item["bbox"] = box

                class_counts[class_name] += 1
                if _is_person(item):
                    people_count += 1
                    if item["track_id"] is not None:
                        person_track_ids.append(item["track_id"])
                        self.unique_track_ids.add(item["track_id"])

                if item["track_id"] is not None:
                    current_track_ids.append(item["track_id"])

                normalised.append(item)

            # Keep IDs unique and preserve detector order.
            current_track_ids = list(dict.fromkeys(current_track_ids))
            person_track_ids = list(dict.fromkeys(person_track_ids))

            frame_no = _safe_int(frame_number, self.total_frames_processed + 1)
            timestamp = _safe_float(timestamp_seconds, 0.0)
            inference_ms = max(0.0, _safe_float(inference_time_ms, 0.0))
            processing_ms = max(0.0, _safe_float(processing_time_ms, 0.0))
            elapsed = max(0.0, time.time() - self.started_at)

            self.total_frames_processed += 1
            self.total_detections += len(normalised)
            self.total_person_detections += people_count
            self.class_detection_totals.update(class_counts)
            self.peak_people_in_frame = max(self.peak_people_in_frame, people_count)
            self.peak_objects_in_frame = max(self.peak_objects_in_frame, len(normalised))

            if inference_time_ms is not None:
                self.inference_times_ms.append(inference_ms)
            if processing_time_ms is not None:
                self.processing_times_ms.append(processing_ms)

            record = FrameAnalytics(
                frame_number=frame_no or self.total_frames_processed,
                timestamp_seconds=timestamp,
                current_people=people_count,
                unique_people_so_far=len(self.unique_track_ids),
                current_track_ids=current_track_ids,
                current_objects=len(normalised),
                class_counts=dict(class_counts),
                detections=normalised,
                inference_time_ms=round(inference_ms, 3),
                processing_time_ms=round(processing_ms, 3),
                elapsed_seconds=round(elapsed, 3),
            ).to_dict()

            self.last_frame = record
            if self.retain_frame_history:
                self.frame_history.append(record)
                if self.max_history_frames and self.max_history_frames > 0:
                    excess = len(self.frame_history) - self.max_history_frames
                    if excess > 0:
                        del self.frame_history[:excess]

            return self._public_frame_record(record)

    @staticmethod
    def _public_frame_record(record: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a shallow JSON-friendly copy of a frame record."""
        return {
            "frame_number": record.get("frame_number", 0),
            "timestamp_seconds": record.get("timestamp_seconds", 0.0),
            "current_people": record.get("current_people", 0),
            "unique_people_so_far": record.get("unique_people_so_far", 0),
            "current_track_ids": list(record.get("current_track_ids", [])),
            "current_objects": record.get("current_objects", 0),
            "class_counts": dict(record.get("class_counts", {})),
            "detections": list(record.get("detections", [])),
            "inference_time_ms": record.get("inference_time_ms", 0.0),
            "processing_time_ms": record.get("processing_time_ms", 0.0),
            "elapsed_seconds": record.get("elapsed_seconds", 0.0),
        }

    @staticmethod
    def _timing_summary(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"average_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "median_ms": 0.0}
        return {
            "average_ms": round(statistics.fmean(values), 3),
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
            "median_ms": round(statistics.median(values), 3),
        }

    def get_current_stats(self) -> Dict[str, Any]:
        """Return the latest frame values and accumulated counters."""
        with self._lock:
            latest = self._public_frame_record(self.last_frame) if self.last_frame else {
                "frame_number": 0,
                "timestamp_seconds": 0.0,
                "current_people": 0,
                "unique_people_so_far": 0,
                "current_track_ids": [],
                "current_objects": 0,
                "class_counts": {},
                "detections": [],
                "inference_time_ms": 0.0,
                "processing_time_ms": 0.0,
                "elapsed_seconds": 0.0,
            }
            return {
                **latest,
                "total_frames_processed": self.total_frames_processed,
                "total_detections": self.total_detections,
                "total_person_detections": self.total_person_detections,
                "unique_people": len(self.unique_track_ids),
                "unique_people_so_far": len(self.unique_track_ids),
                "peak_people_in_frame": self.peak_people_in_frame,
                "peak_objects_in_frame": self.peak_objects_in_frame,
                "class_detection_totals": dict(self.class_detection_totals),
                "tracker": self.tracker_name,
            }

    def finalize(self, total_frames: Optional[int] = None) -> Dict[str, Any]:
        """Mark processing complete and return the final analytics report."""
        with self._lock:
            if self.started_at is None:
                self.started_at = time.time()
            self.finished_at = time.time()
            duration = max(0.0, self.finished_at - self.started_at)
            processed = self.total_frames_processed
            expected = _safe_int(total_frames, processed) or processed

            report = {
                "status": "completed",
                "tracker": self.tracker_name,
                "total_frames": expected,
                "frames_processed": processed,
                "total_detections": self.total_detections,
                "total_person_detections": self.total_person_detections,
                "unique_people": len(self.unique_track_ids),
                "unique_people_count": len(self.unique_track_ids),
                "peak_people_in_frame": self.peak_people_in_frame,
                "peak_objects_in_frame": self.peak_objects_in_frame,
                "class_detection_totals": dict(self.class_detection_totals),
                "processing_duration_seconds": round(duration, 3),
                "average_fps": round(processed / duration, 3) if duration > 0 else 0.0,
                "inference_timing": self._timing_summary(self.inference_times_ms),
                "processing_timing": self._timing_summary(self.processing_times_ms),
                "unique_track_ids": sorted(self.unique_track_ids, key=lambda x: str(x)),
                "last_frame": self._public_frame_record(self.last_frame) if self.last_frame else {},
            }
            return report

    def get_frame_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._public_frame_record(frame) for frame in self.frame_history]

    def export_csv(self, output_path: Union[str, Path]) -> str:
        """Export one row per retained frame. Returns the output path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "frame_number", "timestamp_seconds", "current_people",
            "unique_people_so_far", "current_objects", "current_track_ids",
            "class_counts", "inference_time_ms", "processing_time_ms",
            "elapsed_seconds",
        ]
        with self._lock:
            rows = list(self.frame_history)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                output = {key: row.get(key) for key in fields}
                output["current_track_ids"] = json.dumps(output["current_track_ids"])
                output["class_counts"] = json.dumps(output["class_counts"])
                writer.writerow(output)
        return str(path)

    def export_json(
        self,
        output_path: Union[str, Path],
        include_frames: bool = True,
    ) -> str:
        """Export final report and optionally the retained frame history."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.finalize()
        if include_frames:
            payload["frames"] = self.get_frame_history()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return str(path)


# ---------------------------------------------------------------------------
# Convenience functions for code that does not need a VideoAnalytics instance
# ---------------------------------------------------------------------------

def count_people(detections: Optional[Iterable[Mapping[str, Any]]]) -> int:
    """Count person detections in the current frame."""
    return sum(1 for item in (detections or []) if isinstance(item, Mapping) and _is_person(item))


def count_objects(detections: Optional[Iterable[Mapping[str, Any]]]) -> int:
    """Count all valid detections in the current frame."""
    return sum(1 for item in (detections or []) if isinstance(item, Mapping))


def get_track_ids(
    detections: Optional[Iterable[Mapping[str, Any]]],
    people_only: bool = False,
) -> List[Union[int, str]]:
    """Extract unique track IDs, optionally only for person detections."""
    ids: List[Union[int, str]] = []
    for item in detections or []:
        if not isinstance(item, Mapping):
            continue
        if people_only and not _is_person(item):
            continue
        track_id = _track_id(item)
        if track_id is not None and track_id not in ids:
            ids.append(track_id)
    return ids


def summarize_detections(detections: Optional[Iterable[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Return simple per-frame people/object/class counts."""
    items = [item for item in (detections or []) if isinstance(item, Mapping)]
    counts = Counter(_normalise_class_name(item) for item in items)
    people = sum(1 for item in items if _is_person(item))
    return {
        "current_people": people,
        "current_objects": len(items),
        "current_track_ids": get_track_ids(items),
        "person_track_ids": get_track_ids(items, people_only=True),
        "class_counts": dict(counts),
    }


# Backward-friendly aliases for common naming patterns.
Analytics = VideoAnalytics
AnalyticsTracker = VideoAnalytics
