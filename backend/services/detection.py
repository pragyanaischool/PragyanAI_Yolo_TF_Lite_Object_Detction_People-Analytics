"""
PragyanAI YOLO11 TFLite Person Detection Helper

Detection: YOLO11
Tracking: ByteTrack
Runtime: TensorFlow Lite / LiteRT

The model is loaded by VideoProcessor and passed into this module.
This avoids loading a duplicate YOLO model.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("pragyanai.detection")


def _to_numpy(value: Any) -> np.ndarray:
    """Convert Ultralytics tensors to NumPy arrays."""
    if value is None:
        return np.asarray([])

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def detect_people_and_track(
    model: Any,
    frame: np.ndarray,
    confidence: float = 0.35,
    imgsz: int = 640,
    tracker: str = "bytetrack.yaml",
    persist: bool = True,
    classes: Optional[Sequence[int]] = (0,),
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run YOLO detection and ByteTrack on one video frame.

    Args:
        model: Already-loaded Ultralytics YOLO model.
        frame: OpenCV BGR image.
        confidence: Detection confidence threshold.
        imgsz: Model inference image size.
        tracker: ByteTrack configuration.
        persist: Preserve tracker state between frames.
        classes: (0,) for person only; None for all classes.
        verbose: Ultralytics logging option.

    Returns:
        List of detection dictionaries containing:
        track_id, x1, y1, x2, y2, confidence,
        class_id and class_name.
    """

    if model is None:
        raise ValueError("A loaded YOLO model must be supplied.")

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Frame is empty or invalid.")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Expected HxWx3 OpenCV frame; received {frame.shape}."
        )

    if not 0 <= float(confidence) <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    if int(imgsz) <= 0:
        raise ValueError("Image size must be positive.")

    started = time.perf_counter()
    height, width = frame.shape[:2]

    try:
        results = model.track(
            source=frame,
            persist=bool(persist),
            tracker=tracker,
            conf=float(confidence),
            classes=list(classes) if classes is not None else None,
            imgsz=int(imgsz),
            verbose=bool(verbose),
        )

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            logger.info(
                "[DETECTION COMPLETE] detections=0 | elapsed=%.3fs",
                time.perf_counter() - started,
            )
            return []

        xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
        scores = _to_numpy(boxes.conf).reshape(-1)
        class_ids = _to_numpy(boxes.cls).reshape(-1).astype(int)

        raw_ids = getattr(boxes, "id", None)

        track_ids = (
            _to_numpy(raw_ids).reshape(-1).astype(int)
            if raw_ids is not None
            else None
        )

        names = getattr(result, "names", {}) or {}
        detections = []

        count = min(len(xyxy), len(scores), len(class_ids))

        for index in range(count):
            class_id = int(class_ids[index])

            if classes is not None and class_id not in classes:
                continue

            x1, y1, x2, y2 = xyxy[index].tolist()

            track_id = None

            if track_ids is not None and index < len(track_ids):
                track_id = int(track_ids[index])

            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            elif isinstance(names, (list, tuple)) and class_id < len(names):
                class_name = str(names[class_id])
            else:
                class_name = str(class_id)

            detections.append({
                "track_id": track_id,
                "x1": max(0, int(round(x1))),
                "y1": max(0, int(round(y1))),
                "x2": min(width, int(round(x2))),
                "y2": min(height, int(round(y2))),
                "confidence": float(scores[index]),
                "class_id": class_id,
                "class_name": class_name,
            })

        elapsed = time.perf_counter() - started

        tracked_ids = [
            item["track_id"]
            for item in detections
            if item["track_id"] is not None
        ]

        logger.info(
            "[DETECTION COMPLETE] frame=%sx%s | detections=%s | "
            "track_ids=%s | elapsed=%.3fs",
            width,
            height,
            len(detections),
            tracked_ids[:30],
            elapsed,
        )

        return detections

    except Exception:
        logger.exception(
            "[DETECTION FAILED] frame=%sx%s | elapsed=%.3fs",
            width,
            height,
            time.perf_counter() - started,
        )
        raise


def count_people(
    detections: Sequence[Dict[str, Any]]
) -> int:
    """Count COCO person-class detections."""
    return sum(
        1 for item in detections
        if int(item.get("class_id", 0)) == 0
    )


def get_track_ids(
    detections: Sequence[Dict[str, Any]]
) -> List[int]:
    """Return unique non-null tracking IDs in detection order."""
    seen = set()
    output = []

    for item in detections:
        track_id = item.get("track_id")

        if track_id is None:
            continue

        track_id = int(track_id)

        if track_id not in seen:
            seen.add(track_id)
            output.append(track_id)

    return output
