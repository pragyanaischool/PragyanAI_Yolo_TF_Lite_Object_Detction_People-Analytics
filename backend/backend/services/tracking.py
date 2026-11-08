
"""
PragyanAI People Detection & Counting
File: backend/services/tracking.py

Responsibilities:
1. Track people across video frames.
2. Assign unique tracking IDs.
3. Maintain tracked person records.
4. Count unique tracked people.
5. Return tracking information for analytics.
"""

from ultralytics import YOLO


class PeopleTracker:

    def __init__(
        self,
        model,
        confidence=0.25,
        image_size=640,
        tracker_config="bytetrack.yaml"
    ):

        self.model = model

        self.confidence = confidence
        self.image_size = image_size

        self.tracker_config = tracker_config

        self.unique_person_ids = set()

        self.person_records = {}

        self.frame_count = 0

    # ---------------------------------------------------------
    # Reset tracking information for a new video
    # ---------------------------------------------------------

    def reset(self):

        self.unique_person_ids.clear()

        self.person_records.clear()

        self.frame_count = 0

        predictor = getattr(
            self.model,
            "predictor",
            None
        )

        if predictor is not None:

            trackers = getattr(
                predictor,
                "trackers",
                []
            )

            for tracker in trackers:

                try:
                    tracker.reset()

                except Exception:
                    pass

    # ---------------------------------------------------------
    # Track people in a video frame
    # ---------------------------------------------------------

    def track(
        self,
        frame,
        confidence=None
    ):

        threshold = (
            confidence
            if confidence is not None
            else self.confidence
        )

        results = self.model.track(

            source=frame,

            persist=True,

            tracker=self.tracker_config,

            classes=[0],

            conf=threshold,

            imgsz=self.image_size,

            verbose=False

        )

        result = results[0]

        self.frame_count += 1

        tracked_people = []

        if result.boxes is None:
            return tracked_people

        boxes = result.boxes

        if boxes.id is None:
            return tracked_people

        coordinates = (
            boxes.xyxy.cpu().numpy()
        )

        track_ids = (
            boxes.id.cpu().numpy().astype(int)
        )

        confidences = (
            boxes.conf.cpu().numpy()
        )

        for bbox, track_id, score in zip(
            coordinates,
            track_ids,
            confidences
        ):

            person_id = int(track_id)

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            self.unique_person_ids.add(
                person_id
            )

            # Update individual tracking record.
            if person_id not in self.person_records:

                self.person_records[person_id] = {

                    "person_id": person_id,

                    "first_seen_frame": self.frame_count,

                    "last_seen_frame": self.frame_count,

                    "frames_tracked": 1,

                    "maximum_confidence": float(score)

                }

            else:

                record = self.person_records[person_id]

                record["last_seen_frame"] = (
                    self.frame_count
                )

                record["frames_tracked"] += 1

                record["maximum_confidence"] = max(
                    record["maximum_confidence"],
                    float(score)
                )

            tracked_people.append({

                "person_id": person_id,

                "class_name": "person",

                "confidence": round(
                    float(score),
                    4
                ),

                "bbox": {

                    "x1": x1,

                    "y1": y1,

                    "x2": x2,

                    "y2": y2

                }

            })

        return tracked_people

    # ---------------------------------------------------------
    # Get unique tracked people count
    # ---------------------------------------------------------

    def get_unique_count(self):

        return len(self.unique_person_ids)

    # ---------------------------------------------------------
    # Get current frame people count
    # ---------------------------------------------------------

    def get_current_count(
        self,
        tracked_people
    ):

        return len(tracked_people)

    # ---------------------------------------------------------
    # Get tracking summary
    # ---------------------------------------------------------

    def get_tracking_summary(self):

        return {

            "unique_people_tracked": (
                self.get_unique_count()
            ),

            "total_tracking_frames": (
                self.frame_count
            ),

            "tracked_person_details": list(
                self.person_records.values()
            )

        }
