
"""
PragyanAI People Detection & Counting
File: backend/services/detection.py

Responsibilities:
1. Load YOLO11 TFLite model.
2. Detect people in video frames.
3. Filter detections using confidence threshold.
4. Return bounding boxes and confidence scores.
5. Generate annotated frames.
"""

from pathlib import Path
from ultralytics import YOLO


class PeopleDetector:

    def __init__(
        self,
        model_path="models/yolo11n.tflite",
        confidence=0.25,
        image_size=640
    ):

        self.model_path = Path(model_path)

        self.confidence = confidence
        self.image_size = image_size

        if not self.model_path.is_absolute():
            self.model_path = (
                Path(__file__).resolve().parents[1]
                / self.model_path
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        print(f"Loading YOLO model: {self.model_path}")

        self.model = YOLO(str(self.model_path))

        print("YOLO model loaded successfully.")

    # ---------------------------------------------------------
    # Detect people in a single frame
    # ---------------------------------------------------------

    def detect(
        self,
        frame,
        confidence=None
    ):

        threshold = (
            confidence
            if confidence is not None
            else self.confidence
        )

        results = self.model.predict(
            source=frame,
            classes=[0],  # COCO class 0 = Person
            conf=threshold,
            imgsz=self.image_size,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            score = float(
                box.conf[0].cpu().numpy()
            )

            class_id = int(
                box.cls[0].cpu().numpy()
            )

            if class_id != 0:
                continue

            x1, y1, x2, y2 = map(
                int,
                coordinates
            )

            detections.append({

                "class_id": class_id,

                "class_name": "person",

                "confidence": round(score, 4),

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }

            })

        return detections

    # ---------------------------------------------------------
    # Annotate frame with detection boxes
    # ---------------------------------------------------------

    def annotate(
        self,
        frame,
        detections
    ):

        import cv2

        annotated = frame.copy()

        for detection in detections:

            bbox = detection["bbox"]

            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]

            confidence = detection["confidence"]

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"Person {confidence:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return annotated

    # ---------------------------------------------------------
    # Get people count from detections
    # ---------------------------------------------------------

    def count_people(self, detections):

        return len(detections)
