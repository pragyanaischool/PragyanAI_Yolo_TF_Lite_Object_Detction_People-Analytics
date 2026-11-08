
"""
PragyanAI - People Detection & Counting
Video Processing Engine

Detection     : YOLO11 TFLite
Tracking       : ByteTrack
Video handling : OpenCV
Output         : H.264 MP4 (FFmpeg when available)
Analytics      : CSV + JSON-compatible dictionary

Expected usage:

processor = VideoProcessor("models/yolo11n.tflite")

analytics = processor.process(
    input_video,
    output_video,
    confidence,
    frame_interval,
    csv_path
)
"""

import os
import csv
import time
import logging
import threading
import subprocess

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO


logger = logging.getLogger("pragyanai.video_processor")


# ============================================================
# 1. VIDEO PROCESSING EXCEPTION
# ============================================================

class VideoProcessingError(Exception):
    """Raised when video processing fails."""


# ============================================================
# 2. VIDEO PROCESSOR CLASS
# ============================================================

class VideoProcessor:

    def __init__(
        self,
        model_path,
        imgsz=640,
        tracker="bytetrack.yaml",
        output_fps_fallback=25.0,
    ):
        """
        Initialize YOLO11 TFLite detection engine.

        Args:
            model_path: Path to YOLO11 TFLite model.
            imgsz: Inference image size.
            tracker: Ultralytics tracker configuration.
            output_fps_fallback: FPS fallback for invalid metadata.
        """

        self.model_path = Path(model_path)

        self.imgsz = int(imgsz)
        self.tracker = tracker

        self.output_fps_fallback = float(
            output_fps_fallback
        )

        self.lock = threading.RLock()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        logger.info(
            "Loading YOLO model: %s",
            self.model_path,
        )

        try:

            self.model = YOLO(
                str(self.model_path),
                task="detect",
            )

        except Exception as exc:

            raise VideoProcessingError(
                f"Unable to load YOLO model: {exc}"
            ) from exc

        self.model_loaded = True

        logger.info(
            "YOLO model initialized successfully"
        )

    # ========================================================
    # 3. RESET TRACKER
    # ========================================================

    def reset_tracker(self):
        """
        Reset ByteTrack state between videos.

        Ultralytics initializes its predictor lazily.
        Reset existing trackers when available.
        """

        try:

            predictor = getattr(
                self.model,
                "predictor",
                None,
            )

            if predictor is None:
                return

            trackers = getattr(
                predictor,
                "trackers",
                None,
            )

            if trackers:

                for tracker in trackers:

                    if hasattr(tracker, "reset"):
                        tracker.reset()

            logger.info("Tracker state reset")

        except Exception as exc:

            logger.warning(
                "Tracker reset warning: %s",
                exc,
            )

    # ========================================================
    # 4. VIDEO INFORMATION
    # ========================================================

    @staticmethod
    def get_video_info(video_path):
        """
        Read video metadata.

        Returns:
            Dictionary containing width, height,
            FPS, frame count and duration.
        """

        video_path = str(video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            cap.release()

            raise VideoProcessingError(
                f"Cannot open video: {video_path}"
            )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        cap.release()

        if width <= 0 or height <= 0:

            raise VideoProcessingError(
                "Invalid video dimensions."
            )

        if fps <= 0:
            fps = 25.0

        if frame_count <= 0:

            raise VideoProcessingError(
                "Video contains no readable frames."
            )

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": frame_count / fps,
        }

    # ========================================================
    # 5. DETECTION AND TRACKING
    # ========================================================

    def detect_and_track(
        self,
        frame,
        confidence=0.35,
    ):
        """
        Run YOLO11 inference and ByteTrack.

        Returns:
            List of dictionaries:
            [
                {
                    "track_id": 1,
                    "x1": 10,
                    "y1": 20,
                    "x2": 100,
                    "y2": 200,
                    "confidence": 0.91,
                    "class_id": 0
                }
            ]
        """

        try:

            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                conf=float(confidence),
                classes=[0],
                imgsz=self.imgsz,
                verbose=False,
            )

        except Exception as exc:

            raise VideoProcessingError(
                "YOLO11 inference or ByteTrack failed. "
                "Check whether your installed Ultralytics "
                f"version supports TFLite tracking. Details: {exc}"
            ) from exc

        detections = []

        if not results:
            return detections

        result = results[0]

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()

        confidences = boxes.conf.cpu().numpy()

        classes = boxes.cls.cpu().numpy().astype(int)

        track_ids = None

        if boxes.id is not None:

            track_ids = (
                boxes.id.cpu().numpy().astype(int)
            )

        for index, coordinates in enumerate(xyxy):

            class_id = int(classes[index])

            # Only person class
            if class_id != 0:
                continue

            x1, y1, x2, y2 = coordinates.tolist()

            track_id = None

            if track_ids is not None:
                track_id = int(track_ids[index])

            detections.append({

                "track_id": track_id,

                "x1": int(x1),
                "y1": int(y1),

                "x2": int(x2),
                "y2": int(y2),

                "confidence": float(
                    confidences[index]
                ),

                "class_id": class_id,

            })

        return detections

    # ========================================================
    # 6. DRAW DETECTION BOXES
    # ========================================================

    @staticmethod
    def annotate_frame(
        frame,
        detections,
        frame_number=None,
        people_count=None,
    ):
        """
        Draw bounding boxes and tracking IDs.
        """

        annotated = frame.copy()

        for detection in detections:

            x1 = detection["x1"]
            y1 = detection["y1"]

            x2 = detection["x2"]
            y2 = detection["y2"]

            confidence = detection["confidence"]

            track_id = detection["track_id"]

            # Bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Label
            if track_id is not None:

                label = (
                    f"Person ID: {track_id} "
                    f"({confidence:.2f})"
                )

            else:

                label = (
                    f"Person ({confidence:.2f})"
                )

            label_y = max(20, y1 - 10)

            cv2.putText(
                annotated,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if people_count is None:
            people_count = len(detections)

        # People count overlay
        cv2.rectangle(
            annotated,
            (10, 10),
            (290, 65),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            f"People Count: {people_count}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if frame_number is not None:

            cv2.putText(
                annotated,
                f"Frame: {frame_number}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    # ========================================================
    # 7. CREATE VIDEO WRITER
    # ========================================================

    @staticmethod
    def create_video_writer(
        output_path,
        fps,
        width,
        height,
    ):
        """
        Create OpenCV MP4 writer.

        Uses mp4v as the intermediate codec.
        FFmpeg conversion to H.264 is attempted later.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():

            writer.release()

            raise VideoProcessingError(
                "Unable to initialize MP4 video writer."
            )

        return writer

    # ========================================================
    # 8. CONVERT OUTPUT TO H264
    # ========================================================

    @staticmethod
    def convert_to_h264(
        input_path,
        output_path,
        original_video=None,
    ):
        """
        Convert intermediate MP4 into H.264.

        If FFmpeg is unavailable or conversion fails,
        retain the intermediate MP4 output.

        Original audio is included when possible.
        """

        input_path = Path(input_path)
        output_path = Path(output_path)

        try:

            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(input_path),
            ]

            if original_video is not None:

                command.extend([
                    "-i",
                    str(original_video),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                ])

            command.extend([
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
            ])

            if original_video is not None:

                command.extend([
                    "-c:a",
                    "aac",
                    "-shortest",
                ])

            else:

                command.append("-an")

            command.extend([
                "-movflags",
                "+faststart",
                str(output_path),
            ])

            logger.info(
                "Converting output video to H.264"
            )

            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
            )

            if (
                output_path.exists()
                and output_path.stat().st_size > 0
            ):

                input_path.unlink(missing_ok=True)

                logger.info(
                    "H.264 conversion successful"
                )

                return True

        except Exception as exc:

            logger.warning(
                "H.264 conversion unavailable: %s",
                exc,
            )

        # Fallback: retain intermediate MP4
        if input_path.exists():

            if output_path.exists():
                output_path.unlink()

            os.replace(
                input_path,
                output_path,
            )

            logger.warning(
                "Using MP4 intermediate output instead of H.264"
            )

            return False

        raise VideoProcessingError(
            "Unable to generate final output video."
        )

    # ========================================================
    # 9. CSV ANALYTICS
    # ========================================================

    @staticmethod
    def write_csv(
        csv_path,
        records,
    ):
        """
        Write frame-by-frame analytics.

        Columns:
            frame_number
            timestamp_seconds
            inference_performed
            people_count
            track_ids
            average_confidence
        """

        if not csv_path:
            return

        csv_path = Path(csv_path)

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "frame_number",
            "timestamp_seconds",
            "inference_performed",
            "people_count",
            "track_ids",
            "average_confidence",
        ]

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(records)

        logger.info(
            "CSV analytics saved: %s",
            csv_path,
        )

    # ========================================================
    # 10. PROCESS VIDEO
    # ========================================================

    def process(
        self,
        input_video,
        output_video,
        confidence=0.35,
        frame_interval=1,
        csv_path=None,
    ):
        """
        Complete video processing pipeline.

        Args:
            input_video: Uploaded source video.
            output_video: Processed MP4 output.
            confidence: YOLO confidence threshold.
            frame_interval: Run inference every N frames.
            csv_path: CSV analytics destination.

        Returns:
            Analytics dictionary.
        """

        input_video = Path(input_video)
        output_video = Path(output_video)

        if csv_path:
            csv_path = Path(csv_path)

        if not input_video.is_file():

            raise VideoProcessingError(
                f"Input video not found: {input_video}"
            )

        if not 0.0 <= float(confidence) <= 1.0:

            raise ValueError(
                "Confidence must be between 0 and 1."
            )

        if int(frame_interval) < 1:

            raise ValueError(
                "Frame interval must be at least 1."
            )

        # Lock model/tracker during processing.
        # Avoid concurrent calls corrupting tracker state.
        with self.lock:

            return self._process_video(
                input_video,
                output_video,
                float(confidence),
                int(frame_interval),
                csv_path,
            )

    # ========================================================
    # 11. INTERNAL VIDEO PROCESSING
    # ========================================================

    def _process_video(
        self,
        input_video,
        output_video,
        confidence,
        frame_interval,
        csv_path,
    ):

        start_time = time.perf_counter()

        capture = None
        writer = None

        temp_output = output_video.with_name(
            output_video.stem + "_working.mp4"
        )

        records = []

        unique_people = set()

        detection_count = 0
        processed_frames = 0
        inference_frames = 0

        people_per_frame = []

        confidence_values = []

        last_detections = []

        try:

            # ------------------------------------------------
            # A. READ VIDEO
            # ------------------------------------------------

            video_info = self.get_video_info(
                input_video
            )

            width = video_info["width"]
            height = video_info["height"]

            fps = video_info["fps"]

            total_frames = video_info["frame_count"]

            logger.info(
                "Input video: %s",
                video_info,
            )

            capture = cv2.VideoCapture(
                str(input_video)
            )

            if not capture.isOpened():

                raise VideoProcessingError(
                    "OpenCV cannot open input video."
                )

            # ------------------------------------------------
            # B. CREATE OUTPUT WRITER
            # ------------------------------------------------

            output_video.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temp_output.unlink(
                missing_ok=True
            )

            writer = self.create_video_writer(
                temp_output,
                fps,
                width,
                height,
            )

            # ------------------------------------------------
            # C. RESET TRACKER
            # ------------------------------------------------

            self.reset_tracker()

            logger.info(
                "Starting frame-by-frame processing"
            )

            # ------------------------------------------------
            # D. PROCESS FRAMES
            # ------------------------------------------------

            frame_number = 0

            while True:

                success, frame = capture.read()

                if not success:
                    break

                frame_number += 1

                inference_performed = (
                    (frame_number - 1) % frame_interval == 0
                )

                # --------------------------------------------
                # RUN YOLO11 + BYTETRACK
                # --------------------------------------------

                if inference_performed:

                    detections = self.detect_and_track(
                        frame,
                        confidence,
                    )

                    last_detections = detections

                    inference_frames += 1

                    detection_count += len(detections)

                    for detection in detections:

                        track_id = detection["track_id"]

                        if track_id is not None:

                            unique_people.add(
                                track_id
                            )

                        confidence_values.append(
                            detection["confidence"]
                        )

                else:

                    # Carry last sampled detections forward
                    # for visualization between inference frames.
                    detections = last_detections

                # --------------------------------------------
                # CURRENT PEOPLE COUNT
                # --------------------------------------------

                people_count = len(detections)

                people_per_frame.append(
                    people_count
                )

                # --------------------------------------------
                # DRAW BOUNDING BOXES
                # --------------------------------------------

                annotated_frame = self.annotate_frame(
                    frame,
                    detections,
                    frame_number,
                    people_count,
                )

                # --------------------------------------------
                # WRITE OUTPUT FRAME
                # --------------------------------------------

                writer.write(
                    annotated_frame
                )

                processed_frames += 1

                # --------------------------------------------
                # CSV RECORD
                # --------------------------------------------

                track_ids = [
                    str(item["track_id"])
                    for item in detections
                    if item["track_id"] is not None
                ]

                frame_confidences = [
                    item["confidence"]
                    for item in detections
                ]

                average_confidence = (
                    float(np.mean(frame_confidences))
                    if frame_confidences
                    else 0.0
                )

                timestamp = (
                    (frame_number - 1) / fps
                    if fps > 0
                    else 0.0
                )

                records.append({

                    "frame_number": frame_number,

                    "timestamp_seconds": round(
                        timestamp,
                        3,
                    ),

                    "inference_performed": (
                        inference_performed
                    ),

                    "people_count": people_count,

                    "track_ids": ",".join(
                        track_ids
                    ),

                    "average_confidence": round(
                        average_confidence,
                        4,
                    ),

                })

                # --------------------------------------------
                # LOGGING
                # --------------------------------------------

                if frame_number % 100 == 0:

                    progress = (
                        frame_number / total_frames
                    ) * 100

                    logger.info(
                        "Processing: %.1f%% | Frame %s/%s | People: %s",
                        progress,
                        frame_number,
                        total_frames,
                        people_count,
                    )

            # ------------------------------------------------
            # E. RELEASE VIDEO HANDLES
            # ------------------------------------------------

            capture.release()
            capture = None

            writer.release()
            writer = None

            if processed_frames == 0:

                raise VideoProcessingError(
                    "No frames were processed."
                )

            # ------------------------------------------------
            # F. GENERATE FINAL VIDEO
            # ------------------------------------------------

            self.convert_to_h264(
                temp_output,
                output_video,
                input_video,
            )

            if not output_video.exists():

                raise VideoProcessingError(
                    "Output video was not generated."
                )

            # ------------------------------------------------
            # G. GENERATE CSV
            # ------------------------------------------------

            self.write_csv(
                csv_path,
                records,
            )

            # ------------------------------------------------
            # H. CALCULATE ANALYTICS
            # ------------------------------------------------

            elapsed_time = (
                time.perf_counter() - start_time
            )

            average_people = (
                float(np.mean(people_per_frame))
                if people_per_frame
                else 0.0
            )

            peak_people = (
                max(people_per_frame)
                if people_per_frame
                else 0
            )

            average_confidence = (
                float(np.mean(confidence_values))
                if confidence_values
                else 0.0
            )

            processing_fps = (
                processed_frames / elapsed_time
                if elapsed_time > 0
                else 0.0
            )

            analytics = {

                "total_unique_people": len(
                    unique_people
                ),

                "unique_people": len(
                    unique_people
                ),

                "total_detections": detection_count,

                "peak_people": peak_people,

                "average_people": round(
                    average_people,
                    2,
                ),

                "processed_frames": processed_frames,

                "total_frames": total_frames,

                "inference_frames": inference_frames,

                "processing_time": round(
                    elapsed_time,
                    2,
                ),

                "processing_fps": round(
                    processing_fps,
                    2,
                ),

                "average_confidence": round(
                    average_confidence,
                    4,
                ),

                "confidence_threshold": confidence,

                "frame_interval": frame_interval,

                "video_duration": round(
                    processed_frames / fps,
                    2,
                ),

                "video_width": width,

                "video_height": height,

                "video_fps": fps,

                "detection_engine": "YOLO11",

                "tracking_engine": "ByteTrack",

                "inference_runtime": "TensorFlow Lite",

                "output_video": str(output_video),

                "csv_file": (
                    str(csv_path)
                    if csv_path
                    else None
                ),

                "output_video_size": (
                    output_video.stat().st_size
                ),

            }

            logger.info(
                "VIDEO PROCESSING COMPLETED"
            )

            logger.info(
                "Analytics: %s",
                analytics,
            )

            return analytics

        except Exception as exc:

            logger.exception(
                "Video processing error"
            )

            output_video.unlink(
                missing_ok=True
            )

            raise

        finally:

            if capture is not None:
                capture.release()

            if writer is not None:
                writer.release()

            temp_output.unlink(
                missing_ok=True
            )


# ============================================================
# 12. MODULE INFORMATION
# ============================================================

def get_processor_info():

    return {
        "application": "PragyanAI People Detection",
        "detection_engine": "YOLO11",
        "tracking_engine": "ByteTrack",
        "inference_runtime": "TensorFlow Lite",
        "video_library": "OpenCV",
        "output_format": "MP4",
        "analytics_format": "CSV",
    }

