
"""
===========================================================
PragyanAI - People Detection & Counting
File: backend/services/video_processor.py
===========================================================

Features:
1. Read uploaded video.
2. Extract video frames.
3. Detect people using YOLO11 TFLite.
4. Track people using ByteTrack.
5. Assign tracking IDs.
6. Count people in each processed frame.
7. Draw bounding boxes and tracking IDs.
8. Generate annotated MP4 video.
9. Generate frame-wise CSV analytics.
10. Return processing summary.

Dependencies:
    opencv-python-headless
    ultralytics
    pandas
    imageio-ffmpeg
"""

import cv2
import time
import subprocess
import threading

import pandas as pd
import imageio_ffmpeg

from pathlib import Path

from services.detection import PeopleDetector
from services.tracking import PeopleTracker


class VideoProcessor:

    def __init__(
        self,
        model_path="models/yolo11n.tflite",
        confidence=0.25,
        image_size=640
    ):

        self.model_path = model_path

        self.confidence = confidence

        self.image_size = image_size

        # Initialize YOLO detector.
        self.detector = PeopleDetector(
            model_path=model_path,
            confidence=confidence,
            image_size=image_size
        )

        # Initialize ByteTrack tracker.
        self.tracker = PeopleTracker(
            model=self.detector.model,
            confidence=confidence,
            image_size=image_size
        )

        # Prevent concurrent access to the same model.
        self.model_lock = threading.Lock()

        print("VideoProcessor initialized successfully.")

    # =======================================================
    # 1. VIDEO INFORMATION
    # =======================================================

    def get_video_info(self, video_path):

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Unable to open video.")

        info = {

            "fps": cap.get(cv2.CAP_PROP_FPS),

            "width": int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            ),

            "height": int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ),

            "total_frames": int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

        }

        cap.release()

        if not info["fps"] or info["fps"] <= 0:
            info["fps"] = 25.0

        info["duration_seconds"] = round(
            info["total_frames"] / info["fps"],
            2
        )

        return info

    # =======================================================
    # 2. ANNOTATE VIDEO FRAME
    # =======================================================

    def annotate_frame(
        self,
        frame,
        tracked_people,
        frame_number,
        timestamp
    ):

        annotated = frame.copy()

        # Draw individual person bounding boxes.
        for person in tracked_people:

            bbox = person["bbox"]

            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]

            person_id = person["person_id"]

            confidence = person["confidence"]

            # Bounding box.
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Tracking label.
            label = (
                f"Person ID: {person_id} "
                f"| Conf: {confidence:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # Frame analytics.
        current_count = len(tracked_people)

        unique_count = self.tracker.get_unique_count()

        # Analytics background.
        cv2.rectangle(
            annotated,
            (10, 10),
            (440, 145),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            "PRAGYANAI PEOPLE ANALYTICS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"People in Frame: {current_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Unique Track IDs: {unique_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Frame: {frame_number} | Time: {timestamp:.2f}s",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return annotated

    # =======================================================
    # 3. PROCESS VIDEO
    # =======================================================

    def process(
        self,
        input_video,
        output_video,
        confidence=None,
        frame_interval=1,
        csv_path=None
    ):

        input_video = Path(input_video)

        output_video = Path(output_video)

        if csv_path is not None:
            csv_path = Path(csv_path)

        # Validate input.
        if not input_video.exists():
            raise FileNotFoundError(
                f"Video not found: {input_video}"
            )

        # Create output directories.
        output_video.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if csv_path:
            csv_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

        frame_interval = max(
            1,
            int(frame_interval)
        )

        confidence = (
            confidence
            if confidence is not None
            else self.confidence
        )

        if not 0.05 <= confidence <= 0.95:
            raise ValueError(
                "Confidence must be between 0.05 and 0.95."
            )

        # Get video information.
        video_info = self.get_video_info(
            input_video
        )

        fps = video_info["fps"]

        width = video_info["width"]

        height = video_info["height"]

        total_frames = video_info["total_frames"]

        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions.")

        # Open input video.
        cap = cv2.VideoCapture(
            str(input_video)
        )

        if not cap.isOpened():
            raise ValueError("Unable to read video.")

        # Temporary video file.
        temp_video = output_video.with_name(
            output_video.stem + "_temp.mp4"
        )

        # Initialize video writer.
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(temp_video),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()

            raise RuntimeError(
                "Unable to initialize video writer."
            )

        # Reset tracker before processing a new video.
        frame_number = 0

        processed_frames = 0

        total_detections = 0

        maximum_people = 0

        total_people_observed = 0

        analytics_rows = []

        start_time = time.time()

        try:

            # The same model and tracker must not be
            # accessed simultaneously by multiple requests.
            with self.model_lock:

                self.tracker.reset()

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    frame_number += 1

                    timestamp = (
                        frame_number - 1
                    ) / fps

                    tracked_people = []

                    # ---------------------------------------
                    # 3.1 Extract and process selected frames
                    # ---------------------------------------

                    should_process = (
                        (frame_number - 1)
                        % frame_interval == 0
                    )

                    if should_process:

                        tracked_people = self.tracker.track(
                            frame,
                            confidence=confidence
                        )

                        processed_frames += 1

                        current_count = len(
                            tracked_people
                        )

                        total_detections += current_count

                        total_people_observed += current_count

                        maximum_people = max(
                            maximum_people,
                            current_count
                        )

                        person_ids = [
                            person["person_id"]
                            for person in tracked_people
                        ]

                        # Save frame-wise analytics.
                        analytics_rows.append({

                            "frame_number": frame_number,

                            "timestamp_seconds": round(
                                timestamp,
                                3
                            ),

                            "people_detected": current_count,

                            "tracked_person_ids": ",".join(
                                map(str, person_ids)
                            )

                        })

                        # Draw bounding boxes and counts.
                        annotated_frame = self.annotate_frame(
                            frame,
                            tracked_people,
                            frame_number,
                            timestamp
                        )

                    else:

                        # Preserve original video frame.
                        annotated_frame = frame.copy()

                        cv2.putText(
                            annotated_frame,
                            "Inference skipped",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2
                        )

                    # Write frame to output video.
                    writer.write(
                        annotated_frame
                    )

        finally:

            cap.release()

            writer.release()

        # ===================================================
        # 4. CONVERT OUTPUT TO H264 MP4
        # ===================================================

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        command = [

            ffmpeg_path,

            "-y",

            "-i", str(temp_video),

            "-c:v", "libx264",

            "-pix_fmt", "yuv420p",

            "-preset", "fast",

            "-movflags", "+faststart",

            str(output_video)

        ]

        conversion = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if conversion.returncode != 0:

            if temp_video.exists():
                temp_video.unlink()

            raise RuntimeError(
                "FFmpeg conversion failed: "
                + conversion.stderr[-2000:]
            )

        if temp_video.exists():
            temp_video.unlink()

        # ===================================================
        # 5. GENERATE CSV ANALYTICS
        # ===================================================

        analytics_df = pd.DataFrame(
            analytics_rows,
            columns=[
                "frame_number",
                "timestamp_seconds",
                "people_detected",
                "tracked_person_ids"
            ]
        )

        if csv_path is not None:

            analytics_df.to_csv(
                csv_path,
                index=False
            )

        # ===================================================
        # 6. GENERATE FINAL SUMMARY
        # ===================================================

        processing_time = (
            time.time() - start_time
        )

        tracking_summary = (
            self.tracker.get_tracking_summary()
        )

        average_people = (
            total_people_observed / processed_frames
            if processed_frames > 0
            else 0
        )

        summary = {

            "total_frames": total_frames,

            "processed_frames": processed_frames,

            "frame_interval": frame_interval,

            "total_detections": total_detections,

            "unique_people_tracked": (
                tracking_summary[
                    "unique_people_tracked"
                ]
            ),

            "maximum_people_in_frame": maximum_people,

            "average_people_per_processed_frame": round(
                average_people,
                2
            ),

            "video_fps": round(
                fps,
                2
            ),

            "video_width": width,

            "video_height": height,

            "video_duration_seconds": (
                video_info["duration_seconds"]
            ),

            "processing_time_seconds": round(
                processing_time,
                2
            )

        }

        # ===================================================
        # 7. RETURN RESULTS
        # ===================================================

        return {

            "success": True,

            "summary": summary,

            "tracking_details": tracking_summary,

            "analytics": analytics_rows,

            "output_video": str(output_video),

            "csv_file": (
                str(csv_path)
                if csv_path is not None
                else None
            )

        }
