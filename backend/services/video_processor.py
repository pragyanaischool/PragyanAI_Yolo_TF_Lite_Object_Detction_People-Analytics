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
import importlib.metadata

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

        logger.info("=" * 72)
        logger.info("[MODEL INIT] Starting YOLO11 TFLite initialization")
        logger.info("[MODEL INIT] Model path: %s", self.model_path)
        logger.info("[MODEL INIT] Model size: %.2f MB",
                    self.model_path.stat().st_size / (1024 * 1024))
        logger.info("[MODEL INIT] imgsz=%s | tracker=%s",
                    self.imgsz, self.tracker)

        for package_name in ("ultralytics", "ai-edge-litert", "lap",
                             "tensorflow-cpu", "numpy", "opencv-python-headless"):
            try:
                logger.info("[ENV] %s=%s", package_name,
                            importlib.metadata.version(package_name))
            except importlib.metadata.PackageNotFoundError:
                logger.warning("[ENV] %s distribution not found", package_name)

        model_init_start = time.perf_counter()
        logger.info("[MODEL INIT] Loading YOLO model: %s", self.model_path)

        try:

            self.model = YOLO(
                str(self.model_path),
                task="detect",
            )

        except Exception as exc:

            logger.exception("[MODEL INIT FAILED] Unable to load YOLO model")
            raise VideoProcessingError(
                f"Unable to load YOLO model: {exc}"
            ) from exc

        self.model_loaded = True
        self._inference_call_count = 0

        logger.info(
            "[MODEL INIT COMPLETE] YOLO model initialized in %.2f seconds",
            time.perf_counter() - model_init_start
        )
        logger.info("=" * 72)

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
                logger.info("[TRACKER] Predictor not initialized yet; "
                            "Ultralytics will initialize it on first track call.")
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

            logger.info("[TRACKER] Tracker reset completed. Tracker instances: %s",
                        len(trackers) if trackers else 0)

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
        Run YOLO11 TFLite inference and ByteTrack.

        Logs every inference call, including per-frame detections and timing.

        Returns:
            List of person detection dictionaries.
        """

        call_number = getattr(self, "_inference_call_count", 0) + 1
        self._inference_call_count = call_number

        frame_start = time.perf_counter()
        frame_height, frame_width = frame.shape[:2]

        inference_type = (
            "FIRST_INFERENCE"
            if call_number == 1
            else "SUBSEQUENT_INFERENCE"
        )

        # Log every inference call so Render logs prove that each sampled
        # frame is actively being processed.
        log_this_call = True

        if log_this_call:
            logger.info(
                "[YOLO INFERENCE START] type=%s | call=%s | "
                "frame_size=%sx%s | imgsz=%s | confidence=%.3f",
                inference_type,
                call_number,
                frame_width,
                frame_height,
                self.imgsz,
                float(confidence),
            )

        try:
            # The first call may include lazy predictor, LiteRT interpreter,
            # and tracker initialization. Measure model.track separately.
            model_start = time.perf_counter()

            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                conf=float(confidence),
                classes=[0],
                imgsz=self.imgsz,
                verbose=False,
            )

            model_track_seconds = time.perf_counter() - model_start

            detections = []
            raw_box_count = 0

            if results:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)

                    track_ids = None
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)

                    raw_box_count = len(xyxy)

                    for index, coordinates in enumerate(xyxy):
                        class_id = int(classes[index])

                        if class_id != 0:
                            continue

                        x1, y1, x2, y2 = coordinates.tolist()

                        track_id = None
                        if track_ids is not None and index < len(track_ids):
                            track_id = int(track_ids[index])

                        detections.append({
                            "track_id": track_id,
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "confidence": float(confidences[index]),
                            "class_id": class_id,
                        })

            total_seconds = time.perf_counter() - frame_start
            postprocess_seconds = max(
                total_seconds - model_track_seconds, 0.0
            )

            tracked_ids = [
                item["track_id"]
                for item in detections
                if item["track_id"] is not None
            ]

            average_confidence = (
                float(np.mean([
                    item["confidence"] for item in detections
                ]))
                if detections
                else 0.0
            )

            should_log_result = (
                log_this_call
                or model_track_seconds >= 2.0
                or total_seconds >= 2.0
            )

            if should_log_result:
                logger.info(
                    "[YOLO INFERENCE COMPLETE] type=%s | call=%s | "
                    "model_track=%.3fs | postprocess=%.3fs | "
                    "total=%.3fs | inference_fps=%.3f | "
                    "raw_boxes=%s | people=%s | tracked_ids=%s | "
                    "avg_confidence=%.3f",
                    inference_type,
                    call_number,
                    model_track_seconds,
                    postprocess_seconds,
                    total_seconds,
                    (1.0 / total_seconds) if total_seconds > 0 else 0.0,
                    raw_box_count,
                    len(detections),
                    tracked_ids[:30],
                    average_confidence,
                )

            return detections

        except Exception as exc:
            elapsed = time.perf_counter() - frame_start

            logger.exception(
                "[YOLO INFERENCE FAILED] type=%s | call=%s | "
                "elapsed=%.3fs | frame_size=%sx%s | error=%s",
                inference_type,
                call_number,
                elapsed,
                frame_width,
                frame_height,
                str(exc),
            )

            raise VideoProcessingError(
                "YOLO11 inference or ByteTrack failed. "
                "Check whether your installed Ultralytics version "
                "supports TFLite tracking. "
                f"Details: {exc}"
            ) from exc

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

        logger.info("[OUTPUT] OpenCV writer initialized: %s | codec=mp4v | "
                    "fps=%.2f | resolution=%sx%s",
                    output_path, fps, width, height)
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

            logger.info("[ENCODE] Starting FFmpeg H.264 conversion")
            logger.info("[ENCODE] Input=%s | Output=%s", input_path, output_path)

            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
            )

            logger.info("[ENCODE] FFmpeg process completed successfully")

            if (
                output_path.exists()
                and output_path.stat().st_size > 0
            ):

                input_path.unlink(missing_ok=True)

                logger.info(
                    "H.264 conversion successful"
                )

                return True

        except subprocess.CalledProcessError as exc:

            stderr_text = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr or "")
            )
            logger.exception("[ENCODE] FFmpeg H.264 conversion failed")
            if stderr_text:
                logger.error("[ENCODE] FFmpeg stderr (last 4000 chars): %s",
                             stderr_text[-4000:])

        except Exception as exc:

            logger.exception("[ENCODE] H.264 conversion unavailable: %s", exc)

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

        logger.info("[CSV] Writing %s frame records to %s",
                    len(records), csv_path)

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

        csv_size = csv_path.stat().st_size if csv_path.exists() else 0
        logger.info("[CSV] Analytics saved: %s | records=%s | size=%.2f KB",
                    csv_path, len(records), csv_size / 1024)

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
        progress_callback=None,
    ):
        """
        Complete video processing pipeline.

        Args:
            input_video: Uploaded source video.
            output_video: Processed MP4 output.
            confidence: YOLO confidence threshold.
            frame_interval: Run inference every N frames.
            csv_path: CSV analytics destination.
            progress_callback: Optional callback(progress, current_frame, total_frames, eta_seconds, frame_update).

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

        logger.info("=" * 72)
        logger.info("[REQUEST] Video processing requested")
        logger.info("[REQUEST] Input=%s | size=%.2f MB",
                    input_video, input_video.stat().st_size / (1024 * 1024))
        logger.info("[REQUEST] Output=%s | confidence=%.3f | frame_interval=%s",
                    output_video, float(confidence), int(frame_interval))
        logger.info("[REQUEST] CSV destination=%s", csv_path)

        # Lock model/tracker during processing.
        # Avoid concurrent calls corrupting tracker state.
        logger.info("[LOCK] Waiting for video processor lock")
        with self.lock:
            logger.info("[LOCK] Processor lock acquired")

            return self._process_video(
                input_video,
                output_video,
                float(confidence),
                int(frame_interval),
                csv_path,
                progress_callback,
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
        progress_callback=None,
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

        inference_total_seconds = 0.0
        frame_read_seconds = 0.0
        output_write_seconds = 0.0
        progress_interval = 25
        last_progress_time = start_time
        last_progress_frame = 0

        logger.info("=" * 72)
        logger.info("[PIPELINE START] YOLO11 + ByteTrack video processing")
        logger.info("[CONFIG] confidence=%.3f | frame_interval=%s | imgsz=%s | tracker=%s",
                    confidence, frame_interval, self.imgsz, self.tracker)
        logger.info("[FILES] input=%s | output=%s | temp=%s | csv=%s",
                    input_video, output_video, temp_output, csv_path)

        try:

            # ------------------------------------------------
            # A. READ VIDEO
            # ------------------------------------------------

            stage_start = time.perf_counter()
            logger.info("[STAGE 1/8] Reading video metadata")

            video_info = self.get_video_info(
                input_video
            )

            width = video_info["width"]
            height = video_info["height"]

            fps = video_info["fps"]

            total_frames = video_info["frame_count"]

            logger.info("[VIDEO INFO] %s", video_info)
            logger.info("[STAGE 1/8 COMPLETE] Metadata read in %.3fs",
                        time.perf_counter() - stage_start)

            stage_start = time.perf_counter()
            logger.info("[STAGE 2/8] Opening video capture")

            capture = cv2.VideoCapture(
                str(input_video)
            )

            if not capture.isOpened():

                raise VideoProcessingError(
                    "OpenCV cannot open input video."
                )

            logger.info("[VIDEO INPUT] OpenCV capture opened successfully")

            # ------------------------------------------------
            # B. CREATE OUTPUT WRITER
            # ------------------------------------------------

            stage_start = time.perf_counter()
            logger.info("[STAGE 3/8] Creating temporary output video writer")

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
            logger.info("[STAGE 3/8 COMPLETE] Output writer ready in %.3fs",
                        time.perf_counter() - stage_start)

            # ------------------------------------------------
            # C. RESET TRACKER
            # ------------------------------------------------

            stage_start = time.perf_counter()
            logger.info("[STAGE 4/8] Resetting tracker state")
            self.reset_tracker()
            logger.info("[STAGE 4/8 COMPLETE] Tracker reset check finished in %.3fs",
                        time.perf_counter() - stage_start)

            logger.info("[STAGE 5/8] Starting frame-by-frame processing | "
                        "total_frames=%s | expected_inference_frames=%s",
                        total_frames,
                        (total_frames + frame_interval - 1) // frame_interval)

            # ------------------------------------------------
            # D. PROCESS FRAMES
            # ------------------------------------------------

            frame_number = 0
            self._inference_call_count = 0
            logger.info("[INFERENCE COUNTER] Reset for this video job")

            while True:

                frame_started = time.perf_counter()
                read_start = time.perf_counter()
                success, frame = capture.read()
                current_read_seconds = time.perf_counter() - read_start
                frame_read_seconds += current_read_seconds

                if not success:
                    logger.info("[VIDEO READ] End of stream at frame %s", frame_number)
                    break

                frame_number += 1

                inference_performed = (
                    (frame_number - 1) % frame_interval == 0
                )

                # --------------------------------------------
                # RUN YOLO11 + BYTETRACK
                # --------------------------------------------

                current_inference_seconds = 0.0
                if inference_performed:

                    inference_start = time.perf_counter()
                    detections = self.detect_and_track(
                        frame,
                        confidence,
                    )
                    current_inference_seconds = time.perf_counter() - inference_start
                    inference_total_seconds += current_inference_seconds

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

                annotate_start = time.perf_counter()
                annotated_frame = self.annotate_frame(
                    frame,
                    detections,
                    frame_number,
                    people_count,
                )
                annotate_seconds = time.perf_counter() - annotate_start

                # --------------------------------------------
                # WRITE OUTPUT FRAME
                # --------------------------------------------

                write_start = time.perf_counter()
                writer.write(
                    annotated_frame
                )
                current_write_seconds = time.perf_counter() - write_start
                output_write_seconds += current_write_seconds

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
                # DETAILED PER-FRAME ACTIVITY LOG
                # --------------------------------------------

                frame_total_seconds = time.perf_counter() - frame_started
                frame_status = (
                    "INFERRED" if inference_performed else "TRACKS_CARRIED_FORWARD"
                )
                current_track_ids = [
                    item["track_id"] for item in detections
                    if item["track_id"] is not None
                ]
                logger.info(
                    "[FRAME %s/%s] status=%s | timestamp=%.3fs | "
                    "people=%s | track_ids=%s | detections=%s | "
                    "read=%.4fs | inference=%.4fs | annotate=%.4fs | "
                    "write=%.4fs | frame_total=%.4fs | elapsed=%.2fs",
                    frame_number, total_frames, frame_status, timestamp,
                    people_count, current_track_ids, len(detections),
                    current_read_seconds, current_inference_seconds,
                    annotate_seconds, current_write_seconds,
                    frame_total_seconds, time.perf_counter() - start_time,
                )

                for detection_index, detection in enumerate(detections, start=1):
                    logger.info(
                        "[FRAME %s PERSON %s] track_id=%s | "
                        "confidence=%.4f | bbox=(%s,%s,%s,%s)",
                        frame_number, detection_index,
                        detection.get("track_id"),
                        detection.get("confidence", 0.0),
                        detection.get("x1"), detection.get("y1"),
                        detection.get("x2"), detection.get("y2"),
                    )

                # --------------------------------------------
                # PERIODIC PROGRESS SUMMARY + CALLBACK
                # --------------------------------------------

                if (
                    frame_number == 1
                    or frame_number % progress_interval == 0
                    or frame_number == total_frames
                ):
                    now = time.perf_counter()
                    elapsed = now - start_time
                    interval_elapsed = now - last_progress_time
                    interval_frames = frame_number - last_progress_frame

                    progress = (
                        frame_number / total_frames * 100
                        if total_frames else 0.0
                    )
                    processing_fps = (
                        processed_frames / elapsed if elapsed > 0 else 0.0
                    )
                    remaining_frames = max(total_frames - frame_number, 0)
                    eta_seconds = (
                        remaining_frames / processing_fps
                        if processing_fps > 0 else 0.0
                    )
                    current_ids = [
                        item["track_id"] for item in detections
                        if item["track_id"] is not None
                    ]

                    logger.info(
                        "[PROGRESS] %.1f%% | frame=%s/%s | "
                        "people_in_frame=%s | tracked_ids=%s | "
                        "unique_ids_so_far=%s | detections_total=%s | "
                        "inference_frames=%s | speed=%.2f frames/sec | "
                        "elapsed=%.1fs | ETA=%.1fs",
                        progress,
                        frame_number,
                        total_frames,
                        people_count,
                        current_ids[:20],
                        len(unique_people),
                        detection_count,
                        inference_frames,
                        processing_fps,
                        elapsed,
                        eta_seconds,
                    )
                    logger.info(
                        "[PERFORMANCE] last_%s_frames_in_%.2fs | "
                        "avg_inference=%.3fs | read_total=%.2fs | write_total=%.2fs",
                        interval_frames,
                        interval_elapsed,
                        (inference_total_seconds / inference_frames)
                        if inference_frames else 0.0,
                        frame_read_seconds,
                        output_write_seconds,
                    )

                    last_progress_time = now
                    last_progress_frame = frame_number

                # Publish a live update after EVERY completed frame. The optional
                # fifth argument carries the partial detection result to the API.
                # A four-argument fallback preserves compatibility with older callers.
                if progress_callback is not None:
                    elapsed_now = time.perf_counter() - start_time
                    frame_progress = (
                        frame_number / total_frames * 100
                        if total_frames else 0.0
                    )
                    current_speed = (
                        processed_frames / elapsed_now
                        if elapsed_now > 0 else 0.0
                    )
                    frames_left = max(total_frames - frame_number, 0)
                    current_eta = (
                        frames_left / current_speed
                        if current_speed > 0 else 0.0
                    )
                    frame_update = {
                        "event": "frame_processed",
                        "frame_number": frame_number,
                        "total_frames": total_frames,
                        "progress": round(frame_progress, 2),
                        "timestamp_seconds": round(timestamp, 3),
                        "status": frame_status,
                        "objects_found": len(detections),
                        "people_found": people_count,
                        "track_ids": current_track_ids,
                        "inference_performed": inference_performed,
                        "inference_seconds": round(current_inference_seconds, 4),
                        "frame_seconds": round(frame_total_seconds, 4),
                        "unique_people_so_far": len(unique_people),
                        "eta_seconds": round(current_eta, 1),
                    }
                    try:
                        try:
                            progress_callback(
                                frame_progress,
                                frame_number,
                                total_frames,
                                current_eta,
                                frame_update,
                            )
                        except TypeError:
                            # Backward compatibility: old callbacks accept 4 args.
                            progress_callback(
                                frame_progress,
                                frame_number,
                                total_frames,
                                current_eta,
                            )
                    except Exception:
                        logger.exception(
                            "[PROGRESS CALLBACK ERROR] Unable to publish per-frame update"
                        )

            # ------------------------------------------------
            # E. RELEASE VIDEO HANDLES
            # ------------------------------------------------

            logger.info("[STAGE 5/8 COMPLETE] Frame loop finished | "
                        "frames_read=%s | frames_written=%s | "
                        "inference_frames=%s | detections=%s | unique_ids=%s",
                        frame_number, processed_frames, inference_frames,
                        detection_count, len(unique_people))

            logger.info("[VIDEO RELEASE] Releasing OpenCV capture and writer")
            capture.release()
            capture = None

            writer.release()
            writer = None
            logger.info("[VIDEO RELEASE] Handles released successfully")

            if processed_frames == 0:

                raise VideoProcessingError(
                    "No frames were processed."
                )

            # ------------------------------------------------
            # F. GENERATE FINAL VIDEO
            # ------------------------------------------------

            stage_start = time.perf_counter()
            logger.info("[STAGE 6/8] Converting intermediate MP4 to final H.264 MP4")
            h264_success = self.convert_to_h264(
                temp_output,
                output_video,
                input_video,
            )
            logger.info("[STAGE 6/8 COMPLETE] Final video conversion finished in %.2fs | "
                        "h264_conversion=%s",
                        time.perf_counter() - stage_start, h264_success)

            if not output_video.exists():

                raise VideoProcessingError(
                    "Output video was not generated."
                )

            # ------------------------------------------------
            # G. GENERATE CSV
            # ------------------------------------------------

            stage_start = time.perf_counter()
            logger.info("[STAGE 7/8] Generating CSV analytics")
            self.write_csv(
                csv_path,
                records,
            )
            logger.info("[STAGE 7/8 COMPLETE] CSV generation finished in %.3fs",
                        time.perf_counter() - stage_start)

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

            logger.info("[STAGE 8/8 COMPLETE] Analytics calculated")
            logger.info("[OUTPUT CHECK] video_exists=%s | video_size=%.2f MB | "
                        "csv_exists=%s",
                        output_video.exists(),
                        output_video.stat().st_size / (1024 * 1024),
                        bool(csv_path and csv_path.exists()))
            logger.info("[SUMMARY] frames=%s/%s | inference_frames=%s | "
                        "total_detections=%s | unique_track_ids=%s | "
                        "avg_inference=%.3fs | processing_fps=%.2f",
                        processed_frames, total_frames, inference_frames,
                        detection_count, len(unique_people),
                        (inference_total_seconds / inference_frames)
                        if inference_frames else 0.0,
                        processing_fps)
            logger.info("[PIPELINE COMPLETE] Total processing time=%.2fs",
                        elapsed_time)
            logger.info("Analytics: %s", analytics)
            logger.info("=" * 72)

            return analytics

        except Exception as exc:

            logger.exception("[PIPELINE FAILED] Video processing error at frame=%s "
                             "| processed_frames=%s | inference_frames=%s",
                             processed_frames, processed_frames, inference_frames)

            output_video.unlink(
                missing_ok=True
            )

            raise

        finally:

            logger.info("[CLEANUP] Releasing remaining resources and temporary files")

            if capture is not None:
                capture.release()

            if writer is not None:
                writer.release()

            temp_output.unlink(
                missing_ok=True
            )
            logger.info("[CLEANUP] Cleanup finished")


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
