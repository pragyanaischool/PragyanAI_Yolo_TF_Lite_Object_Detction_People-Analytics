
import os
import uuid
import time
import shutil
import threading
import subprocess

from pathlib import Path

import cv2
import pandas as pd
import imageio_ffmpeg

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
    Request
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "yolo11n.tflite")
    )
)

if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH

UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", "/tmp/pragyanai_uploads")
)

OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", "/tmp/pragyanai_outputs")
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD TFLITE MODEL
# =====================================================

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"TFLite model not found: {MODEL_PATH}. "
        "Place yolo11n.tflite inside backend/models/"
    )

print("Loading YOLO11 TFLite model...")

model = YOLO(str(MODEL_PATH))

print("YOLO11 TFLite model loaded successfully!")


# Prevent simultaneous requests from sharing tracker state.
model_lock = threading.Lock()


# =====================================================
# 3. FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="PragyanAI People Detection API",
    description="YOLO TFLite People Detection and Counting",
    version="1.0.0"
)


# =====================================================
# 4. CORS CONFIGURATION
# =====================================================

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

frontend_origin = os.getenv(
    "FRONTEND_ORIGIN",
    ""
).strip().rstrip("/")

if frontend_origin:
    allowed_origins.append(frontend_origin)

extra_origins = os.getenv("EXTRA_CORS_ORIGINS", "")

allowed_origins.extend([
    origin.strip().rstrip("/")
    for origin in extra_origins.split(",")
    if origin.strip()
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(allowed_origins)),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================================
# 5. SERVE PROCESSED VIDEO AND CSV
# =====================================================

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)


# =====================================================
# 6. HEALTH CHECK
# =====================================================

@app.get("/")
def home():

    return {
        "application": "PragyanAI People Detection",
        "status": "running",
        "model": MODEL_PATH.name
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_exists": MODEL_PATH.exists(),
        "model": MODEL_PATH.name
    }


# =====================================================
# 7. VIDEO CONVERSION
# =====================================================

def convert_video(input_path, output_path):

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Video conversion failed: " + result.stderr[-1500:]
        )


# =====================================================
# 8. RESET TRACKER BETWEEN VIDEO REQUESTS
# =====================================================

def reset_trackers():

    predictor = getattr(model, "predictor", None)

    if predictor is not None:

        trackers = getattr(
            predictor,
            "trackers",
            []
        )

        for tracker in trackers:
            if hasattr(tracker, "reset"):
                tracker.reset()


# =====================================================
# 9. MAIN VIDEO PROCESSING API
# =====================================================

@app.post("/api/process")
def process_video(

    request: Request,

    file: UploadFile = File(...),

    confidence: float = Form(0.25),

    frame_interval: int = Form(1)

):

    # -----------------------------------------------
    # Validate input
    # -----------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload a video."
        )

    if not 0.10 <= confidence <= 0.90:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0.10 and 0.90."
        )

    if not 1 <= frame_interval <= 10:
        raise HTTPException(
            status_code=400,
            detail="Frame interval must be between 1 and 10."
        )

    allowed_extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v"
    ]

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    # -----------------------------------------------
    # Create unique file names
    # -----------------------------------------------

    job_id = uuid.uuid4().hex

    input_path = UPLOAD_DIR / f"{job_id}{extension}"

    raw_path = OUTPUT_DIR / f"{job_id}_raw.mp4"

    output_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"

    csv_path = OUTPUT_DIR / f"{job_id}_report.csv"

    try:

        # -------------------------------------------
        # Save uploaded video
        # -------------------------------------------

        with input_path.open("wb") as destination:

            shutil.copyfileobj(
                file.file,
                destination
            )

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():

            raise HTTPException(
                status_code=400,
                detail="Unable to open uploaded video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if width <= 0 or height <= 0:

            cap.release()

            raise HTTPException(
                status_code=400,
                detail="Invalid video resolution."
            )

        # -------------------------------------------
        # Initialize output video
        # -------------------------------------------

        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():

            cap.release()

            raise HTTPException(
                status_code=500,
                detail="Unable to initialize output video."
            )

        # -------------------------------------------
        # Counting variables
        # -------------------------------------------

        unique_ids = set()

        frame_counts = []

        frame_number = 0

        processed_frames = 0

        maximum_people = 0

        start_time = time.time()

        try:

            # ---------------------------------------
            # YOLO + ByteTrack processing
            # ---------------------------------------

            with model_lock:

                reset_trackers()

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    if frame_number % frame_interval == 0:

                        results = model.track(

                            frame,

                            persist=True,

                            tracker="bytetrack.yaml",

                            classes=[0],

                            conf=confidence,

                            imgsz=640,

                            verbose=False

                        )

                        result = results[0]

                        current_count = (
                            len(result.boxes)
                            if result.boxes is not None
                            else 0
                        )

                        # Get unique tracking IDs

                        if (
                            result.boxes is not None
                            and result.boxes.id is not None
                        ):

                            ids = (
                                result.boxes.id
                                .int()
                                .cpu()
                                .tolist()
                            )

                            unique_ids.update(ids)

                        # Draw bounding boxes

                        annotated = result.plot()

                        processed_frames += 1

                    else:

                        annotated = frame.copy()

                        current_count = None

                        cv2.putText(
                            annotated,
                            "Inference skipped",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2
                        )

                    if current_count is not None:

                        maximum_people = max(
                            maximum_people,
                            current_count
                        )

                    frame_counts.append(current_count)

                    # --------------------------------
                    # Display analytics on video
                    # --------------------------------

                    cv2.rectangle(
                        annotated,
                        (10, 10),
                        (450, 100),
                        (0, 0, 0),
                        -1
                    )

                    count_text = (
                        str(current_count)
                        if current_count is not None
                        else "Not processed"
                    )

                    cv2.putText(
                        annotated,
                        f"People in Frame: {count_text}",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"Unique IDs: {len(unique_ids)}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (255, 255, 255),
                        2
                    )

                    writer.write(annotated)

                    frame_number += 1

        finally:

            cap.release()

            writer.release()

        if frame_number == 0:

            raise HTTPException(
                status_code=400,
                detail="No frames found in uploaded video."
            )

        # -------------------------------------------
        # Convert to browser-compatible MP4
        # -------------------------------------------

        convert_video(
            raw_path,
            output_path
        )

        # -------------------------------------------
        # Generate analytics
        # -------------------------------------------

        processing_time = round(
            time.time() - start_time,
            2
        )

        observed_counts = [
            count
            for count in frame_counts
            if count is not None
        ]

        average_people = (

            round(
                sum(observed_counts) / len(observed_counts),
                2
            )

            if observed_counts
            else 0

        )

        summary = {

            "job_id": job_id,

            "total_video_frames": frame_number,

            "reported_video_frames": total_frames,

            "frames_processed_by_model": processed_frames,

            "unique_tracking_ids_seen": len(unique_ids),

            "maximum_people_in_processed_frame": maximum_people,

            "average_people_in_processed_frames": average_people,

            "processing_time_seconds": processing_time,

            "video_fps": round(fps, 2),

            "resolution": f"{width} x {height}",

            "frame_interval": frame_interval,

            "confidence": confidence,

            "counting_note": (
                "Tracking IDs estimate distinct people. "
                "A person may receive a new ID after re-entering "
                "or when tracking is interrupted."
            )

        }

        # -------------------------------------------
        # Generate CSV report
        # -------------------------------------------

        report_df = pd.DataFrame([

            {
                "Metric": key,
                "Value": value
            }

            for key, value in summary.items()

        ])

        report_df.to_csv(
            csv_path,
            index=False
        )

        # -------------------------------------------
        # Return API response
        # -------------------------------------------

        base_url = str(
            request.base_url
        ).rstrip("/")

        return {

            "success": True,

            "summary": summary,

            "annotated_video_url": (
                f"{base_url}/outputs/{output_path.name}"
            ),

            "csv_url": (
                f"{base_url}/outputs/{csv_path.name}"
            )

        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(error)}"
        )

    finally:

        file.file.close()

        if input_path.exists():
            input_path.unlink(missing_ok=True)

        if raw_path.exists():
            raw_path.unlink(missing_ok=True)
