
import os
import uuid
import shutil
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import cv2
import pandas as pd
import imageio_ffmpeg

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "models/yolo11n.tflite")
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

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PragyanAI People Detection & Counting API",
    description=(
        "YOLO11 TFLite based people detection, "
        "tracking, counting and video analytics."
    ),
    version="1.0.0"
)

frontend_origin = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173"
)

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    frontend_origin.rstrip("/")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(allowed_origins)),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)


# ============================================================
# 3. LOAD YOLO TFLITE MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO TFLite model not found: {MODEL_PATH}"
    )

print("Loading YOLO model:", MODEL_PATH)

model = YOLO(str(MODEL_PATH))

# Prevent simultaneous access to the same tracking model.
model_lock = threading.Lock()


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "application": "PragyanAI People Detection System",
        "status": "Running",
        "version": "1.0.0",
        "model": MODEL_PATH.name
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_exists": MODEL_PATH.exists(),
        "model_name": MODEL_PATH.name
    }


# ============================================================
# 5. TRACKER RESET
# ============================================================

def reset_tracker():

    predictor = getattr(model, "predictor", None)

    if predictor is None:
        return

    trackers = getattr(predictor, "trackers", [])

    for tracker in trackers:
        try:
            tracker.reset()
        except Exception:
            pass


# ============================================================
# 6. VIDEO PROCESSING
# ============================================================

def process_video(
    input_path,
    output_path,
    csv_path,
    confidence=0.25,
    frame_interval=1
):

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")

    frame_interval = max(1, int(frame_interval))

    # Temporary video writer.
    temp_video = output_path.with_name(
        output_path.stem + "_temp.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(temp_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Unable to create output video.")

    frame_number = 0
    processed_frames = 0
    total_detections = 0

    unique_person_ids = set()
    analytics_rows = []

    max_people = 0
    total_people_observed = 0

    start_time = datetime.now(timezone.utc)

    try:

        with model_lock:

            reset_tracker()

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1

                current_people = 0
                tracked_ids = []
                annotated_frame = frame.copy()

                # ------------------------------------------------
                # Run inference on selected frames.
                # ------------------------------------------------

                if (frame_number - 1) % frame_interval == 0:

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0],
                        conf=float(confidence),
                        imgsz=640,
                        verbose=False
                    )

                    result = results[0]

                    annotated_frame = result.plot()

                    if result.boxes is not None:

                        current_people = len(result.boxes)

                        total_detections += current_people

                        if result.boxes.id is not None:

                            ids = (
                                result.boxes.id
                                .cpu()
                                .numpy()
                                .astype(int)
                            )

                            tracked_ids = ids.tolist()

                            unique_person_ids.update(
                                tracked_ids
                            )

                    processed_frames += 1

                    total_people_observed += current_people

                    max_people = max(
                        max_people,
                        current_people
                    )

                    analytics_rows.append({

                        "frame": frame_number,

                        "timestamp_seconds": round(
                            (frame_number - 1) / fps,
                            3
                        ),

                        "people_detected": current_people,

                        "tracked_person_ids": ",".join(
                            map(str, tracked_ids)
                        )

                    })

                    # Overlay analytics.
                    cv2.rectangle(
                        annotated_frame,
                        (10, 10),
                        (440, 115),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        annotated_frame,
                        f"People Detected: {current_people}",
                        (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Unique IDs: {len(unique_person_ids)}",
                        (25, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_number}",
                        (25, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )

                else:

                    # Preserve the original video frame.
                    cv2.putText(
                        annotated_frame,
                        "Inference skipped",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )

                writer.write(annotated_frame)

    finally:

        cap.release()
        writer.release()

    # ------------------------------------------------------------
    # Convert temporary video to browser-compatible H.264 MP4.
    # ------------------------------------------------------------

    ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_executable,
        "-y",
        "-i", str(temp_video),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-movflags", "+faststart",
        str(output_path)
    ]

    conversion = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if conversion.returncode != 0:
        raise RuntimeError(
            "Video conversion failed: "
            + conversion.stderr[-2000:]
        )

    if temp_video.exists():
        temp_video.unlink()

    # ------------------------------------------------------------
    # Generate CSV analytics.
    # ------------------------------------------------------------

    df = pd.DataFrame(analytics_rows)

    df.to_csv(csv_path, index=False)

    processing_time = (
        datetime.now(timezone.utc) - start_time
    ).total_seconds()

    average_people = (
        total_people_observed / processed_frames
        if processed_frames > 0 else 0
    )

    summary = {

        "total_frames": total_frames,

        "processed_frames": processed_frames,

        "frame_interval": frame_interval,

        "total_detections": total_detections,

        "unique_people_tracked": len(unique_person_ids),

        "maximum_people_in_frame": max_people,

        "average_people_per_processed_frame": round(
            average_people,
            2
        ),

        "video_fps": round(fps, 2),

        "video_width": width,

        "video_height": height,

        "video_duration_seconds": round(
            total_frames / fps,
            2
        ),

        "processing_time_seconds": round(
            processing_time,
            2
        )

    }

    return summary


# ============================================================
# 7. VIDEO UPLOAD AND PROCESS API
# ============================================================

@app.post("/api/process")
async def process_uploaded_video(

    file: UploadFile = File(...),

    confidence: float = Form(0.25),

    frame_interval: int = Form(1)

):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No video filename provided."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0.05 and 0.95."
        )

    if not 1 <= frame_interval <= 30:
        raise HTTPException(
            status_code=400,
            detail="Frame interval must be between 1 and 30."
        )

    job_id = str(uuid.uuid4())

    input_path = UPLOAD_DIR / f"{job_id}{extension}"

    output_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"

    csv_path = OUTPUT_DIR / f"{job_id}_analytics.csv"

    try:

        # Save uploaded video in chunks.
        file_size = 0

        with input_path.open("wb") as buffer:

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                file_size += len(chunk)

                if file_size > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds 500 MB limit."
                    )

                buffer.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty."
            )

        # Run video processing.
        summary = process_video(
            input_path=input_path,
            output_path=output_path,
            csv_path=csv_path,
            confidence=confidence,
            frame_interval=frame_interval
        )

        base_url = str(os.getenv("PUBLIC_API_URL", "")).rstrip("/")

        if not base_url:
            # Works behind Render's proxy when forwarded headers
            # are supplied; configure PUBLIC_API_URL for production.
            base_url = ""

        return {

            "success": True,

            "job_id": job_id,

            "summary": summary,

            "annotated_video_url": (
                f"{base_url}/outputs/{output_path.name}"
                if base_url else f"/outputs/{output_path.name}"
            ),

            "csv_url": (
                f"{base_url}/outputs/{csv_path.name}"
                if base_url else f"/outputs/{csv_path.name}"
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

        await file.close()

        if input_path.exists():
            input_path.unlink()


# ============================================================
# 8. RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
