
"""
PragyanAI - People Detection & Counting

Backend:
    FastAPI
    YOLO11 TFLite
    ByteTrack
    OpenCV

Features:
    1. Video upload and validation
    2. YOLO11 TFLite model initialization
    3. People detection and tracking
    4. Video analytics generation
    5. Processed MP4 video output
    6. CSV and JSON report generation
    7. Health and model status endpoints
"""

import os
import json
import uuid
import logging
import time
import threading
from copy import deepcopy
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from services.video_processor import VideoProcessor

from utils.video_utils import (
    ALLOWED_EXTENSIONS,
    VideoValidationError,
    validate_video,
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("pragyanai")


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "yolo11n.tflite"),
    )
)

if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH

UPLOAD_DIR = Path(
    os.getenv(
        "UPLOAD_DIR",
        "/tmp/pragyanai_uploads",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        "/tmp/pragyanai_outputs",
    )
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "https://tflite-objectdetection-peopleanalysis.netlify.app",
).rstrip("/")

PUBLIC_API_URL = os.getenv(
    "PUBLIC_API_URL",
    "https://pragyanai-yolo-tf-lite-object-detction.onrender.com",
).rstrip("/")

MAX_UPLOAD_SIZE = int(
    os.getenv(
        "MAX_UPLOAD_SIZE",
        str(500 * 1024 * 1024),
    )
)

CHUNK_SIZE = 1024 * 1024

# Resource-conservative defaults for the Render instance.
DEFAULT_FRAME_INTERVAL = max(1, int(os.getenv("DEFAULT_FRAME_INTERVAL", "2")))
MAX_CONCURRENT_JOBS = 1
PROCESSING_LOCK = threading.Lock()


# =========================================================
# APPLICATION STATE
# =========================================================

processor = None
model_loaded = False
model_error = None

# In-memory job registry. Jobs survive browser disconnects, but not Render restarts.
jobs = {}
jobs_lock = threading.Lock()


# =========================================================
# APPLICATION LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the YOLO model when the application starts.
    """

    global processor
    global model_loaded
    global model_error

    logger.info("=" * 60)
    logger.info("Starting PragyanAI People Detection Backend")
    logger.info("=" * 60)

    startup_started = time.perf_counter()

    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Upload directory: %s", UPLOAD_DIR)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Frontend origin: %s", FRONTEND_ORIGIN)
    logger.info("Maximum upload size: %s bytes", MAX_UPLOAD_SIZE)
    logger.info("Default frame interval: %s", DEFAULT_FRAME_INTERVAL)
    logger.info("Maximum concurrent video jobs: %s", MAX_CONCURRENT_JOBS)
    logger.info("YOLO inference image size: 320 (configured in VideoProcessor)")
    logger.info("Python process PID: %s", os.getpid())
    logger.info("Model file exists: %s", MODEL_PATH.is_file())

    try:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {MODEL_PATH}"
            )

        logger.info("[STARTUP 1/2] Loading YOLO11 TFLite model...")
        model_load_started = time.perf_counter()

        processor = await run_in_threadpool(
            VideoProcessor,
            str(MODEL_PATH),
        )

        model_loaded = True
        model_error = None

        app.state.processor = processor

        logger.info(
            "[STARTUP 2/2] YOLO11 model loaded successfully in %.2f seconds.",
            time.perf_counter() - model_load_started,
        )

    except Exception as exc:

        processor = None
        model_loaded = False
        model_error = str(exc)

        app.state.processor = None

        logger.exception(
            "Failed to initialize YOLO11 model."
        )

    logger.info(
        "Backend startup completed in %.2f seconds | model_loaded=%s",
        time.perf_counter() - startup_started,
        model_loaded,
    )

    yield

    logger.info("Shutting down PragyanAI backend.")

    processor = None
    model_loaded = False


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="PragyanAI People Detection & Counting API",
    description=(
        "YOLO11 TFLite based people detection, "
        "ByteTrack tracking and video analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

allowed_origins = [
    FRONTEND_ORIGIN,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HTTP REQUEST LOGGING
# =========================================================

@app.middleware("http")
async def request_logging_middleware(request, call_next):
    request_started = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"

    logger.info(
        "[HTTP REQUEST] method=%s path=%s client=%s",
        request.method,
        request.url.path,
        client_host,
    )

    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - request_started

        logger.info(
            "[HTTP RESPONSE] method=%s path=%s status=%s duration=%.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    except Exception:
        elapsed = time.perf_counter() - request_started
        logger.exception(
            "[HTTP ERROR] method=%s path=%s duration=%.3fs",
            request.method,
            request.url.path,
            elapsed,
        )
        raise


# =========================================================
# STATIC OUTPUT DIRECTORY
# =========================================================

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_public_url(file_path):
    """
    Convert an output file path into a public URL.
    """

    filename = Path(file_path).name

    return (
        f"{PUBLIC_API_URL}/outputs/{filename}"
    )


def get_current_processor():
    """
    Return the initialized video processor.
    """

    current_processor = getattr(
        app.state,
        "processor",
        None,
    )

    if current_processor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "YOLO11 model is not ready.",
                "model_loaded": False,
                "model_error": model_error,
            },
        )

    return current_processor


def cleanup_file(file_path):
    """
    Safely delete a temporary file.
    """

    if not file_path:
        return

    try:

        path = Path(file_path)

        if path.exists() and path.is_file():
            path.unlink()

    except Exception:
        logger.exception(
            "Unable to delete temporary file: %s",
            file_path,
        )


def safe_filename(filename):
    """
    Generate a safe filename.
    """

    original_name = Path(
        filename or "uploaded_video.mp4"
    ).name

    extension = Path(
        original_name
    ).suffix.lower()

    return f"{uuid.uuid4().hex}{extension}"


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
async def root():

    return {
        "application": "PragyanAI People Detection & Counting",
        "status": "running",
        "version": "1.0.0",
        "backend": "FastAPI",
        "detection_engine": "YOLO11 TFLite",
        "tracking_engine": "ByteTrack",
        "video_processing": "OpenCV",
        "model_loaded": model_loaded,
        "model_error": model_error,
        "docs": "/docs",
        "health": "/health",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health_check():

    status = (
        "healthy"
        if model_loaded
        else "degraded"
    )

    return {
        "status": status,
        "backend_connected": True,
        "model_loaded": model_loaded,
        "model_status": (
            "ready"
            if model_loaded
            else "not_ready"
        ),
        "model_name": "YOLO11 TFLite",
        "tracking_engine": "ByteTrack",
        "video_processing": "OpenCV",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_error": model_error,
        "processing_busy": PROCESSING_LOCK.locked(),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "default_frame_interval": DEFAULT_FRAME_INTERVAL,
        "default_imgsz": 320,
    }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/api/model-info")
async def model_info():

    return {
        "model_name": "YOLO11n TFLite",
        "model_type": "Object Detection",
        "model_format": "TensorFlow Lite",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": model_loaded,
        "model_status": (
            "ready"
            if model_loaded
            else "not_ready"
        ),
        "model_error": model_error,
        "classes": [
            "person"
        ],
        "class_ids": [
            0
        ],
        "tracking_engine": "ByteTrack",
        "video_processing": "OpenCV",
    }


# =========================================================
# API CONFIGURATION
# =========================================================

@app.get("/api")
async def api_root():

    return {
        "message": "PragyanAI People Detection API",
        "available_endpoints": [
            "/",
            "/health",
            "/api/model-info",
            "/api/jobs",
            "/api/jobs/{job_id}",
            "/api/process",
            "/api/outputs",
        ],
    }



# =========================================================
# ASYNCHRONOUS VIDEO JOB HELPERS
# =========================================================

def update_job(job_id, **changes):
    """Thread-safe update of a job record."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(changes)
            jobs[job_id]["updated_at"] = time.time()


def get_job_snapshot(job_id):
    """Return a copy of a job record."""
    with jobs_lock:
        job = jobs.get(job_id)
        return deepcopy(job) if job else None


async def process_job_background(
    job_id,
    input_path,
    output_path,
    csv_path,
    json_path,
    confidence,
    frame_interval,
    video_info,
    current_processor,
):
    """
    Run the long video-processing operation after the upload request
    has returned. The browser can poll the job endpoint independently.
    """
    started = time.perf_counter()
    update_job(
        job_id,
        status="processing",
        progress=10,
        message="Video accepted. YOLO11 processing has started.",
    )

    logger.info("[ASYNC JOB START] job_id=%s", job_id)

    try:
        def publish_progress(
            percent,
            current_frame,
            total_frames,
            eta_seconds,
            frame_update=None,
        ):
            """Receive progress from the processor thread and publish it safely."""
            overall_progress = 10 + (float(percent) * 0.85)

            # Keep the frame payload small and JSON-safe for the polling API.
            safe_frame_update = None
            if isinstance(frame_update, dict):
                safe_frame_update = {
                    "event": str(frame_update.get("event", "frame_processed")),
                    "frame_number": int(frame_update.get("frame_number", current_frame) or 0),
                    "total_frames": int(frame_update.get("total_frames", total_frames) or 0),
                    "progress": round(float(frame_update.get("progress", percent) or 0), 2),
                    "timestamp_seconds": round(float(frame_update.get("timestamp_seconds", 0) or 0), 3),
                    "status": str(frame_update.get("status", "UNKNOWN")),
                    "objects_found": int(frame_update.get("objects_found", 0) or 0),
                    "people_found": int(frame_update.get("people_found", 0) or 0),
                    "track_ids": [int(value) for value in (frame_update.get("track_ids") or [])],
                    "inference_performed": bool(frame_update.get("inference_performed", False)),
                    "inference_seconds": round(float(frame_update.get("inference_seconds", 0) or 0), 4),
                    "frame_seconds": round(float(frame_update.get("frame_seconds", 0) or 0), 4),
                    "unique_people_so_far": int(frame_update.get("unique_people_so_far", 0) or 0),
                    "eta_seconds": round(float(frame_update.get("eta_seconds", eta_seconds) or 0), 1),
                }

            changes = {
                "progress": min(95, round(overall_progress, 1)),
                "current_frame": int(current_frame),
                "total_frames": int(total_frames),
                "eta_seconds": round(float(eta_seconds), 1),
                "message": f"Processing frame {current_frame} of {total_frames}.",
            }

            if safe_frame_update is not None:
                changes["latest_frame"] = safe_frame_update
                changes["current_people"] = safe_frame_update["people_found"]
                changes["current_objects"] = safe_frame_update["objects_found"]
                changes["unique_people_so_far"] = safe_frame_update["unique_people_so_far"]

                # Retain only the latest 20 updates to keep polling responses small.
                with jobs_lock:
                    current_job = jobs.get(job_id)
                    if current_job is not None:
                        activity = current_job.setdefault("recent_activity", [])
                        activity.append(safe_frame_update)
                        current_job["recent_activity"] = activity[-20:]
                        current_job.update(changes)
                        current_job["updated_at"] = time.time()
            else:
                update_job(job_id, **changes)

        analytics_result = await run_in_threadpool(
            current_processor.process,
            str(input_path),
            str(output_path),
            confidence,
            frame_interval,
            str(csv_path),
            publish_progress,
        )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Video processor did not generate a valid output video.")

        if not isinstance(analytics_result, dict):
            analytics_result = {"result": analytics_result}

        analytics_result["job_id"] = job_id
        analytics_result["video_info"] = video_info
        analytics_result["processing_settings"] = {
            "confidence": confidence,
            "frame_interval": frame_interval,
            "model": "YOLO11 TFLite",
            "tracker": "ByteTrack",
        }

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(analytics_result, json_file, indent=4, default=str)

        result = {
            "success": True,
            "message": "People detection and tracking completed successfully.",
            "job_id": job_id,
            "video_url": get_public_url(output_path),
            "csv_url": get_public_url(csv_path) if csv_path.exists() else None,
            "json_url": get_public_url(json_path),
            "analytics": analytics_result,
            "summary": analytics_result,
            "video_info": video_info,
            "processing_settings": {
                "confidence": confidence,
                "frame_interval": frame_interval,
                "detection_engine": "YOLO11 TFLite",
                "tracking_engine": "ByteTrack",
            },
        }

        update_job(
            job_id,
            status="completed",
            progress=100,
            message="Video processing completed successfully.",
            completed_at=time.time(),
            processing_seconds=round(time.perf_counter() - started, 2),
            result=result,
            error=None,
        )

        logger.info(
            "[ASYNC JOB COMPLETE] job_id=%s elapsed=%.2fs output_size=%.2fMB",
            job_id,
            time.perf_counter() - started,
            output_path.stat().st_size / (1024 * 1024),
        )

    except Exception as exc:
        logger.exception("[ASYNC JOB FAILED] job_id=%s error=%s", job_id, exc)
        update_job(
            job_id,
            status="failed",
            progress=0,
            message="Video processing failed.",
            error=str(exc),
            failed_at=time.time(),
        )

    finally:
        cleanup_file(input_path)
        if PROCESSING_LOCK.locked():
            PROCESSING_LOCK.release()
            logger.info("[PROCESSING SLOT RELEASED] job_id=%s", job_id)


# =========================================================
# ASYNCHRONOUS VIDEO JOB API
# =========================================================

@app.post("/api/jobs", status_code=202)
async def create_video_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    frame_interval: int = Form(DEFAULT_FRAME_INTERVAL),
):
    """
    Upload and validate a video, create a job, and return immediately.
    Processing continues independently of the browser request.
    """
    request_started = time.perf_counter()
    input_path = None
    processing_slot_reserved = False

    try:
        current_processor = get_current_processor()

        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1.")

        if frame_interval < 1:
            raise HTTPException(status_code=400, detail="Frame interval must be at least 1.")

        original_filename = file.filename or "uploaded_video.mp4"
        extension = Path(original_filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported video format. Supported formats: MP4, AVI, MOV, MKV, WEBM.",
            )

        # Reserve the single processing slot before accepting a large upload.
        # This prevents multiple uploads from competing for disk and memory.
        if not PROCESSING_LOCK.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Another video is currently uploading or processing. Please wait until it finishes.",
            )
        processing_slot_reserved = True

        job_id = uuid.uuid4().hex
        input_path = UPLOAD_DIR / safe_filename(original_filename)
        output_path = OUTPUT_DIR / f"{job_id}_processed.mp4"
        csv_path = OUTPUT_DIR / f"{job_id}_analytics.csv"
        json_path = OUTPUT_DIR / f"{job_id}_analytics.json"

        uploaded_size = 0
        with open(input_path, "wb") as destination:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                uploaded_size += len(chunk)
                if uploaded_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the maximum upload size of 500 MB.",
                    )
                destination.write(chunk)

        if uploaded_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty.")

        try:
            video_info = await run_in_threadpool(validate_video, str(input_path))
        except VideoValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        now = time.time()
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 5,
                "message": "Video uploaded and validated. Waiting for processing.",
                "filename": Path(original_filename).name,
                "file_size_bytes": uploaded_size,
                "video_info": video_info,
                "confidence": confidence,
                "frame_interval": frame_interval,
                "created_at": now,
                "updated_at": now,
                "result": None,
                "error": None,
                "latest_frame": None,
                "recent_activity": [],
                "current_people": 0,
                "current_objects": 0,
                "unique_people_so_far": 0,
            }

        background_tasks.add_task(
            process_job_background,
            job_id,
            input_path,
            output_path,
            csv_path,
            json_path,
            confidence,
            frame_interval,
            video_info,
            current_processor,
        )

        logger.info(
            "[ASYNC JOB CREATED] job_id=%s filename=%s size=%.2fMB upload_and_validation=%.2fs",
            job_id,
            original_filename,
            uploaded_size / (1024 * 1024),
            time.perf_counter() - request_started,
        )

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "progress": 5,
            "message": "Video uploaded successfully. Processing has been queued.",
            "status_url": f"/api/jobs/{job_id}",
        }

    except HTTPException:
        cleanup_file(input_path)
        if processing_slot_reserved and PROCESSING_LOCK.locked():
            PROCESSING_LOCK.release()
        raise
    except Exception as exc:
        cleanup_file(input_path)
        if processing_slot_reserved and PROCESSING_LOCK.locked():
            PROCESSING_LOCK.release()
        logger.exception("[ASYNC JOB CREATE FAILED] error=%s", exc)
        raise HTTPException(
            status_code=500,
            detail={"message": "Unable to create video-processing job.", "error": str(exc)},
        )
    finally:
        await file.close()


@app.get("/api/jobs/{job_id}")
async def get_video_job_status(job_id: str):
    """Return job status, live per-frame updates, and completed result when available."""
    job = get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "success": True,
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "filename": job.get("filename"),
        "video_info": job.get("video_info"),
        "current_frame": job.get("current_frame"),
        "total_frames": job.get("total_frames"),
        "eta_seconds": job.get("eta_seconds"),
        "latest_frame": job.get("latest_frame"),
        "recent_activity": job.get("recent_activity", []),
        "current_people": job.get("current_people", 0),
        "current_objects": job.get("current_objects", 0),
        "unique_people_so_far": job.get("unique_people_so_far", 0),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "processing_seconds": job.get("processing_seconds"),
        "error": job.get("error"),
    }

    if job["status"] == "completed":
        response["result"] = job.get("result")

    return response


@app.get("/api/jobs/{job_id}/analytics")
async def get_video_job_analytics(job_id: str):
    """Return analytics for a completed job."""
    job = get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job['status']}; analytics are not ready.",
        )
    return job["result"]["analytics"]


# =========================================================
# VIDEO PROCESSING ENDPOINT
# =========================================================

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    frame_interval: int = Form(DEFAULT_FRAME_INTERVAL),
):
    """
    Upload a video and process it using YOLO11 TFLite
    and ByteTrack.

    Request:
        file: Uploaded video
        confidence: Detection confidence threshold
        frame_interval: Process every Nth frame

    Response:
        Processed video URL
        CSV analytics URL
        JSON analytics URL
        Analytics summary
    """

    request_started = time.perf_counter()
    input_path = None
    job_id = None
    current_stage = "request_received"
    uploaded_size = 0
    processing_slot_reserved = False

    logger.info("=" * 72)
    logger.info(
        "[PROCESS REQUEST] filename=%s confidence=%.3f frame_interval=%s",
        file.filename,
        confidence,
        frame_interval,
    )

    try:

        # -------------------------------------------------
        # 1. CHECK MODEL AVAILABILITY
        # -------------------------------------------------

        current_stage = "check_model"
        logger.info("[JOB STAGE] Checking model availability")

        current_processor = get_current_processor()

        logger.info(
            "[MODEL STATUS] Model is ready | processor=%s",
            type(current_processor).__name__,
        )

        # -------------------------------------------------
        # 2. VALIDATE REQUEST PARAMETERS
        # -------------------------------------------------

        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Confidence must be between 0 and 1."
                ),
            )

        if frame_interval < 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Frame interval must be at least 1."
                ),
            )

        logger.info(
            "[REQUEST SETTINGS] confidence=%.3f frame_interval=%s",
            confidence,
            frame_interval,
        )

        # -------------------------------------------------
        # 3. VALIDATE FILE EXTENSION
        # -------------------------------------------------

        original_filename = (
            file.filename or "uploaded_video.mp4"
        )

        extension = Path(
            original_filename
        ).suffix.lower()

        current_stage = "validate_file_extension"
        logger.info(
            "[FILE VALIDATION] filename=%s extension=%s",
            original_filename,
            extension,
        )

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported video format. "
                    "Supported formats: MP4, AVI, MOV, MKV, WEBM."
                ),
            )

        # -------------------------------------------------
        # 4. GENERATE UNIQUE JOB ID
        # -------------------------------------------------

        job_id = uuid.uuid4().hex

        input_filename = safe_filename(
            original_filename
        )

        input_path = UPLOAD_DIR / input_filename

        output_filename = (
            f"{job_id}_processed.mp4"
        )

        csv_filename = (
            f"{job_id}_analytics.csv"
        )

        json_filename = (
            f"{job_id}_analytics.json"
        )

        output_path = OUTPUT_DIR / output_filename
        csv_path = OUTPUT_DIR / csv_filename
        json_path = OUTPUT_DIR / json_filename

        logger.info(
            "[JOB CREATED] job_id=%s input=%s output=%s",
            job_id,
            input_path.name,
            output_path.name,
        )

        # -------------------------------------------------
        # 5. SAVE UPLOADED VIDEO
        # -------------------------------------------------

        current_stage = "upload_video"
        upload_started = time.perf_counter()

        logger.info(
            "[UPLOAD START] job_id=%s filename=%s",
            job_id,
            original_filename,
        )

        last_upload_log_size = 0

        with open(input_path, "wb") as destination:

            while True:

                chunk = await file.read(CHUNK_SIZE)

                if not chunk:
                    break

                uploaded_size += len(chunk)

                if uploaded_size > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video exceeds the maximum "
                            "upload size of 500 MB."
                        ),
                    )

                destination.write(chunk)

                # Emit upload progress at approximately every 10 MB.
                if uploaded_size - last_upload_log_size >= 10 * 1024 * 1024:
                    logger.info(
                        "[UPLOAD PROGRESS] job_id=%s uploaded=%.2f MB",
                        job_id,
                        uploaded_size / (1024 * 1024),
                    )
                    last_upload_log_size = uploaded_size

        if uploaded_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty.",
            )

        logger.info(
            "[UPLOAD COMPLETE] job_id=%s size=%.2f MB duration=%.2fs",
            job_id,
            uploaded_size / (1024 * 1024),
            time.perf_counter() - upload_started,
        )

        # -------------------------------------------------
        # 6. VALIDATE VIDEO USING OPENCV
        # -------------------------------------------------

        current_stage = "validate_video"
        validation_started = time.perf_counter()
        logger.info("[VALIDATION START] job_id=%s", job_id)

        try:

            video_info = await run_in_threadpool(
                validate_video,
                str(input_path),
            )

        except VideoValidationError as exc:

            logger.warning(
                "[VALIDATION FAILED] job_id=%s error=%s",
                job_id,
                exc,
            )
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        logger.info(
            "[VALIDATION COMPLETE] job_id=%s metadata=%s duration=%.2fs",
            job_id,
            video_info,
            time.perf_counter() - validation_started,
        )

        # -------------------------------------------------
        # 7. PROCESS VIDEO
        # -------------------------------------------------

        current_stage = "yolo_bytetrack_processing"

        if not PROCESSING_LOCK.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Another video is currently processing. Please wait until it finishes.",
            )
        processing_slot_reserved = True

        processing_started = time.perf_counter()

        logger.info(
            "[INFERENCE START] job_id=%s engine=YOLO11 runtime=LiteRT tracker=ByteTrack",
            job_id,
        )
        logger.info(
            "[INFERENCE INPUT] video=%s confidence=%.3f frame_interval=%s",
            input_path.name,
            confidence,
            frame_interval,
        )

        analytics_result = await run_in_threadpool(
            current_processor.process,
            str(input_path),
            str(output_path),
            confidence,
            frame_interval,
            str(csv_path),
        )

        logger.info(
            "[INFERENCE COMPLETE] job_id=%s duration=%.2fs analytics_type=%s",
            job_id,
            time.perf_counter() - processing_started,
            type(analytics_result).__name__,
        )

        # -------------------------------------------------
        # 8. VERIFY OUTPUT FILES
        # -------------------------------------------------

        current_stage = "verify_output_files"
        logger.info("[OUTPUT VERIFY] job_id=%s checking generated files", job_id)

        if not output_path.exists():

            raise RuntimeError(
                "Video processor did not generate output video."
            )

        output_size = output_path.stat().st_size
        logger.info(
            "[OUTPUT VIDEO] job_id=%s path=%s size=%.2f MB",
            job_id,
            output_path.name,
            output_size / (1024 * 1024),
        )

        if output_path.stat().st_size == 0:

            raise RuntimeError(
                "Generated output video is empty."
            )

        if not csv_path.exists():

            logger.warning(
                "[OUTPUT WARNING] job_id=%s CSV analytics file was not generated",
                job_id,
            )
        else:
            logger.info(
                "[OUTPUT CSV] job_id=%s path=%s size=%s bytes",
                job_id,
                csv_path.name,
                csv_path.stat().st_size,
            )

        # -------------------------------------------------
        # 9. BUILD ANALYTICS JSON
        # -------------------------------------------------

        current_stage = "generate_analytics_json"

        if not isinstance(analytics_result, dict):

            analytics_result = {
                "result": analytics_result
            }

        analytics_result["job_id"] = job_id

        analytics_result["video_info"] = video_info

        analytics_result["processing_settings"] = {
            "confidence": confidence,
            "frame_interval": frame_interval,
            "model": "YOLO11 TFLite",
            "tracker": "ByteTrack",
        }

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as json_file:

            json.dump(
                analytics_result,
                json_file,
                indent=4,
                default=str,
            )

        logger.info(
            "[OUTPUT JSON] job_id=%s path=%s size=%s bytes",
            job_id,
            json_path.name,
            json_path.stat().st_size,
        )

        # -------------------------------------------------
        # 10. GENERATE PUBLIC URLS
        # -------------------------------------------------

        video_url = get_public_url(
            output_path
        )

        csv_url = (
            get_public_url(csv_path)
            if csv_path.exists()
            else None
        )

        json_url = get_public_url(
            json_path
        )

        # -------------------------------------------------
        # 11. BUILD API RESPONSE
        # -------------------------------------------------

        response = {
            "success": True,
            "message": (
                "People detection and tracking completed successfully."
            ),
            "job_id": job_id,

            "video_url": video_url,
            "csv_url": csv_url,
            "json_url": json_url,

            "analytics": analytics_result,
            "summary": analytics_result,

            "video_info": video_info,

            "processing_settings": {
                "confidence": confidence,
                "frame_interval": frame_interval,
                "detection_engine": "YOLO11 TFLite",
                "tracking_engine": "ByteTrack",
            },
        }

        total_duration = time.perf_counter() - request_started

        logger.info(
            "[JOB SUCCESS] job_id=%s total_duration=%.2fs "
            "video_size=%.2f MB csv_created=%s json_created=%s",
            job_id,
            total_duration,
            output_path.stat().st_size / (1024 * 1024),
            csv_path.exists(),
            json_path.exists(),
        )
        logger.info(
            "[JOB RESPONSE] job_id=%s video_url=%s csv_url=%s json_url=%s",
            job_id,
            video_url,
            csv_url,
            json_url,
        )
        logger.info(
            "[JOB ANALYTICS SUMMARY] job_id=%s unique_people=%s "
            "total_detections=%s processed_frames=%s processing_time=%s",
            job_id,
            analytics_result.get("unique_people",
                                 analytics_result.get("total_unique_people")),
            analytics_result.get("total_detections"),
            analytics_result.get("processed_frames"),
            analytics_result.get("processing_time"),
        )
        logger.info("=" * 72)

        return response

    except HTTPException as exc:
        logger.warning(
            "[JOB HTTP ERROR] job_id=%s stage=%s status=%s detail=%s "
            "elapsed=%.2fs",
            job_id,
            current_stage,
            exc.status_code,
            exc.detail,
            time.perf_counter() - request_started,
        )
        raise

    except VideoValidationError as exc:

        logger.exception(
            "[JOB VALIDATION ERROR] job_id=%s stage=%s",
            job_id,
            current_stage,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        logger.exception(
            "[JOB FAILED] job_id=%s stage=%s elapsed=%.2fs error=%s",
            job_id,
            current_stage,
            time.perf_counter() - request_started,
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Video processing failed.",
                "error": str(exc),
            },
        )

    finally:

        if processing_slot_reserved and PROCESSING_LOCK.locked():
            PROCESSING_LOCK.release()
            logger.info("[PROCESSING SLOT RELEASED] job_id=%s", job_id)

        # Delete temporary uploaded input.
        cleanup_file(input_path)

        try:
            await file.close()
        except Exception:
            logger.exception(
                "[CLEANUP ERROR] job_id=%s unable to close upload",
                job_id,
            )

        logger.info(
            "[JOB CLEANUP] job_id=%s input_removed=%s total_elapsed=%.2fs",
            job_id,
            not (input_path and input_path.exists()),
            time.perf_counter() - request_started,
        )


# =========================================================
# LIST GENERATED OUTPUT FILES
# =========================================================

@app.get("/api/outputs")
async def list_outputs():

    logger.info("[OUTPUT LIST] Listing files in %s", OUTPUT_DIR)
    files = []

    for path in OUTPUT_DIR.iterdir():

        if path.is_file():

            files.append({
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "url": get_public_url(path),
            })

    logger.info("[OUTPUT LIST COMPLETE] total_files=%s", len(files))

    return {
        "success": True,
        "total_files": len(files),
        "files": files,
    }


# =========================================================
# DELETE GENERATED OUTPUT
# =========================================================

@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):
    """
    Delete an output file by filename.
    """

    safe_name = Path(filename).name

    if safe_name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    target_path = OUTPUT_DIR / safe_name

    if not target_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Output file not found.",
        )

    if not target_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Invalid output file.",
        )

    try:

        logger.info("[OUTPUT DELETE] filename=%s", safe_name)
        target_path.unlink()
        logger.info("[OUTPUT DELETE COMPLETE] filename=%s", safe_name)

        return {
            "success": True,
            "message": "Output file deleted.",
            "filename": safe_name,
        }

    except Exception as exc:

        logger.exception(
            "Failed to delete output file."
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# =========================================================
# RUN LOCALLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
    
