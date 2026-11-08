
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
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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


# =========================================================
# APPLICATION STATE
# =========================================================

processor = None
model_loaded = False
model_error = None


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

    logger.info("Model path: %s", MODEL_PATH)

    try:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {MODEL_PATH}"
            )

        logger.info("Loading YOLO11 TFLite model...")

        processor = await run_in_threadpool(
            VideoProcessor,
            str(MODEL_PATH),
        )

        model_loaded = True
        model_error = None

        app.state.processor = processor

        logger.info(
            "YOLO11 model loaded successfully."
        )

    except Exception as exc:

        processor = None
        model_loaded = False
        model_error = str(exc)

        app.state.processor = None

        logger.exception(
            "Failed to initialize YOLO11 model."
        )

    logger.info("Backend startup completed.")

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
            "/api/process",
            "/api/outputs",
        ],
    }


# =========================================================
# VIDEO PROCESSING ENDPOINT
# =========================================================

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    frame_interval: int = Form(1),
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

    input_path = None

    try:

        # -------------------------------------------------
        # 1. CHECK MODEL AVAILABILITY
        # -------------------------------------------------

        current_processor = get_current_processor()

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

        # -------------------------------------------------
        # 3. VALIDATE FILE EXTENSION
        # -------------------------------------------------

        original_filename = (
            file.filename or "uploaded_video.mp4"
        )

        extension = Path(
            original_filename
        ).suffix.lower()

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

        # -------------------------------------------------
        # 5. SAVE UPLOADED VIDEO
        # -------------------------------------------------

        logger.info(
            "Receiving video: %s",
            original_filename,
        )

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
                        detail=(
                            "Video exceeds the maximum "
                            "upload size of 500 MB."
                        ),
                    )

                destination.write(chunk)

        if uploaded_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty.",
            )

        logger.info(
            "Video uploaded successfully. Size: %.2f MB",
            uploaded_size / (1024 * 1024),
        )

        # -------------------------------------------------
        # 6. VALIDATE VIDEO USING OPENCV
        # -------------------------------------------------

        try:

            video_info = await run_in_threadpool(
                validate_video,
                str(input_path),
            )

        except VideoValidationError as exc:

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        logger.info(
            "Video validation successful: %s",
            video_info,
        )

        # -------------------------------------------------
        # 7. PROCESS VIDEO
        # -------------------------------------------------

        logger.info(
            "Starting YOLO11 + ByteTrack processing."
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
            "Video processing completed."
        )

        # -------------------------------------------------
        # 8. VERIFY OUTPUT FILES
        # -------------------------------------------------

        if not output_path.exists():

            raise RuntimeError(
                "Video processor did not generate output video."
            )

        if output_path.stat().st_size == 0:

            raise RuntimeError(
                "Generated output video is empty."
            )

        if not csv_path.exists():

            logger.warning(
                "CSV analytics file was not generated."
            )

        # -------------------------------------------------
        # 9. BUILD ANALYTICS JSON
        # -------------------------------------------------

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

        logger.info(
            "Job %s completed successfully.",
            job_id,
        )

        return response

    except HTTPException:
        raise

    except VideoValidationError as exc:

        logger.exception(
            "Video validation failed."
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        logger.exception(
            "Video processing failed."
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Video processing failed.",
                "error": str(exc),
            },
        )

    finally:

        # Delete temporary uploaded input.
        cleanup_file(input_path)

        await file.close()


# =========================================================
# LIST GENERATED OUTPUT FILES
# =========================================================

@app.get("/api/outputs")
async def list_outputs():

    files = []

    for path in OUTPUT_DIR.iterdir():

        if path.is_file():

            files.append({
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "url": get_public_url(path),
            })

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

        target_path.unlink()

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
