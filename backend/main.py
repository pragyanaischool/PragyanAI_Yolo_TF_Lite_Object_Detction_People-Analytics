import os
import uuid
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import cv2

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from services.video_processor import VideoProcessor


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("PragyanAI")


# ============================================================
# 2. ENVIRONMENT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "yolo11n.tflite"),
    )
)

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

MAX_UPLOAD_SIZE = int(
    os.getenv(
        "MAX_UPLOAD_SIZE",
        str(500 * 1024 * 1024),
    )
)

PUBLIC_API_URL = os.getenv(
    "PUBLIC_API_URL",
    "",
).rstrip("/")

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "",
)

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

# Create required directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. CORS CONFIGURATION
# ============================================================

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if FRONTEND_ORIGIN:
    origins.extend(
        origin.strip().rstrip("/")
        for origin in FRONTEND_ORIGIN.split(",")
        if origin.strip()
    )

# Remove duplicate origins
origins = list(dict.fromkeys(origins))

logger.info("Allowed CORS origins: %s", origins)


# ============================================================
# 4. GLOBAL PROCESSOR
# ============================================================

video_processor = None
model_error = None


# ============================================================
# 5. APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_processor
    global model_error

    logger.info("Starting PragyanAI People Detection API")

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO TFLite model not found: {MODEL_PATH}"
            )

        logger.info("Loading model: %s", MODEL_PATH)

        # VideoProcessor must support model_path.
        video_processor = VideoProcessor(
            model_path=str(MODEL_PATH)
        )

        model_error = None

        logger.info("Detection engine initialized successfully")

    except Exception as error:
        video_processor = None
        model_error = str(error)

        logger.exception(
            "Failed to initialize detection engine"
        )

    yield

    logger.info("Shutting down PragyanAI API")

    video_processor = None


# ============================================================
# 6. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PragyanAI People Detection & Counting API",
    description=(
        "YOLO11 TFLite based people detection, "
        "ByteTrack tracking and video analytics API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 7. ADD CORS MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 8. STATIC OUTPUT DIRECTORY
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

def get_public_url(relative_path: str) -> str:
    """
    Convert a relative output path into a public URL.
    """

    if PUBLIC_API_URL:
        return f"{PUBLIC_API_URL}{relative_path}"

    return relative_path


def validate_video_file(filename: str):
    """
    Validate uploaded video filename and extension.
    """

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="No video filename received.",
        )

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video format: {extension}. "
                "Supported formats: MP4, AVI, MOV, MKV, WEBM."
            ),
        )

    return extension


def validate_video_content(video_path: Path):
    """
    Check whether OpenCV can open the uploaded video.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid video file. OpenCV could not open "
                "the uploaded video."
            ),
        )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if frame_count <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid video: no readable frames found.",
        )

    if fps <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid video: unable to determine FPS.",
        )

    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid video dimensions.",
        )

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
    }


async def save_uploaded_video(
    file: UploadFile,
    destination: Path,
):
    """
    Save uploaded video in chunks.
    Prevent uploads exceeding MAX_UPLOAD_SIZE.
    """

    total_size = 0

    try:
        with destination.open("wb") as buffer:

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video exceeds the maximum "
                            "upload size of 500 MB."
                        ),
                    )

                buffer.write(chunk)

    except Exception:
        if destination.exists():
            destination.unlink(missing_ok=True)

        raise

    if total_size == 0:
        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty.",
        )

    return total_size


def normalize_analytics(result):
    """
    Normalize analytics returned by VideoProcessor.
    """

    if result is None:
        return {}

    if isinstance(result, dict):
        return result

    if hasattr(result, "to_dict"):
        return result.to_dict()

    if hasattr(result, "__dict__"):
        return result.__dict__

    return {"result": str(result)}


# ============================================================
# 10. ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():

    return {
        "application": "PragyanAI People Detection & Counting",
        "status": "running",
        "version": "1.0.0",
        "framework": "FastAPI",
        "detection_engine": "YOLO11",
        "tracking_engine": "ByteTrack",
        "inference_runtime": "TensorFlow Lite",
        "documentation": "/docs",
        "health": "/health",
    }


# ============================================================
# 11. HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health")
async def health_check():

    model_loaded = video_processor is not None

    return {
        "status": (
            "healthy" if model_loaded else "degraded"
        ),
        "api": "running",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_error": model_error,
        "detection_engine": "YOLO11",
        "tracking_engine": "ByteTrack",
        "inference_runtime": "TensorFlow Lite",
    }


# ============================================================
# 12. MODEL INFORMATION ENDPOINT
# ============================================================

@app.get("/api/model-info")
async def model_info():

    return {
        "model_name": "YOLO11 Nano",
        "model_file": MODEL_PATH.name,
        "model_format": "TensorFlow Lite",
        "model_loaded": video_processor is not None,
        "detection_classes": ["person"],
        "tracking_algorithm": "ByteTrack",
        "confidence_default": 0.35,
    }


# ============================================================
# 13. VIDEO PROCESSING ENDPOINT
# ============================================================

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    frame_interval: int = Form(1),
):
    """
    Upload video and process it using YOLO11 TFLite
    and ByteTrack.

    Form fields:
        file
        confidence
        frame_interval

    Returns:
        Processed video URL
        CSV report URL
        Analytics summary
    """

    input_path = None
    output_path = None
    csv_path = None

    try:

        # ----------------------------------------------------
        # A. CHECK MODEL
        # ----------------------------------------------------

        if video_processor is None:

            raise HTTPException(
                status_code=503,
                detail=(
                    "Detection engine is not initialized. "
                    f"Model error: {model_error}"
                ),
            )

        # ----------------------------------------------------
        # B. VALIDATE PARAMETERS
        # ----------------------------------------------------

        if not 0.0 <= confidence <= 1.0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Confidence must be between 0.0 and 1.0."
                ),
            )

        if frame_interval < 1 or frame_interval > 100:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Frame interval must be between 1 and 100."
                ),
            )

        # ----------------------------------------------------
        # C. VALIDATE UPLOADED FILE
        # ----------------------------------------------------

        extension = validate_video_file(file.filename)

        unique_id = uuid.uuid4().hex

        input_path = UPLOAD_DIR / (
            f"{unique_id}{extension}"
        )

        output_path = OUTPUT_DIR / (
            f"{unique_id}_processed.mp4"
        )

        csv_path = OUTPUT_DIR / (
            f"{unique_id}_analytics.csv"
        )

        logger.info(
            "Received video: %s",
            file.filename,
        )

        # ----------------------------------------------------
        # D. SAVE UPLOADED FILE
        # ----------------------------------------------------

        file_size = await save_uploaded_video(
            file,
            input_path,
        )

        logger.info(
            "Uploaded file size: %.2f MB",
            file_size / (1024 * 1024),
        )

        # ----------------------------------------------------
        # E. VALIDATE VIDEO CONTENT
        # ----------------------------------------------------

        video_info = await run_in_threadpool(
            validate_video_content,
            input_path,
        )

        logger.info(
            "Video information: %s",
            video_info,
        )

        # ----------------------------------------------------
        # F. PROCESS VIDEO
        # ----------------------------------------------------

        logger.info(
            "Starting YOLO11 detection and ByteTrack tracking"
        )

        analytics_result = await run_in_threadpool(
            video_processor.process,
            str(input_path),
            str(output_path),
            confidence,
            frame_interval,
            str(csv_path),
        )

        logger.info(
            "Video processing completed"
        )

        # ----------------------------------------------------
        # G. VERIFY OUTPUT FILES
        # ----------------------------------------------------

        if not output_path.exists():

            raise HTTPException(
                status_code=500,
                detail=(
                    "Video processing completed without "
                    "creating the output video."
                ),
            )

        if output_path.stat().st_size == 0:

            raise HTTPException(
                status_code=500,
                detail="Processed video file is empty.",
            )

        # ----------------------------------------------------
        # H. PREPARE ANALYTICS
        # ----------------------------------------------------

        analytics = normalize_analytics(
            analytics_result
        )

        # ----------------------------------------------------
        # I. PREPARE RESPONSE URLS
        # ----------------------------------------------------

        video_relative_url = (
            f"/outputs/{output_path.name}"
        )

        csv_relative_url = None

        if csv_path.exists():

            csv_relative_url = (
                f"/outputs/{csv_path.name}"
            )

        video_url = get_public_url(
            video_relative_url
        )

        csv_url = (
            get_public_url(csv_relative_url)
            if csv_relative_url
            else None
        )

        # ----------------------------------------------------
        # J. RETURN RESPONSE
        # ----------------------------------------------------

        return {
            "success": True,
            "message": "Video processed successfully",

            "filename": file.filename,
            "file_size": file_size,

            "video_info": video_info,

            "video_url": video_url,
            "output_video_url": video_url,

            "csv_url": csv_url,
            "csv_file_url": csv_url,

            "analytics": analytics,
            "summary": analytics,

            "detection_engine": "YOLO11",
            "tracking_engine": "ByteTrack",
            "inference_runtime": "TensorFlow Lite",

            "confidence": confidence,
            "frame_interval": frame_interval,
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Video processing failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(error)}",
        )

    finally:

        # Remove temporary uploaded video
        if input_path and input_path.exists():

            input_path.unlink(missing_ok=True)

        await file.close()


# ============================================================
# 14. OUTPUT FILE LIST ENDPOINT
# ============================================================

@app.get("/api/outputs")
async def list_outputs():

    files = []

    for path in OUTPUT_DIR.iterdir():

        if path.is_file():

            files.append({
                "filename": path.name,
                "size": path.stat().st_size,
                "url": get_public_url(
                    f"/outputs/{path.name}"
                ),
            })

    return {
        "success": True,
        "count": len(files),
        "files": files,
    }


# ============================================================
# 15. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )

