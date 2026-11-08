
import os
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
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from services.video_processor import VideoProcessor
from utils.video_utils import (
    ALLOWED_EXTENSIONS,
    VideoValidationError,
    validate_video,
)


# ============================================================
# 1. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("pragyanai")


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def resolve_path(value):
    """Resolve relative paths against the backend directory."""
    path = Path(value)

    if path.is_absolute():
        return path

    return BASE_DIR / path


MODEL_PATH = resolve_path(
    os.getenv(
        "MODEL_PATH",
        "models/yolo11n.tflite",
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

# Your actual Render deployment URL
PUBLIC_API_URL = os.getenv(
    "PUBLIC_API_URL",
    "https://pragyanai-yolo-tf-lite-object-detction.onrender.com",
).rstrip("/")

# Your actual Netlify deployment URL
FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "https://tflite-objectdetection-peopleanalysis.netlify.app",
)

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. CORS CONFIGURATION
# ============================================================

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

ALLOWED_ORIGINS.extend(
    origin.strip().rstrip("/")
    for origin in FRONTEND_ORIGIN.split(",")
    if origin.strip()
)

ALLOWED_ORIGINS = list(dict.fromkeys(ALLOWED_ORIGINS))


# ============================================================
# 4. GLOBAL APPLICATION STATE
# ============================================================

processor = None
model_error = None


# ============================================================
# 5. APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize YOLO11 TFLite model when FastAPI starts.
    """

    global processor
    global model_error

    app.state.processor = None
    app.state.model_error = None

    logger.info("=" * 60)
    logger.info("Starting PragyanAI People Detection API")
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Model exists: %s", MODEL_PATH.is_file())
    logger.info("=" * 60)

    try:

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"YOLO11 TFLite model not found: {MODEL_PATH}"
            )

        logger.info("Loading YOLO11 TFLite model...")

        # VideoProcessor constructor:
        # VideoProcessor(model_path)
        app.state.processor = await run_in_threadpool(
            VideoProcessor,
            str(MODEL_PATH),
        )

        processor = app.state.processor
        model_error = None

        logger.info("YOLO11 model loaded successfully.")

    except Exception as exc:

        processor = None
        model_error = str(exc)

        app.state.processor = None
        app.state.model_error = model_error

        logger.exception(
            "Model initialization failed: %s",
            exc,
        )

    yield

    logger.info("Shutting down PragyanAI API")

    app.state.processor = None
    processor = None


# ============================================================
# 6. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PragyanAI People Detection & Counting API",
    description=(
        "YOLO11 TFLite based People Detection, "
        "ByteTrack Object Tracking and Video Analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 7. ADD CORS MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
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

def get_processor():
    """Return initialized processor."""

    current_processor = getattr(
        app.state,
        "processor",
        None,
    )

    if current_processor is None:

        current_error = getattr(
            app.state,
            "model_error",
            None,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Detection engine is not initialized. "
                f"Model error: {current_error}"
            ),
        )

    return current_processor


def get_public_url(filename):
    """Generate publicly accessible output URL."""

    relative_url = f"/outputs/{filename}"

    return f"{PUBLIC_API_URL}{relative_url}"


def validate_filename(filename):
    """Validate video extension."""

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="No filename received.",
        )

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video extension: "
                f"{extension or 'missing'}. "
                "Supported: MP4, MOV, AVI, MKV and WEBM."
            ),
        )

    return extension


async def save_uploaded_file(file, destination):
    """
    Save uploaded video in chunks.
    Avoid loading the entire video into RAM.
    """

    total_size = 0

    try:

        with destination.open("wb") as output:

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
                            f"upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
                        ),
                    )

                output.write(chunk)

        if total_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty.",
            )

        return total_size

    except Exception:

        destination.unlink(missing_ok=True)
        raise


def normalize_analytics(result):
    """Ensure analytics is JSON serializable."""

    if result is None:
        return {}

    if isinstance(result, dict):
        return result

    if hasattr(result, "to_dict"):
        return result.to_dict()

    if hasattr(result, "__dict__"):
        return result.__dict__

    return {
        "result": str(result),
    }


# ============================================================
# 10. ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():

    return {
        "application": "PragyanAI People Detection & Counting",
        "status": "running",
        "version": "1.0.0",
        "backend": "FastAPI",
        "detection_engine": "YOLO11",
        "tracking_engine": "ByteTrack",
        "inference_runtime": "TensorFlow Lite",
        "frontend": (
            "https://tflite-objectdetection-peopleanalysis.netlify.app"
        ),
        "documentation": "/docs",
        "health": "/health",
    }


# ============================================================
# 11. HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():

    current_processor = getattr(
        app.state,
        "processor",
        None,
    )

    current_error = getattr(
        app.state,
        "model_error",
        None,
    )

    model_loaded = current_processor is not None

    return {
        "status": (
            "healthy" if model_loaded else "degraded"
        ),
        "api": "running",
        "model_loaded": model_loaded,
        "model_exists": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
        "model_error": current_error,
        "detection_engine": "YOLO11",
        "tracking_engine": "ByteTrack",
        "inference_runtime": "TensorFlow Lite",
    }


# ============================================================
# 12. MODEL INFORMATION
# ============================================================

@app.get("/api/model-info")
async def model_info():

    current_processor = getattr(
        app.state,
        "processor",
        None,
    )

    return {
        "model_name": "YOLO11 Nano",
        "model_file": MODEL_PATH.name,
        "model_format": "TFLite",
        "model_loaded": current_processor is not None,
        "supported_class": "Person",
        "class_id": 0,
        "tracking_algorithm": "ByteTrack",
        "default_confidence": 0.35,
    }


# ============================================================
# 13. VIDEO PROCESSING API
# ============================================================

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    frame_interval: int = Form(1),
):
    """
    Upload video and perform:

    1. File validation
    2. Video decoding
    3. YOLO11 TFLite inference
    4. ByteTrack tracking
    5. Annotated video generation
    6. CSV analytics generation
    """

    incoming = None
    output = None
    csv_file = None

    try:

        # ----------------------------------------------------
        # A. CHECK MODEL
        # ----------------------------------------------------

        current_processor = get_processor()

        # ----------------------------------------------------
        # B. VALIDATE PARAMETERS
        # ----------------------------------------------------

        if not 0.0 <= confidence <= 1.0:

            raise HTTPException(
                status_code=400,
                detail="Confidence must be between 0.0 and 1.0.",
            )

        if not 1 <= frame_interval <= 100:

            raise HTTPException(
                status_code=400,
                detail="Frame interval must be between 1 and 100.",
            )

        # ----------------------------------------------------
        # C. VALIDATE FILE NAME
        # ----------------------------------------------------

        original_filename = file.filename or ""

        extension = validate_filename(
            original_filename
        )

        # ----------------------------------------------------
        # D. GENERATE UNIQUE FILE NAMES
        # ----------------------------------------------------

        unique_id = uuid.uuid4().hex

        incoming = UPLOAD_DIR / (
            f"{unique_id}{extension}"
        )

        output = OUTPUT_DIR / (
            f"{unique_id}_processed.mp4"
        )

        csv_file = OUTPUT_DIR / (
            f"{unique_id}_analytics.csv"
        )

        # ----------------------------------------------------
        # E. LOG UPLOAD DETAILS
        # ----------------------------------------------------

        logger.info("=" * 60)
        logger.info("VIDEO UPLOAD RECEIVED")
        logger.info("Filename: %s", original_filename)
        logger.info("Content type: %s", file.content_type)
        logger.info("Extension: %s", extension)
        logger.info("Destination: %s", incoming)
        logger.info("=" * 60)

        # ----------------------------------------------------
        # F. SAVE UPLOADED VIDEO
        # ----------------------------------------------------

        file_size = await save_uploaded_file(
            file,
            incoming,
        )

        logger.info(
            "Upload saved successfully: %.2f MB",
            file_size / (1024 * 1024),
        )

        # Log first bytes for file-format diagnostics
        with incoming.open("rb") as uploaded:
            header = uploaded.read(32)

        logger.info(
            "Video file header: %s",
            header.hex(),
        )

        # ----------------------------------------------------
        # G. VALIDATE VIDEO CONTENT
        # ----------------------------------------------------

        logger.info("Starting OpenCV video validation")

        try:

            video_info = await run_in_threadpool(
                validate_video,
                incoming,
            )

        except VideoValidationError as exc:

            logger.warning(
                "Video validation failed: %s",
                exc,
            )

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        logger.info(
            "Video validation successful: %s",
            video_info,
        )

        # ----------------------------------------------------
        # H. PROCESS VIDEO
        # ----------------------------------------------------

        logger.info(
            "Starting YOLO11 + ByteTrack processing"
        )

        analytics_result = await run_in_threadpool(
            current_processor.process,
            str(incoming),
            str(output),
            confidence,
            frame_interval,
            str(csv_file),
        )

        logger.info("Video processing completed")

        # ----------------------------------------------------
        # I. VERIFY OUTPUT VIDEO
        # ----------------------------------------------------

        if not output.exists():

            raise RuntimeError(
                "Output video was not generated."
            )

        if output.stat().st_size == 0:

            raise RuntimeError(
                "Generated output video is empty."
            )

        # ----------------------------------------------------
        # J. VERIFY CSV
        # ----------------------------------------------------

        csv_url = None

        if csv_file.exists():

            if csv_file.stat().st_size > 0:

                csv_url = get_public_url(
                    csv_file.name
                )

        # ----------------------------------------------------
        # K. ANALYTICS
        # ----------------------------------------------------

        analytics = normalize_analytics(
            analytics_result
        )

        # ----------------------------------------------------
        # L. OUTPUT URLS
        # ----------------------------------------------------

        video_url = get_public_url(
            output.name
        )

        # ----------------------------------------------------
        # M. RETURN RESULTS
        # ----------------------------------------------------

        response = {
            "success": True,
            "message": "Video processed successfully",

            "filename": original_filename,
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

        logger.info(
            "Returning successful processing response"
        )

        return response

    # --------------------------------------------------------
    # N. ERROR HANDLING
    # --------------------------------------------------------

    except HTTPException:
        raise

    except VideoValidationError as exc:

        logger.warning(
            "Invalid video file: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Video processing failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Video processing failed: "
                f"{str(exc)}"
            ),
        ) from exc

    # --------------------------------------------------------
    # O. CLEANUP
    # --------------------------------------------------------

    finally:

        await file.close()

        if incoming is not None:
            incoming.unlink(missing_ok=True)

        logger.info(
            "Temporary uploaded video cleaned up"
        )


# ============================================================
# 14. LIST OUTPUT FILES
# ============================================================

@app.get("/api/outputs")
async def list_outputs():

    files = []

    for path in OUTPUT_DIR.iterdir():

        if path.is_file():

            files.append({
                "filename": path.name,
                "size": path.stat().st_size,
                "url": get_public_url(path.name),
            })

    return {
        "success": True,
        "count": len(files),
        "files": files,
    }


# ============================================================
# 15. DELETE OUTPUT FILE
# ============================================================

@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):

    # Prevent path traversal
    safe_name = Path(filename).name

    if safe_name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    target = OUTPUT_DIR / safe_name

    if not target.is_file():

        raise HTTPException(
            status_code=404,
            detail="Output file not found.",
        )

    target.unlink()

    return {
        "success": True,
        "message": "Output file deleted.",
        "filename": safe_name,
    }


# ============================================================
# 16. APPLICATION START
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
