
"""
===========================================================
PragyanAI - People Detection & Counting
File: backend/utils/video_utils.py
===========================================================

Responsibilities:
1. Validate uploaded video files.
2. Extract video metadata.
3. Calculate FPS, duration and resolution.
4. Extract frames using OpenCV.
5. Support frame interval sampling.
6. Check video integrity.
7. Save extracted frames.
8. Provide reusable video utilities.
9. Maintain compatibility with backend/main.py.

Dependencies:
    opencv-python-headless
"""

import cv2
from pathlib import Path


# ===========================================================
# 1. SUPPORTED VIDEO FORMATS
# ===========================================================

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
}

# Compatibility with backend/main.py
ALLOWED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS


# ===========================================================
# 2. CUSTOM EXCEPTION
# ===========================================================

class VideoValidationError(Exception):
    """Raised when an uploaded video is invalid."""
    pass


# ===========================================================
# 3. VALIDATE VIDEO FILE
# ===========================================================

def validate_video(video_path):
    """
    Validate video file extension, existence and readability.

    Args:
        video_path: Path to uploaded video.

    Returns:
        dict: Video metadata if valid.

    Raises:
        VideoValidationError: If the video is invalid.
    """

    video_path = Path(video_path)

    # Check file existence
    if not video_path.exists():
        raise VideoValidationError(
            f"Video file not found: {video_path}"
        )

    # Check that the path is a file
    if not video_path.is_file():
        raise VideoValidationError(
            "Provided path is not a file."
        )

    # Check extension
    extension = video_path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported video format: {extension}"
        )

    # Check file size
    file_size = video_path.stat().st_size

    if file_size <= 0:
        raise VideoValidationError(
            "Uploaded video file is empty."
        )

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(
            "Unable to open video file. "
            "Please upload a valid video."
        )

    try:
        # Read video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Validate dimensions
        if width <= 0 or height <= 0:
            raise VideoValidationError(
                "Invalid video dimensions."
            )

        # Fallback FPS
        if fps <= 0:
            fps = 25.0

        # Verify at least one frame is readable
        success, first_frame = cap.read()

        if not success or first_frame is None:
            raise VideoValidationError(
                "Video contains no readable frames."
            )

        # Duration
        duration = (
            total_frames / fps
            if total_frames > 0
            else 0
        )

        # Metadata response
        return {
            "filename": video_path.name,
            "file_size": file_size,
            "file_size_mb": round(
                file_size / (1024 * 1024), 2
            ),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "frame_count": total_frames,
            "duration": round(duration, 2),
            "duration_seconds": round(duration, 2),
            "resolution": f"{width}x{height}",
            "format": extension.replace(".", "").upper(),
            "valid": True,
        }

    finally:
        cap.release()


# ===========================================================
# 4. GET VIDEO METADATA
# ===========================================================

def get_video_metadata(video_path):
    """
    Extract video metadata.

    Returns:
        Dictionary containing filename, FPS, dimensions,
        frame count, duration and resolution.
    """

    return validate_video(video_path)


# ===========================================================
# 5. GET VIDEO INFORMATION
# ===========================================================

def get_video_info(video_path):
    """
    Compatibility function for backend/main.py
    and other backend modules.
    """

    metadata = get_video_metadata(video_path)

    return {
        "filename": metadata["filename"],
        "fps": metadata["fps"],
        "width": metadata["width"],
        "height": metadata["height"],
        "frame_count": metadata["frame_count"],
        "total_frames": metadata["total_frames"],
        "duration": metadata["duration"],
        "duration_seconds": metadata["duration_seconds"],
        "resolution": metadata["resolution"],
        "file_size": metadata["file_size"],
        "file_size_mb": metadata["file_size_mb"],
    }


# ===========================================================
# 6. GET VIDEO DURATION
# ===========================================================

def get_video_duration(video_path):
    """
    Return video duration in seconds.
    """

    metadata = get_video_metadata(video_path)

    return metadata["duration_seconds"]


# ===========================================================
# 7. GET VIDEO FPS
# ===========================================================

def get_video_fps(video_path):
    """
    Return video FPS.
    """

    metadata = get_video_metadata(video_path)

    return metadata["fps"]


# ===========================================================
# 8. GET VIDEO RESOLUTION
# ===========================================================

def get_video_resolution(video_path):
    """
    Return video width and height.
    """

    metadata = get_video_metadata(video_path)

    return {
        "width": metadata["width"],
        "height": metadata["height"],
        "resolution": metadata["resolution"],
    }


# ===========================================================
# 9. EXTRACT VIDEO FRAMES
# ===========================================================

def extract_video_frames(
    video_path,
    frame_interval=1,
    max_frames=None
):
    """
    Extract frames from a video.

    Args:
        video_path: Input video path.
        frame_interval: Extract every Nth frame.
        max_frames: Optional maximum number of frames.

    Yields:
        Dictionary containing:
            frame_number
            timestamp_seconds
            frame
    """

    video_path = Path(video_path)

    if frame_interval < 1:
        raise ValueError(
            "frame_interval must be at least 1."
        )

    if max_frames is not None and max_frames < 1:
        return

    if not video_path.exists():
        raise VideoValidationError(
            f"Video file not found: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(
            "Unable to open video for frame extraction."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    frame_number = 0
    extracted_count = 0

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Select frames according to interval
            if (frame_number - 1) % frame_interval != 0:
                continue

            timestamp = (frame_number - 1) / fps

            yield {
                "frame_number": frame_number,
                "timestamp_seconds": round(
                    timestamp, 3
                ),
                "frame": frame,
            }

            extracted_count += 1

            if (
                max_frames is not None
                and extracted_count >= max_frames
            ):
                break

    finally:
        cap.release()


# ===========================================================
# 10. EXTRACT FRAMES - COMPATIBILITY FUNCTION
# ===========================================================

def extract_frames(
    video_path,
    frame_interval=1,
    max_frames=None
):
    """
    Compatibility wrapper.

    Yields:
        tuple: (frame_number, frame)
    """

    for item in extract_video_frames(
        video_path,
        frame_interval=frame_interval,
        max_frames=max_frames,
    ):
        yield (
            item["frame_number"],
            item["frame"],
        )


# ===========================================================
# 11. SAVE EXTRACTED FRAME
# ===========================================================

def save_frame(
    frame,
    output_path,
    image_quality=95
):
    """
    Save an extracted video frame as an image.
    """

    if frame is None:
        raise ValueError(
            "Cannot save an empty frame."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extension = output_path.suffix.lower()

    if extension in [".jpg", ".jpeg"]:

        image_quality = max(
            0,
            min(100, int(image_quality))
        )

        success = cv2.imwrite(
            str(output_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                image_quality,
            ],
        )

    else:

        success = cv2.imwrite(
            str(output_path),
            frame,
        )

    if not success:
        raise RuntimeError(
            f"Unable to save frame: {output_path}"
        )

    return str(output_path)


# ===========================================================
# 12. EXTRACT AND SAVE FRAMES
# ===========================================================

def extract_and_save_frames(
    video_path,
    output_directory,
    frame_interval=1,
    max_frames=None
):
    """
    Extract video frames and save them as JPG files.

    Returns:
        List of extracted frame information.
    """

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_frames = []

    for item in extract_video_frames(
        video_path,
        frame_interval=frame_interval,
        max_frames=max_frames,
    ):

        frame_number = item["frame_number"]

        filename = (
            f"frame_{frame_number:06d}.jpg"
        )

        frame_path = (
            output_directory / filename
        )

        save_frame(
            item["frame"],
            frame_path,
        )

        extracted_frames.append({
            "frame_number": frame_number,
            "timestamp_seconds": (
                item["timestamp_seconds"]
            ),
            "frame_path": str(frame_path),
        })

    return extracted_frames


# ===========================================================
# 13. CHECK VIDEO INTEGRITY
# ===========================================================

def check_video_integrity(video_path):
    """
    Read the complete video to check for readable frames.

    Returns:
        Integrity report.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        return {
            "valid": False,
            "message": "Video file does not exist.",
            "readable_frames": 0,
        }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()

        return {
            "valid": False,
            "message": "Unable to open video.",
            "readable_frames": 0,
        }

    readable_frames = 0

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame is not None:
                readable_frames += 1

    finally:
        cap.release()

    return {
        "valid": readable_frames > 0,
        "readable_frames": readable_frames,
        "message": (
            "Video is readable."
            if readable_frames > 0
            else "No readable video frames found."
        ),
    }


# ===========================================================
# 14. RESIZE VIDEO FRAME
# ===========================================================

def resize_frame(
    frame,
    width=None,
    height=None,
    max_dimension=None
):
    """
    Resize a frame while preserving aspect ratio.

    Provide width, height or max_dimension.

    If none is provided, the original frame is returned.
    """

    if frame is None:
        raise ValueError(
            "Input frame is empty."
        )

    original_height, original_width = frame.shape[:2]

    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            "Invalid frame dimensions."
        )

    if width is not None and height is not None:
        raise ValueError(
            "Provide either width or height, not both."
        )

    if width is not None:

        if width <= 0:
            raise ValueError(
                "Width must be greater than zero."
            )

        scale = width / original_width

    elif height is not None:

        if height <= 0:
            raise ValueError(
                "Height must be greater than zero."
            )

        scale = height / original_height

    elif max_dimension is not None:

        if max_dimension <= 0:
            raise ValueError(
                "max_dimension must be greater than zero."
            )

        scale = min(
            1.0,
            max_dimension / max(
                original_width,
                original_height,
            ),
        )

    else:
        return frame

    new_width = max(
        1,
        int(original_width * scale),
    )

    new_height = max(
        1,
        int(original_height * scale),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=interpolation,
    )


# ===========================================================
# 15. GENERATE VIDEO SUMMARY
# ===========================================================

def get_video_summary(video_path):
    """
    Return a compact video summary.
    """

    metadata = get_video_metadata(video_path)

    duration = metadata["duration_seconds"]

    minutes = int(duration // 60)
    seconds = int(duration % 60)

    return {
        "filename": metadata["filename"],
        "duration_seconds": duration,
        "duration_formatted": (
            f"{minutes}m {seconds}s"
        ),
        "fps": metadata["fps"],
        "resolution": metadata["resolution"],
        "width": metadata["width"],
        "height": metadata["height"],
        "total_frames": metadata["total_frames"],
        "file_size_mb": metadata["file_size_mb"],
        "format": metadata["format"],
        "valid": True,
    }


# ===========================================================
# 16. MODULE EXPORTS
# ===========================================================

__all__ = [
    "SUPPORTED_VIDEO_EXTENSIONS",
    "ALLOWED_EXTENSIONS",
    "VideoValidationError",
    "validate_video",
    "get_video_metadata",
    "get_video_info",
    "get_video_duration",
    "get_video_fps",
    "get_video_resolution",
    "extract_video_frames",
    "extract_frames",
    "save_frame",
    "extract_and_save_frames",
    "check_video_integrity",
    "resize_frame",
    "get_video_summary",
]
