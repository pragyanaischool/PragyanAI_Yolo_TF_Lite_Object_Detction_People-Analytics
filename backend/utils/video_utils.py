
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
7. Provide reusable video utilities.

Dependencies:
    opencv-python-headless
    pathlib
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
    ".m4v"

}


# ===========================================================
# 2. VALIDATE VIDEO FILE
# ===========================================================

def validate_video(video_path):
    """
    Validate video file extension and readability.

    Args:
        video_path: Path to uploaded video.

    Returns:
        bool: True if video is valid.

    Raises:
        FileNotFoundError
        ValueError
    """

    video_path = Path(video_path)

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if not video_path.is_file():

        raise ValueError(
            "Provided path is not a file."
        )

    extension = video_path.suffix.lower()

    if extension not in SUPPORTED_VIDEO_EXTENSIONS:

        raise ValueError(
            f"Unsupported video format: {extension}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise ValueError(
            "Unable to open video file."
        )

    success, frame = cap.read()

    cap.release()

    if not success or frame is None:

        raise ValueError(
            "Video contains no readable frames."
        )

    return True


# ===========================================================
# 3. GET VIDEO METADATA
# ===========================================================

def get_video_metadata(video_path):
    """
    Extract video metadata.

    Returns:
        Dictionary containing:
        - filename
        - fps
        - width
        - height
        - total_frames
        - duration_seconds
        - resolution
    """

    video_path = Path(video_path)

    validate_video(video_path)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if fps <= 0:
        fps = 25.0

    duration = (
        total_frames / fps
        if total_frames > 0
        else 0
    )

    return {

        "filename": video_path.name,

        "fps": round(fps, 2),

        "width": width,

        "height": height,

        "total_frames": total_frames,

        "duration_seconds": round(
            duration,
            2
        ),

        "resolution": f"{width}x{height}"

    }


# ===========================================================
# 4. GET VIDEO DURATION
# ===========================================================

def get_video_duration(video_path):
    """
    Return video duration in seconds.
    """

    metadata = get_video_metadata(
        video_path
    )

    return metadata["duration_seconds"]


# ===========================================================
# 5. GET VIDEO FPS
# ===========================================================

def get_video_fps(video_path):
    """
    Return video FPS.
    """

    metadata = get_video_metadata(
        video_path
    )

    return metadata["fps"]


# ===========================================================
# 6. GET VIDEO RESOLUTION
# ===========================================================

def get_video_resolution(video_path):
    """
    Return video width and height.
    """

    metadata = get_video_metadata(
        video_path
    )

    return {

        "width": metadata["width"],

        "height": metadata["height"],

        "resolution": metadata["resolution"]

    }


# ===========================================================
# 7. EXTRACT VIDEO FRAMES
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
        - frame_number
        - timestamp_seconds
        - frame
    """

    video_path = Path(video_path)

    validate_video(video_path)

    frame_interval = max(
        1,
        int(frame_interval)
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

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

            # Extract selected frames.
            if (
                (frame_number - 1)
                % frame_interval != 0
            ):
                continue

            timestamp = (
                frame_number - 1
            ) / fps

            yield {

                "frame_number": frame_number,

                "timestamp_seconds": round(
                    timestamp,
                    3
                ),

                "frame": frame

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
# 8. SAVE EXTRACTED FRAME
# ===========================================================

def save_frame(
    frame,
    output_path,
    image_quality=95
):
    """
    Save an extracted video frame as an image.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    extension = output_path.suffix.lower()

    if extension in [".jpg", ".jpeg"]:

        success = cv2.imwrite(
            str(output_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                image_quality
            ]
        )

    else:

        success = cv2.imwrite(
            str(output_path),
            frame
        )

    if not success:

        raise RuntimeError(
            f"Unable to save frame: {output_path}"
        )

    return str(output_path)


# ===========================================================
# 9. EXTRACT AND SAVE FRAMES
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

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    extracted_frames = []

    for item in extract_video_frames(
        video_path,
        frame_interval=frame_interval,
        max_frames=max_frames
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
            frame_path
        )

        extracted_frames.append({

            "frame_number": frame_number,

            "timestamp_seconds": (
                item["timestamp_seconds"]
            ),

            "frame_path": str(frame_path)

        })

    return extracted_frames


# ===========================================================
# 10. CHECK VIDEO INTEGRITY
# ===========================================================

def check_video_integrity(video_path):
    """
    Read the complete video to check for
    readable frames.

    Returns:
        Integrity report.
    """

    video_path = Path(video_path)

    if not video_path.exists():

        return {

            "valid": False,

            "message": "Video file does not exist.",

            "readable_frames": 0

        }

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return {

            "valid": False,

            "message": "Unable to open video.",

            "readable_frames": 0

        }

    readable_frames = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame is not None:
            readable_frames += 1

    cap.release()

    return {

        "valid": readable_frames > 0,

        "readable_frames": readable_frames,

        "message": (
            "Video is readable."
            if readable_frames > 0
            else "No readable video frames found."
        )

    }


# ===========================================================
# 11. RESIZE VIDEO FRAME
# ===========================================================

def resize_frame(
    frame,
    width=None,
    height=None
):
    """
    Resize a frame while preserving aspect ratio.

    Provide either width or height.
    """

    original_height, original_width = (
        frame.shape[:2]
    )

    if original_width <= 0 or original_height <= 0:

        raise ValueError(
            "Invalid frame dimensions."
        )

    if width is not None:

        scale = width / original_width

        new_width = int(width)

        new_height = int(
            original_height * scale
        )

    elif height is not None:

        scale = height / original_height

        new_height = int(height)

        new_width = int(
            original_width * scale
        )

    else:

        return frame

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ===========================================================
# 12. GENERATE VIDEO SUMMARY
# ===========================================================

def get_video_summary(video_path):
    """
    Return a compact video summary.
    """

    metadata = get_video_metadata(
        video_path
    )

    return {

        "filename": metadata["filename"],

        "duration_seconds": (
            metadata["duration_seconds"]
        ),

        "fps": metadata["fps"],

        "resolution": metadata["resolution"],

        "total_frames": metadata["total_frames"]

    }
