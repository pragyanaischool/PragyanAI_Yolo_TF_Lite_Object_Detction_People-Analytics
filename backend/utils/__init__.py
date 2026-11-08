
"""
PragyanAI People Detection & Counting

Backend Utilities Package

Modules:
    video_utils.py
        Video metadata, validation and frame utilities.

    file_utils.py
        File handling, upload validation and output management.
"""

from .video_utils import (
    get_video_metadata,
    validate_video,
    extract_video_frames
)

from .file_utils import (
    create_directory,
    generate_unique_filename,
    save_uploaded_file,
    delete_file,
    get_file_size
)

__all__ = [
    "get_video_metadata",
    "validate_video",
    "extract_video_frames",
    "create_directory",
    "generate_unique_filename",
    "save_uploaded_file",
    "delete_file",
    "get_file_size"
]
