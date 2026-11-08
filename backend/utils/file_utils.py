
"""
===========================================================
PragyanAI - People Detection & Counting
File: backend/utils/file_utils.py
===========================================================

Responsibilities:
1. Create upload and output directories.
2. Generate unique filenames.
3. Validate uploaded file extensions.
4. Save uploaded video files.
5. Calculate file sizes.
6. Delete temporary files.
7. Clean up old files.
8. List generated files.
9. Validate file paths.
10. Manage video and analytics output files.

Dependencies:
    pathlib
    uuid
    shutil
    datetime
"""

import os
import uuid
import shutil

from pathlib import Path
from datetime import datetime, timezone, timedelta


# ===========================================================
# 1. CONFIGURATION
# ===========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

UPLOAD_DIR = Path(
    os.getenv(
        "UPLOAD_DIR",
        str(BASE_DIR / "uploads")
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(BASE_DIR / "outputs")
    )
)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

ALLOWED_VIDEO_EXTENSIONS = {

    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v"

}

ALLOWED_OUTPUT_EXTENSIONS = {

    ".mp4",
    ".avi",
    ".mov",
    ".csv",
    ".json",
    ".jpg",
    ".jpeg",
    ".png",
    ".xlsx"

}


# ===========================================================
# 2. CREATE DIRECTORY
# ===========================================================

def create_directory(directory_path):
    """
    Create directory if it does not exist.

    Args:
        directory_path: Directory location.

    Returns:
        str: Created directory path.
    """

    directory_path = Path(directory_path)

    directory_path.mkdir(
        parents=True,
        exist_ok=True
    )

    return str(directory_path)


# ===========================================================
# 3. INITIALIZE STORAGE DIRECTORIES
# ===========================================================

def initialize_storage():
    """
    Initialize upload and output directories.
    """

    create_directory(UPLOAD_DIR)

    create_directory(OUTPUT_DIR)

    return {

        "upload_directory": str(UPLOAD_DIR),

        "output_directory": str(OUTPUT_DIR),

        "status": "initialized"

    }


# ===========================================================
# 4. GENERATE UNIQUE FILENAME
# ===========================================================

def generate_unique_filename(
    original_filename,
    suffix=None
):
    """
    Generate a unique filename.

    Example:
        people.mp4
        ->
        people_a82f93.mp4
    """

    original_filename = Path(
        original_filename
    )

    original_stem = original_filename.stem

    extension = original_filename.suffix.lower()

    unique_id = uuid.uuid4().hex[:12]

    if suffix:
        filename = (
            f"{original_stem}_{suffix}_{unique_id}{extension}"
        )

    else:
        filename = (
            f"{original_stem}_{unique_id}{extension}"
        )

    # Avoid unsafe path characters.
    safe_stem = "".join(

        character

        for character in Path(filename).stem

        if character.isalnum()
        or character in ("-", "_")

    )

    return safe_stem + extension


# ===========================================================
# 5. VALIDATE FILE EXTENSION
# ===========================================================

def validate_file_extension(
    filename,
    allowed_extensions=None
):
    """
    Validate file extension.
    """

    if not filename:
        return False

    extension = Path(filename).suffix.lower()

    if allowed_extensions is None:

        allowed_extensions = (
            ALLOWED_VIDEO_EXTENSIONS
        )

    return extension in allowed_extensions


# ===========================================================
# 6. VALIDATE FILE SIZE
# ===========================================================

def validate_file_size(
    file_size,
    max_size=MAX_UPLOAD_SIZE
):
    """
    Validate file size in bytes.
    """

    if file_size <= 0:
        raise ValueError(
            "File is empty."
        )

    if file_size > max_size:
        raise ValueError(
            f"File exceeds maximum size of "
            f"{max_size / (1024 * 1024):.0f} MB."
        )

    return True


# ===========================================================
# 7. GET FILE SIZE
# ===========================================================

def get_file_size(file_path):
    """
    Return file size in bytes.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        return 0

    return file_path.stat().st_size


# ===========================================================
# 8. GET HUMAN-READABLE FILE SIZE
# ===========================================================

def get_readable_file_size(file_path):
    """
    Convert file size into KB, MB or GB.
    """

    size = get_file_size(file_path)

    if size < 1024:
        return f"{size} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"

    if size < 1024 ** 3:
        return f"{size / (1024 ** 2):.2f} MB"

    return f"{size / (1024 ** 3):.2f} GB"


# ===========================================================
# 9. SAVE UPLOADED FILE
# ===========================================================

def save_uploaded_file(
    uploaded_file,
    output_directory=None,
    max_size=MAX_UPLOAD_SIZE
):
    """
    Save an uploaded file-like object.

    The object must support:
        read()
        seek()

    Returns:
        Dictionary containing saved file details.
    """

    if output_directory is None:
        output_directory = UPLOAD_DIR

    output_directory = Path(
        output_directory
    )

    create_directory(
        output_directory
    )

    original_filename = getattr(
        uploaded_file,
        "filename",
        None
    )

    if not original_filename:

        raise ValueError(
            "Uploaded file has no filename."
        )

    if not validate_file_extension(
        original_filename
    ):

        raise ValueError(
            "Unsupported video file format."
        )

    unique_filename = generate_unique_filename(
        original_filename
    )

    file_path = (
        output_directory / unique_filename
    )

    file_size = 0

    try:

        with file_path.open("wb") as destination:

            while True:

                chunk = uploaded_file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                file_size += len(chunk)

                if file_size > max_size:

                    raise ValueError(
                        "Uploaded file exceeds size limit."
                    )

                destination.write(chunk)

        validate_file_size(
            file_size,
            max_size
        )

        return {

            "original_filename": original_filename,

            "saved_filename": unique_filename,

            "file_path": str(file_path),

            "file_size_bytes": file_size,

            "file_size_readable": get_readable_file_size(
                file_path
            ),

            "status": "uploaded"

        }

    except Exception:

        if file_path.exists():
            file_path.unlink()

        raise


# ===========================================================
# 10. DELETE FILE
# ===========================================================

def delete_file(file_path):
    """
    Delete a file if it exists.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        return False

    if not file_path.is_file():
        raise ValueError(
            "Provided path is not a file."
        )

    file_path.unlink()

    return True


# ===========================================================
# 11. DELETE DIRECTORY
# ===========================================================

def delete_directory(directory_path):
    """
    Delete a directory and its contents.

    Only use this on explicitly designated temporary
    directories, never on arbitrary user-supplied paths.
    """

    directory_path = Path(directory_path).resolve()

    protected_paths = {
        UPLOAD_DIR.resolve(),
        OUTPUT_DIR.resolve()
    }

    if directory_path in protected_paths:
        raise ValueError(
            "Refusing to delete a main storage directory."
        )

    if not directory_path.exists():
        return False

    if not directory_path.is_dir():
        raise ValueError(
            "Provided path is not a directory."
        )

    shutil.rmtree(directory_path)

    return True


# ===========================================================
# 12. GENERATE OUTPUT FILE PATH
# ===========================================================

def generate_output_path(
    job_id,
    extension,
    output_directory=None
):
    """
    Generate a path for an output file.

    Examples:
        job123_annotated.mp4
        job123_analytics.csv
    """

    if output_directory is None:
        output_directory = OUTPUT_DIR

    extension = extension.lower()

    if not extension.startswith("."):
        extension = "." + extension

    if extension not in ALLOWED_OUTPUT_EXTENSIONS:
        raise ValueError(
            "Unsupported output file extension."
        )

    output_directory = Path(
        output_directory
    )

    create_directory(
        output_directory
    )

    safe_job_id = "".join(
        character
        for character in str(job_id)
        if character.isalnum() or character in ("-", "_")
    )

    if not safe_job_id:
        raise ValueError("Invalid job ID.")

    filename = (
        f"{safe_job_id}{extension}"
    )

    return str(
        output_directory / filename
    )


# ===========================================================
# 13. GET FILE INFORMATION
# ===========================================================

def get_file_info(file_path):
    """
    Return file metadata.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    stat = file_path.stat()

    return {

        "filename": file_path.name,

        "extension": file_path.suffix.lower(),

        "size_bytes": stat.st_size,

        "size_readable": get_readable_file_size(
            file_path
        ),

        "created_timestamp": stat.st_ctime,

        "modified_timestamp": stat.st_mtime

    }


# ===========================================================
# 14. LIST FILES IN DIRECTORY
# ===========================================================

def list_files(
    directory,
    extensions=None
):
    """
    List files in a directory.

    Optionally filter by extensions.
    """

    directory = Path(directory)

    if not directory.exists():
        return []

    results = []

    for file_path in directory.iterdir():

        if not file_path.is_file():
            continue

        if extensions:

            if file_path.suffix.lower() not in extensions:
                continue

        results.append(
            get_file_info(file_path)
        )

    return results


# ===========================================================
# 15. CLEANUP OLD FILES
# ===========================================================

def cleanup_old_files(
    directory,
    max_age_hours=24
):
    """
    Remove old files from a designated storage directory.

    Returns:
        Number of files deleted.
    """

    directory = Path(directory).resolve()

    allowed_directories = {
        UPLOAD_DIR.resolve(),
        OUTPUT_DIR.resolve()
    }

    if directory not in allowed_directories:
        raise ValueError(
            "Cleanup is restricted to configured "
            "upload and output directories."
        )

    if not directory.exists():
        return 0

    cutoff_time = (
        datetime.now(timezone.utc)
        - timedelta(hours=max_age_hours)
    ).timestamp()

    deleted_count = 0

    for file_path in directory.iterdir():

        if not file_path.is_file():
            continue

        if file_path.name == ".gitkeep":
            continue

        modified_time = file_path.stat().st_mtime

        if modified_time < cutoff_time:

            file_path.unlink()

            deleted_count += 1

    return deleted_count


# ===========================================================
# 16. CHECK FILE EXISTS
# ===========================================================

def file_exists(file_path):
    """
    Check whether a file exists.
    """

    return Path(file_path).is_file()


# ===========================================================
# 17. INITIALIZE STORAGE ON IMPORT
# ===========================================================

initialize_storage()
