
import React, { useRef, useState } from "react";

import {
  Upload,
  FileVideo,
  X,
  CheckCircle,
  AlertCircle,
  Film,
  HardDrive,
} from "lucide-react";


// ======================================================
// CONFIGURATION
// ======================================================

const MAX_FILE_SIZE = 500 * 1024 * 1024;

const ALLOWED_EXTENSIONS = [
  "mp4",
  "avi",
  "mov",
  "mkv",
  "webm",
];


// ======================================================
// FORMAT FILE SIZE
// ======================================================

const formatFileSize = (bytes) => {

  if (!bytes) return "0 Bytes";

  const units = [
    "Bytes",
    "KB",
    "MB",
    "GB",
  ];

  const index = Math.floor(
    Math.log(bytes) / Math.log(1024)
  );

  return (
    (bytes / Math.pow(1024, index)).toFixed(2) +
    " " +
    units[index]
  );
};


// ======================================================
// FORMAT VIDEO DURATION
// ======================================================

const formatDuration = (seconds) => {

  if (!Number.isFinite(seconds)) return "Unknown";

  const minutes = Math.floor(seconds / 60);

  const remainingSeconds = Math.floor(
    seconds % 60
  );

  return `${minutes}:${String(
    remainingSeconds
  ).padStart(2, "0")}`;
};


// ======================================================
// VIDEO UPLOAD COMPONENT
// ======================================================

const VideoUpload = ({
  selectedFile = null,
  onFileSelect,
  disabled = false,
}) => {

  const inputRef = useRef(null);

  const [isDragging, setIsDragging] = useState(false);

  const [error, setError] = useState("");

  const [videoMetadata, setVideoMetadata] = useState(null);


  // ====================================================
  // VALIDATE FILE
  // ====================================================

  const validateFile = (file) => {

    if (!file) {
      return "Please select a video file.";
    }

    if (
      typeof file.name !== "string" ||
      typeof file.size !== "number"
    ) {
      return "Invalid file object.";
    }

    if (file.size <= 0) {
      return "The selected video is empty.";
    }

    const extension = file.name
      .split(".")
      .pop()
      .toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(extension)) {

      return (
        "Unsupported format. Please upload MP4, AVI, MOV, MKV or WEBM."
      );
    }

    if (file.size > MAX_FILE_SIZE) {

      return (
        "Video size exceeds the maximum limit of 500 MB."
      );
    }

    return null;
  };


  // ====================================================
  // SELECT FILE
  // ====================================================

  const handleFile = (file) => {

    if (disabled) return;

    setError("");
    setVideoMetadata(null);

    const validationError = validateFile(file);

    if (validationError) {

      setError(validationError);

      return;
    }

    // IMPORTANT:
    // Pass the actual browser File object to App.jsx.
    if (typeof onFileSelect === "function") {
      onFileSelect(file);
    }

    // Read metadata without uploading the file.
    const objectUrl = URL.createObjectURL(file);

    const video = document.createElement("video");

    video.preload = "metadata";

    video.onloadedmetadata = () => {

      setVideoMetadata({
        duration: video.duration,
        width: video.videoWidth,
        height: video.videoHeight,
      });

      URL.revokeObjectURL(objectUrl);
    };

    video.onerror = () => {

      URL.revokeObjectURL(objectUrl);

      // Metadata parsing can fail for some codecs.
      // Do not reject solely based on browser metadata.
    };

    video.src = objectUrl;
  };


  // ====================================================
  // INPUT CHANGE
  // ====================================================

  const handleInputChange = (event) => {

    const file = event.target.files?.[0];

    if (file) {
      handleFile(file);
    }

    // Allow selecting the same file again.
    event.target.value = "";
  };


  // ====================================================
  // DRAG EVENTS
  // ====================================================

  const handleDragOver = (event) => {

    event.preventDefault();

    if (!disabled) {
      setIsDragging(true);
    }
  };


  const handleDragLeave = (event) => {

    event.preventDefault();

    setIsDragging(false);
  };


  const handleDrop = (event) => {

    event.preventDefault();

    setIsDragging(false);

    if (disabled) return;

    const file = event.dataTransfer.files?.[0];

    if (file) {
      handleFile(file);
    }
  };


  // ====================================================
  // BROWSE BUTTON
  // ====================================================

  const openFilePicker = () => {

    if (disabled) return;

    inputRef.current?.click();
  };


  // ====================================================
  // REMOVE FILE
  // ====================================================

  const removeFile = (event) => {

    event.stopPropagation();

    if (disabled) return;

    setError("");
    setVideoMetadata(null);

    if (typeof onFileSelect === "function") {
      onFileSelect(null);
    }

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };


  // ====================================================
  // RENDER
  // ====================================================

  return (

    <div className="video-upload-wrapper">

      <div className="video-upload-header">

        <div className="section-title">

          <FileVideo size={20} />

          <h3>Upload Video</h3>

        </div>

        <span className="upload-supported">
          MP4, AVI, MOV, MKV, WEBM
        </span>

      </div>


      {/* Hidden input */}

      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.avi,.mov,.mkv,.webm,video/*"
        onChange={handleInputChange}
        disabled={disabled}
        hidden
      />


      {/* Selected file */}

      {selectedFile ? (

        <div className="upload-selected-file">

          <div className="selected-file-icon">
            <FileVideo size={30} />
          </div>

          <div className="selected-file-details">

            <div className="selected-file-name">
              {selectedFile.name}
            </div>

            <div className="selected-file-meta">

              <span>
                <HardDrive size={14} />
                {formatFileSize(selectedFile.size)}
              </span>

              <span>
                <Film size={14} />
                {selectedFile.name.split(".").pop().toUpperCase()}
              </span>

            </div>

            {videoMetadata && (

              <div className="selected-file-meta">

                <span>
                  Duration:{" "}
                  {formatDuration(videoMetadata.duration)}
                </span>

                <span>
                  {videoMetadata.width} × {videoMetadata.height}
                </span>

              </div>

            )}

            <div className="selected-file-status">

              <CheckCircle size={15} />

              <span>Video selected successfully</span>

            </div>

          </div>


          <button
            type="button"
            className="remove-upload-button"
            onClick={removeFile}
            disabled={disabled}
            title="Remove video"
          >

            <X size={18} />

          </button>

        </div>

      ) : (

        <div
          className={`video-upload-area ${
            isDragging ? "dragging" : ""
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={openFilePicker}
          role="button"
          tabIndex={disabled ? -1 : 0}
          aria-label="Upload video"
          aria-disabled={disabled}
          onKeyDown={(event) => {

            if (
              !disabled &&
              (event.key === "Enter" || event.key === " ")
            ) {

              event.preventDefault();

              openFilePicker();
            }

          }}
        >

          <div className="upload-icon">
            <Upload size={36} />
          </div>

          <h4>Drag and drop your video here</h4>

          <p>or click to browse files</p>

          <span className="upload-limit">
            Maximum file size: 500 MB
          </span>

        </div>

      )}


      {/* Error message */}

      {error && (

        <div className="upload-error">

          <AlertCircle size={18} />

          <span>{error}</span>

        </div>

      )}


      {/* Upload information */}

      <div className="upload-information">

        <div>
          <CheckCircle size={14} />
          Supported video formats
        </div>

        <div>
          <CheckCircle size={14} />
          YOLO11 person detection
        </div>

        <div>
          <CheckCircle size={14} />
          ByteTrack people tracking
        </div>

      </div>

    </div>

  );
};


export default VideoUpload;

