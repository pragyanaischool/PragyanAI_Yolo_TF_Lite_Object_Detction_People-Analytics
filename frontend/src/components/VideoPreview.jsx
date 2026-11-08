
import React, { useEffect, useState } from "react";

import {
  Video,
  FileVideo,
  Trash2,
  Clock,
  HardDrive,
  FileType,
  AlertCircle
} from "lucide-react";

// --------------------------------------------------
// Format File Size
// --------------------------------------------------

function formatFileSize(bytes) {
  if (bytes < 1024) {
    return `${bytes} Bytes`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  }

  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// --------------------------------------------------
// Format Duration
// --------------------------------------------------

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) {
    return "Calculating...";
  }

  const totalSeconds = Math.floor(seconds);

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;

  if (hours > 0) {
    return [
      hours,
      String(minutes).padStart(2, "0"),
      String(remainingSeconds).padStart(2, "0")
    ].join(":");
  }

  return [
    String(minutes).padStart(2, "0"),
    String(remainingSeconds).padStart(2, "0")
  ].join(":");
}

// --------------------------------------------------
// Video Preview Component
// --------------------------------------------------

function VideoPreview({
  file,
  onRemove,
  disabled = false
}) {
  const [videoUrl, setVideoUrl] = useState("");
  const [duration, setDuration] = useState(null);
  const [videoError, setVideoError] = useState("");

  // ------------------------------------------------
  // Create and Release Temporary Video URL
  // ------------------------------------------------

  useEffect(() => {
    if (!file) {
      setVideoUrl("");
      setDuration(null);
      setVideoError("");
      return;
    }

    const url = URL.createObjectURL(file);

    setVideoUrl(url);
    setDuration(null);
    setVideoError("");

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  // ------------------------------------------------
  // Extract Video Metadata
  // ------------------------------------------------

  const handleMetadataLoaded = (event) => {
    const video = event.currentTarget;

    if (Number.isFinite(video.duration)) {
      setDuration(video.duration);
    } else {
      setDuration(0);
    }

    setVideoError("");
  };

  // ------------------------------------------------
  // Handle Video Loading Error
  // ------------------------------------------------

  const handleVideoError = () => {
    setVideoError(
      "Unable to preview this video in your browser. The backend may still support processing this format."
    );
  };

  // ------------------------------------------------
  // Remove Video
  // ------------------------------------------------

  const handleRemove = () => {
    if (disabled) return;

    onRemove?.();
  };

  // ------------------------------------------------
  // File Information
  // ------------------------------------------------

  if (!file) {
    return null;
  }

  const fileExtension =
    file.name.split(".").pop()?.toUpperCase() || "VIDEO";

  // ------------------------------------------------
  // Render
  // ------------------------------------------------

  return (
    <section className="preview-card">

      {/* Preview Header */}
      <div className="preview-header">

        <div>
          <h2>Video Preview</h2>

          <p className="card-description">
            Review your uploaded video before processing.
          </p>
        </div>

        <button
          type="button"
          className="remove-button"
          onClick={handleRemove}
          disabled={disabled}
          title="Remove uploaded video"
          aria-label="Remove uploaded video"
        >
          <Trash2 size={15} />
          Remove
        </button>

      </div>

      {/* Video Player */}
      {videoUrl && (
        <div className="video-preview-container">

          {!videoError ? (
            <video
              key={videoUrl}
              className="preview-video"
              src={videoUrl}
              controls
              playsInline
              preload="metadata"
              onLoadedMetadata={handleMetadataLoaded}
              onError={handleVideoError}
            >
              Your browser does not support HTML5 video.
            </video>
          ) : (
            <div className="video-preview-error">
              <AlertCircle size={28} />
              <p>{videoError}</p>
            </div>
          )}

        </div>
      )}

      {/* Video Information */}
      <div className="video-information-grid">

        {/* File Name */}
        <div className="video-info-item">

          <div className="video-info-icon">
            <FileVideo size={18} />
          </div>

          <div>
            <span className="video-info-label">
              File Name
            </span>

            <strong className="video-info-value">
              {file.name}
            </strong>
          </div>

        </div>

        {/* File Size */}
        <div className="video-info-item">

          <div className="video-info-icon">
            <HardDrive size={18} />
          </div>

          <div>
            <span className="video-info-label">
              File Size
            </span>

            <strong className="video-info-value">
              {formatFileSize(file.size)}
            </strong>
          </div>

        </div>

        {/* Duration */}
        <div className="video-info-item">

          <div className="video-info-icon">
            <Clock size={18} />
          </div>

          <div>
            <span className="video-info-label">
              Duration
            </span>

            <strong className="video-info-value">
              {duration === null
                ? "Loading..."
                : formatDuration(duration)}
            </strong>
          </div>

        </div>

        {/* Video Format */}
        <div className="video-info-item">

          <div className="video-info-icon">
            <FileType size={18} />
          </div>

          <div>
            <span className="video-info-label">
              Format
            </span>

            <strong className="video-info-value">
              {fileExtension}
            </strong>
          </div>

        </div>

      </div>

      {/* Preview Information */}
      <div className="preview-information">
        <Video size={16} />

        <span>
          Your video is ready for AI-powered people detection and tracking.
        </span>
      </div>

    </section>
  );
}

export default VideoPreview;
