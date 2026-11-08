
import React from "react";

import {
  CheckCircle,
  Clock,
  LoaderCircle,
  AlertCircle,
  Video,
  BrainCircuit,
  Activity,
  Cpu
} from "lucide-react";

// --------------------------------------------------
// Processing Status Component
// --------------------------------------------------

function ProcessingStatus({
  status = "idle",
  progress = 0,
  error = ""
}) {

  // ------------------------------------------------
  // Normalize Progress
  // ------------------------------------------------

  const safeProgress = Math.min(
    100,
    Math.max(0, Number(progress) || 0)
  );

  // ------------------------------------------------
  // Status Configuration
  // ------------------------------------------------

  const statusInformation = {

    idle: {
      label: "Waiting",
      message: "Upload a video to begin processing.",
      icon: <Clock size={20} />,
      className: "idle"
    },

    processing: {
      label: "Processing",
      message: "AI is detecting and tracking people.",
      icon: (
        <LoaderCircle
          size={20}
          className="rotating-icon"
        />
      ),
      className: "processing"
    },

    completed: {
      label: "Completed",
      message: "Video processing completed successfully.",
      icon: <CheckCircle size={20} />,
      className: "completed"
    },

    failed: {
      label: "Failed",
      message: error || "Video processing failed.",
      icon: <AlertCircle size={20} />,
      className: "failed"
    }

  };

  const currentStatus =
    statusInformation[status] || statusInformation.idle;

  // ------------------------------------------------
  // Render
  // ------------------------------------------------

  return (
    <section className="status-card">

      {/* -------------------------------------------
          Status Header
      -------------------------------------------- */}

      <div className="status-top">

        <div className="status-title">
          <Video size={17} />
          Processing Activity
        </div>

        <span
          className={`status-label ${currentStatus.className}`}
        >
          {currentStatus.label}
        </span>

      </div>

      {/* -------------------------------------------
          Status Message
      -------------------------------------------- */}

      <div
        className={`status-message ${currentStatus.className}`}
        role="status"
        aria-live="polite"
      >

        {currentStatus.icon}

        <p>
          {currentStatus.message}
        </p>

      </div>

      {/* -------------------------------------------
          Progress Section
      -------------------------------------------- */}

      <div className="progress-section">

        <div className="progress-text">

          <span>
            Processing Progress
          </span>

          <strong>
            {safeProgress.toFixed(0)}%
          </strong>

        </div>

        <div
          className="progress-track"
          role="progressbar"
          aria-valuenow={safeProgress}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Video processing progress"
        >

          <div
            className="progress-bar"
            style={{
              width: `${safeProgress}%`
            }}
          />

        </div>

        <div className="progress-caption">

          {status === "idle" && (
            <span>
              Waiting for video upload
            </span>
          )}

          {status === "processing" && (
            <span>
              Please wait while your video is being analyzed.
            </span>
          )}

          {status === "completed" && (
            <span className="success-text">
              Processing finished successfully.
            </span>
          )}

          {status === "failed" && (
            <span className="error-text">
              Processing could not be completed.
            </span>
          )}

        </div>

      </div>

      {/* -------------------------------------------
          Processing Information
      -------------------------------------------- */}

      <div className="status-details">

        <div className="status-detail-row">

          <div className="status-detail-label">
            <BrainCircuit size={15} />
            <span>Detection Engine</span>
          </div>

          <strong>
            YOLO11
          </strong>

        </div>

        <div className="status-detail-row">

          <div className="status-detail-label">
            <Activity size={15} />
            <span>Tracking Engine</span>
          </div>

          <strong>
            ByteTrack
          </strong>

        </div>

        <div className="status-detail-row">

          <div className="status-detail-label">
            <Cpu size={15} />
            <span>Inference Runtime</span>
          </div>

          <strong>
            TensorFlow Lite
          </strong>

        </div>

      </div>

      {/* -------------------------------------------
          Error Information
      -------------------------------------------- */}

      {status === "failed" && error && (

        <div className="status-error" role="alert">

          <AlertCircle size={17} />

          <div>
            <strong>Processing Error</strong>

            <p>
              {error}
            </p>
          </div>

        </div>

      )}

      {/* -------------------------------------------
          Completion Message
      -------------------------------------------- */}

      {status === "completed" && (

        <div className="status-success">

          <CheckCircle size={17} />

          <span>
            Your processed video and analytics are ready.
          </span>

        </div>

      )}

    </section>
  );
}

export default ProcessingStatus;
