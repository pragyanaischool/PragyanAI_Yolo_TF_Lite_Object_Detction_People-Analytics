
import React, { useMemo } from "react";

// =========================================================
// HELPERS
// =========================================================

const formatNumber = (value) => {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "0";
  }

  return Math.round(number).toLocaleString();
};

const formatDuration = (seconds) => {
  const value = Number(seconds);

  if (!Number.isFinite(value) || value < 0) {
    return "Calculating...";
  }

  const totalSeconds = Math.round(value);

  if (totalSeconds < 60) {
    return `${totalSeconds} sec`;
  }

  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;

  if (minutes < 60) {
    return remainingSeconds
      ? `${minutes} min ${remainingSeconds} sec`
      : `${minutes} min`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  return `${hours} hr ${remainingMinutes} min`;
};

const formatClockTime = (value) => {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

const getStatusLabel = (status) => {
  const labels = {
    queued: "Queued",
    pending: "Waiting",
    uploading: "Uploading",
    processing: "Processing",
    completed: "Completed",
    failed: "Failed",
    cancelled: "Cancelled",
  };

  return labels[String(status || "").toLowerCase()] || "Initializing";
};

const getStatusClass = (status) => {
  const value = String(status || "").toLowerCase();

  if (value === "completed") return "success";
  if (value === "failed" || value === "cancelled") return "error";
  if (value === "queued" || value === "pending") return "queued";

  return "processing";
};

// =========================================================
// ANALYTICS CARD
// =========================================================

function MetricCard({
  label,
  value,
  description,
  icon,
}) {
  return (
    <div className="pa-metric-card">

      <div className="pa-metric-top">

        <span className="pa-metric-icon">
          {icon}
        </span>

        <span className="pa-metric-label">
          {label}
        </span>

      </div>

      <strong className="pa-metric-value">
        {formatNumber(value)}
      </strong>

      {description && (
        <small className="pa-metric-description">
          {description}
        </small>
      )}

    </div>
  );
}

// =========================================================
// ACTIVITY ITEM
// =========================================================

function ActivityItem({ item, index }) {
  if (typeof item === "string") {
    return (
      <div className="pa-activity-item">
        <span className="pa-activity-dot" />

        <div className="pa-activity-content">
          <span>{item}</span>
        </div>
      </div>
    );
  }

  const frame =
    item?.frame_number ??
    item?.frame ??
    item?.frame_id;

  const timestamp =
    item?.timestamp ??
    item?.time ??
    item?.created_at;

  const people =
    item?.people_count ??
    item?.person_count ??
    item?.people ??
    item?.persons;

  const objects =
    item?.object_count ??
    item?.objects_count ??
    item?.total_objects;

  const trackId =
    item?.track_id ??
    item?.person_id ??
    item?.id;

  const message =
    item?.message ??
    item?.description ??
    item?.event ??
    item?.activity;

  return (
    <div
      className="pa-activity-item"
      key={item?.id ?? index}
    >

      <span className="pa-activity-dot" />

      <div className="pa-activity-content">

        <div className="pa-activity-heading">

          <strong>
            {message ||
              (trackId !== undefined
                ? `Person detected · ID ${trackId}`
                : "Detection update")}
          </strong>

          {timestamp && (
            <time>
              {formatClockTime(timestamp)}
            </time>
          )}

        </div>

        <div className="pa-activity-meta">

          {frame !== undefined && (
            <span>
              Frame: {formatNumber(frame)}
            </span>
          )}

          {people !== undefined && (
            <span>
              People: {formatNumber(people)}
            </span>
          )}

          {objects !== undefined && (
            <span>
              Objects: {formatNumber(objects)}
            </span>
          )}

          {trackId !== undefined && !message && (
            <span>
              Track ID: {String(trackId)}
            </span>
          )}

        </div>

      </div>

    </div>
  );
}

// =========================================================
// LATEST FRAME PREVIEW
// =========================================================

function LatestFramePreview({ frame }) {
  const imageSource = useMemo(() => {
    if (!frame) return null;

    if (typeof frame === "string") {
      return frame;
    }

    if (typeof frame === "object") {
      return (
        frame.image_url ||
        frame.url ||
        frame.data_url ||
        frame.image ||
        frame.frame_image ||
        null
      );
    }

    return null;
  }, [frame]);

  const frameNumber =
    typeof frame === "object"
      ? frame.frame_number ?? frame.frame ?? null
      : null;

  if (!imageSource) {
    return (
      <div className="pa-frame-empty">

        <div className="pa-frame-empty-icon">
          ▣
        </div>

        <strong>
          Waiting for frame preview
        </strong>

        <span>
          The latest processed frame will appear here
          when the backend publishes an image.
        </span>

      </div>
    );
  }

  return (
    <div className="pa-frame-preview">

      <div className="pa-frame-header">

        <strong>
          Latest processed frame
        </strong>

        {frameNumber !== null && (
          <span>
            Frame {formatNumber(frameNumber)}
          </span>
        )}

      </div>

      <img
        src={imageSource}
        alt="Latest YOLO object detection frame"
        className="pa-frame-image"
        loading="lazy"
        onError={(event) => {
          event.currentTarget.style.display = "none";
        }}
      />

    </div>
  );
}

// =========================================================
// MAIN PROCESSING STATUS COMPONENT
// =========================================================

export default function ProcessingStatus({
  job,
  isPolling = false,
  onStopPolling,
  showActivity = true,
}) {
  if (!job) {
    return null;
  }

  // -------------------------------------------------------
  // JOB STATUS
  // -------------------------------------------------------

  const status = String(
    job.status || "initializing"
  ).toLowerCase();

  const statusLabel = getStatusLabel(status);
  const statusClass = getStatusClass(status);

  const isCompleted = status === "completed";

  const isFailed =
    status === "failed" ||
    status === "cancelled";

  const isActive =
    status === "processing" ||
    status === "queued" ||
    status === "pending" ||
    status === "uploading";

  // -------------------------------------------------------
  // PROGRESS
  // -------------------------------------------------------

  const progressValue = Number(job.progress);

  const progress = Math.max(
    0,
    Math.min(
      100,
      Number.isFinite(progressValue)
        ? progressValue
        : isCompleted
          ? 100
          : 0
    )
  );

  // -------------------------------------------------------
  // FRAME INFORMATION
  // -------------------------------------------------------

  const currentFrame = Number(
    job.current_frame ??
    job.frames_processed ??
    job.processed_frames ??
    0
  );

  const totalFrames = Number(
    job.total_frames ??
    job.frame_count ??
    0
  );

  const remainingFrames = Math.max(
    0,
    totalFrames - currentFrame
  );

  // -------------------------------------------------------
  // PEOPLE / OBJECT COUNTS
  // -------------------------------------------------------

  const currentPeople =
    job.current_people ??
    job.people_count ??
    job.current_person_count ??
    0;

  const currentObjects =
    job.current_objects ??
    job.objects_count ??
    job.object_count ??
    0;

  const uniquePeople =
    job.unique_people_so_far ??
    job.unique_people ??
    job.total_unique_people ??
    0;

  // -------------------------------------------------------
  // ETA
  // -------------------------------------------------------

  const eta =
    job.eta_seconds ??
    job.estimated_remaining_seconds ??
    job.remaining_seconds ??
    null;

  // -------------------------------------------------------
  // ACTIVITY
  // -------------------------------------------------------

  const activity = Array.isArray(job.recent_activity)
    ? job.recent_activity
    : Array.isArray(job.activity)
      ? job.activity
      : [];

  // -------------------------------------------------------
  // LATEST FRAME
  // -------------------------------------------------------

  const latestFrame =
    job.latest_frame ||
    job.preview_frame ||
    job.current_frame_image ||
    null;

  // -------------------------------------------------------
  // JOB ID
  // -------------------------------------------------------

  const jobId =
    job.job_id ||
    job.id ||
    "—";

  // -------------------------------------------------------
  // RENDER
  // -------------------------------------------------------

  return (
    <section className="pa-panel pa-processing-panel">

      {/* HEADER */}
      <div className="pa-section-title">

        <div>

          <span className="pa-kicker">
            LIVE VIDEO ANALYSIS
          </span>

          <h2>
            Processing dashboard
          </h2>

          <p>
            YOLO11 detection and ByteTrack tracking
            progress.
          </p>

        </div>

        <span
          className={`pa-job-status is-${statusClass}`}
        >

          <span className="pa-status-dot" />

          {statusLabel}

        </span>

      </div>

      {/* JOB INFORMATION */}
      <div className="pa-job-information">

        <div>
          <span>Job ID</span>

          <strong title={jobId}>
            {jobId}
          </strong>
        </div>

        <div>
          <span>Started</span>

          <strong>
            {formatClockTime(
              job.started_at ||
              job.created_at
            )}
          </strong>
        </div>

        <div>
          <span>Frame interval</span>

          <strong>
            {job.frame_interval ?? 2}
          </strong>
        </div>

        <div>
          <span>Image size</span>

          <strong>
            {job.imgsz || job.image_size || 320}
          </strong>
        </div>

      </div>

      {/* PROGRESS SECTION */}
      <div className="pa-progress-section">

        <div className="pa-progress-heading">

          <div>
            <span>
              Overall progress
            </span>

            <strong>
              {progress.toFixed(1)}%
            </strong>
          </div>

          <span className="pa-progress-status">
            {isCompleted
              ? "Processing complete"
              : isFailed
                ? "Processing stopped"
                : "Detection in progress"}
          </span>

        </div>

        <div
          className="pa-progress-track"
          role="progressbar"
          aria-label="Video processing progress"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
        >

          <div
            className={`pa-progress-fill ${
              isFailed ? "is-failed" : ""
            }`}
            style={{
              width: `${progress}%`,
            }}
          />

        </div>

        <div className="pa-progress-footer">

          <span>
            {formatNumber(currentFrame)} frames processed
          </span>

          <span>
            {totalFrames > 0
              ? `${formatNumber(totalFrames)} total frames`
              : "Total frames being calculated"}
          </span>

        </div>

      </div>

      {/* FRAME PROCESSING DETAILS */}
      <div className="pa-processing-details">

        <div className="pa-detail-item">

          <span>
            Current frame
          </span>

          <strong>
            {formatNumber(currentFrame)}
          </strong>

        </div>

        <div className="pa-detail-item">

          <span>
            Remaining frames
          </span>

          <strong>
            {formatNumber(remainingFrames)}
          </strong>

        </div>

        <div className="pa-detail-item">

          <span>
            Estimated time remaining
          </span>

          <strong>
            {isCompleted
              ? "Completed"
              : formatDuration(eta)}
          </strong>

        </div>

      </div>

      {/* LIVE ANALYTICS */}
      <div className="pa-live-analytics">

        <div className="pa-subsection-heading">

          <div>
            <h3>
              Detection analytics
            </h3>

            <p>
              Latest available detection statistics
            </p>
          </div>

          {isPolling && (
            <span className="pa-live-indicator">
              <span className="pa-status-dot" />
              LIVE
            </span>
          )}

        </div>

        <div className="pa-metrics-grid">

          <MetricCard
            label="Current people"
            value={currentPeople}
            description="People in the latest processed frame"
            icon="●"
          />

          <MetricCard
            label="Unique people"
            value={uniquePeople}
            description="Unique tracked identities so far"
            icon="◉"
          />

          <MetricCard
            label="Current objects"
            value={currentObjects}
            description="Objects detected in the latest frame"
            icon="▣"
          />

          <MetricCard
            label="Frames processed"
            value={currentFrame}
            description="Frames analysed by the backend"
            icon="▤"
          />

        </div>

      </div>

      {/* LATEST FRAME AND ACTIVITY */}
      <div className="pa-live-grid">

        <div className="pa-live-frame">

          <LatestFramePreview
            frame={latestFrame}
          />

        </div>

        {showActivity && (

          <div className="pa-activity-panel">

            <div className="pa-subsection-heading">

              <div>
                <h3>
                  Recent activity
                </h3>

                <p>
                  Latest detection events
                </p>
              </div>

              <span className="pa-activity-count">
                {activity.length}
              </span>

            </div>

            <div className="pa-activity-list">

              {activity.length > 0 ? (

                activity
                  .slice(-20)
                  .reverse()
                  .map((item, index) => (
                    <ActivityItem
                      key={
                        item?.id ??
                        item?.event_id ??
                        `${index}-${item?.frame_number ?? ""}`
                      }
                      item={item}
                      index={index}
                    />
                  ))

              ) : (

                <div className="pa-activity-empty">

                  <span>
                    No detection activity available yet.
                  </span>

                  <small>
                    Events will appear here as processing continues.
                  </small>

                </div>

              )}

            </div>

          </div>

        )}

      </div>

      {/* ERROR INFORMATION */}
      {isFailed && (

        <div className="pa-error">

          <strong>
            Processing {status}
          </strong>

          <p>
            {job.error ||
              job.message ||
              "The backend could not complete this job."}
          </p>

        </div>

      )}

      {/* COMPLETION INFORMATION */}
      {isCompleted && (

        <div className="pa-completion-message">

          <strong>
            ✓ Video processing completed
          </strong>

          <p>
            Your processed video and analytics
            are ready for download.
          </p>

        </div>

      )}

      {/* STOP POLLING */}
      {isActive && isPolling && onStopPolling && (

        <div className="pa-polling-controls">

          <span>
            Browser is checking job status automatically.
          </span>

          <button
            type="button"
            className="pa-text-button"
            onClick={onStopPolling}
          >
            Stop status updates
          </button>

        </div>

      )}

      {isActive && !isPolling && (

        <div className="pa-polling-paused">
          Status polling is stopped. The backend job
          may still be processing.
        </div>

      )}

    </section>
  );
}
