
import React from "react";

function formatSeconds(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "—";

  if (seconds < 60) return `${seconds.toFixed(2)}s`;

  const minutes = Math.floor(seconds / 60);
  const remaining = Math.floor(seconds % 60);
  return `${minutes}m ${remaining}s`;
}

function formatTimestamp(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const seconds = Number(value);
  return Number.isFinite(seconds) ? `${seconds.toFixed(2)}s` : "—";
}

function MetricCard({ label, value, helper }) {
  return (
    <div className="pa-metric-card">
      <div className="pa-metric-value">{value ?? "—"}</div>
      <div className="pa-metric-label">{label}</div>
      {helper ? <div className="pa-metric-helper">{helper}</div> : null}
    </div>
  );
}

export default function ProcessingStatus({
  job,
  isPolling = false,
  onStopPolling,
  showActivity = true,
}) {
  if (!job) return null;

  const latest = job.latest_frame || null;

  const recentActivity = Array.isArray(job.recent_activity)
    ? [...job.recent_activity].reverse()
    : [];

  const progress = Math.max(
    0,
    Math.min(100, Number(job.progress || 0))
  );

  const currentFrame = job.current_frame ?? latest?.frame_number ?? 0;
  const totalFrames = job.total_frames ?? latest?.total_frames ?? "—";

  const currentPeople =
    job.current_people ?? latest?.people_found ?? 0;

  const currentObjects =
    job.current_objects ?? latest?.objects_found ?? 0;

  const uniquePeople =
    job.unique_people_so_far ??
    latest?.unique_people_so_far ??
    0;

  const status = String(job.status || "queued").toLowerCase();

  const isActive = status === "queued" || status === "processing";

  return (
    <section className="pa-panel pa-live-panel" aria-live="polite">

      <div className="pa-section-heading">
        <div>
          <h2>Live processing activity</h2>
          <p>
            {job.message ||
              "Frame-by-frame detection updates from the backend."}
          </p>
        </div>

        <span className={`pa-job-status status-${status}`}>
          {status}
        </span>
      </div>

      {/* Processing progress */}

      <div className="pa-progress-heading">
        <strong>
          {currentFrame} / {totalFrames} frames
        </strong>

        <strong>{progress.toFixed(1)}%</strong>
      </div>

      <div
        className="pa-progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progress}
        aria-label="Video processing progress"
      >
        <div
          className="pa-progress-fill"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="pa-progress-meta">
        <span>
          Elapsed: {formatSeconds(job.processing_seconds)}
        </span>

        <span>
          Estimated remaining: {formatSeconds(job.eta_seconds)}
        </span>
      </div>

      {/* Live detection metrics */}

      <div className="pa-metrics-grid">
        <MetricCard
          label="People in current frame"
          value={currentPeople}
        />

        <MetricCard
          label="Objects in current frame"
          value={currentObjects}
        />

        <MetricCard
          label="Unique people tracked"
          value={uniquePeople}
        />

        <MetricCard
          label="Latest frame time"
          value={
            latest && latest.frame_seconds !== undefined
              ? `${Number(latest.frame_seconds).toFixed(2)}s`
              : "—"
          }
        />
      </div>

      {/* Latest processed frame */}

      {latest ? (
        <div className="pa-latest-frame">

          <h3>Latest frame result</h3>

          <div className="pa-latest-grid">

            <div>
              <span>Frame</span>
              <strong>
                {latest.frame_number ?? "—"} /{" "}
                {latest.total_frames ?? totalFrames}
              </strong>
            </div>

            <div>
              <span>Video timestamp</span>
              <strong>
                {formatTimestamp(latest.timestamp_seconds)}
              </strong>
            </div>

            <div>
              <span>Inference</span>
              <strong>
                {latest.inference_performed
                  ? "Performed"
                  : "Skipped / carried forward"}
              </strong>
            </div>

            <div>
              <span>YOLO inference time</span>
              <strong>
                {formatSeconds(latest.inference_seconds)}
              </strong>
            </div>

            <div>
              <span>Frame processing time</span>
              <strong>
                {formatSeconds(latest.frame_seconds)}
              </strong>
            </div>

            <div>
              <span>Frame status</span>
              <strong>{latest.status || "—"}</strong>
            </div>

          </div>

          <p className="pa-track-ids">
            <strong>Track IDs:</strong>{" "}
            {Array.isArray(latest.track_ids) &&
            latest.track_ids.length
              ? latest.track_ids.join(", ")
              : "No tracked people in this frame"}
          </p>

        </div>
      ) : (
        <p className="pa-muted">
          Waiting for the first frame update from the video processor…
        </p>
      )}

      {/* Recent frame activity */}

      {showActivity ? (
        <div className="pa-activity">

          <div className="pa-section-heading">
            <h3>Recent frame activity</h3>
            <span>{recentActivity.length} updates</span>
          </div>

          {recentActivity.length > 0 ? (
            <div className="pa-table-wrap">

              <table>
                <thead>
                  <tr>
                    <th>Frame</th>
                    <th>Timestamp</th>
                    <th>People</th>
                    <th>Objects</th>
                    <th>Track IDs</th>
                    <th>Inference</th>
                    <th>Frame time</th>
                    <th>Status</th>
                  </tr>
                </thead>

                <tbody>
                  {recentActivity.map((item, index) => (
                    <tr
                      key={`${item.frame_number ?? "frame"}-${index}`}
                    >
                      <td>{item.frame_number ?? "—"}</td>

                      <td>
                        {formatTimestamp(item.timestamp_seconds)}
                      </td>

                      <td>{item.people_found ?? 0}</td>

                      <td>{item.objects_found ?? 0}</td>

                      <td>
                        {Array.isArray(item.track_ids) &&
                        item.track_ids.length
                          ? item.track_ids.join(", ")
                          : "—"}
                      </td>

                      <td>
                        {item.inference_performed ? "Yes" : "No"}
                      </td>

                      <td>
                        {formatSeconds(item.frame_seconds)}
                      </td>

                      <td>{item.status || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

            </div>
          ) : (
            <p className="pa-muted">
              Frame updates will appear here as processing continues.
            </p>
          )}

        </div>
      ) : null}

      {/* Stop browser-side polling */}

      {isActive &&
      isPolling &&
      typeof onStopPolling === "function" ? (
        <button
          type="button"
          className="pa-secondary-button"
          onClick={onStopPolling}
        >
          Stop checking progress
        </button>
      ) : null}

      {status === "failed" ? (
        <div className="pa-error" role="alert">
          {job.error || job.message || "Video processing failed."}
        </div>
      ) : null}

      {status === "completed" ? (
        <p className="pa-success-message">
          Video processing completed successfully.
        </p>
      ) : null}

    </section>
  );
}
