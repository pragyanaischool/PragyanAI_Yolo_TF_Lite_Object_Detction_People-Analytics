import React from "react";

const formatSeconds = (value) => {
  if (value === null || value === undefined || value === "") return "—";
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(2)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
};

const formatTimestamp = (value) => {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? `${seconds.toFixed(2)}s` : "—";
};

function MetricCard({ label, value, helper, accent }) {
  return (
    <article className={`pa-metric-card ${accent ? "pa-metric-accent" : ""}`}>
      <span className="pa-metric-label">{label}</span>
      <strong className="pa-metric-value">{value ?? "—"}</strong>
      {helper ? <span className="pa-metric-helper">{helper}</span> : null}
    </article>
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
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  const status = String(job.status || "queued").toLowerCase();
  const active = status === "queued" || status === "processing";
  const currentFrame = job.current_frame ?? latest?.frame_number ?? 0;
  const totalFrames = job.total_frames ?? latest?.total_frames ?? "—";
  const currentPeople = job.current_people ?? latest?.people_found ?? 0;
  const currentObjects = job.current_objects ?? latest?.objects_found ?? 0;
  const uniquePeople = job.unique_people_so_far ?? latest?.unique_people_so_far ?? 0;
  const trackIds = Array.isArray(latest?.track_ids) ? latest.track_ids : [];

  return (
    <section className="pa-panel pa-live-panel" aria-live="polite">
      <div className="pa-section-heading">
        <div>
          <span className="pa-kicker">LIVE JOB MONITOR</span>
          <h2>Processing status</h2>
          <p>{job.message || "Waiting for processing updates from the backend…"}</p>
        </div>
        <span className={`pa-job-status status-${status}`}>
          <span className="pa-status-dot" />
          {status}
        </span>
      </div>

      <div className="pa-progress-heading">
        <strong>Frame {currentFrame} <span>/ {totalFrames}</span></strong>
        <strong>{progress.toFixed(1)}%</strong>
      </div>
      <div
        className={`pa-progress-track ${active ? "is-animated" : ""}`}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progress}
        aria-label="Video processing progress"
      >
        <div className="pa-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="pa-progress-meta">
        <span>Elapsed: {formatSeconds(job.processing_seconds)}</span>
        <span>ETA: {formatSeconds(job.eta_seconds)}</span>
      </div>

      {active && (
        <div className="pa-processing-pulse">
          <span className="pa-pulse-dot" />
          {isPolling ? "Receiving live frame updates" : "Job is running; status polling is paused"}
        </div>
      )}

      <div className="pa-metrics-grid">
        <MetricCard label="People in current frame" value={currentPeople} helper="Person class detections" accent />
        <MetricCard label="Objects in current frame" value={currentObjects} helper="All detected classes" />
        <MetricCard label="Unique people tracked" value={uniquePeople} helper="Distinct ByteTrack IDs" />
        <MetricCard label="Current frame time" value={latest ? formatSeconds(latest.frame_seconds) : "—"} helper="Read + inference + write" />
      </div>

      <div className="pa-latest-frame">
        <div className="pa-section-heading">
          <h3>Latest frame details</h3>
          <span className="pa-frame-chip">{latest ? `Frame ${latest.frame_number}` : "Awaiting first frame"}</span>
        </div>

        {latest ? (
          <>
            <div className="pa-latest-grid">
              <div><span>Frame position</span><strong>{latest.frame_number ?? "—"} / {latest.total_frames ?? totalFrames}</strong></div>
              <div><span>Video timestamp</span><strong>{formatTimestamp(latest.timestamp_seconds)}</strong></div>
              <div><span>Inference status</span><strong>{latest.inference_performed ? "YOLO inference" : "Carried forward"}</strong></div>
              <div><span>Inference time</span><strong>{formatSeconds(latest.inference_seconds)}</strong></div>
              <div><span>Frame processing time</span><strong>{formatSeconds(latest.frame_seconds)}</strong></div>
              <div><span>Frame result</span><strong>{latest.status || "—"}</strong></div>
            </div>
            <div className="pa-track-ids">
              <strong>Track IDs in current frame</strong>
              <div className="pa-id-list">
                {trackIds.length
                  ? trackIds.map((id) => <span className="pa-id-chip" key={String(id)}>ID {id}</span>)
                  : <span className="pa-muted">No tracked people in this frame</span>}
              </div>
            </div>
          </>
        ) : (
          <div className="pa-empty-state">
            <span className="pa-spinner" />
            <p>Waiting for the first frame update from YOLO11…</p>
          </div>
        )}
      </div>

      {showActivity && (
        <div className="pa-activity">
          <div className="pa-section-heading">
            <div><h3>Recent frame activity</h3><p>Latest frame-level updates received from FastAPI.</p></div>
            <span className="pa-update-count">{recentActivity.length} updates</span>
          </div>
          {recentActivity.length ? (
            <div className="pa-table-wrap">
              <table className="pa-activity-table">
                <thead>
                  <tr>
                    <th>Frame</th><th>Timestamp</th><th>People</th><th>Objects</th>
                    <th>Track IDs</th><th>Inference</th><th>Frame time</th><th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {recentActivity.map((item, index) => (
                    <tr key={`${item.frame_number ?? "frame"}-${index}`}>
                      <td className="pa-frame-number">{item.frame_number ?? "—"}</td>
                      <td>{formatTimestamp(item.timestamp_seconds)}</td>
                      <td><strong>{item.people_found ?? 0}</strong></td>
                      <td>{item.objects_found ?? 0}</td>
                      <td>{Array.isArray(item.track_ids) && item.track_ids.length ? item.track_ids.join(", ") : "—"}</td>
                      <td><span className={`pa-inference-tag ${item.inference_performed ? "is-done" : "is-skip"}`}>{item.inference_performed ? "Yes" : "Skipped"}</span></td>
                      <td>{formatSeconds(item.frame_seconds)}</td>
                      <td>{item.status || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="pa-empty-state"><p>Frame updates will appear here as video processing continues.</p></div>
          )}
        </div>
      )}

      {active && isPolling && typeof onStopPolling === "function" && (
        <button type="button" className="pa-secondary-button" onClick={onStopPolling}>
          Stop browser status polling
        </button>
      )}

      {status === "failed" && (
        <div className="pa-error" role="alert">{job.error || job.message || "Video processing failed."}</div>
      )}
      {status === "completed" && (
        <div className="pa-success-message">Video processing completed successfully.</div>
      )}
    </section>
  );
}
