
import React, { useCallback, useEffect, useRef, useState } from "react";
import { API_CONFIG, healthCheck } from "./services/api";
import "./style.css";

const API_BASE_URL = API_CONFIG.BASE_URL;
const JOBS_URL = `${API_BASE_URL}/api/jobs`;

async function readResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      typeof data === "string"
        ? data
        : data?.detail?.message ||
          data?.detail ||
          data?.message ||
          `HTTP ${response.status}`;

    throw new Error(message);
  }

  return data;
}

function formatSeconds(value) {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(Number(value))
  ) {
    return "—";
  }

  const seconds = Math.max(0, Math.round(Number(value)));

  if (seconds < 60) return `${seconds}s`;

  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function formatTimestamp(value) {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(Number(value))
  ) {
    return "—";
  }

  return `${Number(value).toFixed(2)}s`;
}

function Metric({ label, value, helper }) {
  return (
    <div className="pa-metric-card">
      <div className="pa-metric-value">{value ?? "—"}</div>
      <div className="pa-metric-label">{label}</div>
      {helper ? (
        <div className="pa-metric-helper">{helper}</div>
      ) : null}
    </div>
  );
}

export default function App() {
  const [backend, setBackend] = useState({
    checking: true,
    connected: false,
    model_loaded: false,
  });

  const [file, setFile] = useState(null);
  const [confidence, setConfidence] = useState(0.35);
  const [frameInterval, setFrameInterval] = useState(1);

  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [isPolling, setIsPolling] = useState(false);

  const pollTimer = useRef(null);
  const activeJobId = useRef(null);
  const mounted = useRef(true);

  // --------------------------------------------------
  // Backend health check
  // --------------------------------------------------

  const refreshBackend = useCallback(async () => {
    setBackend((current) => ({
      ...current,
      checking: true,
    }));

    try {
      const status = await healthCheck();

      if (mounted.current) {
        setBackend({
          ...status,
          checking: false,
        });
      }
    } catch (healthError) {
      if (mounted.current) {
        setBackend({
          checking: false,
          connected: false,
          model_loaded: false,
          error: healthError.message,
        });
      }
    }
  }, []);

  useEffect(() => {
    mounted.current = true;

    refreshBackend();

    return () => {
      mounted.current = false;

      if (pollTimer.current) {
        clearTimeout(pollTimer.current);
      }
    };
  }, [refreshBackend]);

  // --------------------------------------------------
  // Fetch job status
  // --------------------------------------------------

  const fetchJobStatus = useCallback(async (jobId) => {
    const response = await fetch(
      `${JOBS_URL}/${encodeURIComponent(jobId)}`,
      {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
        cache: "no-store",
      }
    );

    return readResponse(response);
  }, []);

  // --------------------------------------------------
  // Poll backend for live frame updates
  // --------------------------------------------------

  const pollJob = useCallback(
    async (jobId) => {
      if (
        !mounted.current ||
        activeJobId.current !== jobId
      ) {
        return;
      }

      setIsPolling(true);

      try {
        const data = await fetchJobStatus(jobId);

        if (
          !mounted.current ||
          activeJobId.current !== jobId
        ) {
          return;
        }

        // Update live progress and frame results.
        setJob(data);

        // Job completed.
        if (data.status === "completed") {
          setResult(data.result || null);
          setIsPolling(false);
          activeJobId.current = null;
          setError("");
          return;
        }

        // Job failed.
        if (
          data.status === "failed" ||
          data.status === "cancelled"
        ) {
          setError(
            data.error ||
              data.message ||
              "Video processing failed."
          );

          setIsPolling(false);
          activeJobId.current = null;
          return;
        }

        // Continue polling.
        pollTimer.current = setTimeout(
          () => pollJob(jobId),
          3000
        );
      } catch (pollError) {
        if (
          !mounted.current ||
          activeJobId.current !== jobId
        ) {
          return;
        }

        // A temporary polling failure does not mean
        // the background video-processing job failed.
        setError(
          `Unable to retrieve the latest job status: ${pollError.message}. Retrying...`
        );

        pollTimer.current = setTimeout(
          () => pollJob(jobId),
          5000
        );
      }
    },
    [fetchJobStatus]
  );

  // --------------------------------------------------
  // Upload video and start processing
  // --------------------------------------------------

  const startProcessing = async (event) => {
    event.preventDefault();

    if (!file) {
      setError("Please select a video file first.");
      return;
    }

    if (pollTimer.current) {
      clearTimeout(pollTimer.current);
    }

    activeJobId.current = null;

    setSubmitting(true);
    setIsPolling(false);
    setError("");
    setJob(null);
    setResult(null);

    try {
      const formData = new FormData();

      formData.append("file", file);
      formData.append(
        "confidence",
        String(confidence)
      );
      formData.append(
        "frame_interval",
        String(frameInterval)
      );

      const response = await fetch(JOBS_URL, {
        method: "POST",
        body: formData,
      });

      const created = await readResponse(response);

      if (!created.job_id) {
        throw new Error(
          "Backend did not return a job ID."
        );
      }

      activeJobId.current = created.job_id;

      setJob({
        ...created,
        status: created.status || "queued",
        latest_frame: null,
        recent_activity: [],
        current_people: 0,
        current_objects: 0,
        unique_people_so_far: 0,
      });

      setSubmitting(false);

      // Start live progress polling.
      pollJob(created.job_id);
    } catch (submitError) {
      setError(
        submitError.message ||
          "Unable to upload the video."
      );

      setSubmitting(false);
    }
  };

  // --------------------------------------------------
  // Stop browser polling
  // --------------------------------------------------

  const stopLocalPolling = () => {
    if (pollTimer.current) {
      clearTimeout(pollTimer.current);
    }

    activeJobId.current = null;

    setIsPolling(false);

    setError(
      "Progress checking stopped in this browser. The backend job may still be running."
    );
  };

  // --------------------------------------------------
  // Live job data
  // --------------------------------------------------

  const latest = job?.latest_frame || null;

  const activity = Array.isArray(job?.recent_activity)
    ? [...job.recent_activity].reverse()
    : [];

  const progress = Math.max(
    0,
    Math.min(100, Number(job?.progress || 0))
  );

  const isActive = [
    "queued",
    "processing",
  ].includes(job?.status);

  const completedResult =
    result || job?.result || null;

  // --------------------------------------------------
  // Output URL helper
  // --------------------------------------------------

  const outputUrl = (url) => {
    if (!url) return "#";

    if (/^https?:\/\//i.test(url)) {
      return url;
    }

    return `${API_BASE_URL}${
      url.startsWith("/") ? "" : "/"
    }${url}`;
  };

  // --------------------------------------------------
  // Main application UI
  // --------------------------------------------------

  return (
    <main className="pa-app">

      {/* HEADER */}

      <header className="pa-header">
        <div>
          <div className="pa-eyebrow">
            PRAGYANAI · COMPUTER VISION
          </div>

          <h1>
            People Detection &amp; Counting
          </h1>

          <p>
            YOLO11 TFLite · ByteTrack · TensorFlow Lite
          </p>
        </div>

        <div
          className={`pa-connection ${
            backend.connected
              ? "is-online"
              : "is-offline"
          }`}
        >
          <span className="pa-status-dot" />

          {backend.checking
            ? "Checking backend"
            : backend.connected
            ? "Backend connected"
            : "Backend disconnected"}

          <button
            type="button"
            className="pa-small-button"
            onClick={refreshBackend}
          >
            Refresh
          </button>
        </div>
      </header>

      {/* VIDEO UPLOAD */}

      <section className="pa-panel">
        <h2>Upload video</h2>

        <form
          onSubmit={startProcessing}
          className="pa-form"
        >
          <label className="pa-file-label">
            <span>
              Select video (MP4, AVI, MOV, MKV, WEBM)
            </span>

            <input
              type="file"
              accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm"
              onChange={(event) =>
                setFile(
                  event.target.files?.[0] || null
                )
              }
            />
          </label>

          {file ? (
            <div className="pa-file-name">
              {file.name} ·{" "}
              {(file.size / (1024 * 1024)).toFixed(2)} MB
            </div>
          ) : null}

          <div className="pa-settings-grid">

            <label>
              Confidence threshold:{" "}
              <strong>
                {confidence.toFixed(2)}
              </strong>

              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={confidence}
                onChange={(event) =>
                  setConfidence(
                    Number(event.target.value)
                  )
                }
              />
            </label>

            <label>
              Frame interval

              <select
                value={frameInterval}
                onChange={(event) =>
                  setFrameInterval(
                    Number(event.target.value)
                  )
                }
              >
                <option value={1}>
                  1 – Process every frame
                </option>

                <option value={2}>
                  2 – Every 2nd frame
                </option>

                <option value={3}>
                  3 – Every 3rd frame
                </option>

                <option value={5}>
                  5 – Every 5th frame
                </option>
              </select>
            </label>

          </div>

          <button
            className="pa-primary-button"
            type="submit"
            disabled={
              !file ||
              submitting ||
              isActive
            }
          >
            {submitting
              ? "Uploading video…"
              : isActive
              ? "Video processing in progress"
              : "Upload & Start Detection"}
          </button>
        </form>

        {!backend.model_loaded &&
        !backend.checking ? (
          <p className="pa-warning">
            The backend is reachable, but the detection
            model is not confirmed as loaded. Check the
            backend health endpoint.
          </p>
        ) : null}
      </section>

      {/* ERROR MESSAGE */}

      {error ? (
        <div className="pa-error" role="alert">
          {error}
        </div>
      ) : null}

      {/* LIVE PROCESSING DASHBOARD */}

      {job ? (
        <section className="pa-panel pa-live-panel">

          <div className="pa-section-heading">
            <div>
              <h2>Live processing activity</h2>

              <p>
                {job.message ||
                  "Waiting for processing updates…"}
              </p>
            </div>

            <span
              className={`pa-job-status status-${
                job.status || "queued"
              }`}
            >
              {job.status || "queued"}
            </span>
          </div>

          {/* PROGRESS */}

          <div className="pa-progress-heading">
            <strong>
              {job.current_frame ?? 0} /{" "}
              {job.total_frames ?? "—"} frames
            </strong>

            <strong>
              {progress.toFixed(1)}%
            </strong>
          </div>

          <div className="pa-progress-track">
            <div
              className="pa-progress-fill"
              style={{
                width: `${progress}%`,
              }}
            />
          </div>

          <div className="pa-progress-meta">
            <span>
              Elapsed:{" "}
              {formatSeconds(
                job.processing_seconds
              )}
            </span>

            <span>
              Estimated remaining:{" "}
              {formatSeconds(job.eta_seconds)}
            </span>
          </div>

          {/* LIVE METRICS */}

          <div className="pa-metrics-grid">

            <Metric
              label="People in current frame"
              value={
                job.current_people ??
                latest?.people_found ??
                0
              }
            />

            <Metric
              label="Objects in current frame"
              value={
                job.current_objects ??
                latest?.objects_found ??
                0
              }
            />

            <Metric
              label="Unique people tracked"
              value={
                job.unique_people_so_far ??
                latest?.unique_people_so_far ??
                0
              }
            />

            <Metric
              label="Latest frame time"
              value={
                latest
                  ? `${Number(
                      latest.frame_seconds || 0
                    ).toFixed(2)}s`
                  : "—"
              }
            />

          </div>

          {/* LATEST FRAME */}

          {latest ? (
            <div className="pa-latest-frame">

              <h3>Latest frame result</h3>

              <div className="pa-latest-grid">

                <div>
                  <span>Frame</span>
                  <strong>
                    {latest.frame_number} /{" "}
                    {latest.total_frames}
                  </strong>
                </div>

                <div>
                  <span>Video timestamp</span>
                  <strong>
                    {formatTimestamp(
                      latest.timestamp_seconds
                    )}
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
                    {Number(
                      latest.inference_seconds || 0
                    ).toFixed(3)}
                    s
                  </strong>
                </div>

                <div>
                  <span>Frame processing time</span>
                  <strong>
                    {Number(
                      latest.frame_seconds || 0
                    ).toFixed(3)}
                    s
                  </strong>
                </div>

                <div>
                  <span>Frame status</span>
                  <strong>
                    {latest.status || "—"}
                  </strong>
                </div>

              </div>

              <p className="pa-track-ids">
                <strong>Track IDs:</strong>{" "}
                {latest.track_ids?.length
                  ? latest.track_ids.join(", ")
                  : "No tracked people in this frame"}
              </p>

            </div>
          ) : (
            <p className="pa-muted">
              Waiting for the first frame update from YOLO11…
            </p>
          )}

          {/* RECENT FRAME ACTIVITY */}

          <div className="pa-activity">

            <div className="pa-section-heading">
              <h3>Recent frame activity</h3>

              <span>
                {activity.length} updates
              </span>
            </div>

            {activity.length ? (

              <div className="pa-table-wrap">

                <table>
                  <thead>
                    <tr>
                      <th>Frame</th>
                      <th>Timestamp</th>
                      <th>People</th>
                      <th>Objects</th>
                      <th>Track IDs</th>
                      <th>Frame time</th>
                      <th>Status</th>
                    </tr>
                  </thead>

                  <tbody>
                    {activity.map((item, index) => (
                      <tr
                        key={`${
                          item.frame_number ?? "frame"
                        }-${index}`}
                      >
                        <td>
                          {item.frame_number ?? "—"}
                        </td>

                        <td>
                          {formatTimestamp(
                            item.timestamp_seconds
                          )}
                        </td>

                        <td>
                          {item.people_found ?? 0}
                        </td>

                        <td>
                          {item.objects_found ?? 0}
                        </td>

                        <td>
                          {item.track_ids?.length
                            ? item.track_ids.join(", ")
                            : "—"}
                        </td>

                        <td>
                          {Number(
                            item.frame_seconds || 0
                          ).toFixed(2)}
                          s
                        </td>

                        <td>
                          {item.status || "—"}
                        </td>
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

          {isActive && isPolling ? (
            <button
              type="button"
              className="pa-secondary-button"
              onClick={stopLocalPolling}
            >
              Stop checking progress
            </button>
          ) : null}

        </section>
      ) : null}

      {/* COMPLETED RESULTS */}

      {completedResult ? (
        <section className="pa-panel">

          <h2>Processing completed</h2>

          <p>
            {completedResult.message ||
              "Your video has been processed successfully."}
          </p>

          <div className="pa-downloads">

            {completedResult.video_url ? (
              <a
                href={outputUrl(
                  completedResult.video_url
                )}
                target="_blank"
                rel="noreferrer"
              >
                View processed video
              </a>
            ) : null}

            {completedResult.csv_url ? (
              <a
                href={outputUrl(
                  completedResult.csv_url
                )}
                target="_blank"
                rel="noreferrer"
              >
                Download CSV report
              </a>
            ) : null}

            {completedResult.json_url ? (
              <a
                href={outputUrl(
                  completedResult.json_url
                )}
                target="_blank"
                rel="noreferrer"
              >
                Download JSON report
              </a>
            ) : null}

          </div>

          {completedResult.analytics ||
          completedResult.summary ? (
            <details className="pa-analytics-details">

              <summary>
                View full analytics JSON
              </summary>

              <pre>
                {JSON.stringify(
                  completedResult.analytics ||
                    completedResult.summary,
                  null,
                  2
                )}
              </pre>

            </details>
          ) : null}

        </section>
      ) : null}

      {/* FOOTER */}

      <footer className="pa-footer">
        PragyanAI · YOLO11 Object Detection &amp; People Analytics
      </footer>

    </main>
  );
}
