
import React, { useCallback, useEffect, useRef, useState } from "react";

import {
  API_CONFIG,
  healthCheck,
  createVideoJob,
  getJobStatus,
  downloadProcessedVideo,
  downloadCSVReport,
  downloadJSONReport,
} from "./services/api";

import ProcessingStatus from "./components/ProcessingStatus";
import "./style.css";

const formatBytes = (bytes) => {
  if (!Number.isFinite(Number(bytes))) return "—";

  const value = Number(bytes);

  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }

  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
};

export default function App() {
  const [backend, setBackend] = useState({
    checking: true,
    connected: false,
    model_loaded: false,
    message: "Checking backend connection…",
  });

  const [file, setFile] = useState(null);
  const [confidence, setConfidence] = useState(0.35);

  // Default to every 2nd frame to reduce Render CPU and memory usage.
  const [frameInterval, setFrameInterval] = useState(2);

  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);

  const [submitting, setSubmitting] = useState(false);
  const [isPolling, setIsPolling] = useState(false);

  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const pollTimer = useRef(null);
  const activeJobId = useRef(null);
  const mounted = useRef(false);
  const consecutivePollErrors = useRef(0);

  // ---------------------------------------------------------
  // BACKEND HEALTH CHECK
  // ---------------------------------------------------------

  const refreshBackend = useCallback(async () => {
    setBackend((old) => ({
      ...old,
      checking: true,
    }));

    const status = await healthCheck();

    if (mounted.current) {
      setBackend({
        ...status,
        checking: false,
      });
    }

    return status;
  }, []);

  useEffect(() => {
    mounted.current = true;

    refreshBackend().catch((err) => {
      if (mounted.current) {
        setBackend({
          checking: false,
          connected: false,
          model_loaded: false,
          message: err.message,
        });
      }
    });

    return () => {
      mounted.current = false;

      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
      }
    };
  }, [refreshBackend]);

  // ---------------------------------------------------------
  // JOB STATUS POLLING
  // ---------------------------------------------------------

  const pollJob = useCallback(async (jobId) => {
    if (!mounted.current || activeJobId.current !== jobId) {
      return;
    }

    setIsPolling(true);

    try {
      const status = await getJobStatus(jobId);

      if (!mounted.current || activeJobId.current !== jobId) {
        return;
      }

      consecutivePollErrors.current = 0;

      setJob(status);
      setError("");

      // Processing completed successfully.
      if (status.status === "completed") {
        setResult(status.result || null);

        setIsPolling(false);
        activeJobId.current = null;

        setNotice("Video processing completed successfully.");

        return;
      }

      // Processing failed or cancelled.
      if (
        status.status === "failed" ||
        status.status === "cancelled"
      ) {
        setError(
          status.error ||
          status.message ||
          `Job ${status.status}.`
        );

        setIsPolling(false);
        activeJobId.current = null;

        return;
      }

      // Continue polling while job is queued or processing.
      pollTimer.current = window.setTimeout(() => {
        pollJob(jobId);
      }, API_CONFIG.POLL_INTERVAL_MS);

    } catch (pollError) {
      if (!mounted.current || activeJobId.current !== jobId) {
        return;
      }

      consecutivePollErrors.current += 1;

      console.error("Job polling error:", pollError);

      // Temporary polling errors do not necessarily mean
      // that the backend processing job has failed.
      if (consecutivePollErrors.current >= 12) {
        setError(
          `Unable to retrieve job status after several attempts. ` +
          `The backend job may still be running. Job ID: ${jobId}. ` +
          `Check Render before uploading the same video again.`
        );

        setIsPolling(false);
        activeJobId.current = null;

        return;
      }

      setNotice(
        `Connection interrupted. Retrying status check ` +
        `(${consecutivePollErrors.current}/12)…`
      );

      pollTimer.current = window.setTimeout(() => {
        pollJob(jobId);
      }, 5000);
    }
  }, []);

  // ---------------------------------------------------------
  // VIDEO FILE SELECTION
  // ---------------------------------------------------------

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0] || null;

    setFile(selected);
    setError("");
    setNotice("");
  };

  // ---------------------------------------------------------
  // START VIDEO PROCESSING
  // ---------------------------------------------------------

  const handleSubmit = async (event) => {
    event.preventDefault();

    setError("");
    setNotice("");

    if (!file) {
      setError("Please select a video file.");
      return;
    }

    if (pollTimer.current) {
      window.clearTimeout(pollTimer.current);
    }

    activeJobId.current = null;

    setSubmitting(true);
    setIsPolling(false);

    setJob(null);
    setResult(null);

    consecutivePollErrors.current = 0;

    try {
      const created = await createVideoJob(
        file,
        confidence,
        frameInterval
      );

      if (!mounted.current) {
        return;
      }

      const initialJob = {
        ...created,

        status: created.status || "queued",

        latest_frame: null,
        recent_activity: [],

        current_people: 0,
        current_objects: 0,
        unique_people_so_far: 0,
      };

      setJob(initialJob);

      activeJobId.current = created.job_id;

      setNotice(`Job created: ${created.job_id}`);

      setSubmitting(false);

      // Begin monitoring the backend job.
      pollJob(created.job_id);

    } catch (submitError) {
      setError(
        submitError.message ||
        "Unable to upload the video."
      );

      setSubmitting(false);
    }
  };

  // ---------------------------------------------------------
  // STOP BROWSER POLLING
  // ---------------------------------------------------------

  const stopPolling = () => {
    if (pollTimer.current) {
      window.clearTimeout(pollTimer.current);
    }

    activeJobId.current = null;

    setIsPolling(false);

    setNotice(
      "Browser status polling stopped. The backend job may continue processing."
    );
  };

  // ---------------------------------------------------------
  // JOB STATUS
  // ---------------------------------------------------------

  const active = ["queued", "processing"].includes(
    job?.status
  );

  const completed =
    result ||
    (job?.status === "completed" ? job.result : null);

  // ---------------------------------------------------------
  // DOWNLOAD PROCESSED OUTPUTS
  // ---------------------------------------------------------

  const handleDownload = async (kind) => {
    try {
      if (kind === "video") {
        await downloadProcessedVideo(
          completed?.video_url
        );
      }

      if (kind === "csv") {
        await downloadCSVReport(
          completed?.csv_url
        );
      }

      if (kind === "json") {
        await downloadJSONReport(
          completed?.json_url
        );
      }

    } catch (downloadError) {
      setError(
        downloadError.message ||
        "Unable to download the output file."
      );
    }
  };

  // ---------------------------------------------------------
  // USER INTERFACE
  // ---------------------------------------------------------

  return (
    <main className="pa-app">

      {/* HEADER */}
      <header className="pa-header">

        <div className="pa-brand">

          <div className="pa-brand-mark">
            P
          </div>

          <div>
            <div className="pa-eyebrow">
              PRAGYANAI · COMPUTER VISION
            </div>

            <h1>
              People Detection &amp; Counting
            </h1>

            <p>
              YOLO11 TFLite · ByteTrack · OpenCV
            </p>
          </div>

        </div>

        {/* BACKEND CONNECTION STATUS */}
        <div
          className={`pa-connection ${
            backend.connected
              ? "is-online"
              : "is-offline"
          }`}
        >

          <span className="pa-status-dot" />

          <span>
            {backend.checking
              ? "Checking backend"
              : backend.connected
                ? backend.model_loaded
                  ? "Backend & model ready"
                  : "Backend connected · Model not ready"
                : "Backend disconnected"}
          </span>

          <button
            type="button"
            className="pa-small-button"
            onClick={() =>
              refreshBackend().catch((err) =>
                setError(err.message)
              )
            }
            disabled={backend.checking}
          >
            Refresh
          </button>

        </div>

      </header>

      {/* VIDEO UPLOAD PANEL */}
      <section className="pa-panel pa-upload-panel">

        <div className="pa-section-title">

          <div>
            <span className="pa-kicker">
              VIDEO ANALYSIS
            </span>

            <h2>
              Upload a video
            </h2>

            <p>
              Upload a video to start frame-by-frame
              people detection and tracking.
            </p>
          </div>

          <span className="pa-engine-tag">
            TensorFlow Lite Runtime
          </span>

        </div>

        <form
          onSubmit={handleSubmit}
          className="pa-form"
        >

          {/* VIDEO FILE INPUT */}
          <label className="pa-file-drop">

            <span
              className="pa-upload-icon"
              aria-hidden="true"
            >
              ↑
            </span>

            <strong>
              {file
                ? "Video selected"
                : "Choose a video file"}
            </strong>

            <span>
              MP4, AVI, MOV, MKV or WEBM · Maximum 500 MB
            </span>

            <input
              type="file"
              accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm"
              onChange={handleFileChange}
              disabled={submitting || active}
            />

          </label>

          {/* SELECTED FILE INFORMATION */}
          {file && (
            <div className="pa-selected-file">

              <div className="pa-file-symbol">
                VIDEO
              </div>

              <div className="pa-file-details">

                <strong title={file.name}>
                  {file.name}
                </strong>

                <span>
                  {formatBytes(file.size)}
                </span>

              </div>

              <button
                type="button"
                className="pa-text-button"
                onClick={() => setFile(null)}
                disabled={submitting || active}
              >
                Remove
              </button>

            </div>
          )}

          {/* PROCESSING SETTINGS */}
          <div className="pa-settings-grid">

            {/* CONFIDENCE THRESHOLD */}
            <label className="pa-setting">

              <span>
                Confidence threshold{" "}
                <strong>
                  {confidence.toFixed(2)}
                </strong>
              </span>

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
                disabled={submitting || active}
              />

              <small>
                Higher values reduce low-confidence detections.
              </small>

            </label>

            {/* FRAME INTERVAL */}
            <label className="pa-setting">

              <span>
                Frame interval
              </span>

              <select
                value={frameInterval}
                onChange={(event) =>
                  setFrameInterval(
                    Number(event.target.value)
                  )
                }
                disabled={submitting || active}
              >

                <option value={1}>
                  Every frame (best detail)
                </option>

                <option value={2}>
                  Every 2nd frame (recommended)
                </option>

                <option value={3}>
                  Every 3rd frame
                </option>

                <option value={5}>
                  Every 5th frame
                </option>

              </select>

              <small>
                Processing every 2nd frame reduces inference
                workload. Every-frame processing may improve
                tracking of fast-moving people.
              </small>

            </label>

          </div>

          {/* SUBMIT BUTTON */}
          <button
            className="pa-primary-button"
            type="submit"
            disabled={
              !file ||
              submitting ||
              active
            }
          >

            {submitting
              ? "Uploading video…"
              : active
                ? "Processing in progress…"
                : "Upload & Start Detection"}

          </button>

        </form>

        {/* MODEL STATUS WARNING */}
        {!backend.model_loaded &&
          !backend.checking && (

            <div className="pa-inline-warning">
              Backend health check has not confirmed
              that the YOLO model is loaded. Check the
              Render health endpoint before starting
              a long job.
            </div>

          )}

      </section>

      {/* ERROR MESSAGE */}
      {error && (
        <div
          className="pa-error"
          role="alert"
        >
          {error}
        </div>
      )}

      {/* STATUS / NOTICE MESSAGE */}
      {!error && notice && (
        <div
          className="pa-notice"
          role="status"
        >
          {notice}
        </div>
      )}

      {/* LIVE PROCESSING STATUS */}
      {job && (
        <ProcessingStatus
          job={job}
          isPolling={isPolling}
          onStopPolling={stopPolling}
          showActivity
        />
      )}

      {/* COMPLETED RESULTS */}
      {completed && (

        <section className="pa-panel pa-results-panel">

          <div className="pa-section-title">

            <div>
              <span className="pa-kicker">
                ANALYSIS COMPLETE
              </span>

              <h2>
                Processing results
              </h2>

              <p>
                {completed.message ||
                  "Your processed video and analytics reports are ready."}
              </p>
            </div>

            <span className="pa-complete-mark">
              ✓ Completed
            </span>

          </div>

          {/* DOWNLOAD CARDS */}
          <div className="pa-download-grid">

            {completed.video_url && (

              <button
                type="button"
                className="pa-download-card"
                onClick={() => handleDownload("video")}
              >

                <span className="pa-download-icon">
                  ▶
                </span>

                <span>
                  <strong>
                    Processed video
                  </strong>

                  <small>
                    Annotated MP4 output
                  </small>
                </span>

                <span className="pa-download-arrow">
                  ↓
                </span>

              </button>

            )}

            {completed.csv_url && (

              <button
                type="button"
                className="pa-download-card"
                onClick={() => handleDownload("csv")}
              >

                <span className="pa-download-icon">
                  CSV
                </span>

                <span>
                  <strong>
                    Frame analytics
                  </strong>

                  <small>
                    Download CSV report
                  </small>
                </span>

                <span className="pa-download-arrow">
                  ↓
                </span>

              </button>

            )}

            {completed.json_url && (

              <button
                type="button"
                className="pa-download-card"
                onClick={() => handleDownload("json")}
              >

                <span className="pa-download-icon">
                  JSON
                </span>

                <span>
                  <strong>
                    Full analytics
                  </strong>

                  <small>
                    Download JSON report
                  </small>
                </span>

                <span className="pa-download-arrow">
                  ↓
                </span>

              </button>

            )}

          </div>

          {/* FINAL ANALYTICS JSON */}
          {(completed.analytics || completed.summary) && (

            <details className="pa-analytics-details">

              <summary>
                View final analytics details
              </summary>

              <pre>
                {JSON.stringify(
                  completed.analytics ||
                  completed.summary,
                  null,
                  2
                )}
              </pre>

            </details>

          )}

        </section>

      )}

      {/* FOOTER */}
      <footer className="pa-footer">

        <span>
          PRAGYANAI
        </span>

        <span>
          YOLO11 Object Detection &amp; People Analytics
        </span>

      </footer>

    </main>
  );
}

