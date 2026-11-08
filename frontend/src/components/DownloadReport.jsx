
import React, { useState } from "react";

import {
  Download,
  FileVideo,
  FileSpreadsheet,
  FileJson,
  AlertCircle,
  CheckCircle,
  LoaderCircle,
  FolderDown,
} from "lucide-react";


/* ============================================================
   CONFIGURATION
   ============================================================ */

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000"
).replace(/\/+$/, "");


/* ============================================================
   HELPER: RESOLVE FILE URL
   ============================================================ */

function resolveFileUrl(url) {
  if (!url || typeof url !== "string") {
    return null;
  }

  try {
    return new URL(url, `${API_BASE_URL}/`).href;
  } catch {
    return null;
  }
}


/* ============================================================
   HELPER: SAVE BLOB TO DEVICE
   ============================================================ */

function saveBlob(blob, filename) {
  const objectUrl = URL.createObjectURL(blob);

  const link = document.createElement("a");

  link.href = objectUrl;
  link.download = filename;

  document.body.appendChild(link);
  link.click();
  link.remove();

  // Give the browser time to initiate the download.
  window.setTimeout(() => {
    URL.revokeObjectURL(objectUrl);
  }, 10000);
}


/* ============================================================
   HELPER: DOWNLOAD FILE
   ============================================================ */

async function downloadFile(url, filename) {
  const resolvedUrl = resolveFileUrl(url);

  if (!resolvedUrl) {
    throw new Error("Download URL is not available.");
  }

  const response = await fetch(resolvedUrl, {
    method: "GET",
    mode: "cors",
  });

  if (!response.ok) {
    throw new Error(
      `Download failed. Server returned ${response.status}.`
    );
  }

  const blob = await response.blob();

  if (blob.size === 0) {
    throw new Error("The downloaded file is empty.");
  }

  saveBlob(blob, filename);
}


/* ============================================================
   HELPER: GENERATE JSON REPORT
   ============================================================ */

function downloadJsonReport(results) {
  if (!results) {
    throw new Error("No report data is available.");
  }

  const report = {
    application: "PragyanAI People Detection & Counting",
    generated_at: new Date().toISOString(),
    model: "YOLO11 TensorFlow Lite",
    tracking: "ByteTrack",
    results,
  };

  const jsonString = JSON.stringify(report, null, 2);

  const blob = new Blob([jsonString], {
    type: "application/json;charset=utf-8",
  });

  saveBlob(
    blob,
    "pragyanai_people_detection_report.json"
  );
}


/* ============================================================
   DOWNLOAD BUTTON
   ============================================================ */

function DownloadButton({
  icon: Icon,
  title,
  description,
  onClick,
  loading = false,
  disabled = false,
  variant = "primary",
}) {
  return (
    <button
      type="button"
      className={`report-download-button ${variant}`}
      onClick={onClick}
      disabled={disabled || loading}
      aria-busy={loading}
    >
      <div className="report-download-icon">
        {loading ? (
          <LoaderCircle
            size={22}
            className="rotating-icon"
          />
        ) : (
          <Icon size={22} />
        )}
      </div>

      <div className="report-download-content">
        <strong>
          {loading ? "Downloading..." : title}
        </strong>

        <span>{description}</span>
      </div>

      <Download
        size={18}
        className="report-download-arrow"
      />
    </button>
  );
}


/* ============================================================
   MAIN COMPONENT
   ============================================================ */

export default function DownloadReport({
  videoUrl,
  csvUrl,
  results,
}) {
  const [downloading, setDownloading] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const resolvedVideoUrl = resolveFileUrl(videoUrl);
  const resolvedCsvUrl = resolveFileUrl(csvUrl);

  const hasVideo = Boolean(resolvedVideoUrl);
  const hasCsv = Boolean(resolvedCsvUrl);
  const hasResults = Boolean(results);

  const availableCount = [
    hasVideo,
    hasCsv,
    hasResults,
  ].filter(Boolean).length;


  /* ----------------------------------------------------------
     HANDLE DOWNLOAD
     ---------------------------------------------------------- */

  async function handleDownload(type) {
    if (downloading) {
      return;
    }

    setError("");
    setSuccess("");
    setDownloading(type);

    try {
      switch (type) {
        case "video":
          await downloadFile(
            resolvedVideoUrl,
            "pragyanai_people_detection_processed.mp4"
          );

          setSuccess(
            "Processed video download started successfully."
          );
          break;

        case "csv":
          await downloadFile(
            resolvedCsvUrl,
            "pragyanai_people_detection_analytics.csv"
          );

          setSuccess(
            "CSV analytics download started successfully."
          );
          break;

        case "json":
          downloadJsonReport(results);

          setSuccess(
            "JSON report download started successfully."
          );
          break;

        default:
          throw new Error("Unsupported download format.");
      }
    } catch (err) {
      console.error("Download error:", err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to download the requested file."
      );
    } finally {
      setDownloading("");
    }
  }


  /* ----------------------------------------------------------
     COMPONENT UI
     ---------------------------------------------------------- */

  return (
    <section className="download-card">

      {/* ================= HEADER ================= */}

      <div className="download-header">

        <div className="download-heading-icon">
          <FolderDown size={23} />
        </div>

        <div>
          <h3>Download Reports</h3>

          <p>
            Download your processed video and AI analytics.
          </p>
        </div>

      </div>


      {/* ================= AVAILABILITY ================= */}

      <div className="download-availability">

        <CheckCircle size={15} />

        <span>
          {availableCount === 3
            ? "All reports are ready for download."
            : `${availableCount} of 3 reports available.`}
        </span>

      </div>


      {/* ================= DOWNLOAD OPTIONS ================= */}

      <div className="report-download-list">

        {/* MP4 Video */}

        <DownloadButton
          icon={FileVideo}
          title="Processed Video (MP4)"
          description="Video with YOLO11 bounding boxes and tracking IDs"
          onClick={() => handleDownload("video")}
          loading={downloading === "video"}
          disabled={!hasVideo || Boolean(downloading)}
          variant="primary"
        />


        {/* CSV Analytics */}

        <DownloadButton
          icon={FileSpreadsheet}
          title="Analytics Report (CSV)"
          description="Frame-wise people counting and tracking data"
          onClick={() => handleDownload("csv")}
          loading={downloading === "csv"}
          disabled={!hasCsv || Boolean(downloading)}
          variant="secondary"
        />


        {/* JSON Report */}

        <DownloadButton
          icon={FileJson}
          title="Complete Report (JSON)"
          description="Detection summary, tracking information and analytics"
          onClick={() => handleDownload("json")}
          loading={downloading === "json"}
          disabled={!hasResults || Boolean(downloading)}
          variant="secondary"
        />

      </div>


      {/* ================= ERROR MESSAGE ================= */}

      {error && (
        <div
          className="download-error"
          role="alert"
        >
          <AlertCircle size={17} />

          <span>{error}</span>
        </div>
      )}


      {/* ================= SUCCESS MESSAGE ================= */}

      {success && (
        <div
          className="download-success"
          role="status"
        >
          <CheckCircle size={17} />

          <span>{success}</span>
        </div>
      )}


      {/* ================= FOOTER ================= */}

      <div className="download-footer">

        <span>
          <FileVideo size={13} />
          MP4 Video
        </span>

        <span>
          <FileSpreadsheet size={13} />
          CSV Analytics
        </span>

        <span>
          <FileJson size={13} />
          JSON Report
        </span>

      </div>

    </section>
  );
}
