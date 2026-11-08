
/*
============================================================
PragyanAI - People Detection & Counting
File: frontend/src/services/api.js

Backend: FastAPI / Render
Detection: YOLO11 TFLite
Tracking: ByteTrack
Video processing: OpenCV

Supports:
- Backend health and model checks
- Video validation and upload
- Asynchronous job creation
- Live frame-by-frame job polling
- People/object counts and tracking IDs
- Processing analytics
- Output listing, deletion and downloads
============================================================
*/

// ----------------------------------------------------------
// 1. API CONFIGURATION
// ----------------------------------------------------------

const DEFAULT_API_URL =
  "https://pragyanai-yolo-tf-lite-object-detction.onrender.com";

const API_BASE_URL = String(
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_URL
)
  .trim()
  .replace(/\/+$/, "");

const API_URL = `${API_BASE_URL}/api`;

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  API_URL,
  ROOT_URL: API_BASE_URL,
  HEALTH_URL: `${API_BASE_URL}/health`,
  MODEL_INFO_URL: `${API_URL}/model-info`,
  PROCESS_URL: `${API_URL}/process`, // Legacy synchronous endpoint
  JOBS_URL: `${API_URL}/jobs`,
  OUTPUTS_URL: `${API_URL}/outputs`,
  MAX_UPLOAD_SIZE: 500 * 1024 * 1024,
  ALLOWED_EXTENSIONS: ["mp4", "avi", "mov", "mkv", "webm"],
  POLL_INTERVAL_MS: 3000,
  MAX_JOB_WAIT_MS: 3 * 60 * 60 * 1000,
};

console.info("PragyanAI API base URL:", API_BASE_URL);


// ----------------------------------------------------------
// 2. COMMON RESPONSE HANDLER
// ----------------------------------------------------------

export const parseResponse = async (response) => {
  const contentType = response.headers.get("content-type") || "";
  let data;

  if (contentType.includes("application/json")) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    let message;

    if (typeof data === "string") {
      message = data;
    } else if (typeof data?.detail === "string") {
      message = data.detail;
    } else if (data?.detail?.message) {
      message = data.detail.message;
    } else if (data?.message) {
      message = data.message;
    } else {
      message = `Request failed with HTTP ${response.status}`;
    }

    const error = new Error(message);
    error.status = response.status;
    error.response = data;
    throw error;
  }

  return data;
};


// ----------------------------------------------------------
// 3. GENERIC GET REQUEST
// ----------------------------------------------------------

const get = async (url, options = {}) => {
  const response = await fetch(url, {
    method: "GET",
    headers: {
      Accept: "application/json",
      ...(options.headers || {}),
    },
    cache: "no-store",
    signal: options.signal,
  });

  return parseResponse(response);
};


// ----------------------------------------------------------
// 4. BACKEND HEALTH CHECK
// ----------------------------------------------------------

export const healthCheck = async () => {
  try {
    const data = await get(API_CONFIG.HEALTH_URL);

    const connected =
      data.backend_connected === true ||
      data.api === "running" ||
      data.status === "healthy" ||
      data.status === "degraded";

    const modelLoaded = data.model_loaded === true;

    return {
      connected,
      checking: false,
      model_loaded: modelLoaded,
      model_status:
        data.model_status ||
        (modelLoaded ? "ready" : "not_ready"),
      status: data.status || "unknown",
      api: data.api || null,
      model_exists: data.model_exists ?? null,
      model_path: data.model_path || null,
      model_error: data.model_error || null,
      detection_engine: data.detection_engine || "YOLO11",
      tracking_engine: data.tracking_engine || "ByteTrack",
      inference_runtime:
        data.inference_runtime || "TensorFlow Lite",
      message: modelLoaded
        ? "Backend connected. YOLO11 model is ready."
        : connected
          ? "Backend connected, but the model is not ready."
          : "Backend connection could not be verified.",
      raw: data,
    };
  } catch (error) {
    console.error("PragyanAI health check failed:", error);

    return {
      connected: false,
      checking: false,
      model_loaded: false,
      model_status: "unknown",
      status: "disconnected",
      api: null,
      model_exists: null,
      model_path: null,
      model_error: error.message,
      detection_engine: "YOLO11",
      tracking_engine: "ByteTrack",
      inference_runtime: "TensorFlow Lite",
      message: error.message,
      raw: null,
    };
  }
};

export const getBackendStatus = async () => healthCheck();

export const testBackendConnection = async () => {
  const result = await healthCheck();

  return {
    success: result.connected,
    connected: result.connected,
    model_loaded: result.model_loaded,
    model_status: result.model_status,
    message: result.message,
    details: result,
  };
};


// ----------------------------------------------------------
// 5. MODEL INFORMATION
// ----------------------------------------------------------

export const getModelInfo = async () =>
  get(API_CONFIG.MODEL_INFO_URL);


// ----------------------------------------------------------
// 6. VIDEO VALIDATION
// ----------------------------------------------------------

export const validateVideoFile = (file) => {
  if (!file) {
    throw new Error("Please select a video file.");
  }

  const extension = String(file.name || "")
    .split(".")
    .pop()
    .toLowerCase();

  if (!API_CONFIG.ALLOWED_EXTENSIONS.includes(extension)) {
    throw new Error(
      "Unsupported video format. Please upload MP4, AVI, MOV, MKV or WEBM."
    );
  }

  if (!file.size) {
    throw new Error("The selected video file is empty.");
  }

  if (file.size > API_CONFIG.MAX_UPLOAD_SIZE) {
    throw new Error(
      "Video size exceeds the maximum limit of 500 MB."
    );
  }

  return {
    valid: true,
    filename: file.name,
    size: file.size,
    size_mb: Number(
      (file.size / (1024 * 1024)).toFixed(2)
    ),
    extension,
  };
};


// ----------------------------------------------------------
// 7. CREATE ASYNC VIDEO JOB
// ----------------------------------------------------------

export const createVideoJob = async (
  file,
  confidence = 0.35,
  frameInterval = 1,
  options = {}
) => {
  validateVideoFile(file);

  const confidenceValue = Number(confidence);
  const intervalValue = Number(frameInterval);

  if (
    !Number.isFinite(confidenceValue) ||
    confidenceValue < 0 ||
    confidenceValue > 1
  ) {
    throw new Error("Confidence must be between 0 and 1.");
  }

  if (
    !Number.isInteger(intervalValue) ||
    intervalValue < 1
  ) {
    throw new Error(
      "Frame interval must be a positive integer."
    );
  }

  const formData = new FormData();

  formData.append("file", file);
  formData.append("confidence", String(confidenceValue));
  formData.append("frame_interval", String(intervalValue));

  let response;

  try {
    response = await fetch(API_CONFIG.JOBS_URL, {
      method: "POST",
      body: formData,
      signal: options.signal,
    });
  } catch (error) {
    throw new Error(
      `Could not connect to the backend while uploading the video: ${error.message}`
    );
  }

  const created = await parseResponse(response);

  if (!created?.job_id) {
    throw new Error(
      "Backend did not return a valid job ID."
    );
  }

  console.info(
    "PragyanAI processing job created:",
    created.job_id
  );

  return created;
};


// ----------------------------------------------------------
// 8. GET JOB STATUS / LIVE FRAME UPDATES
// ----------------------------------------------------------

export const getJobStatus = async (
  jobId,
  options = {}
) => {
  if (!jobId) {
    throw new Error("Job ID is required.");
  }

  return get(
    `${API_CONFIG.JOBS_URL}/${encodeURIComponent(jobId)}`,
    options
  );
};

export const getJobAnalytics = async (
  jobId,
  options = {}
) => {
  if (!jobId) {
    throw new Error("Job ID is required.");
  }

  return get(
    `${API_CONFIG.JOBS_URL}/${encodeURIComponent(jobId)}/analytics`,
    options
  );
};

// Alias retained for compatibility with earlier components.
export const getProcessingResults = async (jobId) =>
  getJobAnalytics(jobId);


// ----------------------------------------------------------
// 9. POLL A JOB AND REPORT EVERY STATUS RESPONSE
// ----------------------------------------------------------

const sleep = (milliseconds) =>
  new Promise((resolve) =>
    setTimeout(resolve, milliseconds)
  );

/**
 * Polls the job-status endpoint.
 *
 * onUpdate receives the complete status object on every
 * successful poll, including latest_frame and recent_activity.
 *
 * onProgress receives:
 * (percentage, completeStatusObject)
 */

export const pollVideoJob = async (
  jobId,
  {
    onUpdate = null,
    onProgress = null,
    pollIntervalMs = API_CONFIG.POLL_INTERVAL_MS,
    maxWaitMs = API_CONFIG.MAX_JOB_WAIT_MS,
    signal,
    maxConsecutiveErrors = 10,
  } = {}
) => {
  if (!jobId) {
    throw new Error("Job ID is required.");
  }

  const startedAt = Date.now();
  let consecutiveErrors = 0;

  while (Date.now() - startedAt < maxWaitMs) {
    if (signal?.aborted) {
      throw new DOMException(
        "Job polling was cancelled.",
        "AbortError"
      );
    }

    try {
      const status = await getJobStatus(jobId, { signal });

      consecutiveErrors = 0;

      if (typeof onUpdate === "function") {
        onUpdate(status);
      }

      const percentage = Math.max(
        0,
        Math.min(100, Number(status.progress) || 0)
      );

      if (typeof onProgress === "function") {
        onProgress(percentage, status);
      }

      if (status.status === "completed") {
        if (!status.result?.success) {
          throw new Error(
            "The job completed, but the backend did not return a valid result."
          );
        }

        return status;
      }

      if (
        status.status === "failed" ||
        status.status === "cancelled"
      ) {
        throw new Error(
          status.error ||
            status.message ||
            `Video job ${status.status}.`
        );
      }
    } catch (error) {
      if (error.name === "AbortError") {
        throw error;
      }

      // A failed job is a terminal application error,
      // not a transient polling error.

      if (
        error.message &&
        !error.message.startsWith("Request failed with HTTP 5") &&
        !error.message.startsWith("Failed to fetch") &&
        !error.message.includes("NetworkError") &&
        !error.message.includes("network") &&
        !error.message.includes("Load failed")
      ) {
        throw error;
      }

      consecutiveErrors += 1;

      console.warn(
        `Job status request failed (${consecutiveErrors}/${maxConsecutiveErrors}):`,
        error
      );

      if (consecutiveErrors >= maxConsecutiveErrors) {
        throw new Error(
          `Unable to retrieve job progress after ${maxConsecutiveErrors} attempts. The job may still be running. Job ID: ${jobId}. Check Render before submitting the video again.`
        );
      }
    }

    await sleep(pollIntervalMs);
  }

  throw new Error(
    `Video processing exceeded the maximum wait time. Job ID: ${jobId}. Check its status before uploading the same video again.`
  );
};


// ----------------------------------------------------------
// 10. PROCESS VIDEO (CREATE JOB + LIVE POLLING)
// ----------------------------------------------------------

/**
 * Backward-compatible helper.
 *
 * onProgress is called as:
 * onProgress(percent, jobStatus).
 *
 * onUpdate is called with the complete status object
 * every successful poll.
 */

export const processVideo = async (
  file,
  confidence = 0.35,
  frameInterval = 1,
  onProgress = null,
  options = {}
) => {
  const created = await createVideoJob(
    file,
    confidence,
    frameInterval,
    options
  );

  if (typeof onProgress === "function") {
    onProgress(
      Number(created.progress) || 0,
      created
    );
  }

  const completedStatus = await pollVideoJob(
    created.job_id,
    {
      ...options,
      onProgress,
      onUpdate: options.onUpdate,
    }
  );

  const result = completedStatus.result;

  return {
    ...result,
    job_id: created.job_id,
    video_url: getFileUrl(result.video_url),
    csv_url: getFileUrl(result.csv_url),
    json_url: getFileUrl(result.json_url),
    analytics: result.analytics || result.summary || {},
    summary: result.summary || result.analytics || {},
  };
};


// ----------------------------------------------------------
// 11. FILE URL HELPER
// ----------------------------------------------------------

export const getFileUrl = (url) => {
  if (!url) return null;

  if (/^https?:\/\//i.test(url)) {
    return url;
  }

  return `${API_BASE_URL}${
    url.startsWith("/") ? "" : "/"
  }${url}`;
};


// ----------------------------------------------------------
// 12. DOWNLOAD HELPER
// ----------------------------------------------------------

export const downloadFile = async (
  fileUrl,
  filename = "download"
) => {
  if (!fileUrl) {
    throw new Error("Download URL is not available.");
  }

  const response = await fetch(getFileUrl(fileUrl));

  if (!response.ok) {
    throw new Error(
      `Download failed with HTTP ${response.status}.`
    );
  }

  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");

  try {
    link.href = objectUrl;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
  } finally {
    link.remove();

    window.setTimeout(
      () => URL.revokeObjectURL(objectUrl),
      1500
    );
  }
};

export const downloadProcessedVideo = (url) =>
  downloadFile(
    url,
    "pragyanai_people_detection.mp4"
  );

export const downloadCSVReport = (url) =>
  downloadFile(
    url,
    "pragyanai_people_analytics.csv"
  );

export const downloadJSONReport = (url) =>
  downloadFile(
    url,
    "pragyanai_people_analytics.json"
  );


// ----------------------------------------------------------
// 13. OUTPUT FILE MANAGEMENT
// ----------------------------------------------------------

export const getOutputFiles = async () =>
  get(API_CONFIG.OUTPUTS_URL);

export const deleteOutputFile = async (filename) => {
  if (!filename) {
    throw new Error("Filename is required.");
  }

  const response = await fetch(
    `${API_CONFIG.OUTPUTS_URL}/${encodeURIComponent(filename)}`,
    {
      method: "DELETE",
      headers: {
        Accept: "application/json",
      },
    }
  );

  return parseResponse(response);
};


// ----------------------------------------------------------
// 14. DEFAULT EXPORT
// ----------------------------------------------------------

const api = {
  API_CONFIG,
  parseResponse,
  healthCheck,
  getBackendStatus,
  testBackendConnection,
  getModelInfo,
  validateVideoFile,
  createVideoJob,
  getJobStatus,
  getJobAnalytics,
  pollVideoJob,
  processVideo,
  getProcessingResults,
  getFileUrl,
  downloadFile,
  downloadProcessedVideo,
  downloadCSVReport,
  downloadJSONReport,
  getOutputFiles,
  deleteOutputFile,
};

export default api;
