
/*
 * PragyanAI People Detection & Counting - API client
 * FastAPI backend: Render
 * Frontend: React + Vite / Netlify
 */

// =========================================================
// 1. API CONFIGURATION
// =========================================================

const DEFAULT_API_URL =
  "https://pragyanai-yolo-tf-lite-object-detction.onrender.com";

const API_BASE_URL = String(
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_URL
).trim().replace(/\/+$/, "");

const API_URL = `${API_BASE_URL}/api`;

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  API_URL,
  HEALTH_URL: `${API_BASE_URL}/health`,
  MODEL_INFO_URL: `${API_URL}/model-info`,
  JOBS_URL: `${API_URL}/jobs`,
  OUTPUTS_URL: `${API_URL}/outputs`,
  MAX_UPLOAD_SIZE: 500 * 1024 * 1024,
  ALLOWED_EXTENSIONS: ["mp4", "avi", "mov", "mkv", "webm"],
  POLL_INTERVAL_MS: 3000,
  MAX_JOB_WAIT_MS: 3 * 60 * 60 * 1000,
};

// =========================================================
// 2. COMMON RESPONSE HANDLER
// =========================================================

export const parseResponse = async (response) => {
  const contentType = response.headers.get("content-type") || "";

  let data;

  try {
    data = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
  } catch {
    data = null;
  }

  if (!response.ok) {
    let message = `Request failed with HTTP ${response.status}`;

    if (typeof data === "string" && data) {
      message = data;
    } else if (typeof data?.detail === "string") {
      message = data.detail;
    } else if (data?.detail?.message) {
      message = data.detail.message;
    } else if (data?.message) {
      message = data.message;
    }

    const error = new Error(message);
    error.status = response.status;
    error.response = data;

    throw error;
  }

  return data;
};

// =========================================================
// 3. COMMON FETCH REQUEST
// =========================================================

const request = async (url, options = {}) => {
  let response;

  try {
    response = await fetch(url, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    });
  } catch (error) {
    const wrapped = new Error(
      `Unable to connect to the PragyanAI backend: ${error.message}`
    );

    wrapped.cause = error;
    throw wrapped;
  }

  return parseResponse(response);
};

// =========================================================
// 4. BACKEND HEALTH CHECK
// =========================================================

export const healthCheck = async () => {
  try {
    const data = await request(
      API_CONFIG.HEALTH_URL,
      {
        method: "GET",
        cache: "no-store",
      }
    );

    const connected =
      data?.api === "running" ||
      data?.backend_connected === true ||
      data?.status === "healthy" ||
      data?.status === "degraded" ||
      data?.status === "ok";

    const modelLoaded = data?.model_loaded === true;

    return {
      connected,
      checking: false,
      model_loaded: modelLoaded,

      model_status:
        data?.model_status ||
        (modelLoaded ? "ready" : "not_ready"),

      status: data?.status || "unknown",
      api: data?.api || null,

      model_exists: data?.model_exists ?? null,
      model_path: data?.model_path || null,
      model_error: data?.model_error || null,

      detection_engine:
        data?.detection_engine || "YOLO11",

      tracking_engine:
        data?.tracking_engine || "ByteTrack",

      inference_runtime:
        data?.inference_runtime || "TensorFlow Lite",

      message: modelLoaded
        ? "Backend connected. YOLO11 model is ready."
        : connected
          ? "Backend connected, but the model is not confirmed as ready."
          : "Backend connection could not be verified.",

      raw: data,
    };

  } catch (error) {
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

export const getBackendStatus = healthCheck;

// =========================================================
// 5. GET MODEL INFORMATION
// =========================================================

export const getModelInfo = () =>
  request(
    API_CONFIG.MODEL_INFO_URL,
    {
      method: "GET",
      cache: "no-store",
    }
  );

// =========================================================
// 6. VIDEO FILE VALIDATION
// =========================================================

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
      "Unsupported video format. Upload MP4, AVI, MOV, MKV or WEBM."
    );
  }

  if (!file.size) {
    throw new Error("The selected video file is empty.");
  }

  if (file.size > API_CONFIG.MAX_UPLOAD_SIZE) {
    throw new Error(
      "Video exceeds the maximum upload size of 500 MB."
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

// =========================================================
// 7. CREATE VIDEO PROCESSING JOB
// =========================================================

export const createVideoJob = async (
  file,
  confidence = 0.35,
  frameInterval = 2,
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
    throw new Error(
      "Confidence must be between 0 and 1."
    );
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
  formData.append(
    "confidence",
    String(confidenceValue)
  );
  formData.append(
    "frame_interval",
    String(intervalValue)
  );

  return request(
    API_CONFIG.JOBS_URL,
    {
      method: "POST",
      body: formData,
      signal: options.signal,
    }
  );
};

// =========================================================
// 8. GET JOB STATUS
// =========================================================

export const getJobStatus = async (
  jobId,
  options = {}
) => {
  if (!jobId) {
    throw new Error("Job ID is required.");
  }

  return request(
    `${API_CONFIG.JOBS_URL}/${encodeURIComponent(jobId)}`,
    {
      method: "GET",
      cache: "no-store",
      signal: options.signal,
    }
  );
};

// =========================================================
// 9. GET JOB ANALYTICS
// =========================================================

export const getJobAnalytics = async (
  jobId,
  options = {}
) => {
  if (!jobId) {
    throw new Error("Job ID is required.");
  }

  return request(
    `${API_CONFIG.JOBS_URL}/${encodeURIComponent(jobId)}/analytics`,
    {
      method: "GET",
      cache: "no-store",
      signal: options.signal,
    }
  );
};

export const getProcessingResults = getJobAnalytics;

// =========================================================
// 10. POLL VIDEO PROCESSING JOB
// =========================================================

export const pollVideoJob = async (
  jobId,
  {
    onUpdate,
    onProgress,
    pollIntervalMs = API_CONFIG.POLL_INTERVAL_MS,
    maxWaitMs = API_CONFIG.MAX_JOB_WAIT_MS,
    signal,
    maxConsecutiveErrors = 12,
  } = {}
) => {
  if (!jobId) {
    throw new Error("Job ID is required.");
  }

  const startedAt = Date.now();
  let errors = 0;

  while (Date.now() - startedAt < maxWaitMs) {
    if (signal?.aborted) {
      throw new DOMException(
        "Polling cancelled.",
        "AbortError"
      );
    }

    try {
      const status = await getJobStatus(
        jobId,
        { signal }
      );

      errors = 0;

      onUpdate?.(status);

      onProgress?.(
        Math.max(
          0,
          Math.min(
            100,
            Number(status.progress) || 0
          )
        ),
        status
      );

      if (status.status === "completed") {
        return status;
      }

      if (
        status.status === "failed" ||
        status.status === "cancelled"
      ) {
        throw new Error(
          status.error ||
          status.message ||
          `Job ${status.status}.`
        );
      }

    } catch (error) {
      if (error.name === "AbortError") {
        throw error;
      }

      if (
        ["failed", "cancelled"].includes(
          error.response?.status
        )
      ) {
        throw error;
      }

      if (
        error.status === 404 ||
        error.status === 400 ||
        error.status === 422
      ) {
        throw error;
      }

      errors += 1;

      if (errors >= maxConsecutiveErrors) {
        throw new Error(
          `Unable to retrieve job status after ${maxConsecutiveErrors} attempts. ` +
          `The job may still be running. Job ID: ${jobId}. ` +
          `Check Render before resubmitting.`
        );
      }
    }

    await new Promise((resolve) =>
      window.setTimeout(resolve, pollIntervalMs)
    );
  }

  throw new Error(
    `Polling timed out. Check job ${jobId} on the backend before resubmitting.`
  );
};

// =========================================================
// 11. RESOLVE BACKEND FILE URL
// =========================================================

export const getFileUrl = (url) => {
  if (!url) {
    return null;
  }

  if (/^https?:\/\//i.test(url)) {
    return url;
  }

  return (
    `${API_BASE_URL}` +
    `${url.startsWith("/") ? "" : "/"}` +
    `${url}`
  );
};

// =========================================================
// 12. DOWNLOAD FILE
// =========================================================

export const downloadFile = async (
  url,
  filename = "download"
) => {
  const resolvedUrl = getFileUrl(url);

  if (!resolvedUrl) {
    throw new Error(
      "Download URL is not available."
    );
  }

  const response = await fetch(resolvedUrl);

  if (!response.ok) {
    throw new Error(
      `Download failed with HTTP ${response.status}.`
    );
  }

  const blob = await response.blob();

  const objectUrl = URL.createObjectURL(blob);

  const anchor = document.createElement("a");

  anchor.href = objectUrl;
  anchor.download = filename;

  document.body.appendChild(anchor);

  anchor.click();
  anchor.remove();

  window.setTimeout(
    () => URL.revokeObjectURL(objectUrl),
    1500
  );
};

// =========================================================
// 13. DOWNLOAD PROCESSED VIDEO
// =========================================================

export const downloadProcessedVideo = (url) =>
  downloadFile(
    url,
    "pragyanai_people_detection_processed.mp4"
  );

// =========================================================
// 14. DOWNLOAD CSV REPORT
// =========================================================

export const downloadCSVReport = (url) =>
  downloadFile(
    url,
    "pragyanai_people_analytics.csv"
  );

// =========================================================
// 15. DOWNLOAD JSON REPORT
// =========================================================

export const downloadJSONReport = (url) =>
  downloadFile(
    url,
    "pragyanai_people_analytics.json"
  );

// =========================================================
// 16. LIST OUTPUT FILES
// =========================================================

export const getOutputFiles = () =>
  request(
    API_CONFIG.OUTPUTS_URL,
    {
      method: "GET",
      cache: "no-store",
    }
  );

// =========================================================
// 17. DELETE OUTPUT FILE
// =========================================================

export const deleteOutputFile = (filename) =>
  request(
    `${API_CONFIG.OUTPUTS_URL}/${encodeURIComponent(filename)}`,
    {
      method: "DELETE",
    }
  );

// =========================================================
// 18. DEFAULT API EXPORT
// =========================================================

const api = {
  API_CONFIG,
  parseResponse,
  healthCheck,
  getBackendStatus,
  getModelInfo,
  validateVideoFile,
  createVideoJob,
  getJobStatus,
  getJobAnalytics,
  getProcessingResults,
  pollVideoJob,
  getFileUrl,
  downloadFile,
  downloadProcessedVideo,
  downloadCSVReport,
  downloadJSONReport,
  getOutputFiles,
  deleteOutputFile,
};

export default api;
