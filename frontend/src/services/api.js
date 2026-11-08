
/*
 * PragyanAI - People Detection & Counting
 * API Service
 *
 * Frontend: React + Vite + Netlify
 * Backend: FastAPI + Render
 */

// --------------------------------------------------
// 1. API CONFIGURATION
// --------------------------------------------------

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000"
).replace(/\/+$/, "");

const API_BASE_PATH = "/api";

// Request timeout: 10 minutes
const REQUEST_TIMEOUT = 10 * 60 * 1000;

// --------------------------------------------------
// 2. HELPER FUNCTIONS
// --------------------------------------------------

/**
 * Build complete backend API URL.
 */
const getApiUrl = (endpoint) => {
  const cleanEndpoint = endpoint.startsWith("/")
    ? endpoint
    : `/${endpoint}`;

  return `${API_BASE_URL}${cleanEndpoint}`;
};

/**
 * Build output URL.
 * Supports absolute and relative backend URLs.
 */
export const getFileUrl = (url) => {
  if (!url) return "";

  if (
    url.startsWith("http://") ||
    url.startsWith("https://") ||
    url.startsWith("blob:")
  ) {
    return url;
  }

  return `${API_BASE_URL}${
    url.startsWith("/") ? url : `/${url}`
  }`;
};

/**
 * Extract useful error message.
 */
const getErrorMessage = (errorData, fallback) => {
  if (!errorData) return fallback;

  if (typeof errorData.detail === "string") {
    return errorData.detail;
  }

  if (Array.isArray(errorData.detail)) {
    return errorData.detail
      .map((item) => item.msg || JSON.stringify(item))
      .join(", ");
  }

  if (typeof errorData.message === "string") {
    return errorData.message;
  }

  return fallback;
};

/**
 * Safely parse API response.
 */
const parseResponse = async (response) => {
  const contentType =
    response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  const text = await response.text();

  return {
    message: text || response.statusText,
  };
};

// --------------------------------------------------
// 3. GENERIC API REQUEST
// --------------------------------------------------

const apiRequest = async (
  endpoint,
  options = {},
  timeout = REQUEST_TIMEOUT
) => {
  const controller = new AbortController();

  const timeoutId = setTimeout(
    () => controller.abort(),
    timeout
  );

  try {
    const response = await fetch(getApiUrl(endpoint), {
      ...options,
      signal: controller.signal,
    });

    const data = await parseResponse(response);

    if (!response.ok) {
      throw new Error(
        getErrorMessage(
          data,
          `Server error: ${response.status}`
        )
      );
    }

    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(
        "Request timed out. The backend may still be processing your video. Please check the Render logs."
      );
    }

    if (
      error instanceof TypeError ||
      error.message === "Failed to fetch"
    ) {
      throw new Error(
        "Cannot connect to the backend. Please verify that the Render API is running, the API URL is correct, and CORS is configured."
      );
    }

    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
};

// --------------------------------------------------
// 4. BACKEND HEALTH CHECK
// --------------------------------------------------

export const healthCheck = async () => {
  try {
    const data = await apiRequest(
      "/health",
      {
        method: "GET",
      },
      30000
    );

    return {
      success: true,
      connected: true,
      data,
    };
  } catch (error) {
    return {
      success: false,
      connected: false,
      error: error.message,
    };
  }
};

// --------------------------------------------------
// 5. ROOT API STATUS
// --------------------------------------------------

export const getBackendStatus = async () => {
  return apiRequest(
    "/",
    {
      method: "GET",
    },
    30000
  );
};

// --------------------------------------------------
// 6. PROCESS VIDEO
// --------------------------------------------------

/**
 * Upload video and start people detection.
 *
 * Backend endpoint:
 * POST /api/process
 *
 * Form fields:
 * file
 * confidence
 * frame_interval
 */

export const processVideo = async (
  file,
  confidence = 0.35,
  frameInterval = 1,
  onProgress = null
) => {
  if (!file) {
    throw new Error("Please select a video file.");
  }

  if (!(file instanceof File)) {
    throw new Error("Invalid video file.");
  }

  const allowedExtensions = [
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm",
  ];

  const extension = file.name
    .split(".")
    .pop()
    .toLowerCase();

  if (!allowedExtensions.includes(extension)) {
    throw new Error(
      "Unsupported video format. Upload MP4, AVI, MOV, MKV or WEBM."
    );
  }

  const MAX_SIZE = 500 * 1024 * 1024;

  if (file.size > MAX_SIZE) {
    throw new Error(
      "Video size exceeds the maximum limit of 500 MB."
    );
  }

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

  const controller = new AbortController();

  const timeoutId = setTimeout(
    () => controller.abort(),
    REQUEST_TIMEOUT
  );

  try {
    // Fetch-based upload.
    // The browser manages multipart boundaries automatically.
    const response = await fetch(
      getApiUrl(`${API_BASE_PATH}/process`),
      {
        method: "POST",
        body: formData,
        signal: controller.signal,
      }
    );

    const data = await parseResponse(response);

    if (!response.ok) {
      throw new Error(
        getErrorMessage(
          data,
          `Video processing failed: ${response.status}`
        )
      );
    }

    if (typeof onProgress === "function") {
      onProgress(100);
    }

    // Normalize output URLs.
    const videoUrl =
      data.video_url ||
      data.output_video_url ||
      data.processed_video_url ||
      null;

    const csvUrl =
      data.csv_url ||
      data.csv_file_url ||
      data.report_url ||
      null;

    const jsonUrl =
      data.json_url ||
      data.json_file_url ||
      null;

    return {
      success: true,

      ...data,

      video_url: getFileUrl(videoUrl),
      output_video_url: getFileUrl(videoUrl),

      csv_url: getFileUrl(csvUrl),
      csv_file_url: getFileUrl(csvUrl),

      json_url: getFileUrl(jsonUrl),

      analytics:
        data.analytics ||
        data.summary ||
        {},

      summary:
        data.summary ||
        data.analytics ||
        {},
    };
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(
        "Video processing request timed out. Check your Render service logs and video processing status."
      );
    }

    if (
      error instanceof TypeError ||
      error.message === "Failed to fetch"
    ) {
      throw new Error(
        "Cannot connect to the backend. Please verify that the Render API is running, the API URL is correct, and CORS is configured."
      );
    }

    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
};

// --------------------------------------------------
// 7. GET PROCESSING RESULTS
// --------------------------------------------------

/**
 * Optional endpoint.
 * Enable this only if the backend implements:
 * GET /api/results/{job_id}
 */

export const getProcessingResults = async (jobId) => {
  if (!jobId) {
    throw new Error("Processing job ID is required.");
  }

  return apiRequest(
    `${API_BASE_PATH}/results/${encodeURIComponent(jobId)}`,
    {
      method: "GET",
    }
  );
};

// --------------------------------------------------
// 8. DOWNLOAD HELPERS
// --------------------------------------------------

/**
 * Download a file from backend.
 */
export const downloadFile = async (
  fileUrl,
  filename = "download"
) => {
  if (!fileUrl) {
    throw new Error("Download URL is unavailable.");
  }

  try {
    const response = await fetch(getFileUrl(fileUrl));

    if (!response.ok) {
      throw new Error(
        `Download failed: ${response.status}`
      );
    }

    const blob = await response.blob();

    const objectUrl = window.URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.href = objectUrl;
    link.download = filename;

    document.body.appendChild(link);

    link.click();

    link.remove();

    window.URL.revokeObjectURL(objectUrl);

    return true;
  } catch (error) {
    throw new Error(
      `Unable to download file: ${error.message}`
    );
  }
};

// --------------------------------------------------
// 9. EXPORT API CONFIGURATION
// --------------------------------------------------

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  API_PATH: API_BASE_PATH,
  HEALTH_URL: getApiUrl("/health"),
  PROCESS_URL: getApiUrl(`${API_BASE_PATH}/process`),
};

// --------------------------------------------------
// 10. DEFAULT EXPORT
// --------------------------------------------------

const api = {
  healthCheck,
  getBackendStatus,
  processVideo,
  getProcessingResults,
  downloadFile,
  getFileUrl,
  API_CONFIG,
};

export default api;
