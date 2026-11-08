
/*
========================================================
 PragyanAI - People Detection & Counting
 API Service
========================================================

Frontend : React + Vite + Netlify
Backend  : FastAPI + Render
Model    : YOLO11 TFLite
Tracking : ByteTrack
========================================================
*/

// ======================================================
// 1. API CONFIGURATION
// ======================================================

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "https://pragyanai-yolo-tf-lite-object-detction.onrender.com"
).replace(/\/+$/, "");

const API_BASE_PATH = "/api";

const MAX_FILE_SIZE = 500 * 1024 * 1024;

const ALLOWED_EXTENSIONS = [
  "mp4",
  "avi",
  "mov",
  "mkv",
  "webm",
];

const REQUEST_TIMEOUT = 10 * 60 * 1000;


// ======================================================
// 2. API URL HELPER
// ======================================================

export const getApiUrl = (endpoint) => {
  const cleanEndpoint = endpoint.startsWith("/")
    ? endpoint
    : `/${endpoint}`;

  return `${API_BASE_URL}${cleanEndpoint}`;
};


// ======================================================
// 3. OUTPUT FILE URL HELPER
// ======================================================

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


// ======================================================
// 4. ERROR MESSAGE HELPER
// ======================================================

const extractErrorMessage = (data, fallback) => {
  if (!data) return fallback;

  if (typeof data.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item) => item.msg || JSON.stringify(item))
      .join(", ");
  }

  if (typeof data.message === "string") {
    return data.message;
  }

  return fallback;
};


// ======================================================
// 5. RESPONSE PARSER
// ======================================================

const parseResponse = async (response) => {
  const contentType =
    response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return await response.json();
  }

  const text = await response.text();

  return {
    message: text || response.statusText,
  };
};


// ======================================================
// 6. GENERIC API REQUEST
// ======================================================

const apiRequest = async (
  endpoint,
  options = {},
  timeout = 30000
) => {
  const controller = new AbortController();

  const timeoutId = setTimeout(
    () => controller.abort(),
    timeout
  );

  try {
    const response = await fetch(
      getApiUrl(endpoint),
      {
        ...options,
        signal: controller.signal,
      }
    );

    const data = await parseResponse(response);

    if (!response.ok) {
      throw new Error(
        extractErrorMessage(
          data,
          `Server error: ${response.status}`
        )
      );
    }

    return data;

  } catch (error) {

    if (error.name === "AbortError") {
      throw new Error(
        "Backend request timed out. Please retry."
      );
    }

    if (
      error instanceof TypeError ||
      error.message === "Failed to fetch"
    ) {
      throw new Error(
        "Cannot connect to Render backend. Check backend availability, API URL and CORS configuration."
      );
    }

    throw error;

  } finally {
    clearTimeout(timeoutId);
  }
};


// ======================================================
// 7. BACKEND HEALTH CHECK
// ======================================================

export const healthCheck = async () => {
  try {
    const data = await apiRequest(
      "/health",
      { method: "GET" },
      30000
    );

    return {
      success: true,
      connected: true,
      modelLoaded: data.model_loaded === true,
      data,
    };

  } catch (error) {

    return {
      success: false,
      connected: false,
      modelLoaded: false,
      error: error.message,
    };
  }
};


// ======================================================
// 8. BACKEND ROOT STATUS
// ======================================================

export const getBackendStatus = async () => {
  return apiRequest(
    "/",
    { method: "GET" },
    30000
  );
};


// ======================================================
// 9. MODEL INFORMATION
// ======================================================

export const getModelInfo = async () => {
  return apiRequest(
    `${API_BASE_PATH}/model-info`,
    { method: "GET" },
    30000
  );
};


// ======================================================
// 10. VALIDATE VIDEO FILE
// ======================================================

export const validateVideoFile = (file) => {

  if (!file) {
    throw new Error("Please select a video file.");
  }

  if (
    typeof file.name !== "string" ||
    typeof file.size !== "number"
  ) {
    throw new Error(
      "Invalid video file object. Please select the video again."
    );
  }

  if (file.size <= 0) {
    throw new Error("The selected video file is empty.");
  }

  const extension = file.name
    .split(".")
    .pop()
    .toLowerCase();

  if (!ALLOWED_EXTENSIONS.includes(extension)) {
    throw new Error(
      "Unsupported video format. Supported formats: MP4, AVI, MOV, MKV and WEBM."
    );
  }

  if (file.size > MAX_FILE_SIZE) {
    throw new Error(
      "Video exceeds the maximum upload size of 500 MB."
    );
  }

  return true;
};


// ======================================================
// 11. PROCESS VIDEO
// ======================================================

export const processVideo = async (
  file,
  confidence = 0.35,
  frameInterval = 1,
  onProgress = null
) => {

  // Validate actual browser File object
  validateVideoFile(file);

  if (
    !Number.isFinite(Number(confidence)) ||
    Number(confidence) < 0 ||
    Number(confidence) > 1
  ) {
    throw new Error(
      "Confidence must be between 0 and 1."
    );
  }

  if (
    !Number.isInteger(Number(frameInterval)) ||
    Number(frameInterval) < 1
  ) {
    throw new Error(
      "Frame interval must be a positive integer."
    );
  }

  // Prepare multipart upload
  const formData = new FormData();

  formData.append("file", file, file.name);

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

    console.info("PragyanAI video upload started");

    console.info("Backend URL:", API_BASE_URL);

    console.info("Video filename:", file.name);

    console.info("Video size:", file.size);

    console.info("Video type:", file.type);

    // IMPORTANT:
    // Do not manually set Content-Type.
    // Browser automatically adds multipart boundary.

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
        extractErrorMessage(
          data,
          `Video processing failed: HTTP ${response.status}`
        )
      );
    }

    if (data.success === false) {
      throw new Error(
        data.message || "Video processing failed."
      );
    }

    const videoUrl =
      data.video_url ||
      data.output_video_url ||
      data.processed_video_url ||
      null;

    const csvUrl =
      data.csv_url ||
      data.csv_file_url ||
      null;

    const jsonUrl =
      data.json_url ||
      data.json_file_url ||
      null;

    if (!videoUrl) {
      throw new Error(
        "Backend did not return a processed video URL."
      );
    }

    if (typeof onProgress === "function") {
      onProgress(100);
    }

    console.info("Video processing completed", data);

    return {
      ...data,

      success: true,

      video_url: getFileUrl(videoUrl),

      output_video_url: getFileUrl(videoUrl),

      csv_url: csvUrl
        ? getFileUrl(csvUrl)
        : null,

      csv_file_url: csvUrl
        ? getFileUrl(csvUrl)
        : null,

      json_url: jsonUrl
        ? getFileUrl(jsonUrl)
        : null,

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

    console.error(
      "PragyanAI processing error:",
      error
    );

    if (error.name === "AbortError") {

      throw new Error(
        "Processing request timed out. The Render service may have stopped or the video may take longer to process."
      );
    }

    if (
      error instanceof TypeError ||
      error.message === "Failed to fetch"
    ) {

      throw new Error(
        "Cannot connect to the backend. Check Render status, API URL and CORS configuration."
      );
    }

    throw error;

  } finally {

    clearTimeout(timeoutId);

  }
};


// ======================================================
// 12. OPTIONAL PROCESSING RESULTS
// ======================================================

// Requires a corresponding backend route.
// Do not call until that route is implemented.

export const getProcessingResults = async (jobId) => {

  if (!jobId) {
    throw new Error("Processing job ID is required.");
  }

  return apiRequest(
    `${API_BASE_PATH}/results/${encodeURIComponent(jobId)}`,
    { method: "GET" }
  );
};


// ======================================================
// 13. DOWNLOAD FILE
// ======================================================

export const downloadFile = async (
  fileUrl,
  filename = "download"
) => {

  if (!fileUrl) {
    throw new Error("Download URL is unavailable.");
  }

  const response = await fetch(
    getFileUrl(fileUrl)
  );

  if (!response.ok) {
    throw new Error(
      `Download failed: HTTP ${response.status}`
    );
  }

  const blob = await response.blob();

  const objectUrl = URL.createObjectURL(blob);

  const link = document.createElement("a");

  link.href = objectUrl;
  link.download = filename;

  document.body.appendChild(link);

  link.click();

  link.remove();

  URL.revokeObjectURL(objectUrl);

  return true;
};


// ======================================================
// 14. API CONFIGURATION
// ======================================================

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  API_PATH: API_BASE_PATH,

  HEALTH_URL: getApiUrl("/health"),

  PROCESS_URL: getApiUrl(
    `${API_BASE_PATH}/process`
  ),

  MODEL_INFO_URL: getApiUrl(
    `${API_BASE_PATH}/model-info`
  ),

  MAX_FILE_SIZE,
  ALLOWED_EXTENSIONS,
};


// ======================================================
// 15. DEFAULT EXPORT
// ======================================================

const api = {
  healthCheck,
  getBackendStatus,
  getModelInfo,
  validateVideoFile,
  processVideo,
  getProcessingResults,
  downloadFile,
  getFileUrl,
  getApiUrl,
  API_CONFIG,
};

export default api;
