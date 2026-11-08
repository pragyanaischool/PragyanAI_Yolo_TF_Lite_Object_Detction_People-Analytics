/*
===========================================================
PragyanAI - People Detection & Counting
File: frontend/src/services/api.js
===========================================================

Backend:
    FastAPI on Render

Detection:
    YOLO11 TFLite

Tracking:
    ByteTrack

Video Processing:
    OpenCV

Features:
    1. Backend health check
    2. Model readiness check
    3. Video upload
    4. Asynchronous video processing
    5. Job status polling
    6. Processing results
    7. Output URL handling
    8. MP4, CSV and JSON downloads
===========================================================
*/


// =========================================================
// 1. API CONFIGURATION
// =========================================================

const DEFAULT_API_URL =
  "https://pragyanai-yolo-tf-lite-object-detction.onrender.com";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_URL
).trim().replace(/\/+$/, "");

const API_URL = `${API_BASE_URL}/api`;

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  API_URL,
  HEALTH_URL: `${API_BASE_URL}/health`,
  ROOT_URL: API_BASE_URL,
  MODEL_INFO_URL: `${API_URL}/model-info`,
  PROCESS_URL: `${API_URL}/process`,
  JOBS_URL: `${API_URL}/jobs`,
  OUTPUTS_URL: `${API_URL}/outputs`,
  MAX_UPLOAD_SIZE: 500 * 1024 * 1024,
  ALLOWED_EXTENSIONS: [
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm",
  ],
};

console.info(
  "PragyanAI Backend URL:",
  API_BASE_URL
);


// =========================================================
// 2. COMMON API ERROR HANDLER
// =========================================================

const parseResponse = async (response) => {
  const contentType =
    response.headers.get("content-type") || "";

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

    throw new Error(message);
  }

  return data;
};


// =========================================================
// 3. BACKEND HEALTH CHECK
// =========================================================

export const healthCheck = async () => {
  try {
    const response = await fetch(
      API_CONFIG.HEALTH_URL,
      {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
        cache: "no-store",
      }
    );

    const data = await parseResponse(response);

    const connected =
      data.api === "running" ||
      data.backend_connected === true ||
      data.status === "healthy" ||
      data.status === "degraded";

    const modelLoaded =
      data.model_loaded === true;

    return {
      connected,
      model_loaded: modelLoaded,
      model_status: modelLoaded
        ? "ready"
        : connected
          ? "not_ready"
          : "unknown",

      status: data.status || "unknown",

      api: data.api || null,

      model_exists: data.model_exists ?? null,

      model_path: data.model_path || null,

      model_error: data.model_error || null,

      detection_engine:
        data.detection_engine || "YOLO11",

      tracking_engine:
        data.tracking_engine || "ByteTrack",

      inference_runtime:
        data.inference_runtime || "TensorFlow Lite",

      message: modelLoaded
        ? "Backend connected. YOLO11 model is ready."
        : connected
          ? "Backend connected, but model is not ready."
          : "Backend connection could not be verified.",

      raw: data,
    };

  } catch (error) {
    console.error(
      "PragyanAI backend health check failed:",
      error
    );

    return {
      connected: false,
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


// =========================================================
// 4. BACKEND STATUS
// =========================================================

export const getBackendStatus = async () => {
  return await healthCheck();
};


// =========================================================
// 5. GET MODEL INFORMATION
// =========================================================

export const getModelInfo = async () => {
  try {
    const response = await fetch(
      API_CONFIG.MODEL_INFO_URL,
      {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
        cache: "no-store",
      }
    );

    return await parseResponse(response);

  } catch (error) {
    console.error(
      "Model information error:",
      error
    );

    throw error;
  }
};


// =========================================================
// 6. VIDEO FILE VALIDATION
// =========================================================

export const validateVideoFile = (file) => {

  if (!file) {
    throw new Error(
      "Please select a video file."
    );
  }

  if (!(file instanceof File)) {
    throw new Error(
      "Invalid video file."
    );
  }

  const extension = file.name
    .split(".")
    .pop()
    .toLowerCase();

  if (
    !API_CONFIG.ALLOWED_EXTENSIONS.includes(extension)
  ) {
    throw new Error(
      "Unsupported video format. Please upload MP4, AVI, MOV, MKV or WEBM."
    );
  }

  if (file.size === 0) {
    throw new Error(
      "The selected video file is empty."
    );
  }

  if (
    file.size > API_CONFIG.MAX_UPLOAD_SIZE
  ) {
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


// =========================================================
// 7. PROCESS VIDEO - ASYNCHRONOUS JOB
// =========================================================

const sleep = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));


export const processVideo = async (
  file,
  confidence = 0.35,
  frameInterval = 1,
  onProgress = null
) => {

  // -------------------------------------------------------
  // 7.1 Validate video
  // -------------------------------------------------------

  validateVideoFile(file);


  // -------------------------------------------------------
  // 7.2 Validate confidence
  // -------------------------------------------------------

  if (
    !Number.isFinite(Number(confidence)) ||
    Number(confidence) < 0 ||
    Number(confidence) > 1
  ) {
    throw new Error(
      "Confidence must be between 0 and 1."
    );
  }


  // -------------------------------------------------------
  // 7.3 Validate frame interval
  // -------------------------------------------------------

  if (
    !Number.isInteger(Number(frameInterval)) ||
    Number(frameInterval) < 1
  ) {
    throw new Error(
      "Frame interval must be a positive integer."
    );
  }


  // -------------------------------------------------------
  // 7.4 Prepare multipart request
  // -------------------------------------------------------

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


  if (onProgress) {
    onProgress(5);
  }


  console.info(
    "Creating PragyanAI video-processing job:",
    file.name
  );


  // -------------------------------------------------------
  // 7.5 Create processing job
  // -------------------------------------------------------

  let createResponse;

  try {

    createResponse = await fetch(
      API_CONFIG.JOBS_URL,
      {
        method: "POST",
        body: formData,
      }
    );

  } catch (error) {

    throw new Error(
      `Could not reach the backend while uploading the video. Check the Render service and network connection. ${error.message || ""}`
    );

  }


  const createdJob = await parseResponse(
    createResponse
  );


  if (!createdJob?.job_id) {

    throw new Error(
      "Backend did not return a valid processing job ID."
    );

  }


  const jobId = createdJob.job_id;


  console.info(
    "PragyanAI job created:",
    jobId
  );


  if (onProgress) {
    onProgress(
      createdJob.progress ?? 5
    );
  }


  // -------------------------------------------------------
  // 7.6 Poll job status
  // -------------------------------------------------------

  const pollIntervalMs = 2000;

  const maxWaitMs =
    3 * 60 * 60 * 1000;

  const startedAt = Date.now();

  let transientFailures = 0;


  while (
    Date.now() - startedAt < maxWaitMs
  ) {

    await sleep(pollIntervalMs);

    let statusResponse;


    // -----------------------------------------------------
    // Request job status
    // -----------------------------------------------------

    try {

      statusResponse = await fetch(
        `${API_CONFIG.JOBS_URL}/${encodeURIComponent(jobId)}`,
        {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
          cache: "no-store",
        }
      );

      transientFailures = 0;

    } catch (error) {

      transientFailures += 1;

      console.warn(
        `Job status polling network error (${transientFailures}/5):`,
        error
      );


      if (transientFailures >= 5) {

        throw new Error(
          "Lost connection to the backend while checking job progress. The job may still be running; check the Render service before submitting the video again."
        );

      }

      continue;

    }


    const status = await parseResponse(
      statusResponse
    );


    // -----------------------------------------------------
    // Update frontend progress
    // -----------------------------------------------------

    if (onProgress) {

      onProgress(
        status.status === "completed"
          ? 100
          : status.status === "queued"
            ? Math.max(
                5,
                Number(status.progress) || 5
              )
            : Math.max(
                10,
                Number(status.progress) || 10
              )
      );

    }


    // -----------------------------------------------------
    // Failed job
    // -----------------------------------------------------

    if (status.status === "failed") {

      throw new Error(
        status.error ||
        status.message ||
        "Video processing failed."
      );

    }


    // -----------------------------------------------------
    // Completed job
    // -----------------------------------------------------

    if (status.status === "completed") {

      const result = status.result;


      if (!result?.success) {

        throw new Error(
          "The job completed but the backend returned no valid result."
        );

      }


      console.info(
        "PragyanAI video processing completed:",
        result
      );


      return {

        ...result,

        job_id: jobId,

        video_url: getFileUrl(
          result.video_url
        ),

        csv_url: getFileUrl(
          result.csv_url
        ),

        json_url: getFileUrl(
          result.json_url
        ),

        analytics:
          result.analytics ||
          result.summary ||
          {},

        summary:
          result.summary ||
          result.analytics ||
          {},

      };

    }

  }


  // -------------------------------------------------------
  // 7.7 Processing timeout
  // -------------------------------------------------------

  throw new Error(
    `Video processing is taking longer than expected. Job ID: ${jobId}. Check the job status before submitting the same video again.`
  );

};


// =========================================================
// 8. PROCESSING RESULTS
// =========================================================

export const getProcessingResults = async (
  jobId
) => {

  if (!jobId) {

    throw new Error(
      "Job ID is required."
    );

  }


  const response = await fetch(
    `${API_CONFIG.JOBS_URL}/${encodeURIComponent(jobId)}/analytics`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      cache: "no-store",
    }
  );


  return parseResponse(response);

};


// =========================================================
// 9. OUTPUT FILE URL
// =========================================================

export const getFileUrl = (url) => {

  if (!url) {
    return null;
  }


  if (
    url.startsWith("http://") ||
    url.startsWith("https://")
  ) {
    return url;
  }


  return `${API_BASE_URL}${
    url.startsWith("/") ? "" : "/"
  }${url}`;

};


// =========================================================
// 10. DOWNLOAD FILE
// =========================================================

export const downloadFile = async (
  fileUrl,
  filename = "download"
) => {

  if (!fileUrl) {

    throw new Error(
      "Download URL is not available."
    );

  }


  const url = getFileUrl(fileUrl);

  const response = await fetch(url);


  if (!response.ok) {

    throw new Error(
      `Download failed with HTTP ${response.status}`
    );

  }


  const blob = await response.blob();

  const objectUrl = URL.createObjectURL(blob);


  try {

    const link = document.createElement("a");

    link.href = objectUrl;

    link.download = filename;

    document.body.appendChild(link);

    link.click();

    link.remove();

  } finally {

    // Allow the browser to start reading the blob.
    setTimeout(() => {
      URL.revokeObjectURL(objectUrl);
    }, 1000);

  }

};


// =========================================================
// 11. DOWNLOAD PROCESSED VIDEO
// =========================================================

export const downloadProcessedVideo = async (
  videoUrl
) => {

  return downloadFile(
    videoUrl,
    "pragyanai_people_detection.mp4"
  );

};


// =========================================================
// 12. DOWNLOAD CSV REPORT
// =========================================================

export const downloadCSVReport = async (
  csvUrl
) => {

  return downloadFile(
    csvUrl,
    "pragyanai_people_analytics.csv"
  );

};


// =========================================================
// 13. DOWNLOAD JSON REPORT
// =========================================================

export const downloadJSONReport = async (
  jsonUrl
) => {

  return downloadFile(
    jsonUrl,
    "pragyanai_people_analytics.json"
  );

};


// =========================================================
// 14. GET OUTPUT FILES
// =========================================================

export const getOutputFiles = async () => {

  const response = await fetch(
    API_CONFIG.OUTPUTS_URL,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    }
  );


  return parseResponse(response);

};


// =========================================================
// 15. DELETE OUTPUT FILE
// =========================================================

export const deleteOutputFile = async (
  filename
) => {

  if (!filename) {

    throw new Error(
      "Filename is required."
    );

  }


  const response = await fetch(
    `${API_CONFIG.OUTPUTS_URL}/${encodeURIComponent(filename)}`,
    {
      method: "DELETE",
    }
  );


  return parseResponse(response);

};


// =========================================================
// 16. BACKEND CONNECTION TEST
// =========================================================

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


// =========================================================
// 17. DEFAULT EXPORT
// =========================================================

const api = {

  API_CONFIG,

  healthCheck,

  getBackendStatus,

  getModelInfo,

  validateVideoFile,

  processVideo,

  getProcessingResults,

  getFileUrl,

  downloadFile,

  downloadProcessedVideo,

  downloadCSVReport,

  downloadJSONReport,

  getOutputFiles,

  deleteOutputFile,

  testBackendConnection,

};


export default api;
