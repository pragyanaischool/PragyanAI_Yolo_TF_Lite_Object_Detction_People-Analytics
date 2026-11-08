
/**
 * PragyanAI - People Detection & Counting
 * API Service
 *
 * Frontend: React + Vite (Netlify)
 * Backend: FastAPI (Render)
 */

// --------------------------------------------------
// 1. Backend API Configuration
// --------------------------------------------------

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000"
).replace(/\/+$/, "");

const API_BASE_PATH = "/api";

// --------------------------------------------------
// 2. Common API Helper
// --------------------------------------------------

async function handleResponse(response) {
  let data;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      `API request failed with status ${response.status}`;

    throw new Error(
      typeof message === "string"
        ? message
        : JSON.stringify(message)
    );
  }

  return data;
}

// --------------------------------------------------
// 3. Health Check
// --------------------------------------------------

export async function checkBackendHealth() {
  try {
    const response = await fetch(
      `${API_BASE_URL}/health`,
      {
        method: "GET",
        headers: {
          Accept: "application/json"
        }
      }
    );

    return await handleResponse(response);

  } catch (error) {
    console.error("Backend health check failed:", error);
    throw new Error(
      "Unable to connect to the PragyanAI backend."
    );
  }
}

// --------------------------------------------------
// 4. Process Uploaded Video
// --------------------------------------------------

export async function processVideo({
  file,
  confidence = 0.35,
  frameInterval = 1,
  onProgress
}) {
  if (!file) {
    throw new Error("Please select a video file.");
  }

  if (!(file instanceof File)) {
    throw new Error("Invalid video file.");
  }

  const formData = new FormData();

  // Append uploaded video
  formData.append("file", file);

  // Append detection configuration
  formData.append(
    "confidence",
    String(confidence)
  );

  formData.append(
    "frame_interval",
    String(frameInterval)
  );

  try {
    onProgress?.(10);

    const response = await fetch(
      `${API_BASE_URL}${API_BASE_PATH}/process`,
      {
        method: "POST",
        body: formData
      }
    );

    onProgress?.(90);

    const result = await handleResponse(response);

    onProgress?.(100);

    return normalizeProcessingResult(result);

  } catch (error) {
    console.error("Video processing failed:", error);

    if (error instanceof TypeError) {
      throw new Error(
        "Cannot connect to the backend. Please verify that the Render API is running and the API URL is correct."
      );
    }

    throw error;
  }
}

// --------------------------------------------------
// 5. Normalize Backend Response
// --------------------------------------------------

function normalizeProcessingResult(data) {
  if (!data || typeof data !== "object") {
    throw new Error("Invalid response received from backend.");
  }

  const videoPath =
    data.video_url ||
    data.output_video_url ||
    data.output_video ||
    data.video;

  const csvPath =
    data.csv_url ||
    data.csv_file_url ||
    data.csv_file ||
    data.csv;

  return {
    ...data,

    // Convert relative paths into absolute backend URLs
    video_url: buildBackendUrl(videoPath),

    csv_url: buildBackendUrl(csvPath),

    // Preserve backend analytics and summary
    summary: data.summary || {},

    analytics: data.analytics || data.summary || {},

    tracking_details: data.tracking_details || {}
  };
}

// --------------------------------------------------
// 6. Build Backend URL
// --------------------------------------------------

export function buildBackendUrl(path) {
  if (!path) {
    return "";
  }

  // Already an absolute URL
  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  // Handle root-relative paths
  const normalizedPath = path.startsWith("/")
    ? path
    : `/${path}`;

  return `${API_BASE_URL}${normalizedPath}`;
}

// --------------------------------------------------
// 7. Download Processed Video
// --------------------------------------------------

export function getVideoDownloadUrl(videoPath) {
  return buildBackendUrl(videoPath);
}

// --------------------------------------------------
// 8. Download CSV Report
// --------------------------------------------------

export function getCSVDownloadUrl(csvPath) {
  return buildBackendUrl(csvPath);
}

// --------------------------------------------------
// 9. Fetch Processing Results
// --------------------------------------------------

export async function getProcessingResults(resultId) {
  if (!resultId) {
    throw new Error("Result ID is required.");
  }

  const response = await fetch(
    `${API_BASE_URL}${API_BASE_PATH}/results/${encodeURIComponent(resultId)}`,
    {
      method: "GET",
      headers: {
        Accept: "application/json"
      }
    }
  );

  return await handleResponse(response);
}

// --------------------------------------------------
// 10. Export API Configuration
// --------------------------------------------------

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  API_PATH: API_BASE_PATH
};
