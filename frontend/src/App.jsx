
import React, {
  useState,
  useEffect,
  useCallback,
} from "react";

import {
  Activity,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Server,
  Cpu,
  Video,
  Users,
  Wifi,
  WifiOff,
} from "lucide-react";

// ============================================================
// COMPONENTS
// ============================================================

import Header from "./components/Header";
import VideoUpload from "./components/VideoUpload";
import VideoPreview from "./components/VideoPreview";
import DetectionSettings from "./components/DetectionSettings";
import ProcessingStatus from "./components/ProcessingStatus";
import ResultsVideo from "./components/ResultsVideo";
import AnalyticsCards from "./components/AnalyticsCards";
import DownloadReport from "./components/DownloadReport";

// ============================================================
// API SERVICES
// ============================================================

import {
  healthCheck,
  processVideo,
} from "./services/api";


// ============================================================
// APPLICATION CONFIGURATION
// ============================================================

const APP_CONFIG = {
  name: "PragyanAI People Detection & Counting",

  backend:
    "https://pragyanai-yolo-tf-lite-object-detction.onrender.com",

  defaultConfidence: 0.35,

  defaultFrameInterval: 1,

  maxFileSize: 500 * 1024 * 1024,
};


// ============================================================
// MAIN APPLICATION
// ============================================================

function App() {

  // ==========================================================
  // 1. APPLICATION STATE
  // ==========================================================

  const [videoFile, setVideoFile] = useState(null);

  const [confidence, setConfidence] = useState(
    APP_CONFIG.defaultConfidence
  );

  const [frameInterval, setFrameInterval] = useState(
    APP_CONFIG.defaultFrameInterval
  );

  const [isProcessing, setIsProcessing] = useState(false);

  const [processingStatus, setProcessingStatus] = useState("idle");

  const [processingProgress, setProcessingProgress] = useState(0);

  const [error, setError] = useState("");

  const [results, setResults] = useState(null);


  // ==========================================================
  // 2. BACKEND AND MODEL STATUS
  // ==========================================================

  const [backendStatus, setBackendStatus] = useState("checking");

  const [modelLoaded, setModelLoaded] = useState(false);

  const [backendMessage, setBackendMessage] = useState(
    "Checking backend connection..."
  );

  const [modelError, setModelError] = useState(null);


  // ==========================================================
  // 3. BACKEND HEALTH CHECK
  // ==========================================================

  const checkBackend = useCallback(async () => {

    setBackendStatus("checking");

    setBackendMessage("Connecting to Render backend...");

    try {

      const response = await healthCheck();

      console.info(
        "PragyanAI backend health response:",
        response
      );

      /*
       IMPORTANT FIX:

       Current api.js returns:

       {
         connected: true,
         model_loaded: true
       }

       Older versions may return:

       {
         success: true,
         modelLoaded: true
       }

       Support both response formats.
      */

      const connected =
        response?.connected === true ||
        response?.success === true;

      const loaded =
        response?.model_loaded === true ||
        response?.modelLoaded === true;

      if (connected) {

        setBackendStatus("connected");

        setModelLoaded(loaded);

        setModelError(
          response?.model_error || null
        );

        setBackendMessage(
          loaded
            ? "Backend connected. YOLO11 model loaded successfully."
            : "Backend connected, but YOLO11 model is not ready."
        );

        return {
          connected: true,
          model_loaded: loaded,
        };

      }

      setBackendStatus("disconnected");

      setModelLoaded(false);

      setModelError(null);

      setBackendMessage(
        response?.message ||
        response?.error ||
        "Backend connection failed."
      );

      return {
        connected: false,
        model_loaded: false,
      };

    } catch (err) {

      console.error(
        "Backend health check failed:",
        err
      );

      setBackendStatus("disconnected");

      setModelLoaded(false);

      setModelError(null);

      setBackendMessage(
        err.message ||
        "Cannot connect to Render backend."
      );

      return {
        connected: false,
        model_loaded: false,
      };

    }

  }, []);


  // ==========================================================
  // 4. INITIAL BACKEND CONNECTION
  // ==========================================================

  useEffect(() => {

    checkBackend();

  }, [checkBackend]);


  // ==========================================================
  // 5. FILE SELECTION
  // ==========================================================

  const handleFileSelect = (file) => {

    if (isProcessing) return;

    setError("");

    setResults(null);

    setProcessingProgress(0);

    setProcessingStatus("idle");

    if (!file) {

      setVideoFile(null);

      return;
    }

    // Check browser File object

    if (!(file instanceof File)) {

      setVideoFile(null);

      setError(
        "Invalid video file. Please select a video again."
      );

      return;
    }

    // Check file size

    if (file.size <= 0) {

      setVideoFile(null);

      setError(
        "The selected video file is empty."
      );

      return;
    }

    if (file.size > APP_CONFIG.maxFileSize) {

      setVideoFile(null);

      setError(
        "Video exceeds the maximum allowed size of 500 MB."
      );

      return;
    }

    // Supported formats

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

      setVideoFile(null);

      setError(
        "Unsupported video format. Upload MP4, AVI, MOV, MKV or WEBM."
      );

      return;
    }

    setVideoFile(file);

    console.info(
      "Video selected:",
      file.name
    );

  };


  // ==========================================================
  // 6. RESET APPLICATION
  // ==========================================================

  const resetApplication = () => {

    if (isProcessing) return;

    setVideoFile(null);

    setResults(null);

    setError("");

    setProcessingStatus("idle");

    setProcessingProgress(0);

    setConfidence(
      APP_CONFIG.defaultConfidence
    );

    setFrameInterval(
      APP_CONFIG.defaultFrameInterval
    );

  };


  // ==========================================================
  // 7. PROCESS VIDEO
  // ==========================================================

  const handleProcessVideo = async () => {

    if (isProcessing) return;

    setError("");

    setResults(null);

    setProcessingProgress(0);

    // --------------------------------------------------------
    // Validate video
    // --------------------------------------------------------

    if (!videoFile) {

      setError(
        "Please upload a video first."
      );

      return;
    }

    if (!(videoFile instanceof File)) {

      setError(
        "Invalid video file. Please select the video again."
      );

      return;
    }

    // --------------------------------------------------------
    // Set processing state
    // --------------------------------------------------------

    setIsProcessing(true);

    setProcessingStatus("processing");

    setProcessingProgress(10);

    try {

      console.info("================================");

      console.info(
        "PragyanAI People Detection Started"
      );

      console.info(
        "Video:",
        videoFile.name
      );

      console.info(
        "File size:",
        videoFile.size
      );

      console.info(
        "Confidence:",
        confidence
      );

      console.info(
        "Frame interval:",
        frameInterval
      );

      console.info("================================");

      // ------------------------------------------------------
      // Recheck backend before processing
      // ------------------------------------------------------

      const backend = await checkBackend();

      if (!backend.connected) {

        throw new Error(
          "Cannot connect to Render backend. Please check your connection."
        );

      }

      if (!backend.model_loaded) {

        throw new Error(
          "YOLO11 model is not ready on Render. Please check the backend logs."
        );

      }

      // ------------------------------------------------------
      // Send video to FastAPI
      // ------------------------------------------------------

      const result = await processVideo(
        videoFile,
        confidence,
        frameInterval,
        (progress) => {
          setProcessingProgress(progress);
        }
      );

      console.info(
        "Backend processing response:",
        result
      );

      // ------------------------------------------------------
      // Validate backend response
      // ------------------------------------------------------

      if (!result) {

        throw new Error(
          "No response received from the backend."
        );

      }

      if (result.success !== true) {

        throw new Error(
          result.message ||
          "Video processing could not be completed."
        );

      }

      if (!result.video_url) {

        throw new Error(
          "Processed video URL was not returned by the backend."
        );

      }

      // ------------------------------------------------------
      // Store processing results
      // ------------------------------------------------------

      setResults(result);

      setProcessingProgress(100);

      setProcessingStatus("completed");

      setError("");

      console.info(
        "PragyanAI processing completed successfully.",
        result
      );

    } catch (err) {

      console.error(
        "Video processing failed:",
        err
      );

      setError(
        err.message ||
        "Video processing failed. Please try again."
      );

      setProcessingStatus("failed");

      setProcessingProgress(0);

    } finally {

      setIsProcessing(false);

    }

  };


  // ==========================================================
  // 8. RETRY PROCESSING
  // ==========================================================

  const retryProcessing = () => {

    if (!videoFile) {

      setError(
        "Please upload a video first."
      );

      return;
    }

    handleProcessVideo();

  };


  // ==========================================================
  // 9. DERIVED STATUS
  // ==========================================================

  const isBackendConnected =
    backendStatus === "connected";

  const isBackendReady =
    isBackendConnected && modelLoaded;

  const modelStatusLabel =
    backendStatus === "checking"
      ? "Checking..."
      : !isBackendConnected
        ? "Unknown"
        : modelLoaded
          ? "Ready"
          : "Not Ready";


  // ==========================================================
  // 10. RENDER APPLICATION
  // ==========================================================

  return (

    <div className="app-container">

      {/* ==================================================
          HEADER
      ================================================== */}

      <Header />


      {/* ==================================================
          MAIN CONTENT
      ================================================== */}

      <main className="main-content">

        {/* ==================================================
            HERO SECTION
        ================================================== */}

        <section className="hero-section">

          <div className="hero-content">

            <div className="hero-badge">

              <Activity size={16} />

              AI-POWERED VIDEO ANALYTICS

            </div>

            <h1>
              People Detection
              <span> & Counting</span>
            </h1>

            <p>
              Upload your video, detect people,
              track individuals and generate
              intelligent video analytics using
              YOLO11 and ByteTrack.
            </p>

          </div>

        </section>


        {/* ==================================================
            BACKEND STATUS SECTION
        ================================================== */}

        <section className="backend-status-section">

          <div className="backend-status-card">

            <div className="backend-status-left">

              <div
                className={`backend-status-icon ${backendStatus}`}
              >

                {isBackendConnected ? (
                  <Server size={22} />
                ) : (
                  <WifiOff size={22} />
                )}

              </div>

              <div className="backend-status-text">

                <h3>Backend Connection</h3>

                <p>{backendMessage}</p>

              </div>

            </div>


            <div className="backend-status-right">

              <span
                className={`backend-status-badge ${backendStatus}`}
              >

                {backendStatus === "checking"
                  ? "Checking"
                  : isBackendConnected
                    ? "Connected"
                    : "Disconnected"}

              </span>

              <button
                type="button"
                className="refresh-status-button"
                onClick={checkBackend}
                disabled={backendStatus === "checking"}
                title="Refresh backend status"
              >

                <RefreshCw
                  size={17}
                  className={
                    backendStatus === "checking"
                      ? "spin"
                      : ""
                  }
                />

              </button>

            </div>

          </div>


          {/* ENGINE INFORMATION */}

          <div className="engine-status-grid">

            {/* Detection engine */}

            <div className="engine-status-item">

              <Cpu size={18} />

              <div>

                <span>Detection Engine</span>

                <strong>YOLO11 TFLite</strong>

              </div>

            </div>


            {/* Tracking engine */}

            <div className="engine-status-item">

              <Activity size={18} />

              <div>

                <span>Tracking Engine</span>

                <strong>ByteTrack</strong>

              </div>

            </div>


            {/* Video processing */}

            <div className="engine-status-item">

              <Video size={18} />

              <div>

                <span>Video Processing</span>

                <strong>OpenCV</strong>

              </div>

            </div>


            {/* Model status */}

            <div className="engine-status-item">

              {isBackendReady ? (
                <Wifi size={18} />
              ) : (
                <WifiOff size={18} />
              )}

              <div>

                <span>Model Status</span>

                <strong>{modelStatusLabel}</strong>

              </div>

            </div>

          </div>

        </section>


        {/* ==================================================
            VIDEO PROCESSING WORKSPACE
        ================================================== */}

        <section className="workspace-section">

          <div className="workspace-header">

            <div>

              <h2>Video Processing Workspace</h2>

              <p>
                Upload a video and configure
                your detection settings.
              </p>

            </div>

            <button
              type="button"
              className="reset-button"
              onClick={resetApplication}
              disabled={isProcessing}
            >

              <RefreshCw size={16} />

              Reset

            </button>

          </div>


          <div className="workspace-grid">

            {/* ==================================================
                LEFT COLUMN
            ================================================== */}

            <div className="workspace-left">

              {/* VIDEO UPLOAD */}

              <VideoUpload
                selectedFile={videoFile}
                onFileSelect={handleFileSelect}
                disabled={isProcessing}
              />


              {/* VIDEO PREVIEW */}

              {videoFile && (

                <VideoPreview
                  file={videoFile}
                  onRemove={() => {
                    handleFileSelect(null);
                  }}
                  disabled={isProcessing}
                />

              )}


              {/* DETECTION SETTINGS */}

              <DetectionSettings
                confidence={confidence}
                frameInterval={frameInterval}
                onConfidenceChange={setConfidence}
                onFrameIntervalChange={setFrameInterval}
                disabled={isProcessing}
              />


              {/* PROCESS VIDEO BUTTON */}

              <button
                type="button"
                className="process-button"
                onClick={handleProcessVideo}
                disabled={
                  !videoFile ||
                  isProcessing ||
                  !isBackendReady
                }
              >

                {isProcessing ? (

                  <>
                    <RefreshCw
                      size={20}
                      className="spin"
                    />

                    Processing Video...

                  </>

                ) : (

                  <>

                    <Activity size={20} />

                    Detect & Count People

                  </>

                )}

              </button>


              {/* BACKEND WARNING */}

              {!isBackendReady && (

                <div className="backend-warning">

                  <AlertCircle size={17} />

                  <span>

                    {backendStatus === "checking"
                      ? "Checking backend connection..."
                      : !isBackendConnected
                        ? "Backend is unavailable. Please check Render and retry connection."
                        : "YOLO11 detection model is not ready on Render."}

                  </span>

                </div>

              )}


              {/* MODEL ERROR */}

              {isBackendConnected && modelError && (

                <div className="backend-warning">

                  <AlertCircle size={17} />

                  <span>
                    Model error: {modelError}
                  </span>

                </div>

              )}

            </div>


            {/* ==================================================
                RIGHT COLUMN
            ================================================== */}

            <div className="workspace-right">

              <ProcessingStatus
                status={processingStatus}
                progress={processingProgress}
                error={error}
              />


              {/* PROCESSING ERROR */}

              {error && (

                <div className="processing-error">

                  <AlertCircle size={20} />

                  <div>

                    <strong>Processing Error</strong>

                    <p>{error}</p>

                  </div>

                </div>

              )}


              {/* PROCESSING SUCCESS */}

              {processingStatus === "completed" && (

                <div className="processing-success">

                  <CheckCircle size={20} />

                  <div>

                    <strong>Processing Completed</strong>

                    <p>
                      Your video has been processed successfully.
                    </p>

                  </div>

                </div>

              )}


              {/* RETRY BUTTON */}

              {processingStatus === "failed" && videoFile && (

                <button
                  type="button"
                  className="process-button"
                  onClick={retryProcessing}
                  disabled={isProcessing}
                >

                  <RefreshCw size={18} />

                  Retry Processing

                </button>

              )}

            </div>

          </div>

        </section>


        {/* ==================================================
            PROCESSING RESULTS
        ================================================== */}

        {results && (

          <section className="results-section">

            <div className="results-section-header">

              <div>

                <h2>Processing Results</h2>

                <p>
                  Detection, tracking and analytics
                  generated from your video.
                </p>

              </div>

              <span className="results-success-badge">

                <CheckCircle size={16} />

                Completed

              </span>

            </div>


            {/* PROCESSED VIDEO */}

            {results.video_url && (

              <ResultsVideo
                videoUrl={results.video_url}
              />

            )}


            {/* ANALYTICS */}

            <AnalyticsCards
              analytics={
                results.analytics ||
                results.summary ||
                {}
              }
            />


            {/* DOWNLOAD REPORTS */}

            <DownloadReport
              videoUrl={results.video_url}
              csvUrl={results.csv_url}
              jsonUrl={results.json_url}
              analytics={
                results.analytics ||
                results.summary ||
                {}
              }
            />

          </section>

        )}


        {/* ==================================================
            HOW IT WORKS
        ================================================== */}

        <section className="workflow-section">

          <div className="section-heading">

            <h2>How It Works</h2>

            <p>
              AI-powered people detection and video analytics.
            </p>

          </div>


          <div className="workflow-grid">

            {/* STEP 1 */}

            <div className="workflow-card">

              <div className="workflow-number">
                01
              </div>

              <Video size={26} />

              <h3>Upload Video</h3>

              <p>
                Upload an MP4, AVI, MOV, MKV or WEBM video.
              </p>

            </div>


            {/* STEP 2 */}

            <div className="workflow-card">

              <div className="workflow-number">
                02
              </div>

              <div className="workflow-icon-text">
                AI
              </div>

              <h3>Person Detection</h3>

              <p>
                YOLO11 TFLite detects people in video frames.
              </p>

            </div>


            {/* STEP 3 */}

            <div className="workflow-card">

              <div className="workflow-number">
                03
              </div>

              <Users size={26} />

              <h3>Person Tracking</h3>

              <p>
                ByteTrack maintains individual tracking IDs.
              </p>

            </div>


            {/* STEP 4 */}

            <div className="workflow-card">

              <div className="workflow-number">
                04
              </div>

              <Activity size={26} />

              <h3>Analytics & Reports</h3>

              <p>
                Generate annotated videos and people-counting reports.
              </p>

            </div>

          </div>

        </section>

      </main>


      {/* ==================================================
          FOOTER
      ================================================== */}

      <footer className="app-footer">

        <div className="footer-brand">

          <strong>PragyanAI</strong>

          <span>
            AI-Powered People Detection & Counting
          </span>

        </div>

        <div className="footer-engine">

          YOLO11 TFLite | ByteTrack | OpenCV

        </div>

        <div className="footer-copyright">

          © {new Date().getFullYear()} PragyanAI.
          All rights reserved.

        </div>

      </footer>

    </div>

  );

}


export default App;

