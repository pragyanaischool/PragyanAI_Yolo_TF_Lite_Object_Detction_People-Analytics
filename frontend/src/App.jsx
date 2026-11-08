
import React, { useState, useEffect, useCallback } from "react";

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

// Components
import Header from "./components/Header";
import VideoUpload from "./components/VideoUpload";
import VideoPreview from "./components/VideoPreview";
import DetectionSettings from "./components/DetectionSettings";
import ProcessingStatus from "./components/ProcessingStatus";
import ResultsVideo from "./components/ResultsVideo";
import AnalyticsCards from "./components/AnalyticsCards";
import DownloadReport from "./components/DownloadReport";

// API
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

  healthEndpoint: "/health",

  defaultConfidence: 0.35,

  defaultFrameInterval: 1,

  maxFileSize: 500 * 1024 * 1024,
};


// ============================================================
// APPLICATION
// ============================================================

function App() {

  // ----------------------------------------------------------
  // 1. STATE
  // ----------------------------------------------------------

  const [videoFile, setVideoFile] = useState(null);

  const [confidence, setConfidence] = useState(
    APP_CONFIG.defaultConfidence
  );

  const [frameInterval, setFrameInterval] = useState(
    APP_CONFIG.defaultFrameInterval
  );

  const [isProcessing, setIsProcessing] = useState(false);

  const [processingStatus, setProcessingStatus] = useState(
    "idle"
  );

  const [processingProgress, setProcessingProgress] = useState(0);

  const [error, setError] = useState("");

  const [results, setResults] = useState(null);

  const [backendStatus, setBackendStatus] = useState(
    "checking"
  );

  const [modelLoaded, setModelLoaded] = useState(false);

  const [backendMessage, setBackendMessage] = useState(
    "Checking backend connection..."
  );


  // ----------------------------------------------------------
  // 2. BACKEND HEALTH CHECK
  // ----------------------------------------------------------

  const checkBackend = useCallback(async () => {

    setBackendStatus("checking");

    setBackendMessage("Connecting to Render backend...");

    try {

      const response = await healthCheck();

      if (response.success) {

        setBackendStatus("connected");

        setModelLoaded(
          response.modelLoaded === true
        );

        setBackendMessage(
          response.modelLoaded
            ? "Backend connected. Detection model loaded."
            : "Backend connected, but detection model is not loaded."
        );

      } else {

        setBackendStatus("disconnected");

        setModelLoaded(false);

        setBackendMessage(
          response.error || "Backend connection failed."
        );

      }

    } catch (err) {

      setBackendStatus("disconnected");

      setModelLoaded(false);

      setBackendMessage(err.message);

    }

  }, []);


  // ----------------------------------------------------------
  // 3. INITIAL BACKEND CHECK
  // ----------------------------------------------------------

  useEffect(() => {

    checkBackend();

  }, [checkBackend]);


  // ----------------------------------------------------------
  // 4. FILE SELECTION
  // ----------------------------------------------------------

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

    if (!(file instanceof File)) {

      setVideoFile(null);

      setError(
        "Invalid video file. Please select a video again."
      );

      return;
    }

    if (file.size <= 0) {

      setVideoFile(null);

      setError("The selected video file is empty.");

      return;
    }

    if (file.size > APP_CONFIG.maxFileSize) {

      setVideoFile(null);

      setError(
        "Video exceeds the maximum allowed size of 500 MB."
      );

      return;
    }

    setVideoFile(file);

  };


  // ----------------------------------------------------------
  // 5. RESET APPLICATION
  // ----------------------------------------------------------

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


  // ----------------------------------------------------------
  // 6. PROCESSING FUNCTION
  // ----------------------------------------------------------

  const handleProcessVideo = async () => {

    if (isProcessing) return;

    setError("");

    setResults(null);

    setProcessingProgress(0);

    // ----------------------------------------
    // Validate file
    // ----------------------------------------

    if (!videoFile) {

      setError("Please upload a video first.");

      return;
    }

    if (!(videoFile instanceof File)) {

      setError(
        "Invalid video file object. Please select the video again."
      );

      return;
    }

    if (backendStatus === "disconnected") {

      setError(
        "Cannot connect to Render backend. Please check backend availability."
      );

      return;
    }

    // ----------------------------------------
    // Start processing
    // ----------------------------------------

    setIsProcessing(true);

    setProcessingStatus("processing");

    setProcessingProgress(10);

    try {

      console.info("================================");

      console.info(
        "PragyanAI People Detection Started"
      );

      console.info("Video:", videoFile.name);

      console.info("File size:", videoFile.size);

      console.info("Confidence:", confidence);

      console.info("Frame interval:", frameInterval);

      console.info("================================");

      // --------------------------------------
      // API request
      // --------------------------------------

      const result = await processVideo(
        videoFile,
        confidence,
        frameInterval
      );

      // --------------------------------------
      // Validate response
      // --------------------------------------

      if (!result || result.success !== true) {

        throw new Error(
          result?.message ||
          "Video processing could not be completed."
        );

      }

      if (!result.video_url) {

        throw new Error(
          "Processed video URL was not returned by the backend."
        );

      }

      // --------------------------------------
      // Store results
      // --------------------------------------

      setResults(result);

      setProcessingProgress(100);

      setProcessingStatus("completed");

      console.info(
        "PragyanAI processing completed",
        result
      );

    } catch (err) {

      console.error(
        "Processing failed:",
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


  // ----------------------------------------------------------
  // 7. RETRY PROCESSING
  // ----------------------------------------------------------

  const retryProcessing = () => {

    if (!videoFile) {

      setError("Please upload a video first.");

      return;
    }

    handleProcessVideo();

  };


  // ----------------------------------------------------------
  // 8. STATUS LABELS
  // ----------------------------------------------------------

  const isBackendConnected =
    backendStatus === "connected";

  const isBackendReady =
    isBackendConnected && modelLoaded;


  // ----------------------------------------------------------
  // 9. RENDER APPLICATION
  // ----------------------------------------------------------

  return (

    <div className="app-container">

      {/* ================================================
          HEADER
      ================================================= */}

      <Header />


      {/* ================================================
          MAIN CONTENT
      ================================================= */}

      <main className="main-content">

        {/* --------------------------------------------
            HERO SECTION
        -------------------------------------------- */}

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


        {/* --------------------------------------------
            BACKEND STATUS
        -------------------------------------------- */}

        <section className="backend-status-section">

          <div className="backend-status-card">

            <div className="backend-status-left">

              <div
                className={`backend-status-icon ${
                  backendStatus
                }`}
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
                className={`backend-status-badge ${
                  backendStatus
                }`}
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

                <RefreshCw size={17} />

              </button>

            </div>

          </div>


          {/* Engine information */}

          <div className="engine-status-grid">

            <div className="engine-status-item">

              <Cpu size={18} />

              <div>

                <span>Detection Engine</span>

                <strong>YOLO11 TFLite</strong>

              </div>

            </div>


            <div className="engine-status-item">

              <Activity size={18} />

              <div>

                <span>Tracking Engine</span>

                <strong>ByteTrack</strong>

              </div>

            </div>


            <div className="engine-status-item">

              <Video size={18} />

              <div>

                <span>Video Processing</span>

                <strong>OpenCV</strong>

              </div>

            </div>


            <div className="engine-status-item">

              {isBackendReady ? (
                <Wifi size={18} />
              ) : (
                <WifiOff size={18} />
              )}

              <div>

                <span>Model Status</span>

                <strong>
                  {modelLoaded
                    ? "Loaded"
                    : "Not Ready"}
                </strong>

              </div>

            </div>

          </div>

        </section>


        {/* ================================================
            PROCESSING WORKSPACE
        ================================================= */}

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

            {/* --------------------------------------
                LEFT COLUMN
            -------------------------------------- */}

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


              {/* PROCESS BUTTON */}

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


              {/* BACKEND NOT READY */}

              {!isBackendReady && (

                <div className="backend-warning">

                  <AlertCircle size={17} />

                  <span>
                    {backendStatus === "checking"
                      ? "Checking backend connection..."
                      : backendStatus === "disconnected"
                      ? "Backend is unavailable. Please retry connection."
                      : "Detection model is not loaded on Render."}
                  </span>

                </div>

              )}

            </div>


            {/* --------------------------------------
                RIGHT COLUMN
            -------------------------------------- */}

            <div className="workspace-right">

              <ProcessingStatus
                status={processingStatus}
                progress={processingProgress}
                error={error}
              />

              {/* ERROR MESSAGE */}

              {error && (

                <div className="processing-error">

                  <AlertCircle size={20} />

                  <div>

                    <strong>Processing Error</strong>

                    <p>{error}</p>

                  </div>

                </div>

              )}


              {/* SUCCESS MESSAGE */}

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

            </div>

          </div>

        </section>


        {/* ================================================
            RESULTS SECTION
        ================================================= */}

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


            {/* DOWNLOADS */}

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


        {/* ================================================
            WORKFLOW INFORMATION
        ================================================= */}

        <section className="workflow-section">

          <div className="section-heading">

            <h2>How It Works</h2>

            <p>
              AI-powered people detection and video analytics.
            </p>

          </div>


          <div className="workflow-grid">

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


      {/* ================================================
          FOOTER
      ================================================= */}

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

