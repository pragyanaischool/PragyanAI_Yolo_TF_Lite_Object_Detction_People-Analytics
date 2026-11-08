
import React, { useState } from "react";

import Header from "./components/Header";
import VideoUpload from "./components/VideoUpload";
import VideoPreview from "./components/VideoPreview";
import DetectionSettings from "./components/DetectionSettings";
import ProcessingStatus from "./components/ProcessingStatus";
import ResultsVideo from "./components/ResultsVideo";
import AnalyticsCards from "./components/AnalyticsCards";
import DownloadReport from "./components/DownloadReport";

import { processVideo } from "./services/api";

function App() {
  // Uploaded video
  const [videoFile, setVideoFile] = useState(null);

  // Detection settings
  const [confidence, setConfidence] = useState(0.35);
  const [frameInterval, setFrameInterval] = useState(1);

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState("idle");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);

  // Backend response
  const [results, setResults] = useState(null);

  // Handle video selection
  const handleVideoSelect = (file) => {
    if (!file) return;

    setVideoFile(file);
    setResults(null);
    setError("");
    setProgress(0);
    setProcessingStatus("idle");
  };

  // Handle detection confidence change
  const handleConfidenceChange = (value) => {
    setConfidence(Number(value));
  };

  // Handle frame interval change
  const handleFrameIntervalChange = (value) => {
    setFrameInterval(Number(value));
  };

  // Process uploaded video
  const handleProcessVideo = async () => {
    if (!videoFile) {
      setError("Please upload a video before starting detection.");
      return;
    }

    try {
      setIsProcessing(true);
      setProcessingStatus("processing");
      setError("");
      setResults(null);
      setProgress(10);

      const response = await processVideo({
        file: videoFile,
        confidence,
        frameInterval
      });

      setProgress(100);
      setResults(response);
      setProcessingStatus("completed");
    } catch (err) {
      console.error("Video processing error:", err);

      setError(
        err?.message ||
        "Unable to process the video. Please check the backend server."
      );

      setProcessingStatus("failed");
    } finally {
      setIsProcessing(false);
    }
  };

  // Reset the application
  const handleReset = () => {
    setVideoFile(null);
    setResults(null);
    setError("");
    setProgress(0);
    setProcessingStatus("idle");
    setIsProcessing(false);
    setConfidence(0.35);
    setFrameInterval(1);
  };

  return (
    <div className="app-container">

      {/* Application Header */}
      <Header />

      <main className="main-content">

        {/* Page Introduction */}
        <section className="hero-section">
          <div className="hero-badge">
            <span className="status-dot"></span>
            AI-Powered Video Analytics
          </div>

          <h1>
            People Detection
            <span className="gradient-text"> & Counting</span>
          </h1>

          <p>
            Upload your video and automatically detect, track, and count
            people using YOLO11, TensorFlow Lite and ByteTrack.
          </p>

          <div className="technology-tags">
            <span>YOLO11</span>
            <span>TensorFlow Lite</span>
            <span>ByteTrack</span>
            <span>FastAPI</span>
          </div>
        </section>

        {/* Upload and Configuration Section */}
        <section className="workspace-grid">

          <div className="left-panel">

            {/* Video Upload */}
            <VideoUpload
              selectedFile={videoFile}
              onFileSelect={handleVideoSelect}
              disabled={isProcessing}
            />

            {/* Video Preview */}
            {videoFile && (
              <VideoPreview
                file={videoFile}
                onRemove={() => {
                  setVideoFile(null);
                  setResults(null);
                  setError("");
                  setProcessingStatus("idle");
                }}
                disabled={isProcessing}
              />
            )}

            {/* Detection Settings */}
            <DetectionSettings
              confidence={confidence}
              frameInterval={frameInterval}
              onConfidenceChange={handleConfidenceChange}
              onFrameIntervalChange={handleFrameIntervalChange}
              disabled={isProcessing}
            />

            {/* Process Button */}
            <button
              type="button"
              className="process-button"
              onClick={handleProcessVideo}
              disabled={!videoFile || isProcessing}
            >
              {isProcessing ? (
                <>
                  <span className="button-spinner"></span>
                  Processing Video...
                </>
              ) : (
                <>
                  <span>▶</span>
                  Start People Detection
                </>
              )}
            </button>

            {/* Reset Button */}
            <button
              type="button"
              className="reset-button"
              onClick={handleReset}
              disabled={isProcessing}
            >
              Reset Application
            </button>

          </div>

          {/* Processing Information */}
          <div className="right-panel">

            <div className="panel-heading">
              <h2>Processing Status</h2>
              <p>Monitor your video processing activity.</p>
            </div>

            <ProcessingStatus
              status={processingStatus}
              progress={progress}
              error={error}
            />

            {error && (
              <div className="error-message" role="alert">
                <strong>Processing Error</strong>
                <p>{error}</p>
              </div>
            )}

            <div className="workflow-card">
              <h3>Detection Workflow</h3>

              <div className="workflow-step">
                <span className="workflow-number">01</span>
                <div>
                  <strong>Video Upload</strong>
                  <p>Upload your video file.</p>
                </div>
              </div>

              <div className="workflow-step">
                <span className="workflow-number">02</span>
                <div>
                  <strong>Frame Extraction</strong>
                  <p>Extract frames from the uploaded video.</p>
                </div>
              </div>

              <div className="workflow-step">
                <span className="workflow-number">03</span>
                <div>
                  <strong>AI Detection & Tracking</strong>
                  <p>Detect and track people using YOLO11 and ByteTrack.</p>
                </div>
              </div>

              <div className="workflow-step">
                <span className="workflow-number">04</span>
                <div>
                  <strong>Analytics & Reports</strong>
                  <p>Generate people-counting analytics and reports.</p>
                </div>
              </div>
            </div>

          </div>
        </section>

        {/* Results Section */}
        {results && (
          <section className="results-section">

            <div className="section-heading">
              <div>
                <span className="section-eyebrow">ANALYSIS COMPLETE</span>
                <h2>Detection Results</h2>
                <p>
                  Review the annotated video and people-counting analytics.
                </p>
              </div>

              <span className="success-label">
                Completed
              </span>
            </div>

            {/* Analytics Cards */}
            <AnalyticsCards
              analytics={results.analytics || results.summary || results}
            />

            {/* Processed Video */}
            <ResultsVideo
              videoUrl={results.video_url || results.output_video_url}
            />

            {/* Download Reports */}
            <DownloadReport
              videoUrl={results.video_url || results.output_video_url}
              csvUrl={results.csv_url || results.csv_file_url}
              results={results}
            />

          </section>
        )}

        {/* Footer */}
        <footer className="app-footer">
          <p>
            Powered by <strong>PragyanAI</strong>
          </p>

          <p>
            YOLO11 | TensorFlow Lite | ByteTrack | FastAPI | React
          </p>

          <p>
            AI Engineering & Intelligent Video Analytics
          </p>
        </footer>

      </main>
    </div>
  );
}

export default App;
