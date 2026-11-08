import React, { useState } from "react";
import {
  Video,
  ExternalLink,
  AlertCircle,
  PlayCircle,
  Download,
  CheckCircle,
} from "lucide-react";

export default function ResultsVideo({ videoUrl }) {
  const [videoError, setVideoError] = useState(false);
  const [videoLoaded, setVideoLoaded] = useState(false);

  if (!videoUrl) {
    return null;
  }

  const handleVideoError = () => {
    setVideoError(true);
    setVideoLoaded(false);
  };

  const handleVideoLoaded = () => {
    setVideoError(false);
    setVideoLoaded(true);
  };

  return (
    <section className="results-card">

      {/* ================= VIDEO HEADER ================= */}

      <div className="results-video-header">

        <div>
          <h3>
            <Video size={19} />
            Processed Video
          </h3>

          <p>
            YOLO11 people detection and ByteTrack tracking results
          </p>
        </div>

        <div className="video-result-status">
          <CheckCircle size={15} />
          <span>Processing Complete</span>
        </div>

      </div>


      {/* ================= VIDEO PLAYER ================= */}

      {!videoError ? (

        <div className="results-video-container">

          {!videoLoaded && (
            <div className="video-loading-overlay">
              <PlayCircle size={35} />
              <p>Loading processed video...</p>
            </div>
          )}

          <video
            className="results-video"
            src={videoUrl}
            controls
            playsInline
            preload="metadata"
            onLoadedMetadata={handleVideoLoaded}
            onError={handleVideoError}
          >
            Your browser does not support HTML video.
          </video>

        </div>

      ) : (

        <div className="video-preview-error">

          <AlertCircle size={32} />

          <h4>Unable to Preview Video</h4>

          <p>
            The processed video could not be played in this browser.
            You can still try opening or downloading the video.
          </p>

        </div>

      )}


      {/* ================= VIDEO ACTIONS ================= */}

      <div className="results-video-actions">

        <a
          href={videoUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="video-open-link"
        >
          <ExternalLink size={15} />
          Open Video
        </a>

        <a
          href={videoUrl}
          download="pragyanai_people_detection.mp4"
          className="video-download-link"
        >
          <Download size={15} />
          Download MP4
        </a>

      </div>


      {/* ================= VIDEO FOOTER ================= */}

      <div className="results-video-footer">

        <span>
          <CheckCircle size={13} />
          YOLO11 Detection
        </span>

        <span>
          ByteTrack Tracking
        </span>

        <span>
          MP4 Output
        </span>

      </div>

    </section>
  );
}
