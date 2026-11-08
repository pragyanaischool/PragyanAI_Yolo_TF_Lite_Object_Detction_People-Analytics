
import React, { useState } from "react";

import {
  Video,
  ExternalLink,
  AlertCircle
} from "lucide-react";

function ResultsVideo({ videoUrl }) {
  const [videoError, setVideoError] = useState(false);

  if (!videoUrl) {
    return null;
  }

  const handleVideoError = () => {
    setVideoError(true);
  };

  return (
    <section className="results-card">

      <div className="results-video-header">

        <div>
          <h3>
            <Video size={19} />
            Processed Video
          </h3>

          <p>
            Annotated video with person detection boxes,
            tracking IDs and counting information.
          </p>
        </div>

        {!videoError && (
          <a
            href={videoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="video-open-link"
          >
            Open Video
            <ExternalLink size={15} />
          </a>
        )}

      </div>

      {videoError ? (
        <div className="video-preview-error">
          <AlertCircle size={28} />

          <p>
            Unable to load the processed video.
            Please check whether the backend output file exists.
          </p>
        </div>
      ) : (
        <video
          key={videoUrl}
          className="results-video"
          src={videoUrl}
          controls
          playsInline
          preload="metadata"
          onError={handleVideoError}
        >
          Your browser does not support HTML5 video.
        </video>
      )}

      <div className="results-video-footer">
        <span>AI Annotated Output</span>
        <span>YOLO11 + ByteTrack</span>
      </div>

    </section>
  );
}

export default ResultsVideo;
