
import React from "react";

import {
  Settings,
  SlidersHorizontal,
  Gauge,
  Info
} from "lucide-react";

function DetectionSettings({
  confidence = 0.35,
  frameInterval = 1,
  onConfidenceChange,
  onFrameIntervalChange,
  disabled = false
}) {
  const handleConfidenceChange = (event) => {
    onConfidenceChange?.(Number(event.target.value));
  };

  const handleFrameIntervalChange = (event) => {
    onFrameIntervalChange?.(Number(event.target.value));
  };

  return (
    <section className="settings-card">

      {/* Heading */}
      <div className="settings-heading">
        <div className="settings-title-row">
          <Settings size={20} color="#2563eb" />

          <h2>Detection Settings</h2>
        </div>

        <p>
          Configure the AI detection parameters before processing.
        </p>
      </div>

      {/* Confidence Threshold */}
      <div className="setting-group">

        <div className="setting-label">
          <span className="setting-label-title">
            <SlidersHorizontal size={15} />
            Confidence Threshold
          </span>

          <span className="setting-value">
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>

        <input
          type="range"
          className="setting-slider"
          min="0.10"
          max="0.90"
          step="0.05"
          value={confidence}
          onChange={handleConfidenceChange}
          disabled={disabled}
          aria-label="Detection confidence threshold"
        />

        <div className="slider-labels">
          <span>10% - More detections</span>
          <span>90% - Higher confidence</span>
        </div>

        <p className="setting-description">
          Controls the minimum confidence required to accept a person
          detection. Higher values may reduce false detections but
          can miss some people.
        </p>

      </div>

      {/* Frame Interval */}
      <div className="setting-group">

        <div className="setting-label">
          <span className="setting-label-title">
            <Gauge size={15} />
            Frame Processing Interval
          </span>
        </div>

        <select
          className="setting-select"
          value={frameInterval}
          onChange={handleFrameIntervalChange}
          disabled={disabled}
          aria-label="Frame processing interval"
        >
          <option value={1}>Every Frame</option>
          <option value={2}>Every 2 Frames</option>
          <option value={3}>Every 3 Frames</option>
          <option value={5}>Every 5 Frames</option>
          <option value={10}>Every 10 Frames</option>
        </select>

        <p className="setting-description">
          Select how frequently frames are sent for detection.
          Processing fewer frames can improve speed but may reduce
          tracking continuity and counting accuracy.
        </p>

      </div>

      {/* Information Box */}
      <div className="settings-information">
        <Info size={17} />

        <p>
          Recommended configuration: 35% confidence and every-frame
          processing. Adjust these settings depending on video quality
          and processing requirements.
        </p>
      </div>

    </section>
  );
}

export default DetectionSettings;
