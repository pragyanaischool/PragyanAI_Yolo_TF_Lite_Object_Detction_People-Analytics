
import React from "react";

import {
  Users,
  UserCheck,
  TrendingUp,
  Video,
  Activity,
  Clock,
  Target,
  BarChart3,
} from "lucide-react";


/* ============================================================
   HELPER FUNCTIONS
   ============================================================ */

// Get the first available metric from possible API field names.
function getMetric(data, keys, fallback = 0) {
  for (const key of keys) {
    const value = data?.[key];

    if (
      value !== undefined &&
      value !== null &&
      value !== ""
    ) {
      return value;
    }
  }

  return fallback;
}


// Format numbers for dashboard display.
function formatNumber(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "0";
  }

  return number.toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  });
}


// Format duration in seconds.
function formatDuration(value) {
  const seconds = Number(value);

  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0 sec";
  }

  if (seconds < 60) {
    return `${seconds.toFixed(1)} sec`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  return `${minutes} min ${remainingSeconds} sec`;
}


/* ============================================================
   ANALYTICS CARD COMPONENT
   ============================================================ */

function AnalyticsCard({
  title,
  value,
  description,
  icon: Icon,
  color = "blue",
}) {
  return (
    <article className={`analytics-card analytics-${color}`}>

      <div className="analytics-card-header">

        <div className="analytics-icon">
          <Icon size={22} />
        </div>

      </div>

      <div className="analytics-value">
        {value}
      </div>

      <div className="analytics-title">
        {title}
      </div>

      <div className="analytics-description">
        {description}
      </div>

    </article>
  );
}


/* ============================================================
   MAIN ANALYTICS DASHBOARD
   ============================================================ */

export default function AnalyticsCards({ analytics = {} }) {

  // ----------------------------------------------------------
  // Extract analytics from backend response
  // ----------------------------------------------------------

  const uniquePeople = getMetric(analytics, [
    "unique_people",
    "unique_count",
    "total_unique_people",
    "unique_persons",
    "unique_ids",
  ]);

  const totalDetections = getMetric(analytics, [
    "total_detections",
    "total_people",
    "total_count",
    "detection_count",
    "total_person_detections",
  ]);

  const peakPeople = getMetric(analytics, [
    "max_people",
    "maximum_people",
    "peak_people",
    "max_count",
    "maximum_count",
  ]);

  const processedFrames = getMetric(analytics, [
    "processed_frames",
    "total_frames",
    "frames_processed",
    "frame_count",
  ]);

  const averagePeople = getMetric(analytics, [
    "average_people",
    "avg_people",
    "average_count",
    "mean_people_per_frame",
  ]);

  const processingTime = getMetric(analytics, [
    "processing_time",
    "processing_time_seconds",
    "total_processing_time",
  ]);

  const averageConfidence = getMetric(analytics, [
    "average_confidence",
    "avg_confidence",
    "mean_confidence",
  ]);

  const fps = getMetric(analytics, [
    "processing_fps",
    "inference_fps",
    "fps",
  ]);


  // ----------------------------------------------------------
  // Primary analytics cards
  // ----------------------------------------------------------

  const primaryMetrics = [
    {
      title: "Unique People",
      value: formatNumber(uniquePeople),
      description: "Distinct individuals tracked",
      icon: UserCheck,
      color: "blue",
    },
    {
      title: "Total Detections",
      value: formatNumber(totalDetections),
      description: "Total person detections",
      icon: Users,
      color: "green",
    },
    {
      title: "Peak People",
      value: formatNumber(peakPeople),
      description: "Maximum people in one frame",
      icon: TrendingUp,
      color: "purple",
    },
    {
      title: "Processed Frames",
      value: formatNumber(processedFrames),
      description: "Frames analysed by AI",
      icon: Video,
      color: "orange",
    },
  ];


  // ----------------------------------------------------------
  // Additional analytics
  // ----------------------------------------------------------

  const additionalMetrics = [
    {
      title: "Average People",
      value: formatNumber(averagePeople),
      icon: BarChart3,
    },
    {
      title: "Processing Time",
      value: formatDuration(processingTime),
      icon: Clock,
    },
    {
      title: "Average Confidence",
      value: `${(Number(averageConfidence) * 100).toFixed(1)}%`,
      icon: Target,
    },
    {
      title: "Processing FPS",
      value: formatNumber(fps),
      icon: Activity,
    },
  ];


  // ----------------------------------------------------------
  // Dashboard UI
  // ----------------------------------------------------------

  return (
    <section className="analytics-dashboard">

      {/* Dashboard Header */}

      <div className="analytics-dashboard-header">

        <div>
          <div className="section-eyebrow">
            AI VIDEO ANALYTICS
          </div>

          <h2>People Detection Analytics</h2>

          <p>
            Summary of people detection and tracking results.
          </p>
        </div>

        <div className="analytics-live-status">
          <span className="status-dot"></span>
          Analysis Complete
        </div>

      </div>


      {/* Primary Metrics */}

      <div className="analytics-grid">

        {primaryMetrics.map((metric) => (

          <AnalyticsCard
            key={metric.title}
            title={metric.title}
            value={metric.value}
            description={metric.description}
            icon={metric.icon}
            color={metric.color}
          />

        ))}

      </div>


      {/* Additional Metrics */}

      <div className="additional-analytics">

        <h3>Additional Performance Metrics</h3>

        <div className="additional-analytics-grid">

          {additionalMetrics.map((metric) => {

            const Icon = metric.icon;

            return (
              <div
                className="additional-analytics-card"
                key={metric.title}
              >

                <div className="additional-analytics-icon">
                  <Icon size={19} />
                </div>

                <div className="additional-analytics-content">

                  <span className="additional-analytics-title">
                    {metric.title}
                  </span>

                  <strong className="additional-analytics-value">
                    {metric.value}
                  </strong>

                </div>

              </div>
            );

          })}

        </div>

      </div>

    </section>
  );
}
