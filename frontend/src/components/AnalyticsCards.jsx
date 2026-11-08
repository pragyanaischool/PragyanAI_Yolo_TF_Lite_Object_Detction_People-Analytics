
import React from "react";

import {
  Users,
  UserCheck,
  TrendingUp,
  Layers
} from "lucide-react";

function getMetric(data, keys, fallback = 0) {
  for (const key of keys) {
    if (
      data &&
      data[key] !== undefined &&
      data[key] !== null
    ) {
      return data[key];
    }
  }

  return fallback;
}

function AnalyticsCards({ analytics = {} }) {
  const data = analytics || {};

  const metrics = [
    {
      title: "Unique People",
      value: getMetric(data, [
        "unique_people",
        "total_people",
        "unique_count",
        "total_unique_people"
      ]),
      icon: <Users size={21} />,
      description: "Distinct tracked people"
    },

    {
      title: "Maximum People",
      value: getMetric(data, [
        "max_people",
        "maximum_people",
        "max_count",
        "peak_occupancy"
      ]),
      icon: <UserCheck size={21} />,
      description: "Highest simultaneous count"
    },

    {
      title: "Average People",
      value: getMetric(data, [
        "average_people",
        "avg_people",
        "average_count",
        "avg_count"
      ]),
      icon: <TrendingUp size={21} />,
      description: "Average count per frame"
    },

    {
      title: "Processed Frames",
      value: getMetric(data, [
        "processed_frames",
        "total_frames",
        "frames_processed",
        "frame_count"
      ]),
      icon: <Layers size={21} />,
      description: "Frames analyzed"
    }
  ];

  const formatValue = (value) => {
    if (typeof value === "number") {
      return Number.isInteger(value)
        ? value.toLocaleString()
        : value.toFixed(2);
    }

    return String(value);
  };

  return (
    <section className="analytics-grid">

      {metrics.map((metric) => (
        <div
          className="analytics-card"
          key={metric.title}
        >

          <div className="analytics-card-header">

            <div className="analytics-icon">
              {metric.icon}
            </div>

          </div>

          <div className="analytics-value">
            {formatValue(metric.value)}
          </div>

          <div className="analytics-title">
            {metric.title}
          </div>

          <p className="analytics-description">
            {metric.description}
          </p>

        </div>
      ))}

    </section>
  );
}

export default AnalyticsCards;
