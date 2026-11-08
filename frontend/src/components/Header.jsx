
import React from "react";
import {
  Activity,
  BrainCircuit,
  ShieldCheck
} from "lucide-react";

function Header() {
  return (
    <header className="app-header">
      <div className="header-inner">

        {/* PragyanAI Branding */}
        <div className="brand-section">

          <div className="brand-logo-container">
            <img
              src="/favicon.svg"
              alt="PragyanAI Logo"
              className="brand-logo"
            />
          </div>

          <div className="brand-information">
            <div className="brand-name">
              PragyanAI
            </div>

            <div className="brand-tagline">
              Artificial Intelligence & Intelligent Automation
            </div>
          </div>

        </div>

        {/* Application Information */}
        <div className="header-right">

          <div className="header-feature">
            <BrainCircuit size={16} />
            <span>YOLO11 AI</span>
          </div>

          <div className="header-feature">
            <ShieldCheck size={16} />
            <span>Smart Detection</span>
          </div>

          <div className="header-status">
            <span className="status-dot"></span>
            <Activity size={15} />
            <span>Application Ready</span>
          </div>

        </div>

      </div>
    </header>
  );
}

export default Header;
