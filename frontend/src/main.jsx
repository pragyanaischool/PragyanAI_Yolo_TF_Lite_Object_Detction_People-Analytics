
import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import "./style.css";

// Get the root element from index.html
const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Root element with id 'root' was not found.");
}

// Initialize and render the React application
ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
