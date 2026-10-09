import React from "react";
import ReactDOM from "react-dom/client";
import { MotionConfig } from "framer-motion";
import App from "./App";
import Workspace from "./Workspace";
import "./styles.css";
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MotionConfig reducedMotion="user">
      {window.location.pathname === "/workspace" ? <Workspace /> : <App />}
    </MotionConfig>
  </React.StrictMode>,
);
