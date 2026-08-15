import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { initializeTheme, ThemeProvider } from "@/components/theme-provider";
import "./index.css";

initializeTheme();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>,
);
