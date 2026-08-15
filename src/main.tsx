import React from "react";
import ReactDOM from "react-dom/client";
import { I18nProvider } from "@lingui/react";
import App from "./App";
import { initializeTheme, ThemeProvider } from "@/components/theme-provider";
import { activateLocale, getInitialLocale, i18n } from "./i18n";
import "./index.css";

initializeTheme();

async function bootstrap(): Promise<void> {
  await activateLocale(getInitialLocale());

  ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
      <I18nProvider i18n={i18n}>
        <ThemeProvider>
          <App />
        </ThemeProvider>
      </I18nProvider>
    </React.StrictMode>,
  );
}

void bootstrap();
