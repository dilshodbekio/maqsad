import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { applyHtmlLang } from "./i18n";
import "./styles.css";

applyHtmlLang();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
