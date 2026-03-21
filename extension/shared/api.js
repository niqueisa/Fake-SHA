/**
 * Shared backend API helpers for FAKE-SHA extension surfaces (popup, settings, etc.).
 */
(function (global) {
  "use strict";

  function normalizeBackendBaseUrl(raw) {
    return String(raw || "").trim().replace(/\/+$/, "");
  }

  /**
   * POST /analyze with JSON body. Returns parsed JSON or throws Error.
   */
  async function postAnalyze(baseUrl, payload) {
    const base = normalizeBackendBaseUrl(baseUrl);
    const response = await fetch(`${base}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(
        `Server error (${response.status}): ${errText.slice(0, 100) || "Unknown error"}`
      );
    }
    const jsonText = await response.text();
    try {
      return JSON.parse(jsonText);
    } catch (e) {
      throw new Error("Invalid JSON response from backend.");
    }
  }

  /**
   * GET /health. Validates JSON body. Optional AbortSignal for timeouts.
   */
  async function getHealth(baseUrl, signal) {
    const base = normalizeBackendBaseUrl(baseUrl);
    const healthUrl = new URL("/health", base).href;
    const response = await fetch(healthUrl, { method: "GET", signal });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const text = await response.text();
    JSON.parse(text);
    return text;
  }

  global.FakeShaApi = {
    normalizeBackendBaseUrl,
    postAnalyze,
    getHealth,
  };
})(typeof self !== "undefined" ? self : this);
