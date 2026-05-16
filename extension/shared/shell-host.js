/**
 * Fixed-size extension popup shell. Inner routes load in #fakeShaAppFrame so the
 * browser does not resize the popup when switching popup / settings / history.
 */
(function () {
  "use strict";

  const NAV_KEY = "fakeShaActionPopup";
  const DEFAULT_ROUTE = "popup/popup.html";
  const SHELL_POPUP = "shell.html";

  const frame = document.getElementById("fakeShaAppFrame");

  function normalizeRoute(path) {
    return String(path || DEFAULT_ROUTE)
      .replace(/^\.\.\//, "")
      .replace(/^\//, "");
  }

  function persistRoute(route) {
    try {
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ [NAV_KEY]: route });
      }
    } catch (e) {
      // ignore
    }
  }

  function registerShellAsActionPopup() {
    try {
      if (typeof chrome !== "undefined" && chrome.action && chrome.action.setPopup) {
        chrome.action.setPopup({ popup: SHELL_POPUP });
      }
    } catch (e) {
      // ignore
    }
  }

  function loadRoute(path) {
    if (!frame || typeof chrome === "undefined" || !chrome.runtime?.getURL) return;
    const route = normalizeRoute(path);
    persistRoute(route);
    frame.src = chrome.runtime.getURL(route);
  }

  registerShellAsActionPopup();

  try {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(NAV_KEY, (result) => {
        loadRoute(result && result[NAV_KEY]);
      });
    } else {
      loadRoute(DEFAULT_ROUTE);
    }
  } catch (e) {
    loadRoute(DEFAULT_ROUTE);
  }

  window.addEventListener("message", (event) => {
    if (!frame || event.source !== frame.contentWindow) return;
    const data = event.data;
    if (data && data.type === "fakeSha_navigate" && data.path) {
      loadRoute(data.path);
    }
  });
})();
