/**
 * In-extension navigation without resizing the browser action popup.
 *
 * Routes load inside shell.html's iframe; only the last route is stored for reopen.
 */
(function (global) {
  "use strict";

  const NAV_KEY = "fakeShaActionPopup";
  const DEFAULT_POPUP = "popup/popup.html";
  const SHELL_POPUP = "shell.html";

  function normalizePopupPath(path) {
    return String(path || DEFAULT_POPUP)
      .replace(/^\.\.\//, "")
      .replace(/^\//, "");
  }

  function persistRoute(path) {
    const route = normalizePopupPath(path);
    try {
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ [NAV_KEY]: route });
      }
    } catch (e) {
      // ignore
    }
    return route;
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

  function navigateTo(relativePath) {
    const route = persistRoute(relativePath);

    if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
      const url = chrome.runtime.getURL(route);

      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: "fakeSha_navigate", path: route }, "*");
        return;
      }

      const frame = document.getElementById("fakeShaAppFrame");
      if (frame) {
        frame.src = url;
        return;
      }
    }

    const href = relativePath.startsWith("..") ? relativePath : `../${relativePath}`;
    window.location.href = href;
  }

  function setActionPopup(path) {
    persistRoute(path);
    registerShellAsActionPopup();
  }

  function restoreActionPopupFromStorage() {
    registerShellAsActionPopup();
  }

  function savePageUiState(key, state) {
    try {
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ [key]: state });
      }
    } catch (e) {
      // ignore
    }
  }

  function loadPageUiState(key, callback) {
    try {
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.get(key, (result) => {
          callback(result && result[key] ? result[key] : null);
        });
        return;
      }
    } catch (e) {
      // ignore
    }
    callback(null);
  }

  global.FakeShaNav = {
    NAV_KEY,
    DEFAULT_POPUP,
    SHELL_POPUP,
    setActionPopup,
    navigateTo,
    restoreActionPopupFromStorage,
    savePageUiState,
    loadPageUiState,
  };
})(typeof self !== "undefined" ? self : window);
