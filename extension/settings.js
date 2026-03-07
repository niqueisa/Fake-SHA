// Settings page logic for FAKE-SHA

document.addEventListener("DOMContentLoaded", () => {
  const backendUrlInput = document.getElementById("backendUrl");
  const modeSelectionOnly = document.getElementById("modeSelectionOnly");
  const modeSelectionFallback = document.getElementById("modeSelectionFallback");
  const toggleHighlightTokens = document.getElementById("toggleHighlightTokens");
  const toggleHistoryEnabled = document.getElementById("toggleHistoryEnabled");

  const btnSaveSettings = document.getElementById("btnSaveSettings");
  const saveStatus = document.getElementById("saveStatus");
  const btnBackToPopup = document.getElementById("btnBackToPopup");
  const btnClearHistory = document.getElementById("btnClearHistory");
  const clearHistoryStatus = document.getElementById("clearHistoryStatus");
  const btnTestConnection = document.getElementById("btnTestConnection");
  const testConnectionStatus = document.getElementById("testConnectionStatus");

  const DEFAULT_SETTINGS = {
    backendUrl: "http://localhost:8000",
    analysisMode: "selection_only",
    highlightTokens: true,
    historyEnabled: true,
  };

  const SETTINGS_KEY = "fakeShaSettings";
  const HISTORY_KEY = "fakeShaHistory";

  function getStorage() {
    try {
      // Prefer browser.* if available, then chrome.*; fall back to localStorage for non-extension contexts.
      if (typeof browser !== "undefined" && browser.storage && browser.storage.local) {
        return browser.storage.local;
      }
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        return chrome.storage.local;
      }
    } catch (e) {
      // ignore
    }
    return null;
  }

  const storage = getStorage();

  function loadSettings() {
    if (!storage) {
      applySettingsToUI(DEFAULT_SETTINGS);
      return;
    }

    storage.get(SETTINGS_KEY, (result) => {
      const stored = result && result[SETTINGS_KEY];
      const settings = Object.assign({}, DEFAULT_SETTINGS, stored || {});
      applySettingsToUI(settings);
    });
  }

  function applySettingsToUI(settings) {
    backendUrlInput.value = settings.backendUrl || DEFAULT_SETTINGS.backendUrl;

    if (settings.analysisMode === "selection_fallback") {
      modeSelectionFallback.checked = true;
    } else {
      modeSelectionOnly.checked = true;
    }

    if (toggleHighlightTokens) {
      toggleHighlightTokens.checked =
        typeof settings.highlightTokens === "boolean"
          ? settings.highlightTokens
          : DEFAULT_SETTINGS.highlightTokens;
    }

    toggleHistoryEnabled.checked =
      typeof settings.historyEnabled === "boolean"
        ? settings.historyEnabled
        : DEFAULT_SETTINGS.historyEnabled;
  }

  function collectSettingsFromUI() {
    const analysisMode = modeSelectionFallback.checked
      ? "selection_fallback"
      : "selection_only";

    return {
      backendUrl: backendUrlInput.value.trim() || DEFAULT_SETTINGS.backendUrl,
      analysisMode,
      highlightTokens: !!(toggleHighlightTokens && toggleHighlightTokens.checked),
      historyEnabled: !!toggleHistoryEnabled.checked,
    };
  }

  function showSaveStatus() {
    if (!saveStatus) return;
    saveStatus.textContent = "Settings saved.";
    saveStatus.style.opacity = "1";
    setTimeout(() => {
      saveStatus.style.opacity = "0";
    }, 1500);
  }

  function showClearHistoryStatus(message) {
    if (!clearHistoryStatus) return;
    clearHistoryStatus.textContent = message;
  }

  if (btnSaveSettings) {
    btnSaveSettings.addEventListener("click", () => {
      const settings = collectSettingsFromUI();

      if (!storage) {
        try {
          localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        } catch (e) {
          // ignore errors for now
        }
        showSaveStatus();
        return;
      }

      storage.set({ [SETTINGS_KEY]: settings }, () => {
        showSaveStatus();
      });
    });
  }

  if (btnBackToPopup) {
    btnBackToPopup.addEventListener("click", () => {
      window.location.href = "popup.html";
    });
  }

  if (btnClearHistory) {
    btnClearHistory.addEventListener("click", () => {
      if (!storage) {
        try {
          localStorage.removeItem(HISTORY_KEY);
          showClearHistoryStatus("History cleared (local storage).");
        } catch (e) {
          showClearHistoryStatus("Unable to clear history in this context.");
        }
        return;
      }

      storage.remove(HISTORY_KEY, () => {
        showClearHistoryStatus("History cleared from extension storage.");
      });
    });
  }

  if (btnTestConnection) {
    btnTestConnection.addEventListener("click", () => {
      if (testConnectionStatus) {
        testConnectionStatus.textContent =
          "Test connection is a UI-only placeholder. No network request is made yet.";
      }
    });
  }

  // Initial load
  try {
    // Attempt to load from extension storage; if unavailable, fall back to localStorage snapshot.
    if (storage) {
      loadSettings();
    } else {
      const raw = localStorage.getItem(SETTINGS_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      const settings = Object.assign({}, DEFAULT_SETTINGS, parsed || {});
      applySettingsToUI(settings);
    }
  } catch (e) {
    applySettingsToUI(DEFAULT_SETTINGS);
  }
});

