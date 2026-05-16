/**
 * Background service worker: runs analysis after the popup closes and persists tab sessions.
 */
importScripts("shared/session.js", "shared/api.js", "shared/nav.js");

const SETTINGS_KEY = "fakeShaSettings";
const DEFAULT_SETTINGS = {
  backendUrl: "http://localhost:8000",
  analysisMode: "selection_only",
  highlightTokens: true,
  historyEnabled: true,
};

/** @type {Map<number, { generation: number, controller: AbortController }>} */
const analyzeJobsByTab = new Map();

function getSettings() {
  return new Promise((resolve) => {
    try {
      chrome.storage.local.get(SETTINGS_KEY, (result) => {
        const stored = result && result[SETTINGS_KEY];
        resolve({ ...DEFAULT_SETTINGS, ...(stored || {}) });
      });
    } catch (e) {
      resolve({ ...DEFAULT_SETTINGS });
    }
  });
}

function filterTokensWithinScope(tokens, scopeText) {
  const scope = String(scopeText || "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[\u2018\u2019`´]/g, "'")
    .replace(/[^\w\s']/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!scope) return [];

  return (Array.isArray(tokens) ? tokens : [])
    .map((t) => (typeof t === "string" ? t : t && t.text ? t.text : ""))
    .map((t) => String(t || "").trim())
    .filter((token) => {
      if (!token || token.length < 2) return false;
      const needle = token
        .replace(/\s+/g, " ")
        .trim()
        .toLowerCase();
      if (scope.includes(needle)) return true;
      const needleFlat = needle.replace(/'/g, "");
      const scopeFlat = scope.replace(/'/g, "");
      if (needleFlat.length >= 2 && scopeFlat.includes(needleFlat)) return true;
      const words = needle.split(" ").filter((w) => w.length >= 2);
      if (words.length > 1) return words.every((w) => scope.includes(w));
      return false;
    });
}

function sendHighlights(tabId, tokens, scopeText, mode) {
  if (tabId == null) return;
  const tokenTexts = filterTokensWithinScope(tokens, scopeText);
  if (tokenTexts.length === 0) return;
  chrome.tabs.sendMessage(
    tabId,
    { type: "fakeSha_highlightTokens", tokens: tokenTexts, scopeText, mode },
    () => {
      void chrome.runtime.lastError;
    }
  );
}

function isJobCurrent(tabId, generation) {
  const job = analyzeJobsByTab.get(tabId);
  return Boolean(job && job.generation === generation);
}

async function runAnalyze(tabId, message) {
  const prev = analyzeJobsByTab.get(tabId);
  if (prev && prev.controller) {
    prev.controller.abort();
  }

  const generation = (prev?.generation || 0) + 1;
  const controller = new AbortController();
  analyzeJobsByTab.set(tabId, { generation, controller });

  await FakeShaSession.setForTab(tabId, {
    view: "loading",
    generation,
    meta: message.meta || {},
  });

  try {
    const settings = await getSettings();
    const backendUrl = String(settings.backendUrl || DEFAULT_SETTINGS.backendUrl).trim();
    if (!backendUrl) {
      if (!isJobCurrent(tabId, generation)) return;
      await FakeShaSession.setForTab(tabId, {
        view: "error",
        errorMessage: "Backend URL is not configured. Open Settings to set it.",
        meta: message.meta || {},
      });
      return;
    }

    const baseUrl = self.FakeShaApi
      ? self.FakeShaApi.normalizeBackendBaseUrl(backendUrl)
      : backendUrl.replace(/\/+$/, "");

    const backendResult = await self.FakeShaApi.postAnalyze(baseUrl, message.payload, {
      signal: controller.signal,
    });

    if (!isJobCurrent(tabId, generation)) return;

    const isFake = String(backendResult.verdict || "").toUpperCase() === "FAKE";
    await FakeShaSession.setForTab(tabId, {
      view: "result",
      generation,
      backendResult,
      meta: message.meta || {},
      historySaved: false,
    });

    if (message.meta && message.meta.highlightTokens) {
      const shapTopTokens = Array.isArray(backendResult?.explanation?.top_tokens)
        ? backendResult.explanation.top_tokens
        : [];
      const rawTokens = shapTopTokens.length > 0
        ? shapTopTokens
        : (Array.isArray(backendResult.tokens) ? backendResult.tokens : []);
      const highlightMode = isFake ? "fake" : "real";
      sendHighlights(tabId, rawTokens, message.meta.textToAnalyze || "", highlightMode);
    }
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (!isJobCurrent(tabId, generation)) return;
    const errorMessage =
      err instanceof Error ? err.message : "An unexpected error occurred.";
    await FakeShaSession.setForTab(tabId, {
      view: "error",
      errorMessage,
      meta: message.meta || {},
    });
  } finally {
    const job = analyzeJobsByTab.get(tabId);
    if (job && job.generation === generation) {
      analyzeJobsByTab.delete(tabId);
    }
  }
}

function cancelAnalyze(tabId) {
  const job = analyzeJobsByTab.get(tabId);
  if (job && job.controller) {
    job.controller.abort();
    analyzeJobsByTab.delete(tabId);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  FakeShaSession.markInterruptedLoadingSessions();
  if (self.FakeShaNav) {
    self.FakeShaNav.restoreActionPopupFromStorage();
  }
});

chrome.runtime.onStartup.addListener(() => {
  FakeShaSession.markInterruptedLoadingSessions();
  if (self.FakeShaNav) {
    self.FakeShaNav.restoreActionPopupFromStorage();
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || typeof message.type !== "string") return;

  if (message.type === "fakeSha_startAnalyze") {
    const tabId = message.tabId;
    if (tabId == null) {
      sendResponse({ ok: false, error: "Missing tab id." });
      return;
    }
    runAnalyze(tabId, message)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => {
        sendResponse({
          ok: false,
          error: err instanceof Error ? err.message : "Analyze failed.",
        });
      });
    return true;
  }

  if (message.type === "fakeSha_cancelAnalyze") {
    const tabId = message.tabId;
    if (tabId != null) {
      cancelAnalyze(tabId);
      FakeShaSession.clearForTab(tabId).then(() => sendResponse({ ok: true }));
    } else {
      sendResponse({ ok: false });
    }
    return true;
  }

  if (message.type === "fakeSha_setActionPopup") {
    if (self.FakeShaNav && message.popup) {
      self.FakeShaNav.setActionPopup(message.popup);
    }
    sendResponse({ ok: true });
    return;
  }
});
