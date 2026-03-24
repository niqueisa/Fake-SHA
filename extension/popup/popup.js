document.addEventListener("DOMContentLoaded", () => {
  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const resultState = document.getElementById("resultState");

  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnClear = document.getElementById("btnClear");
  const btnCancelLoading = document.getElementById("btnCancelLoading");
  const btnOpenHistory = document.getElementById("btnOpenHistory");
  const btnOpenSettings = document.getElementById("btnOpenSettings");
  const selectedTextValue = document.getElementById("selectedTextValue");
  const selectionHeader = document.getElementById("selectionHeader");

  // -----------------------------
  // Settings (loaded from extension storage on popup open)
  // -----------------------------
  const DEFAULT_POPUP_SETTINGS = {
    backendUrl: "http://localhost:8000",
    analysisMode: "selection_only",
    highlightTokens: true,
    historyEnabled: true,
  };

  let popupSettings = { ...DEFAULT_POPUP_SETTINGS };

  const HISTORY_KEY = "fakeShaHistory";

  function getStorage() {
    try {
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

  function getExtensionStorage(callback) {
    try {
      const api = typeof chrome !== "undefined" && chrome.storage && chrome.storage.local
        ? chrome.storage
        : typeof browser !== "undefined" && browser.storage && browser.storage.local
          ? browser.storage
          : null;
      if (api && api.local) {
        api.local.get("fakeShaSettings", (result) => {
          if (chrome?.runtime?.lastError) {
            callback({ ...DEFAULT_POPUP_SETTINGS });
            return;
          }
          const stored = result && result.fakeShaSettings;
          if (stored && typeof stored === "object") {
            popupSettings = {
              backendUrl: typeof stored.backendUrl === "string" && stored.backendUrl.trim()
                ? stored.backendUrl.trim()
                : DEFAULT_POPUP_SETTINGS.backendUrl,
              analysisMode: stored.analysisMode === "selection_fallback" || stored.analysisMode === "selection_only"
                ? stored.analysisMode
                : DEFAULT_POPUP_SETTINGS.analysisMode,
              highlightTokens: typeof stored.highlightTokens === "boolean"
                ? stored.highlightTokens
                : DEFAULT_POPUP_SETTINGS.highlightTokens,
              historyEnabled: typeof stored.historyEnabled === "boolean"
                ? stored.historyEnabled
                : DEFAULT_POPUP_SETTINGS.historyEnabled,
            };
          }
          callback(popupSettings);
        });
        return;
      }
    } catch (e) {
      // ignore
    }
    callback({ ...DEFAULT_POPUP_SETTINGS });
  }

  function loadPopupSettingsThenRefresh() {
    getExtensionStorage(() => {
      refreshSelectionFromPage();
    });
  }
  // Default state
  showEmpty();
  loadPopupSettingsThenRefresh();

  btnAnalyze.addEventListener("click", () => {
    doAnalyze();
  });

  /**
   * Main Analyze flow: get selection, call backend, render result.
   * Uses async/await for clean error handling.
   */
  async function doAnalyze() {
    showLoading();
    hideError();

    try {
      // 1. Load settings (backendUrl, mode, etc.)
      const settings = await new Promise((resolve) => {
        getExtensionStorage(resolve);
      });

      const backendUrl = (settings.backendUrl || DEFAULT_POPUP_SETTINGS.backendUrl).trim();
      if (!backendUrl) {
        showErrorAndReset("Backend URL is not configured. Open Settings to set it.");
        return;
      }

      const baseUrl = (window.FakeShaApi && window.FakeShaApi.normalizeBackendBaseUrl
        ? window.FakeShaApi.normalizeBackendBaseUrl(backendUrl)
        : backendUrl.replace(/\/+$/, ""));

      // 2. Get text to analyze: selected text first, or page content in fallback mode
      let textToAnalyze = await getSelectedTextFromPage();
      textToAnalyze = (textToAnalyze || "").trim();

      if (!textToAnalyze) {
        if (settings.analysisMode === "selection_fallback") {
          // Fallback mode: try to extract page content
          const pageContent = await getPageContentFromPage();
          textToAnalyze = (pageContent.text || "").trim();
          if (!textToAnalyze) {
            showErrorAndReset(
              "Could not extract page content. Try selecting text manually, or open an article page."
            );
            return;
          }
          // Use page title from extraction if available (tab title is fetched below)
        } else {
          showErrorAndReset("Please select text on the page first.");
          return;
        }
      }

      // 3. Get active tab URL and title (fallback to empty if unavailable)
      const tabInfo = await getActiveTabInfo();
      const pageUrl = tabInfo.url || "";
      const pageTitle = tabInfo.title || "Untitled";

      // 4. Build request payload
      const payload = {
        text: textToAnalyze,
        url: pageUrl,
        title: pageTitle,
        mode: settings.analysisMode || "selection_only",
      };

      // 5. Send POST request to backend (shared helper)
      if (!window.FakeShaApi || typeof window.FakeShaApi.postAnalyze !== "function") {
        throw new Error("Extension API module missing (shared/api.js).");
      }
      const backendResult = await window.FakeShaApi.postAnalyze(baseUrl, payload);

      // 6. Map backend response to popup's render format
      const mappedData = mapBackendResponseToPopupFormat(backendResult, {
        articleTitle: pageTitle,
        sourceUrl: pageUrl,
      });

      // 7. Render result and save to history
      renderResult(mappedData);
      saveHistoryIfEnabled(mappedData);
      showResult();

      // 8. If highlightTokens is enabled, highlight contributing phrases on the page
      if (settings.highlightTokens && tabInfo.tabId != null) {
        const tokens = Array.isArray(backendResult.tokens) ? backendResult.tokens : [];
        sendHighlightTokensToPage(tabInfo.tabId, tokens);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      showErrorAndReset(getUserFriendlyError(message));
    }
  }

  /**
   * Get page content (article/main/body text) from the active tab via content script.
   * Used when selection_fallback mode is on and no text is selected.
   */
  function getPageContentFromPage() {
    return new Promise((resolve) => {
      const fallback = { text: "", pageTitle: "", extractionSource: "body" };
      if (typeof chrome === "undefined" || !chrome.tabs?.query) {
        resolve(fallback);
        return;
      }
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (chrome.runtime.lastError || !tabs?.length || tabs[0].id == null) {
          resolve(fallback);
          return;
        }
        chrome.tabs.sendMessage(
          tabs[0].id,
          { type: "fakeSha_getPageContent" },
          (response) => {
            if (chrome.runtime.lastError || !response) {
              resolve(fallback);
              return;
            }
            resolve({
              text: response.text || "",
              pageTitle: response.pageTitle || "",
              extractionSource: response.extractionSource || "body",
            });
          }
        );
      });
    });
  }

  /**
   * Get selected text from the active tab via content script messaging.
   */
  function getSelectedTextFromPage() {
    return new Promise((resolve) => {
      if (typeof chrome === "undefined" || !chrome.tabs?.query) {
        resolve((selectedTextValue && selectedTextValue.textContent) || "");
        return;
      }
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (chrome.runtime.lastError || !tabs?.length || tabs[0].id == null) {
          resolve((selectedTextValue && selectedTextValue.textContent) || "");
          return;
        }
        chrome.tabs.sendMessage(
          tabs[0].id,
          { type: "fakeSha_getSelection" },
          (response) => {
            if (chrome.runtime.lastError || !response) {
              resolve((selectedTextValue && selectedTextValue.textContent) || "");
              return;
            }
            resolve((response.text || "").trim());
          }
        );
      });
    });
  }

  /**
   * Get active tab URL, title, and id using Chrome extension API.
   */
  function getActiveTabInfo() {
    return new Promise((resolve) => {
      const fallback = { url: "", title: "", tabId: null };
      if (typeof chrome === "undefined" || !chrome.tabs?.query) {
        resolve(fallback);
        return;
      }
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (chrome.runtime.lastError || !tabs?.length || !tabs[0]) {
          resolve(fallback);
          return;
        }
        resolve({
          url: tabs[0].url || "",
          title: tabs[0].title || "",
          tabId: tabs[0].id,
        });
      });
    });
  }

  /**
   * Send tokens to content script for page highlighting (when highlightTokens setting is on).
   */
  function sendHighlightTokensToPage(tabId, tokens) {
    if (tabId == null || typeof chrome === "undefined" || !chrome.tabs?.sendMessage) return;
    const tokenTexts = Array.isArray(tokens)
      ? tokens.map((t) => (typeof t === "string" ? t : t && t.text ? t.text : "")).filter(Boolean)
      : [];
    if (tokenTexts.length === 0) return;
    chrome.tabs.sendMessage(tabId, { type: "fakeSha_highlightTokens", tokens: tokenTexts }, () => {
      if (chrome.runtime.lastError) {
        // Ignore: content script may not be loaded (e.g. chrome:// page)
      }
    });
  }

  /**
   * Clear FAKE-SHA highlights on the active tab.
   */
  function clearHighlightsOnPage() {
    if (typeof chrome === "undefined" || !chrome.tabs?.query) return;
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (chrome.runtime.lastError || !tabs?.length || tabs[0].id == null) return;
      chrome.tabs.sendMessage(tabs[0].id, { type: "fakeSha_clearHighlights" }, () => {
        if (chrome.runtime.lastError) {
          // Ignore
        }
      });
    });
  }

  /**
   * Map backend /analyze response to popup's render format.
   * Backend returns: verdict, confidence (0-1), summary, indicators (string[]), tokens ({text, impact, label}[])
   * Popup expects: label, confidence (0-100), indicators ({name, shap, contributionPct}[]), topTokens ({text, shap, impact}[]), etc.
   */
  function mapBackendResponseToPopupFormat(backend, meta = {}) {
    const isFake = String(backend.verdict || "").toUpperCase() === "FAKE";
    const confidencePct =
      typeof backend.confidence === "number"
        ? backend.confidence * 100
        : parseFloat(String(backend.confidence || "0")) * 100 || 0;

    const label = isFake ? "FAKE NEWS DETECTED" : "REAL NEWS DETECTED";
    const topTokensTitle = isFake
      ? "Top Tokens Contributing to Misinformation"
      : "Top Tokens Contributing to Authenticity";
    const topTokensLegend = isFake
      ? "High Misinformation Impact"
      : "High Authenticity Impact";

    const indicatorsRaw = Array.isArray(backend.indicators)
      ? backend.indicators
      : [];
    const indicators = indicatorsRaw.map((name, i) => {
      const contributionPct = Math.max(10, 80 - i * 15);
      const shap = isFake ? -(20 + i * 5) : 20 + i * 5;
      return {
        name: typeof name === "string" ? name : "Indicator",
        shap,
        contributionPct,
      };
    });

    const tokensRaw = Array.isArray(backend.tokens) ? backend.tokens : [];
    const topTokens = tokensRaw.map((t) => {
      const impact = t.impact || "medium";
      const isFakeToken =
        String(t.label || "").toLowerCase().includes("fake") || isFake;
      const shapMap = { high: 6, medium: 4, low: 2 };
      const val = shapMap[impact] || shapMap.medium;
      const shap = isFakeToken ? -val : val;
      return {
        text: t.text || "",
        impact,
        shap,
      };
    });

    return {
      articleTitle: meta.articleTitle || "Untitled",
      sourceUrl: meta.sourceUrl || "",
      label,
      confidence: Math.min(100, Math.max(0, confidencePct)),
      indicators,
      topTokensTitle,
      topTokensLegend,
      topTokens,
      summary: backend.summary || "No summary available.",
    };
  }

  /**
   * Convert technical errors to user-friendly messages.
   */
  function getUserFriendlyError(message) {
    if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
      return "Backend unavailable. Check that the server is running and the URL in Settings is correct.";
    }
    if (message.includes("invalid") || message.includes("Invalid")) {
      return "Invalid backend response. Please try again.";
    }
    if (message.includes("CORS") || message.includes("blocked")) {
      return "Request was blocked. Ensure the backend allows your extension origin.";
    }
    return message;
  }

  /**
   * Show error message and reset to empty state.
   */
  function showErrorAndReset(message) {
    showEmpty();
    showError(message);
    refreshSelectionFromPage();
  }

  /**
   * Display an error message in the popup (in empty state).
   */
  function showError(message) {
    hideError();
    if (!emptyState || !message) return;
    const errDiv = document.createElement("div");
    errDiv.id = "analyzeError";
    errDiv.className =
      "mb-3 p-3 rounded-xl border border-red-200 bg-red-50 text-sm text-red-800";
    errDiv.textContent = message;
    emptyState.insertBefore(errDiv, emptyState.firstChild);
  }

  /**
   * Remove any displayed error message.
   */
  function hideError() {
    const existing = document.getElementById("analyzeError");
    if (existing) existing.remove();
  }

  btnClear.addEventListener("click", () => {
    showEmpty();
    hideError();

    if (selectedTextValue) {
      selectedTextValue.textContent = "Nothing selected yet.";
      selectedTextValue.classList.remove("text-gray-800");
      selectedTextValue.classList.add("text-gray-400");
    }

    disableAnalyzeButton();
    clearSelectionOnPage();
    clearHighlightsOnPage();
  });
  btnCancelLoading.addEventListener("click", () => {
    showEmpty();
    hideError();
  });

  if (btnOpenHistory) {
    btnOpenHistory.addEventListener("click", () => {
      window.location.href = "../history/history.html";
    });
  }

  if (btnOpenSettings) {
    btnOpenSettings.addEventListener("click", () => {
      window.location.href = "../settings/settings.html";
    });
  }

  // -----------------------------
  // State helpers
  // -----------------------------
  function showEmpty() {
    emptyState.classList.remove("hidden");
    loadingState.classList.add("hidden");
    resultState.classList.add("hidden");

    if (selectionHeader) {
      selectionHeader.textContent = "No text selected";
    }
  }

  function showLoading() {
    emptyState.classList.add("hidden");
    loadingState.classList.remove("hidden");
    resultState.classList.add("hidden");
  }

  function showResult() {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultState.classList.remove("hidden");
    document.querySelector(".popup-main")?.scrollTo({ top: 0, behavior: "auto" });
  }

  function enableAnalyzeButton() {
    btnAnalyze.disabled = false;
    btnAnalyze.classList.remove("bg-gray-200", "text-gray-500", "cursor-not-allowed");
    btnAnalyze.classList.add(
      "bg-[#1e2c3e]",
      "text-white",
      "hover:opacity-95",
      "transition"
    );
    btnAnalyze.title = "";
  }

  function disableAnalyzeButton() {
    btnAnalyze.disabled = true;
    btnAnalyze.classList.remove("bg-[#1e2c3e]", "text-white", "hover:opacity-95");
    btnAnalyze.classList.add("bg-gray-200", "text-gray-500", "cursor-not-allowed");
    btnAnalyze.title = "Select text first to enable Analyze";
  }

  function refreshSelectionFromPage() {
    if (!selectedTextValue) return;

    const fallbackMessage =
      popupSettings.analysisMode === "selection_fallback"
        ? "No text selected. Fallback mode: page content will be analyzed when you click Analyze."
        : "Nothing selected yet.";

    // Default UI while we attempt to read selection
    selectedTextValue.textContent = fallbackMessage;
    // In selection_only mode, disable until text is selected. In fallback mode, enable so user can analyze page content.
    if (popupSettings.analysisMode === "selection_fallback") {
      enableAnalyzeButton();
    } else {
      disableAnalyzeButton();
    }

    if (typeof chrome === "undefined" || !chrome.tabs || !chrome.tabs.query) {
      selectedTextValue.textContent =
        popupSettings.analysisMode === "selection_fallback"
          ? fallbackMessage
          : "Selection not available in this context.";
      return;
    }

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (chrome.runtime.lastError || !tabs || !tabs.length || tabs[0].id == null) {
        selectedTextValue.textContent =
          popupSettings.analysisMode === "selection_fallback"
            ? fallbackMessage
            : "Unable to read selected text on this page.";
        return;
      }

      const tabId = tabs[0].id;

      chrome.tabs.sendMessage(
        tabId,
        { type: "fakeSha_getSelection" },
        (response) => {
          if (chrome.runtime.lastError || !response) {
            selectedTextValue.textContent =
              popupSettings.analysisMode === "selection_fallback"
                ? fallbackMessage
                : "Unable to read selected text on this page.";
            return;
          }

          const text = (response.text || "").trim();
          if (text) {
            selectedTextValue.textContent = text;
            selectedTextValue.classList.remove("text-gray-400");
            selectedTextValue.classList.add("text-gray-800");

            if (selectionHeader) {
              selectionHeader.textContent = "Text selected";
            }
            enableAnalyzeButton();
          } else {
            selectedTextValue.textContent = fallbackMessage;
            selectedTextValue.classList.remove("text-gray-800");
            selectedTextValue.classList.add("text-gray-400");

            if (selectionHeader) {
              selectionHeader.textContent = "No text selected";
            }
            // In fallback mode, keep Analyze enabled (page content can be used)
            if (popupSettings.analysisMode !== "selection_fallback") {
              disableAnalyzeButton();
            } else {
              enableAnalyzeButton();
            }
          }
        }
      );
    });
  }

  function clearSelectionOnPage() {
    if (typeof chrome === "undefined" || !chrome.tabs || !chrome.tabs.query) {
      return;
    }

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (chrome.runtime.lastError || !tabs || !tabs.length || tabs[0].id == null) {
        return;
      }

      const tabId = tabs[0].id;
      chrome.tabs.sendMessage(tabId, { type: "fakeSha_clearSelection" }, () => {
        // Ignore response; best-effort clear.
      });
    });
  }

  // -----------------------------
  // Rendering
  // -----------------------------
  function renderResult(data) {
    const theme = getThemeForData(data);

    const impactColor = (impact) => {
      if (impact === "high") return theme.tokenHigh;
      if (impact === "medium") return theme.tokenMed;
      return theme.tokenLow;
    };

    const indicatorRows = data.indicators
      .map((ind) => {
        const width = clamp(ind.contributionPct ?? 0, 0, 100);
        const shapStr = `${formatSigned(ind.shap)}%`;
        return `
          <div class="mt-4">
            <div class="h-3 w-full rounded-full" style="background:${theme.indicatorBg};">
              <div class="h-3 rounded-full" style="background:${theme.indicatorProgress}; width:${width}%;"></div>
            </div>

            <div class="mt-2 flex items-center justify-between">
              <div class="text-sm text-gray-400">${escapeHtml(ind.name)}</div>
              <div class="flex items-center gap-2">
                <div class="text-sm font-semibold text-[#1e2c3e]">${shapStr}</div>
                <div class="h-5 w-5 rounded-full flex items-center justify-center" style="background:${theme.indicatorBg};">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M6 20V10" stroke="${theme.indicatorProgress}" stroke-width="2" stroke-linecap="round"/>
                    <path d="M12 20V6" stroke="${theme.indicatorProgress}" stroke-width="2" stroke-linecap="round"/>
                    <path d="M18 20V14" stroke="${theme.indicatorProgress}" stroke-width="2" stroke-linecap="round"/>
                  </svg>
                </div>
              </div>
            </div>
          </div>
        `;
      })
      .join(" ");

    const tokenRows = data.topTokens
      .map((t) => {
        const dotLow = impactColor("low");
        const dotMed = impactColor("medium");
        const dotHigh = impactColor("high");

        // match your mockup style: one active dot (low OR medium OR high)
        const active = t.impact || "low";
        const activeColors = {
          low: [dotLow, "#ffffff", "#ffffff"],
          medium: ["#ffffff", dotMed, "#ffffff"],
          high: ["#ffffff", "#ffffff", dotHigh],
        };
        const [c1, c2, c3] = activeColors[active] || activeColors.low;

        return `
          <div class="flex items-center justify-between py-2 border-t border-gray-100">
            <div class="flex items-center gap-3 min-w-0">
              <div class="flex items-center gap-2 flex-shrink-0">
                <span class="h-3.5 w-3.5 rounded-full border" style="background:${c1}; border-color:#d1d5db;"></span>
                <span class="h-3.5 w-3.5 rounded-full border" style="background:${c2}; border-color:#d1d5db;"></span>
                <span class="h-3.5 w-3.5 rounded-full border" style="background:${c3}; border-color:#d1d5db;"></span>
              </div>
              <div class="text-sm tracking-wide text-gray-900 truncate">${escapeHtml(t.text)}</div>
            </div>

            <div class="ml-3 flex-shrink-0 text-xs font-semibold text-gray-700 px-2 py-1 rounded-md" style="background:#e5e7eb;">
              (${formatSigned(t.shap)}%)
            </div>
          </div>
        `;
      })
      .join(" ");

    resultState.innerHTML = `
      <section>
        <div class="mt-2 text-base font-bold text-[#1e2c3e]">Article: “${escapeHtml(data.articleTitle)}”</div>
        <div class="mt-1 text-xs text-gray-400 break-all">Source: ${escapeHtml(data.sourceUrl)}</div>

        <!-- Result Banner -->
        <div class="mt-4 rounded-xl border-2 p-4 flex gap-3 items-center" style="border-color:${theme.bannerBorder}; background:${theme.bannerBg};">
          <div class="h-9 w-9 rounded-lg flex items-center justify-center flex-shrink-0" style="background:${theme.indicatorBg};">
            ${theme.iconSvg}
          </div>

          <div class="min-w-0">
            <div class="text-sm font-extrabold tracking-wide" style="color:${theme.bannerText};">${escapeHtml(data.label)}</div>
            <div class="mt-1 text-sm" style="color:${theme.bannerText};">Confidence: <span class="font-extrabold">${data.confidence.toFixed(1)}%</span></div>
          </div>
        </div>

        <!-- Key Indicators -->
        <div class="mt-6">
          <div class="flex items-end justify-between">
            <div class="text-base font-bold text-[#1e2c3e]">Key Indicators</div>
            <div class="text-sm text-gray-500">SHAP Value</div>
          </div>
          ${indicatorRows}
        </div>

        <!-- Top Tokens -->
        <div class="mt-6">
          <div class="text-base font-bold text-[#1e2c3e]">${escapeHtml(data.topTokensTitle)}</div>

          <div class="mt-2 flex items-center gap-3">
            <div class="flex items-center gap-2">
              <span class="h-3.5 w-3.5 rounded-full" style="background:${theme.tokenLow};"></span>
              <span class="h-3.5 w-3.5 rounded-full" style="background:${theme.tokenMed};"></span>
              <span class="h-3.5 w-3.5 rounded-full" style="background:${theme.tokenHigh};"></span>
            </div>
            <div class="text-sm text-gray-400">${escapeHtml(data.topTokensLegend)}</div>
          </div>

          <div class="mt-3 border-b border-gray-200">
            ${tokenRows}
          </div>
        </div>

        <!-- Summary -->
        <div class="mt-5 rounded-xl border-2 p-4" style="border-color:#b7d4ff; background:#eaf3ff;">
          <div class="flex items-center gap-2">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M3 17L9 11" stroke="#2f6fd6" stroke-width="2" stroke-linecap="round"/>
              <path d="M13 7L21 3" stroke="#2f6fd6" stroke-width="2" stroke-linecap="round"/>
              <path d="M14 7L17 10" stroke="#2f6fd6" stroke-width="2" stroke-linecap="round"/>
              <path d="M7 13L10 16" stroke="#2f6fd6" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <div class="text-sm font-extrabold" style="color:#2f6fd6;">SUMMARY</div>
          </div>
          <div class="mt-2 text-sm text-[#1e2c3e] leading-relaxed">${escapeHtml(data.summary)}</div>
        </div>

        <!-- Actions -->
        <button
          id="btnReportIssue"
          type="button"
          class="mt-4 w-full py-3 rounded-xl font-extrabold tracking-wide text-white hover:opacity-95 transition"
          style="background:#1e2c3e;"
        >
          REPORT ISSUE
        </button>

        <button
          id="btnBack"
          type="button"
          class="mt-3 w-full py-2 rounded-xl text-sm font-semibold border border-gray-300 text-gray-800 hover:bg-gray-50 transition"
        >
          Back
        </button>

        <div class="mt-4 text-sm text-gray-400 italic">What does the indicators mean?</div>
      </section>
    `;

    // Back
    const btnBack = document.getElementById("btnBack");
    if (btnBack) btnBack.addEventListener("click", () => { hideError(); showEmpty(); });

    // Report issue (dummy)
    const btnReport = document.getElementById("btnReportIssue");
    if (btnReport) {
      btnReport.addEventListener("click", () => {
        btnReport.textContent = "REPORTED (DUMMY)";
        btnReport.disabled = true;
        btnReport.classList.add("opacity-80", "cursor-not-allowed");
      });
    }
  }

  function saveHistoryIfEnabled(resultData) {
    const proceedWithSave = (enabled) => {
      if (!enabled) return;

      const label = String(resultData.label || "").toUpperCase();
      const isFake = label.includes("FAKE");
      const verdict = isFake ? "Fake News" : "Real News";

      const confidenceNum =
        typeof resultData.confidence === "number"
          ? resultData.confidence
          : parseFloat(String(resultData.confidence || "0").replace("%", "")) || 0;

      const now = new Date();
      const record = {
        id: `${now.getTime()}-${Math.random().toString(36).slice(2, 8)}`,
        articleTitle: resultData.articleTitle || "Untitled",
        sourceUrl: resultData.sourceUrl || "",
        selectedText: (selectedTextValue && selectedTextValue.textContent) || "",
        verdict,
        confidence: confidenceNum,
        indicators: Array.isArray(resultData.indicators) ? resultData.indicators : [],
        summary: resultData.summary || "",
        label: resultData.label || (isFake ? "FAKE NEWS DETECTED" : "REAL NEWS DETECTED"),
        topTokensTitle: resultData.topTokensTitle || "Key tokens",
        topTokensLegend: resultData.topTokensLegend || "Impact",
        topTokens: Array.isArray(resultData.topTokens) ? resultData.topTokens : [],
        timestamp: now.toISOString(),
      };

      const persist = (records) => {
        const updated = [record, ...(Array.isArray(records) ? records : [])];

        try {
          if (storage) {
            storage.set({ [HISTORY_KEY]: updated }, () => {});
          } else {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
          }
        } catch (e) {
          // ignore write errors for now
        }
      };

      try {
        if (storage) {
          storage.get(HISTORY_KEY, (result) => {
            const existing = result && Array.isArray(result[HISTORY_KEY]) ? result[HISTORY_KEY] : [];
            persist(existing);
          });
        } else {
          const raw = localStorage.getItem(HISTORY_KEY);
          const parsed = raw ? JSON.parse(raw) : [];
          const existing = Array.isArray(parsed) ? parsed : [];
          persist(existing);
        }
      } catch (e) {
        // ignore read errors
      }
    };

    try {
      if (storage) {
        storage.get("fakeShaSettings", (result) => {
          const stored = result && result.fakeShaSettings;
          const enabled =
            stored && typeof stored.historyEnabled === "boolean"
              ? stored.historyEnabled
              : DEFAULT_POPUP_SETTINGS.historyEnabled;
          proceedWithSave(enabled);
        });
      } else {
        const rawSettings = localStorage.getItem("fakeShaSettings");
        let parsedSettings = null;
        try {
          parsedSettings = rawSettings ? JSON.parse(rawSettings) : null;
        } catch (e) {
          parsedSettings = null;
        }
        const enabled =
          parsedSettings && typeof parsedSettings.historyEnabled === "boolean"
            ? parsedSettings.historyEnabled
            : DEFAULT_POPUP_SETTINGS.historyEnabled;
        proceedWithSave(enabled);
      }
    } catch (e) {
      proceedWithSave(DEFAULT_POPUP_SETTINGS.historyEnabled);
    }
  }

  // -----------------------------
  // Utilities
  // -----------------------------
  function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
  }

  function formatSigned(n) {
    const num = Number(n);
    if (Number.isNaN(num)) return "0.0";
    const fixed = Math.abs(num).toFixed(1);
    return num < 0 ? `-${fixed}` : `+${fixed}`;
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function getThemeForData(data) {
    const isReal = String(data.label || "").toUpperCase().includes("REAL");
    return getThemeForMode(isReal ? "real" : "fake");
  }

  function getThemeForMode(mode) {
    if (mode === "real") {
      // Real News (Positive State)
      return {
        bannerText: "#035323",
        bannerBorder: "#16a34a",
        bannerBg: "#e9fff1",
        indicatorBg: "#dfffe9",
        indicatorProgress: "#16a34a",
        tokenLow: "#d0e6de",
        tokenMed: "#a5dfbe",
        tokenHigh: "#83cfa0",
        iconSvg: `
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 6L9 17L4 12" stroke="#035323" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        `.trim(),
      };
    }

    // Fake News (Negative State)
    return {
      bannerText: "#ad0516",
      bannerBorder: "#f56f70",
      bannerBg: "#fde9ea",
      indicatorBg: "#f6c6c8",
      indicatorProgress: "#f56f70",
      tokenLow: "#f9cbc7",
      tokenMed: "#f8a19e",
      tokenHigh: "#f25e5d",
      iconSvg: `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 9V13" stroke="#ad0516" stroke-width="2" stroke-linecap="round"/>
          <path d="M12 17H12.01" stroke="#ad0516" stroke-width="2" stroke-linecap="round"/>
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" stroke="#ad0516" stroke-width="2" stroke-linejoin="round"/>
        </svg>
      `.trim(),
    };
  }
});
