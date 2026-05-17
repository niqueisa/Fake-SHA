document.addEventListener("DOMContentLoaded", () => {
  const historyListView = document.getElementById("historyListView");
  const historyList = document.getElementById("historyList");
  const detailPanel = document.getElementById("detailPanel");
  const detailContent = document.getElementById("detailContent");
  const btnBackToRecords = document.getElementById("btnBackToRecords");
  const btnBack = document.getElementById("btnBack");
  const btnOpenSettings = document.getElementById("btnOpenSettings");
  const searchHistoryInput = document.getElementById("searchHistory");
  const btnSearchHistory = document.getElementById("btnSearchHistory");

  const HISTORY_KEY = "fakeShaHistory";
  const HISTORY_UI_KEY = "fakeShaHistoryUi";
  const popupMain = document.querySelector(".popup-main");

  // Store full records for client-side search (avoids re-fetching on each keystroke)
  let allRecords = [];
  let currentDetailId = null;

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

  function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
  }

  function confidenceToPercent(value) {
    const n =
      typeof value === "number"
        ? value
        : parseFloat(String(value || "0").replace("%", "")) || 0;
    if (n > 0 && n <= 1) {
      return (n * 100).toFixed(1);
    }
    return (Math.min(100, Math.max(0, n))).toFixed(1);
  }

  function formatConfidencePercent(value) {
    return `${confidenceToPercent(value)}%`;
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

  function normalizeRecord(item) {
    // Treat popup-saved records as canonical even when indicators are empty.
    if (item.articleTitle != null && Array.isArray(item.indicators) && (item.label || item.summary || item.topTokens)) {
      return {
        articleTitle: item.articleTitle || item.title || "Untitled",
        sourceUrl: item.sourceUrl || "",
        label: item.label || (item.isFake ? "FAKE NEWS DETECTED" : "REAL NEWS DETECTED"),
        confidence: parseFloat(confidenceToPercent(
          typeof item.confidenceNum === "number" ? item.confidenceNum : item.confidence
        )),
        indicators: item.indicators,
        summary: item.summary || item.explanation || "No summary available.",
        topTokensTitle: item.topTokensTitle || "Key tokens",
        topTokensLegend: item.topTokensLegend || "Impact",
        topTokens: Array.isArray(item.topTokens) ? item.topTokens : [],
      };
    }
    // Legacy fallback: rebuild minimal shape for older stored records.
    const confNum = parseFloat(confidenceToPercent(item.confidence));
    const indNames = Array.isArray(item.indicators) ? item.indicators : [];
    const indicators = indNames.map((name, i) => ({
      name: typeof name === "string" ? name : "Indicator",
      contributionPct: Math.max(5, 80 - i * 15),
    }));
    return {
      articleTitle: item.articleTitle || item.title || "Untitled",
      sourceUrl: item.sourceUrl || "",
      label: item.isFake ? "FAKE NEWS DETECTED" : "REAL NEWS DETECTED",
      confidence: confNum,
      indicators,
      summary: item.summary || item.explanation || "No summary available.",
      topTokensTitle: item.topTokensTitle || "Key tokens",
      topTokensLegend: item.topTokensLegend || "Impact",
      topTokens: Array.isArray(item.topTokens) ? item.topTokens : [],
    };
  }

  function renderResultDetail(data) {
    const theme = getThemeForData(data);

    function impactFromContributionPct(value) {
      const pct = clamp(Number(value ?? 0), 0, 100);
      if (pct >= 66.67) return "high";
      if (pct >= 33.34) return "medium";
      return "low";
    }

    const impactColor = (impact) => {
      if (impact === "high") return theme.tokenHigh;
      if (impact === "medium") return theme.tokenMed;
      return theme.tokenLow;
    };

    const indicatorRows = (data.indicators || [])
      .map((ind, idx) => {
        const width = clamp(Number(ind.contributionPct ?? 0), 0, 100);
        const contributionStr = `${width.toFixed(1)}%`;
        return `
          <div class="mt-4">
            <div class="h-3 w-full rounded-full" style="background:${theme.indicatorBg};">
              <div class="h-3 rounded-full" style="background:${theme.indicatorProgress}; width:${width}%;"></div>
            </div>
            <div class="mt-2 flex items-center justify-between">
              <div class="text-sm text-gray-400">${escapeHtml(ind.name)}</div>
              <div class="flex items-center gap-2">
                <div class="text-sm font-semibold text-[#1e2c3e]">${contributionStr}</div>
                <button
                  type="button"
                  class="detail-indicator-token-filter h-5 w-5 rounded-full flex items-center justify-center"
                  data-indicator-idx="${idx}"
                  title="Filter top tokens by this indicator"
                  style="background:${theme.indicatorBg};"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M6 20V10" stroke="${theme.indicatorProgress}" stroke-width="2" stroke-linecap="round"/>
                    <path d="M12 20V6" stroke="${theme.indicatorProgress}" stroke-width="2" stroke-linecap="round"/>
                    <path d="M18 20V14" stroke="${theme.indicatorProgress}" stroke-width="2" stroke-linecap="round"/>
                  </svg>
                </button>
              </div>
            </div>
          </div>
        `;
      })
      .join(" ");

    const tokens = data.topTokens || [];
    const tokenRows = tokens
      .map((t, i) => {
        const dotLow = impactColor("low");
        const dotMed = impactColor("medium");
        const dotHigh = impactColor("high");
        // Dot severity is based on token contribution percentage.
        const active = impactFromContributionPct(t.contributionPct);
        const activeColors = {
          low: [dotLow, "#ffffff", "#ffffff"],
          medium: ["#ffffff", dotMed, "#ffffff"],
          high: ["#ffffff", "#ffffff", dotHigh],
        };
        const [c1, c2, c3] = activeColors[active] || activeColors.low;
        return `
          <div
            class="detail-token-row ${i >= 5 ? "hidden" : ""} flex items-center justify-between py-2 border-t border-gray-100"
            data-token="${encodeURIComponent(String(t.text || "").toLowerCase())}"
          >
            <div class="flex items-center gap-3 min-w-0">
              <div class="flex items-center gap-2 flex-shrink-0">
                <span class="h-3.5 w-3.5 rounded-full border" style="background:${c1}; border-color:#d1d5db;"></span>
                <span class="h-3.5 w-3.5 rounded-full border" style="background:${c2}; border-color:#d1d5db;"></span>
                <span class="h-3.5 w-3.5 rounded-full border" style="background:${c3}; border-color:#d1d5db;"></span>
              </div>
              <div class="text-sm tracking-wide text-gray-900 truncate">${escapeHtml(t.text)}</div>
            </div>
            <div class="ml-3 flex-shrink-0 text-xs font-semibold text-gray-700 px-2 py-1 rounded-md" style="background:#e5e7eb;">
              (${Number(t.contributionPct || 0).toFixed(1)}%)
            </div>
          </div>
        `;
      })
      .join(" ");

    const confidenceVal = data.confidence;

    const tokensSection = tokens.length
      ? `
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
          <div id="detailTokenRowsContainer" class="mt-3 border-b border-gray-200">
            ${tokenRows}
          </div>
          <button
            id="btnToggleDetailTokens"
            type="button"
            data-expanded="false"
            class="${tokens.length > 5 ? "" : "hidden"} mt-2 text-xs font-semibold text-[#1e2c3e] hover:underline"
          >
            Show more tokens
          </button>
          <div id="detailTokenFilterHint" class="mt-1 text-xs text-gray-500 hidden"></div>
        </div>
      `
      : "";

    return `
      <section>
        <div class="text-base font-bold text-[#1e2c3e]">Article: "${escapeHtml(data.articleTitle)}"</div>
        <div class="mt-1 text-xs text-gray-400 break-all">Source: ${escapeHtml(data.sourceUrl)}</div>

        <div class="mt-4 rounded-xl border-2 p-4 flex gap-3 items-center" style="border-color:${theme.bannerBorder}; background:${theme.bannerBg};">
          <div class="h-9 w-9 rounded-lg flex items-center justify-center flex-shrink-0" style="background:${theme.indicatorBg};">
            ${theme.iconSvg}
          </div>
          <div class="min-w-0">
            <div class="text-sm font-extrabold tracking-wide" style="color:${theme.bannerText};">${escapeHtml(data.label)}</div>
            <div class="mt-1 text-sm" style="color:${theme.bannerText};">Confidence: <span class="font-extrabold">${formatConfidencePercent(confidenceVal)}</span></div>
          </div>
        </div>

        <div class="mt-6">
          <div class="flex items-end justify-between">
            <div class="text-base font-bold text-[#1e2c3e]">Key Indicators</div>
            <div class="text-sm text-gray-500">Contribution</div>
          </div>
          ${indicatorRows}
        </div>

        ${tokensSection}

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

        ${buildIndicatorGlossarySection(data)}
      </section>
    `;
  }

  function buildIndicatorGlossarySection(data) {
    const detectedNames = (data?.indicators || []).map((ind) => ind?.name).filter(Boolean);
    if (window.FakeShaIndicatorGlossary?.buildIndicatorGlossaryHtml) {
      return window.FakeShaIndicatorGlossary.buildIndicatorGlossaryHtml({ detectedNames });
    }
    return `
      <div class="mt-4 rounded-xl border border-gray-200 bg-gray-50 p-4">
        <div class="text-sm font-semibold text-[#1e2c3e]">What do the indicators mean?</div>
        <p class="mt-1 text-xs text-gray-600 leading-relaxed">
          Indicators group SHAP token contributions into readable categories.
        </p>
      </div>
    `;
  }

  function saveHistoryUiState() {
    if (!window.FakeShaNav) return;
    const inDetail = detailPanel && !detailPanel.classList.contains("hidden");
    window.FakeShaNav.savePageUiState(HISTORY_UI_KEY, {
      view: inDetail ? "detail" : "list",
      recordId: currentDetailId,
      searchQuery: searchHistoryInput ? searchHistoryInput.value : "",
      scrollTop: popupMain ? popupMain.scrollTop : 0,
    });
  }

  function showListView() {
    currentDetailId = null;
    if (historyListView) historyListView.classList.remove("hidden");
    if (detailPanel) detailPanel.classList.add("hidden");
    saveHistoryUiState();
  }

  function showDetailView() {
    if (historyListView) historyListView.classList.add("hidden");
    if (detailPanel) detailPanel.classList.remove("hidden");
    document.querySelector(".popup-main")?.scrollTo({ top: 0, behavior: "auto" });
    saveHistoryUiState();
  }

  function showDetails(item) {
    currentDetailId = item && item.id ? item.id : null;
    const data = normalizeRecord(item);
    if (detailContent) detailContent.innerHTML = renderResultDetail(data);
    setupDetailTokenInteractions(data);
    if (window.FakeShaIndicatorGlossary?.setupIndicatorGlossaryToggle) {
      window.FakeShaIndicatorGlossary.setupIndicatorGlossaryToggle(detailContent);
    }
    showDetailView();
  }

  function setupDetailTokenInteractions(data) {
    if (!detailContent) return;
    const btnToggle = detailContent.querySelector("#btnToggleDetailTokens");
    const filterHint = detailContent.querySelector("#detailTokenFilterHint");
    let expanded = false;
    let activeIndicatorIdx = null;

    function applyTokenVisibility() {
      const rows = Array.from(detailContent.querySelectorAll(".detail-token-row"));
      const selectedIndicator = activeIndicatorIdx == null ? null : data.indicators?.[activeIndicatorIdx];
      const tokenSet = new Set((selectedIndicator?.tokens || []).map((s) => encodeURIComponent(String(s).toLowerCase())));

      // Apply both indicator filter and "show more" pagination in one pass.
      let visibleCounter = 0;
      rows.forEach((row) => {
        const token = row.getAttribute("data-token") || "";
        const passFilter = !selectedIndicator || tokenSet.has(token);
        row.classList.toggle("hidden", !passFilter);
        if (passFilter) {
          visibleCounter += 1;
          const hideForOverflow = !expanded && visibleCounter > 5;
          row.classList.toggle("hidden", hideForOverflow);
        }
      });

      if (btnToggle) {
        btnToggle.classList.toggle("hidden", visibleCounter <= 5);
        btnToggle.textContent = expanded ? "Show fewer tokens" : "Show more tokens";
        btnToggle.dataset.expanded = expanded ? "true" : "false";
      }

      if (filterHint) {
        if (!selectedIndicator) {
          filterHint.classList.add("hidden");
          filterHint.textContent = "";
        } else {
          filterHint.classList.remove("hidden");
          filterHint.textContent = `Filtered by: ${selectedIndicator.name || "Indicator"}`;
        }
      }
    }

    if (btnToggle) {
      btnToggle.addEventListener("click", () => {
        expanded = !expanded;
        applyTokenVisibility();
      });
    }

    detailContent.querySelectorAll(".detail-indicator-token-filter").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.getAttribute("data-indicator-idx"));
        if (!Number.isFinite(idx)) return;
        activeIndicatorIdx = activeIndicatorIdx === idx ? null : idx;
        expanded = false;
        applyTokenVisibility();
      });
    });

    applyTokenVisibility();
  }

  /**
   * Filter records by search query (keyword or URI).
   * Matches against: article title, source URL, selected text, verdict, summary.
   */
  function filterRecords(records, query) {
    if (!query || !String(query).trim()) return records;
    const q = String(query).trim().toLowerCase();
    return records.filter((r) => {
      const title = String(r.articleTitle || r.title || "").toLowerCase();
      const url = String(r.sourceUrl || "").toLowerCase();
      const selected = String(r.selectedText || "").toLowerCase();
      const verdict = String(r.verdict || r.label || "").toLowerCase();
      const summary = String(r.summary || "").toLowerCase();
      return (
        title.includes(q) ||
        url.includes(q) ||
        selected.includes(q) ||
        verdict.includes(q) ||
        summary.includes(q)
      );
    });
  }

  function renderRecords(records) {
    if (!historyList) return;

      historyList.innerHTML = "";

      if (!Array.isArray(records) || records.length === 0) {
        const empty = document.createElement("div");
        empty.className = "text-sm text-gray-400 text-center mt-4";
        const query = searchHistoryInput ? searchHistoryInput.value.trim() : "";
        empty.textContent = query ? "No matches found." : "No history entries yet.";
        historyList.appendChild(empty);
        return;
      }

      records.forEach((raw) => {
        const label = String(raw.label || "").toUpperCase();
        const isFake = label.includes("FAKE");
        const verdict = raw.verdict || (isFake ? "Fake News" : "Real News");

        const confidenceNum = parseFloat(confidenceToPercent(raw.confidence));

        const title = raw.articleTitle || raw.title || "Untitled";

        let dateText = "";
        if (raw.timestamp) {
          const d = new Date(raw.timestamp);
          if (!Number.isNaN(d.getTime())) {
            dateText = d.toLocaleDateString();
          }
        }

        const card = document.createElement("div");
        card.className =
          "p-3 rounded-xl border border-gray-100 bg-white shadow-sm hover:border-blue-200 cursor-pointer transition";

        const iconColor = isFake ? "text-red-500" : "text-green-500";
        const iconPath = isFake
          ? "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          : "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z";

        const confidenceDisplay = formatConfidencePercent(confidenceNum);

        card.innerHTML = `
          <div class="flex items-start gap-3">
            <div class="mt-1 ${iconColor}">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${iconPath}" />
              </svg>
            </div>
            <div>
              <div class="text-sm font-semibold text-gray-800">${escapeHtml(title)}</div>
              <div class="text-xs text-gray-500 mt-1">
                Classified: <span class="font-medium">${verdict} (${confidenceDisplay})</span>
              </div>
              ${
                dateText
                  ? `<div class="text-xs text-gray-400">Date: ${escapeHtml(dateText)}</div>`
                  : ""
              }
            </div>
          </div>
        `;

        card.addEventListener("click", () => showDetails(raw));
        historyList.appendChild(card);
      });
  }

  /**
   * Apply current search query to allRecords and re-render the list.
   */
  function applySearchAndRender() {
    const query = searchHistoryInput ? searchHistoryInput.value.trim() : "";
    const filtered = filterRecords(allRecords, query);
    renderRecords(filtered);
  }

  function restoreHistoryUiState() {
    if (!window.FakeShaNav) return;
    window.FakeShaNav.loadPageUiState(HISTORY_UI_KEY, (ui) => {
      if (!ui) return;
      if (searchHistoryInput && typeof ui.searchQuery === "string") {
        searchHistoryInput.value = ui.searchQuery;
      }
      applySearchAndRender();
      if (ui.view === "detail" && ui.recordId) {
        const record = allRecords.find((r) => r && r.id === ui.recordId);
        if (record) {
          showDetails(record);
        }
      }
      if (popupMain && typeof ui.scrollTop === "number") {
        popupMain.scrollTop = ui.scrollTop;
      }
    });
  }

  function loadAndRenderHistory(done) {
    const finish = () => {
      applySearchAndRender();
      restoreHistoryUiState();
      if (typeof done === "function") done();
    };
    try {
      if (storage) {
        storage.get(HISTORY_KEY, (result) => {
          allRecords = result && Array.isArray(result[HISTORY_KEY]) ? result[HISTORY_KEY] : [];
          finish();
        });
      } else {
        const raw = localStorage.getItem(HISTORY_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        allRecords = Array.isArray(parsed) ? parsed : [];
        finish();
      }
    } catch (e) {
      allRecords = [];
      finish();
    }
  }

  if (btnBackToRecords) {
    btnBackToRecords.addEventListener("click", showListView);
  }

  btnBack.addEventListener("click", () => {
    if (window.FakeShaNav) {
      window.FakeShaNav.navigateTo("popup/popup.html");
    } else {
      window.location.href = "../popup/popup.html";
    }
  });

  if (btnOpenSettings) {
    btnOpenSettings.addEventListener("click", () => {
      if (window.FakeShaNav) {
        window.FakeShaNav.navigateTo("settings/settings.html");
      } else {
        window.location.href = "../settings/settings.html";
      }
    });
  }

  window.addEventListener("pagehide", saveHistoryUiState);

  // Search: filter on input and on button click
  if (searchHistoryInput) {
    searchHistoryInput.addEventListener("input", applySearchAndRender);
    searchHistoryInput.addEventListener("keyup", (e) => {
      if (e.key === "Enter") applySearchAndRender();
    });
  }
  if (btnSearchHistory) {
    btnSearchHistory.addEventListener("click", applySearchAndRender);
  }

  loadAndRenderHistory();
});
