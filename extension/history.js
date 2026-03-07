document.addEventListener("DOMContentLoaded", () => {
  const historyListView = document.getElementById("historyListView");
  const historyList = document.getElementById("historyList");
  const detailPanel = document.getElementById("detailPanel");
  const detailContent = document.getElementById("detailContent");
  const btnBackToRecords = document.getElementById("btnBackToRecords");
  const btnBack = document.getElementById("btnBack");
  const btnOpenSettings = document.getElementById("btnOpenSettings");

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
    if (item.articleTitle != null && Array.isArray(item.indicators) && item.indicators.length > 0 && typeof item.indicators[0] === "object") {
      return {
        articleTitle: item.articleTitle || item.title || "Untitled",
        sourceUrl: item.sourceUrl || "",
        label: item.label || (item.isFake ? "FAKE NEWS DETECTED" : "REAL NEWS DETECTED"),
        confidence: typeof item.confidenceNum === "number" ? item.confidenceNum : parseFloat(String(item.confidence || "0").replace("%", "")) || 0,
        indicators: item.indicators,
        summary: item.summary || item.explanation || "No summary available.",
        topTokensTitle: item.topTokensTitle || "Key tokens",
        topTokensLegend: item.topTokensLegend || "Impact",
        topTokens: Array.isArray(item.topTokens) ? item.topTokens : [],
      };
    }
    const confNum = parseFloat(String(item.confidence || "0").replace("%", "")) || 0;
    const indNames = Array.isArray(item.indicators) ? item.indicators : [];
    const indicators = indNames.map((name, i) => ({
      name: typeof name === "string" ? name : "Indicator",
      shap: item.isFake ? -15 : 20,
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

    const impactColor = (impact) => {
      if (impact === "high") return theme.tokenHigh;
      if (impact === "medium") return theme.tokenMed;
      return theme.tokenLow;
    };

    const indicatorRows = (data.indicators || [])
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

    const tokens = data.topTokens || [];
    const tokenRows = tokens
      .map((t) => {
        const dotLow = impactColor("low");
        const dotMed = impactColor("medium");
        const dotHigh = impactColor("high");
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

    const confidenceVal = typeof data.confidence === "number" ? data.confidence : parseFloat(String(data.confidence || "0").replace("%", "")) || 0;

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
          <div class="mt-3 border-b border-gray-200">
            ${tokenRows}
          </div>
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
            <div class="mt-1 text-sm" style="color:${theme.bannerText};">Confidence: <span class="font-extrabold">${confidenceVal.toFixed(1)}%</span></div>
          </div>
        </div>

        <div class="mt-6">
          <div class="flex items-end justify-between">
            <div class="text-base font-bold text-[#1e2c3e]">Key Indicators</div>
            <div class="text-sm text-gray-500">SHAP Value</div>
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

        <div class="mt-4 text-sm text-gray-400 italic">What does the indicators mean?</div>
      </section>
    `;
  }

  function showListView() {
    if (historyListView) historyListView.classList.remove("hidden");
    if (detailPanel) detailPanel.classList.add("hidden");
  }

  function showDetailView() {
    if (historyListView) historyListView.classList.add("hidden");
    if (detailPanel) detailPanel.classList.remove("hidden");
    document.querySelector(".popup-main")?.scrollTo({ top: 0, behavior: "auto" });
  }

  function showDetails(item) {
    const data = normalizeRecord(item);
    if (detailContent) detailContent.innerHTML = renderResultDetail(data);
    showDetailView();
  }

  function renderHistory() {
    if (!historyList) return;

    const renderRecords = (records) => {
      historyList.innerHTML = "";

      if (!Array.isArray(records) || records.length === 0) {
        const empty = document.createElement("div");
        empty.className = "text-sm text-gray-400 text-center mt-4";
        empty.textContent = "No history entries yet.";
        historyList.appendChild(empty);
        return;
      }

      records.forEach((raw) => {
        const label = String(raw.label || "").toUpperCase();
        const isFake = label.includes("FAKE");
        const verdict = raw.verdict || (isFake ? "Fake News" : "Real News");

        const confidenceNum =
          typeof raw.confidence === "number"
            ? raw.confidence
            : parseFloat(String(raw.confidence || "0").replace("%", "")) || 0;

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

        const confidenceDisplay = `${confidenceNum.toFixed(1)}%`;

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
    };

    try {
      if (storage) {
        storage.get(HISTORY_KEY, (result) => {
          const records = result && Array.isArray(result[HISTORY_KEY]) ? result[HISTORY_KEY] : [];
          renderRecords(records);
        });
      } else {
        const raw = localStorage.getItem(HISTORY_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        const records = Array.isArray(parsed) ? parsed : [];
        renderRecords(records);
      }
    } catch (e) {
      renderRecords([]);
    }
  }

  if (btnBackToRecords) {
    btnBackToRecords.addEventListener("click", showListView);
  }

  btnBack.addEventListener("click", () => {
    window.location.href = "popup.html";
  });

  if (btnOpenSettings) {
    btnOpenSettings.addEventListener("click", () => {
      window.location.href = "settings.html";
    });
  }

  renderHistory();
});
