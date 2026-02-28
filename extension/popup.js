document.addEventListener("DOMContentLoaded", () => {
  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const resultState = document.getElementById("resultState");

  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnClear = document.getElementById("btnClear");
  const btnCancelLoading = document.getElementById("btnCancelLoading");
  const btnOpenHistory = document.getElementById("btnOpenHistory");

  // -----------------------------
  // Dummy data (placeholders)
  // -----------------------------
  const dummyFakeResult = {
    articleTitle: "K12 NO MORE DEPED",
    sourceUrl: "https://prcjobhiring.blogspot.com/2025/04/k12-no-more.html",
    label: "FAKE NEWS DETECTED",
    confidence: 94.7,
    indicators: [
      { name: "Source Credibility", shap: -25.0, contributionPct: 22 },
      { name: "Claim Verification", shap: -30.0, contributionPct: 28 },
      { name: "Language Tone", shap: -15.2, contributionPct: 14 },
      { name: "Sensational Wording", shap: -14.5, contributionPct: 13 },
      { name: "Consistency with Known Facts", shap: -10.0, contributionPct: 10 },
    ],
    topTokensTitle: "Top Tokens Contributing to Misinformation",
    topTokensLegend: "High Misinformation Impact",
    topTokens: [
      { text: "CLICK HERE", shap: -6.2, impact: "high" },
      { text: "NO GRADE 11", shap: -5.8, impact: "high" },
      { text: "K12 NO MORE", shap: -5.0, impact: "high" },
      { text: "KUNG TAGA SAAN KA?", shap: -4.5, impact: "high" },
      { text: "SCHOOL YEAR 2025 -2016", shap: -3.1, impact: "medium" },
      { text: "ANNOUNCEMENT ||", shap: -2.2, impact: "low" },
    ],
    summary:
      "Primary Reason: High-Confidence Misinformation Alert for Claim Mismatch & Sensationalism",
  };

  const dummyRealResult = {
    articleTitle: "Duterte stays in detention; ICC Appeals …",
    sourceUrl: "https://verafiles.org/articles/duterte-stays-in-detention-icc-appeals-…",
    label: "REAL NEWS DETECTED",
    confidence: 98.2,
    indicators: [
      { name: "Source Credibility", shap: 45.0, contributionPct: 62 },
      { name: "Claim Verification", shap: 40.0, contributionPct: 56 },
      { name: "Language Tone", shap: 5.2, contributionPct: 18 },
      { name: "Sensational Wording", shap: 4.5, contributionPct: 16 },
      { name: "Consistency with Known Facts", shap: 3.5, contributionPct: 14 },
    ],
    topTokensTitle: "Top Tokens Contributing to Authenticity",
    topTokensLegend: "High Authenticity Impact",
    topTokens: [
      {
        text: "Appeals Chamber of the International Criminal Court",
        shap: 18.5,
        impact: "high",
      },
      { text: "rejected all three grounds", shap: 10.0, impact: "medium" },
      { text: "Sept. 26 ruling (released Oct.10)", shap: 6.5, impact: "low" },
      {
        text: "Presiding Judge Luz del Carmen Ibañez Carranza",
        shap: 5.0,
        impact: "low",
      },
    ],
    summary: "Primary Reason: Verified Institutional Source & Objective Reporting",
  };

  // Temporary UI toggle state (dummy only)
  let currentMode = "fake"; // "fake" | "real"

  // Default state
  showEmpty();

  // Simulate enabling Analyze (dummy flow)
  enableAnalyzeButton();

  btnAnalyze.addEventListener("click", () => {
    showLoading();
    setTimeout(() => {
      renderResult(currentMode === "real" ? dummyRealResult : dummyFakeResult);
      showResult();
    }, 1200);
  });

  btnClear.addEventListener("click", () => showEmpty());
  btnCancelLoading.addEventListener("click", () => showEmpty());

  if (btnOpenHistory) {
    btnOpenHistory.addEventListener("click", () => {
      window.location.href = "history.html";
    });
  }

  // -----------------------------
  // State helpers
  // -----------------------------
  function showEmpty() {
    emptyState.classList.remove("hidden");
    loadingState.classList.add("hidden");
    resultState.classList.add("hidden");
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

    // Toggle colors: fake uses fake progress color, real uses real progress color
    const toggleBg = currentMode === "real" ? getThemeForMode("real").indicatorProgress : getThemeForMode("fake").indicatorProgress;

    resultState.innerHTML = `
      <section>
        <!-- TEMP toggle (dummy only) -->
        <div class="flex items-center justify-end">
          <div class="flex items-center gap-2 rounded-full border border-gray-200 bg-white px-2 py-1">
            <span class="text-[11px] font-semibold ${currentMode === "fake" ? "text-[#1e2c3e]" : "text-gray-400"}">Fake</span>
            <button
              id="modeToggle"
              type="button"
              class="relative h-6 w-11 rounded-full transition"
              style="background:${toggleBg};"
              aria-label="Toggle result mode"
            >
              <span
                class="absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition"
                style="left:${currentMode === "real" ? "24px" : "2px"};"
              ></span>
            </button>
            <span class="text-[11px] font-semibold ${currentMode === "real" ? "text-[#1e2c3e]" : "text-gray-400"}">Real</span>
          </div>
        </div>

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

    // Toggle handler
    const modeToggle = document.getElementById("modeToggle");
    if (modeToggle) {
      modeToggle.addEventListener("click", () => {
        currentMode = currentMode === "real" ? "fake" : "real";
        renderResult(currentMode === "real" ? dummyRealResult : dummyFakeResult);
        document.querySelector(".popup-main")?.scrollTo({ top: 0, behavior: "auto" });
      });
    }

    // Back
    const btnBack = document.getElementById("btnBack");
    if (btnBack) btnBack.addEventListener("click", showEmpty);

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
