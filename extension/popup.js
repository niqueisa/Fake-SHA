document.addEventListener("DOMContentLoaded", () => {
  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const resultState = document.getElementById("resultState");

  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnClear = document.getElementById("btnClear");
  const btnCancelLoading = document.getElementById("btnCancelLoading");

  // Default state
  showEmpty();

  // Enable Analyze button (simulate ready to analyze)
  btnAnalyze.disabled = false;
  btnAnalyze.classList.remove("bg-gray-200", "text-gray-500", "cursor-not-allowed");
  btnAnalyze.classList.add("bg-[#1e2c3e]", "text-white", "hover:opacity-95", "transition");

  // ---------- Event Listeners ----------
  btnAnalyze.addEventListener("click", () => {
    showLoading();

    // Simulate 2-second analysis delay
    setTimeout(() => {
      showResultPlaceholder();
    }, 2000);
  });

  btnClear.addEventListener("click", showEmpty);
  btnCancelLoading.addEventListener("click", showEmpty);

  // ---------- State Functions ----------
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

  function showResultPlaceholder() {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultState.classList.remove("hidden");

    // Inject Result UI with Back button
    resultState.innerHTML = `
      <!-- Verdict & Confidence Banner -->
      <div id="verdictBanner" class="p-3 rounded-xl text-white font-bold">
        Verdict Text
        <p id="confidenceText" class="text-sm font-normal mt-1">Confidence: 0%</p>
      </div>

      <!-- Indicators Section (5 indicators) -->
      <div id="indicatorsSection" class="space-y-3">
        ${[1,2,3,4,5].map(i => `
          <div class="flex justify-between text-sm">
            <span>Indicator ${i}</span>
            <span class="shap-value px-1 rounded bg-gray-200 text-xs">0.5</span>
          </div>
          <div class="h-2 bg-gray-200 rounded overflow-hidden">
            <div class="indicator-bar h-full"></div>
          </div>
        `).join('')}
      </div>

      <!-- Chart Placeholder -->
      <div id="chartSection" class="h-24 bg-gray-200 rounded flex items-center justify-center text-gray-500">
        Chart Placeholder
      </div>

      <!-- Explainability / Top Tokens -->
      <div id="explainSection" class="p-2 bg-gray-50 rounded space-y-1">
        <p class="font-semibold text-sm">Top Tokens:</p>
        <div class="flex flex-wrap gap-1">
          <span class="px-2 py-1 bg-gray-200 rounded text-xs">token1</span>
          <span class="px-2 py-1 bg-gray-200 rounded text-xs">token2</span>
          <span class="px-2 py-1 bg-gray-200 rounded text-xs">token3</span>
          <span class="px-2 py-1 bg-gray-200 rounded text-xs">token4</span>
          <span class="px-2 py-1 bg-gray-200 rounded text-xs">token5</span>
        </div>
      </div>

      <!-- Back Button -->
      <button id="btnBackToEmpty" class="mt-4 w-full py-2 rounded-xl bg-[#1e2c3e] text-white font-semibold hover:opacity-95 transition">
        Back
      </button>
    `;

    // <-- Add Back button listener here -->
    document.getElementById("btnBackToEmpty").addEventListener("click", showEmpty);

    // Initialize Fake/Real toggle for demo
    setupDummyDataToggle();
  }

  // ---------- Dummy Data Toggle ----------
  function setupDummyDataToggle() {
    const states = {
      fake: {
        verdict: "Fake News",
        confidence: "72%",
        bannerBg: "#f6c6c8",
        textColor: "#ad0516",
        indicatorColors: ["#f9cbc7","#f8a19e","#f25e5d","#f9cbc7","#f8a19e"]
      },
      real: {
        verdict: "Real News",
        confidence: "88%",
        bannerBg: "#dfffe9",
        textColor: "#035323",
        indicatorColors: ["#d0e6de","#a5dfbe","#83cfa0","#d0e6de","#a5dfbe"]
      }
    };

    let currentState = "fake";

    const verdictBanner = document.getElementById("verdictBanner");
    const confidenceText = document.getElementById("confidenceText");
    const indicatorBars = document.querySelectorAll(".indicator-bar");

    // Update immediately
    updateResultUI(states[currentState], verdictBanner, confidenceText, indicatorBars);

    // Clicking Analyze again toggles Fake ↔ Real
    btnAnalyze.addEventListener("click", () => {
      currentState = currentState === "fake" ? "real" : "fake";
      updateResultUI(states[currentState], verdictBanner, confidenceText, indicatorBars);
    });
  }

  // ---------- Update UI Helper ----------
  function updateResultUI(data, verdictBanner, confidenceText, indicatorBars) {
    // Verdict banner
    verdictBanner.textContent = data.verdict;
    verdictBanner.style.backgroundColor = data.bannerBg;
    verdictBanner.style.color = data.textColor;

    // Confidence text
    confidenceText.textContent = `Confidence: ${data.confidence}`;
    confidenceText.style.color = data.textColor;

    // Indicator bars
    indicatorBars.forEach((bar, i) => {
      bar.style.backgroundColor = data.indicatorColors[i];
      bar.style.width = `${Math.floor(Math.random() * 70 + 30)}%`; // random width for demo
    });
  }
});