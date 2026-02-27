document.addEventListener("DOMContentLoaded", () => {
  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const resultState = document.getElementById("resultState");

  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnClear = document.getElementById("btnClear");
  const btnCancelLoading = document.getElementById("btnCancelLoading");

  // Default state
  showEmpty();

  // Simulate enabling Analyze (for now)
  btnAnalyze.disabled = false;
  btnAnalyze.classList.remove("bg-gray-200", "text-gray-500", "cursor-not-allowed");
  btnAnalyze.classList.add("bg-[#1e2c3e]", "text-white", "hover:opacity-95", "transition");

  btnAnalyze.addEventListener("click", () => {
    showLoading();

    // Simulate analysis delay (2 seconds)
    setTimeout(() => {
      showResultPlaceholder();
    }, 2000);
  });

  btnClear.addEventListener("click", () => {
    showEmpty();
  });

  btnCancelLoading.addEventListener("click", () => {
    showEmpty();
  });

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

    resultState.innerHTML = `
      <div class="text-sm text-gray-700 font-semibold">
        Analysis complete (placeholder).
      </div>
      <p class="mt-2 text-sm text-gray-600">
        Here magpapakita yung results.
      </p>
      <button
        class="mt-4 w-full py-2 rounded-xl bg-[#1e2c3e] text-white font-semibold hover:opacity-95 transition"
        id="btnBackToEmpty"
      >
        Back
      </button>
    `;

    document.getElementById("btnBackToEmpty").addEventListener("click", showEmpty);
  }
});

const btnOpenHistory = document.getElementById("btnOpenHistory");
btnOpenHistory.addEventListener("click", () => {
  window.location.href = "history.html";
});