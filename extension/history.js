document.addEventListener("DOMContentLoaded", () => {
  const historyList = document.getElementById("historyList");
  const detailPanel = document.getElementById("detailPanel");
  const detailContent = document.getElementById("detailContent");
  const btnCloseDetail = document.getElementById("btnCloseDetail");
  const btnBack = document.getElementById("btnBack");

  // Dummy Data
  const dummyHistory = [
    {
      id: 1,
      title: "Climate change hoax article",
      verdict: "Fake News",
      confidence: "92.3%",
      date: "2025-11-15",
      isFake: true,
      explanation: "Lacks scientific consensus and uses emotionally charged language.",
      indicators: ["Sensationalism", "Unverified Source"]
    },
    {
      id: 2,
      title: "Scientific research findings",
      verdict: "Real News",
      confidence: "97.8%",
      date: "2025-11-14",
      isFake: false,
      explanation: "Cross-referenced with peer-reviewed journals and official data.",
      indicators: ["Cites Sources", "Neutral Tone"]
    },
    {
      id: 3,
      title: "Celebrity scandal story",
      verdict: "Fake News",
      confidence: "89.4%",
      date: "2025-11-13",
      isFake: true,
      explanation: "Source is a known satirical website not clearly labeled.",
      indicators: ["Satire", "Missing Attribution"]
    }
  ];

  function renderHistory() {
    historyList.innerHTML = "";
    dummyHistory.forEach(item => {
      const card = document.createElement("div");
      card.className = "p-3 rounded-xl border border-gray-100 bg-white shadow-sm hover:border-blue-200 cursor-pointer transition";
      
      const iconColor = item.isFake ? "text-red-500" : "text-green-500";
      const iconPath = item.isFake 
        ? "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
        : "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z";

      card.innerHTML = `
        <div class="flex items-start gap-3">
          <div class="mt-1 ${iconColor}">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${iconPath}" />
            </svg>
          </div>
          <div>
            <div class="text-sm font-semibold text-gray-800">${item.title}</div>
            <div class="text-xs text-gray-500 mt-1">
              Classified: <span class="font-medium">${item.verdict} (${item.confidence})</span>
            </div>
            <div class="text-xs text-gray-400">Date: ${item.date}</div>
          </div>
        </div>
      `;

      card.addEventListener("click", () => showDetails(item));
      historyList.appendChild(card);
    });
  }

  function showDetails(item) {
    historyList.classList.add("hidden");
    detailPanel.classList.remove("hidden");
    
    detailContent.innerHTML = `
      <div class="text-sm space-y-2">
        <p><strong>Verdict:</strong> <span class="${item.isFake ? 'text-red-500' : 'text-green-500'} font-bold">${item.verdict}</span></p>
        <p><strong>Confidence:</strong> ${item.confidence}</p>
        <p><strong>Indicators:</strong> ${item.indicators.join(", ")}</p>
        <p class="mt-2 text-gray-600 italic border-l-2 border-gray-200 pl-2">"${item.explanation}"</p>
      </div>
    `;
  }

  btnCloseDetail.addEventListener("click", () => {
    detailPanel.classList.add("hidden");
    historyList.classList.remove("hidden");
  });

  btnBack.addEventListener("click", () => {
    window.location.href = "popup.html";
  });

  renderHistory();
});