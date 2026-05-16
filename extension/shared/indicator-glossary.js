/**
 * Static glossary for SHAP indicator categories (popup + history).
 * Names align with backend explainability groupings in xlmr_shap.py.
 */
(function (global) {
  "use strict";

  const INDICATOR_GLOSSARY = [
    {
      name: "Textual Source Attribution Mentions",
      description:
        "References to people, organizations, documents, or channels presented as the source of a claim (e.g., officials, agencies, witnesses, “according to”).",
    },
    {
      name: "Presence of Evidence-related Language",
      description:
        "Wording that frames proof, data, studies, records, or verification (e.g., evidence, report, confirmed, documented).",
    },
    {
      name: "Sensationalism",
      description:
        "Attention-grabbing or emotionally charged phrasing (e.g., shocking, exclusive, urgent, viral) that may amplify perceived misinformation risk.",
    },
    {
      name: "Claim Certainty",
      description:
        "Signals of strong certainty or hedging (e.g., definitely, proven, alleged, reportedly) that affect how decisive the text sounds.",
    },
    {
      name: "Linguistic Tone",
      description:
        "Emotionally loaded or evaluative tone (e.g., outrage, praise, insults, fear) that can steer the model toward FAKE or REAL patterns.",
    },
  ];

  const CHEVRON_SVG = `
    <svg class="indicator-glossary-chevron" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M7 10l5 5 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  `;

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  /**
   * @param {{ detectedNames?: string[], heading?: string }} [options]
   */
  function buildIndicatorGlossaryHtml(options) {
    const opts = options && typeof options === "object" ? options : {};
    const heading = opts.heading || "What do the indicators mean?";
    const detected = new Set(
      (opts.detectedNames || [])
        .map((name) => String(name || "").trim())
        .filter(Boolean)
    );

    const rows = INDICATOR_GLOSSARY.map((item, idx) => {
      const inResult = detected.has(item.name);
      const badge = inResult
        ? '<span class="indicator-glossary-item-badge">In this result</span>'
        : "";
      return `
        <div class="indicator-glossary-item">
          <button
            type="button"
            class="indicator-glossary-item-toggle"
            data-glossary-idx="${idx}"
            aria-expanded="false"
            aria-controls="indicator-glossary-item-${idx}"
          >
            <span class="indicator-glossary-item-label">
              <span class="indicator-glossary-item-title">${escapeHtml(item.name)}</span>
              ${badge}
            </span>
            ${CHEVRON_SVG}
          </button>
          <div
            id="indicator-glossary-item-${idx}"
            class="indicator-glossary-item-panel is-collapsed"
            role="region"
          >
            <p class="indicator-glossary-item-desc">${escapeHtml(item.description)}</p>
          </div>
        </div>
      `;
    }).join("");

    return `
      <div class="indicator-glossary">
        <button
          type="button"
          class="indicator-glossary-toggle"
          aria-expanded="false"
          aria-controls="indicator-glossary-panel"
        >
          <span class="indicator-glossary-toggle-label">${escapeHtml(heading)}</span>
          ${CHEVRON_SVG}
        </button>
        <div
          id="indicator-glossary-panel"
          class="indicator-glossary-panel is-collapsed"
          role="region"
        >
          <p class="indicator-glossary-intro">
            Indicators group SHAP token contributions into readable categories. They show which linguistic patterns influenced the model—not whether claims are factually true.
          </p>
          <div class="indicator-glossary-list">
            ${rows}
          </div>
        </div>
      </div>
    `;
  }

  function setToggleExpanded(btn, expanded) {
    if (!btn) return;
    btn.setAttribute("aria-expanded", expanded ? "true" : "false");
    const chevron = btn.querySelector(".indicator-glossary-chevron");
    if (chevron) {
      chevron.classList.toggle("is-expanded", expanded);
    }
  }

  function setPanelCollapsed(panel, collapsed) {
    if (!panel) return;
    panel.classList.toggle("is-collapsed", collapsed);
  }

  function setupIndicatorGlossaryToggle(root) {
    if (!root) return;

    const mainToggle = root.querySelector(".indicator-glossary-toggle");
    const mainPanel = root.querySelector(".indicator-glossary-panel");
    if (mainToggle && mainPanel) {
      mainToggle.addEventListener("click", () => {
        const expanded = mainToggle.getAttribute("aria-expanded") === "true";
        const next = !expanded;
        setPanelCollapsed(mainPanel, !next);
        setToggleExpanded(mainToggle, next);
      });
    }

    root.querySelectorAll(".indicator-glossary-item-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = btn.getAttribute("data-glossary-idx");
        const panel = root.querySelector(`#indicator-glossary-item-${idx}`);
        if (!panel) return;
        const expanded = btn.getAttribute("aria-expanded") === "true";
        const next = !expanded;
        setPanelCollapsed(panel, !next);
        setToggleExpanded(btn, next);
      });
    });
  }

  global.FakeShaIndicatorGlossary = {
    INDICATOR_GLOSSARY,
    buildIndicatorGlossaryHtml,
    setupIndicatorGlossaryToggle,
  };
})(typeof self !== "undefined" ? self : window);
