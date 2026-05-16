/**
 * Layout helpers for popup pages (loaded inside shell iframe or standalone).
 */
(function () {
  "use strict";

  const POPUP_WIDTH_PX = 360;
  const POPUP_HEIGHT_PX = 580;

  function lockPopupSize() {
    const html = document.documentElement;
    const body = document.body;
    if (!html || !body) return;

    const inShellFrame = window.parent && window.parent !== window;

    html.classList.add("fake-sha-popup-root");
    body.classList.add("popup-fixed");

    if (inShellFrame) {
      for (const el of [html, body]) {
        el.style.setProperty("width", "100%", "important");
        el.style.setProperty("min-width", "100%", "important");
        el.style.setProperty("max-width", "100%", "important");
        el.style.setProperty("height", "100%", "important");
        el.style.setProperty("min-height", "100%", "important");
        el.style.setProperty("max-height", "100%", "important");
        el.style.setProperty("overflow", "hidden", "important");
        el.style.setProperty("margin", "0", "important");
        el.style.setProperty("box-sizing", "border-box", "important");
      }
      return;
    }

    for (const el of [html, body]) {
      el.style.setProperty("width", `${POPUP_WIDTH_PX}px`, "important");
      el.style.setProperty("min-width", `${POPUP_WIDTH_PX}px`, "important");
      el.style.setProperty("max-width", `${POPUP_WIDTH_PX}px`, "important");
      el.style.setProperty("height", `${POPUP_HEIGHT_PX}px`, "important");
      el.style.setProperty("min-height", `${POPUP_HEIGHT_PX}px`, "important");
      el.style.setProperty("max-height", `${POPUP_HEIGHT_PX}px`, "important");
      el.style.setProperty("overflow", "hidden", "important");
      el.style.setProperty("box-sizing", "border-box", "important");
      el.style.setProperty("margin", "0", "important");
    }
  }

  lockPopupSize();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", lockPopupSize);
  }
})();
