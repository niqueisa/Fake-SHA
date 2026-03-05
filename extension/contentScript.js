// Content script for FAKE-SHA
// Responsible for returning and clearing the current page's selected text.

function getCurrentSelectionText() {
  let text = "";

  try {
    const selection = window.getSelection ? window.getSelection() : null;
    if (selection && selection.rangeCount > 0) {
      text = selection.toString();
    }
  } catch (e) {
    // ignore selection errors
  }

  if (!text) {
    const active = document.activeElement;
    if (
      active &&
      (active.tagName === "TEXTAREA" ||
        (active.tagName === "INPUT" &&
          /^(text|search|url|tel|email|password)$/i.test(active.type)))
    ) {
      try {
        const start = active.selectionStart ?? 0;
        const end = active.selectionEnd ?? 0;
        if (end > start) {
          text = active.value.substring(start, end);
        }
      } catch (e) {
        // ignore
      }
    }
  }

  return (text || "").trim();
}

function clearCurrentSelection() {
  try {
    const selection = window.getSelection ? window.getSelection() : null;
    if (selection && selection.removeAllRanges) {
      selection.removeAllRanges();
    }
  } catch (e) {
    // ignore
  }

  const active = document.activeElement;
  if (
    active &&
    (active.tagName === "TEXTAREA" ||
      (active.tagName === "INPUT" &&
        /^(text|search|url|tel|email|password)$/i.test(active.type)))
  ) {
    try {
      const len = active.value.length;
      active.setSelectionRange(len, len);
    } catch (e) {
      // ignore
    }
  }
}

// Listen for messages from the popup
if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.onMessage) {
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (!request || !request.type) {
      return;
    }

    if (request.type === "fakeSha_getSelection") {
      try {
        const text = getCurrentSelectionText();
        sendResponse({ text });
      } catch (e) {
        sendResponse({ text: "", error: "selection_failed" });
      }
    } else if (request.type === "fakeSha_clearSelection") {
      try {
        clearCurrentSelection();
        sendResponse({ ok: true });
      } catch (e) {
        sendResponse({ ok: false });
      }
    }
    // Synchronous responses only; no need to return true.
  });
}

