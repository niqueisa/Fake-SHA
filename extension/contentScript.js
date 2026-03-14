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

/**
 * Extract main article/page text from the document.
 * Strategy: article → main → body (first non-empty wins).
 * Returns { text, extractionSource }.
 */
function getPageContent() {
  let text = "";
  let source = "body";

  try {
    // 1. Try <article> (common for news/blog posts)
    const article = document.querySelector("article");
    if (article && article.innerText && article.innerText.trim()) {
      text = article.innerText.trim();
      source = "article";
    }

    // 2. If not found or empty, try <main>
    if (!text) {
      const main = document.querySelector("main");
      if (main && main.innerText && main.innerText.trim()) {
        text = main.innerText.trim();
        source = "main";
      }
    }

    // 3. Fallback to document body
    if (!text) {
      const body = document.body;
      if (body && body.innerText && body.innerText.trim()) {
        text = body.innerText.trim();
        source = "body";
      }
    }
  } catch (e) {
    // ignore extraction errors
  }

  return {
    text: (text || "").trim(),
    pageTitle: document.title || "",
    extractionSource: source,
  };
}

// -----------------------------------------------------------------------------
// Token highlighting (for "Highlight Contributing Phrases" setting)
// -----------------------------------------------------------------------------

const HIGHLIGHT_CLASS = "fake-sha-highlight";
const MAX_TOKENS_TO_HIGHLIGHT = 10;
const MIN_TOKEN_LENGTH = 2;

/**
 * Inject CSS for highlight styling. Idempotent.
 */
function injectHighlightStyles() {
  if (document.getElementById("fake-sha-highlight-styles")) return;
  const style = document.createElement("style");
  style.id = "fake-sha-highlight-styles";
  style.textContent = `
    .fake-sha-highlight {
      background-color: #fef08a;
      padding: 0 1px;
      border-radius: 2px;
    }
  `;
  (document.head || document.documentElement).appendChild(style);
}

/**
 * Check if a node should be skipped (inside script, style, etc.).
 */
function shouldSkipNode(node) {
  if (!node || !node.parentElement) return true;
  const parent = node.parentElement;
  const tag = parent.tagName && parent.tagName.toUpperCase();
  if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") return true;
  if (parent.closest && parent.closest("." + HIGHLIGHT_CLASS)) return true;
  return false;
}

/**
 * Highlight one occurrence of a token in a text node. Returns true if a match was wrapped.
 */
function highlightOneMatch(textNode, token) {
  const text = textNode.textContent;
  const tokenLower = token.toLowerCase();
  const idx = text.toLowerCase().indexOf(tokenLower);
  if (idx === -1) return false;

  const before = text.slice(0, idx);
  const match = text.slice(idx, idx + token.length);
  const after = text.slice(idx + token.length);

  const parent = textNode.parentNode;
  if (!parent) return false;

  if (before) parent.insertBefore(document.createTextNode(before), textNode);
  const mark = document.createElement("mark");
  mark.className = HIGHLIGHT_CLASS;
  mark.textContent = match;
  parent.insertBefore(mark, textNode);
  if (after) parent.insertBefore(document.createTextNode(after), textNode);
  parent.removeChild(textNode);
  return true;
}

/**
 * Highlight all occurrences of a token in the document.
 */
function highlightToken(token) {
  if (!token || typeof token !== "string") return;
  const t = token.trim();
  if (t.length < MIN_TOKEN_LENGTH) return;

  let found;
  do {
    found = false;
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (shouldSkipNode(node)) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        },
      }
    );
    while (walker.nextNode()) {
      if (highlightOneMatch(walker.currentNode, t)) {
        found = true;
        break;
      }
    }
  } while (found);
}

/**
 * Clear all FAKE-SHA highlights from the page.
 */
function clearHighlights() {
  document.querySelectorAll("." + HIGHLIGHT_CLASS).forEach((mark) => {
    const text = document.createTextNode(mark.textContent);
    if (mark.parentNode) {
      mark.parentNode.replaceChild(text, mark);
    }
  });
}

/**
 * Apply highlights for the given token texts.
 * Clears previous highlights first, then highlights up to MAX_TOKENS_TO_HIGHLIGHT.
 */
function applyTokenHighlights(tokens) {
  clearHighlights();
  if (!Array.isArray(tokens) || tokens.length === 0) return;

  injectHighlightStyles();

  const tokenTexts = [];
  for (const t of tokens) {
    const text = typeof t === "string" ? t : (t && t.text ? t.text : "");
    if (text && text.trim().length >= MIN_TOKEN_LENGTH) {
      tokenTexts.push(text.trim());
    }
    if (tokenTexts.length >= MAX_TOKENS_TO_HIGHLIGHT) break;
  }

  for (const token of tokenTexts) {
    highlightToken(token);
  }
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
    } else if (request.type === "fakeSha_getPageContent") {
      try {
        const result = getPageContent();
        sendResponse(result);
      } catch (e) {
        sendResponse({ text: "", pageTitle: "", extractionSource: "body", error: "extraction_failed" });
      }
    } else if (request.type === "fakeSha_highlightTokens") {
      try {
        applyTokenHighlights(request.tokens || []);
        sendResponse({ ok: true });
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    } else if (request.type === "fakeSha_clearHighlights") {
      try {
        clearHighlights();
        sendResponse({ ok: true });
      } catch (e) {
        sendResponse({ ok: false });
      }
    }
    // Synchronous responses only; no need to return true.
  });
}

