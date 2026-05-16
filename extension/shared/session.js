/**
 * Per-tab popup session persistence (popup, background service worker).
 */
(function (global) {
  "use strict";

  const POPUP_SESSIONS_KEY = "fakeShaTabSessions";
  const MAX_TAB_SESSIONS = 30;

  function getStorage() {
    try {
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        return chrome.storage.local;
      }
      if (typeof browser !== "undefined" && browser.storage && browser.storage.local) {
        return browser.storage.local;
      }
    } catch (e) {
      // ignore
    }
    return null;
  }

  const storage = getStorage();

  function pruneTabSessions(sessions) {
    const entries = Object.entries(sessions || {});
    if (entries.length <= MAX_TAB_SESSIONS) return sessions;
    entries.sort((a, b) => Number(b[1]?.updatedAt || 0) - Number(a[1]?.updatedAt || 0));
    return Object.fromEntries(entries.slice(0, MAX_TAB_SESSIONS));
  }

  function readAllTabSessions() {
    return new Promise((resolve) => {
      try {
        if (storage) {
          storage.get(POPUP_SESSIONS_KEY, (result) => {
            const sessions = result && result[POPUP_SESSIONS_KEY];
            resolve(sessions && typeof sessions === "object" ? sessions : {});
          });
          return;
        }
        const raw = localStorage.getItem(POPUP_SESSIONS_KEY);
        resolve(raw ? JSON.parse(raw) : {});
      } catch (e) {
        resolve({});
      }
    });
  }

  function writeAllTabSessions(sessions) {
    return new Promise((resolve) => {
      try {
        const pruned = pruneTabSessions(sessions);
        if (storage) {
          storage.set({ [POPUP_SESSIONS_KEY]: pruned }, () => resolve());
          return;
        }
        localStorage.setItem(POPUP_SESSIONS_KEY, JSON.stringify(pruned));
        resolve();
      } catch (e) {
        resolve();
      }
    });
  }

  async function getForTab(tabId) {
    if (tabId == null) return null;
    const sessions = await readAllTabSessions();
    return sessions[String(tabId)] || null;
  }

  async function setForTab(tabId, sessionData) {
    if (tabId == null) return;
    const key = String(tabId);
    const sessions = await readAllTabSessions();
    sessions[key] = {
      ...sessionData,
      tabId,
      updatedAt: Date.now(),
    };
    await writeAllTabSessions(sessions);
  }

  async function clearForTab(tabId) {
    if (tabId == null) return;
    const key = String(tabId);
    const sessions = await readAllTabSessions();
    if (!sessions[key]) return;
    delete sessions[key];
    await writeAllTabSessions(sessions);
  }

  async function markInterruptedLoadingSessions() {
    const sessions = await readAllTabSessions();
    let changed = false;
    for (const [key, session] of Object.entries(sessions)) {
      if (session && session.view === "loading") {
        sessions[key] = {
          ...session,
          view: "error",
          errorMessage: "Analysis was interrupted. Please analyze again.",
          updatedAt: Date.now(),
        };
        changed = true;
      }
    }
    if (changed) {
      await writeAllTabSessions(sessions);
    }
  }

  global.FakeShaSession = {
    POPUP_SESSIONS_KEY,
    readAllTabSessions,
    getForTab,
    setForTab,
    clearForTab,
    markInterruptedLoadingSessions,
  };
})(typeof self !== "undefined" ? self : window);
