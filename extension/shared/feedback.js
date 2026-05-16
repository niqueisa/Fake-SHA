/**
 * Report-issue / feedback mailto helpers for FAKE-SHA extension surfaces.
 */
(function (global) {
  "use strict";

  const FEEDBACK_RECIPIENTS = [
    "dicl2022-4733-98895@bicol-u.edu.ph",
    "jcnm2022-8677-11726@bicol-u.edu.ph",
    "tqg2022-5560-42938@bicol-u.edu.ph",
  ];

  const DEFAULT_SUBJECT = "FAKE-SHA issue report";

  function buildFeedbackMailtoUrl(subject, bodyLines) {
    const to = FEEDBACK_RECIPIENTS.join(",");
    const subj = encodeURIComponent(subject || DEFAULT_SUBJECT);
    const body = encodeURIComponent((bodyLines || []).join("\n"));
    return `mailto:${to}?subject=${subj}&body=${body}`;
  }

  function openFeedbackMailto(subject, bodyLines) {
    const url = buildFeedbackMailtoUrl(subject, bodyLines);
    try {
      window.location.assign(url);
    } catch (e) {
      window.open(url, "_self");
    }
  }

  global.FakeShaFeedback = {
    FEEDBACK_RECIPIENTS,
    DEFAULT_SUBJECT,
    buildFeedbackMailtoUrl,
    openFeedbackMailto,
  };
})(typeof self !== "undefined" ? self : window);
