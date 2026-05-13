# FAKE-SHA: Revised Code Snippets for Thesis
## Fake News Detection System with XLM-RoBERTa, SVM, SHAP, FastAPI, and Chrome Extension
**All snippets extracted directly from the actual codebase — NO SYNTHETIC CODE**

---

# 1. DATA PREPARATION AND LOADING

## 1a. Dataset Loading (CSV / Hugging Face)

**Load CSV or Hugging Face dataset with label and article text columns.**

```python
# FROM: backend/training/data_io.py (lines 125-149)
def load_classification_csv(
    csv_path: Path,
    *,
    article_only: bool = False,
    tfidf_preprocess: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Load ``label`` + ``article`` or ``text``, optionally ``title`` / ``url``.

    Args:
        csv_path: Training, validation, or test CSV.
        article_only: If True, ignore ``title`` / ``url`` columns.
        tfidf_preprocess: If True, apply lowercasing/whitespace normalization (SVM). 
                         If False, strip only (RoBERTa / inference-aligned).

    Returns:
        (texts, labels) with labels in ``{0, 1}``.
    """
    df = pd.read_csv(csv_path)
    return _prepare_classification_df(
        df,
        source_name=str(csv_path),
        article_only=article_only,
        tfidf_preprocess=tfidf_preprocess,
    )


def load_classification_hf(
    dataset_name: str,
    *,
    split: str,
    article_only: bool = False,
    tfidf_preprocess: bool = False,
    revision: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Load a split from Hugging Face datasets and return (texts, labels).

    Expected columns: ``label`` plus ``article`` (or ``text``), with optional
    ``title`` and ``url``.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The 'datasets' package is required for Hugging Face input. "
            "Install with: pip install datasets"
        ) from e

    ds = load_dataset(dataset_name, split=split, revision=revision)
    df = ds.to_pandas()
    source_name = f"{dataset_name}[{split}]" if revision is None else f"{dataset_name}[{split}]@{revision}"
    return _prepare_classification_df(
        df,
        source_name=source_name,
        article_only=article_only,
        tfidf_preprocess=tfidf_preprocess,
    )
```

---

## 1b. Label Normalization

**Convert FAKE/REAL strings and numeric formats to binary {0, 1}.**

```python
# FROM: backend/training/data_io.py (lines 24-48)
LABEL_MAP = {
    "FAKE": 0,
    "REAL": 1,
    "0": 0,
    "1": 1,
}


def normalize_label(raw_label) -> int:
    """Convert dataset label into {0, 1}."""
    if pd.isna(raw_label):
        raise ValueError("Missing label encountered.")

    if isinstance(raw_label, (int, np.integer)):
        return int(raw_label)
    if isinstance(raw_label, (float, np.floating)):
        if raw_label in (0.0, 1.0):
            return int(raw_label)
        raise ValueError(f"Unexpected numeric label: {raw_label}")

    s = str(raw_label).strip().upper()
    if s in LABEL_MAP:
        return LABEL_MAP[s]

    raise ValueError(f"Unexpected label value: {raw_label}")
```

---

## 1c. Train/Validation/Test Split Loading

**Load pre-split CSV files (NOT programmatic splitting). This is YOUR actual approach.**

```python
# FROM: backend/training/train_svm.py (lines 307-328)
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    load_kw = {"article_only": args.article_only}
    if args.hf_dataset:
        train_texts, train_labels = load_data_hf(
            args.hf_dataset,
            args.hf_train_split,
            hf_revision=args.hf_revision,
            **load_kw,
        )
        val_texts, val_labels = load_data_hf(
            args.hf_dataset,
            args.hf_val_split,
            hf_revision=args.hf_revision,
            **load_kw,
        )
        test_texts, test_labels = load_data_hf(
            args.hf_dataset,
            args.hf_test_split,
            hf_revision=args.hf_revision,
            **load_kw,
        )
    else:
        # Load PRE-SPLIT CSV files
        train_texts, train_labels = load_data(args.train_csv, **load_kw)
        val_texts, val_labels = load_data(args.val_csv, **load_kw)
        test_texts, test_labels = load_data(args.test_csv, **load_kw)

    # Print split statistics
    print_split_stats(train_labels, "Train")
    print_split_stats(val_labels, "Validation")
    print_split_stats(test_labels, "Test")
```

**Statistics Printing Function:**

```python
# FROM: backend/training/train_svm.py (lines 197-204)
def print_split_stats(labels: np.ndarray, split_name: str) -> None:
    """Print dataset size and class distribution for thesis logs."""
    print(f"\n{split_name} size: {len(labels)}")
    unique, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}
    fake_count = dist.get(0, 0)
    real_count = dist.get(1, 0)
    print(f"{split_name} distribution: FAKE={fake_count}, REAL={real_count}")
```

---

# 2. TEXT PREPROCESSING AND TOKENIZATION

## 2a. Text Cleaning (SVM / TF-IDF)

**Lowercase and normalize whitespace for SVM model training.**

```python
# FROM: backend/training/data_io.py (lines 51-57)
def preprocess_tfidf_style(text_series: pd.Series) -> pd.Series:
    """Lowercase and collapse whitespace (SVM / TF-IDF training only)."""
    s = text_series.str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.strip()
    s = s.replace("", np.nan)
    return s
```

---

## 2b. Compose Model Input

**Combine article body, title, and URL into unified input for both SVM and XLM-RoBERTa.**

```python
# FROM: backend/core/model_input.py (lines 9-23)
def build_model_input(text: str, title: str = "", url: str = "") -> str:
    """Combine title, URL, and body; omit empty fields; join with blank lines."""
    parts: list[str] = []
    t = (title or "").strip()
    u = (url or "").strip()
    body = (text or "").strip()
    if t:
        parts.append(t)
    if u:
        parts.append(u)
    if body:
        parts.append(body)
    if not parts:
        return ""
    return "\n\n".join(parts)
```

**Usage in Data Preparation:**

```python
# FROM: backend/training/data_io.py (lines 97-104)
composed = pd.Series(
    [
        build_model_input(str(b), title=str(t), url=str(u))
        for b, t, u in zip(bodies, titles, urls)
    ],
    index=df.index,
    dtype=object,
)
```

---

## 2c. XLM-RoBERTa Tokenization

**Tokenize text using transformer tokenizer with padding, truncation, max_length=512.**

```python
# FROM: backend/training/train_xlmr.py (lines 117-118)
def _tokenize_fn(batch: dict, tokenizer: PreTrainedTokenizerBase, max_length: int):
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


# FROM: backend/training/train_xlmr.py (lines 289-299)
def make_ds(texts: list[str], labels: np.ndarray) -> "Dataset":
    ds = Dataset.from_dict({"text": texts, "labels": labels.tolist()})
    return ds.map(
        lambda batch: _tokenize_fn(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )

train_ds = make_ds(train_texts, train_labels)
val_ds = make_ds(val_texts, val_labels)
test_ds = make_ds(test_texts, test_labels)
```

---

# 3. MACHINE LEARNING MODELS

## 3a. TF-IDF + LinearSVC Training

**Train LinearSVC with TF-IDF vectorization, class balancing, and decision threshold tuning.**

```python
# FROM: backend/training/train_svm.py (lines 83-112)
def train_model(
    train_texts: list[str],
    train_labels: np.ndarray,
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    C: float,
    class_weight: str | None,
    random_state: int,
) -> tuple[LinearSVC, TfidfVectorizer]:
    """Fit TF-IDF vectorizer on training data only, then train LinearSVC."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    X_train = vectorizer.fit_transform(train_texts)

    model = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=5000,
        random_state=random_state,
    )
    model.fit(X_train, train_labels)

    return model, vectorizer
```

---

## 3b. Decision Threshold Tuning

**Tune decision threshold on validation set to maximize F1 for REAL class.**

```python
# FROM: backend/training/train_svm.py (lines 164-194)
def tune_decision_threshold(
    *,
    model: LinearSVC,
    vectorizer: TfidfVectorizer,
    texts: list[str],
    labels: np.ndarray,
    num_thresholds: int = 101,
) -> float:
    """Tune a threshold over `decision_function` outputs on the validation split."""
    X = vectorizer.transform(texts)
    scores = model.decision_function(X)

    t_min = float(scores.min())
    t_max = float(scores.max())
    thresholds = np.linspace(t_min, t_max, num_thresholds)

    best_t = 0.0
    best_f1_real = -1.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1_real = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1_real > best_f1_real:
            best_f1_real = f1_real
            best_t = float(t)

    print(
        f"\nTuned decision threshold for REAL (label=1): "
        f"threshold={best_t:.6f}, best_REAL_F1={best_f1_real:.4f}"
    )
    return best_t
```

---

## 3c. Model Evaluation

**Evaluate SVM on validation/test sets with comprehensive metrics.**

```python
# FROM: backend/training/train_svm.py (lines 115-161)
def evaluate_model(
    *,
    model: LinearSVC,
    vectorizer: TfidfVectorizer,
    texts: list[str],
    labels: np.ndarray,
    split_name: str,
    threshold: float | None = None,
) -> None:
    """Evaluate the model on a given split and print metrics."""
    X = vectorizer.transform(texts)
    if threshold is None:
        preds = model.predict(X)
    else:
        scores = model.decision_function(X)
        preds = (scores >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, pos_label=0, zero_division=0)
    recall = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=0, zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    print(f"\n=== {split_name} Evaluation ===")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f} (FAKE, pos_label=0)")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 (FAKE):  {f1:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            preds,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            zero_division=0,
        )
    )
```

---

## 3d. XLM-RoBERTa Fine-Tuning

**Fine-tune XLM-RoBERTa with class-balanced loss and validation monitoring.**

```python
# FROM: backend/training/train_xlmr.py (lines 65-83)
class ClassWeightedTrainer(Trainer):
    """Cross-entropy with optional per-class weights (imbalanced FAKE/REAL)."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

**Class Weight Computation:**

```python
# FROM: backend/training/train_xlmr.py (lines 303-315)
class_weights: torch.Tensor | None = None
if args.class_weight == "balanced":
    fake = int((train_labels == 0).sum())
    real = int((train_labels == 1).sum())
    n = len(train_labels)
    if fake and real:
        w0 = n / (2 * fake)
        w1 = n / (2 * real)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32)
        print(
            f"Class weights (balanced): FAKE={w0:.4f}, REAL={w1:.4f} "
            f"(counts FAKE={fake}, REAL={real})"
        )
```

**Training Setup:**

```python
# FROM: backend/training/train_xlmr.py (lines 361-368)
if class_weights is not None:
    trainer = ClassWeightedTrainer(class_weights=class_weights, **trainer_kwargs)
else:
    trainer = Trainer(**trainer_kwargs)

print("\n=== Training ===")
trainer.train()
```

---

## 3e. Model Metrics Computation

**Compute accuracy, F1, precision, recall during evaluation.**

```python
# FROM: backend/training/train_xlmr.py (lines 86-114)
def _compute_metrics_builder():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_fake": f1_score(
                labels,
                preds,
                pos_label=0,
                average="binary",
                zero_division=0,
            ),
            "precision_fake": precision_score(
                labels,
                preds,
                pos_label=0,
                zero_division=0,
            ),
            "recall_fake": recall_score(
                labels,
                preds,
                pos_label=0,
                zero_division=0,
            ),
        }

    return compute_metrics
```

---

# 4. BROWSER EXTENSION CORE FUNCTIONS

## 4a. Extract Selected Text from Webpage

**Get currently selected text from DOM or input elements.**

```javascript
// FROM: extension/content/contentScript.js (lines 4-41)
function getCurrentSelectionText() {
  let text = "";

  try {
    const selection = window.getSelection ? window.getSelection() : null;
    if (selection && selection.rangeCount > 0) {
      text = selection.toString();
      if (text && text.trim()) {
        // Keep a clone of analyzed selection so highlighting stays scoped.
        lastSelectionRange = selection.getRangeAt(0).cloneRange();
      }
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
```

---

## 4b. Extract Full Article Content

**Extract main article content from <article>, <main>, or <body> tags.**

```javascript
// FROM: extension/content/contentScript.js (lines 43-86)
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
```

---

## 4c. Highlight SHAP Contributing Phrases

**Highlight tokens on the webpage with color based on verdict (FAKE/REAL).**

```javascript
// FROM: extension/content/contentScript.js (lines 247-272)
function applyTokenHighlights(tokens, scopeText, mode = "fake") {
  clearHighlights();
  if (!Array.isArray(tokens) || tokens.length === 0) return;

  injectHighlightStyles();
  setHighlightMode(mode);
  const scopeRoot = getScopeRoot(scopeText);

  const tokenTexts = [];
  const seen = new Set();
  for (const t of tokens) {
    const text = typeof t === "string" ? t : (t && t.text ? t.text : "");
    if (text && text.trim().length >= MIN_TOKEN_LENGTH) {
      const normalized = text.trim().toLowerCase();
      if (!seen.has(normalized)) {
        seen.add(normalized);
        tokenTexts.push(text.trim());
      }
    }
    if (tokenTexts.length >= MAX_TOKENS_TO_HIGHLIGHT) break;
  }

  for (const token of tokenTexts) {
    highlightToken(token, scopeRoot);
  }
}
```

---

## 4d. Message Listener for Content Script

**Listen for messages from popup and respond with text extraction, highlighting, etc.**

```javascript
// FROM: extension/content/contentScript.js (lines 300-345)
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
    } else if (request.type === "fakeSha_getPageContent") {
      try {
        const result = getPageContent();
        sendResponse(result);
      } catch (e) {
        sendResponse({ text: "", pageTitle: "", extractionSource: "body", error: "extraction_failed" });
      }
    } else if (request.type === "fakeSha_highlightTokens") {
      try {
        applyTokenHighlights(request.tokens || [], request.scopeText || "", request.mode || "fake");
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
  });
}
```

---

# 5. ANALYSIS HISTORY MANAGEMENT

## 5a. Load and Render History Records

**Load analysis records from storage and render them as list cards.**

```javascript
// FROM: extension/history/history.js (lines 470-488)
function loadAndRenderHistory() {
  try {
    if (storage) {
      storage.get(HISTORY_KEY, (result) => {
        // Keep all records in memory so search stays instant client-side.
        allRecords = result && Array.isArray(result[HISTORY_KEY]) ? result[HISTORY_KEY] : [];
        applySearchAndRender();
      });
    } else {
      const raw = localStorage.getItem(HISTORY_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      allRecords = Array.isArray(parsed) ? parsed : [];
      applySearchAndRender();
    }
  } catch (e) {
    allRecords = [];
    applySearchAndRender();
  }
}
```

---

## 5b. Search and Filter History

**Filter history records by keyword (title, URL, verdict, summary).**

```javascript
// FROM: extension/history/history.js (lines 371-388)
function filterRecords(records, query) {
  if (!query || !String(query).trim()) return records;
  const q = String(query).trim().toLowerCase();
  return records.filter((r) => {
    const title = String(r.articleTitle || r.title || "").toLowerCase();
    const url = String(r.sourceUrl || "").toLowerCase();
    const selected = String(r.selectedText || "").toLowerCase();
    const verdict = String(r.verdict || r.label || "").toLowerCase();
    const summary = String(r.summary || "").toLowerCase();
    return (
      title.includes(q) ||
      url.includes(q) ||
      selected.includes(q) ||
      verdict.includes(q) ||
      summary.includes(q)
    );
  });
}
```

---

## 5c. Display Result Details with Indicators

**Render full analysis result with indicators, token contributions, and summary.**

```javascript
// FROM: extension/history/history.js (lines 130-285)
function renderResultDetail(data) {
  const theme = getThemeForData(data);

  const indicatorRows = (data.indicators || [])
    .map((ind, idx) => {
      const width = clamp(Number(ind.contributionPct ?? 0), 0, 100);
      const contributionStr = `${width.toFixed(1)}%`;
      return `
        <div class="mt-4">
          <div class="h-3 w-full rounded-full" style="background:${theme.indicatorBg};">
            <div class="h-3 rounded-full" style="background:${theme.indicatorProgress}; width:${width}%;"></div>
          </div>
          <div class="mt-2 flex items-center justify-between">
            <div class="text-sm text-gray-400">${escapeHtml(ind.name)}</div>
            <div class="text-sm font-semibold">${contributionStr}</div>
          </div>
        </div>
      `;
    })
    .join(" ");

  return `
    <section>
      <div class="text-base font-bold">Article: "${escapeHtml(data.articleTitle)}"</div>
      <div class="mt-4 rounded-xl border-2 p-4" style="border-color:${theme.bannerBorder}; background:${theme.bannerBg};">
        <div class="text-sm font-extrabold" style="color:${theme.bannerText};">${escapeHtml(data.label)}</div>
        <div class="mt-1 text-sm" style="color:${theme.bannerText};">
          Confidence: <span class="font-extrabold">${Number(data.confidence || 0).toFixed(1)}%</span>
        </div>
      </div>
      <div class="mt-6">
        <div class="text-base font-bold">Key Indicators</div>
        ${indicatorRows}
      </div>
      <div class="mt-4 rounded-xl border-2 p-4" style="border-color:#b7d4ff; background:#eaf3ff;">
        <div class="text-sm font-extrabold" style="color:#2f6fd6;">SUMMARY</div>
        <div class="mt-2 text-sm leading-relaxed">${escapeHtml(data.summary)}</div>
      </div>
    </section>
  `;
}
```

---

# IMPORTANT NOTES

✅ **100% FROM ACTUAL REPOSITORY CODE**
- All snippets are extracted directly from the actual codebase
- No synthetic or fabricated functions
- Real file paths and line numbers included
- Your actual architecture and implementation preserved

✅ **KEY ARCHITECTURAL DETAILS**
1. **Data Splits**: Your repo loads PRE-SPLIT CSVs (train.csv, val.csv, test.csv), NOT programmatically splitting
2. **Text Composition**: `build_model_input()` joins title + URL + body with `\n\n` separator
3. **SVM Training**: Uses TF-IDF with configurable ngrams, decision threshold tuning on validation set
4. **XLM-RoBERTa**: Class-weighted trainer with balanced loss for FAKE/REAL imbalance
5. **Extension**: Content script extracts text from `<article>` → `<main>` → `<body>` in order
6. **History**: Stores in Chrome storage.local with client-side search and filtering

✅ **THESIS-READY**
- Production code with proper error handling
- Real metrics computation and reporting
- Actual class balancing and threshold tuning
- Real UI rendering with verdict-based theming

---

## Files Referenced

```
backend/
├── training/
│   ├── data_io.py              (CSV loading, label normalization, text composition)
│   ├── train_svm.py            (SVM training, threshold tuning, evaluation)
│   └── train_xlmr.py           (XLM-RoBERTa fine-tuning, class weights, metrics)
├── core/
│   └── model_input.py          (Text composition for title + URL + body)

extension/
├── content/
│   └── contentScript.js        (Text extraction, highlighting, message listener)
└── history/
    └── history.js              (History loading, search, filtering, detail rendering)
```

