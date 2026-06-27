# FAKE-SHA

FAKE-SHA is a browser extension and backend analyzer developed as a Bachelor of Science in Computer Science (BSCS 4B) thesis project. The system helps users analyze selected text from web pages and classifies content as FAKE or REAL, returning a confidence score, a short summary, and indicator breakdowns. The project has progressed beyond the prototype stage and includes a trained model and a public dataset hosted on Hugging Face.

## Developers

* Guibao, Tricia Q.
* Luces, Dominique Isabelle C.
* Marbida, Jan Christian N.

## Project Overview

FAKE-SHA consists of two primary components:

### 1. Browser Extension (Frontend)

* Built using HTML, JavaScript, and Tailwind CSS
* Provides a popup interface for user interaction
* Extracts selected text from web pages
* Displays analysis results, confidence scores, and indicators
* Designed to be compatible with major browsers (Chromium-based and Firefox)

### 2. Backend API

* Built using Python (FastAPI)
* Performs analysis and returns verdict (FAKE/REAL), confidence, summary, and indicators
* Integrates with Supabase (PostgreSQL) for optional analysis record storage
* Integrates with a trained transformer model for inference (Hugging Face model)

## Trained Model & Dataset

The dataset used for training and the resulting trained model are publicly available on Hugging Face:

* Dataset: https://huggingface.co/datasets/niqueisa/fake-sha_taglish_dataset
* Trained model: https://huggingface.co/niqueisa/fake-sha_xlmr-roberta

Quick notes on using the Hugging Face model:

- You can perform inference using the Hugging Face Transformers library. Example (Python):

```python
from transformers import pipeline

model_id = "niqueisa/fake-sha_xlmr-roberta"
classifier = pipeline("text-classification", model=model_id, device=-1)  # device=0 for GPU

result = classifier("Sample text to analyze")
print(result)
```

- The dataset contains Taglish text samples and annotations used for training and evaluation.
- If you plan to run inference in the backend, consider caching the model locally or using the Hugging Face Inference API for production deployment.

## Features

* Text selection-based analysis
* Confidence scoring and indicator breakdown
* Phrase highlighting of relevant tokens
* Analysis history (stored in extension)
* Configurable backend endpoint
* Fallback mode when backend is unavailable
* Optional Supabase storage for analysis records

## Project Structure

```
FAKE-SHA/
├── extension/        # Browser extension (see subfolders below)
│   ├── popup/        # Popup UI
│   ├── settings/     # Settings page
│   ├── history/      # History page
│   ├── content/      # Content script(s)
│   ├── shared/       # Shared JS (e.g. backend API client)
│   └── assets/       # CSS, icons, logo (Tailwind output: assets/styles.css)
├── backend/          # FastAPI API + Supabase (see backend/README.md)
├── data/             # Train/validation/test CSVs (if present)
├── ui/               # Tailwind CSS source (input.css)
├── package.json      # Tailwind build scripts
└── README.md         # This file
```

## Development Setup

### Requirements

* Node.js (LTS recommended)
* npm
* Git
* Python 3.x (for backend)

### Extension (Frontend)

1. Install dependencies:

   ```bash
   npm install
   ```

2. Build Tailwind CSS:

   - Development (watch mode): `npm run dev:css`
   - Production: `npm run build:css`

   Output: `extension/assets/styles.css` (do not edit manually)

### Backend (API)

See **[backend/README.md](backend/README.md)** for:

* Virtual environment setup
* Dependencies (`pip install -r requirements.txt`)
* Optional Supabase configuration
* Running the server (`uvicorn main:app --reload --host 0.0.0.0 --port 8000`)
* Notes on integrating the Hugging Face model: set the model id to `niqueisa/fake-sha_xlmr-roberta` or provide a local model path.

## Loading the Extension (Chromium-Based Browsers)

1. Open `chrome://extensions`
2. Enable Developer Mode
3. Click "Load unpacked"
4. Select the `extension/` folder

## README Guide

| File | Purpose |
|------|---------|
| **README.md** (root) | Project overview, structure, extension setup, how to load the extension |
| **backend/README.md** | Backend-only: API setup, Supabase, testing, and deployment-focused analyzer config |

## Branching and Contribution Workflow

* All development must be done on feature branches.
* Direct pushes to `main` are not allowed.
* Pull Requests require review before merging.
* Contributors must not merge their own Pull Requests.
* Only designated maintainers may modify `manifest.json`.

## License

This project is developed for academic purposes as part of a BSCS thesis requirement.
