# FAKE-SHA

FAKE-SHA is a browser extension developed as a Bachelor of Science in Computer Science (BSCS 4B) thesis project. The system assists users in analyzing selected text from web pages and identifying the likelihood of misinformation using machine learning and natural language processing techniques.

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
* Designed for future ML/NLP model integration (e.g., RoBERTa, SHAP)

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
├── data/             # Train/validation/test CSVs
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

## Loading the Extension (Chromium-Based Browsers)

1. Open `chrome://extensions`
2. Enable Developer Mode
3. Click "Load unpacked"
4. Select the `extension/` folder

## README Guide

| File | Purpose |
|------|---------|
| **README.md** (root) | Project overview, structure, extension setup, how to load the extension |
| **backend/README.md** | Backend-only: API setup, Supabase, testing, replacing the mock analyzer |

## Branching and Contribution Workflow

* All development must be done on feature branches.
* Direct pushes to `main` are not allowed.
* Pull Requests require review before merging.
* Contributors must not merge their own Pull Requests.
* Only designated maintainers may modify `manifest.json`.

## License

This project is developed for academic purposes as part of a BSCS thesis requirement.
