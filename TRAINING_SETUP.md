# Fake-SHA Training Setup (Windows + PowerShell)

Use this guide before training SVM or RoBERTa.

## 1) Clone and open the project

```powershell
git clone <your-repo-url>
cd Fake-SHA
```

## 2) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked in PowerShell, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

## 4) (Optional but recommended) Login to Hugging Face

Public datasets work without login, but login gives better rate limits and smoother downloads.

```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

Paste your Hugging Face token when prompted.

## 5) Quick dependency check

```powershell
python -c "import torch, transformers, datasets, sklearn, pandas, numpy, joblib; print('OK')"
```

## 6) Train models from Hugging Face dataset

Dataset ID used by this project:

`niqueisa/fake-sha-dataset`

### SVM

```powershell
python -m backend.training.train_svm --hf-dataset "niqueisa/fake-sha-dataset"
```

### RoBERTa

```powershell
python -m backend.training.train_roberta --hf-dataset "niqueisa/fake-sha-dataset"
```

## 7) Optional reproducibility flag

Pin dataset version for consistent runs across teammates:

```powershell
python -m backend.training.train_roberta --hf-dataset "niqueisa/fake-sha-dataset" --hf-revision "<commit_hash>"
```

Use the same `--hf-revision` value for SVM too.

## 8) Daily workflow (each new terminal)

```powershell
cd <path-to-Fake-SHA>
.\.venv\Scripts\Activate.ps1
```

Then run training commands.

## Notes

- Splits detected from dataset hub: `train`, `validation`, `test`.
- Required columns detected: `label`, `title`, `article`, `url`.
- Trained artifacts are written under `backend/artifacts/`.
