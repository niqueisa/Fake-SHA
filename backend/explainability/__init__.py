"""
Explainability hooks (SHAP, attention, etc.).

Keep separate from `inference/` so SVM and RoBERTa can each have their own
explainers without coupling training code to the API layer.
"""
