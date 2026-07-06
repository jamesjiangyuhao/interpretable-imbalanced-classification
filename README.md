# Interpretable Gradient Boosting on an Imbalanced Binary Classification Problem

## Overview

This repository contains a fully synthetic machine learning workflow for an imbalanced binary classification problem. It demonstrates cohort filtering, class-imbalance handling, gradient boosting, held-out evaluation, explainability, partial dependence analysis, and a simplified lookup scoring artifact that stakeholders can inspect.

## Business Problem

Many operational classification problems involve rare outcomes where accuracy alone is misleading. A model can appear accurate by predicting the majority class while failing to identify the records that matter most. This project shows how to build, evaluate, and explain a model in that setting using only synthetic data and generic business logic.

## Why This Project Matters

The workflow connects model performance with decision support. It evaluates precision, recall, F1, ROC-AUC, and PR-AUC, then converts top model drivers into a lightweight lookup table that can be reviewed independently from the full model.

## Synthetic Data Design

The dataset contains 55,000 synthetic records with numeric features, categorical features, segment and region labels, record maturity, data quality scores, cohort eligibility, and an imbalanced binary target. The positive class rate is intentionally low, and the target is shaped by linear, nonlinear, correlated, noisy, and interacting features.

## Methodology

1. Generate synthetic structured data.
2. Apply cohort and maturity filters.
3. Split features and target.
4. Impute missing values and one-hot encode categorical variables.
5. Optionally undersample the majority class in training only.
6. Train a gradient boosting classifier.
7. Evaluate held-out performance across multiple metrics.
8. Explain top drivers with SHAP when available or robust fallback importance.
9. Build a partial-dependence-inspired lookup scoring table.
10. Compare full model probabilities against lookup scores.

## Class-Imbalance Handling

The project avoids using accuracy as the primary success metric. It supports training-set undersampling and evaluates precision-recall behavior across thresholds so the operating point can be selected based on the cost of false positives and false negatives.

## Model Design

The classifier uses scikit-learn gradient boosting with a reusable preprocessing pipeline. The code includes lightweight hyperparameter search utilities and a final training function designed to run locally without excessive compute.

## Evaluation Strategy

The project reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC curve, precision-recall curve, and threshold sensitivity. This mix is important because the positive class is intentionally rare.

## Explainability Layer

The project attempts SHAP-based global explanations when compatible. If SHAP is unavailable or incompatible in the local environment, it falls back gracefully to feature importance and permutation importance so the workflow remains runnable.

## Partial Dependence Lookup Scoring

Top numeric drivers are binned into quantile-based buckets. Each bucket receives a lookup score based on marginal model behavior. The result is a transparent scoring artifact with feature names, bucket boundaries, and lookup values.

## Dashboard Features

- Cohort and maturity filtering
- Segment and region filters
- Classification threshold controls
- Optional undersampling
- Class balance diagnostics
- Model performance metrics
- Threshold sensitivity analysis
- Feature importance and top-driver interpretation
- Lookup table creation and download
- Model probability versus lookup score comparison

## Repository Structure

```text
interpretable-imbalanced-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   ├── synthetic_binary_classification_data.csv
│   └── partial_dependence_lookup.csv
├── src/
│   ├── generate_synthetic_data.py
│   ├── data_processing.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── lookup_scoring.py
│   └── visualization.py
├── notebooks/
│   └── interpretable_classification_workflow.ipynb
└── outputs/
    ├── class_balance.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── feature_importance.png
    ├── shap_summary.png
    ├── partial_dependence_top_features.png
    └── model_vs_lookup_score.png
```

## How to Run

```bash
pip install -r requirements.txt
python src/generate_synthetic_data.py
streamlit run app.py
```

## Example Insights

- A high accuracy score can be misleading when the positive class is rare.
- Precision and recall move in opposite directions as the operating threshold changes.
- A small number of structured features explain a meaningful share of model behavior.
- Lookup scoring tracks the full model directionally while sacrificing some detail.
- A transparent lookup table can support stakeholder review before operational use.

## Skills Demonstrated

- Structured data preparation
- Cohort filtering
- Imbalanced binary classification
- Undersampling
- Gradient boosting
- Hyperparameter tuning
- Cross-validation
- Held-out model evaluation
- Precision, recall, F1, ROC-AUC, and PR-AUC analysis
- Confusion matrix interpretation
- SHAP-based explainability with fallback logic
- Partial dependence analysis
- Lookup-based scoring
- Stakeholder-oriented model interpretation
- Streamlit dashboard development

## Confidentiality Note

This project is built entirely with synthetic data and generic business logic. It does not contain any proprietary employer data, code, table names, target definitions, eligibility rules, business rules, screenshots, URLs, customer information, or confidential information.

## Future Enhancements

- Add calibration diagnostics.
- Add monotonicity constraints for selected features.
- Add cost-sensitive threshold optimization.
- Add unit tests for lookup scoring.
- Add model monitoring examples for drift and score stability.
