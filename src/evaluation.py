"""Evaluation metrics and plots for imbalanced binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics_summary(y_true, y_pred, y_prob) -> dict:
    """Return core held-out performance metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def create_confusion_matrix_plot(y_true, y_pred, output_path=None):
    """Create and optionally save a confusion matrix plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
    return fig


def create_roc_curve_plot(y_true, y_prob, output_path=None):
    """Create and optionally save an ROC curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.4, label=f"ROC-AUC {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
    return fig


def create_precision_recall_curve_plot(y_true, y_prob, output_path=None):
    """Create and optionally save a precision-recall curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, color="#0f766e", linewidth=2.4, label=f"PR-AUC {average_precision_score(y_true, y_prob):.3f}")
    ax.axhline(np.mean(y_true), color="#9ca3af", linestyle="--", label="Base rate")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
    return fig


def threshold_sensitivity_table(y_true, y_prob) -> pd.DataFrame:
    """Show metric tradeoffs across operating thresholds."""
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "predicted_positive_rate": y_pred.mean(),
            }
        )
    return pd.DataFrame(rows)


def find_operating_threshold(y_true, y_prob, target_recall=None, target_precision=None) -> float:
    """Find a threshold meeting a target recall or precision constraint."""
    table = threshold_sensitivity_table(y_true, y_prob)
    if target_recall is not None:
        eligible = table[table["recall"] >= target_recall]
        if not eligible.empty:
            return float(eligible.sort_values("precision", ascending=False).iloc[0]["threshold"])
    if target_precision is not None:
        eligible = table[table["precision"] >= target_precision]
        if not eligible.empty:
            return float(eligible.sort_values("recall", ascending=False).iloc[0]["threshold"])
    return float(table.sort_values("f1", ascending=False).iloc[0]["threshold"])
