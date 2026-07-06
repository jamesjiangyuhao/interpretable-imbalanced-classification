"""Visualization helpers for the interpretable classification workflow."""

from __future__ import annotations

import pandas as pd


def plot_class_balance(df: pd.DataFrame, output_path=None):
    """Plot binary target class balance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df["binary_target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(["Negative", "Positive"], counts.values, color=["#2563eb", "#b91c1c"])
    ax.set_title("Class Balance")
    ax.set_ylabel("Records")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
    return fig


def plot_outcome_rate_by_segment(df: pd.DataFrame, segment_col):
    """Return a segment-level outcome rate table."""
    return df.groupby(segment_col)["binary_target"].agg(["count", "mean"]).sort_values("mean", ascending=False)


def plot_score_distribution(y_prob, y_true):
    """Create a Plotly score distribution chart."""
    import plotly.express as px

    frame = pd.DataFrame({"model_probability": y_prob, "observed_class": y_true.astype(str)})
    return px.histogram(frame, x="model_probability", color="observed_class", nbins=40, barmode="overlay", title="Model Score Distribution")


def plot_model_vs_lookup_score(model_scores, lookup_scores, output_path=None):
    """Plot full-model probability against lookup score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(lookup_scores, model_scores, s=10, alpha=0.35, color="#2563eb")
    ax.set_title("Full Model Probability vs. Lookup Score")
    ax.set_xlabel("Lookup Score")
    ax.set_ylabel("Model Probability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
    return fig


def plot_threshold_sensitivity(threshold_df: pd.DataFrame):
    """Create a Plotly threshold sensitivity chart."""
    import plotly.express as px

    melted = threshold_df.melt(id_vars="threshold", value_vars=["precision", "recall", "f1", "predicted_positive_rate"])
    return px.line(melted, x="threshold", y="value", color="variable", markers=True, title="Threshold Sensitivity")


def plot_top_driver_effects(lookup_df: pd.DataFrame):
    """Create a Plotly chart of lookup effects by feature bucket."""
    import plotly.express as px

    return px.line(lookup_df, x="bucket_id", y="lookup_score", color="feature_name", markers=True, title="Lookup Score by Feature Bucket")
