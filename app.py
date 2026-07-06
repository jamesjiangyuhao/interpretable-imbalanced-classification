"""Streamlit dashboard for interpretable imbalanced binary classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from data_processing import (
    apply_cohort_filters,
    identify_feature_types,
    load_data,
    preprocess_features,
    split_features_target,
    train_test_split_data,
    undersample_majority_class,
)
from evaluation import classification_metrics_summary, find_operating_threshold, threshold_sensitivity_table
from explainability import extract_top_features_from_importance, get_feature_names, permutation_importance_table, transformed_feature_frame
from generate_synthetic_data import main as generate_data
from lookup_scoring import build_partial_dependence_lookup, compare_model_vs_lookup, save_lookup_table, score_with_lookup
from modeling import predict_classes, predict_probabilities, train_final_model
from visualization import (
    plot_class_balance,
    plot_model_vs_lookup_score,
    plot_outcome_rate_by_segment,
    plot_score_distribution,
    plot_threshold_sensitivity,
    plot_top_driver_effects,
)


DATA_PATH = ROOT / "data" / "synthetic_binary_classification_data.csv"
LOOKUP_PATH = ROOT / "data" / "partial_dependence_lookup.csv"

st.set_page_config(page_title="Interpretable Imbalanced Classification", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        generate_data()
    return load_data(DATA_PATH)


@st.cache_resource(show_spinner=False)
def train_workflow(data: pd.DataFrame, min_age: int, use_undersampling: bool):
    filtered = apply_cohort_filters(data, min_record_age=min_age, eligible_only=True)
    X, y = split_features_target(filtered)
    numeric, categorical = identify_feature_types(X)
    preprocessor = preprocess_features(X, numeric, categorical)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    if use_undersampling:
        X_fit, y_fit = undersample_majority_class(X_train, y_train)
    else:
        X_fit, y_fit = X_train, y_train
    model = train_final_model(X_fit, y_fit, preprocessor)
    feature_names = get_feature_names(model.named_steps["preprocessor"])
    return filtered, X, y, X_test, y_test, model, feature_names


def pct(value: float) -> str:
    return f"{value:.1%}"


records = get_data()

st.title("Interpretable Imbalanced Classification Dashboard")
st.caption(
    "Synthetic portfolio project demonstrating gradient boosting, class-imbalance handling, SHAP explainability, "
    "partial dependence, and lightweight lookup-based scoring."
)

with st.sidebar:
    st.header("Controls")
    min_age = st.slider("Minimum record age", 1, 36, 12)
    selected_segments = st.multiselect("Segment filter", sorted(records["segment_code"].unique()), default=sorted(records["segment_code"].unique()))
    selected_regions = st.multiselect("Region filter", sorted(records["region_code"].unique()), default=sorted(records["region_code"].unique()))
    threshold = st.slider("Classification threshold", 0.10, 0.90, 0.35, 0.05)
    use_undersampling = st.checkbox("Use undersampling", value=True)
    top_n = st.slider("Top features for interpretation", 3, 8, 5)
    lookup_bins = st.slider("Lookup bins", 5, 15, 10)

filtered_records = records[records["segment_code"].isin(selected_segments) & records["region_code"].isin(selected_regions)].copy()
if filtered_records.empty:
    st.warning("No records match the selected filters.")
    st.stop()

filtered, X, y, X_test, y_test, model, feature_names = train_workflow(filtered_records, min_age, use_undersampling)
y_prob = predict_probabilities(model, X_test)
y_pred = (y_prob >= threshold).astype(int)
metrics = classification_metrics_summary(y_test, y_pred, y_prob)
threshold_table = threshold_sensitivity_table(y_test, y_prob)

top_raw = permutation_importance_table(model, X_test, y_test).head(top_n)["feature"].tolist()
lookup_df = build_partial_dependence_lookup(model, X, top_raw, bins=lookup_bins)
LOOKUP_PATH.parent.mkdir(exist_ok=True)
save_lookup_table(lookup_df, LOOKUP_PATH)
model_scores = predict_probabilities(model, X)
lookup_scores = score_with_lookup(X, lookup_df)
lookup_comparison = compare_model_vs_lookup(model_scores, lookup_scores)

st.subheader("Project Overview")
st.write(
    "This synthetic project demonstrates an interpretable machine learning workflow for an imbalanced binary "
    "classification problem. It moves from cohort selection and model training to explainability and simplified "
    "scoring logic that non-technical stakeholders can inspect."
)

st.subheader("Cohort and Class Balance")
cols = st.columns(4)
cols[0].metric("Total Records", f"{len(records):,}")
cols[1].metric("Eligible Records", f"{len(filtered):,}")
cols[2].metric("Positive Class Rate", pct(y.mean()))
cols[3].metric("Best F1 Threshold", f"{find_operating_threshold(y_test, y_prob):.2f}")
st.pyplot(plot_class_balance(filtered), clear_figure=True)

st.subheader("Model Performance")
metric_cols = st.columns(6)
for col, key in zip(metric_cols, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
    col.metric(key.upper().replace("_", "-"), f"{metrics[key]:.3f}")
st.plotly_chart(plot_score_distribution(y_prob, y_test), width="stretch")

st.subheader("Threshold Sensitivity")
st.plotly_chart(plot_threshold_sensitivity(threshold_table), width="stretch")
st.dataframe(threshold_table, width="stretch")

st.subheader("Feature Importance and Explainability")
importance = permutation_importance_table(model, X_test, y_test).head(15)
st.dataframe(importance, width="stretch")
st.write("Top driver candidates:", ", ".join(top_raw))

st.subheader("Partial Dependence and Lookup Scoring")
lookup_cols = st.columns([1, 1])
lookup_cols[0].plotly_chart(plot_top_driver_effects(lookup_df), width="stretch")
lookup_cols[1].pyplot(plot_model_vs_lookup_score(model_scores, lookup_scores), clear_figure=True)
st.metric("Model vs. Lookup Correlation", f"{lookup_comparison['correlation']:.3f}")
st.dataframe(lookup_df, width="stretch")

st.subheader("Business Interpretation")
insights = [
    f"The filtered cohort has a {pct(y.mean())} positive class rate, so accuracy alone can overstate model usefulness.",
    f"At the selected threshold, recall is {pct(metrics['recall'])} and precision is {pct(metrics['precision'])}, showing the operating tradeoff.",
    f"The strongest permutation driver in this run is `{top_raw[0]}`." if top_raw else "Top drivers are unavailable for this filtered cohort.",
    f"The simplified lookup score has {lookup_comparison['correlation']:.2f} correlation with full model probability.",
    "The lookup artifact is useful for stakeholder review because it turns model behavior into auditable feature buckets.",
]
for insight in insights:
    st.write(f"- {insight}")

st.subheader("Data Preview and Download")
score_output = X.copy()
score_output["model_probability"] = model_scores
score_output["lookup_score"] = lookup_scores
score_output["binary_target"] = y.values
tabs = st.tabs(["Filtered Data", "Model Scoring Output", "Lookup Table"])
tabs[0].dataframe(filtered.head(500), width="stretch")
tabs[1].dataframe(score_output.head(500), width="stretch")
tabs[2].dataframe(lookup_df, width="stretch")
st.download_button(
    "Download lookup table",
    lookup_df.to_csv(index=False).encode("utf-8"),
    file_name="partial_dependence_lookup.csv",
    mime="text/csv",
)
