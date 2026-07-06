"""Partial-dependence-inspired lookup scoring artifact."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_partial_dependence_lookup(model, X: pd.DataFrame, top_features: list[str], bins: int = 10) -> pd.DataFrame:
    """Build a transparent lookup table from binned feature-level model behavior."""
    rows = []
    numeric_features = [f for f in top_features if f in X.columns and pd.api.types.is_numeric_dtype(X[f])]
    baseline = model.predict_proba(X.sample(min(len(X), 3000), random_state=42))[:, 1].mean()
    for feature in numeric_features:
        values = X[feature].dropna()
        if values.nunique() < 3:
            continue
        quantiles = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(quantiles) < 3:
            continue
        bucketed = pd.cut(X[feature], bins=quantiles, include_lowest=True, duplicates="drop")
        for bucket_id, interval in enumerate(bucketed.cat.categories, start=1):
            X_temp = X.sample(min(len(X), 2500), random_state=42).copy()
            midpoint = (interval.left + interval.right) / 2
            X_temp[feature] = midpoint
            avg_prob = model.predict_proba(X_temp)[:, 1].mean()
            rows.append(
                {
                    "feature_name": feature,
                    "bucket_id": bucket_id,
                    "bucket_min": float(interval.left),
                    "bucket_max": float(interval.right),
                    "lookup_score": float(avg_prob - baseline),
                }
            )
    return pd.DataFrame(rows)


def save_lookup_table(lookup_df: pd.DataFrame, path: str) -> None:
    """Save lookup table to CSV."""
    lookup_df.to_csv(path, index=False)


def load_lookup_table(path: str) -> pd.DataFrame:
    """Load lookup table from CSV."""
    return pd.read_csv(path)


def score_with_lookup(df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.Series:
    """Score each record by averaging matched feature bucket scores."""
    scores = pd.Series(0.0, index=df.index)
    counts = pd.Series(0, index=df.index)
    for feature, feature_lookup in lookup_df.groupby("feature_name"):
        if feature not in df.columns:
            continue
        for row in feature_lookup.itertuples(index=False):
            mask = df[feature].between(row.bucket_min, row.bucket_max, inclusive="both")
            scores.loc[mask] += row.lookup_score
            counts.loc[mask] += 1
    counts = counts.replace(0, np.nan)
    return (scores / counts).fillna(0.0)


def compare_model_vs_lookup(model_scores, lookup_scores) -> dict:
    """Compare full model probabilities against lookup-based scores."""
    corr = pd.Series(model_scores).corr(pd.Series(lookup_scores))
    return {"correlation": float(corr) if pd.notna(corr) else 0.0}
