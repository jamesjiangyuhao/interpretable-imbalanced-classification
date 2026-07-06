"""Data loading, cohort filtering, preprocessing, and train/test utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    """Load synthetic classification data."""
    return pd.read_csv(path)


def apply_cohort_filters(df: pd.DataFrame, min_record_age: int = 12, eligible_only: bool = True) -> pd.DataFrame:
    """Apply modeling eligibility and maturity filters."""
    output = df.copy()
    if eligible_only:
        output = output[output["cohort_flag"] == 1]
    output = output[output["record_age_months"] >= min_record_age]
    return output.reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target_col: str = "binary_target") -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix and binary target."""
    drop_cols = [target_col, "record_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].astype(int)
    return X, y


def identify_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns."""
    numeric_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in df.columns if c not in numeric_features]
    return numeric_features, categorical_features


def preprocess_features(X: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for mixed tabular data."""
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create held-out train/test splits."""
    stratify_values = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_values)


def undersample_majority_class(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Undersample the majority class in the training set only."""
    rng = np.random.default_rng(random_state)
    positives = y_train[y_train == 1].index.to_numpy()
    negatives = y_train[y_train == 0].index.to_numpy()
    if len(positives) == 0 or len(negatives) <= len(positives):
        return X_train, y_train
    sampled_negatives = rng.choice(negatives, size=min(len(negatives), len(positives) * 3), replace=False)
    keep = np.concatenate([positives, sampled_negatives])
    rng.shuffle(keep)
    return X_train.loc[keep].reset_index(drop=True), y_train.loc[keep].reset_index(drop=True)
