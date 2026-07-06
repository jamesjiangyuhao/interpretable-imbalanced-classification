"""Gradient boosting model training and prediction helpers."""

from __future__ import annotations

import numpy as np
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:  # Older scikit-learn fallback
    HistGradientBoostingClassifier = None
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def _gradient_boosting_classifier(**params):
    """Return the newest available scikit-learn gradient boosting classifier."""
    if HistGradientBoostingClassifier is not None:
        return HistGradientBoostingClassifier(**params)
    fallback = {
        "learning_rate": params.get("learning_rate", 0.06),
        "n_estimators": params.get("max_iter", 180),
        "max_depth": 3,
        "min_samples_leaf": params.get("min_samples_leaf", 40),
        "random_state": params.get("random_state", 42),
    }
    return GradientBoostingClassifier(**fallback)


def train_baseline_model(X_train, y_train, preprocessor) -> Pipeline:
    """Train a baseline gradient boosting classifier."""
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", _gradient_boosting_classifier(max_iter=120, learning_rate=0.06, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def tune_gradient_boosting_model(X_train, y_train, preprocessor, random_state: int = 42) -> RandomizedSearchCV:
    """Run a lightweight cross-validated hyperparameter search."""
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", _gradient_boosting_classifier(random_state=random_state)),
        ]
    )
    if HistGradientBoostingClassifier is not None:
        param_distributions = {
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "classifier__max_iter": [100, 160, 220],
            "classifier__max_leaf_nodes": [15, 31, 45],
            "classifier__l2_regularization": [0.0, 0.05, 0.1, 0.2],
            "classifier__min_samples_leaf": [20, 40, 80],
        }
    else:
        param_distributions = {
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "classifier__n_estimators": [100, 160, 220],
            "classifier__max_depth": [2, 3, 4],
            "classifier__min_samples_leaf": [20, 40, 80],
        }
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="average_precision",
        cv=3,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def train_final_model(X_train, y_train, preprocessor, best_params: dict | None = None) -> Pipeline:
    """Train the final model using optional tuned parameters."""
    params = {
        "learning_rate": 0.06,
        "max_iter": 180,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.05,
        "min_samples_leaf": 40,
        "random_state": 42,
    }
    if best_params:
        cleaned = {key.replace("classifier__", ""): value for key, value in best_params.items()}
        params.update(cleaned)
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", _gradient_boosting_classifier(**params)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def predict_probabilities(model: Pipeline, X) -> np.ndarray:
    """Predict positive-class probabilities."""
    return model.predict_proba(X)[:, 1]


def predict_classes(model: Pipeline, X, threshold: float = 0.50) -> np.ndarray:
    """Convert positive-class probabilities to binary labels."""
    return (predict_probabilities(model, X) >= threshold).astype(int)
