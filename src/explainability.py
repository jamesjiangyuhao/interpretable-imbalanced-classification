"""Explainability utilities using SHAP when available and robust fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
try:
    from sklearn.inspection import PartialDependenceDisplay
except ImportError:
    PartialDependenceDisplay = None
try:
    from sklearn.inspection import plot_partial_dependence
except ImportError:
    plot_partial_dependence = None


def get_feature_names(preprocessor) -> list[str]:
    """Return transformed feature names across newer and older scikit-learn versions."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        names = []
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "remainder" or transformer == "drop":
                continue
            if transformer_name == "numeric":
                names.extend(list(columns))
            else:
                encoder = transformer.named_steps.get("onehot")
                try:
                    encoded = encoder.get_feature_names_out(columns).tolist()
                except AttributeError:
                    encoded = encoder.get_feature_names(columns).tolist()
                names.extend(encoded)
        return names


def transformed_feature_frame(model, X: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """Return preprocessed model features as a named DataFrame."""
    sample = X.sample(min(len(X), max_rows), random_state=42) if max_rows else X
    preprocessor = model.named_steps["preprocessor"]
    arr = preprocessor.transform(sample)
    names = get_feature_names(preprocessor)
    return pd.DataFrame(arr, columns=names, index=sample.index)


def compute_shap_values(model, X_sample: pd.DataFrame, max_rows: int = 2000):
    """Compute SHAP values when compatible; otherwise return None."""
    try:
        import shap

        X_transformed = transformed_feature_frame(model, X_sample, max_rows=max_rows)
        classifier = model.named_steps["classifier"]
        explainer = shap.Explainer(classifier.predict_proba, X_transformed)
        values = explainer(X_transformed)
        return values[:, :, 1] if len(values.shape) == 3 else values
    except Exception:
        return None


def create_shap_summary_plot(shap_values, X_sample: pd.DataFrame, output_path=None):
    """Create a SHAP summary plot if SHAP values are available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 6))
    if shap_values is not None:
        try:
            import shap

            shap.summary_plot(shap_values, show=False, max_display=15)
            fig = plt.gcf()
        except Exception:
            plt.text(0.05, 0.55, "SHAP summary unavailable; see feature importance fallback.", fontsize=13)
            plt.axis("off")
    else:
        plt.text(0.05, 0.55, "SHAP summary unavailable; see feature importance fallback.", fontsize=13)
        plt.axis("off")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def create_feature_importance_plot(model, feature_names, output_path=None):
    """Create a feature importance plot using classifier or permutation-compatible importances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    else:
        values = np.zeros(len(feature_names))
    if not np.any(values):
        values = np.ones(len(feature_names)) / max(len(feature_names), 1)
    importance = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1], color="#2563eb")
    ax.set_title("Top Model Drivers")
    ax.set_xlabel("Relative Importance")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
    return fig


def permutation_importance_table(model, X, y, n_repeats: int = 5) -> pd.DataFrame:
    """Compute fallback permutation importances on raw feature columns."""
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, scoring="average_precision", n_jobs=1)
    return pd.DataFrame({"feature": X.columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)


def create_partial_dependence_plots(model, X, features, output_path=None):
    """Create partial dependence plots for top raw features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = [feature for feature in features if feature in X.columns and pd.api.types.is_numeric_dtype(X[feature])][:5]
    fig, ax = plt.subplots(figsize=(11, 6))
    if features:
        if PartialDependenceDisplay is not None and hasattr(PartialDependenceDisplay, "from_estimator"):
            PartialDependenceDisplay.from_estimator(model, X, features, ax=ax, kind="average")
            fig = plt.gcf()
        elif plot_partial_dependence is not None:
            plot_partial_dependence(model, X, features, ax=ax)
            fig = plt.gcf()
        else:
            ax.text(0.05, 0.55, "Partial dependence plotting is unavailable in this scikit-learn version.", fontsize=13)
            ax.axis("off")
    else:
        ax.text(0.05, 0.55, "No numeric features available for partial dependence.", fontsize=13)
        ax.axis("off")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
    return fig


def extract_top_features_from_importance(model, feature_names, top_n: int = 5) -> list[str]:
    """Extract top raw-ish feature names from transformed feature names."""
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    else:
        values = np.ones(len(feature_names)) / max(len(feature_names), 1)
    ordered = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False)
    raw_features = []
    for name in ordered["feature"]:
        raw = name
        for prefix in ["segment_code_", "region_code_", "categorical_feature_x_", "categorical_feature_y_", "categorical_feature_z_"]:
            if raw.startswith(prefix):
                raw = prefix[:-1]
        if raw.startswith("numeric_feature_") or raw == "data_quality_score":
            candidate = raw
        else:
            candidate = raw
        if candidate not in raw_features:
            raw_features.append(candidate)
        if len(raw_features) >= top_n:
            break
    return raw_features
