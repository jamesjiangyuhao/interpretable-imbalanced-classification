"""Generate a fully synthetic imbalanced binary classification dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
ROW_COUNT = 55_000


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def build_synthetic_data(row_count: int = ROW_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create deterministic structured data with nonlinear signal and class imbalance."""
    rng = np.random.default_rng(seed)

    segment_code = rng.choice(["Segment A", "Segment B", "Segment C", "Segment D"], row_count, p=[0.34, 0.28, 0.23, 0.15])
    region_code = rng.choice(["Region 1", "Region 2", "Region 3", "Region 4"], row_count, p=[0.30, 0.26, 0.24, 0.20])
    categorical_feature_x = rng.choice(["X Low", "X Medium", "X High"], row_count, p=[0.48, 0.34, 0.18])
    categorical_feature_y = rng.choice(["Y Stable", "Y Watch", "Y New"], row_count, p=[0.60, 0.27, 0.13])
    categorical_feature_z = rng.choice(["Z1", "Z2", "Z3", "Z4"], row_count, p=[0.38, 0.30, 0.20, 0.12])

    cohort_flag = rng.binomial(1, 0.86, row_count)
    record_age_months = rng.gamma(shape=3.2, scale=7.5, size=row_count).clip(1, 72).round().astype(int)

    base = rng.normal(0, 1, row_count)
    numeric_feature_a = base + rng.normal(0, 0.7, row_count)
    numeric_feature_b = 0.65 * numeric_feature_a + rng.normal(0, 0.9, row_count)
    numeric_feature_c = rng.normal(0, 1.15, row_count)
    numeric_feature_d = rng.lognormal(mean=0.2, sigma=0.55, size=row_count)
    numeric_feature_e = rng.normal(0, 1, row_count)
    numeric_feature_f = rng.uniform(0, 1, row_count)
    numeric_feature_g = rng.normal(0, 1, row_count)
    numeric_feature_h = rng.normal(0, 1, row_count)
    numeric_feature_i = 0.45 * numeric_feature_c + rng.normal(0, 1, row_count)
    numeric_feature_j = rng.normal(0, 1, row_count)
    data_quality_score = np.clip(rng.normal(78, 12, row_count), 25, 100)

    linear_score = -2.65
    linear_score += 0.55 * numeric_feature_a
    linear_score += 0.45 * (numeric_feature_c > 0.85)
    linear_score += 0.85 * np.maximum(numeric_feature_f - 0.68, 0) * 3.0
    linear_score += 0.30 * np.sin(numeric_feature_d * 1.8)
    linear_score += 0.24 * np.where(categorical_feature_x == "X High", 1, 0)
    linear_score += 0.38 * np.where(segment_code == "Segment D", 1, 0)
    linear_score += 0.22 * np.where((segment_code == "Segment B") & (categorical_feature_y == "Y Watch"), 1, 0)
    linear_score += 0.32 * np.where((numeric_feature_a > 0.8) & (categorical_feature_x == "X High"), 1, 0)
    linear_score += 0.25 * np.where(region_code == "Region 3", 1, 0)
    linear_score += -0.012 * (data_quality_score - 75)
    linear_score += rng.normal(0, 0.35, row_count)

    probability = sigmoid(linear_score)
    binary_target = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "record_id": [f"REC{i:07d}" for i in range(row_count)],
            "cohort_flag": cohort_flag,
            "segment_code": segment_code,
            "region_code": region_code,
            "record_age_months": record_age_months,
            "numeric_feature_a": numeric_feature_a,
            "numeric_feature_b": numeric_feature_b,
            "numeric_feature_c": numeric_feature_c,
            "numeric_feature_d": numeric_feature_d,
            "numeric_feature_e": numeric_feature_e,
            "numeric_feature_f": numeric_feature_f,
            "numeric_feature_g": numeric_feature_g,
            "numeric_feature_h": numeric_feature_h,
            "numeric_feature_i": numeric_feature_i,
            "numeric_feature_j": numeric_feature_j,
            "categorical_feature_x": categorical_feature_x,
            "categorical_feature_y": categorical_feature_y,
            "categorical_feature_z": categorical_feature_z,
            "data_quality_score": data_quality_score,
            "binary_target": binary_target,
        }
    )

    missing_numeric = ["numeric_feature_b", "numeric_feature_e", "numeric_feature_i"]
    for col in missing_numeric:
        mask = rng.random(row_count) < 0.025
        df.loc[mask, col] = np.nan
    for col in ["categorical_feature_y", "categorical_feature_z"]:
        mask = rng.random(row_count) < 0.015
        df.loc[mask, col] = np.nan

    numeric_cols = [c for c in df.columns if c.startswith("numeric_feature_")] + ["data_quality_score"]
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


def main() -> None:
    """Write the synthetic dataset to the data directory."""
    output_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_binary_classification_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_synthetic_data()
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows to {output_path}")
    print(f"Positive class rate: {df['binary_target'].mean():.2%}")


if __name__ == "__main__":
    main()
