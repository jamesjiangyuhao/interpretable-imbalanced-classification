"""Generate public-safe portfolio charts for the early water loss modeling story."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


RANDOM_SEED = 42


def create_state_problem_chart(output_path: Path) -> None:
    """Create a synthetic state-level early water loss disproportionality chart."""
    states = ["AZ", "CA", "CO", "GA", "IL", "NV", "TX", "WA"]
    total_share = np.array([0.08, 0.14, 0.14, 0.27, 0.38, 0.41, 0.43, 0.92])
    early_share = np.array([0.035, 0.055, 0.064, 0.074, 0.095, 0.098, 0.127, 0.136])
    x = np.arange(len(states))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(x - width / 2, total_share, width, label="Total Data", color="#f59e0b")
    ax.bar(x + width / 2, early_share, width, label="Early Water Loss = 1", color="#1d4ed8")
    ax.set_title("Percentage of Early Water Loss by State", fontsize=16, weight="bold")
    ax.set_xlabel("State")
    ax.set_ylabel("Share of Records")
    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    ax.text(
        0.01,
        -0.16,
        "Synthetic state labels; chart illustrates disproportionate early water loss concentration.",
        transform=ax.transAxes,
        fontsize=10,
        color="#4b5563",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def create_shap_strength_chart(output_path: Path) -> None:
    """Create a SHAP-style beeswarm plot with generic feature labels."""
    rng = np.random.default_rng(RANDOM_SEED)
    features = [f"Feature {letter}" for letter in list("ABCDEFGHIJKLMNOPQRST")]
    spreads = np.array([3.0, 2.4, 2.0, 1.7, 1.45, 1.2, 1.05, 0.95, 0.85, 0.78, 0.70, 0.62, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.27, 0.24])
    signs = np.array([-1, 1, -1, 1, -1, 1, -1, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, 1])

    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    for idx, (feature, spread, sign) in enumerate(zip(features, spreads, signs)):
        n = 240
        feature_value = rng.uniform(0, 1, n)
        noise = rng.normal(0, spread * 0.18, n)
        if sign == 0:
            shap_value = rng.normal(0, spread * 0.28, n)
        else:
            shap_value = sign * (feature_value - 0.5) * spread * 2.1 + noise
        outlier_mask = rng.random(n) < max(0.02, spread / 90)
        shap_value[outlier_mask] += rng.normal(0, spread * 1.15, outlier_mask.sum())
        y = np.full(n, len(features) - 1 - idx) + rng.normal(0, 0.065, n)
        scatter = ax.scatter(shap_value, y, c=feature_value, cmap="cool", s=12, alpha=0.78, edgecolors="none")

    ax.axvline(0, color="#6b7280", linewidth=1.4)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features[::-1])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("Top Features by Predictive Strength", fontsize=16, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.025)
    cbar.set_label("Feature value")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def create_pd_overlap_chart(output_path: Path) -> None:
    """Create partial dependence and PD-score overlap charts showing over-blocking risk."""
    rng = np.random.default_rng(RANDOM_SEED)
    feature_x = np.linspace(0, 1, 100)
    pd_a = 0.45 - 3.5 * feature_x
    feature_b = np.arange(1, 14)
    pd_b = np.array([1.72, 1.58, 1.55, 1.55, 0.48, 0.55, -0.34, -0.58, -0.58, -2.18, -2.18, -2.18, -2.18])

    indicator_0 = np.concatenate(
        [
            rng.normal(-0.1, 0.33, 1800),
            rng.normal(1.0, 0.22, 1150),
            rng.normal(-2.2, 0.55, 850),
            rng.normal(2.6, 0.28, 420),
            rng.normal(-4.0, 0.55, 300),
        ]
    )
    indicator_1 = np.concatenate(
        [
            rng.normal(0.42, 0.58, 900),
            rng.normal(1.2, 0.45, 700),
            rng.normal(-1.0, 0.85, 500),
            rng.normal(2.55, 0.42, 260),
        ]
    )
    indicator_0 = np.clip(indicator_0, -8.0, 4.5)
    indicator_1 = np.clip(indicator_1, -8.0, 4.5)

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.65], hspace=0.42, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.plot(feature_x, pd_a, color="#3b82f6", linewidth=2.2)
    ax1.set_title("Feature A Partial Dependence", fontsize=12, weight="bold")
    ax1.set_ylabel("Partial dependence")
    ax1.set_xlabel("Feature A")
    ax1.grid(alpha=0.18)
    ax1.vlines([0, 1], ymin=-3.35, ymax=-3.0, color="#111827", linewidth=2)

    ax2.plot(feature_b, pd_b, color="#3b82f6", linewidth=2.2)
    ax2.set_title("Feature B Partial Dependence", fontsize=12, weight="bold")
    ax2.set_ylabel("Partial dependence")
    ax2.set_xlabel("Feature B bucket")
    ax2.grid(alpha=0.18)
    ax2.vlines([2, 9], ymin=-2.45, ymax=-2.1, color="#111827", linewidth=2)

    sns.kdeplot(indicator_0, ax=ax3, fill=True, color="#60a5fa", alpha=0.35, linewidth=2, label="early_water_loss_90_ind = 0")
    sns.kdeplot(indicator_1, ax=ax3, fill=True, color="#f59e0b", alpha=0.30, linewidth=2, label="early_water_loss_90_ind = 1")
    ax3.set_title("Distribution of PD Score by Early Water Loss Indicator", fontsize=15, weight="bold")
    ax3.set_xlabel("PD Score")
    ax3.set_ylabel("Density")
    ax3.legend(frameon=True, loc="upper right")
    ax3.grid(alpha=0.18)
    ax3.text(
        0.02,
        0.95,
        "Large overlap implies the current score would over-block many non-event records.",
        transform=ax3.transAxes,
        fontsize=10.5,
        color="#4b5563",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    create_state_problem_chart(output_dir / "early_water_state_problem.png")
    create_shap_strength_chart(output_dir / "early_water_shap_strength.png")
    create_pd_overlap_chart(output_dir / "early_water_pd_overlap.png")
    print(f"Wrote public portfolio charts to {output_dir}")


if __name__ == "__main__":
    main()
