"""
Exploratory Data Analysis for Chronic Kidney Disease Dataset
=============================================================
Generates and saves visualisation plots for:
  - Target distribution
  - Missing value heatmap
  - Numerical feature distributions
  - Correlation heatmap
  - Box plots by class
  - Categorical feature distributions
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pandas as pd
import numpy as np

from data_preprocessing import (
    load_data, clean_data, NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL
)

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Style ----------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"ckd": "#e74c3c", "notckd": "#2ecc71"}


def save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {name}")


def plot_target_distribution(df: pd.DataFrame):
    """Bar chart of CKD vs Not-CKD counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COL].value_counts()
    labels = ["CKD (1)" if v == 1 else "Not CKD (0)" for v in counts.index]
    bars = ax.bar(labels, counts.values, color=["#e74c3c", "#2ecc71"], edgecolor="black")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Target Distribution: CKD vs Not CKD", fontweight="bold")
    ax.set_ylabel("Count")
    save(fig, "01_target_distribution.png")


def plot_missing_heatmap(df: pd.DataFrame):
    """Missing-value matrix plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    msno.matrix(df, ax=ax, sparkline=False, fontsize=9, color=(0.27, 0.52, 0.96))
    ax.set_title("Missing Value Matrix", fontweight="bold", fontsize=14)
    save(fig, "02_missing_value_matrix.png")


def plot_missing_bar(df: pd.DataFrame):
    """Bar chart of missing counts per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("  ℹ️  No missing values found — skipping missing bar plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind="bar", ax=ax, color="#3498db", edgecolor="black")
    ax.set_title("Missing Values Per Feature", fontweight="bold")
    ax.set_ylabel("Count")
    for i, (col, val) in enumerate(missing.items()):
        ax.text(i, val + 1, str(val), ha="center", fontsize=9)
    save(fig, "03_missing_bar.png")


def plot_numerical_distributions(df: pd.DataFrame):
    """Histograms + KDE for all numerical features."""
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    n = len(num_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        for label, color in [("CKD", "#e74c3c"), ("Not CKD", "#2ecc71")]:
            subset = df[df[TARGET_COL] == (1 if label == "CKD" else 0)][col].dropna()
            ax.hist(subset, bins=20, alpha=0.5, label=label, color=color, edgecolor="black")
        ax.set_title(col, fontweight="bold")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Numerical Feature Distributions by Class", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "04_numerical_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix heatmap for numerical features."""
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=14)
    save(fig, "05_correlation_heatmap.png")


def plot_box_plots(df: pd.DataFrame):
    """Box plots of key numerical features split by target."""
    key_features = ["hemo", "sc", "bu", "bgr", "pcv", "rc", "sod", "al"]
    available = [f for f in key_features if f in df.columns]
    n = len(available)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(available):
        ax = axes[i]
        sns.boxplot(
            data=df, x=TARGET_COL, y=col, ax=ax,
            palette={1: "#e74c3c", 0: "#2ecc71"}, hue=TARGET_COL, legend=False
        )
        ax.set_xticklabels(["Not CKD", "CKD"])
        ax.set_title(col, fontweight="bold")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Box Plots of Key Features by Class", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "06_box_plots.png")


def plot_categorical_distributions(df: pd.DataFrame):
    """Count plots of categorical features split by target."""
    cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    n = len(cats)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cats):
        ax = axes[i]
        ct = df.groupby([col, TARGET_COL]).size().unstack(fill_value=0)
        ct.plot(kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black")
        ax.set_title(col, fontweight="bold")
        ax.legend(["Not CKD", "CKD"], fontsize=8)
        ax.tick_params(axis='x', rotation=0)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Categorical Feature Distributions by Class", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "07_categorical_distributions.png")


def run_eda():
    """Run all EDA plots."""
    print("=" * 60)
    print("  PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df_raw = load_data()
    df_clean = clean_data(df_raw)

    print("\n--- Dataset Overview ---")
    print(f"  Shape: {df_clean.shape}")
    print(f"  Target Distribution:")
    counts = df_clean[TARGET_COL].value_counts()
    for idx, cnt in counts.items():
        lbl = "CKD" if idx == 1 else "Not CKD"
        print(f"    {lbl}: {cnt} ({cnt/len(df_clean)*100:.1f}%)")

    print("\n--- Generating Plots ---")
    plot_target_distribution(df_clean)
    plot_missing_heatmap(df_clean)
    plot_missing_bar(df_clean)
    plot_numerical_distributions(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_box_plots(df_clean)
    plot_categorical_distributions(df_clean)

    print(f"\n✅ EDA complete. All plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    run_eda()
