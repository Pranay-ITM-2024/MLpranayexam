"""
Feature Engineering & Selection for CKD Classification
=======================================================
- Correlation filtering (drop >0.95 correlated pairs)
- Random Forest feature importances
- Mutual Information scores
- Combined ranking
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def drop_high_correlation(X: np.ndarray, feature_names: list, threshold: float = 0.95):
    """Drop one of each pair of features with correlation > threshold."""
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    keep = [f for f in feature_names if f not in to_drop]
    print(f"  Dropping {len(to_drop)} highly correlated features: {to_drop}")
    return to_drop, keep


def rf_feature_importance(X_train, y_train, feature_names):
    """Train a quick RF and return sorted feature importances."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return fi


def mutual_information_scores(X_train, y_train, feature_names):
    """Compute mutual information between each feature and target."""
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({
        "feature": feature_names,
        "mi_score": mi
    }).sort_values("mi_score", ascending=False)
    return mi_df


def plot_feature_importance(fi: pd.DataFrame, mi_df: pd.DataFrame):
    """Side-by-side: RF importance and mutual info scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # RF importance
    colors = sns.color_palette("viridis", len(fi))
    ax1.barh(fi["feature"], fi["importance"], color=colors, edgecolor="black")
    ax1.set_xlabel("Importance")
    ax1.set_title("Random Forest Feature Importance", fontweight="bold")
    ax1.invert_yaxis()

    # Mutual information
    colors2 = sns.color_palette("magma", len(mi_df))
    ax2.barh(mi_df["feature"], mi_df["mi_score"], color=colors2, edgecolor="black")
    ax2.set_xlabel("Mutual Information Score")
    ax2.set_title("Mutual Information Scores", fontweight="bold")
    ax2.invert_yaxis()

    fig.suptitle("Feature Selection Analysis", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "08_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: 08_feature_importance.png")


def run_feature_engineering(X_train, y_train, feature_names):
    """Run all feature engineering steps. Returns combined ranking."""
    print("=" * 60)
    print("  PHASE 4: FEATURE ENGINEERING & SELECTION")
    print("=" * 60)

    # 1. Correlation filter
    print("\n[1/3] Checking for highly correlated features (>0.95) ...")
    to_drop, keep = drop_high_correlation(X_train, feature_names)

    # 2. RF importance
    print("[2/3] Computing Random Forest feature importances ...")
    fi = rf_feature_importance(X_train, y_train, feature_names)
    print(fi.to_string(index=False))

    # 3. Mutual information
    print("\n[3/3] Computing Mutual Information scores ...")
    mi_df = mutual_information_scores(X_train, y_train, feature_names)
    print(mi_df.to_string(index=False))

    # Combined ranking
    fi_rank = fi.reset_index(drop=True)
    fi_rank["rf_rank"] = range(1, len(fi_rank) + 1)

    mi_rank = mi_df.reset_index(drop=True)
    mi_rank["mi_rank"] = range(1, len(mi_rank) + 1)

    combined = fi_rank[["feature", "rf_rank"]].merge(
        mi_rank[["feature", "mi_rank"]], on="feature"
    )
    combined["avg_rank"] = (combined["rf_rank"] + combined["mi_rank"]) / 2
    combined = combined.sort_values("avg_rank")

    print("\n--- Combined Feature Ranking ---")
    print(combined.to_string(index=False))

    # Plot
    plot_feature_importance(fi, mi_df)

    print("\n✅ Feature engineering complete.")
    return fi, mi_df, combined


if __name__ == "__main__":
    from data_preprocessing import run_full_pipeline
    (_, _, _, _, X_train, X_test, y_train, y_test,
     scaler, label_encoders, feature_names, _) = run_full_pipeline()
    run_feature_engineering(X_train, y_train, feature_names)
