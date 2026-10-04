"""
Model Training & Evaluation for CKD Classification
====================================================
Trains 5 models with GridSearchCV, evaluates on test set,
produces comparison tables, confusion matrices, and ROC curves.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from data_preprocessing import run_full_pipeline
from feature_engineering import run_feature_engineering

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Model Definitions + Hyperparameter Grids ----------
MODELS = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_leaf": [1, 2, 4],
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        }
    },
}


def train_models(X_train, y_train):
    """
    Train each model with GridSearchCV (5-fold stratified CV).
    Returns dict of {name: best_estimator}.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained = {}

    for name, config in MODELS.items():
        print(f"\n  🔧 Training: {name} ...")
        grid = GridSearchCV(
            config["model"], config["params"],
            cv=cv, scoring="f1", n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        trained[name] = grid.best_estimator_
        print(f"     Best params: {grid.best_params_}")
        print(f"     Best CV F1:  {grid.best_score_:.4f}")

    return trained


def evaluate_models(trained: dict, X_test, y_test):
    """
    Evaluate each trained model on the test set.
    Returns results DataFrame.
    """
    results = []

    for name, model in trained.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "TN": cm[0][0],
            "FP": cm[0][1],
            "FN": cm[1][0],
            "TP": cm[1][1],
        })

        print(f"\n  📋 {name}:")
        print(f"     Accuracy:  {acc:.4f}")
        print(f"     Precision: {prec:.4f}")
        print(f"     Recall:    {rec:.4f}")
        print(f"     F1-Score:  {f1:.4f}")
        print(f"     Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    return pd.DataFrame(results)


def plot_confusion_matrices(trained: dict, X_test, y_test):
    """Plot confusion matrices for all models side by side."""
    n = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Not CKD", "CKD"],
            yticklabels=["Not CKD", "CKD"],
            cbar=False
        )
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices — All Models", fontweight="bold", fontsize=14, y=1.05)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "09_confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved: 09_confusion_matrices.png")


def plot_roc_curves(trained: dict, X_test, y_test):
    """Overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for (name, model), color in zip(trained.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = os.path.join(PLOTS_DIR, "10_roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved: 10_roc_curves.png")


def plot_metrics_comparison(results_df: pd.DataFrame):
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(results_df))
    width = 0.18
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, results_df[metric], width,
                      label=metric, color=colors[i], edgecolor="black")
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.set_ylim([0.8, 1.05])
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "11_metrics_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved: 11_metrics_comparison.png")


def plot_cv_boxplot(trained: dict, X_train, y_train):
    """Box plot of 5-fold CV accuracy scores per model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_data = []

    for name, model in trained.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        for s in scores:
            cv_data.append({"Model": name, "Accuracy": s})

    cv_df = pd.DataFrame(cv_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=cv_df, x="Model", y="Accuracy", ax=ax,
                palette="Set2", hue="Model", legend=False)
    ax.set_title("Cross-Validation Accuracy Distribution", fontweight="bold", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_ylim([0.85, 1.02])
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "12_cv_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved: 12_cv_boxplot.png")


def select_and_save_best(trained: dict, results_df: pd.DataFrame):
    """Select the best model by F1-Score and save it."""
    best_idx = results_df["F1-Score"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    best_model = trained[best_name]

    print(f"\n  🏆 Best Model: {best_name}")
    print(f"     F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")
    print(f"     Recall:   {results_df.loc[best_idx, 'Recall']:.4f}")

    path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, path)
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))
    print(f"     Saved to: {path}")

    return best_name, best_model


def run_training():
    """Full training pipeline."""
    # --- Preprocessing ---
    (df_raw, df_clean, df_imputed, df_encoded,
     X_train, X_test, y_train, y_test,
     scaler, label_encoders, feature_names,
     missing_before) = run_full_pipeline()

    # --- Feature Engineering ---
    fi, mi_df, combined = run_feature_engineering(X_train, y_train, feature_names)

    # --- Training ---
    print("\n" + "=" * 60)
    print("  PHASE 5: MODEL TRAINING")
    print("=" * 60)
    trained = train_models(X_train, y_train)

    # --- Evaluation ---
    print("\n" + "=" * 60)
    print("  PHASE 6: MODEL EVALUATION & COMPARISON")
    print("=" * 60)
    results_df = evaluate_models(trained, X_test, y_test)

    # --- Visualisations ---
    print("\n--- Generating Evaluation Plots ---")
    plot_confusion_matrices(trained, X_test, y_test)
    plot_roc_curves(trained, X_test, y_test)
    plot_metrics_comparison(results_df)
    plot_cv_boxplot(trained, X_train, y_train)

    # --- Select Best ---
    best_name, best_model = select_and_save_best(trained, results_df)

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print("  COMPARATIVE RESULTS SUMMARY")
    print("=" * 60)
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    print(results_df[display_cols].to_string(index=False))

    # Save results table
    results_df.to_csv(os.path.join(BASE_DIR, "models", "results_summary.csv"), index=False)
    print("\n✅ Training & evaluation complete. Results saved.")

    return trained, results_df, best_name


if __name__ == "__main__":
    run_training()
