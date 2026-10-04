"""
Data Preprocessing Pipeline for Chronic Kidney Disease Classification
=====================================================================
Handles: loading, type correction, whitespace stripping, missing value
imputation (KNN for numerical, mode for categorical), label encoding,
feature scaling, and stratified train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import joblib
import os

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "kidney_disease.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Columns that should be numeric but are sometimes stored as strings
NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc"
]

# Truly categorical (binary) columns
CATEGORICAL_COLS = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
]

TARGET_COL = "classification"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix dtypes, strip whitespace from strings, replace '?' with NaN,
    and drop the 'id' column.
    """
    df = df.copy()

    # Drop id column
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # Strip whitespace from all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Replace ? and empty strings with NaN
    df.replace({"?": np.nan, "": np.nan, "\t": np.nan}, inplace=True)

    # Fix numeric columns stored as object
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix target column
    df[TARGET_COL] = df[TARGET_COL].str.strip().str.replace("\t", "")
    df[TARGET_COL] = df[TARGET_COL].map({"ckd": 1, "notckd": 0, "ckd\t": 1})

    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return summary[summary["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-pass imputation:
      1. Categorical columns → mode imputation
      2. Numerical columns → KNN imputer (k=5)
    """
    df = df.copy()

    # --- Categorical: mode imputation ---
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # --- Numerical: KNN imputation ---
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])

    return df


def encode_features(df: pd.DataFrame):
    """
    Label-encode categorical features.
    Returns (df_encoded, label_encoders_dict).
    """
    df = df.copy()
    label_encoders = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split + StandardScaler (fit on train only).
    Returns X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def run_full_pipeline():
    """
    Execute the full preprocessing pipeline and return all artefacts.
    """
    print("=" * 60)
    print("  PHASE 2: DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading data ...")
    df_raw = load_data()
    print(f"  Shape: {df_raw.shape}")

    # 2. Clean
    print("[2/6] Cleaning data ...")
    df_clean = clean_data(df_raw)

    # 3. Missing summary (before imputation)
    print("[3/6] Missing value summary BEFORE imputation:")
    missing_before = get_missing_summary(df_clean)
    print(missing_before.to_string())

    # 4. Impute
    print("\n[4/6] Imputing missing values ...")
    df_imputed = impute_data(df_clean)
    missing_after = get_missing_summary(df_imputed)
    print(f"  Missing values remaining: {df_imputed.isnull().sum().sum()}")

    # 5. Encode
    print("[5/6] Encoding categorical features ...")
    df_encoded, label_encoders = encode_features(df_imputed)

    # 6. Split & Scale
    print("[6/6] Splitting (80/20 stratified) and scaling ...")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(
        df_encoded
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Features: {len(feature_names)}")

    # Save artefacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    print("\n✅ Preprocessing complete. Artefacts saved to models/")

    return (
        df_raw, df_clean, df_imputed, df_encoded,
        X_train, X_test, y_train, y_test,
        scaler, label_encoders, feature_names,
        missing_before
    )


if __name__ == "__main__":
    run_full_pipeline()
