"""
preprocessing.py
----------------
Phase 2 — Build features and split data for training.

Steps (mirroring the notebook):
  - Label-encode binary/ordinal categorical columns
  - One-hot encode multi-class categorical columns (PaymentMethod, Contract, InternetService)
  - Standard-scale numerical columns (tenure, MonthlyCharges, TotalCharges)
  - Encode the target column (Churn: Yes → 1, No → 0)
  - Train/test split (70/30, stratified, random_state=40)

Returns:
    X_train, X_test, y_train, y_test, preprocessor
"""

import logging
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")

# ─── Column Groups (same logic as notebook Cell 24) ──────────────────────────
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Multi-class columns → OneHotEncoder
OHE_COLS = ["PaymentMethod", "Contract", "InternetService"]

# Demographic columns to drop for bias/fairness check
DEMOGRAPHIC_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents"]

# Binary/ordinal columns → LabelEncoder (handled separately before pipeline)
LABEL_ENCODE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling",
]

TARGET_COL = "Churn"


def label_encode_binary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all binary/ordinal categorical columns in-place."""
    df = df.copy()
    le = LabelEncoder()
    for col in LABEL_ENCODE_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    # Encode target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = le.fit_transform(df[TARGET_COL].astype(str))
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer:
      - Numerical cols → StandardScaler
      - OHE cols       → OneHotEncoder (drop='first' to avoid multicollinearity)
    """
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL_COLS),
        ("cat", categorical_pipeline, OHE_COLS),
    ], remainder="passthrough")   # Keep label-encoded cols as-is
    return preprocessor


def preprocess(df: pd.DataFrame, save_preprocessor: bool = True):
    """
    Full preprocessing pipeline.

    Args:
        df: Cleaned DataFrame from data_ingestion.ingest()
        save_preprocessor: If True, saves the fitted preprocessor to disk.

    Returns:
        X_train, X_test, y_train, y_test (all numpy arrays)
    """
    logger.info("Starting preprocessing...")

    # Step 1: Drop demographic columns for fairness (Bias Check)
    df = df.drop(columns=DEMOGRAPHIC_COLS, errors="ignore")

    # Step 2: Label-encode binary columns + target
    df = label_encode_binary_cols(df)

    # Step 3: Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    logger.info(f"Features shape: {X.shape} | Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Step 4: Train/test split (70/30, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=40, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Step 5: Fit preprocessor on train, transform both
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Step 6 (optional): Save fitted preprocessor for inference
    if save_preprocessor:
        os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        logger.info(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_ingestion import ingest
    df = ingest()
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Churn rate in train: {y_train.mean():.2%}")
    print(f"Churn rate in test:  {y_test.mean():.2%}")
