"""
preprocessing.py
----------------
Phase 2 - Modularize ML Pipeline

Builds a Scikit-learn Pipeline that handles:
  - Missing values (TotalCharges)
  - Categorical encoding (OneHotEncoder)
  - Numerical scaling (StandardScaler)
  - Binary label encoding (Churn: Yes/No → 1/0)

TODO: Implement full pipeline — to be refactored from notebook.
"""

import logging
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def get_preprocessing_pipeline(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """Build and return the full preprocessing ColumnTransformer."""
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


def preprocess(df: pd.DataFrame) -> tuple:
    """
    Clean and split data into X (features) and y (target).
    Returns (X_processed_df, y_series, fitted_preprocessor)
    """
    # TODO: Full implementation — refactor from notebook
    raise NotImplementedError("Full implementation in Phase 2")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Preprocessing script placeholder — Phase 2 implementation pending.")
