"""
data_ingestion.py
-----------------
Phase 2 — Load raw Telco Churn CSV and do initial cleaning:
  - Drop customerID
  - Convert TotalCharges to numeric
  - Remove rows where tenure == 0 (no real customers)
  - Map SeniorCitizen 0/1 → "No"/"Yes"
"""

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV file into a DataFrame."""
    logger.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply initial cleaning steps to the raw DataFrame:
      1. Drop the customerID column (not a feature)
      2. Convert TotalCharges from string to numeric (coerce errors → NaN)
      3. Remove rows where tenure == 0 (invalid / no real billing history)
      4. Fill remaining NaN in TotalCharges with column mean
      5. Map SeniorCitizen integer flag to human-readable string
    """
    logger.info("Cleaning data...")

    # 1. Drop non-feature identifier
    df = df.drop(columns=["customerID"], errors="ignore")

    # 2. TotalCharges sometimes has whitespace strings → coerce to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Customers with tenure == 0 have no charges — drop them
    zero_tenure = df[df["tenure"] == 0].index
    logger.info(f"Dropping {len(zero_tenure)} rows with tenure == 0")
    df.drop(labels=zero_tenure, axis=0, inplace=True)

    # 4. Fill remaining NaN TotalCharges with mean
    df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

    # 5. Make SeniorCitizen readable (consistent with other binary columns)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def ingest(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Full ingestion pipeline: load + clean."""
    df = load_raw_data(path)
    df = clean_data(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = ingest()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
