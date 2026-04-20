"""
data_ingestion.py
-----------------
Phase 2 - Modularize ML Pipeline

Loads raw Telco Churn CSV from data/raw/ directory.
"""

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    logger.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_raw_data()
    print(df.head())
    print(df.dtypes)
