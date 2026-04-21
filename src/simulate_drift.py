"""
simulate_drift.py
-----------------
Script that loads the reference dataset and intentionally injects 
data drift into some features (e.g., tenure, MonthlyCharges, Contract)
so that we can demonstrate the Data Drift dashboard.
"""
import os
import pandas as pd
import numpy as np

# Path to the raw data
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
SIMULATED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "simulated_churn_data.csv")

def simulate_drift(input_path: str, output_path: str):
    print(f"Loading reference data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print("Injecting Drift to data...")
    # Drift 1: Customer tenure drops significantly (e.g. many new customers)
    # tenure is numeric
    df['tenure'] = df['tenure'].apply(lambda x: max(0, x - np.random.randint(20, 50)) if pd.notnull(x) else x)
    
    # Drift 2: Monthly Charges spike (e.g. new pricing update)
    if pd.api.types.is_numeric_dtype(df['MonthlyCharges']):
        df['MonthlyCharges'] = df['MonthlyCharges'] * np.random.uniform(1.5, 2.5, size=len(df))
    
    # Drift 3: Shift in categorical distribution (Everyone switches to Month-to-month contracts)
    # We will randomly convert 70% of One year / Two year contracts to Month-to-month
    if 'Contract' in df.columns:
        mask = df['Contract'] != 'Month-to-month'
        indices_to_change = df[mask].sample(frac=0.7, random_state=42).index
        df.loc[indices_to_change, 'Contract'] = 'Month-to-month'
    
    print(f"Saving simulated drifted data to: {output_path}")
    df.to_csv(output_path, index=False)
    print("DONE! You can now run monitor_drift.py to detect this drift.")

if __name__ == "__main__":
    simulate_drift(RAW_DATA_PATH, SIMULATED_DATA_PATH)
