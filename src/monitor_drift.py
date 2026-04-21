"""
monitor_drift.py
----------------
Script that uses Evidently AI to compare the original Reference Data
against the Current Data and generates a Dashboard.
"""
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
SIMULATED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "simulated_churn_data.csv")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "monitoring_report.html")

def generate_drift_report(ref_path: str, curr_path: str, output_html: str):
    print("Loading datasets...")
    
    if not os.path.exists(ref_path):
        print(f"Error: Could not find reference data at {ref_path}.")
        return
        
    reference_data = pd.read_csv(ref_path)
    
    if not os.path.exists(curr_path):
        print(f"Error: Could not find current data at {curr_path}.")
        print("Please run simulate_drift.py first!")
        return

    current_data = pd.read_csv(curr_path)
    
    print("Generating Evidently Report (Data Drift & Data Quality) ...")
    # Using Evidently preset to automatically generate standard drift metrics
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Save the report
    drift_report.save_html(output_html)
    print(f"Report successfully saved to {output_html}")
    print("Open this HTML file in your browser to view the drift dashboard.")

if __name__ == "__main__":
    generate_drift_report(RAW_DATA_PATH, SIMULATED_DATA_PATH, REPORT_PATH)
