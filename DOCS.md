Detail Report: Customer Churn Prediction System

1. Project Overview

    Problem Statement: The telecommunications industry faces high competition, where the cost of acquiring new customers is 5-25 times higher than retaining existing ones. Current strategies are often "Reactive," leading to lost revenue.

    Project Goal: To reduce the Churn Rate by transitioning to a "Targeted Retention" strategy using predictive analytics.

    System Goal: Develop a Decision Support System that provides a "Churn Risk Score" for marketing and customer retention teams.

2. Dataset Specification

    Source: IBM "Telco Customer Churn" dataset from Kaggle.

    Size: 7,043 rows with 21 features.

    Key Features: CustomerID, gender, SeniorCitizen, Partner, and various service usage metrics.

    Target Variable: Churn (Binary: Yes/No).

3. Machine Learning Task & Metrics

    Task: Binary Classification.

    Primary Metric: Recall (Sensitivity).

        Reasoning: In business, a False Negative (failing to detect a churning customer) is more costly than a False Positive (offering a promotion to a loyal customer).

    Secondary Metrics: Accuracy (target ~82%) and F1-Score.

4. Technical Implementation Details
4.1 Data Preprocessing & Engineering

    Pipeline: Automated Preprocessing Pipeline.

    Class Imbalance Handling:

        Use SMOTE (Synthetic Minority Over-sampling Technique).

        Apply class_weight adjustments during training.

    Anonymization: Remove identifiable information like customerID during preprocessing for privacy.

4.2 Model Selection

    Chosen Model: Voting Classifier (Ensemble).

    Constituent Models: Combined strengths of multiple algorithms (e.g., Gradient Boosting, AdaBoost) to capture non-linear patterns and reduce bias/variance.

    Performance: Achieved approximately 81.7% - 82% Accuracy with high stability against noise.

4.3 Advanced Tuning

    Threshold Tuning: Implement Decision-making threshold adjustment to balance Precision and Recall based on marketing budget and business impact.

    Cost Simulation: Evaluate the financial impact of False Positives vs. False Negatives.

5. System Architecture & Deployment

    Workflow: Data Source -> Preprocessing -> Training Pipeline -> Model Registry -> Inference API.

    Prediction Mode: Batch Prediction (Monthly processing aligned with billing cycles).

    Interpretability: Use tools like SHAP or LIME (planned) to explain risk scores to end-users.

    Fallback Plan: Return to Rule-based screening if model performance drifts or fails.

6. Security & Ethics

    Data Minimization: Only use billing and network usage data necessary for prediction.

    Access Control: Limit access to raw data and Risk Scores to authorized retention teams only.

    Human-in-the-loop: Final decisions on promotions or customer contact remain with human staff.