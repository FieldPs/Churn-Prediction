import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_ingestion import ingest
from preprocessing import build_preprocessor

df = ingest()

# Drop demographics for bias check
df = df.drop(columns=["gender", "SeniorCitizen", "Partner", "Dependents"], errors="ignore")

# Label encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"].astype(str))

# Encode remaining binary cols
binary_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", 
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
for col in binary_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Churn"])
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

preprocessor = build_preprocessor()
# We need to redefine columns in preprocessing
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
ohe_cols = ["PaymentMethod", "Contract", "InternetService"]

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ohe_cols),
], remainder="passthrough")

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Try SMOTE
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
except ImportError:
    print("imbalanced-learn not installed, skipping SMOTE")
    X_train_sm, y_train_sm = X_train, y_train

from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

clf1 = GradientBoostingClassifier(random_state=42)
clf2 = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
clf3 = AdaBoostClassifier(random_state=42)
model = VotingClassifier(estimators=[("gbc", clf1), ("lr", clf2), ("abc", clf3)], voting="soft")

model.fit(X_train_sm, y_train_sm)
probs = model.predict_proba(X_test)[:, 1]

for thresh in [0.3, 0.4, 0.5, 0.6]:
    preds = (probs >= thresh).astype(int)
    rec = recall_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    print(f"Threshold: {thresh} -> Recall: {rec:.4f}, Accuracy: {acc:.4f}")

