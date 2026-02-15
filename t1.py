#!/usr/bin/env python3
"""
COMPLETE PIPELINE: GitHub → Train → Commit → Push
"""

import pandas as pd
import numpy as np
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
import pickle
import warnings
warnings.filterwarnings("ignore")

# YOUR GITHUB SETTINGS
REPO_NAME = "2025AA05420_ML2_project"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/Rudresh-BITS/2025AA05420_ML2_project/main/dow_jones_index.csv"  # Replace
GIT_USERNAME = "Rudresh-BITS"  # Replace
GIT_EMAIL = " 2025aa05420@wilp.bits-pilani.ac.in"    # Replace
REPO_URL = f"https://github.com/{GIT_USERNAME}/{REPO_NAME}.git"

print("🚀 ML Assignment Pipeline Starting...")

# 1. LOAD DATA FROM GITHUB
print("📥 Downloading from GitHub...")
df = pd.read_csv(GITHUB_RAW_URL)
print(f"✅ Data loaded: {df.shape}")

# 2. CLEAN + TRAIN (same logic, shortened)
def clean_money(series): 
    return pd.to_numeric(series.astype(str).str.replace("$","").str.replace(",",""), errors="coerce")

money_cols = ["open","high","low","close","next_weeks_open","next_weeks_close"]
for col in money_cols: 
    if col in df.columns: df[col] = clean_money(df[col])

numeric_cols = ["percent_change_price","percent_change_volume_over_last_wk","percent_change_next_weeks_price","percent_return_next_dividend","volume","previous_weeks_volume","days_to_next_dividend","quarter"]
for col in numeric_cols:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

# Target
df = df.dropna(subset=["percent_change_next_weeks_price"])
df["target_up"] = (df["percent_change_next_weeks_price"] > 0).astype(int)

# Features
feature_cols = ["quarter","open","high","low","close","volume","percent_change_price","percent_change_volume_over_last_wk","previous_weeks_volume","days_to_next_dividend","percent_return_next_dividend"]
feature_cols = [c for c in feature_cols if c in df.columns]
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

df_encoded = pd.get_dummies(df, columns=["stock"], drop_first=True)
stock_features = [c for c in df_encoded if c.startswith("stock_")]
X = df_encoded[feature_cols + stock_features]
y = df_encoded["target_up"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save test data
X_test.to_csv("test_data.csv", index=False)

# 3. TRAIN MODELS
models = {
    "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(n_estimators=100, random_state=42)
}

metrics = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    pickle.dump(model, open(f"{name}.pkl", "wb"))
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

print("✅ All models trained & saved!")

# 4. CREATE README WITH METRICS
readme_content = """# Dow Jones Stock Price Prediction

## Dataset
Weekly Dow Jones Industrial Average stock data (30 stocks).

## Problem Statement
Predict if stock price will **increase next week** (binary classification).

## Models Performance

| Model              | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------|----------|-------|-----------|--------|-------|-------|
"""
for name, m in metrics.items():
    readme_content += f"| {name.replace('_',' ').title()} | {m['accuracy']:.3f} | {m['auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['mcc']:.3f} |\n"

readme_content += """
|--------------------|----------|-------|-----------|--------|-------|-------|

## Files
- `test_data.csv` - Test features for Streamlit app
- `*.pkl` - 6 trained models
- `app.py` - Streamlit app (add separately)

## Deployment
Streamlit app: [LIVE LINK HERE]
"""

with open("README.md", "w") as f:
    f.write(readme_content)

print("📝 README.md created with metrics!")

# 5. GIT SETUP + COMMIT + PUSH
print("🐙 Setting up Git...")

# Git config
subprocess.run(["git", "config", "user.name", GIT_USERNAME], check=True)
subprocess.run(["git", "config", "user.email", GIT_EMAIL], check=True)

# Init/Add/Commit
subprocess.run(["git", "init"], check=True)
subprocess.run(["git", "add", "."], check=True)
subprocess.run(["git", "commit", "-m", "Add trained models + test data + README"], check=True)

# Add remote + push
subprocess.run(["git", "remote", "add", "origin", REPO_URL], check=True)
subprocess.run(["git", "branch", "-M", "main"], check=True)
subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

print("🎉 SUCCESS!")
print("✅ Files pushed to GitHub:")
print(f"   Repo: https://github.com/{GIT_USERNAME}/{REPO_NAME}")
print("   Files: test_data.csv + 6 .pkl + README.md")
print("✅ Copy REPO URL to your assignment PDF!")
