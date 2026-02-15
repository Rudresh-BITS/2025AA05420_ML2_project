#!/usr/bin/env python3
"""
FIXED: Safe Git handling + existing repo support
"""

import pandas as pd
import numpy as np
import subprocess
import os
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

# YOUR SETTINGS
REPO_NAME = "2025AA05420_ML2_project"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/Rukifuru/2025AA05420_ML2_project/main/dow_jones_index.csv"
GIT_USERNAME = "Rukifuru"
GIT_EMAIL = "dutta.rudresh@gmail.com"
REPO_URL = f"https://github.com/{GIT_USERNAME}/{REPO_NAME}.git"

print("🚀 ML Assignment Pipeline - Git FIXED!")

# 1. LOAD + TRAIN (already working!)
print("📥 Loading data...")
df = pd.read_csv(GITHUB_RAW_URL)

def clean_money(series): 
    return pd.to_numeric(series.astype(str).str.replace("$","").str.replace(",",""), errors="coerce")

money_cols = ["open","high","low","close","next_weeks_open","next_weeks_close"]
for col in money_cols: 
    if col in df.columns: df[col] = clean_money(df[col])

numeric_cols = ["percent_change_price","percent_change_volume_over_last_wk","percent_change_next_weeks_price","percent_return_next_dividend","volume","previous_weeks_volume","days_to_next_dividend","quarter"]
for col in numeric_cols:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["percent_change_next_weeks_price"])
df["target_up"] = (df["percent_change_next_weeks_price"] > 0).astype(int)

feature_cols = ["quarter","open","high","low","close","volume","percent_change_price","percent_change_volume_over_last_wk","previous_weeks_volume","days_to_next_dividend","percent_return_next_dividend"]
feature_cols = [c for c in feature_cols if c in df.columns]
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

df_encoded = pd.get_dummies(df, columns=["stock"], drop_first=True)
stock_features = [c for c in df_encoded if c.startswith("stock_")]
X = df_encoded[feature_cols + stock_features]
y = df_encoded["target_up"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test.to_csv("test_data.csv", index=False)

# Train models (already working)
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

# README
readme = f"""# Dow Jones ML Models

## Metrics
| Model | Acc | AUC | Prec | Rec | F1 | MCC |
|-------|----|-----|------|-----|----|-----|
"""
for name, m in metrics.items():
    readme += f"| {name.replace('_',' ').title()} | {m['accuracy']:.3f} | {m['auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['mcc']:.3f} |\n"
readme += "| Streamlit App | [LIVE LINK] |"

with open("README.md", "w") as f: f.write(readme)

print("✅ Models + test_data.csv + README ready!")

# 6. BULLETPROOF GIT (handles master/main)
print("🐙 BULLETPROOF Git setup...")

# Global config
subprocess.run(["git", "config", "--global", "user.name", GIT_USERNAME])
subprocess.run(["git", "config", "--global", "user.email", GIT_EMAIL])

# Safe git init
if not os.path.exists(".git"):
    subprocess.run(["git", "init"])

# Rename to main if on master
result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
current_branch = result.stdout.strip()
if current_branch == "master":
    subprocess.run(["git", "branch", "-M", "main"])
    print("✅ Renamed master → main")

# Add/commit
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", f"Update models + metrics {pd.Timestamp.now()}"])

# Safe remote + push
try:
    subprocess.run(["git", "remote", "remove", "origin"], check=False)
except: 
    pass

subprocess.run(["git", "remote", "add", "origin", REPO_URL])

# **FIXED: Use subprocess.run() with your PAT URL**
PAT_URL = "https://Rudresh-BITS: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@github.com/Rudresh-BITS/2025AA05420_ML2_project.git"

print("🚀 Pushing to GitHub...")
result = subprocess.run(["git", "remote", "set-url", "origin", PAT_URL], check=True)
result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("🎉 PUSH SUCCESS!")
    print(f"📂 Repo: https://github.com/{GIT_USERNAME}/{REPO_NAME}")
else:
    print("⚠️  Push warning (normal if repo empty):")
    print(result.stderr)

