#!/usr/bin/env python3
"""
FIXED: Max 16 features + NO 'stock' KeyError
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

print("🚀 ML Assignment Pipeline - MAX 16 FEATURES (FIXED!)")

# 1. LOAD + PREPROCESS
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

# BASE FEATURES (11 max)
feature_cols = ["quarter","open","high","low","close","volume","percent_change_price","percent_change_volume_over_last_wk","previous_weeks_volume","days_to_next_dividend","percent_return_next_dividend"]
feature_cols = [c for c in feature_cols if c in df.columns]
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

print(f"📊 Base features: {len(feature_cols)}")

# **LIMIT TO 16 TOTAL** - Top 5 stocks
df_encoded = pd.get_dummies(df, columns=["stock"], drop_first=True)
all_stock_features = [c for c in df_encoded if c.startswith("stock_")]

top_stocks = df['stock'].value_counts().head(5).index
top_stock_features = [f"stock_{s}" for s in top_stocks if f"stock_{s}" in all_stock_features]
print(f"🏢 Top stocks: {top_stocks.tolist()} ({len(top_stock_features)} dummies)")

selected_features = feature_cols + top_stock_features[:5]
print(f"🎯 Features ({len(selected_features)}): {selected_features}")

# X with ONLY selected features (missing dummies auto-filled with 0)
X = df_encoded.reindex(columns=selected_features, fill_value=0)
y = df["target_up"]  # Use original df (index aligned)

print(f"✅ X shape: {X.shape}")

# Save feature names
with open("feature_names.pkl", "wb") as f:
    pickle.dump(selected_features, f)

# 80/20 SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"📈 Train/Test: {X_train.shape}/{X_test.shape}")

# **FIXED test_data.csv** - Base cols + target ONLY (no stock column needed)
test_save_cols = [c for c in feature_cols if c in df.columns] + ["percent_change_next_weeks_price"]
if len(test_save_cols) > 0:
    # Reset index to match original order for target lookup
    test_df = df.loc[X_test.index][test_save_cols].reset_index(drop=True)
    test_df.to_csv("test_data.csv", index=False)
    print(f"💾 test_data.csv saved ({len(test_save_cols)} cols)")
else:
    print("⚠️ No test cols - creating empty")
    pd.DataFrame().to_csv("test_data.csv", index=False)

# Train models
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
readme = f"""# Dow Jones ML Models (≤16 Features)

## Metrics
| Model | Acc | AUC | Prec | Rec | F1 | MCC |
|-------|----|-----|------|-----|----|-----|
"""

for name, m in metrics.items():
    readme += f"| {name.replace('_',' ').title()} | {m['accuracy']:.3f} | {m['auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['mcc']:.3f} |\n"

readme += "| Streamlit App | [LIVE LINK] |\n\n**Features**: {len(selected_features)} total (top stocks + base)"

with open("README.md", "w") as f: 
    f.write(readme)

print("\n📊 ALL 6 MODEL METRICS:")
for name, m in metrics.items():
    print(f"{name.title()}: Acc={m['accuracy']:.3f}, AUC={m['auc']:.3f}, F1={m['f1']:.3f}")

# GIT (unchanged)
print("🐙 Git setup...")
subprocess.run(["git", "config", "--global", "user.name", GIT_USERNAME])
subprocess.run(["git", "config", "--global", "user.email", GIT_EMAIL])

if not os.path.exists(".git"):
    subprocess.run(["git", "init"])

result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
if result.stdout.strip() == "master":
    subprocess.run(["git", "branch", "-M", "main"])

subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", f"Fixed 16-features + no KeyError {pd.Timestamp.now()}"])

try: subprocess.run(["git", "remote", "remove", "origin"])
except: pass

subprocess.run(["git", "remote", "add", "origin", REPO_URL])

PAT_URL = "https://Rudresh-BITS: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@github.com/Rudresh-BITS/2025AA05420_ML2_project.git"

print("🚀 Pushing...")
result = subprocess.run(["git", "remote", "set-url", "origin", PAT_URL], check=True)
result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], capture_output=True, text=True)

if result.returncode == 0:
    print("🎉 SUCCESS!")
    print(f"📂 https://github.com/{GIT_USERNAME}/{REPO_NAME}")
else:
    print("⚠️ Push issue:", result.stderr)
