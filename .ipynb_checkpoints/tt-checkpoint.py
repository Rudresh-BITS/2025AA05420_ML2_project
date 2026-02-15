import requests
import io
import pandas as pd
import numpy as np
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

# FIXED: Proper Google Drive file ID extraction and download
def download_google_drive_file(file_id):
    """Download CSV from Google Drive file ID."""
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(URL)
    
    # Handle confirmation token if large file
    token = None
    for key, value in response.cookies.items():
        if 'download_warning' in key:
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params)
    
    return io.BytesIO(response.content)

# Your Google Drive file ID
file_id = "1bZhAXStOt5JrRVpll00qlFMwnUAE2Z6-"
RAW_PATH = download_google_drive_file(file_id)

# File paths
TEST_PATH = "test_data.csv"  # Save locally

print("1. Loading raw data from Google Drive...")
df = pd.read_csv(RAW_PATH)
print(f"Original shape: {df.shape}")
print("Columns:", df.columns.tolist())

# -----------------------------
# 2. CLEANING AND TYPE CONVERSION
# -----------------------------
def clean_money_column(series):
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace("", np.nan)
        .astype(float)
    )

# Convert money columns
money_cols = ["open", "high", "low", "close", "next_weeks_open", "next_weeks_close"]
for col in money_cols:
    if col in df.columns:
        df[col] = clean_money_column(df[col])

# Convert numeric columns
float_cols = [
    "percent_change_price", "percent_change_volume_over_last_wk",
    "percent_change_next_weeks_price", "percent_return_next_dividend"
]
for col in float_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

int_cols = ["quarter", "volume", "previous_weeks_volume", "days_to_next_dividend"]
for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

print("2. Cleaning complete.")

# -----------------------------
# 3. CREATE BINARY TARGET
# -----------------------------
target_col = "percent_change_next_weeks_price"
df = df.dropna(subset=[target_col])
df["target_up"] = (df[target_col] > 0).astype(int)

print(f"3. Target created. Class distribution:\n{df['target_up'].value_counts(normalize=True)}")

# -----------------------------
# 4. FIXED FEATURES SELECTION
# -----------------------------
feature_cols = [
    "quarter", "open", "high", "low", "close", "volume",
    "percent_change_price", "percent_change_volume_over_last_wk",
    "previous_weeks_volume", "days_to_next_dividend",
    "percent_return_next_dividend"
]

feature_cols = [col for col in feature_cols if col in df.columns]
cat_cols = ["stock"] if "stock" in df.columns else []

print(f"4. Using features: {feature_cols}")

# Fill missing values
num_cols = [c for c in feature_cols if c not in cat_cols]
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

for col in cat_cols:
    mode_val = df[col].mode()
    if len(mode_val) > 0:
        df[col] = df[col].fillna(mode_val[0])

# One-hot encode
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Select ONLY the features we want
stock_cols = [col for col in df_encoded.columns if col.startswith("stock_")]
X = df_encoded[feature_cols + stock_cols]
y = df_encoded["target_up"]

print(f"Final features shape: {X.shape}")

# -----------------------------
# 5. SPLIT & SAVE TEST DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

test_for_app = X_test.reset_index(drop=True)
test_for_app.to_csv(TEST_PATH, index=False)
print(f"5. Test data saved to {TEST_PATH} (shape: {X_test.shape})")

# -----------------------------
# 6. TRAIN 6 MODELS
# -----------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5,
                            random_state=42, eval_metric="logloss")
}

metrics = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Save model
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Predict & metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }
    
    print(f"{name}: AUC={metrics[name]['auc']:.4f}")

print("6. All 6 models saved!")

# -----------------------------
# 7. METRICS TABLE
# -----------------------------
print("\n" + "="*100)
print("COPY THIS TABLE TO README.md:")
print("="*100)
print("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
print("|---------------|----------|-----|-----------|--------|----|-----|")
for name, m in metrics.items():
    print(f"| {name.replace('_', ' ').title()} | "
          f"{m['accuracy']:.4f} | {m['auc']:.4f} | "
          f"{m['precision']:.4f} | {m['recall']:.4f} | "
          f"{m['f1']:.4f} | {m['mcc']:.4f} |")

print("\n✅ SUCCESS! Generated test_data.csv + 6 .pkl files")
