import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showfileUploaderEncoding', False)

# Load all your models (@st.cache_data for speed)
@st.cache_data
def load_models():
    models = {}
    model_names = ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
    for name in model_names:
        try:
            models[name] = pickle.load(open(f"{name}.pkl", "rb"))
        except FileNotFoundError:
            st.error(f"Missing {name}.pkl – run training script first!")
            st.stop()
    return models

models = load_models()

# Feature cols from your training
feature_cols = ["quarter", "open", "high", "low", "close", "volume", "percent_change_price", 
                "percent_change_volume_over_last_wk", "previous_weeks_volume", "days_to_next_dividend", 
                "percent_return_next_dividend"]
money_cols = ["open", "high", "low", "close", "next_weeks_open", "next_weeks_close"]
numeric_cols = ["percent_change_price", "percent_change_volume_over_last_wk", "percent_change_next_weeks_price", 
                "percent_return_next_dividend", "volume", "previous_weeks_volume", "days_to_next_dividend", "quarter"]

st.title("🚀 Dow Jones Stock Prediction App")
st.markdown("Upload test CSV (like `test_data.csv`), select model, get predictions & metrics.")

# Sidebar: Upload & Model select
st.sidebar.header("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader("CSV only (test data structure)", type="csv")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Preview test_data.csv if no upload
@st.cache_data
def load_demo_data():
    try:
        return pd.read_csv("test_data.csv")
    except:
        st.warning("No test_data.csv – upload one!")
        return None

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
else:
    df_test = load_demo_data()
    if df_test is None:
        st.stop()

if df_test is not None:
    st.success(f"Loaded {len(df_test)} rows")
    st.dataframe(df_test.head())

    # Clean & preprocess (exact match to training)
    for col in money_cols:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col].astype(str).str.replace("$", "").str.replace(",", ""), errors="coerce")
    for col in numeric_cols:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
    
    # Target (for metrics)
    if "percent_change_next_weeks_price" in df_test.columns:
        y_true = (df_test["percent_change_next_weeks_price"] > 0).astype(int)
    else:
        y_true = None  # No metrics if no target
    
    df_test = df_test.dropna(subset=["percent_change_next_weeks_price"]) if "percent_change_next_weeks_price" in df_test else df_test
    df_encoded = pd.get_dummies(df_test, columns=["stock"], drop_first=True)
    stock_features = [c for c in df_encoded.columns if c.startswith("stock_")]
    X_test = df_encoded[feature_cols + stock_features].fillna(df_encoded[feature_cols + stock_features].median())

    model = models[model_name]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Predictions table
    st.subheader("🔮 Predictions (last 10)")
    pred_df = pd.DataFrame({
        "True (if avail)": y_true.tail(10) if y_true is not None else "N/A",
        "Predicted Up": y_pred[-10:],
        "Proba Up": np.round(y_proba[-10:], 3)
    })
    st.dataframe(pred_df)

    # Metrics (if y_true exists)
    if y_true is not None:
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }
        col1, col2, col3 = st.columns(3)
        for i, (k, v) in enumerate(metrics.items()):
            with (col1, col2, col3)[i % 3]:
                st.metric(k, f"{v:.3f}")

        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        st.subheader("📈 Classification Report")
        st.text(classification_report(y_true, y_pred))
