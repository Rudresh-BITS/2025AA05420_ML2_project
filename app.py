import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_artifacts():
    models = {}
    for name in ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]:
        models[name] = pickle.load(open(f"{name}.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
    return models, feature_names

models, feature_names = load_artifacts()

st.title("🚀 Dow Jones Stock Prediction")
st.markdown("15 features | 6 models | Live demo")

st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# Download button
try:
    demo_df = pd.read_csv("test_data.csv")
    csv_data = demo_df.to_csv(index=False)
    st.sidebar.download_button("📥 Download test_data.csv", csv_data, "test_data.csv", "text/csv")
except:
    st.sidebar.info("ℹ️ Run training script first")

model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Load data
@st.cache_data
def load_demo_data():
    return pd.read_csv("test_data.csv")

df_test = pd.read_csv(uploaded_file) if uploaded_file else load_demo_data()
if df_test is None:
    st.warning("Upload CSV or run training!")
    st.stop()

st.success(f"✅ Loaded {len(df_test)} rows")
st.dataframe(df_test.head())

# Clean existing columns only
clean_cols = ["open", "high", "low", "close", "quarter", "percent_change_price", 
              "percent_change_volume_over_last_wk", "percent_change_next_weeks_price", 
              "percent_return_next_dividend", "volume", "previous_weeks_volume", "days_to_next_dividend"]
for col in clean_cols:
    if col in df_test.columns:
        df_test[col] = pd.to_numeric(df_test[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")

# Fill base features (like training)
base_cols = ["quarter", "open", "high", "low", "close", "volume", "percent_change_price", 
             "percent_change_volume_over_last_wk", "previous_weeks_volume", "days_to_next_dividend", "percent_return_next_dividend"]
base_cols = [c for c in base_cols if c in df_test.columns]
if base_cols:
    df_test[base_cols] = df_test[base_cols].fillna(df_test[base_cols].median())

# Target (optional)
target_col = "percent_change_next_weeks_price"
if target_col in df_test.columns:
    df_test = df_test.dropna(subset=[target_col])
    y_true = (df_test[target_col] > 0).astype(int)
    st.success("✅ Target column found")
else:
    y_true = None
    st.warning("⚠️ No target column - predictions only")

# Safe encoding
df_encoded = df_test.copy()
if "stock" in df_test.columns:
    df_encoded = pd.get_dummies(df_encoded, columns=["stock"], drop_first=True)

# **CRITICAL**: Exact feature alignment
X_test = df_encoded.reindex(columns=feature_names, fill_value=0)

st.success(f"✅ Features aligned: {X_test.shape[1]} (matches training)")

# Predict
model = models[model_name]
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Show predictions
st.subheader("🔮 Predictions (Last 10)")
pred_df = pd.DataFrame({
    "Predicted (1=Up)": y_pred[-10:],
    "Probability Up": np.round(y_proba[-10:], 3)
})
if y_true is not None:
    pred_df["True"] = y_true.tail(10).values
st.dataframe(pred_df)

# Metrics (SAFE length check)
if y_true is not None and len(y_true) == len(y_pred):
    st.subheader("📈 Model Performance")
    metrics_dict = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for idx, (metric, value) in enumerate(metrics_dict.items()):
        with cols[idx % 3]:
            st.metric(metric, f"{value:.3f}")


    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Down (0)", "Up (1)"], 
                yticklabels=["Down (0)", "Up (1)"], ax=ax)
    st.pyplot(fig)

    # Classification Report
    st.subheader("📋 Detailed Report")
    st.code(classification_report(y_true, y_pred, target_names=["Down (0)", "Up (1)"]))
else:
    st.info("➕ Add `percent_change_next_weeks_price` column for full metrics")

st.markdown("---")
st.caption(f"**Training Features** ({len(feature_names)}): {', '.join(feature_names[:5])}...")
