# 📦 Imports
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import joblib
import plotly.express as px
import json
from datetime import datetime
from pathlib import Path
import os

# === File Paths ===
# Update for Streamlit Cloud Solution
MODEL_PATH = os.path.join("model", "keras_lstm_legacy.h5")
SCALER_PATH = os.path.join("model", "scaler.pkl")
SAMPLE_INPUT = os.path.join("model", "sample_input.csv")
MULTI_ENGINE_INPUT = os.path.join("model", "multi_engine_test_input.csv")
MODEL_METRICS_PATH = os.path.join("model", "model_metrics.json")
SHAP_SAMPLE_PATH = os.path.join("model", "sample_shap_values.csv")
IMAGE_PATH = os.path.join("model", "hytec-illustration.png")

# === UI Config ===
st.set_page_config(page_title="Turbofan Engine RUL Predictor", layout="wide")

# === Feature Names Mapping ===
COLUMN_MAP = {
    'op_setting_1': 'Altitude (ft)',
    'op_setting_2': 'Mach Number',
    'op_setting_3': 'Throttle Resolver Angle (°)',
    'sensor_2': 'Total Temperature at LPC Outlet (°R)',
    'sensor_3': 'Total Pressure at HPC Outlet (psia)',
    'sensor_4': 'Physical Fan Speed (rpm)',
    'sensor_7': 'Physical Core Speed (rpm)',
    'sensor_8': 'Engine Pressure Ratio',
    'sensor_9': 'HPC Outlet Static Pressure (psia)',
    'sensor_11': 'Fuel Flow (pps)',
    'sensor_12': 'Fan Inlet Total Temperature (°R)',
    'sensor_13': 'Bypass Duct Pressure (psia)',
    'sensor_14': 'Bypass-duct Total Pressure (psia)',
    'sensor_15': 'HPT Coolant Bleed (lb/min)',
    'sensor_17': 'LPT Coolant Bleed (lb/min)',
    'sensor_20': 'Static Pressure at LPT Outlet (psia)',
    'sensor_21': 'Exhaust Gas Temperature (°R)',
}
FEATURES = list(COLUMN_MAP.keys())

# === Load Model & Scaler ===
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# === Navigation Sidebar ===
page = st.sidebar.radio("📂 Navigate", ["🏠 Home", "🔍 Predict", "🧠 SHAP Explainability", "📊 Model Comparison", "ℹ️ About"])

# === PAGE: Home ===
if page == "🏠 Home":
    st.title("✈️ Turbofan Engine RUL Predictor")
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, width=600)
    else:
        st.warning("🔗 Illustration not found.")
    st.markdown("Welcome to the **Remaining Useful Life (RUL)** prediction platform for aircraft turbofan engines.")

    with st.expander("ℹ️ What is RUL?"):
        st.info("RUL is the estimated number of cycles remaining before failure. It's key for maintenance planning.")

    with st.expander("📘 Sensor Descriptions"):
        for key, val in COLUMN_MAP.items():
            st.markdown(f"• **{key}**: {val}")

# === PAGE: Predict ===
elif page == "🔍 Predict":
    st.title("🔍 Predict Remaining Useful Life")
    mode = st.radio("Choose input mode:", ["Random Sample", "Upload CSV", "Multi-Engine Batch"])

    if mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload engine CSV (30 timesteps)", type="csv")
    elif mode == "Multi-Engine Batch":
        st.info("Predicting on multiple preloaded engines.")

    if st.button("Run Prediction"):
        try:
            if mode == "Random Sample":
                input_df = pd.read_csv(SAMPLE_INPUT)
            elif mode == "Upload CSV" and uploaded_file:
                input_df = pd.read_csv(uploaded_file)
            elif mode == "Multi-Engine Batch":
                input_df = pd.read_csv(MULTI_ENGINE_INPUT)
            else:
                st.warning("Provide a valid input.")
                st.stop()

            input_scaled = input_df.copy()
            input_scaled[FEATURES] = scaler.transform(input_df[FEATURES])
            X_input = input_scaled[FEATURES].values.reshape(-1, 30, len(FEATURES))
            y_pred = model.predict(X_input).flatten()

            if len(y_pred) == 1:
                st.metric("Predicted RUL", f"{y_pred[0]:.2f} cycles")
            else:
                df_result = pd.DataFrame({"Engine_ID": np.arange(1, len(y_pred)+1), "Predicted_RUL": y_pred})
                st.dataframe(df_result)
                st.download_button("📥 Download Predictions", df_result.to_csv(index=False).encode(), "batch_rul.csv")

            log_df = pd.DataFrame({
                "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]*len(y_pred),
                "mode": [mode]*len(y_pred),
                "predicted_RUL": y_pred
            })
            log_df.to_csv("logs.csv", mode='a', index=False, header=not Path("logs.csv").exists())

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# === PAGE: SHAP Explainability ===
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Explainability")
    st.markdown("This explains each sensor's impact on the RUL prediction.")

    # Load SHAP values
    shap_df = pd.read_csv(SHAP_SAMPLE_PATH)

    # Rename SHAP columns for clarity
    readable_columns = {"Timestep": "Timestep"}
    for col in shap_df.columns:
        if col in COLUMN_MAP:
            readable_columns[col] = COLUMN_MAP[col]
    shap_df_renamed = shap_df.rename(columns=readable_columns)

    # Plot SHAP values with readable labels
    st.plotly_chart(px.line(
        shap_df_renamed,
        x="Timestep",
        y=[v for k, v in readable_columns.items() if k != "Timestep"],
        title="SHAP Value Trends Over Time"
    ))

    # Download button for renamed SHAP values
    st.download_button(
        "📥 Download SHAP Values (Readable Names)",
        shap_df_renamed.to_csv(index=False).encode(),
        "shap_values_readable.csv"
    )

    with st.expander("ℹ️ What are SHAP values?"):
        st.info("SHAP values help explain how each input (e.g., sensor) contributed to the model prediction.")


# === PAGE: Model Comparison ===
elif page == "📊 Model Comparison":
    st.title("📊 Compare Models")
    with open(MODEL_METRICS_PATH, "r") as f:
        metrics = json.load(f)

    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics.columns = ["Model", "MAE", "RMSE"]
    st.plotly_chart(px.bar(df_metrics, x="Model", y=["MAE", "RMSE"], barmode="group", title="Model MAE/RMSE"))
    st.dataframe(df_metrics)

# === PAGE: About ===
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
This app predicts the Remaining Useful Life (RUL) of aircraft engines using machine learning models trained on NASA's C-MAPSS dataset.

**Features:**
- 🔍 LSTM prediction with CSV or batch inputs  
- 🧠 SHAP value visualizations  
- 📊 Model comparison: LSTM, RF, XGBoost  

**Tech Stack:**  
Streamlit · TensorFlow · SHAP · Plotly · Scikit-learn
""")
