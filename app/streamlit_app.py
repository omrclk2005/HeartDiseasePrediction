import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

# --------------------------------------------------
# Load trained Logistic Regression pipeline
# --------------------------------------------------
MODEL_PATH = os.path.join("models", "lr_pipeline.pkl")
lr_pipeline = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Feature order (must match training)
# --------------------------------------------------
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# --------------------------------------------------
# App title & description
# --------------------------------------------------
st.title("Heart Disease Prediction System")
st.write(
    "This application predicts the risk of heart disease using a "
    "Logistic Regression model with proper feature preprocessing."
)

st.divider()

# --------------------------------------------------
# User input section
# --------------------------------------------------
st.subheader("Patient Information")

age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol Level", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression", value=1.0)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()

if st.button("Predict Heart Disease Risk"):
    input_df = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=FEATURE_COLUMNS)

    risk_probability = lr_pipeline.predict_proba(input_df)[0][1]
    risk_percentage = risk_probability * 100

    st.subheader("Prediction Result")
    st.write(f"Estimated probability of heart disease: **{risk_percentage:.2f}%**")

    if risk_percentage >= 60:
        st.error("High Risk – Medical consultation advised.")
    elif risk_percentage >= 30:
        st.warning("Moderate Risk – Lifestyle changes recommended.")
    else:
        st.success("Low Risk – Maintain healthy habits.")
