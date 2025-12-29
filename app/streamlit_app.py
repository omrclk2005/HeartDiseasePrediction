import streamlit as st
import numpy as np
import pickle
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

# --------------------------------------------------
# Load trained model (correct relative path)
# --------------------------------------------------
MODEL_PATH = os.path.join("models", "heart_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------
# App title & description
# --------------------------------------------------
st.title("Heart Disease Prediction System")
st.write(
    "This application predicts the risk of heart disease using a "
    "machine learning model trained on clinical health data."
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
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])

    # Predict probability
    risk_probability = model.predict_proba(input_data)[0][1]
    risk_percentage = risk_probability * 100

    st.subheader("Prediction Result")
    st.metric(
        label="Estimated Heart Disease Risk",
        value=f"{risk_percentage:.2f}%"
    )

    # Optional interpretation
    if risk_percentage >= 70:
        st.error("Very High Risk – Medical consultation strongly advised.")
    elif risk_percentage >= 40:
        st.warning("Moderate Risk – Lifestyle and medical follow-up recommended.")
    else:
        st.success("Low Risk – Maintain healthy lifestyle.")
