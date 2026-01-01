import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="Heart Health AI", page_icon="❤️", layout="centered")

# --- Load Artifacts with Correct Paths ---
# This looks one level up (..) from the current file's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_disease_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'data_scaler.pkl')


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None


model, scaler = load_model()

# --- User Interface ---
st.title("❤️ Heart Disease Prediction System")
st.write("Fill in the patient details to assess the risk of heart disease.")

if model is None:
    st.error(f"Error: Model files not found! Looking at: {MODEL_PATH}")
    st.info("Check that the 'models' folder contains your .pkl files.")
else:
    # Form layout
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 1, 110, 50)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
            chol = st.number_input("Cholesterol", 100, 500, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                               format_func=lambda x: "True" if x == 1 else "False")

        with col2:
            restecg = st.selectbox("Resting ECG Result (0-2)", [0, 1, 2])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3])

        submit = st.form_submit_button("Analyze Risk")

    if submit:
        # Prepare and Scale input
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        features_scaled = scaler.transform(features)

        # Get Probabilities (Label 0 is Disease, Label 1 is Healthy)
        probs = model.predict_proba(features_scaled)[0]
        risk_score = probs[0] * 100

        st.divider()
        st.subheader(f"Risk Assessment Result")

        if risk_score >= 70:
            st.error(f"High Risk Detected: {risk_score:.1f}% probability of heart disease.")
        elif risk_score >= 30:
            st.warning(f"Moderate Risk: {risk_score:.1f}% probability of heart disease.")
        else:
            st.success(f"Low Risk: {risk_score:.1f}% probability. Indicators are within healthy ranges.")

st.caption("Developed with Python, Scikit-Learn, and Streamlit.")