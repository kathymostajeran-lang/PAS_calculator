import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and imputer
@st.cache_resource
def load_model():
    rf_model = joblib.load('pas_rf_model.pkl')
    imputer = joblib.load('pas_imputer.pkl')
    return rf_model, imputer

rf_model, imputer = load_model()

st.title("PAS Emergent Delivery Risk Calculator")
st.markdown("Predicting bleeding-indicated delivery <34 weeks in PAS.")

# Sidebar inputs
bmi = st.sidebar.number_input("Maternal BMI", 15.0, 60.0, 30.0)
cs = st.sidebar.number_input("Prior Cesarean Sections", 0, 8, 2)
prior_bleed = st.sidebar.selectbox("Prior Admission for Bleeding?", ("No", "Yes"))
us_percreta = st.sidebar.selectbox("Ultrasound Suspicion of Percreta?", ("No", "Yes"))
para_preterm = st.sidebar.number_input("Prior Preterm Births", 0, 5, 0)
tvus_cl = st.sidebar.number_input("TVUS Cervical Length (mm)", 0.0, 60.0, 35.0)
ga_weeks = st.sidebar.number_input("GA at TVUS (Weeks)", 12.0, 34.0, 24.0)

if st.button("Calculate Risk Tier"):
    input_data = pd.DataFrame([[bmi, cs, 1 if prior_bleed=="Yes" else 0, 1 if us_percreta=="Yes" else 0, para_preterm, ga_weeks*7, tvus_cl]], 
                              columns=['bmi', 'cs', 'priorbleed_admit', 'us_percreta', 'para_preterm', 'cl_ga_days', 'tvus_cl'])
    input_imp = imputer.transform(input_data)
    prob = rf_model.predict_proba(input_imp)[0][1] * 100
    
    st.subheader(f"Predicted Risk: {prob:.1f}%")
    if prob > 50:
        st.error("### 🔴 HIGH RISK TIER")
    elif prob >= 15:
        st.warning("### 🟡 MODERATE RISK TIER")
    else:
        st.success("### 🟢 LOW RISK TIER")
