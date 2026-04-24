import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Health Predictor", page_icon="🏥", layout="centered")

# Pure White Theme UI with Optimized Button Alignment
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 12px;
        padding: 1.5rem !important;
    }

    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    label, p, span, h1, h3 {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* FIX BUTTON WIDTH AND ALIGNMENT */
    .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    .stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        height: 3.5rem !important; /* Increased height Slightly */
        width: 100% !important;   /* Full Width */
        margin-top: 1rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .stButton > button p {
        color: #ffffff !important;
        font-size: 1.1rem !important;
    }

    .bmi-container {
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 8px !important;
        text-align: center;
        margin: 0.5rem 0 !important;
    }
    .bmi-value {
        font-size: 1.2rem;
        font-weight: 900;
    }

    [data-testid="stVerticalBlock"] > div {
        gap: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("models/model.pkl", "rb"))
        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        return model, scaler
    except: return None, None

model, scaler = load_assets()

if model is None:
    st.error("Model files missing.")
    st.stop()

st.markdown("<h1 style='text-align: center;'>🏥 Health Risk Predictor</h1>", unsafe_allow_html=True)

with st.container(border=True):
    st.write("### Demographic & Weight")
    c1, c2 = st.columns(2)
    with c1: gender = st.selectbox("Select Gender", ["Female", "Male"])
    with c2: weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, step=0.1)
    
    st.write("### Precise Measurements")
    c3, c4 = st.columns(2)
    with c3: height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, step=0.1)
    with c4: arm_length = st.number_input("Arm Length (cm)", 10.0, 60.0, 35.0, step=0.1)
    
    c5, c6 = st.columns(2)
    with c5: leg_length = st.number_input("Leg Length (cm)", 10.0, 60.0, 40.0, step=0.1)
    with c6: arm_circum = st.number_input("Arm Circ (cm)", 10.0, 70.0, 30.0, step=0.1)

    bmi = weight / ((height / 100) ** 2)
    st.markdown(f'<div class="bmi-container"><span>Calculated BMI</span><br><span class="bmi-value">{bmi:.2f}</span></div>', unsafe_allow_html=True)

    # Added use_container_width=True back
    if st.button("EXECUTE ANALYSIS", use_container_width=True):
        f_names = ['BMXWT', 'BMXHT', 'BMI', 'BMXARML', 'BMXLEG', 'BMXARMC', 'Gender']
        input_df = pd.DataFrame([[weight, height, bmi, arm_length, leg_length, arm_circum, (1 if gender == "Female" else 0)]], columns=f_names)
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        st.markdown("<hr style='margin: 0.5rem 0; border-top: 2px solid #000;'>", unsafe_allow_html=True)
        if prediction == 1:
            st.error(f"### ⚠️ High Risk Detected ({prob:.1%})")
        else:
            st.success(f"### ✅ Low Risk Detected ({prob:.1%})")

st.markdown("<p style='text-align: center; color: #000; font-size: 0.8rem; font-weight: bold;'>High Precision Analytics Engine</p>", unsafe_allow_html=True)