import streamlit as st # using a venv
import pandas as pd
import numpy as np
import os
import joblib
import altair as alt
import base64
from tensorflow.keras.models import load_model
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials


# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Allergy Flare-Up Predictor",
    page_icon="🌿",
    layout="wide"
)


def center_image(image_path, width=100):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded}" width="{width}">
        </div>
        """,
        unsafe_allow_html=True
    )

center_image("longlogo.png", width=350)


st.write(" ")
st.write(" ")

st.markdown("<h1 style='text-align: center;'> 🌿 Allergy Flare-Up & Trigger Dashboard</h1>",
            unsafe_allow_html=True)

st.write(" ")
st.write(" ")

# -----------------------
# Load models
# -----------------------

@st.cache_resource
def load_models():
    lstm_model = load_model("flare_lstm_model.keras")
    trigger_model = joblib.load("trigger_global_rf.joblib")
    return lstm_model, trigger_model

lstm_model, trigger_model = load_models()

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Daily User Inputs")

season = st.sidebar.selectbox(
    "Season",
    ["dry", "rainy_early", "rainy_peak", "rainy_late"]
)

temperature = st.sidebar.slider("Temperature (°C)", 5.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 70.0)
dust_level = st.sidebar.slider("Dust Level", 0.0, 50.0, 10.0)
smoke_level = st.sidebar.slider("Smoke Level", 0.0, 50.0, 10.0)
outdoor_time = st.sidebar.slider("Outdoor Time (minutes)", 0, 300, 60)
medication_used = st.sidebar.selectbox("Medication Used", [0, 1])
symptom_score = st.sidebar.slider("Current Symptom Score", 0, 10, 2)

# -----------------------
# Encode input
# -----------------------
season_map = {
    "dry": 0,
    "rainy_early": 1,
    "rainy_peak": 2,
    "rainy_late": 3
}

input_df = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "dust_level": dust_level,
    "smoke_level": smoke_level,
    "outdoor_time_min": outdoor_time,
    "medication_used": medication_used,
}])

# -----------------------
# LSTM expects 3D input
# (samples, timesteps, features)
# -----------------------
X_lstm = np.array(input_df).reshape(1, 1, input_df.shape[1])

# -----------------------
# Layout
# -----------------------

col1, col2, col3= st.columns(3)

# =======================
# Download the user data
# =======================

# Authorize
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=scope
)

client = gspread.authorize(creds)

# Open sheet
sheet = client.open("AllerSafe_User_Data").worksheet("data")


def save_user_input(season, temp, humidity, dust, smoke, outdoor_time, medication, symptoms, flare_risk):
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        season,
        temp,
        humidity,
        dust,
        smoke,
        outdoor_time,
        medication,
        symptoms,
        flare_risk
    ])


# =======================
# Flare-Up Prediction
# =======================
with col1:
    # -----------------------
    # Flare-up session state
    # -----------------------
    if "flare_prob" not in st.session_state:
        st.session_state.flare_prob = None

    # =======================
    # Flare-Up Prediction
    # =======================
    st.subheader("🔮 Flare-Up Prediction")

    if st.button("Predict Flare-Up"):
        with st.spinner("Predicting flare-up risk..."):
            st.session_state.flare_prob = float(
                lstm_model.predict(X_lstm)[0][0]
            )

    # -----------------------
    # Display result (persistent)
    # -----------------------
    if st.session_state.flare_prob is not None:
        prob = st.session_state.flare_prob

        st.metric(
            label="Flare-Up Probability",
            value=f"{prob * 100:.1f}%"
        )

        if prob >= 0.5:
            st.error("⚠️ High Risk of Allergy Flare-Up")
        else:
            st.success("✅ Low Risk of Flare-Up")
        user_data = {
            "Season":season,
            "Temperature": temperature,
            "Humidity": humidity,
            "Dust Level": dust_level,
            "Smoke Level": smoke_level,
            "Outdoor Time (minutes)": outdoor_time,
            "Medication Used": medication_used,
            "Current Symptom Score": symptom_score,
        }
        save_user_input(season, temperature, humidity, dust_level, smoke_level, outdoor_time, medication_used, symptom_score, prob)



# =======================
# Trigger Analysis
# =======================
with col3:
    st.subheader("🎯 Trigger Analysis")

    if st.button("Analyze Triggers"):
        importances = trigger_model.feature_importances_

        feature_names = trigger_model.feature_names_in_

        trigger_df = pd.DataFrame({
            "Trigger": feature_names,
            "Impact": importances
        }).sort_values("Impact", ascending=False)

        #st.bar_chart(trigger_df.set_index("Trigger"))

        chart = alt.Chart(trigger_df).mark_bar().encode(
            x=alt.X("Trigger:N", sort="-y"),
            y="Impact:Q",
            color=alt.Color(
                "Impact:Q",
                scale=alt.Scale(scheme="greens"),
                legend=None
            ),
        tooltip=["Trigger", "Impact"]
        )
        st.altair_chart(chart, width='stretch')

        st.markdown("### 🔥 Top Personal Triggers")
        for _, row in trigger_df.head(3).iterrows():
            st.write(f"• **{row['Trigger']}** (impact: {row['Impact']:.3f})")

# -----------------------
# Recommendations
# -----------------------

#st.subheader("🛡️ Personalized Guidance")
st.write(" ")
st.write(" ")
st.markdown("<h3 style='text-align: center;'> 🛡️ Personalized Guidance</h3>",
        unsafe_allow_html=True)

if dust_level > 25:
    st.warning("High dust exposure — consider wearing a mask outdoors.")

if smoke_level > 25:
    st.error("Smoke levels are high — reduce outdoor activities.")

if medication_used == 0 and symptom_score > 5:
    st.info("Medication may help reduce symptom severity today.")

if outdoor_time > 120:
    st.info("Long outdoor exposure detected — consider limiting time outside.")

st.write(" ")
st.markdown("<h3 style='text-align: center; color:green;'>   Predict. Prevent. Thrive.. 🌿</h3>",
        unsafe_allow_html=True)
