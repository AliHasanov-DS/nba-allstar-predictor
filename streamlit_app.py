import streamlit as st
import pandas as pd
import joblib
import os
st.set_page_config(page_title="NBA All-Star AI Predictor", page_icon="🏀")

st.title("🏀 NBA All-Star Performance Predictor")
st.markdown("Enter player statistics below to analyze their **All-Star potential** based on historical NBA data.")
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'nba_xgboost_real_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_model()
if model is None:
    st.error("Model file not found! Please ensure 'nba_xgboost_real_model.joblib' is in the project folder.")
    st.stop()
st.sidebar.header("📊 Player Statistics")
points = st.sidebar.slider("Points", 0, 100, 25)
assists = st.sidebar.slider("Assists", 0, 30, 5)
rebounds = st.sidebar.slider("Rebounds", 0, 40, 10)
blocks = st.sidebar.slider("Blocks", 0, 15, 1)
steals = st.sidebar.slider("Steals", 0, 10, 1)
turnovers = st.sidebar.slider("Turnovers", 0, 15, 2)
plus_minus = st.sidebar.slider("Plus/Minus (+/-)", -50, 50, 5)

st.sidebar.header("🧍 Physical Attributes")
height = st.sidebar.slider("Height (Inches)", 65, 95, 78)
weight = st.sidebar.slider("Weight (Lbs)", 150, 350, 220)
position = st.sidebar.selectbox("Primary Position", ["Guard", "Forward", "Center"])

st.write("---")

if st.button("🔮 Predict Result", use_container_width=True):

    input_data = pd.DataFrame({
        'points': [points],
        'assists': [assists],
        'blocks': [blocks],
        'steals': [steals],
        'reboundsTotal': [rebounds],
        'turnovers': [turnovers],
        'plusMinusPoints': [plus_minus],
        'heightInches': [height],
        'bodyWeightLbs': [weight],
        'guard': [1 if position == "Guard" else 0],
        'forward': [1 if position == "Forward" else 0],
        'center': [1 if position == "Center" else 0]
    })
    prob = model.predict_proba(input_data)[0][1]
    col1, col2 = st.columns(2)
    col1.metric("AI Probability Score", f"{prob * 100:.2f}%")

    threshold = 0.25

    if prob >= threshold:
        if prob >= 0.40:
            st.balloons()
    else:
        col2.error("❌ STANDARD PERFORMANCE")

    st.info(f"The model compares these stats against over 30 years of NBA historical data. "
            f"A score above {threshold * 100}% indicates elite, All-Star caliber play.")