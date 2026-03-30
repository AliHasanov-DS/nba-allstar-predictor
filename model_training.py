import pandas as pd
import joblib
import streamlit as st
try:
    model = joblib.load('nba_xgboost_model.joblib')
except FileNotFoundError:
    st.error("Error: Model file not found. Please ensure 'nba_xgboost_model.joblib' exists.")
    exit()
new_players = pd.DataFrame({
    'points': [45, 2],
    'assists': [12, 0],
    'blocks': [3, 0],
    'steals': [4, 0],
    'reboundsTotal': [15, 1],
    'turnovers': [2, 1],
    'plusMinusPoints': [25, -12],
    'heightInches': [81, 75],
    'bodyWeightLbs': [240, 190],
    'guard': [0, 1],
    'forward': [1, 0],
    'center': [0, 0]
})
probabilities = model.predict_proba(new_players)[:, 1]
player_names = ["MVP Candidate", "Standard Player"]

threshold = 0.25

for i in range(len(player_names)):
    prob = probabilities[i]
    if prob >= threshold:
        status = "ALL-STAR LEVEL PERFORMANCE"
        st.success(f"⭐ {player_names[i]}: {status}")
    else:
        status = "STANDARD PERFORMANCE"
        st.error(f"❌ {player_names[i]}: {status}")
    print(f"Player Profile: {player_names[i]}")
    print(f"AI Analysis Score: {prob * 100:.2f}%")
    print(f"Final Result: {status}\n")