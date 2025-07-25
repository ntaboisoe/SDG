import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model = joblib.load("random_forest_model.pkl")

st.title("🔮 Predict Student Grade")

# ─────────────────────────────
# 📌 Single prediction section
# ─────────────────────────────
st.header("📌 Predict for One Student")
stress = st.selectbox("Stress Level", options=["Low", "Moderate", "High"])
stress_map = {"Low": 0, "Moderate": 1, "High": 2}
study_hours = st.slider("Study Hours Per Day", 0.0, 10.0, 2.0)
sleep_hours = st.slider("Sleep Hours Per Day", 0.0, 12.0, 6.0)

if st.button("Predict Grade"):
    X_input = pd.DataFrame([[study_hours, sleep_hours, stress_map[stress]]],
                           columns=["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "Stress_Level"])
    prediction = model.predict(X_input)[0]
    st.success(f"🎯 Predicted Grade: **{prediction:.2f}**")

# ─────────────────────────────
# 📁 Batch prediction section
# ─────────────────────────────
st.header("📁 Predict from CSV")


# Go up one level from /pages/ to project root, then into /data/
sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'lifestyle_data.csv'))


# Offer the sample file as download
with open(sample_path, 'rb') as f:
    st.download_button(
        label="📥 Download Sample CSV",
        data=f,
        file_name="lifestyle_data.csv",
        mime="text/csv"
    )

# File upload for batch prediction
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Map categorical stress levels to numeric
    df['Stress_Level'] = df['Stress_Level'].replace({"Low": 0, "Moderate": 1, "High": 2})

    # Use only relevant features
    features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Stress_Level']
    df = df[features]

    # Predict grades
    preds = model.predict(df)
    df['Predicted_Grade'] = preds

    # Show results
    st.dataframe(df)
    st.success("🎉 Predictions completed!")