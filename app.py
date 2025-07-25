import streamlit as st

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.success("Select a page above")

st.title("🎓 Student Lifestyle & Academic Performance App")
st.markdown("""
Welcome! This app explores how student lifestyle factors influence academic performance using machine learning.

👉 Use the sidebar to navigate:
- **Home**: Project overview  
- **Data Exploration**: See the code & analysis  
- **Prediction**: Try the predictor yourself!
""")
