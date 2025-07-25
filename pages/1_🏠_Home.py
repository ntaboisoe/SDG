import streamlit as st

st.title("🏠 Project Overview")

st.markdown("""
### 🎯 Objective:
Predict student grades based on lifestyle factors using machine learning.

### ❓ Key Questions:
- What lifestyle factors affect academic performance?
- Which model performs best?
- How can we use this for real-time prediction?

### 📊 Key Findings:
- **Study Hours** had the strongest impact on grades.
- **Sleep Hours** and **Stress Level** were also critical.
- **Random Forest** outperformed XGBoost (R² = 0.4066).

### 🤖 Models Used:
- RandomForestRegressor ✅
- XGBoostRegressor

### 📈 Next Steps:
- Collect more nuanced behavioral data
- Add interpretability (e.g., SHAP values)
""")