import streamlit as st
import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Title
st.title("📊 Data Exploration & Modeling")

# --- SECTION 1: Importing and Loading Dataset ---
st.header("1. Load and Display Data")

code_block_1 = '''
# Download latest version
path = kagglehub.dataset_download("charlottebennett1234/lifestyle-factors-and-their-impact-on-students")
csv_file_name = 'student_lifestyle_dataset..csv'
full_csv_path = os.path.join(path, csv_file_name)

# Read the CSV
df = pd.read_csv(full_csv_path)
'''
st.code(code_block_1, language='python')

# # Execute actual code
# path = kagglehub.dataset_download("charlottebennett1234/lifestyle-factors-and-their-impact-on-students")
# csv_file_name = 'student_lifestyle_dataset..csv'
# full_csv_path = os.path.join(path, csv_file_name)

# try:
#     df = pd.read_csv(full_csv_path)
#     st.success("Data loaded successfully.")
#     st.dataframe(df.head())
# except Exception as e:
#     st.error(f"Error loading data: {e}")

df = pd.read_csv('data/sample_student_data.csv')

# --- SECTION 2: Display Column Types ---
st.header("2. Data Types and Info")

code_block_2 = '''
# Display columns data types
df.info()
'''
st.code(code_block_2, language='python')

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.subheader("DataFrame Info")
st.text(s)


# --- SECTION 3: Preprocess ---
st.header("3. Preprocessing")

code_block_3 = '''
df.drop('Student_ID', axis=1, inplace=True)
df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
'''
st.code(code_block_3, language='python')

df.drop('Student_ID', axis=1, inplace=True)
df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
st.success("Preprocessing complete.")

# --- SECTION 4: Correlation Matrix ---
st.header("4. Correlation Matrix Heatmap")

code_block_4 = '''
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Student Lifestyle Factors and Grades')
plt.show()
'''
st.code(code_block_4, language='python')

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Student Lifestyle Factors and Grades')
st.pyplot(fig=plt.gcf())

# --- SECTION 5: Histograms ---
st.header("5. Histograms of Numerical Features")

code_block_5 = '''
df.hist(bins=20, figsize=(14, 10))
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
'''
st.code(code_block_5, language='python')

df.hist(bins=20, figsize=(14, 10))
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(fig=plt.gcf())

# --- SECTION 6: Scatter Plot ---
st.header("6. Study Hours vs. Grades")

code_block_6 = '''
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Study_Hours_Per_Day', y='Grades', data=df)
plt.title('Study Hours per Day vs. Grades')
plt.xlabel('Study Hours per Day')
plt.ylabel('Grades')
plt.grid(True)
plt.show()
'''
st.code(code_block_6, language='python')

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Study_Hours_Per_Day', y='Grades', data=df)
plt.title('Study Hours per Day vs. Grades')
plt.xlabel('Study Hours per Day')
plt.ylabel('Grades')
plt.grid(True)
st.pyplot(fig=plt.gcf())

# --- SECTION 7: Modeling ---
st.header("7. Model Training and Evaluation")

code_block_7 = '''
features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Stress_Level']
target = 'Grades'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

# Evaluation
rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

xgb_r2 = r2_score(y_test, xgb_predictions)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
'''
st.code(code_block_7, language='python')

# Execute code
features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Stress_Level']
target = 'Grades'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

st.subheader("📈 Model Performance")
st.write("**Random Forest**")
st.write(f"R² Score: {rf_r2:.4f} | MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f}")
st.write("**XGBoost**")
st.write(f"R² Score: {xgb_r2:.4f} | MAE: {xgb_mae:.2f} | RMSE: {xgb_rmse:.2f}")

# --- SECTION 8: Feature Importance ---
st.header("8. Feature Importance Comparison")

code_block_8 = '''
rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_

rf_importance_df = pd.DataFrame({'Feature': features, 'Importance': rf_importances}).sort_values(by='Importance', ascending=False)
xgb_importance_df = pd.DataFrame({'Feature': features, 'Importance': xgb_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Importance', y='Feature', data=rf_importance_df)
plt.title('Feature Importance - Random Forest')
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=xgb_importance_df)
plt.title('Feature Importance - XGBoost')
plt.tight_layout()
plt.show()
'''
st.code(code_block_8, language='python')

rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_
rf_importance_df = pd.DataFrame({'Feature': features, 'Importance': rf_importances}).sort_values(by='Importance', ascending=False)
xgb_importance_df = pd.DataFrame({'Feature': features, 'Importance': xgb_importances}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Importance', y='Feature', data=rf_importance_df)
plt.title('Feature Importance - Random Forest')
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=xgb_importance_df)
plt.title('Feature Importance - XGBoost')
plt.tight_layout()
st.pyplot(fig=plt.gcf())
