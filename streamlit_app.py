import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ============================================
# Page config
# ============================================
st.set_page_config(
    page_title="ANN Rainfall Landslide Prediction",
    layout="wide"
)

st.title("🤖 Artificial Neural Network to Predict")
st.info(
    "This app trains an ANN model to predict Factor of Safety (FOS) "
    "based on rainfall and soil parameters."
)

# ============================================
# Step 1: Load Data
# ============================================
with st.expander("📂 Dataset"):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ArifAzhar243/artificialneuralnetworkbyaa/refs/heads/master/ML%20OBJ2.csv"
    )

    # Remove duplicate header row if exists
    df = df.drop(index=0, errors="ignore")

    st.write("Raw Data Preview")
    st.dataframe(df)
print("✅ Data Loaded Successfully")

# Handle Missing Values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Prepare Features & Target
X = df_imputed.drop(columns=['FOS'])
y = df_imputed['FOS']

# Feature scaling (CRUCIAL: We will reuse this 'scaler' object for live data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dataset split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Build ANN Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n🚀 Training ANN Model...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0) # set verbose=1 to see training logs
print("✅ Training Complete.")

import joblib

# Save ANN model
model.save("ann_model.h5")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Model saved!")
