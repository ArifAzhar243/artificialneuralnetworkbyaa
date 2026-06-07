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

# ============================================
# Step 2: Handle Missing Values
# ============================================
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# ============================================
# Step 3: Correlation Heatmap
# ============================================
st.subheader("📊 Correlation Heatmap")
correlation_matrix = df_imputed.corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)
st.pyplot(fig)

# ============================================
# Step 4: Prepare Features & Target
# ============================================
if "FOS" not in df_imputed.columns:
    st.error("Dataset must contain 'FOS' column.")
    st.stop()

X = df_imputed.drop(columns=["FOS"])
y = df_imputed["FOS"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Validation / Test split (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)


# ============================================
# Step 5: Build ANN Model
# ============================================
st.subheader("🧠 ANN Model Training")

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

with st.spinner("Training model..."):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=0
    )

st.success("Model training completed.")

# ============================================
# Step 6: Model Evaluation
# ============================================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"MSE  : {mse:.4f}")
st.write(f"RMSE : {rmse:.4f}")
st.write(f"R²   : {r2:.4f}")

# Actual vs Predicted
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
ax2.set_xlabel("Actual Factor of Safety")
ax2.set_ylabel("Predicted Factor of Safety")
ax2.set_title("Actual vs Predicted Factor of Safety")
ax2.grid(True, linestyle="--", alpha=0.6)
st.pyplot(fig2)

# ============================================
# Step 7: User Input Prediction
# ============================================
st.subheader("🔮 Predict Factor of Safety from New Parameters")

friction_angle = st.number_input("Friction Angle (°)", 0, 90, 30)
cohesion = st.number_input("Cohesion (kPa)", 0, 20)
slope_angle = st.number_input("Slope Angle (°)", 0, 90, 25)
rainfall_intensity = st.number_input("Rainfall Intensity (mm/hr)", 0, 50)
rainfall_duration = st.number_input("Rainfall Duration (hours)", 0, 5)
permeability = st.number_input(
    "Permeability (m/s)",
    min_value=0.0,
    value=1e-5,
    format="%.6e"
)

if st.button("Predict FOS"):
    try:
        input_data = pd.DataFrame([[
            friction_angle,
            cohesion,
            slope_angle,
            rainfall_intensity,
            rainfall_duration,
            permeability
        ]], columns=X.columns)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.success(
            f"Predicted Factor of Safety (FOS): {prediction[0][0]:.4f}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
