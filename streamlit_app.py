import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

st.set_page_config(page_title="ANN Rainfall Landslide Prediction", layout="wide")

# Title and description
st.title("🤖 Artificial Neural Network to Predict Rainfall Induced Landslide")
st.info("This app builds and trains an ANN model to predict Factor of Safety from rainfall and soil strength parameters. Created by Arif Azhar.")

# Step 1: Load Data
with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv('https://raw.githubusercontent.com/ArifAzhar243/artificialneuralnetworkbyaa/refs/heads/master/aa%20Machine%20Learning.csv')
  df


# Step 2: Correlation Heatmap
st.subheader("📊 Correlation Heatmap")
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Step 3: Preprocess Data
st.subheader("Data Preprocessing")
if 'FOS' not in df.columns:
    st.error("Dataset must contain a 'FOS' column.")
    st.stop()

X = df.drop(columns=['FOS'])
y = df['FOS']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

st.write(f"Training samples: {X_train.shape[0]}")
st.write(f"Validation samples: {X_val.shape[0]}")
st.write(f"Test samples: {X_test.shape[0]}")

# Step 4: Build ANN Model (fixed parameters)
st.subheader("🧠 ANN Model Training")
epochs = 100  # fixed
batch_size = 32  # fixed

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # regression output
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

with st.spinner("Training model..."):
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        verbose=0)

st.success("Model training complete.")

# Step 5: Model Evaluation
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
st.write(f"Test Loss (MSE): {test_loss:.4f}")
st.write(f"Test MAE: {test_mae:.4f}")

# Predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

st.write(f"RMSE: {rmse:.4f}")
st.write(f"R² Score: {r2:.4f}")

# Step 6: Actual vs Predicted Plot
st.subheader("📈 Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax1.set_xlabel('Actual Factor of Safety')
ax1.set_ylabel('Predicted Factor of Safety')
ax1.set_title('Actual vs Predicted Factor of Safety')
st.pyplot(fig1)


# Step 7: User Input Prediction
st.subheader("🔮 Predict Factor of Safety from New Parameters")

friction_angle = st.number_input("Friction Angle (°)", min_value=0, max_value=90, value=30, step=1, format="%d")
cohesion = st.number_input("Cohesion (kPa)", min_value=0, value=20, step=1, format="%d")
slope_angle = st.number_input("Slope Angle (°)", min_value=0, max_value=90, value=25, step=1, format="%d")
rainfall_intensity = st.number_input("Rainfall Intensity (mm/hr)", min_value=0, value=50, step=1, format="%d")
rainfall_duration = st.number_input("Rainfall Duration (hours)", min_value=0, value=5, step=1, format="%d")

if st.button("Predict FOS"):
    try:
        input_data = pd.DataFrame([[friction_angle, cohesion, slope_angle, rainfall_intensity, rainfall_duration]],
                                  columns=X.columns)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Factor of Safety: {prediction[0][0]:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
