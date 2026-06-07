from tensorflow.keras.models import load_model
import joblib

model = load_model("ann_model.h5")
scaler = joblib.load("scaler.pkl")

print("SUCCESS")
