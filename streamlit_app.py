import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
# Added Plotly imports for advanced dual-axis plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================
# 🖥️ STREAMLIT CONFIG & PAGE INITIALIZATION
# ============================================
st.set_page_config(page_title="Slope FOS Real-Time Monitor", layout="wide", page_icon="⛰️")

st.title("⛰️ Real-Time Slope Monitoring & Early Warning System for Rainfall Induced Landslide")
st.write("A real-time slope stability monitoring system leveraging Artificial Neural Networks (ANN) and integrated Telegram alerts.")

# Initialize session state for tracking rainfall duration and history
if 'current_duration_hrs' not in st.session_state:
    st.session_state.current_duration_hrs = 0.0
if 'monitoring_history' not in st.session_state:
    st.session_state.monitoring_history = []

# ============================================
# 📥 PHASE 1: Load, Train & Cache ANN (Runs Once)
# ============================================
@st.cache_resource
def init_and_train_ann():
    try:
        # Load dataset directly from your GitHub repository
        csv_url = "https://raw.githubusercontent.com/ArifAzhar243/artificialneuralnetworkbyaa/refs/heads/master/ML%20OBJ2.csv"
        df = pd.read_csv(csv_url)
        df = df.drop(index=0, errors='ignore') 
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # Split features & target
        X = df_imputed.drop(columns=['FOS'])
        y = df_imputed['FOS']
        
        # Fit Scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
        
        # Build Model
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)
        
        return model, scaler, X.columns.tolist()
    except Exception as e:
        st.error(f"Critical error during ANN model training: {e}")
        return None, None, None

with st.spinner("🤖 Fetching data from GitHub & training ANN model... Please wait."):
    model, scaler, feature_columns = init_and_train_ann()

if model is not None:
    st.sidebar.success("✅ ANN Model & Scaler ready for use!")
else:
    st.error("❌ Failed to load the AI system. Please verify your internet connection or the GitHub URL.")
    st.stop()

# ============================================
# 🛠️ SIDEBAR: USER INPUT CONFIGURATIONS
# ============================================
st.sidebar.header("📋 Geotechnical Slope Parameters")
slope_cohesion = st.sidebar.number_input("Cohesion (kN/m²)", value=24.0, step=0.1)
slope_friction = st.sidebar.number_input("Friction Angle (°)", value=10.0, step=0.1)
slope_angle = st.sidebar.number_input("Slope Angle (°)", value=50.0, step=0.1)
slope_permeability = st.sidebar.number_input("Permeability (m/s)", value=0.0005, format="%.5f")

st.sidebar.markdown("---")
st.sidebar.header("🌍 API & System Configuration")
api_key = st.sidebar.text_input("WeatherAPI Key", value="941088bba63c4425a7062821260206")
lat_lon = st.sidebar.text_input("Location Coordinates (Lat,Lon)", value="1.4927,103.7414")

st.sidebar.markdown("---")
st.sidebar.header("🤖 Telegram Alert Integration")
telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password", placeholder="Token from BotFather")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", placeholder="Group ID / Personal ID")

st.sidebar.markdown("---")
st.sidebar.header("⏰ Real-Time Monitoring Settings")
time_unit = st.sidebar.selectbox("Monitoring Interval Unit", ["Seconds", "Hours"])
interval_value = st.sidebar.number_input("Interval Step Value", min_value=1, value=5 if "Seconds" in time_unit else 1)

# Convert monitoring interval to seconds for the processing loop sleep timer
sleep_seconds = interval_value if "Seconds" in time_unit else interval_value * 3600

st.sidebar.markdown("---")
st.sidebar.header("🧪 Simulation Mode (Testing)")
simulation_mode = st.sidebar.toggle("Enable Simulation Mode", value=False)
if simulation_mode:
    st.sidebar.warning("API bypassed. Using manual inputs below:")
    sim_intensity = st.sidebar.slider("Simulated Rainfall (mm/hr)", 0.0, 300.0, 20.0, step=1.0)
    sim_duration_hours = st.sidebar.slider("Simulated Duration (Hours)", 0.0, 168.0, 12.0, step=1.0)

# ============================================
# 🧮 HELPER FUNCTIONS (Classification & Alerts)
# ============================================
def get_fos_classification(fos):
    if fos > 1.3:
        return "Safe Zone", "green", "#28a745"
    elif 1.15 <= fos <= 1.3:
        return "Alert Zone", "yellow", "#ffc107"
    elif 1.00 <= fos < 1.15:
        return "Danger Zone", "orange", "#fd7e14"
    else:
        return "Failure Zone", "red", "#dc3545"

def send_telegram_alert(token, chat_id, fos, zone, intensity, duration, lat_lon):
    if not token or not chat_id:
        return False
    
    emoji_map = {"Safe Zone": "🟢", "Alert Zone": "🟡", "Danger Zone": "🟠", "Failure Zone": "🔴🚨"}
    emoji = emoji_map.get(zone, "⚠️")
    
    message = (
        f"{emoji} *SLOPE STABILITY ALERT* {emoji}\n\n"
        f"📍 *Coordinates:* {lat_lon}\n"
        f"📊 *FoS Value:* `{fos:.3f}`\n"
        f"🏷️ *Classification:* {zone}\n"
        f"🌧️ *Rainfall Intensity:* {intensity} mm/hr\n"
        f"⏳ *Accumulated Duration:* {duration:.2f} Hours\n\n"
        f"⚠️ _Please execute safety protocols and precautionary measures immediately!_"
    )
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        return res.status_code == 200
    except:
        return False

# ============================================
# 📊 MAIN DASHBOARD INTERFACE
# ============================================
start_monitoring = st.toggle("▶_ Activate Real-Time Monitoring Engine", value=False)

metric_container = st.empty()
chart_container = st.empty()
log_container = st.empty()

# ============================================
# 🔄 MONITORING LOOP RUNNER
# ============================================
while start_monitoring:
    try:
        if not simulation_mode:
            # 1. Fetch data from WeatherAPI (LIVE MODE)
            weather_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat_lon}"
            response = requests.get(weather_url, timeout=10).json()
            
            if 'current' not in response:
                with log_container:
                    st.error(f"❌ WeatherAPI Error: {response.get('error', {}).get('message', 'Connection issue or invalid API key')}")
                time.sleep(5)
                continue
                
            current_intensity_mm = response['current']['precip_mm']
            
            # Update Rainfall Duration dynamically
            if current_intensity_mm > 0.0:
                added_hours = (interval_value / 3600.0) if "Seconds" in time_unit else float(interval_value)
                st.session_state.current_duration_hrs += added_hours
            else:
                st.session_state.current_duration_hrs = 0.0
                
        else:
            # 1. Bypass API and pass manual simulation hours straight to the pipeline
            current_intensity_mm = float(sim_intensity)
            st.session_state.current_duration_hrs = float(sim_duration_hours)
            
        # 2. Restructure Live/Simulated Data for the ANN Pipeline
        live_data = pd.DataFrame({
            'Friction_Angle': [slope_friction],
            'Cohesion': [slope_cohesion],
            'Slope_Angle': [slope_angle],
            'Rainfall_Intensity': [current_intensity_mm],
            'Rainfall_Duration': [st.session_state.current_duration_hrs],
            'Permeability': [slope_permeability]
        })
        
        # Align columns explicitly with the structural layout the model was trained on
        live_data = live_data[feature_columns]
        
        # 3. Scale and Predict via ANN Model
        live_data_scaled = scaler.transform(live_data)
        predicted_fos_raw = model.predict(live_data_scaled, verbose=0)
        live_fos = float(predicted_fos_raw[0][0])
        
        # 4. Extract Zone Metrics and Classification Colors
        zone_name, bootstrap_color, hex_color = get_fos_classification(live_fos)
        
        # UPDATED: Logs combined Date & Time now
        timestamp_now = time.strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.monitoring_history.append({
            "DateTime": timestamp_now, 
            "FOS": round(live_fos, 3), 
            "Rainfall (mm/hr)": current_intensity_mm
        })
        if len(st.session_state.monitoring_history) > 30:
            st.session_state.monitoring_history.pop(0)
            
        # 5. Render Dashboard Display Layout Elements
        with metric_container.container():
            mode_badge = "🧪 SIMULATION MODE ACTIVE" if simulation_mode else "🌍 LIVE API MODE"
            st.markdown(f"### 📍 Monitored Location: `{lat_lon}` | Last Updated: `{timestamp_now}` | {mode_badge}")
            
            # CSS Injection to dynamically handle live alert banner status colors
            st.markdown(
                f"<div style='background-color:{hex_color}; padding:20px; border-radius:10px; text-align:center; color:white; font-weight:bold; font-size:24px;'>"
                f"STATUS: {zone_name.upper()} (FoS = {live_fos:.3f})"
                f"</div>", 
                unsafe_allow_html=True
            )
            st.write("")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="Factor of Safety (FoS)", value=f"{live_fos:.3f}")
            col2.metric(label="Rainfall Intensity", value=f"{current_intensity_mm} mm/hr")
            col3.metric(label="Accumulated Rain Duration", value=f"{st.session_state.current_duration_hrs:.2f} Hrs")
            col4.metric(label="Monitoring Step Rate", value=f"{interval_value} {time_unit}")
            
        # ============================================
        # 📊 6. UPDATED: ADVANCED DUAL Y-AXIS CHART
        # ============================================
        with chart_container.container():
            st.markdown("---")
            st.subheader("📈 Live Slope Stability & Weather Telemetry Trend Logs")
            
            history_df = pd.DataFrame(st.session_state.monitoring_history)
            
            if not history_df.empty:
                # Create a figure canvas with a secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Trace 1: Factor of Safety (Primary Left Y-Axis)
                fig.add_trace(
                    go.Scatter(
                        x=history_df['DateTime'], 
                        y=history_df['FOS'], 
                        name="Factor of Safety (FoS)", 
                        mode='lines+markers',
                        line=dict(color='#dc3545' if live_fos < 1.15 else '#28a745', width=3)
                    ),
                    secondary_y=False,
                )
                
                # Trace 2: Rainfall Intensity (Secondary Right Y-Axis)
                fig.add_trace(
                    go.Scatter(
                        x=history_df['DateTime'], 
                        y=history_df['Rainfall (mm/hr)'], 
                        name="Rainfall Intensity (mm/hr)", 
                        mode='lines+markers',
                        line=dict(color='#007bff', width=2, dash='dot')
                    ),
                    secondary_y=True,
                )
                
                # Set explicit labels for all axes
                fig.update_xaxes(title_text="<b>Date & Time of Data Taken</b>")
                fig.update_yaxes(title_text="<b>Factor of Safety (FoS)</b>", secondary_y=False)
                fig.update_yaxes(title_text="<b>Rainfall Intensity (mm/hr)</b>", secondary_y=True)
                
                # Layout formatting configurations
                fig.update_layout(
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=450
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        # 7. Trigger Telegram Alert Dispatch System (For Non-Safe Conditions)
        if zone_name != "Safe Zone" and telegram_token and telegram_chat_id:
            alert_success = send_telegram_alert(
                telegram_token, telegram_chat_id, live_fos, 
                zone_name, current_intensity_mm, st.session_state.current_duration_hrs, lat_lon
            )
            with log_container:
                if alert_success:
                    st.toast(f"🔔 Security alert successfully pushed to Telegram channel!", icon="⚠️")
                else:
                    st.toast(f"❌ Warning: Telegram alert delivery failure. Check bot credentials.", icon="🚨")

        time.sleep(sleep_seconds)
        st.rerun()

    except Exception as e:
        with log_container:
            st.error(f"⚠️ System interruption or loop error occurred: {e}")
        time.sleep(5)

# Render informative banner when engine status toggle is set to off
if not start_monitoring:
    st.info("ℹ️ Please configure your parameters on the sidebar panel and toggle 'Activate Real-Time Monitoring Engine' to launch live telemetry monitoring.")
