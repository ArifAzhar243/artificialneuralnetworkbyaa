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

# ============================================
# 🖥️ STREAMLIT CONFIG & PAGE INITIALIZATION
# ============================================
st.set_page_config(page_title="Slope FOS Real-Time Monitor", layout="wide", page_icon="⛰️")

st.title("⛰️ Real-Time Slope Stability & FoS Monitoring System")
st.write("Sistem pemantauan kestabilan cerun masa nyata menggunakan kecerdasan buatan (ANN) dan integrasi amaran Telegram.")

# Initialize session state untuk menjejaki hujan
if 'current_duration_hrs' not in st.session_state:
    st.session_state.current_duration_hrs = 0.0
if 'monitoring_history' not in st.session_state:
    st.session_state.monitoring_history = []

# ============================================
# 📥 PHASE 1: Load, Train & Cache ANN (Satu Kali Sahaja)
# ============================================
@st.cache_resource
def init_and_train_ann():
    try:
        # Load dataset asli
        df = pd.read_excel("ML OBJ2.xlsx")
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
        st.error(f"Ralat kritikal semasa latihan model ANN: {e}")
        return None, None, None

with st.spinner("🤖 Sedang melatih model ANN di latar belakang... Sila tunggu sebentar."):
    model, scaler, feature_columns = init_and_train_ann()

if model is not None:
    st.sidebar.success("✅ Model ANN & Scaler sedia untuk digunakan!")
else:
    st.error("❌ Gagal memuatkan sistem kecerdasan buatan. Sila semak fail 'ML OBJ2.xlsx' anda.")
    st.stop()

# ============================================
# 🛠️ SIDEBAR: USER INPUT CONFIGURATIONS
# ============================================
st.sidebar.header("📋 Parameter Geoteknik Cerun")
slope_cohesion = st.sidebar.number_input("Cohesion (kN/m²)", value=24.0, step=0.1)
slope_friction = st.sidebar.number_input("Friction Angle (°)", value=10.0, step=0.1)
slope_angle = st.sidebar.number_input("Slope Angle (°)", value=50.0, step=0.1)
slope_permeability = st.sidebar.number_input("Permeability (m/s)", value=0.0005, format="%.5f")

st.sidebar.markdown("---")
st.sidebar.header("🌍 API & Konfigurasi Sistem")
api_key = st.sidebar.text_input("WeatherAPI Key", value="941088bba63c4425a7062821260206")
lat_lon = st.sidebar.text_input("Koordinat Lokasi (Lat,Lon)", value="1.4927,103.7414")

st.sidebar.markdown("---")
st.sidebar.header("🤖 Integrasi Amaran Telegram")
telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password", placeholder="Token dari BotFather")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", placeholder="ID Group / ID Personal")

st.sidebar.markdown("---")
st.sidebar.header("⏰ Tetapan Pemantauan Masa Nyata")
time_unit = st.sidebar.selectbox("Unit Masa Selang", ["Seconds (Saat)", "Hours (Jam)"])
interval_value = st.sidebar.number_input("Kadar Kemaskini (Sela Masa)", min_value=1, value=5 if "Seconds" in time_unit else 1)

# Tukar sela masa ke unit saat untuk script sleep loop
sleep_seconds = interval_value if "Seconds" in time_unit else interval_value * 3600

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
        f"{emoji} *AMARAN KESTABILAN CERUN* {emoji}\n\n"
        f"📍 *Koordinat:* {lat_lon}\n"
        f"📊 *Nilai FoS:* `{fos:.3f}`\n"
        f"🏷️ *Klasifikasi:* {zone}\n"
        f"🌧️ *Intensiti Hujan:* {intensity} mm/hr\n"
        f"⏳ *Durasi Hujan:* {duration:.2f} Jam\n\n"
        f"⚠️ _Sila ambil tindakan keselamatan dan langkah berjaga-jaga segera!_"
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
# Layout untuk status semasa pemantauan
start_monitoring = st.toggle("▶️ Aktifkan Enjin Pemantauan Masa Nyata", value=False)

# Placeholders dinamik supaya UI tidak berkelip (flicker) semasa refresh loop
metric_container = st.empty()
chart_container = st.empty()
log_container = st.empty()

# Aturan susunan lajur data mengikut keperluan model ANN asal anda
# Urutan asal: 'Friction_Angle', 'Cohesion', 'Slope_Angle', 'Rainfall_Intensity', 'Rainfall_Duration', 'Permeability'
# Sila pastikan urutan ini match 100% dengan susunan kolum dalam dataframe latihan (X.columns)

# ============================================
# 🔄 MONITORING LOOP RUNNER
# ============================================
while start_monitoring:
    try:
        # 1. Hubungi WeatherAPI
        weather_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat_lon}"
        response = requests.get(weather_url, timeout=10).json()
        
        if 'current' not in response:
            with log_container:
                st.error(f"❌ Ralat WeatherAPI: {response.get('error', {}).get('message', 'Masalah sambungan API')}")
            time.sleep(5)
            continue
            
        current_intensity_mm = response['current']['precip_mm']
        
        # 2. Kemaskini Durasi Hujan (Guna jam sebagai unit perkiraan model)
        if current_intensity_mm > 0.0:
            added_hours = (interval_value / 3600.0) if "Seconds" in time_unit else float(interval_value)
            st.session_state.current_duration_hrs += added_hours
        else:
            st.session_state.current_duration_hrs = 0.0
            
        # 3. Penyediaan data input live untuk disuapkan ke ANN
        live_data = pd.DataFrame({
            'Friction_Angle': [slope_friction],
            'Cohesion': [slope_cohesion],
            'Slope_Angle': [slope_angle],
            'Rainfall_Intensity': [current_intensity_mm],
            'Rainfall_Duration': [st.session_state.current_duration_hrs],
            'Permeability': [slope_permeability]
        })
        
        # Susun semula mengikut susunan eksak model dilatih
        live_data = live_data[feature_columns]
        
        # 4. Penskalaan Data (Scaling) & Ramalan ANN
        live_data_scaled = scaler.transform(live_data)
        predicted_fos_raw = model.predict(live_data_scaled, verbose=0)
        live_fos = float(predicted_fos_raw[0][0])
        
        # 5. Dapatkan Klasifikasi Zon & Warna
        zone_name, bootstrap_color, hex_color = get_fos_classification(live_fos)
        
        # Simpan dalam history untuk graf trend
        timestamp_now = time.strftime('%H:%M:%S')
        st.session_state.monitoring_history.append({
            "Masa": timestamp_now, 
            "FOS": round(live_fos, 3), 
            "Rainfall (mm)": current_intensity_mm
        })
        # Hadkan simpanan data graf setakat 20 rekod terakhir sahaja
        if len(st.session_state.monitoring_history) > 20:
            st.session_state.monitoring_history.pop(0)
            
        # 6. Reka bentuk Paparan Widget Metrik Utama (Dynamic Update)
        with metric_container.container():
            st.markdown(f"### 📍 Lokasi Pantauan: `{lat_lon}` | Terakhir Diperbaharui: `{timestamp_now}`")
            
            # Highlight Kotak Status Utama Menggunakan Warna Bootstrap Streamlit
            st.markdown(
                f"<div style='background-color:{hex_color}; padding:20px; border-radius:10px; text-align:center; color:white; font-weight:bold; font-size:24px;'>"
                f"STATUS: {zone_name.upper()} (FoS = {live_fos:.3f})"
                f"</div>", 
                unsafe_style_allowed=True
            )
            st.write("")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="Factor of Safety (FoS)", value=f"{live_fos:.3f}")
            col2.metric(label="Intensiti Hujan", value=f"{current_intensity_mm} mm/hr")
            col3.metric(label="Durasi Hujan Mengumpul", value=f"{st.session_state.current_duration_hrs:.2f} Jam")
            col4.metric(label="Sela Masa Semakan", value=f"{interval_value} {time_unit.split()[0]}")
            
        # 7. Paparan Graf Trend
        with chart_container.container():
            st.markdown("---")
            st.subheader("📈 Graf Trend Kestabilan Cerun (Live)")
            history_df = pd.DataFrame(st.session_state.monitoring_history)
            if not history_df.empty:
                st.line_chart(history_df.set_index("Masa")[["FOS", "Rainfall (mm)"]])
                
        # 8. Pemicu Amaran Telegram (Hantar jika zon bukan Safe Zone)
        if zone_name != "Safe Zone" and telegram_token and telegram_chat_id:
            alert_success = send_telegram_alert(
                telegram_token, telegram_chat_id, live_fos, 
                zone_name, current_intensity_mm, st.session_state.current_duration_hrs, lat_lon
            )
            with log_container:
                if alert_success:
                    st.toast(f"🔔 Isyarat amaran Telegram berjaya dihantar ke sistem!", icon="⚠️")
                else:
                    st.toast(f"❌ Gagal menghantar isyarat Telegram. Sila periksa token/ID.", icon="🚨")

        # Rehat mengikut sela masa yang ditetapkan oleh user
        time.sleep(sleep_seconds)
        st.rerun()

    except Exception as e:
        with log_container:
            st.error(f"⚠️ Gangguan sistem atau ralat gelung: {e}")
        time.sleep(5)

# Paparan jika pemantauan dimatikan (Off)
if not start_monitoring:
    st.info("ℹ️ Sila konfigurasikan parameter di bar sisi (sidebar) dan petik suis 'Aktifkan Enjin Pemantauan' untuk memulakan sistem.")
