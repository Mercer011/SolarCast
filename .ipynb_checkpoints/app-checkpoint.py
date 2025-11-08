# -------------------------------
# 🌞 DeepEnergyCast Full Weather Dashboard
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# -------------------------------
# ⚙️ Load trained model & scaler
# -------------------------------
@st.cache_resource
def load_assets():
    model = load_model("models/deepenergy_best.keras")   # use your best model
    scaler_data = joblib.load("models/scaler_final.pkl")
    return model, scaler_data

model, scaler_data = load_assets()
scaler = scaler_data["scaler"]
features = scaler_data["features"]

# -------------------------------
# 🌍 Page Setup
# -------------------------------
st.set_page_config(page_title="🌦️ DeepEnergyCast Dashboard", layout="wide")
st.title("⚡ DeepEnergyCast: Full Weather + Solar Energy Forecast")
st.markdown("""
Predict next-day **solar radiation** and view **complete weather forecast**  
powered by **NASA POWER + OpenWeatherMap APIs + Deep Learning (LSTM)**.
""")

# -------------------------------
# 🏙️ User Input
# -------------------------------
city = st.text_input("🏙️ Enter City Name:", "Bengaluru")

if st.button("🔍 Fetch & Predict"):
    try:
        API_KEY = "315f293daca93eaf2847d326cec66326"

        # -------------------------------
        # 🌦️ 1. Fetch Current Weather Data
        # -------------------------------
        owm_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(owm_url)
        data = response.json()

        if data.get("cod") != 200:
            st.error("❌ City not found. Please enter a valid city name.")
        else:
            lon, lat = data["coord"]["lon"], data["coord"]["lat"]

            # -------------------------------
            # ☀️ 2. Fetch NASA Data (past 7 days)
            # -------------------------------
            end_date = datetime.now()
            start_date = end_date - timedelta(days=89)
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")

            nasa_url = (
                f"https://power.larc.nasa.gov/api/temporal/daily/point?"
                f"parameters=ALLSKY_SFC_SW_DWN,WS10M,T2M,RH2M,PRECTOTCORR,ALLSKY_KT"
                f"&community=RE&longitude={lon}&latitude={lat}"
                f"&start={start_str}&end={end_str}&format=JSON"
            )
            nasa_res = requests.get(nasa_url).json()

            if "properties" not in nasa_res:
                st.error("⚠️ Failed to fetch NASA data.")
            else:
                records = nasa_res["properties"]["parameter"]
                df = pd.DataFrame({
                    "solar_radiation": list(records["ALLSKY_SFC_SW_DWN"].values()),
                    "wind_speed": list(records["WS10M"].values()),
                    "temperature": list(records["T2M"].values()),
                    "humidity": list(records.get("RH2M", {}).values()) if "RH2M" in records else [0]*7,
                    "rainfall": list(records.get("PRECTOTCORR", {}).values()) if "PRECTOTCORR" in records else [0]*7,
                    "clearsky_index": list(records.get("ALLSKY_KT", {}).values()) if "ALLSKY_KT" in records else [0]*7
                })

                # Temporal + static features
                df["pressure"] = data["main"]["pressure"]
                df["clouds"] = data["clouds"]["all"]
                df["month"] = end_date.month
                df["dayofyear"] = end_date.timetuple().tm_yday
                df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
                df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

                for col in features:
                    if col not in df.columns:
                        df[col] = 0

                df = df[features]

                # Scale + predict
                scaled = scaler.transform(df)
                X_input = np.expand_dims(scaled, axis=0)
                y_pred = model.predict(X_input)[0][0]

                # -------------------------------
                # 🌤️ 3. Fetch Weather Forecast (Next 3 days)
                # -------------------------------
                forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
                forecast_res = requests.get(forecast_url).json()

                if "list" in forecast_res:
                    forecast_list = []
                    for item in forecast_res["list"][:24:3]:  # next 3 days (8 intervals)
                        forecast_list.append({
                            "datetime": item["dt_txt"],
                            "temp": item["main"]["temp"],
                            "humidity": item["main"]["humidity"],
                            "pressure": item["main"]["pressure"],
                            "wind_speed": item["wind"]["speed"],
                            "clouds": item["clouds"]["all"],
                            "weather": item["weather"][0]["description"].title()
                        })
                    forecast_df = pd.DataFrame(forecast_list)
                else:
                    forecast_df = pd.DataFrame()

                # -------------------------------
                # 📊 4. Display Results
                # -------------------------------
                st.success(f"✅ Forecast for {city.title()} fetched successfully!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Solar Energy (kWh/m²/day)", f"{y_pred:.3f}")
                with col2:
                    st.metric("Current Temperature (°C)", f"{data['main']['temp']}")

                # Weather forecast table
                st.subheader("📅 Next 3-Day Weather Forecast")
                st.dataframe(forecast_df)

                # Visualization
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(df["solar_radiation"].values, color="orange", marker="o", label="Past 7 Days")
                ax[0].axhline(y=y_pred, color="r", linestyle="--", label="Predicted Next Day")
                ax[0].set_title("Solar Radiation Trend")
                ax[0].legend()
                ax[0].grid(alpha=0.3)

                if not forecast_df.empty:
                    ax[1].plot(forecast_df["datetime"], forecast_df["temp"], color="skyblue", marker="o", label="Temperature")
                    ax[1].set_title("Next 3-Day Temperature Forecast")
                    ax[1].tick_params(axis='x', rotation=45)
                    ax[1].legend()
                    ax[1].grid(alpha=0.3)

                st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
