import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

# ---------------------- STREAMLIT PAGE CONFIG ----------------------
st.set_page_config(page_title="BNB/USD Forecast", layout="wide")
st.title("🔮 BNB/USD Price Prediction with Prophet")

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    # Load CSV dataset
    df = pd.read_csv("BNBUSDT (1).csv")  # Make sure this file is in same folder
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert to datetime
    return df

df = load_data()

st.subheader("📈 Historical Data (BNB/USD)")
st.dataframe(df.tail())

# ---------------------- HANDLE LARGE DATA ----------------------
if len(df) > 500000:  # If too large, keep only last 200k rows
    st.warning("⚠️ Large dataset detected, using last 200,000 rows for training.")
    df = df.tail(200000)

# ---------------------- PREPARE DATA FOR PROPHET ----------------------
df_train = df[['timestamp', 'close']].rename(columns={"timestamp": "ds", "close": "y"})

# ---------------------- USER INPUT FOR FORECAST ----------------------
n_days = st.slider("Forecast days:", min_value=7, max_value=180, value=30)

# ---------------------- TRAIN MODEL ----------------------
@st.cache_resource
def train_model(data):
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    return model

model = train_model(df_train)

# ---------------------- MAKE FUTURE PREDICTIONS ----------------------
future = model.make_future_dataframe(periods=n_days)
forecast = model.predict(future)

# ---------------------- DISPLAY FORECAST ----------------------
st.subheader("🔮 Forecasted Prices")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Data (last 10 days)")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

st.success("✅ Prediction completed successfully!")
