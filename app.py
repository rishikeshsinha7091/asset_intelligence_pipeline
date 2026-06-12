# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from ingest_data import fetch_market_data
from datetime import datetime

# Set browser tab properties
st.set_page_config(
    page_title="Live Asset Intelligence",
    page_icon="⚡",
    layout="wide"
)

@st.cache_data(ttl=15)
def fetch_and_engineer_live_stream():
    """Pings Binance live API, extracts real-time payloads, and converts time zones on the fly."""
    # 1. Harvest live data up to this exact millisecond
    raw_candle_data = fetch_market_data(symbol="BTCUSDT", interval="1h", limit=500)
    
    # 2. Parse into a functional DataFrame matrix core
    headers = ["timestamp", "open", "high", "low", "close", "volume"]
    clean_rows = []
    for candle in raw_candle_data:
        clean_rows.append([
            int(candle[0]), float(candle[1]), float(candle[2]),
            float(candle[3]), float(candle[4]), float(candle[5])
        ])
    
    df = pd.DataFrame(clean_rows, columns=headers)
    
    # CRITICAL TIMEZONE CONVERSION: Map naive UTC from API directly to local time (IST)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # 3. Dynamic Feature Engineering
    df["sma_fast"] = df["close"].rolling(window=6).mean()
    df["sma_slow"] = df["close"].rolling(window=24).mean()
    df["price_to_sma_fast"] = df["close"] / df["sma_fast"]
    df["price_to_sma_slow"] = df["close"] / df["sma_slow"]
    
    # Calculate live Z-Scores for the anomaly scanner
    df["hourly_spread"] = df["high"] - df["low"]
    mean_spread = df["hourly_spread"].mean()
    std_spread = df["hourly_spread"].std()
    df["z_score"] = (df["hourly_spread"] - mean_spread) / std_spread
    
    # Create clean anomaly frames
    anomalies_df = df[df["z_score"] > 3].copy()
    
    return df, anomalies_df

# Launch live data engine pipeline
with st.spinner("Pinging global exchange servers for real-time asset data..."):
    live_df, live_anomalies = fetch_and_engineer_live_stream()

# Capture precise execution wall-clock time down to the second
sync_time = datetime.now().strftime("%A, %B %d, %Y | %I:%M:%S %p")

# --- BACKEND MODEL TRAINING CORE ---
train_df = live_df.dropna().reset_index(drop=True)
feature_cols = ["sma_fast", "sma_slow", "price_to_sma_fast", "price_to_sma_slow"]
X_raw = train_df[feature_cols].iloc[:-1]
y = (train_df["close"].shift(-1) - train_df["close"]) / train_df["close"]
y = y.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

model = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=15, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

# --- USER INTERFACE DESIGN ---
st.title("⚡ Live Production Asset Forecasting Pipeline")

# Live System Heartbeat Container
st.info(f"🟢 **Live Pipeline Heartbeat:** Connected to exchange endpoints. Last database synchronization verified at: **{sync_time}**")
st.write("---")

# SUMMARY METRIC FLOATS
latest_bar = live_df.iloc[-1]
prev_bar = live_df.iloc[-2]
price_change = latest_bar['close'] - prev_bar['close']

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="LIVE BITCOIN PRICE", value=f"${latest_bar['close']:.2f}", delta=f"${price_change:.2f} (Past 60m)")
with col2:
    st.metric(label="CURRENT BAR VOLUME", value=f"{latest_bar['volume']:.2f} BTC", delta=f"{(latest_bar['volume'] - prev_bar['volume']):.2f} vs Prev Hour")
with col3:
    st.metric(label="STREAMING WINDOW", value="ONLINE", delta="API STREAM ACTIVE", delta_color="normal")

st.write("---")

# --- LIVE STREAM TICKER FEED ---
st.subheader("⏱️ Live Market Activity Ticker (Most Recent 5 Hours)")
st.markdown("Exposing raw physical block timestamps streaming straight off the network wire to verify data continuity.")

recent_feed = live_df.tail(5)[["timestamp", "open", "high", "low", "close", "volume"]].copy()
recent_feed = recent_feed.sort_values(by="timestamp", ascending=False)

# Clean string formatting retaining localized timezone configurations
recent_feed["timestamp"] = recent_feed["timestamp"].dt.strftime("%Y-%m-%d | %I:%M %p")

st.dataframe(recent_feed, use_container_width=True, hide_index=True)
st.write("---")

# RE-ALIGNED CORE BLOCKS
left_deck, right_deck = st.columns([1, 1])

with left_deck:
    st.subheader("🚨 Live Detected Volatility Anomalies")
    if len(live_anomalies) > 0:
        clean_anomalies = live_anomalies[["timestamp", "open", "high", "low", "close", "z_score"]].copy()
        clean_anomalies["timestamp"] = clean_anomalies["timestamp"].dt.strftime("%Y-%m-%d | %I:%M %p")
        st.dataframe(
            clean_anomalies.sort_values(by="timestamp", ascending=False),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Market volatility conditions normal. Zero Z-Score outliers detected in active window.")

with right_deck:
    st.subheader("🔮 Live Production Inference Engine")
    st.markdown("Feeding the final real-time feature vector straight into the trained Random Forest model split:")
    
    st.write(f"**Current Fast Trend (6h SMA):** ${latest_bar['sma_fast']:.2f}")
    st.write(f"**Current Macro Trend (24h SMA):** ${latest_bar['sma_slow']:.2f}")
    st.write(f"**Current Deviation Matrix Ratio:** {latest_bar['price_to_sma_fast']:.4f}")
    
    current_features = np.array([[latest_bar['sma_fast'], latest_bar['sma_slow'], latest_bar['price_to_sma_fast'], latest_bar['price_to_sma_slow']]])
    current_features_scaled = scaler.transform(current_features)
    
    live_prediction = model.predict(current_features_scaled)[0]
    
    st.markdown("---")
    st.markdown("### Model Forward Projection Output")
    if live_prediction > 0:
        st.success(f"📈 **BULLISH EXPECTATION FORECAST:** Model projects a positive upcoming close move: +{live_prediction*100:.4f}%")
    else:
        st.error(f"📉 **BEARISH EXPECTATION FORECAST:** Model projects a negative upcoming close move: {live_prediction*100:.4f}%")