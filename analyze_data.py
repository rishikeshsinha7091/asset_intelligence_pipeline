import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# 1. Load data and calculate metrics
df = pd.read_csv("raw_market_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df["hourly_spread"] = df["high"] - df["low"]
df["hourly_return"] = df["close"] - df["open"]

mean_spread = df["hourly_spread"].mean()
std_spread = df["hourly_spread"].std()
df["spread_z_score"] = (df["hourly_spread"] - mean_spread) / std_spread

# Filter out the extreme outliers
anomaly_threshold = 3
anomalies = df[df["spread_z_score"] > anomaly_threshold]

# 2. Initialize the Canvas
plt.figure(figsize=(14, 7))

# Layer 1: Plot the baseline continuous volatility line
plt.plot(df["timestamp"], df["hourly_spread"], color="royalblue", alpha=0.6, label="Hourly Price Spread", zorder=1)

# Layer 2: Draw the mathematical threshold cutoff line (Mean + 3 * StdDev)
cutoff_value = mean_spread + (anomaly_threshold * std_spread)
plt.axhline(y=cutoff_value, color="darkorange", linestyle="--", linewidth=1.5, label="Anomaly Threshold (Z = 3)", zorder=2)

# Layer 3: Overlay the high-signal anomaly events as prominent red dots
plt.scatter(anomalies["timestamp"], anomalies["hourly_spread"], color="crimson", edgecolors="black", s=50, label="Detected Anomalies", zorder=3)

# 3. Canvas Formatting and Styling
plt.title("Bitcoin High-Frequency Volatility & Statistical Anomaly Engine", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Timeline (June 2026 Window)", fontsize=11, labelpad=10)
plt.ylabel("Hourly Price Variance (High - Low) in USD", fontsize=11, labelpad=10)
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

# 4. Bulletproof Export System
# Instead of launching a pop-up window that can crash, we save the image directly to disk
output_image_path = "market_anomalies.png"
plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
print(f"Visual engine execution complete. Analytical plot saved to: {output_image_path}")