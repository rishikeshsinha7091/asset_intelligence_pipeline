import pandas as pd

def build_feature_pipeline(input_path="raw_market_data.csv", output_path="processed_features.csv"):
    """Loads raw market arrays and engineers advanced rolling mathematical indicators."""
    print("Extracting clean asset data matrix...")
    df = pd.read_csv(input_path)
    
    # Ensure correct datetime ordering for time-series calculations
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    print("Engineering advanced mathematical features...")
    
    # 1. Trend Features: Rolling Window Moving Averages
    df["sma_fast"] = df["close"].rolling(window=6).mean()   # 6-Hour Short Trend
    df["sma_slow"] = df["close"].rolling(window=24).mean()  # 24-Hour Macro Trend
    
    # 2. Momentum Features: Price Distance Ratios
    # Calculates how far the current price has deviated from its moving trend line
    df["price_to_sma_fast"] = df["close"] / df["sma_fast"]
    df["price_to_sma_slow"] = df["close"] / df["sma_slow"]
    
    # 3. Volatility Features: 24-Hour Rolling Standard Deviation
    df["rolling_volatility_24h"] = df["close"].rolling(window=24).std()
    
    # 4. The Target Label: 1-Hour Future Return (What our ML model will predict)
    # .shift(-1) moves the data column backwards by 1 row, bringing the future into the current row context
    df["future_close_1h"] = df["close"].shift(-1)
    df["target_return_1h"] = (df["future_close_1h"] - df["close"]) / df["close"]
    
    # 5. Clean up boundary artifacts
    # Rolling windows require historical context. The first 23 rows won't have enough data to calculate a 24-hour average, leaving them blank (NaN).
    # The last row won't have a future price to look ahead to. We prune these incomplete records.
    initial_row_count = len(df)
    df = df.dropna()
    pruned_rows = initial_row_count - len(df)
    
    # Save the processed dataset to disk
    df.to_csv(output_path, index=False)
    print(f"Feature processing complete! Pruned {pruned_rows} boundary padding rows.")
    print(f"Engineered dataset securely saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    processed_df = build_feature_pipeline()
    
    print("\n=== INSTANT FEATURE MATRIX PREVIEW ===")
    feature_columns = ["timestamp", "close", "sma_fast", "price_to_sma_slow", "target_return_1h"]
    print(processed_df[feature_columns].head(5))