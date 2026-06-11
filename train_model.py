import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def execute_model_training(input_path="processed_features.csv"):
    """Loads engineered feature datasets, splits them chronologically, trains a Random Forest, and audits its brain."""
    print("Loading engineered feature matrices...")
    df = pd.read_csv(input_path)
    
    # 1. Isolate inputs (X) from output targets (y)
    feature_cols = ["sma_fast", "sma_slow", "price_to_sma_fast", "price_to_sma_slow", "rolling_volatility_24h"]
    X = df[feature_cols]
    y = df["target_return_1h"]
    
    # 2. Chronological Split (80% Train / 20% Test to prevent data leakage)
    split_index = int(len(df) * 0.80)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print("Dataset carved successfully.")
    print(f"Training Allocation: {X_train.shape[0]} records | Forward Validation Windows: {X_test.shape[0]} records")
    
    # 3. Model Architecture Initialization (Leveraging multi-core processing)
    print("\nInitializing Random Forest Regressor and spinning up decision trees...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 4. Training Execution
    print("Fitting model parameters against historical patterns...")
    model.fit(X_train, y_train)
    print("Model optimization complete.")
    
    # 5. Prediction Inference
    print("\nRunning predictive inference across the forward test window...")
    predictions = model.predict(X_test)
    
    # 6. Statistical Diagnostics Audit
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n=== HISTORICAL VALIDATION PERFORMANCE DIAGNOSTICS ===")
    print(f"Mean Absolute Error (MAE): {mae:.6f} (Average return variance per hour)")
    print(f"R-Squared (R2) Stability Score: {r2:.4f}")
    
    # 7. Structural Feature Importance Audit (Mean Decrease in Impurity)
    importances = model.feature_importances_
    
    importance_matrix = pd.DataFrame({
        "Feature_Indicator": feature_cols,
        "Relative_Importance_Weight": importances
    }).sort_values(by="Relative_Importance_Weight", ascending=False)
    
    print("\n=== MODEL ARCHITECTURE FEATURE IMPORTANCE WEIGHTS ===")
    print(importance_matrix.to_string(index=False))
    
    return model

if __name__ == "__main__":
    trained_model = execute_model_training()