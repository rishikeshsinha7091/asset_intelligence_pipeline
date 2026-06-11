import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
# pyrefly: ignore [missing-import]
import shap

def execute_final_production_training(input_path="processed_features.csv"):
    print("Loading feature assets into production memory...")
    df = pd.read_csv(input_path)
    
    # 1. Feature Pruning: We completely drop the lowest-performing SHAP indicators
    feature_cols = ["sma_fast", "sma_slow", "price_to_sma_fast", "price_to_sma_slow"]
    X = df[feature_cols]
    y = df["target_return_1h"]
    
    # 2. Chronological Train/Test Split (80% / 20%)
    split_index = int(len(df) * 0.80)
    X_train_raw, X_test_raw = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # 3. Production Feature Standardization (Scaling)
    # This transforms our data columns so they all share a uniform scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Convert back to clean DataFrames so SHAP can read the column labels right
    X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    # 4. Constrained Model Architecture Initialization
    print("Initializing fully regularized predictive engine...")
    model = RandomForestRegressor(
        n_estimators=200,      # Increased trees for smoother variance reductions
        max_depth=3,           # Further capped depth to strictly enforce macro rules
        min_samples_leaf=15,   # Increased leaf constraints to eliminate tail-noise
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting standardized parameters against core signals...")
    model.fit(X_train, y_train)
    
    # 5. Inference Testing
    predictions = model.predict(X_test)
    
    # 6. Final Performance Evaluation Audit
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n=== SYSTEM OPTIMIZED PERFORMANCE DIAGNOSTICS ===")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Final Stability Score (R2): {r2:.4f}")
    
    # 7. Explainable AI Verification
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_shap_impact = np.abs(shap_values).mean(axis=0)
    
    shap_matrix = pd.DataFrame({
        "Feature_Indicator": feature_cols,
        "Mean_Absolute_SHAP_Impact": mean_shap_impact
    }).sort_values(by="Mean_Absolute_SHAP_Impact", ascending=False)
    
    print("\n=== VERIFIED EXPLAINABLE AI (SHAP) ATTRIBUTIONS ===")
    print(shap_matrix.to_string(index=False))
    
    return model

if __name__ == "__main__":
    production_model = execute_final_production_training()