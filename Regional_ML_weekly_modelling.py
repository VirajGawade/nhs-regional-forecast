import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler  
import joblib

# Directories
MODEL_DIR = "regional_ml_weekly_models"
FORECAST_DIR = "regional_ml_weekly_forecasts"
SCALER_DIR = "regional_ml_weekly_scalers"  
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)  

# Load dataset
df = pd.read_csv("regional_engineered_features_weekly.csv")
df["Week"] = pd.to_datetime(df["Week"])
df = df.sort_values("Week")

# Constants
target_col = "Total_Attendances"
region_col = "Region_unified"
non_feature_cols = [target_col, "Week", "Region", "Type", region_col]
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in non_feature_cols]
split_date = pd.to_datetime("2024-11-01")

# Models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
}

# Loop over regions
regions = sorted(df[region_col].dropna().unique())

summary_results = []

for region in regions:
    print(f"\n[INFO] Processing region: {region}")
    region_df = df[df[region_col] == region].copy().sort_values("Week")

    train_df = region_df[region_df["Week"] < split_date]
    test_df = region_df[region_df["Week"] >= split_date]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Scale features and save scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(SCALER_DIR, f"weekly_scaler_{region}.save")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Saved scaler for {region} at {scaler_path}")

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        acc = 100 - mape

        print(f"""
[INFO] Region: {region}
[INFO] Model: {model_name}
[INFO] MAE: {mae:,.0f}
[INFO] RMSE: {rmse:,.0f}
[INFO] MAPE: {mape:.2f}%
[INFO] Accuracy: {acc:.2f}%
[INFO] R²: {r2:.4f}
""")

        # Append for summary
        summary_results.append({
            "Region": region,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Accuracy (%)": acc,
            "R²": r2
        })

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{region.replace(' ', '_')}_{model_name}_weekly.pkl")
        joblib.dump(model, model_path)

        # Save forecast
        results_df = test_df[["Week"]].copy()
        results_df["Actual"] = y_test.values
        results_df["Predicted"] = y_pred
        results_df["Region"] = region
        results_df["Model"] = model_name
        results_df.to_csv(
            os.path.join(FORECAST_DIR, f"{region.replace(' ', '_')}_{model_name}_weekly_forecast.csv"),
            index=False
        )

# Print summary table
summary_df = pd.DataFrame(summary_results)
print("[INFO] ML Weekly Modeling Summary:")
print(summary_df.to_string(index=False))

print("\n[INFO] All regional weekly models trained and forecasts saved.")
