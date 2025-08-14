import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.ticker as ticker

# Load dataset
df = pd.read_csv("regional_engineered_features.csv")

# Define key columns
target_col = "Total_Attendances"
region_col = "Region_unified"
date_col = "Month"

# Drop any rows with missing target
df = df.dropna(subset=[target_col])

# Ensure output directories exist
os.makedirs("regional_ml_monthly_outputs", exist_ok=True)
os.makedirs("regional_ml_monthly_models", exist_ok=True)
os.makedirs("regional_ml_monthly_scalers", exist_ok=True)  

# Identify numeric feature columns only
non_features = [target_col, region_col, date_col, "Region", "Type"]
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in feature_cols if col not in non_features]

summary_results = []  # summary info for all regions/models

# Process each of the 7 broader NHS regions
for region in sorted(df[region_col].dropna().unique()):
    print(f"\n[INFO] Processing region: {region}")

    region_df = df[df[region_col] == region].copy()
    region_df = region_df.sort_values(by=date_col)

    X = region_df[feature_cols]
    y = region_df[target_col]

    if len(X) < 10:
        print(f"[INFO] Not enough data for region: {region} (only {len(X)} rows). Skipping...")
        continue

    # Scale features and save scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_path = os.path.join("regional_ml_monthly_scalers", f"{region.replace(' ', '_')}_scaler.save")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Saved scaler for {region} at {scaler_path}")

    # Train-test split
    train_size = len(X_scaled) - 7
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    dates_test = pd.to_datetime(region_df[date_col].iloc[train_size:])

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    }

    for model_name, model in models.items():
        print(f"[INFO] Training {model_name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_save_path = os.path.join("regional_ml_monthly_models", f"{region.replace(' ', '_')}_{model_name}_monthly.pkl")
        joblib.dump(model, model_save_path)
        print(f"[INFO] Saved {model_name} model for {region} at {model_save_path}")

        # Evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        accuracy = 100 - mape
        r2 = r2_score(y_test, y_pred)

        print(f"""
[INFO] {model_name} metrics for {region}:
[INFO] MAE: {mae:,.2f}
[INFO] RMSE: {rmse:,.2f}
[INFO] MAPE: {mape:.2f}%
[INFO] Accuracy: {accuracy:.2f}%
[INFO] R²: {r2:.4f}
""")

        # Save forecast CSV
        results_df = pd.DataFrame({
            "Month": dates_test.dt.strftime('%Y-%m'),
            "Actual": y_test.values,
            "Predicted": y_pred
        })
        filename = f"{region.replace(' ', '_')}_{model_name}_forecast.csv"
        results_df.to_csv(os.path.join("regional_ml_monthly_outputs", filename), index=False)

        
        # Plot forecast as line chart 
        plt.figure(figsize=(6, 4))
        plt.plot(dates_test, y_test.values, marker="o", label="Actual")
        plt.plot(dates_test, y_pred, marker="x", label="Predicted")

        plt.xticks(rotation=45)
        plt.ylabel("Total Attendances")
        plt.title(f"{region} – {model_name} Monthly Forecast")
        plt.legend()
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
        plt.tight_layout()

        plot_filename = f"{region.replace(' ', '_')}_{model_name}_forecast_plot.png"
        plt.savefig(os.path.join("regional_ml_monthly_outputs", plot_filename))
        plt.close()


        # Append summary info
        summary_results.append({
            "Region": region,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Accuracy (%)": accuracy,
            "R2": r2
        })

# print summary table
summary_df = pd.DataFrame(summary_results)
print("[INFO] ML Monthly Modeling Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv("regional_ml_monthly_models_summary.csv", index=False)

print("\n[INFO] All regional models trained, scaled, saved, evaluated, and forecasts saved.")
