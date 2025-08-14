import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import joblib
import matplotlib.ticker as ticker

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
args = parser.parse_args()

target_date = pd.Timestamp(year=args.year, month=args.month, day=1)

# Load dataset
df = pd.read_csv("regional_engineered_features.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month")
df = df.dropna(subset=["Total_Attendances"])

# Define columns
target_col = "Total_Attendances"
region_col = "Region_unified"
date_col = "Month"
non_features = [target_col, region_col, date_col, "Region", "Type"]
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in non_features]

# Output folder
os.makedirs("regional_ml_1month_predictions", exist_ok=True)

results = []  # For summary

# Loop over regions
for region in sorted(df[region_col].dropna().unique()):
    print(f"\nPredicting for region: {region}")

    region_df = df[df[region_col] == region].copy().sort_values(date_col)
    region_df = region_df.reset_index(drop=True)

    # Load saved scaler and model paths
    scaler_path = f"regional_ml_monthly_scalers/{region.replace(' ', '_')}_scaler.save"

    models_to_load = {
        "RandomForest": f"regional_ml_monthly_models/{region.replace(' ', '_')}_RandomForest_monthly.pkl",
        "XGBoost": f"regional_ml_monthly_models/{region.replace(' ', '_')}_XGBoost_monthly.pkl"
    }

    # Load scaler
    if not os.path.exists(scaler_path):
        print(f"Scaler not found for region {region} at {scaler_path}, skipping...")
        continue
    scaler = joblib.load(scaler_path)

    # Prepare data
    last_date = region_df[date_col].max()
    full_features = region_df[feature_cols].copy()

    if target_date <= last_date:
        # Validation mode
        match_row = region_df[region_df[date_col] == target_date]
        if match_row.empty:
            print(f" No data available for {target_date.strftime('%B %Y')} in region {region}. Skipping...")
            continue
        X_val = scaler.transform(match_row[feature_cols])
        actual = match_row[target_col].values[0]
        mode = "validation"
    else:
        # Forecast mode: average of last 12 months features
        forecast_input = region_df[region_df[date_col] < target_date].copy().tail(12)
        if len(forecast_input) < 12:
            print(f" Not enough past data to forecast for {region}. Skipping...")
            continue
        X_val = scaler.transform(forecast_input[feature_cols]).mean(axis=0).reshape(1, -1)
        actual = 0  # For plot only
        mode = "forecast"

    # Predict with saved models
    for model_name, model_path in models_to_load.items():
        if not os.path.exists(model_path):
            print(f"[INFO] Model not found: {model_path}, skipping...")
            continue

        model = joblib.load(model_path)
        pred = model.predict(X_val)[0]

        if mode == "validation":
            mae = np.abs(actual - pred)
            mape = np.abs((actual - pred) / actual) * 100
            rmse = np.sqrt((actual - pred) ** 2)

            print(f"""
[INFO] Region: {region}
[INFO] Month: {target_date.strftime('%Y-%m')}
[INFO] Model: {model_name}
[INFO] Predicted: {int(pred):,}

[INFO] Actual: {int(actual):,}
[INFO] MAE: {mae:,.2f}
[INFO] RMSE: {rmse:,.2f}
[INFO] MAPE: {mape:.2f}%
[INFO] Accuracy: {100 - mape:.2f}%
""")

            results.append({
                "Region": region,
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "Accuracy (%)": 100 - mape
            })

            result_df = pd.DataFrame({
                "Month": [target_date],
                "Region": [region],
                "Model": [model_name],
                "Actual": [actual],
                "Predicted": [pred],
                "MAE": [mae],
                "RMSE": [rmse],
                "MAPE (%)": [mape],
                "Accuracy (%)": [100 - mape]
            })
        else:
            print(f"""
[INFO] Region: {region}
[INFO] Month: {target_date.strftime('%Y-%m')}
[INFO] Model: {model_name}
[INFO] Predicted: {int(pred):,}
""")

            result_df = pd.DataFrame({
                "Month": [target_date],
                "Region": [region],
                "Model": [model_name],
                "Predicted": [pred]
            })

        # Save prediction CSV
        out_csv = os.path.join("regional_ml_1month_predictions", f"{region.replace(' ', '_')}_{model_name}_prediction.csv")
        result_df.to_csv(out_csv, index=False)

        # Save bar plot
        plt.figure(figsize=(5, 5))
        plt.bar(["Actual", "Predicted"], [actual, pred], color=["blue", "orange"])
        plt.title(f"{region} - {model_name} ({target_date.strftime('%b %Y')})")
        plt.ylabel("Attendances")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
        plt.tight_layout()
        out_png = os.path.join("regional_ml_1month_predictions", f"{region.replace(' ', '_')}_{model_name}_prediction_plot.png")
        plt.savefig(out_png)
        plt.close()

print("\nAll regional 1-month predictions completed.")

# Summary output like LSTM monthly modeling
summary_df = pd.DataFrame(results)
summary_df = summary_df[["Region", "Model", "MAE", "RMSE", "MAPE (%)", "Accuracy (%)"]]
print("\n[INFO] ML Monthly 1-Month Prediction Summary:")
print(summary_df.to_string(index=False))
