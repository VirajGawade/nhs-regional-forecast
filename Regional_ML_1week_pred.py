import pandas as pd
import numpy as np
import argparse
import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as ticker

# Paths
DATA_PATH = "regional_engineered_features_weekly.csv"
MODEL_DIR = "regional_ml_weekly_models"
SCALER_DIR = "regional_ml_weekly_scalers"  
FORECAST_DIR = "regional_ml_weekly_forecasts"
PLOT_DIR = "regional_ml_weekly_plots"
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument("--day", type=int, required=True)
args = parser.parse_args()

# Convert input date to ISO week start 
target_date = datetime(args.year, args.month, args.day)
iso_year, iso_week, _ = target_date.isocalendar()
week_start_date = datetime.strptime(f"{iso_year}-W{iso_week}-1", "%G-W%V-%u") + timedelta(days=6)
target_week_str = f"{iso_year}-W{str(iso_week).zfill(2)}"

# Load dataset
df = pd.read_csv(DATA_PATH)
df["Week"] = pd.to_datetime(df["Week"])
df = df.sort_values("Week")
target_col = "Total_Attendances"
region_col = "Region_unified"
non_features = [target_col, "Region", region_col, "Type", "Week"]
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in non_features]

# Regions
regions = sorted(df[region_col].dropna().unique())

results = []

# Loop through regions
for region in regions:
    print(f"\n[INFO] Processing region: {region}")
    region_df = df[df[region_col] == region].copy()
    region_df = region_df.sort_values("Week").reset_index(drop=True)

    if week_start_date in list(region_df["Week"]):
        # Validation mode
        mode = "validation"
        test_row = region_df[region_df["Week"] == week_start_date].copy()
        input_row = region_df[region_df["Week"] < week_start_date].tail(12)
    else:
        # Forecast mode
        mode = "forecast"
        input_row = region_df.tail(12)

    # Load scaler
    scaler_path = os.path.join(SCALER_DIR, f"weekly_scaler_{region}.save")
    scaler = joblib.load(scaler_path)

    # Prepare input features
    mean_features = input_row[feature_cols].mean()
    input_features = pd.DataFrame([mean_features], columns=feature_cols)

    # Align columns to scaler's expected features (exclude target if present)
    expected_cols = list(scaler.feature_names_in_)
    if target_col in expected_cols:
        expected_cols.remove(target_col)
    input_features = input_features[expected_cols]

    # Scale input
    input_scaled = scaler.transform(input_features)

    # Load models
    rf_path = os.path.join(MODEL_DIR, f"{region.replace(' ', '_')}_RandomForest_weekly.pkl")
    xgb_path = os.path.join(MODEL_DIR, f"{region.replace(' ', '_')}_XGBoost_weekly.pkl")
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    # Predict
    rf_pred = rf_model.predict(input_scaled)[0]
    xgb_pred = xgb_model.predict(input_scaled)[0]

    if mode == "validation":
        actual = test_row[target_col].values[0]
        rf_mae = mean_absolute_error([actual], [rf_pred])
        rf_rmse = np.sqrt(mean_squared_error([actual], [rf_pred]))
        rf_mape = np.mean(np.abs((actual - rf_pred) / actual)) * 100
        rf_acc = 100 - rf_mape

        xgb_mae = mean_absolute_error([actual], [xgb_pred])
        xgb_rmse = np.sqrt(mean_squared_error([actual], [xgb_pred]))
        xgb_mape = np.mean(np.abs((actual - xgb_pred) / actual)) * 100
        xgb_acc = 100 - xgb_mape

        print(f"""
[INFO] Region: {region}
[INFO] Week: {week_start_date.date()}
[INFO] Model: RandomForest
[INFO] Predicted: {int(round(rf_pred)):,}

[INFO] Actual: {int(actual):,}
[INFO] MAE: {rf_mae:,.2f}
[INFO] RMSE: {rf_rmse:,.2f}
[INFO] MAPE: {rf_mape:.2f}%
[INFO] Accuracy: {rf_acc:.2f}%
""")

        print(f"""
[INFO] Region: {region}
[INFO] Week: {week_start_date.date()}
[INFO] Model: XGBoost
[INFO] Predicted: {int(round(xgb_pred)):,}

[INFO] Actual: {int(actual):,}
[INFO] MAE: {xgb_mae:,.2f}
[INFO] RMSE: {xgb_rmse:,.2f}
[INFO] MAPE: {xgb_mape:.2f}%
[INFO] Accuracy: {xgb_acc:.2f}%
""")

        results.append({
            "Region": region,
            "Model": "RandomForest",
            "MAE": rf_mae,
            "RMSE": rf_rmse,
            "MAPE (%)": rf_mape,
            "Accuracy (%)": rf_acc
        })
        results.append({
            "Region": region,
            "Model": "XGBoost",
            "MAE": xgb_mae,
            "RMSE": xgb_rmse,
            "MAPE (%)": xgb_mape,
            "Accuracy (%)": xgb_acc
        })

    else:
        print(f"""
[INFO] Region: {region}
[INFO] Week: {week_start_date.date()}
[INFO] Model: RandomForest
[INFO] Predicted: {int(round(rf_pred)):,}
""")
        print(f"""
[INFO] Region: {region}
[INFO] Week: {week_start_date.date()}
[INFO] Model: XGBoost
[INFO] Predicted: {int(round(xgb_pred)):,}
""")

        results.append({
            "Region": region,
            "Model": "RandomForest",
            "MAE": None,
            "RMSE": None,
            "MAPE (%)": None,
            "Accuracy (%)": None
        })
        results.append({
            "Region": region,
            "Model": "XGBoost",
            "MAE": None,
            "RMSE": None,
            "MAPE (%)": None,
            "Accuracy (%)": None
        })

    # Save forecast CSV
    forecast_df = pd.DataFrame({
        "Region": [region, region],
        "Week": [week_start_date.date(), week_start_date.date()],
        "Model": ["RandomForest", "XGBoost"],
        "Predicted": [rf_pred, xgb_pred]
    })
    forecast_path = os.path.join(FORECAST_DIR, f"{region.replace(' ', '_')}_week_{target_week_str}.csv")
    forecast_df.to_csv(forecast_path, index=False)

    # Plot
    if mode == "validation":
        actual_val = actual
    else:
        actual_val = None  # No actual available in forecast mode

    # RandomForest plot
    plt.figure(figsize=(5, 5))
    if actual_val is not None:
        labels = ["Actual", "RandomForest"]
        values = [actual_val, rf_pred]
        colors = ["blue", "green"]
    else:
        labels = ["RandomForest"]
        values = [rf_pred]
        colors = ["green"]

    plt.bar(labels, values, color=colors)
    plt.title(f"{region} - Week {target_week_str} (RF)")
    plt.ylabel("Total Attendances")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    plt.tight_layout()
    rf_plot_path = os.path.join(PLOT_DIR, f"{region.replace(' ', '_')}_week_{target_week_str}_rf.png")
    plt.savefig(rf_plot_path)
    plt.close()

    # XGBoost plot
    plt.figure(figsize=(5, 5))
    if actual_val is not None:
        labels = ["Actual", "XGBoost"]
        values = [actual_val, xgb_pred]
        colors = ["blue", "orange"]
    else:
        labels = ["XGBoost"]
        values = [xgb_pred]
        colors = ["orange"]

    plt.bar(labels, values, color=colors)
    plt.title(f"{region} - Week {target_week_str} (XGB)")
    plt.ylabel("Total Attendances")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    plt.tight_layout()
    xgb_plot_path = os.path.join(PLOT_DIR, f"{region.replace(' ', '_')}_week_{target_week_str}_xgb.png")
    plt.savefig(xgb_plot_path)
    plt.close()


# Summary Table
summary_df = pd.DataFrame(results)
print("\n[INFO] ML Weekly 1-week Prediction Summary:")
print(summary_df.to_string(index=False))

print("\n[INFO] All regional 1-week predictions completed.")
