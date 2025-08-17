import pandas as pd
import numpy as np
import argparse
import os
import joblib
from datetime import datetime, timedelta

# Config
DATA_PATH = "regional_engineered_features_weekly.csv"
MODEL_DIR = "regional_ml_weekly_models"
SCALER_DIR = "regional_ml_weekly_scalers"
FORECAST_DIR = "regional_ml_7day_daily_forecasts"
os.makedirs(FORECAST_DIR, exist_ok=True)

# Daily proportions 
DAILY_PROPORTIONS = {
    "Monday": 0.175,
    "Tuesday": 0.165,
    "Wednesday": 0.155,
    "Thursday": 0.145,
    "Friday": 0.135,
    "Saturday": 0.110,
    "Sunday": 0.115
}

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument("--day", type=int, required=True)
args = parser.parse_args()

# Determine ISO week start (Monday) and week end (Sunday)
target_date = datetime(args.year, args.month, args.day)
iso_year, iso_week, _ = target_date.isocalendar()
week_start_date = datetime.strptime(f"{iso_year}-W{iso_week}-1", "%G-W%V-%u")
week_end_date = week_start_date + timedelta(days=6)
target_week_str = f"{iso_year}-W{str(iso_week).zfill(2)}"

# Load data
df = pd.read_csv(DATA_PATH)
df["Week"] = pd.to_datetime(df["Week"])
df = df.sort_values("Week")
target_col = "Total_Attendances"
region_col = "Region_unified"
non_features = [target_col, "Region", region_col, "Type", "Week"]
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in non_features]

regions = sorted(df[region_col].dropna().unique())

# Collect all results for plotting
all_results = []

# Prediction loop
for region in regions:
    print(f"\n[INFO] Processing region: {region}")
    region_df = df[df[region_col] == region].copy().sort_values("Week").reset_index(drop=True)

    # Check if validation or forecast
    if week_end_date in list(region_df["Week"]):
        mode = "validation"
        test_row = region_df[region_df["Week"] == week_end_date].copy()
        input_row = region_df[region_df["Week"] < week_end_date].tail(12)
    else:
        mode = "forecast"
        input_row = region_df.tail(12)

    # Load scaler
    scaler_path = os.path.join(SCALER_DIR, f"weekly_scaler_{region}.save")
    scaler = joblib.load(scaler_path)

    # Prepare features
    mean_features = input_row[feature_cols].mean()
    input_features = pd.DataFrame([mean_features], columns=feature_cols)

    expected_cols = list(scaler.feature_names_in_)
    if target_col in expected_cols:
        expected_cols.remove(target_col)
    input_features = input_features[expected_cols]

    input_scaled = scaler.transform(input_features)

    # Load models
    rf_path = os.path.join(MODEL_DIR, f"{region.replace(' ', '_')}_RandomForest_weekly.pkl")
    xgb_path = os.path.join(MODEL_DIR, f"{region.replace(' ', '_')}_XGBoost_weekly.pkl")
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    # Predict weekly totals
    rf_weekly_pred = rf_model.predict(input_scaled)[0]
    xgb_weekly_pred = xgb_model.predict(input_scaled)[0]

    # Split into daily predictions (floor Mon to Sat, leftover on Sunday)
    def split_week_to_days(total):
        total_int = int(round(total))
        days = list(DAILY_PROPORTIONS.keys())
        daily = []
        remaining = total_int
        for i, day in enumerate(days):
            if i < 6:  # Monday..Saturday
                v = int(np.floor(total_int * DAILY_PROPORTIONS[day]))
                daily.append(v)
                remaining -= v
            else:      # Sunday = leftover so the 7 days sum to total_int
                daily.append(remaining)
        return dict(zip(days, daily))

    rf_daily_ints  = split_week_to_days(rf_weekly_pred)
    xgb_daily_ints = split_week_to_days(xgb_weekly_pred)

    # Print results 
    print(f"[INFO] RandomForest Forecasted Weekly Total: {int(round(rf_weekly_pred)):,}")
    print(f"[INFO] XGBoost   Forecasted Weekly Total: {int(round(xgb_weekly_pred)):,}")
    print("[INFO] 7-Day Breakdown:")
    for i, day in enumerate(DAILY_PROPORTIONS.keys()):
        day_date = week_start_date + timedelta(days=i)
        print(f"[INFO]   {day} ({day_date.date()}): RF={rf_daily_ints[day]:,} | XGB={xgb_daily_ints[day]:,}")

    # Save forecast CSV 
    forecast_data = []
    for i, day in enumerate(DAILY_PROPORTIONS.keys()):
        day_date = week_start_date + timedelta(days=i)
        forecast_data.append({
            "Region": region,
            "Date": day_date.date(),
            "Day": day,
            "RandomForest": rf_daily_ints[day],
            "XGBoost": xgb_daily_ints[day]
        })

        # Store average prediction for plotting
        all_results.append({
            "Region": region,
            "Date": day_date.date(),
            "Day": day,
            "Weekly_RF": int(round(rf_weekly_pred)),
            "Daily_RF": rf_daily_ints[day],
            "Weekly_XGB": int(round(xgb_weekly_pred)),
            "Daily_XGB": xgb_daily_ints[day]
        })


# Create dataframe for plotting
df_result = pd.DataFrame(all_results)

# Plot ML 7-day daily forecast 
# Plot 1: RandomForest 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure(figsize=(12, 6))
for region in regions:
    df_r = df_result[df_result["Region"] == region]
    plt.plot(df_r["Date"], df_r["Daily_RF"], marker='o', label=region)

plt.xlabel("Date")
plt.ylabel("Total Attendances")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
plt.title(f"Regional ML 7-Day Daily Forecast (RandomForest, Week of {week_start_date.date()})")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("regional_ml_7day_daily_forecast_RF.png")
plt.show()

# Plot 2: XGBoost 
plt.figure(figsize=(12, 6))
for region in regions:
    df_r = df_result[df_result["Region"] == region]
    plt.plot(df_r["Date"], df_r["Daily_XGB"], marker='s', label=region)

plt.xlabel("Date")
plt.ylabel("Total Attendances")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
plt.title(f"Regional ML 7-Day Daily Forecast (XGBoost, Week of {week_start_date.date()})")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("regional_ml_7day_daily_forecast_XGB.png")
plt.show()


print("\n[INFO] All regional 7-day daily predictions completed.")

# Save combined summary 
output_file = f"regional_ml_7day_daily_predictions_{iso_year}-W{iso_week:02d}.csv"
ml_df = pd.DataFrame(all_results)
ml_df.to_csv(output_file, index=False)
print(f"[INFO] Saved ML daily predictions to {output_file}")


