import os
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Config
DATA_PATH = "regional_engineered_features_weekly.csv"

# ML 
ML_MODEL_DIR = "regional_ml_weekly_models"
ML_SCALER_DIR = "regional_ml_weekly_scalers"

# LSTM 
LSTM_MODEL_DIR = "."
LSTM_SCALER_DIR = "."

# Output
OUT_DIR = "hybrid_ml_lstm_7day_daily_forecasts"
os.makedirs(OUT_DIR, exist_ok=True)

# Daily proportions
DAILY_PROPORTIONS = {
    "Monday":    0.175,
    "Tuesday":   0.165,
    "Wednesday": 0.155,
    "Thursday":  0.145,
    "Friday":    0.135,
    "Saturday":  0.110,
    "Sunday":    0.115,
}

SEQ_LEN = 12
ML_WEIGHT = 0.7
LSTM_WEIGHT = 0.3
TARGET_COL = "Total_Attendances"
REGION_COL = "Region_unified"

# CLI
parser = argparse.ArgumentParser(description="Hybrid (ML+LSTM) 7-Day Daily Forecast")
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument("--day", type=int, required=True)
args = parser.parse_args()

# Determine ISO week
input_date = datetime(args.year, args.month, args.day)
week_start = input_date - timedelta(days=input_date.weekday())
week_end = week_start + timedelta(days=6)
iso_year, iso_week, _ = week_end.isocalendar()
iso_week_str = f"{iso_year}-W{str(iso_week).zfill(2)}"

print(f"\n[INFO] Hybrid 7-day daily forecast for week {iso_week_str} (Mon {week_start.date()} – Sun {week_end.date()})")

# Load dataset
df_all = pd.read_csv(DATA_PATH)
df_all["Week"] = pd.to_datetime(df_all["Week"])
df_all = df_all.sort_values("Week").reset_index(drop=True)

non_features = [TARGET_COL, "Region", REGION_COL, "Type", "Week"]
feature_cols = [c for c in df_all.select_dtypes(include=[np.number]).columns if c not in non_features]
regions = sorted(df_all[REGION_COL].dropna().unique())

# Helpers
def split_week_to_days(total: float, proportions: dict) -> dict:
    total_int = int(round(total))
    days = list(proportions.keys())
    props = list(proportions.values())
    raw = [total_int * p for p in props]
    daily = [int(np.floor(x)) for x in raw]
    diff = total_int - sum(daily)
    daily[-1] += diff
    return dict(zip(days, daily))

def predict_ml_week(region_df: pd.DataFrame, region: str, target_week_end: pd.Timestamp):
    if target_week_end in list(region_df["Week"]):
        input_win = region_df[region_df["Week"] < target_week_end].tail(SEQ_LEN)
    else:
        input_win = region_df.tail(SEQ_LEN)

    scaler_path = os.path.join(ML_SCALER_DIR, f"weekly_scaler_{region}.save")
    rf_path = os.path.join(ML_MODEL_DIR, f"{region.replace(' ', '_')}_RandomForest_weekly.pkl")
    xgb_path = os.path.join(ML_MODEL_DIR, f"{region.replace(' ', '_')}_XGBoost_weekly.pkl")

    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    mean_features = input_win[feature_cols].mean()
    X = pd.DataFrame([mean_features], columns=feature_cols)

    expected = list(scaler.feature_names_in_)
    if TARGET_COL in expected:
        expected.remove(TARGET_COL)
    X = X[expected]
    Xs = scaler.transform(X)

    rf_pred = float(rf_model.predict(Xs)[0])
    xgb_pred = float(xgb_model.predict(Xs)[0])
    return (rf_pred + xgb_pred) / 2.0

def predict_lstm_week(region_df: pd.DataFrame, region: str, target_week_end: pd.Timestamp):
    model_path = os.path.join(LSTM_MODEL_DIR, f"best_weekly_lstm_{region}.keras")
    scaler_path = os.path.join(LSTM_SCALER_DIR, f"weekly_scaler_{region}.save")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    expected_features = list(scaler.feature_names_in_)
    input_features = [c for c in expected_features if c != TARGET_COL]

    if target_week_end in list(region_df["Week"]):
        df_hist = region_df[region_df["Week"] < target_week_end].copy()
    else:
        df_hist = region_df.copy()

    scaled = scaler.transform(df_hist[expected_features])
    target_index = expected_features.index(TARGET_COL)

    if len(scaled) < SEQ_LEN:
        raise ValueError("Not enough history for LSTM sequence.")

    X_seq = scaled[-SEQ_LEN:, [expected_features.index(f) for f in input_features]][None, :, :]
    y_scaled = model.predict(X_seq, verbose=0)[0][0]

    last_frame = scaled[-1].copy()
    last_frame[target_index] = y_scaled
    inv = scaler.inverse_transform([last_frame])[0]
    return float(inv[target_index])

# Main loop
all_rows = []

for region in regions:
    print(f"\n[INFO] Processing region: {region}")
    region_df = df_all[df_all[REGION_COL] == region].copy().sort_values("Week").reset_index(drop=True)

    try:
        ml_week = predict_ml_week(region_df, region, week_end)
    except Exception as e:
        print(f"[WARN] ML weekly prediction failed for {region}: {e}")
        ml_week = None

    try:
        lstm_week = predict_lstm_week(region_df, region, week_end)
    except Exception as e:
        print(f"[WARN] LSTM weekly prediction failed for {region}: {e}")
        lstm_week = None

    if (ml_week is None) and (lstm_week is None):
        print(f"[ERROR] No weekly prediction for {region}, skipping daily split.")
        continue
    elif ml_week is None:
        hybrid_week = lstm_week
    elif lstm_week is None:
        hybrid_week = ml_week
    else:
        hybrid_week = ML_WEIGHT * ml_week + LSTM_WEIGHT * lstm_week

    print(f"[INFO] Weekly totals — ML: {int(round(ml_week)) if ml_week is not None else 'N/A'}, "
          f"LSTM: {int(round(lstm_week)) if lstm_week is not None else 'N/A'}, "
          f"Hybrid: {int(round(hybrid_week))}")

    daily_hybrid = split_week_to_days(hybrid_week, DAILY_PROPORTIONS)

    print("[INFO] 7-Day Breakdown (Hybrid):")
    for i, (day, val) in enumerate(daily_hybrid.items()):
        d = (week_start + timedelta(days=i)).date()
        print(f"[INFO]   {day} ({d}): {val:,}")
        all_rows.append({
            "Region": region,
            "ISO_Week": iso_week_str,
            "Date": d,
            "Day": day,
            "ML_Weekly": int(round(ml_week)) if ml_week is not None else None,
            "LSTM_Weekly": int(round(lstm_week)) if lstm_week is not None else None,
            "Hybrid_Weekly": int(round(hybrid_week)),
            "Hybrid_Daily": int(val),
        })

    # Per-region daily total line plot
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    date_str = week_end.strftime("%Y-W%U")

    # Prepare daily dates and values
    daily_dates = [(week_start + timedelta(days=i)).date() for i in range(7)]
    daily_values = list(daily_hybrid.values())

    plt.figure(figsize=(8, 5))
    plt.plot(daily_dates, daily_values, marker='o', color='orange', label='Hybrid Prediction')

    plt.xlabel("Date")
    plt.ylabel("Predicted Attendances (Daily)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    plt.title(f"{region} - Hybrid 7-Day Daily Forecast ({date_str})")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"forecast_plot_{region.replace(' ', '_')}_hybrid_7day_daily_{date_str}.png")
    plt.close()


# Save CSV
out_path = os.path.join(OUT_DIR, f"hybrid_7day_daily_{iso_week_str}.csv")
pd.DataFrame(all_rows).to_csv(out_path, index=False)
print(f"\n[INFO] Saved hybrid 7-day daily forecasts to {out_path}")

# Combined regional line plot 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_result = pd.DataFrame(all_rows)
plt.figure(figsize=(12, 6))
for region in sorted(df_result["Region"].unique()):
    df_r = df_result[df_result["Region"] == region]
    plt.plot(df_r["Date"], df_r["Hybrid_Daily"], marker='o', label=region)

plt.xlabel("Date")
plt.ylabel("Total Attendances")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
plt.title(f"Regional Hybrid 7-Day Daily Forecast (Week of {week_start.date()})")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("regional_hybrid_7day_daily_forecast_plot.png")
plt.show()
