import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import timedelta
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
from keras.models import load_model
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
tf.get_logger().setLevel("ERROR")
sys.stderr = stderr

# Args 
parser = argparse.ArgumentParser(description="Regional 7-Day Daily Forecast Using LSTM")
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument("--day", type=int, required=True)
args = parser.parse_args()

# Config 
SEQ_LEN = 12
TARGET_COL = "Total_Attendances"
DATA_PATH = "regional_engineered_features_weekly.csv"
DAILY_PROPORTIONS = {
    'Monday': 0.175,
    'Tuesday': 0.165,
    'Wednesday': 0.155,
    'Thursday': 0.145,
    'Friday': 0.135,
    'Saturday': 0.110,
    'Sunday': 0.115
}

# Input Date and Week 
input_date = pd.Timestamp(f"{args.year:04d}-{args.month:02d}-{args.day:02d}")
week_start = input_date - pd.Timedelta(days=input_date.weekday())  # Monday
week_dates = [week_start + timedelta(days=i) for i in range(7)]
week_labels = [d.strftime('%A') for d in week_dates]

# Load dataset
df_all = pd.read_csv(DATA_PATH, parse_dates=["Week"])
regions = df_all["Region_unified"].unique()
results = []

for region in regions:
    print(f"\n[INFO] Processing region: {region}")
    model_path = f"best_weekly_lstm_{region}.keras"
    scaler_path = f"weekly_scaler_{region}.save"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[INFO] Skipping {region}: model or scaler not found.")
        continue

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"[INFO] Error loading model/scaler for {region}: {e}")
        continue

    df_region = df_all[df_all["Region_unified"] == region].sort_values("Week").reset_index(drop=True)
    expected_features = scaler.feature_names_in_
    input_features = [col for col in expected_features if col != TARGET_COL]

    # Determine if validation or forecast mode
    max_available_week = df_region["Week"].max().date()

    if week_start.date() <= max_available_week:
        # Validation mode: exclude current week
        df_region = df_region[df_region["Week"].dt.date < week_start.date()]
    else:
        # Forecast mode: use all available history
        pass

    try:
        scaled = scaler.transform(df_region[expected_features])
    except Exception as e:
        print(f"[INFO] Skipping {region}: Scaling failed — {e}")
        continue

    scaled_df = pd.DataFrame(scaled, columns=expected_features)
    scaled_df["Week"] = df_region["Week"]
    target_index = list(expected_features).index(TARGET_COL)

    # Prepare input sequence
    if len(scaled_df) < SEQ_LEN:
        print(f"[INFO] Not enough history for forecast for {region}.")
        continue

    input_seq = scaled_df[input_features].tail(SEQ_LEN).values
    input_seq = np.expand_dims(input_seq, axis=0)
    pred_scaled = model.predict(input_seq, verbose=0)[0][0]

    # Inverse scale full row
    last_frame = np.zeros(len(expected_features))
    for i, col in enumerate(input_features):
        idx = list(expected_features).index(col)
        last_frame[idx] = input_seq[0, -1, i]
    last_frame[target_index] = pred_scaled
    pred_full = scaler.inverse_transform([last_frame])[0]
    weekly_total = pred_full[target_index]

    print(f"[INFO] Forecasted Weekly Total: {int(weekly_total):,}")
    print("[INFO] 7-Day Breakdown:")
    
    
    # Split into 7 days (floor Mon to Sat, leftover on Sunday)
    total_int = int(round(weekly_total))
    daily_ints = []
    remaining = total_int
    for i, day in enumerate(week_labels):
        if i < 6:  # Monday to Saturday
            v = int(np.floor(total_int * DAILY_PROPORTIONS[day]))
            daily_ints.append(v)
            remaining -= v
        else:      # Sunday = leftover
            v = remaining
            daily_ints.append(v)

    for i, day in enumerate(week_labels):
        daily_value_rounded = daily_ints[i]
        print(f"[INFO]   {day} ({week_dates[i].date()}): {daily_value_rounded:,}")
        results.append({
            "Region": region,
            "Date": week_dates[i].date(),
            "Day": day,
            "Weekly_LSTM": int(round(weekly_total)),
            "Daily_LSTM": daily_value_rounded
        })

df_result = pd.DataFrame(results)

# Save results
output_file = f"regional_LSTM_7day_daily_predictions_{week_start.isocalendar()[0]}-W{week_start.isocalendar()[1]:02d}.csv"
df_result.to_csv(output_file, index=False)
print(f"[INFO] Saved LSTM daily predictions to {output_file}")


# Plot
# Plot
plt.figure(figsize=(12, 6))
for region in regions:
    df_r = df_result[df_result["Region"] == region]
    plt.plot(df_r["Date"], df_r["Daily_LSTM"], marker='o', label=region)

plt.xlabel("Date")
plt.ylabel("Total Attendances")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
plt.title(f"Regional LSTM 7-Day Daily Forecast (Week of {week_start.date()})")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("regional_lstm_7day_daily_forecast_plot.png")
plt.show()

