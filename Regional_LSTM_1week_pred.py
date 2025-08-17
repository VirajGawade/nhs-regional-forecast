import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow messages
import sys
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.ticker as ticker
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from keras.models import load_model

# Restore normal stderr after TensorFlow setup
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

sys.stderr = stderr

# CLI args 
parser = argparse.ArgumentParser(description="Regional Weekly LSTM Prediction for All Regions")
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument("--day", type=int, required=True)
args = parser.parse_args()

input_date = pd.Timestamp(f"{args.year:04d}-{args.month:02d}-{args.day:02d}")
week_start = input_date - pd.Timedelta(days=input_date.weekday())
TARGET_DATE = week_start + pd.Timedelta(days=6)
SEQ_LEN = 12
TARGET_COL = "Total_Attendances"
DATA_PATH = "regional_engineered_features_weekly.csv"

# Load dataset and region list
df_all = pd.read_csv(DATA_PATH, parse_dates=["Week"])
regions = df_all["Region_unified"].unique()

results = []

for region in regions:
    print(f"\n[INFO] Processing region: {region}")
    model_path = f"best_weekly_lstm_{region}.keras"
    scaler_path = f"weekly_scaler_{region}.save"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[INFO] Skipping {region}: model or scaler missing.")
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

    try:
        scaled = scaler.transform(df_region[expected_features])
    except Exception as e:
        print(f"[INFO] Skipping {region}: Scaling failed due to missing columns — {e}")
        continue

    scaled_df = pd.DataFrame(scaled, columns=expected_features)
    scaled_df["Week"] = df_region["Week"]
    target_index = list(expected_features).index(TARGET_COL)

    def predict_week(input_seq):
        input_seq = np.expand_dims(input_seq, axis=0)
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        last_frame = np.zeros(len(expected_features))
        for i, col in enumerate(input_features):
            idx = list(expected_features).index(col)
            last_frame[idx] = input_seq[0, -1, i]
        last_frame[target_index] = pred_scaled
        pred_full = scaler.inverse_transform([last_frame])[0]
        return pred_full[target_index]

    # Validation Mode
    if (df_region["Week"] == TARGET_DATE).any():
        target_idx = df_region.index[df_region["Week"] == TARGET_DATE][0]
        if target_idx < SEQ_LEN:
            print("[INFO] Not enough history for input sequence.")
            continue
        input_seq = scaled_df.iloc[target_idx - SEQ_LEN:target_idx][input_features].values
        actual = df_region.loc[target_idx, TARGET_COL]
        pred = predict_week(input_seq)
        mape = mean_absolute_percentage_error([actual], [pred]) * 100
        mae = mean_absolute_error([actual], [pred])
        rmse = np.sqrt(mean_squared_error([actual], [pred]))


        print(f"""
[INFO] Region: {region}
[INFO] Week: {TARGET_DATE.date()}
[INFO] Predicted: {int(pred):,}

[INFO] Actual: {int(actual):,}
[INFO] MAE: {mae:,.2f}
[INFO] RMSE: {rmse:,.2f}
[INFO] MAPE: {mape:.2f}%
[INFO] Accuracy: {100 - mape:.2f}%
""")

        results.append({
            "Region": region,
            "Week": TARGET_DATE,
            "Actual": actual,
            "Predicted": pred,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Accuracy": 100 - mape
        })

    else:
        print(f"[INFO] Forecasting for {TARGET_DATE.date()}")
        if len(scaled_df) < SEQ_LEN:
            print("[INFO] Not enough history for forecast.")
            continue
        input_seq = scaled_df[input_features].tail(SEQ_LEN).values
        pred = predict_week(input_seq)

        print(f"""
[INFO] Region: {region}
[INFO] Week: {TARGET_DATE.date()}
[INFO] Predicted: {int(pred):,}
""")
        results.append({
            "Region": region,
            "Week": TARGET_DATE,
            "Actual": None,
            "Predicted": pred,
            "MAE": None,
            "RMSE": None,
            "MAPE": None,
            "Accuracy": None
        })

summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values("Region")

# Plot
PLOT_DIR = "regional_lstm_weekly_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

for _, row in summary_df.iterrows():
    region = row["Region"]
    actual = row["Actual"]
    predicted = row["Predicted"]

    plt.figure(figsize=(5, 5))  

    # bars 
    if pd.notna(actual):
        plt.bar(["Actual", "Predicted"], [actual, predicted], color=["blue", "orange"])
        plt.title(f"{region} - LSTM Validation (Week Ending {TARGET_DATE.date()})")
    else:
        plt.bar(["Actual", "Predicted"], [0, predicted], color=["blue", "orange"])
        plt.title(f"{region} - LSTM Forecast (Week Ending {TARGET_DATE.date()})")

    plt.ylabel("Total Attendances")
    
    bars = plt.bar(
    ["Actual", "Predicted"],
    [actual if pd.notna(actual) else 0, predicted],
    color=["blue", "orange"],
    width=0.5
    )

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{int(height):,}",
                 ha='center', va='bottom', fontsize=10)
    # Format y-axis 
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))    
    plt.grid(False)
    plt.tight_layout()

    # Save plot
    plot_filename = f"{region.replace(' ', '_')}_weekly_lstm_{TARGET_DATE.date()}.png"
    plt.savefig(os.path.join(PLOT_DIR, plot_filename))
    plt.close()

    print(f"[INFO] Saved plot to {plot_filename}")



# Summary Output
print("\n[INFO] Weekly LSTM 1-week Prediction Summary:")
print(summary_df.to_string(index=False))

summary_filename = f"weekly_lstm_1week_prediction_summary_{TARGET_DATE.date()}.csv"
summary_df.to_csv(summary_filename, index=False)

print(f"[INFO] All regional forecasts complete. Saved to '{summary_filename}'.")
