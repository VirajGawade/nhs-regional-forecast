import os 
import warnings
import argparse
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF msg
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore Keras UserWarnings
tf.get_logger().setLevel('ERROR')  # Suppress TF logs below ERROR level


# cli arg
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
args = parser.parse_args()

YEAR = args.year
MONTH = args.month

# Config 
DATA_PATH = "regional_engineered_features.csv"
REGIONS = [
    "London", "East of England", "South East", "South West",
    "Midlands", "North West", "North East and Yorkshire"
]
SEQ_LEN = 12
target_date = pd.Timestamp(year=YEAR, month=MONTH, day=1)
date_str = target_date.strftime("%Y-%m")  

# Load Data 
df_all = pd.read_csv(DATA_PATH, parse_dates=["Month"])
df_all = df_all.sort_values("Month")

# Collect all results
all_results = []
results = []  # summary 

os.makedirs("regional_lstm_1month_plots", exist_ok=True)

for REGION in REGIONS:
    df = df_all[df_all["Region_unified"] == REGION].copy()

    if len(df) < SEQ_LEN + 1:
        print(f"[WARNING] Not enough data for {REGION}. Skipping.")
        continue

    try:
        scaler = joblib.load(f"scaler_{REGION}.save")
        model = load_model(f"lstm_model_{REGION}.keras")
    except Exception as e:
        print(f"[ERROR] Failed to load model/scaler for {REGION}: {e}")
        continue

    X_all = df.drop(columns=["Month", "Region", "Region_unified"])
    X_all = X_all.select_dtypes(include=["number"])

    try:
        X_scaled = scaler.transform(X_all)
    except Exception as e:
        print(f"[ERROR] Scaling failed for {REGION}: {e}")
        continue

    # Build sequences
    X_seq = []
    dates = []
    for i in range(len(df) - SEQ_LEN):
        end_month = df.iloc[i + SEQ_LEN]["Month"]
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        dates.append(end_month)

    if target_date not in dates:
        print(f"[INFO] Forecasting future date for {REGION} - {date_str}")
        X_input = np.array([X_scaled[-SEQ_LEN:]])
        is_forecast = True
    else:
        print(f"[INFO] Validating known date for {REGION} - {date_str}")
        target_index = dates.index(target_date)
        X_input = np.array(X_seq[target_index:target_index+1])
        is_forecast = False

    # Predict (scaled)
    y_pred_scaled = model.predict(X_input)

    # fixed inverse scaling
    if hasattr(scaler, "feature_names_in_"):
        feature_cols = list(scaler.feature_names_in_)
    else:
        feature_cols = list(X_all.columns)
    target_col = "Total_Attendances"
    target_idx = feature_cols.index(target_col)

    last_frame_scaled = X_input[0, -1, :].copy()
    last_frame_scaled[target_idx] = y_pred_scaled[0][0]

    last_frame_unscaled = scaler.inverse_transform([last_frame_scaled])
    y_pred = last_frame_unscaled[0][target_idx]
    
    # Output
    result_df = pd.DataFrame({
        "Region": [REGION],
        "Month": [target_date],
        "Predicted_Total_Attendances": [int(y_pred)],
    })

    if not is_forecast:
        actual_value = df[df["Month"] == target_date]["Total_Attendances"].values[0]
        result_df["Actual_Total_Attendances"] = int(actual_value)
        result_df["MAE"] = round(mean_absolute_error([actual_value], [y_pred]), 2)
        result_df["RMSE"] = round(np.sqrt(mean_squared_error([actual_value], [y_pred])), 2)
        result_df["MAPE (%)"] = round(np.mean(np.abs((actual_value - y_pred) / actual_value)) * 100, 2)
        result_df["Accuracy (%)"] = round(100 - result_df["MAPE (%)"], 2)

    # Append to summary
    all_results.append(result_df)

    # append to results list for combined summary after loop
    results.append({
        "Region": REGION,
        "Month": date_str,
        "Predicted": int(y_pred),
        "Actual": int(actual_value) if not is_forecast else None,
        "MAE": result_df["MAE"].values[0] if not is_forecast else None,
        "RMSE": result_df["RMSE"].values[0] if not is_forecast else None,
        "MAPE (%)": result_df["MAPE (%)"].values[0] if not is_forecast else None,
        "Accuracy (%)": result_df["Accuracy (%)"].values[0] if not is_forecast else None,
    })

    # Plot with date in filename (separate per region)
    plt.figure(figsize=(5, 5))
    if not is_forecast:
        plt.bar(["Actual", "Predicted"], [actual_value, y_pred], color=["blue", "orange"])
        plt.title(f"{REGION} - LSTM Validation ({date_str})")
    else:
        plt.bar(["Actual", "Predicted"], [0, y_pred], color=["blue", "orange"])
        plt.title(f"{REGION} - LSTM Forecast ({date_str})")

    plt.ylabel("Total Attendances")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    plt.tight_layout()
    for p in plt.gca().patches:
        plt.gca().annotate(
            f'{p.get_height():,.0f}',          
            (p.get_x() + p.get_width() / 2, p.get_height()), 
            ha='center', va='bottom', fontsize=9
        )

    plt.savefig(f"regional_lstm_1month_plots/{REGION.replace(' ', '_')}_prediction_{date_str}.png")
    plt.close()

    print(f"[INFO] Forecast for {REGION} ({date_str}) complete.")
    print(f"""
[INFO] Region: {REGION}
[INFO] Month: {date_str}
[INFO] Predicted: {int(y_pred):,}
""")
    if not is_forecast:
        print(f"""[INFO] Actual: {int(actual_value):,}
[INFO] MAE: {result_df['MAE'].values[0]:,.2f}
[INFO] RMSE: {result_df['RMSE'].values[0]:,.2f}
[INFO] MAPE: {result_df['MAPE (%)'].values[0]:.2f}%
[INFO] Accuracy: {result_df['Accuracy (%)'].values[0]:.2f}%
""")

    tf.keras.backend.clear_session()

# Save one summary CSV 
summary_df = pd.DataFrame(results)
print("\n[INFO] Monthly LSTM 1-month Prediction Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv(f"monthly_lstm_1month_prediction_summary_{date_str}.csv", index=False)
print(f"\n[INFO] All regional forecasts complete. Saved to 'monthly_lstm_1month_prediction_summary_{date_str}.csv'.")
